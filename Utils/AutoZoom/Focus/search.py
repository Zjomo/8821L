"""
常用自动对焦搜索策略。

实现的策略：
  - HillClimbSearch      : 试探方向 + 沿方向递进 + 回退最佳位置（原有算法，新增自适应步长）
  - FullSweepSearch      : 在 [-range, range] 范围内全扫描并回到最佳位置
  - CurveFitSearch       : 等间距采样后抛物线/高斯拟合，跳到预测峰值并局部微调
  - GoldenSectionSearch  : 先粗扫 bracket 峰值，再用黄金分割法细化

参考文献 / 业界常用方法：
  - Bonet Sanz et al., "An algorithm selection methodology for automated focusing in optical microscopy", Microsc Res Tech 2022
  - LeSage & Kron, "Design and implementation of algorithms for focus automation in digital imaging time-lapse microscopy", Cytometry 2002
  - Malivert et al., "Active image optimization for lattice light sheet microscopy", Biomed Opt Express 2022
  - OpenCV autofocus focus measures comparative study
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import AutofocusConfig

MoveFn = Callable[[int], None]
MeasureFn = Callable[[str, int], Tuple[Optional[float], Optional[Dict[str, float]]]]
LogFn = Callable[[str], None]


class BaseFocusSearch:
    """搜索策略基类。"""

    def __init__(
        self,
        cfg: AutofocusConfig,
        move_fn: MoveFn,
        measure_fn: MeasureFn,
        log_fn: LogFn,
    ):
        self.cfg = cfg
        self.move_fn = move_fn
        self.measure_fn = measure_fn
        self.log_fn = log_fn

        self.pos = 0
        self.history: List[Dict[str, Any]] = []
        self._cache: Dict[int, Tuple[Optional[float], Optional[Dict[str, float]]]] = {}

    def _move(self, delta: int) -> None:
        delta = int(round(delta))
        if delta == 0:
            return
        self.move_fn(delta)
        self.pos += delta

    def _settle(self) -> None:
        settle_s = max(0.0, float(getattr(self.cfg, "z_settle_time_s", 0.0)))
        if settle_s > 0:
            time.sleep(settle_s)

    def _measure(
        self,
        phase: str,
        iteration: int,
        use_cache: bool = False,
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """在 self.pos 处测量分数。可选缓存，避免同一位置重复拍照。"""
        if use_cache and self.pos in self._cache:
            score, comp = self._cache[self.pos]
        else:
            score, comp = self.measure_fn(phase, iteration)
            if use_cache:
                self._cache[self.pos] = (score, comp)

        entry: Dict[str, Any] = {
            "iter": iteration,
            "relative_z_steps": self.pos,
            "score": score,
            "phase": phase,
        }
        if comp is not None:
            entry["component_ratios"] = comp
        self.history.append(entry)
        if score is not None:
            self.log_fn(
                f"[补焦-{phase}] iter={iteration}, z={self.pos}, score={score:.4f}"
            )
        else:
            self.log_fn(
                f"[补焦-{phase}] iter={iteration}, z={self.pos}, score=None"
            )
        return score, comp

    def _goto(
        self,
        target_pos: int,
        phase: str,
        iteration: int,
        use_cache: bool = False,
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """移动到 target_pos 并测量。"""
        self._move(target_pos - self.pos)
        self._settle()
        return self._measure(phase, iteration, use_cache=use_cache)

    def search(
        self,
        initial_score: Optional[float],
        target: float,
        max_total_steps: int,
        max_iter: int,
        patience: int,
        min_improve: float,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class HillClimbSearch(BaseFocusSearch):
    """
    改进的爬山搜索。

    在原算法基础上增加：
      - 自适应步长：连续无进步时按 z_adaptive_step_decay 缩小 search_steps；
      - 方向确定时直接使用动态步长。
    """

    def search(
        self,
        initial_score: Optional[float],
        target: float,
        max_total_steps: int,
        max_iter: int,
        patience: int,
        min_improve: float,
    ) -> Dict[str, Any]:
        probe_steps = max(1, int(self.cfg.z_probe_steps))
        search_steps = max(1, int(self.cfg.z_search_steps))
        decay = max(0.0, min(1.0, float(self.cfg.z_adaptive_step_decay)))

        if initial_score is None:
            score, _ = self._measure("start", 0)
            initial_score = score
        else:
            self._measure("start", 0)

        if initial_score is None:
            return {"ok": False, "reason": "no_initial_score"}

        best_score = float(initial_score)
        best_pos = 0

        if best_score >= target:
            self.log_fn(
                f"[补焦] 初始 FocusScore={best_score:.4f} >= {target}，无需移动"
            )
            return {
                "ok": True,
                "reason": "already_above_target",
                "initial_score": initial_score,
                "best_score": best_score,
                "best_relative_z_steps": 0,
                "history": self.history,
            }

        # ---- 试探 + 方向 ----
        direction = None
        score_plus, _ = self._goto(+probe_steps, "probe_plus", 1)

        if score_plus is not None and score_plus > best_score * (1.0 + min_improve):
            direction = +1
            best_score = float(score_plus)
            best_pos = self.pos
        else:
            # 回到原点再试反方向
            self._goto(0, "probe_return", 2)
            score_minus, _ = self._goto(-probe_steps, "probe_minus", 3)

            if score_minus is not None and score_minus > best_score * (1.0 + min_improve):
                direction = -1
                best_score = float(score_minus)
                best_pos = self.pos
            else:
                # 回到原点
                if self.pos != 0:
                    self._goto(0, "probe_return", 4)
                self.log_fn("[补焦] 双向试探均无明显提升，停止")
                return {
                    "ok": True,
                    "reason": "both_directions_no_improve",
                    "initial_score": initial_score,
                    "best_score": best_score,
                    "best_relative_z_steps": 0,
                    "history": self.history,
                }

        # ---- 沿方向递进搜索 ----
        no_improve_count = 0
        iteration = 4
        while iteration <= max_iter:
            if best_score >= target:
                self.log_fn(
                    f"[补焦] best_score={best_score:.4f} >= target={target}，停止"
                )
                break
            if abs(self.pos) >= max_total_steps:
                self.log_fn(
                    f"[补焦] 达最大步数 |{self.pos}| >= {max_total_steps}，停止"
                )
                break

            self._goto(self.pos + int(direction * search_steps), "search", iteration)
            score = self.history[-1]["score"]

            improved = score is not None and score > best_score * (1.0 + min_improve)
            if improved:
                best_score = float(score)
                best_pos = self.pos
                no_improve_count = 0
            else:
                no_improve_count += 1
                if decay < 1.0:
                    search_steps = max(1, int(search_steps * decay))
                    self.log_fn(
                        f"[补焦] 无进步，步长衰减为 {search_steps}"
                    )

            if no_improve_count >= patience:
                self.log_fn(f"[补焦] 连续 {no_improve_count} 次无提升，停止")
                break
            iteration += 1

        # ---- 回退到最佳位置 ----
        if best_pos != self.pos:
            self.log_fn(
                f"[补焦] 回退到最佳位置: best_pos={best_pos}, current_pos={self.pos}"
            )
            self._goto(best_pos, "return_to_best", iteration + 1)

        ok = best_score >= min(target, float(self.cfg.autofocus_focus_trigger_ratio))
        return {
            "ok": bool(ok),
            "reason": "done",
            "initial_score": float(initial_score),
            "best_score": float(best_score),
            "best_relative_z_steps": int(best_pos),
            "final_relative_z_steps": int(self.pos),
            "target": target,
            "history": self.history,
        }


class FullSweepSearch(BaseFocusSearch):
    """在 [-range, range] 等间距全扫描，返回最佳位置。"""

    def search(
        self,
        initial_score: Optional[float],
        target: float,
        max_total_steps: int,
        max_iter: int,
        patience: int,
        min_improve: float,
    ) -> Dict[str, Any]:
        half_range = min(max_total_steps, max(1, int(self.cfg.z_sweep_range_steps)))
        step = max(1, int(self.cfg.z_search_steps))

        if initial_score is None:
            score, _ = self._measure("start", 0)
            initial_score = score
        else:
            self.history.append(
                {
                    "iter": 0,
                    "relative_z_steps": 0,
                    "score": initial_score,
                    "phase": "start",
                }
            )

        if initial_score is None:
            return {"ok": False, "reason": "no_initial_score"}

        best_score = float(initial_score)
        best_pos = 0

        positions = [0]
        p = step
        while p <= half_range:
            positions.extend([-p, p])
            p += step
        positions = sorted(set(positions))

        iteration = 1
        for pos in positions:
            if pos == 0:
                continue
            if abs(pos) > max_total_steps:
                continue
            score, _ = self._goto(pos, "sweep", iteration, use_cache=True)
            iteration += 1
            if score is not None and score > best_score:
                best_score = float(score)
                best_pos = pos
            if score is not None and score >= target:
                self.log_fn(
                    f"[全扫] 已达目标 score={score:.4f} >= {target}"
                )
                break
            if iteration > max_iter:
                break

        # 回到最佳位置
        if best_pos != self.pos:
            self._goto(best_pos, "return_to_best", iteration, use_cache=True)

        ok = best_score >= min(target, float(self.cfg.autofocus_focus_trigger_ratio))
        return {
            "ok": bool(ok),
            "reason": "sweep_done",
            "initial_score": float(initial_score),
            "best_score": float(best_score),
            "best_relative_z_steps": int(best_pos),
            "final_relative_z_steps": int(self.pos),
            "target": target,
            "history": self.history,
        }


class CurveFitSearch(BaseFocusSearch):
    """
    曲线拟合搜索。

    1. 在 [-range, range] 等间距采集 n 个点；
    2. 用最小二乘拟合抛物线 y = a x^2 + b x + c；
       若 a < 0，峰值位置 xp = -b / (2a)；
    3. 移动到预测峰值并局部微调。
    """

    def search(
        self,
        initial_score: Optional[float],
        target: float,
        max_total_steps: int,
        max_iter: int,
        patience: int,
        min_improve: float,
    ) -> Dict[str, Any]:
        half_range = min(max_total_steps, max(1, int(self.cfg.z_sweep_range_steps)))
        n_points = max(5, int(self.cfg.z_curve_fit_points))
        if n_points % 2 == 0:
            n_points += 1

        if initial_score is None:
            score, _ = self._measure("start", 0)
            initial_score = score
        else:
            self.history.append(
                {
                    "iter": 0,
                    "relative_z_steps": 0,
                    "score": initial_score,
                    "phase": "start",
                }
            )

        if initial_score is None:
            return {"ok": False, "reason": "no_initial_score"}

        # 等间距采样
        step = max(1, int(round(2 * half_range / (n_points - 1))))
        positions = [int(round(-half_range + i * step)) for i in range(n_points)]
        positions = [max(-max_total_steps, min(max_total_steps, p)) for p in positions]
        positions = sorted(set(positions))

        measured: List[Tuple[int, Optional[float]]] = [(0, initial_score)]
        iteration = 1
        for pos in positions:
            if pos == 0:
                continue
            score, _ = self._goto(pos, "curve_sample", iteration, use_cache=True)
            iteration += 1
            measured.append((pos, score))

        valid = [(p, s) for p, s in measured if s is not None]
        if not valid:
            return {"ok": False, "reason": "no_valid_samples"}

        best_pos = max(valid, key=lambda x: x[1])[0]
        best_score = max(s for _, s in valid)

        if best_score >= target:
            self._goto(best_pos, "return_to_best", iteration, use_cache=True)
            return {
                "ok": True,
                "reason": "target_reached_by_sample",
                "initial_score": float(initial_score),
                "best_score": float(best_score),
                "best_relative_z_steps": int(best_pos),
                "final_relative_z_steps": int(self.pos),
                "target": target,
                "history": self.history,
            }

        # 抛物线拟合
        xs = np.array([float(p) for p, _ in valid])
        ys = np.array([float(s) for _, s in valid])
        try:
            coeffs = np.polyfit(xs, ys, 2)
            a, b, c = coeffs
        except Exception:
            a, b, c = 0.0, 0.0, 0.0

        predicted_peak = None
        if a < -1e-12:
            predicted_peak = int(round(-b / (2.0 * a)))
            predicted_peak = max(-max_total_steps, min(max_total_steps, predicted_peak))

        if predicted_peak is not None and predicted_peak != best_pos:
            score_peak, _ = self._goto(
                predicted_peak, "curve_peak", iteration, use_cache=True
            )
            iteration += 1
            if score_peak is not None and score_peak > best_score:
                best_score = float(score_peak)
                best_pos = predicted_peak

        # 局部微调
        refine_step = max(1, int(step // 2))
        no_improve = 0
        refine_patience = max(1, patience)
        while iteration <= max_iter and no_improve < refine_patience:
            candidates = [
                (best_pos + refine_step, "refine_plus"),
                (best_pos - refine_step, "refine_minus"),
            ]
            local_best = best_score
            local_best_pos = best_pos
            for pos, phase in candidates:
                pos = max(-max_total_steps, min(max_total_steps, pos))
                score, _ = self._goto(pos, phase, iteration, use_cache=True)
                iteration += 1
                if score is not None and score > local_best * (1.0 + min_improve):
                    local_best = float(score)
                    local_best_pos = pos
            if local_best_pos != best_pos:
                best_score = local_best
                best_pos = local_best_pos
                no_improve = 0
                if best_score >= target:
                    break
            else:
                no_improve += 1
                refine_step = max(1, int(refine_step * 0.5))

        if best_pos != self.pos:
            self._goto(best_pos, "return_to_best", iteration, use_cache=True)

        ok = best_score >= min(target, float(self.cfg.autofocus_focus_trigger_ratio))
        return {
            "ok": bool(ok),
            "reason": "curve_fit_done",
            "initial_score": float(initial_score),
            "best_score": float(best_score),
            "best_relative_z_steps": int(best_pos),
            "final_relative_z_steps": int(self.pos),
            "target": target,
            "history": self.history,
        }


class GoldenSectionSearch(BaseFocusSearch):
    """
    黄金分割搜索。

    1. 先粗扫确定峰值区间 [a, c]，满足 f(b) >= f(a) 且 f(b) >= f(c)；
    2. 在区间内使用黄金分割法细化，直到区间宽度 <= tol。
    """

    def search(
        self,
        initial_score: Optional[float],
        target: float,
        max_total_steps: int,
        max_iter: int,
        patience: int,
        min_improve: float,
    ) -> Dict[str, Any]:
        half_range = min(max_total_steps, max(1, int(self.cfg.z_sweep_range_steps)))
        tol = max(1, int(self.cfg.z_golden_section_tol))
        coarse_step = max(1, int(self.cfg.z_search_steps))

        if initial_score is None:
            score, _ = self._measure("start", 0)
            initial_score = score
        else:
            self.history.append(
                {
                    "iter": 0,
                    "relative_z_steps": 0,
                    "score": initial_score,
                    "phase": "start",
                }
            )

        if initial_score is None:
            return {"ok": False, "reason": "no_initial_score"}

        # ---- 粗扫 bracket ----
        positions = [0]
        p = coarse_step
        while p <= half_range:
            positions.extend([-p, p])
            p += coarse_step
        positions = sorted(set(positions))

        samples: Dict[int, float] = {0: float(initial_score)}
        iteration = 1
        for pos in positions:
            if pos == 0:
                continue
            if abs(pos) > max_total_steps:
                continue
            score, _ = self._goto(pos, "bracket", iteration, use_cache=True)
            iteration += 1
            if score is not None:
                samples[pos] = float(score)
            if iteration > max_iter:
                break

        # 找到最佳点，用相邻三点作为 bracket
        if not samples:
            return {"ok": False, "reason": "no_bracket_samples"}

        sorted_positions = sorted(samples.keys())
        best_idx = max(range(len(sorted_positions)), key=lambda i: samples[sorted_positions[i]])
        best_pos = sorted_positions[best_idx]
        best_score = samples[best_pos]

        if best_score >= target:
            self._goto(best_pos, "return_to_best", iteration, use_cache=True)
            return {
                "ok": True,
                "reason": "target_reached_by_bracket",
                "initial_score": float(initial_score),
                "best_score": float(best_score),
                "best_relative_z_steps": int(best_pos),
                "final_relative_z_steps": int(self.pos),
                "target": target,
                "history": self.history,
            }

        # 初始化 a < b < c，b 为当前最佳
        if best_idx == 0:
            a = max(-max_total_steps, best_pos - coarse_step)
            b = best_pos
            c = sorted_positions[1] if len(sorted_positions) > 1 else min(max_total_steps, best_pos + coarse_step)
        elif best_idx == len(sorted_positions) - 1:
            a = sorted_positions[-2]
            b = best_pos
            c = min(max_total_steps, best_pos + coarse_step)
        else:
            a = sorted_positions[best_idx - 1]
            b = best_pos
            c = sorted_positions[best_idx + 1]

        a = max(-max_total_steps, a)
        c = min(max_total_steps, c)
        if not (a < b < c):
            # 退化为最佳点
            self._goto(best_pos, "return_to_best", iteration, use_cache=True)
            return {
                "ok": best_score >= min(target, float(self.cfg.autofocus_focus_trigger_ratio)),
                "reason": "invalid_bracket",
                "initial_score": float(initial_score),
                "best_score": float(best_score),
                "best_relative_z_steps": int(best_pos),
                "final_relative_z_steps": int(self.pos),
                "target": target,
                "history": self.history,
            }

        def get_score(z: int) -> float:
            nonlocal iteration
            if z in samples:
                return samples[z]
            score, _ = self._goto(z, "golden", iteration, use_cache=True)
            iteration += 1
            if score is None:
                score = -float("inf")
            samples[z] = float(score)
            return samples[z]

        # 黄金分割（整数位置）
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        resphi = 2.0 - phi  # ~= 0.382

        x1 = int(round(b - resphi * (b - a)))
        x2 = int(round(a + resphi * (c - a)))
        x1 = max(a, min(x2 - 1, x1))
        x2 = min(c, max(x1 + 1, x2))

        f1 = get_score(x1)
        f2 = get_score(x2)

        inner_iter = 0
        max_inner = max_iter - iteration
        while (c - a) > tol and inner_iter < max_inner:
            if f1 < f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = int(round(a + resphi * (c - a)))
                x2 = min(c, max(x1 + 1, x2))
                f2 = get_score(x2)
            else:
                c = x2
                x2 = x1
                f2 = f1
                x1 = int(round(b - resphi * (b - a)))
                x1 = max(a, min(x2 - 1, x1))
                f1 = get_score(x1)
            inner_iter += 1

        # 在最终区间内取最佳实测点
        interval_positions = [p for p in samples.keys() if a <= p <= c]
        if interval_positions:
            final_best_pos = max(interval_positions, key=lambda p: samples[p])
            final_best_score = samples[final_best_pos]
        else:
            final_best_pos = best_pos
            final_best_score = best_score

        if final_best_pos != self.pos:
            self._goto(final_best_pos, "return_to_best", iteration, use_cache=True)

        ok = final_best_score >= min(target, float(self.cfg.autofocus_focus_trigger_ratio))
        return {
            "ok": bool(ok),
            "reason": "golden_section_done",
            "initial_score": float(initial_score),
            "best_score": float(final_best_score),
            "best_relative_z_steps": int(final_best_pos),
            "final_relative_z_steps": int(self.pos),
            "target": target,
            "history": self.history,
        }


def create_search(
    strategy: str,
    cfg: AutofocusConfig,
    move_fn: MoveFn,
    measure_fn: MeasureFn,
    log_fn: LogFn,
) -> BaseFocusSearch:
    """根据策略名创建搜索器。"""
    strategy = (strategy or "hill_climb").lower().strip()
    if strategy == "full_sweep":
        return FullSweepSearch(cfg, move_fn, measure_fn, log_fn)
    if strategy == "curve_fit":
        return CurveFitSearch(cfg, move_fn, measure_fn, log_fn)
    if strategy == "golden_section":
        return GoldenSectionSearch(cfg, move_fn, measure_fn, log_fn)
    return HillClimbSearch(cfg, move_fn, measure_fn, log_fn)
