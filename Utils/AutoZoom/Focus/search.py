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
ShouldStopFn = Callable[[], bool]


class FocusSearchStopped(Exception):
    """用户点击停止时抛出，用于快速中断搜索。"""

    pass


class BaseFocusSearch:
    """搜索策略基类。"""

    def __init__(
        self,
        cfg: AutofocusConfig,
        move_fn: MoveFn,
        measure_fn: MeasureFn,
        log_fn: LogFn,
        should_stop: Optional[ShouldStopFn] = None,
    ):
        self.cfg = cfg
        self.move_fn = move_fn
        self.measure_fn = measure_fn
        self.log_fn = log_fn
        self.should_stop = should_stop

        self.pos = 0
        self.history: List[Dict[str, Any]] = []
        self._cache: Dict[int, Tuple[Optional[float], Optional[Dict[str, float]]]] = {}

    def _check_stop(self, phase: str = "") -> None:
        """若用户请求停止则抛出 FocusSearchStopped。"""
        if self.should_stop is not None and self.should_stop():
            msg = "用户停止"
            if phase:
                msg = f"[{phase}] {msg}"
            self.log_fn(f"[补焦] {msg}")
            raise FocusSearchStopped(msg)

    @staticmethod
    def _in_tolerance(
        score: Optional[float],
        target: float,
        upper_target: Optional[float],
    ) -> bool:
        """判断分数是否在允许区间 [target, upper_target] 内。

        - score < target  → False（偏低，需搜索更好位置）
        - score > upper_target → False（偏高，需搜索峰值后由上层更新参考）
        - target <= score <= upper_target → True（在容差内，无需移动）
        upper_target 为 None 时退化为单边下限 score >= target。
        """
        if score is None:
            return False
        if score < target:
            return False
        if upper_target is not None and score > upper_target:
            return False
        return True

    def _move(self, delta: int) -> None:
        self._check_stop("move")
        delta = int(round(delta))
        if delta == 0:
            return
        self.move_fn(delta)
        self.pos += delta

    def _settle(self) -> None:
        settle_s = max(0.0, float(getattr(self.cfg, "z_settle_time_s", 0.0)))
        if settle_s > 0:
            # 将长时间 sleep 拆分为小段，便于响应停止
            deadline = time.time() + settle_s
            while time.time() < deadline:
                self._check_stop("settle")
                time.sleep(min(0.05, deadline - time.time()))

    def _measure(
        self,
        phase: str,
        iteration: int,
        use_cache: bool = False,
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """在 self.pos 处测量分数。可选缓存，避免同一位置重复拍照。"""
        self._check_stop(phase)
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
        upper_target: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class HillClimbSearch(BaseFocusSearch):
    """
    改进的爬山搜索。

    在原算法基础上增加：
      - 自适应步长：连续无进步时按 z_adaptive_step_decay 缩小 search_steps；
      - 方向确定时先按步长序列做双向采样，再选更优方向；
      - 粗搜后回到 best_pos，用逐步缩小的左右探测做局部细搜。
    """

    def _score_is_better_for_target(
        self,
        score: Optional[float],
        best_score: float,
        target: float,
        upper_target: Optional[float],
        min_improve: float,
    ) -> bool:
        """按目标范围选择候选点；有上限时优先保持在容差区间内。"""
        if score is None:
            return False
        score_f = float(score)
        if upper_target is None:
            return score_f > best_score * (1.0 + min_improve)
        score_in = self._in_tolerance(score_f, target, upper_target)
        best_in = self._in_tolerance(best_score, target, upper_target)
        if score_in:
            if not best_in:
                return True
            return abs(score_f - 1.0) < abs(best_score - 1.0)
        if best_in:
            return False
        return score_f > best_score * (1.0 + min_improve)

    def _measure_median_at_current_position(
        self,
        phase: str,
        iteration: int,
        sample_count: int,
    ) -> Optional[float]:
        """在当前 Z 位置重复采样，返回中位数分数。"""
        scores: List[float] = []
        for sample_idx in range(max(1, int(sample_count))):
            self._check_stop(phase)
            score, _ = self._measure(f"{phase}_sample{sample_idx + 1}", iteration, use_cache=False)
            if score is not None:
                scores.append(float(score))
        if not scores:
            return None
        return float(np.median(scores))

    def _measure_median_at_position(
        self,
        target_pos: int,
        phase: str,
        iteration: int,
        sample_count: int,
    ) -> Optional[float]:
        """移动到指定位置后重复采样，返回中位数分数。"""
        self._move(target_pos - self.pos)
        self._settle()
        return self._measure_median_at_current_position(phase, iteration, sample_count)

    def _determine_direction_with_dynamic_sampling(
        self,
        probe_steps: List[int],
        min_improve: float,
        sample_count: int,
        points_per_step: int = 5,
    ) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[int]]:
        """按基础步长序列做多点采样，选择更优方向。"""
        if not probe_steps:
            probe_steps = [max(1, int(self.cfg.z_probe_steps))]
        points_per_step = max(1, int(points_per_step))

        center_score_final: Optional[float] = None

        for stage_idx, probe_step in enumerate(probe_steps, start=1):
            probe_step = max(1, int(probe_step))
            stage_center_pos = self.pos
            self.log_fn(
                f"[补焦] 方向判断阶段 {stage_idx}/{len(probe_steps)}: "
                f"probe_step={probe_step}, points={points_per_step}"
            )
            center_score = self._measure_median_at_current_position(
                phase=f"direction_center_s{probe_step}",
                iteration=stage_idx * 10 - 2,
                sample_count=sample_count,
            )
            center_score_final = center_score
            if center_score is None:
                self.log_fn(
                    f"[补焦] 中心点采样失败，跳过 probe_step={probe_step}"
                )
                continue

            threshold = center_score * (1.0 + min_improve)
            stage_candidates: List[Tuple[int, int, float, float]] = []
            ratio_base = max(abs(center_score), 1e-12)
            sample_summary: List[str] = []

            for point_idx in range(1, points_per_step + 1):
                target_pos = stage_center_pos + probe_step * point_idx
                probe_score = self._measure_median_at_position(
                    target_pos=target_pos,
                    phase=f"direction_probe_s{probe_step}_p{point_idx}",
                    iteration=stage_idx * 100 + point_idx,
                    sample_count=sample_count,
                )
                sample_summary.append(
                    f"{target_pos}:{probe_score:.4f}" if probe_score is not None else f"{target_pos}:None"
                )
                if probe_score is not None and probe_score > threshold:
                    stage_candidates.append(
                        (
                            +1,
                            int(target_pos),
                            float(probe_score),
                            float(probe_score / ratio_base),
                        )
                    )

            if self.pos != stage_center_pos:
                self._move(stage_center_pos - self.pos)
                self._settle()

            self.log_fn(
                f"[补焦] 方向判断阶段 {stage_idx} 结果: "
                f"center={center_score:.4f}, samples=[{', '.join(sample_summary)}], "
                f"threshold={threshold:.4f}"
            )

            if not stage_candidates:
                continue

            stage_direction, stage_pos, stage_score, stage_ratio = max(
                stage_candidates, key=lambda item: item[2]
            )
            self.log_fn(
                f"[补焦] 动态多点采样确定方向: direction={stage_direction}, "
                f"probe_step={probe_step}, score={stage_score:.4f}, ratio={stage_ratio:.4f}, "
                f"best_pos={stage_pos}"
            )
            return stage_direction, probe_step, stage_score, stage_pos

        if center_score_final is not None:
            self.log_fn(
                f"[补焦] 动态多点采样未找到明显提升，center={center_score_final:.4f}"
            )
        else:
            self.log_fn("[补焦] 动态多点采样全部失败，无法确定方向")
        return None, None, center_score_final, None

    def _refine_around_best(
        self,
        best_score: float,
        best_pos: int,
        initial_step: int,
        target: float,
        upper_target: Optional[float],
        min_improve: float,
        max_total_steps: int,
        iteration: int,
    ) -> Tuple[float, int, int]:
        """回到 best_pos 左右做局部细搜，返回更新后的 best_score/best_pos/iteration。"""
        if not bool(getattr(self.cfg, "z_local_refine_enabled", True)):
            return best_score, best_pos, iteration

        refine_step = max(1, int(initial_step))
        refine_step = max(1, refine_step // 2)
        min_step = max(1, int(getattr(self.cfg, "z_local_refine_min_step", 1)))
        max_rounds = max(0, int(getattr(self.cfg, "z_local_refine_max_rounds", 4)))
        decay = float(getattr(self.cfg, "z_local_refine_decay", 0.5))
        decay = max(0.1, min(0.9, decay))

        if max_rounds <= 0:
            return best_score, best_pos, iteration

        self.log_fn(
            f"[补焦] 开始局部细搜: best_pos={best_pos}, step={refine_step}, "
            f"min_step={min_step}, rounds={max_rounds}"
        )

        round_index = 0
        while refine_step >= min_step and round_index < max_rounds:
            self._check_stop("local_refine")
            round_index += 1

            if self.pos != best_pos:
                iteration += 1
                self._goto(best_pos, "local_refine_center", iteration, use_cache=True)

            round_improved = False
            candidates = [best_pos - refine_step, best_pos + refine_step]
            for candidate_pos in candidates:
                self._check_stop("local_refine")
                if abs(candidate_pos) > max_total_steps:
                    self.log_fn(
                        f"[补焦] 局部细搜跳过越界位置: z={candidate_pos}, "
                        f"limit={max_total_steps}"
                    )
                    continue
                iteration += 1
                score, _ = self._goto(
                    candidate_pos, "local_refine", iteration, use_cache=True
                )
                improved = self._score_is_better_for_target(
                    score, best_score, target, upper_target, min_improve
                )
                if improved:
                    best_score = float(score)
                    best_pos = self.pos
                    round_improved = True
                    self.log_fn(
                        f"[补焦] 局部细搜更新最佳: best_pos={best_pos}, "
                        f"best_score={best_score:.4f}"
                    )

            if self.pos != best_pos:
                iteration += 1
                self._goto(best_pos, "local_refine_return", iteration, use_cache=True)

            next_step = max(min_step, int(refine_step * decay))
            if next_step >= refine_step:
                next_step = refine_step - 1
            refine_step = next_step
            if refine_step < min_step:
                break
            if not round_improved:
                self.log_fn(
                    f"[补焦] 局部细搜本轮无提升，继续缩小步长为 {refine_step}"
                )

        return best_score, best_pos, iteration

    def search(
        self,
        initial_score: Optional[float],
        target: float,
        max_total_steps: int,
        max_iter: int,
        patience: int,
        min_improve: float,
        upper_target: Optional[float] = None,
    ) -> Dict[str, Any]:
        probe_steps = max(1, int(self.cfg.z_probe_steps))
        search_steps = max(1, int(self.cfg.z_search_steps))
        decay = max(0.0, min(1.0, float(self.cfg.z_adaptive_step_decay)))
        stage_count = int(getattr(self.cfg, "z_direction_probe_stage_count", 0) or 0)
        step_interval = int(getattr(self.cfg, "z_direction_probe_step_interval", 0) or 0)
        if stage_count > 0 and step_interval > 0:
            probe_schedule = [step_interval * i for i in range(1, stage_count + 1)]
        else:
            probe_schedule_raw = getattr(self.cfg, "z_direction_probe_steps", (probe_steps,))
            if isinstance(probe_schedule_raw, str):
                probe_schedule = [
                    int(v.strip())
                    for v in probe_schedule_raw.split(",")
                    if v.strip()
                ]
            elif isinstance(probe_schedule_raw, (tuple, list)):
                probe_schedule = [int(v) for v in probe_schedule_raw if int(v) > 0]
            else:
                probe_schedule = [int(probe_steps)]
        sample_count = max(1, int(getattr(self.cfg, "z_direction_probe_samples", 3)))
        points_per_step = max(1, int(getattr(self.cfg, "z_direction_probe_points_per_step", 5)))

        if initial_score is None:
            score, _ = self._measure("start", 0)
            initial_score = score
        else:
            self._measure("start", 0)

        if initial_score is None:
            return {"ok": False, "reason": "no_initial_score"}

        best_score = float(initial_score)
        best_pos = 0

        if self._in_tolerance(best_score, target, upper_target):
            self.log_fn(
                f"[补焦] 初始 FocusScore={best_score:.4f} 在容差区间内，无需移动"
            )
            return {
                "ok": True,
                "reason": "already_in_tolerance",
                "initial_score": initial_score,
                "best_score": best_score,
                "best_relative_z_steps": 0,
                "history": self.history,
            }

        # ---- 动态多点采样 + 方向；本轮只判断一次，后续续搜复用该方向 ----
        direction, chosen_probe_step, chosen_score, chosen_pos = self._determine_direction_with_dynamic_sampling(
            probe_steps=probe_schedule,
            min_improve=min_improve,
            sample_count=sample_count,
            points_per_step=points_per_step,
        )
        if direction is None or chosen_probe_step is None or chosen_score is None or chosen_pos is None:
            if self.pos != 0:
                self._goto(0, "probe_return", 4)
            return {
                "ok": True,
                "reason": "both_directions_no_improve",
                "initial_score": initial_score,
                "best_score": best_score,
                "best_relative_z_steps": 0,
                "direction": None,
                "direction_probe_step": None,
                "direction_marked": False,
                "history": self.history,
            }

        direction_marked = True
        best_score = float(chosen_score)
        best_pos = int(chosen_pos)
        if self.pos != best_pos:
            self.log_fn(
                f"[补焦] 移动到方向探测最佳点: best_pos={best_pos}, current_pos={self.pos}"
            )
            self._move(best_pos - self.pos)
            self._settle()

        # ---- 沿已标记方向递进搜索 ----
        no_improve_count = 0
        iteration = 4
        while iteration <= max_iter:
            self._check_stop("hill_climb")
            if self._in_tolerance(best_score, target, upper_target):
                self.log_fn(
                    f"[补焦] best_score={best_score:.4f} 进入容差区间，停止"
                )
                break
            if abs(self.pos) >= max_total_steps:
                self.log_fn(
                    f"[补焦] 达最大步数 |{self.pos}| >= {max_total_steps}，停止"
                )
                break

            self._goto(self.pos + int(direction * search_steps), "search", iteration)
            score = self.history[-1]["score"]

            improved = self._score_is_better_for_target(
                score, best_score, target, upper_target, min_improve
            )
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

        # ---- 回退到最佳位置；只有进入目标范围后才允许停止或局部细搜 ----
        if best_pos != self.pos:
            self.log_fn(
                f"[补焦] 回退到最佳位置: best_pos={best_pos}, current_pos={self.pos}"
            )
            self._goto(best_pos, "return_to_best", iteration + 1)
            iteration += 1

        while (
            self._in_tolerance(best_score, target, upper_target)
            and bool(getattr(self.cfg, "z_local_refine_enabled", True))
        ):
            before_refine_iteration = iteration
            best_score, best_pos, iteration = self._refine_around_best(
                best_score=best_score,
                best_pos=best_pos,
                initial_step=search_steps,
                target=target,
                upper_target=upper_target,
                min_improve=min_improve,
                max_total_steps=max_total_steps,
                iteration=iteration,
            )

            if best_pos != self.pos:
                self.log_fn(
                    f"[补焦] 细搜后回退到最佳位置: best_pos={best_pos}, current_pos={self.pos}"
                )
                self._goto(best_pos, "return_to_best", iteration + 1)
                iteration += 1

            if self._in_tolerance(best_score, target, upper_target):
                break
            if iteration <= before_refine_iteration:
                iteration += 1
            self.log_fn(
                "[补焦] 局部细搜后仍未进入目标范围，继续沿已标记方向补焦，不重新方向判断"
            )
            no_improve_count = 0
            while iteration <= max_iter:
                self._check_stop("hill_climb_continue")
                if self._in_tolerance(best_score, target, upper_target):
                    break
                if abs(self.pos) >= max_total_steps:
                    self.log_fn(
                        f"[补焦] 达最大步数 |{self.pos}| >= {max_total_steps}，停止"
                    )
                    break
                self._goto(self.pos + int(direction * search_steps), "search_continue", iteration)
                score = self.history[-1]["score"]
                improved = self._score_is_better_for_target(
                    score, best_score, target, upper_target, min_improve
                )
                if improved:
                    best_score = float(score)
                    best_pos = self.pos
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count >= patience:
                    self.log_fn(f"[补焦] 继续补焦连续 {no_improve_count} 次无提升，停止")
                    break
                iteration += 1

            if best_pos != self.pos:
                self.log_fn(
                    f"[补焦] 继续补焦后回退到最佳位置: best_pos={best_pos}, current_pos={self.pos}"
                )
                self._goto(best_pos, "return_to_best", iteration + 1)
                iteration += 1

        if not self._in_tolerance(best_score, target, upper_target):
            self.log_fn(
                f"[补焦] best_score={best_score:.4f} 尚未进入目标范围，"
                "本轮已复用标记方向且不再重新方向判断"
            )

        ok = self._in_tolerance(best_score, target, upper_target)
        return {
            "ok": bool(ok),
            "reason": "done",
            "initial_score": float(initial_score),
            "best_score": float(best_score),
            "best_relative_z_steps": int(best_pos),
            "final_relative_z_steps": int(self.pos),
            "target": target,
            "direction": int(direction),
            "direction_probe_step": int(chosen_probe_step),
            "direction_marked": bool(direction_marked),
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
        upper_target: Optional[float] = None,
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
            self._check_stop("full_sweep")
            if pos == 0:
                continue
            if abs(pos) > max_total_steps:
                continue
            score, _ = self._goto(pos, "sweep", iteration, use_cache=True)
            iteration += 1
            if score is not None and score > best_score:
                best_score = float(score)
                best_pos = pos
            if score is not None and self._in_tolerance(score, target, upper_target):
                self.log_fn(
                    f"[全扫] 已进入容差区间 score={score:.4f}"
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
        upper_target: Optional[float] = None,
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
            self._check_stop("curve_sample")
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

        if self._in_tolerance(best_score, target, upper_target):
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
            self._check_stop("curve_refine")
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
                if self._in_tolerance(best_score, target, upper_target):
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
        upper_target: Optional[float] = None,
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
            self._check_stop("bracket")
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

        if self._in_tolerance(best_score, target, upper_target):
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
            self._check_stop("golden_section")
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
    should_stop: Optional[ShouldStopFn] = None,
) -> BaseFocusSearch:
    """根据策略名创建搜索器。"""
    strategy = (strategy or "hill_climb").lower().strip()
    if strategy == "full_sweep":
        return FullSweepSearch(cfg, move_fn, measure_fn, log_fn, should_stop)
    if strategy == "curve_fit":
        return CurveFitSearch(cfg, move_fn, measure_fn, log_fn, should_stop)
    if strategy == "golden_section":
        return GoldenSectionSearch(cfg, move_fn, measure_fn, log_fn, should_stop)
    return HillClimbSearch(cfg, move_fn, measure_fn, log_fn, should_stop)
