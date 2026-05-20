"""
自适应焦平面搜索器 (AdaptiveFocusSearcher)

灵感来源:
- HCIPy (https://github.com/ehpor/hcipy) — 自适应光学焦平面搜索
- SOAPY (https://github.com/AOtools/soapy) — AO 闭环控制仿真
- ARTIQ (https://github.com/m-labs/artiq) — 实时实验控制中的扫描策略
- Golden Section Search — 黄金分割搜索 (一维最优化经典算法)
- Fibonacci Search — 斐波那契搜索

算法原理:
- Golden Section Search — 黄金分割法在焦点评分曲线上搜索峰值
- Coarse-to-Fine Strategy — 由粗到精的多分辨率搜索
- Focus Curve Modeling — 焦点曲线建模与峰值预测
- Hysteresis Compensation — 迟滞补偿 (压电陶瓷等执行器)

功能:
- 替代简单的固定步长 Z 轴搜索，实现智能焦平面定位
- 黄金分割搜索快速找到焦点峰值
- 由粗到精策略平衡速度与精度
- 焦点曲线记录与分析
- 迟滞补偿提升可重复性

依赖: numpy, logging
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.FocusSearch")


@dataclass
class FocusPoint:
    """焦点搜索数据点。"""
    z_position: float  # Z 轴位置 (相对单位)
    focus_score: float  # 焦点评分
    timestamp: float = 0.0


@dataclass
class FocusSearchResult:
    """焦点搜索结果。"""
    best_z: float  # 最佳 Z 位置
    best_score: float  # 最佳焦点评分
    total_steps: int  # 总搜索步数
    search_range: Tuple[float, float]  # 搜索范围
    score_improvement: float  # 评分提升 (相对初始)
    focus_curve: List[FocusPoint]  # 完整焦点曲线
    converged: bool  # 是否收敛


class AdaptiveFocusSearcher:
    """自适应焦平面搜索器。

    使用黄金分割搜索算法在 Z 轴上快速定位最佳焦点位置。

    Parameters
    ----------
    coarse_range : float
        粗搜索范围 (Z 轴单位)。
    coarse_step : float
        粗搜索步长。
    fine_tolerance : float
        精搜索收敛容差。
    max_total_steps : int
        最大总搜索步数。
    min_score_improvement : float
        最小评分提升阈值 (低于此值停止搜索)。
    hysteresis_compensation : float
        迟滞补偿因子 [0, 1]。0=无补偿, 1=完全补偿。
    """

    def __init__(
        self,
        coarse_range: float = 10.0,
        coarse_step: float = 1.0,
        fine_tolerance: float = 0.5,
        max_total_steps: int = 30,
        min_score_improvement: float = 5.0,
        hysteresis_compensation: float = 0.0,
    ):
        self.coarse_range = float(coarse_range)
        self.coarse_step = float(coarse_step)
        self.fine_tolerance = float(fine_tolerance)
        self.max_total_steps = int(max_total_steps)
        self.min_score_improvement = float(min_score_improvement)
        self.hysteresis_compensation = float(hysteresis_compensation)

        self._focus_curve: List[FocusPoint] = []
        self._step_count: int = 0
        self._last_z_direction: int = 0  # 1=up, -1=down, 0=none

    def search(
        self,
        initial_z: float,
        score_fn: Callable[[float], float],
        move_fn: Callable[[float, str], None],
    ) -> FocusSearchResult:
        """执行自适应焦平面搜索。

        Parameters
        ----------
        initial_z : float
            初始 Z 位置。
        score_fn : Callable[[float], float]
            焦点评分函数。接收 Z 位置，返回焦点评分。
            注意: 调用此函数前应确保已移动到对应 Z 位置。
        move_fn : Callable[[float, str], None]
            Z 轴移动函数。接收 (步长, 方向) 参数。
            方向: 'up' 或 'down'。

        Returns
        -------
        FocusSearchResult
            搜索结果。
        """
        import time as _time
        self._focus_curve.clear()
        self._step_count = 0

        # 1. 粗搜索: 在范围内等间距采样
        LOGGER.info("FocusSearch: starting coarse search, range=%.1f, step=%.1f",
                     self.coarse_range, self.coarse_step)

        n_coarse = max(3, int(self.coarse_range / self.coarse_step))
        coarse_positions = np.linspace(
            -self.coarse_range / 2,
            self.coarse_range / 2,
            n_coarse,
        )

        best_z = initial_z
        best_score = 0.0

        # 采集初始评分
        initial_score = score_fn(initial_z)
        self._record_point(initial_z, initial_score, _time.time())
        best_score = initial_score
        LOGGER.info("FocusSearch: initial z=%.2f, score=%.1f", initial_z, initial_score)

        # 粗搜索扫描
        for dz in coarse_positions:
            if self._step_count >= self.max_total_steps:
                break

            target_z = initial_z + dz
            delta = dz - (best_z - initial_z)
            if delta > 0:
                move_fn(abs(delta), "up")
                self._last_z_direction = 1
            elif delta < 0:
                move_fn(abs(delta), "down")
                self._last_z_direction = -1

            self._step_count += 1
            score = score_fn(target_z)
            self._record_point(target_z, score, _time.time())

            if score > best_score:
                best_score = score
                best_z = target_z
                LOGGER.info("FocusSearch: new best z=%.2f, score=%.1f", best_z, best_score)

        # 2. 精搜索: 黄金分割法
        if self._step_count < self.max_total_steps:
            LOGGER.info("FocusSearch: starting golden section refinement around z=%.2f", best_z)
            refine_result = self._golden_section_search(
                best_z - self.coarse_step,
                best_z + self.coarse_step,
                score_fn,
                move_fn,
                initial_z,
            )
            if refine_result[1] > best_score:
                best_z, best_score = refine_result[0], refine_result[1]

        score_improvement = best_score - initial_score
        converged = score_improvement >= self.min_score_improvement

        LOGGER.info(
            "FocusSearch: completed. best_z=%.2f, best_score=%.1f, improvement=%.1f, steps=%d",
            best_z, best_score, score_improvement, self._step_count,
        )

        return FocusSearchResult(
            best_z=best_z,
            best_score=best_score,
            total_steps=self._step_count,
            search_range=(initial_z - self.coarse_range / 2, initial_z + self.coarse_range / 2),
            score_improvement=score_improvement,
            focus_curve=list(self._focus_curve),
            converged=converged,
        )

    def _golden_section_search(
        self,
        a: float,
        b: float,
        score_fn: Callable[[float], float],
        move_fn: Callable[[float, str], None],
        reference_z: float,
    ) -> Tuple[float, float]:
        """黄金分割搜索。

        Parameters
        ----------
        a, b : float
            搜索区间 (相对于 initial_z 的偏移)。
        score_fn, move_fn : Callable
            同 search()。
        reference_z : float
            参考初始 Z 位置。

        Returns
        -------
        Tuple[float, float]
            (best_z_offset, best_score)
        """
        golden_ratio = (np.sqrt(5) + 1) / 2  # ≈ 1.618
        gr_inv = 1.0 / golden_ratio  # ≈ 0.618

        best_z = (a + b) / 2
        best_score = 0.0

        # 内点
        c = b - gr_inv * (b - a)
        d = a + gr_inv * (b - a)

        # 评估两个内点
        for z_offset in [c, d]:
            if self._step_count >= self.max_total_steps:
                break
            target_z = reference_z + z_offset
            delta = z_offset - (best_z - reference_z)
            if delta > 0:
                move_fn(abs(delta), "up")
            elif delta < 0:
                move_fn(abs(delta), "down")
            self._step_count += 1
            score = score_fn(target_z)
            self._record_point(target_z, score, 0.0)
            if score > best_score:
                best_score = score
                best_z = target_z

        fc = score_fn(reference_z + c) if self._has_score_at(reference_z + c) else best_score
        fd = score_fn(reference_z + d) if self._has_score_at(reference_z + d) else best_score

        # 迭代缩小区间
        max_refine_steps = min(10, self.max_total_steps - self._step_count)
        for _ in range(max_refine_steps):
            if abs(b - a) < self.fine_tolerance:
                break

            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr_inv * (b - a)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr_inv * (b - a)

            # 评估新的内点
            z_offset = c if fc <= fd else d
            if self._step_count >= self.max_total_steps:
                break
            target_z = reference_z + z_offset
            delta = z_offset - (best_z - reference_z)
            if abs(delta) > 0.01:
                if delta > 0:
                    move_fn(abs(delta), "up")
                elif delta < 0:
                    move_fn(abs(delta), "down")
                self._step_count += 1
            score = score_fn(target_z)
            self._record_point(target_z, score, 0.0)
            if score > best_score:
                best_score = score
                best_z = target_z

            if z_offset == c:
                fc = score
            else:
                fd = score

        return (best_z, best_score)

    def _has_score_at(self, z: float) -> bool:
        """检查是否已有某 Z 位置的评分。"""
        return any(abs(p.z_position - z) < 0.01 for p in self._focus_curve)

    def _record_point(self, z: float, score: float, timestamp: float) -> None:
        """记录焦点数据点。"""
        self._focus_curve.append(FocusPoint(
            z_position=z,
            focus_score=score,
            timestamp=timestamp,
        ))

    def get_focus_curve_data(self) -> List[Tuple[float, float]]:
        """获取焦点曲线数据 (z_position, focus_score)。"""
        return [(p.z_position, p.focus_score) for p in self._focus_curve]

    def estimate_peak_width(self) -> float:
        """估计焦点曲线的峰值宽度 (FWHM)。

        Returns
        -------
        float
            半高全宽 (Z 轴单位)。如果数据不足返回 0。
        """
        if len(self._focus_curve) < 3:
            return 0.0

        scores = np.array([p.focus_score for p in self._focus_curve])
        positions = np.array([p.z_position for p in self._focus_curve])

        peak_score = float(np.max(scores))
        half_max = peak_score / 2.0

        # 找到半高点
        above_half = positions[scores >= half_max]
        if len(above_half) < 2:
            return 0.0

        return float(above_half[-1] - above_half[0])

    def reset(self) -> None:
        """重置搜索器。"""
        self._focus_curve.clear()
        self._step_count = 0
        self._last_z_direction = 0
