"""
运动轨迹记录与分析器 (TrajectoryRecorder)

灵感来源:
- 实验自动化框架 (labscript, artiq) 的数据记录
- 自适应光学系统中的性能监控
- SpotZoom 现有的 RunReporter

功能:
- 记录完整的对准运动轨迹 (位置、偏差、时间戳)
- 生成收敛曲线数据
- 计算关键性能指标 (收敛时间、最终精度、总移动距离)
- 支持 JSON 导出用于后续分析
"""

import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class TrajectoryPoint:
    """单个轨迹数据点。"""
    timestamp_s: float
    detection_center: Optional[Tuple[int, int]]
    target_center: Optional[Tuple[int, int]]
    pixel_error: Optional[Tuple[int, int]]
    confidence: float
    focus_score: float
    action: str  # 'detect', 'move_x', 'move_y', 'move_z_up', 'move_z_down', 'converged'
    iteration: int = 0
    align_round: int = 0


@dataclass
class TrajectorySummary:
    """轨迹分析摘要。"""
    total_duration_s: float
    total_iterations: int
    total_align_rounds: int
    total_detect_attempts: int
    total_detect_success: int
    total_x_moves: int
    total_y_moves: int
    total_z_moves: int
    total_x_distance_steps: int
    total_y_distance_steps: int
    final_pixel_error: Optional[Tuple[int, int]]
    convergence_achieved: bool
    convergence_iteration: Optional[int]
    initial_pixel_error: Optional[Tuple[int, int]]
    avg_convergence_rate: float  # pixels/iteration
    max_pixel_error: Optional[Tuple[int, int]]


class TrajectoryRecorder:
    """对准运动轨迹记录器。

    Parameters
    ----------
    max_points : int
        最大记录点数。超过后自动丢弃最旧的点。
    auto_export_path : str or None
        自动导出路径。为 None 时不自动导出。
    """

    def __init__(self, max_points: int = 5000, auto_export_path: Optional[str] = None):
        self._points: Deque[TrajectoryPoint] = deque(maxlen=max_points)
        self._start_time: float = time.time()
        self._auto_export_path = Path(auto_export_path) if auto_export_path else None
        self._iteration = 0
        self._align_round = 0
        self._total_x_steps = 0
        self._total_y_steps = 0
        self._total_z_moves = 0
        self._total_detect = 0
        self._total_detect_ok = 0
        self._convergence_iteration: Optional[int] = None
        self._initial_error: Optional[Tuple[int, int]] = None

    def record_detection(
        self,
        center: Optional[Tuple[int, int]],
        target: Optional[Tuple[int, int]],
        confidence: float,
        focus_score: float,
    ) -> None:
        """记录一次检测结果。"""
        self._total_detect += 1
        if center is not None:
            self._total_detect_ok += 1

        pixel_error = None
        if center is not None and target is not None:
            pixel_error = (target[0] - center[0], target[1] - center[1])
            if self._initial_error is None:
                self._initial_error = pixel_error

        point = TrajectoryPoint(
            timestamp_s=time.time() - self._start_time,
            detection_center=center,
            target_center=target,
            pixel_error=pixel_error,
            confidence=confidence,
            focus_score=focus_score,
            action="detect",
            iteration=self._iteration,
            align_round=self._align_round,
        )
        self._points.append(point)

    def record_move_x(self, steps: int) -> None:
        """记录 X 轴移动。"""
        self._total_x_steps += abs(steps)
        point = TrajectoryPoint(
            timestamp_s=time.time() - self._start_time,
            detection_center=None, target_center=None, pixel_error=None,
            confidence=0.0, focus_score=0.0,
            action=f"move_x({steps})",
            iteration=self._iteration,
            align_round=self._align_round,
        )
        self._points.append(point)

    def record_move_y(self, steps: int) -> None:
        """记录 Y 轴移动。"""
        self._total_y_steps += abs(steps)
        point = TrajectoryPoint(
            timestamp_s=time.time() - self._start_time,
            detection_center=None, target_center=None, pixel_error=None,
            confidence=0.0, focus_score=0.0,
            action=f"move_y({steps})",
            iteration=self._iteration,
            align_round=self._align_round,
        )
        self._points.append(point)

    def record_z_move(self, direction: str) -> None:
        """记录 Z 轴移动。"""
        self._total_z_moves += 1
        point = TrajectoryPoint(
            timestamp_s=time.time() - self._start_time,
            detection_center=None, target_center=None, pixel_error=None,
            confidence=0.0, focus_score=0.0,
            action=f"move_z_{direction}",
            iteration=self._iteration,
            align_round=self._align_round,
        )
        self._points.append(point)

    def set_iteration(self, iteration: int) -> None:
        """设置当前迭代编号。"""
        self._iteration = iteration

    def set_align_round(self, round_idx: int) -> None:
        """设置当前对准轮次编号。"""
        self._align_round = round_idx

    def mark_converged(self) -> None:
        """标记收敛。"""
        self._convergence_iteration = self._iteration
        point = TrajectoryPoint(
            timestamp_s=time.time() - self._start_time,
            detection_center=None, target_center=None, pixel_error=None,
            confidence=0.0, focus_score=0.0,
            action="converged",
            iteration=self._iteration,
            align_round=self._align_round,
        )
        self._points.append(point)

    def get_summary(self) -> TrajectorySummary:
        """生成轨迹分析摘要。"""
        detect_points = [p for p in self._points if p.action == "detect" and p.pixel_error is not None]

        final_error = None
        max_error = None
        if detect_points:
            final_error = detect_points[-1].pixel_error
            max_err_val = 0
            for p in detect_points:
                err_mag = (p.pixel_error[0] ** 2 + p.pixel_error[1] ** 2) ** 0.5
                if err_mag > max_err_val:
                    max_err_val = err_mag
                    max_error = p.pixel_error

        # 平均收敛速率
        avg_rate = 0.0
        if len(detect_points) >= 2 and self._initial_error is not None:
            init_mag = (self._initial_error[0] ** 2 + self._initial_error[1] ** 2) ** 0.5
            if init_mag > 0:
                final_mag = (final_error[0] ** 2 + final_error[1] ** 2) ** 0.5 if final_error else 0
                n_iters = max(self._iteration, 1)
                avg_rate = (init_mag - final_mag) / n_iters

        return TrajectorySummary(
            total_duration_s=round(time.time() - self._start_time, 3),
            total_iterations=self._iteration,
            total_align_rounds=self._align_round,
            total_detect_attempts=self._total_detect,
            total_detect_success=self._total_detect_ok,
            total_x_moves=self._total_x_steps,
            total_y_moves=self._total_y_steps,
            total_z_moves=self._total_z_moves,
            total_x_distance_steps=self._total_x_steps,
            total_y_distance_steps=self._total_y_steps,
            final_pixel_error=final_error,
            convergence_achieved=self._convergence_iteration is not None,
            convergence_iteration=self._convergence_iteration,
            initial_pixel_error=self._initial_error,
            avg_convergence_rate=round(avg_rate, 3),
            max_pixel_error=max_error,
        )

    def export_json(self, path: Optional[str] = None) -> None:
        """导出轨迹数据为 JSON 文件。"""
        export_path = Path(path) if path else self._auto_export_path
        if export_path is None:
            return

        export_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()

        data = {
            "summary": asdict(summary),
            "points": [
                {
                    "ts": round(p.timestamp_s, 4),
                    "center": list(p.detection_center) if p.detection_center else None,
                    "target": list(p.target_center) if p.target_center else None,
                    "error": list(p.pixel_error) if p.pixel_error else None,
                    "conf": round(p.confidence, 4),
                    "focus": round(p.focus_score, 3),
                    "action": p.action,
                    "iter": p.iteration,
                    "round": p.align_round,
                }
                for p in self._points
            ],
        }

        with export_path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

    def get_convergence_data(self) -> List[Tuple[float, float]]:
        """获取收敛曲线数据 (时间, 偏差幅度)。"""
        data = []
        for p in self._points:
            if p.action == "detect" and p.pixel_error is not None:
                mag = (p.pixel_error[0] ** 2 + p.pixel_error[1] ** 2) ** 0.5
                data.append((p.timestamp_s, mag))
        return data

    def reset(self) -> None:
        """重置记录器。"""
        self._points.clear()
        self._start_time = time.time()
        self._iteration = 0
        self._align_round = 0
        self._total_x_steps = 0
        self._total_y_steps = 0
        self._total_z_moves = 0
        self._total_detect = 0
        self._total_detect_ok = 0
        self._convergence_iteration = None
        self._initial_error = None
