"""
Convergence Guard - 收敛保护器

Inspired by:
- Bluesky (bluesky): Plan-based experiment execution with abort handling
- ARTIQ (m-labs): Real-time experiment control with watchdog
- python-control: Stability analysis and guard filters

Core Innovation:
- 多维度收敛判断: 位置/速度/加速度/振荡
- 早停机制: 检测发散趋势提前终止
- 收敛质量评分: 量化对准精度
- 异常模式识别: 检测极限环、振荡、漂移
- 纯 numpy 实现，零外部依赖
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class ConvergenceGuardConfig:
    """收敛保护器配置"""
    # 位置收敛阈值 (像素)
    position_threshold_px: float = 2.0
    # 速度收敛阈值 (像素/帧)
    velocity_threshold_px: float = 0.5
    # 加速度阈值 (像素/帧²)
    acceleration_threshold_px: float = 0.1
    # 收敛确认帧数
    convergence_confirm_frames: int = 5
    # 发散检测阈值 (连续增大帧数)
    divergence_threshold: int = 10
    # 振荡检测窗口
    oscillation_window: int = 20
    # 振荡检测阈值 (方向变化次数)
    oscillation_threshold: int = 8
    # 最大运行帧数
    max_frames: int = 5000
    # 极限环检测半径 (像素)
    limit_cycle_radius_px: float = 3.0
    # 是否启用早停
    early_stop_enabled: bool = True


@dataclass
class ConvergenceReport:
    """收敛分析报告"""
    # 是否已收敛
    is_converged: bool
    # 是否发散
    is_diverging: bool
    # 是否振荡
    is_oscillating: bool
    # 是否极限环
    is_limit_cycle: bool
    # 收敛质量评分 (0-100)
    convergence_score: float
    # 当前位置误差 (像素)
    position_error: float
    # 当前速度 (像素/帧)
    velocity: float
    # 收敛帧数
    convergence_frame: int
    # 总运行帧数
    total_frames: int
    # 建议动作
    recommended_action: str
    # 诊断信息
    diagnostics: dict


class ConvergenceGuard:
    """收敛保护器

    监控对准过程的收敛状态，检测异常模式，
    提供早停和恢复建议。

    Inspired by Bluesky's plan execution with abort handling
    and ARTIQ's real-time watchdog mechanism.
    """

    def __init__(self, config: Optional[ConvergenceGuardConfig] = None):
        self.config = config or ConvergenceGuardConfig()
        self._error_history: Deque[float] = deque(maxlen=200)
        self._position_history: Deque[Tuple[float, float]] = deque(maxlen=200)
        self._convergence_counter: int = 0
        self._divergence_counter: int = 0
        self._total_frames: int = 0
        self._convergence_frame: int = -1
        self._prev_error: float = 0.0
        self._prev_velocity: float = 0.0
        self._direction_changes: int = 0

    def reset(self) -> None:
        """重置保护器状态"""
        self._error_history.clear()
        self._position_history.clear()
        self._convergence_counter = 0
        self._divergence_counter = 0
        self._total_frames = 0
        self._convergence_frame = -1
        self._prev_error = 0.0
        self._prev_velocity = 0.0
        self._direction_changes = 0

    def _compute_velocity(self) -> float:
        """计算当前位置变化速度"""
        if len(self._error_history) < 2:
            return 0.0
        errors = list(self._error_history)
        return float(errors[-1] - errors[-2])

    def _compute_acceleration(self) -> float:
        """计算加速度"""
        if len(self._error_history) < 3:
            return 0.0
        errors = list(self._error_history)
        v1 = errors[-1] - errors[-2]
        v2 = errors[-2] - errors[-3]
        return float(v1 - v2)

    def _detect_oscillation(self) -> bool:
        """检测振荡模式"""
        if len(self._error_history) < self.config.oscillation_window:
            return False

        recent = list(self._error_history)[-self.config.oscillation_window:]

        # 计算方向变化次数
        changes = 0
        for i in range(1, len(recent)):
            if (recent[i] - recent[i - 1]) * (recent[i - 1] - recent[i - 2]) < 0:
                changes += 1

        return changes >= self.config.oscillation_threshold

    def _detect_limit_cycle(self) -> bool:
        """检测极限环"""
        if len(self._position_history) < self.config.oscillation_window:
            return False

        positions = list(self._position_history)[-self.config.oscillation_window:]
        positions = np.array(positions)

        # 计算质心
        centroid = np.mean(positions, axis=0)

        # 计算到质心的距离
        distances = np.sqrt(np.sum((positions - centroid) ** 2, axis=1))

        # 如果距离集中在某个范围，可能是极限环
        if np.mean(distances) < self.config.limit_cycle_radius_px:
            return False  # 太小，不是极限环

        # 检查距离方差是否小 (一致的距离 = 极限环)
        if np.std(distances) < np.mean(distances) * 0.5:
            return True

        return False

    def _detect_divergence(self) -> bool:
        """检测发散趋势"""
        if len(self._error_history) < 5:
            return False

        recent = list(self._error_history)[-5:]
        # 检查误差是否持续增大
        increasing = all(recent[i] > recent[i - 1] for i in range(1, len(recent)))
        if increasing and recent[-1] > recent[0] * 2:
            return True
        return False

    def _compute_convergence_score(self) -> float:
        """计算收敛质量评分 (0-100)"""
        if len(self._error_history) < 3:
            return 0.0

        errors = np.array(list(self._error_history))

        # 1. 最终误差评分 (40%)
        final_error = abs(errors[-1])
        error_score = max(0, 100 * (1 - final_error / (self.config.position_threshold_px * 5)))

        # 2. 收敛速度评分 (30%)
        if errors[0] > 1e-6:
            decay = errors[-1] / errors[0]
            speed_score = max(0, 100 * (1 - decay))
        else:
            speed_score = 100.0

        # 3. 稳定性评分 (30%)
        if len(errors) >= 10:
            recent_errors = errors[-10:]
            stability = 1.0 - np.std(recent_errors) / (np.mean(np.abs(recent_errors)) + 1e-6)
            stability_score = max(0, 100 * stability)
        else:
            stability_score = 50.0

        return float(0.4 * error_score + 0.3 * speed_score + 0.3 * stability_score)

    def update(
        self, error_x: float, error_y: float
    ) -> ConvergenceReport:
        """更新收敛状态

        Args:
            error_x: X 方向误差 (像素)
            error_y: Y 方向误差 (像素)

        Returns:
            ConvergenceReport 收敛分析报告
        """
        self._total_frames += 1
        position_error = float(np.sqrt(error_x ** 2 + error_y ** 2))
        self._error_history.append(position_error)
        self._position_history.append((error_x, error_y))

        velocity = self._compute_velocity()
        acceleration = self._compute_acceleration()

        # 检测各种异常模式
        is_oscillating = self._detect_oscillation()
        is_limit_cycle = self._detect_limit_cycle()
        is_diverging = self._detect_divergence()

        # 收敛判断
        is_converged = False
        if (position_error < self.config.position_threshold_px
                and abs(velocity) < self.config.velocity_threshold_px
                and abs(acceleration) < self.config.acceleration_threshold_px):
            self._convergence_counter += 1
            if self._convergence_counter >= self.config.convergence_confirm_frames:
                is_converged = True
                if self._convergence_frame < 0:
                    self._convergence_frame = self._total_frames
        else:
            self._convergence_counter = 0

        # 发散计数
        if position_error > abs(self._prev_error) * 1.1:
            self._divergence_counter += 1
        else:
            self._divergence_counter = max(0, self._divergence_counter - 1)

        self._prev_error = position_error
        self._prev_velocity = velocity

        # 收敛质量评分
        score = self._compute_convergence_score()

        # 推荐动作
        if is_diverging or self._divergence_counter >= self.config.divergence_threshold:
            action = "ABORT: 检测到发散趋势，建议立即停止并检查系统"
        elif is_oscillating:
            action = "REDUCE_GAIN: 检测到振荡，建议降低 PID 增益"
        elif is_limit_cycle:
            action = "CHANGE_STRATEGY: 检测到极限环，建议切换控制策略"
        elif self._total_frames >= self.config.max_frames:
            action = "TIMEOUT: 超过最大运行帧数，建议终止"
        elif is_converged:
            action = "CONVERGED: 对准已完成"
        elif self._convergence_counter > 0:
            action = "CONVERGING: 正在收敛中..."
        else:
            action = "RUNNING: 正常运行"

        return ConvergenceReport(
            is_converged=is_converged,
            is_diverging=is_diverging or self._divergence_counter >= self.config.divergence_threshold,
            is_oscillating=is_oscillating,
            is_limit_cycle=is_limit_cycle,
            convergence_score=score,
            position_error=position_error,
            velocity=velocity,
            convergence_frame=self._convergence_frame,
            total_frames=self._total_frames,
            recommended_action=action,
            diagnostics={
                "acceleration": acceleration,
                "divergence_counter": self._divergence_counter,
                "convergence_counter": self._convergence_counter,
                "error_history_len": len(self._error_history),
            },
        )

    def should_stop(self) -> bool:
        """是否应该停止运行"""
        if not self.config.early_stop_enabled:
            return False

        if self._total_frames >= self.config.max_frames:
            return True

        if self._divergence_counter >= self.config.divergence_threshold:
            return True

        return False

    def get_diagnostics(self) -> dict:
        """获取保护器诊断信息"""
        return {
            "total_frames": self._total_frames,
            "convergence_counter": self._convergence_counter,
            "divergence_counter": self._divergence_counter,
            "convergence_frame": self._convergence_frame,
            "history_length": len(self._error_history),
        }
