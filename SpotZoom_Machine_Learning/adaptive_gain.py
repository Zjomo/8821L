"""
自适应增益调度控制器 (AdaptiveGainScheduler)

基于经典控制理论与自适应控制策略的 PID 增益调度算法。

算法原理:
- Gain Scheduling — 增益调度 (Astrom & Wittenmark, 自适应控制)
- Deviation-based Factor — 偏差分段增益因子
- Confidence-based Factor — 置信度线性插值增益
- Oscillation Detection — 振荡检测 (符号变化计数法)

功能:
- 根据检测置信度、偏差大小、历史收敛速度动态调整 PID 增益
- 提供多种调度策略: 基于偏差、基于置信度、基于收敛速率
- 防止积分饱和和振荡

依赖: numpy (仅用于类型标注)
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class GainSet:
    """PID 增益集合。"""
    kp: float
    ki: float
    kd: float


class AdaptiveGainScheduler:
    """自适应 PID 增益调度器。

    根据实时控制状态动态调整 PID 增益，实现不同偏差条件下的
    最优控制性能。
    """

    def __init__(
        self,
        base_kp: float = 1.0,
        base_ki: float = 0.0,
        base_kd: float = 0.0,
        strategy: str = "combined",
        history_window: int = 10,
    ):
        self.base_gains = GainSet(kp=float(base_kp), ki=float(base_ki), kd=float(base_kd))
        self.strategy = strategy

        if strategy not in ("deviation", "confidence", "convergence", "combined"):
            raise ValueError(f"Unknown strategy: {strategy}")

        self._error_history: deque = deque(maxlen=history_window)
        self._confidence_history: deque = deque(maxlen=history_window)

        # 调度参数
        self.large_error_threshold = 50.0  # 像素
        self.small_error_threshold = 10.0  # 像素
        self.low_confidence_threshold = 0.5
        self.min_gain_factor = 0.2
        self.max_gain_factor = 3.0

    def compute_gains(
        self,
        pixel_error: float,
        confidence: float = 1.0,
    ) -> GainSet:
        """根据当前状态计算自适应 PID 增益。

        Parameters
        ----------
        pixel_error : float
            当前像素偏差 (标量，取 x/y 偏差的范数)。
        confidence : float
            当前检测置信度 [0, 1]。

        Returns
        -------
        GainSet
            调整后的 PID 增益。
        """
        self._error_history.append(float(pixel_error))
        self._confidence_history.append(float(confidence))

        kp_factor = 1.0
        ki_factor = 1.0
        kd_factor = 1.0

        if self.strategy in ("deviation", "combined"):
            kp_f, ki_f, kd_f = self._deviation_factors(pixel_error)
            kp_factor *= kp_f
            ki_factor *= ki_f
            kd_factor *= kd_f

        if self.strategy in ("confidence", "combined"):
            cf = self._confidence_factor(confidence)
            kp_factor *= cf
            ki_factor *= cf
            kd_factor *= cf

        if self.strategy in ("convergence", "combined"):
            kp_f, ki_f, kd_f = self._convergence_factors()
            kp_factor *= kp_f
            ki_factor *= ki_f
            kd_factor *= kd_f

        # 限幅
        kp_factor = max(self.min_gain_factor, min(self.max_gain_factor, kp_factor))
        ki_factor = max(self.min_gain_factor, min(self.max_gain_factor, ki_factor))
        kd_factor = max(self.min_gain_factor, min(self.max_gain_factor, kd_factor))

        return GainSet(
            kp=self.base_gains.kp * kp_factor,
            ki=self.base_gains.ki * ki_factor,
            kd=self.base_gains.kd * kd_factor,
        )

    def _deviation_factors(self, error: float) -> Tuple[float, float, float]:
        """基于偏差大小的增益因子。

        大偏差: 增大比例增益加速收敛，减小微分增益避免过冲
        小偏差: 减小比例增益避免振荡，增大微分增益提高阻尼
        """
        abs_error = abs(error)

        if abs_error > self.large_error_threshold:
            # 大偏差: 激进策略
            kp_f = 1.5 + 0.5 * min(abs_error / 100.0, 1.0)
            ki_f = 0.5  # 减小积分防止饱和
            kd_f = 0.3  # 减小微分避免过冲
        elif abs_error < self.small_error_threshold:
            # 小偏差: 精细策略
            ratio = abs_error / max(self.small_error_threshold, 1.0)
            kp_f = 0.3 + 0.7 * ratio  # 小比例增益
            ki_f = 0.5 + 0.5 * (1.0 - ratio)  # 适度积分
            kd_f = 1.5  # 大微分增益提高阻尼
        else:
            # 中等偏差: 标准策略
            ratio = (abs_error - self.small_error_threshold) / max(
                self.large_error_threshold - self.small_error_threshold, 1.0
            )
            kp_f = 0.8 + 0.7 * ratio
            ki_f = 0.5
            kd_f = 0.5 + 0.5 * (1.0 - ratio)

        return (kp_f, ki_f, kd_f)

    def _confidence_factor(self, confidence: float) -> float:
        """基于检测置信度的增益因子。

        低置信度: 降低增益，保守移动
        高置信度: 正常增益
        """
        if confidence >= 0.9:
            return 1.0
        elif confidence < self.low_confidence_threshold:
            return self.min_gain_factor
        else:
            # 线性插值
            ratio = (confidence - self.low_confidence_threshold) / max(
                0.9 - self.low_confidence_threshold, 0.01
            )
            return self.min_gain_factor + (1.0 - self.min_gain_factor) * ratio

    def _convergence_factors(self) -> Tuple[float, float, float]:
        """基于历史收敛速率的增益因子。

        快速收敛: 保持当前增益
        缓慢收敛/振荡: 调整增益
        """
        if len(self._error_history) < 3:
            return (1.0, 1.0, 1.0)

        errors = list(self._error_history)
        recent_errors = errors[-5:] if len(errors) >= 5 else errors

        # 计算收敛速率
        if len(recent_errors) >= 2:
            rate = recent_errors[-1] - recent_errors[0]
            if rate < -1:
                # 正在收敛
                return (1.0, 1.0, 1.0)
            elif rate > 1:
                # 发散趋势
                return (0.5, 0.2, 1.5)  # 降低增益，增加阻尼

        # 检测振荡
        if len(recent_errors) >= 4:
            sign_changes = sum(
                1 for i in range(1, len(recent_errors))
                if (recent_errors[i] - recent_errors[i - 1]) * (recent_errors[i - 1] - recent_errors[max(0, i - 2)]) < 0
            )
            if sign_changes >= 2:
                # 振荡: 降低比例和积分，增加微分
                return (0.5, 0.1, 2.0)

        return (1.0, 1.0, 1.0)

    def reset(self) -> None:
        """重置历史记录。"""
        self._error_history.clear()
        self._confidence_history.clear()
