"""
Adaptive Feedforward Controller - 自适应前馈控制器

Inspired by:
- leap-c (leap-c): Learned predictive control for optics
- python-control: Feedforward + feedback control architecture
- do-mpc: Model predictive control with disturbance estimation

Core Innovation:
- 基于运动轨迹预测的前馈补偿
- 自适应扰动估计与抑制
- 与 PID 反馈控制器的协同工作
- 纯 numpy 实现，零外部依赖
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class FeedforwardConfig:
    """前馈控制器配置"""
    # 预测窗口长度
    prediction_horizon: int = 10
    # 历史窗口长度
    history_length: int = 50
    # 前馈增益
    feedforward_gain: float = 0.3
    # 最大前馈补偿量 (像素)
    max_compensation_px: float = 5.0
    # 扰动估计学习率
    disturbance_learning_rate: float = 0.1
    # 是否启用前馈
    enabled: bool = True
    # 低通滤波器截止频率 (0-1, 归一化)
    lpf_cutoff: float = 0.3
    # 最小可信预测步数
    min_confident_steps: int = 3


@dataclass
class FeedforwardOutput:
    """前馈控制器输出"""
    compensation_x: float  # X 方向补偿量
    compensation_y: float  # Y 方向补偿量
    predicted_trajectory_x: List[float]  # 预测 X 轨迹
    predicted_trajectory_y: List[float]  # 预测 Y 轨迹
    disturbance_estimate_x: float  # 扰动估计 X
    disturbance_estimate_y: float  # 扰动估计 Y
    confidence: float  # 预测置信度


class AdaptiveFeedforwardController:
    """自适应前馈控制器

    通过分析光斑运动历史，预测未来运动趋势，
    生成前馈补偿信号与 PID 反馈控制协同工作。

    Inspired by leap-c's learned predictive control and
    python-control's feedforward architecture.
    """

    def __init__(self, config: Optional[FeedforwardConfig] = None):
        self.config = config or FeedforwardConfig()
        self._history_x: Deque[float] = deque(maxlen=self.config.history_length)
        self._history_y: Deque[float] = deque(maxlen=self.config.history_length)
        self._disturbance_x: float = 0.0
        self._disturbance_y: float = 0.0
        self._prev_error_x: float = 0.0
        self._prev_error_y: float = 0.0

    def reset(self) -> None:
        """重置控制器状态"""
        self._history_x.clear()
        self._history_y.clear()
        self._disturbance_x = 0.0
        self._disturbance_y = 0.0
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0

    def _low_pass_filter(self, signal: List[float], cutoff: float) -> List[float]:
        """简单低通滤波器"""
        if len(signal) < 2:
            return list(signal)

        alpha = cutoff
        filtered = [signal[0]]
        for i in range(1, len(signal)):
            filtered.append(alpha * signal[i] + (1 - alpha) * filtered[-1])
        return filtered

    def _linear_predict(
        self, history: Deque[float], horizon: int
    ) -> Tuple[List[float], float]:
        """线性回归预测

        Args:
            history: 历史数据
            horizon: 预测步数

        Returns:
            (预测轨迹, 置信度)
        """
        if len(history) < 3:
            return [history[-1] if history else 0.0] * horizon, 0.0

        data = np.array(list(history))
        n = len(data)
        x = np.arange(n, dtype=np.float64)

        # 线性回归
        x_mean = np.mean(x)
        y_mean = np.mean(data)
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (data - y_mean))

        if ss_xx < 1e-10:
            return [float(y_mean)] * horizon, 0.0

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean

        # 残差标准差
        predicted = slope * x + intercept
        residuals = data - predicted
        std_residual = float(np.std(residuals))

        # 置信度: 基于拟合 R²
        ss_tot = np.sum((data - y_mean) ** 2)
        r_squared = 1.0 - (np.sum(residuals ** 2) / (ss_tot + 1e-10))
        confidence = float(max(0, min(1, r_squared)))

        # 预测未来
        future_x = np.arange(n, n + horizon, dtype=np.float64)
        future_pred = slope * future_x + intercept

        # 预测不确定性随步数增长
        for i in range(horizon):
            future_pred[i] += np.random.normal(0, std_residual * (i + 1) * 0.1)

        return [float(v) for v in future_pred], confidence

    def _estimate_disturbance(
        self, error_x: float, error_y: float, dt: float = 1.0
    ) -> Tuple[float, float]:
        """估计外部扰动

        使用误差变化率估计扰动方向和大小。

        Args:
            error_x: X 方向误差
            error_y: Y 方向误差
            dt: 时间步长

        Returns:
            (扰动估计 X, 扰动估计 Y)
        """
        lr = self.config.disturbance_learning_rate

        # 误差变化率 = 扰动估计
        de_x = (error_x - self._prev_error_x) / dt
        de_y = (error_y - self._prev_error_y) / dt

        # 指数移动平均更新扰动估计
        self._disturbance_x = (1 - lr) * self._disturbance_x + lr * de_x
        self._disturbance_y = (1 - lr) * self._disturbance_y + lr * de_y

        self._prev_error_x = error_x
        self._prev_error_y = error_y

        return self._disturbance_x, self._disturbance_y

    def compute(
        self, error_x: float, error_y: float
    ) -> FeedforwardOutput:
        """计算前馈补偿

        Args:
            error_x: X 方向当前误差 (像素)
            error_y: Y 方向当前误差 (像素)

        Returns:
            FeedforwardOutput 前馈补偿输出
        """
        if not self.config.enabled:
            return FeedforwardOutput(
                compensation_x=0.0, compensation_y=0.0,
                predicted_trajectory_x=[], predicted_trajectory_y=[],
                disturbance_estimate_x=0.0, disturbance_estimate_y=0.0,
                confidence=0.0,
            )

        # 记录历史
        self._history_x.append(error_x)
        self._history_y.append(error_y)

        # 估计扰动
        dist_x, dist_y = self._estimate_disturbance(error_x, error_y)

        # 预测未来轨迹
        pred_x, conf_x = self._linear_predict(
            self._history_x, self.config.prediction_horizon
        )
        pred_y, conf_y = self._linear_predict(
            self._history_y, self.config.prediction_horizon
        )

        # 低通滤波平滑预测
        pred_x = self._low_pass_filter(pred_x, self.config.lpf_cutoff)
        pred_y = self._low_pass_filter(pred_y, self.config.lpf_cutoff)

        # 计算前馈补偿: 基于预测的未来偏差
        confidence = min(conf_x, conf_y)

        if confidence > 0.3 and len(pred_x) >= self.config.min_confident_steps:
            # 取预测窗口中点的值作为补偿目标
            mid = min(self.config.min_confident_steps, len(pred_x) - 1)
            comp_x = -pred_x[mid] * self.config.feedforward_gain
            comp_y = -pred_y[mid] * self.config.feedforward_gain
        else:
            comp_x = 0.0
            comp_y = 0.0

        # 加上扰动补偿
        comp_x -= dist_x * self.config.feedforward_gain * 0.5
        comp_y -= dist_y * self.config.feedforward_gain * 0.5

        # 限幅
        max_comp = self.config.max_compensation_px
        comp_x = float(np.clip(comp_x, -max_comp, max_comp))
        comp_y = float(np.clip(comp_y, -max_comp, max_comp))

        return FeedforwardOutput(
            compensation_x=comp_x,
            compensation_y=comp_y,
            predicted_trajectory_x=pred_x,
            predicted_trajectory_y=pred_y,
            disturbance_estimate_x=dist_x,
            disturbance_estimate_y=dist_y,
            confidence=confidence,
        )

    def get_diagnostics(self) -> dict:
        """获取控制器诊断信息"""
        return {
            "history_length": len(self._history_x),
            "disturbance_estimate": {
                "x": self._disturbance_x,
                "y": self._disturbance_y,
            },
            "enabled": self.config.enabled,
            "feedforward_gain": self.config.feedforward_gain,
        }
