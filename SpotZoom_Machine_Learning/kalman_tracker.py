"""
卡尔曼滤波光斑追踪器 (KalmanSpotTracker)

基于经典机器视觉 / 统计信号处理的实时追踪算法。

算法原理:
- Kalman Filter (1960, R.E. Kalman) — 最优线性状态估计器
- Constant Velocity Motion Model — 恒速运动假设
- Joseph Form 协方差更新 — 保证数值稳定性

功能:
- 对光斑检测结果进行卡尔曼滤波时序平滑
- 预测下一帧光斑位置，减少检测抖动
- 处理检测丢失时的短时预测

依赖: numpy (无线性代数库之外的外部依赖)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """卡尔曼滤波器状态向量 [x, y, vx, vy]"""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


class KalmanSpotTracker:
    """基于卡尔曼滤波的光斑位置追踪器。

    使用恒速运动模型 (Constant Velocity Model) 对光斑位置进行
    时序滤波和平滑，有效减少检测帧间抖动。

    状态向量: [x, y, vx, vy]
    观测向量: [x, y]

    Parameters
    ----------
    process_noise_q : float
        过程噪声协方差 Q 的缩放因子。值越大，滤波器越信任观测值，
        响应越快但平滑效果减弱。典型值 0.01 ~ 1.0。
    measurement_noise_r : float
        测量噪声协方差 R 的缩放因子。值越大，滤波器越信任预测值，
        平滑效果更强但响应变慢。典型值 0.1 ~ 10.0。
    max_predict_steps : int
        允许的最大连续预测步数（无观测更新）。超过后重置滤波器。
    """

    def __init__(
        self,
        process_noise_q: float = 0.1,
        measurement_noise_r: float = 1.0,
        max_predict_steps: int = 5,
    ):
        self._process_noise_q = float(process_noise_q)
        self._measurement_noise_r = float(measurement_noise_r)
        self._max_predict_steps = int(max_predict_steps)

        # State: [x, y, vx, vy]
        self._x = np.zeros(4, dtype=np.float64)
        # State covariance
        self._P = np.eye(4, dtype=np.float64) * 100.0

        # State transition matrix (constant velocity)
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Observation matrix
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        self._initialized = False
        self._predict_count = 0
        self._last_measurement: Optional[Tuple[int, int]] = None

    @property
    def is_initialized(self) -> bool:
        """滤波器是否已用首次观测初始化。"""
        return self._initialized

    @property
    def predict_count(self) -> int:
        """连续预测步数（无观测更新）。"""
        return self._predict_count

    def initialize(self, cx: int, cy: int) -> None:
        """用首次观测初始化滤波器状态。

        Parameters
        ----------
        cx, cy : int
            光斑中心像素坐标。
        """
        self._x = np.array([float(cx), float(cy), 0.0, 0.0], dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 100.0
        self._P[0, 0] = 1.0
        self._P[1, 1] = 1.0
        self._initialized = True
        self._predict_count = 0
        self._last_measurement = (cx, cy)
        LOGGER.info("KalmanSpotTracker: 初始化完成 (cx=%d, cy=%d)", cx, cy)

    def update(self, cx: int, cy: int) -> Tuple[int, int]:
        """用新的观测更新滤波器并返回平滑后的位置。

        Parameters
        ----------
        cx, cy : int
            光斑检测器输出的光斑中心坐标。

        Returns
        -------
        Tuple[int, int]
            卡尔曼滤波后的平滑光斑中心坐标。
        """
        if not self._initialized:
            self.initialize(cx, cy)
            return (cx, cy)

        z = np.array([float(cx), float(cy)], dtype=np.float64)

        # --- Predict ---
        x_pred = self._F @ self._x
        Q = self._process_noise_q * np.array([
            [0.25, 0, 0.5, 0],
            [0, 0.25, 0, 0.5],
            [0.5, 0, 1, 0],
            [0, 0.5, 0, 1],
        ], dtype=np.float64)
        P_pred = self._F @ self._P @ self._F.T + Q

        # --- Update ---
        y_residual = z - self._H @ x_pred
        R = self._measurement_noise_r * np.eye(2, dtype=np.float64)
        S = self._H @ P_pred @ self._H.T + R
        try:
            K = P_pred @ self._H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            LOGGER.warning(
                "KalmanSpotTracker: 协方差矩阵奇异，跳过本次更新 (cx=%d, cy=%d)",
                cx, cy,
            )
            return (int(round(self._x[0])), int(round(self._x[1])))

        self._x = x_pred + K @ y_residual
        I_KH = np.eye(4) - K @ self._H
        self._P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T  # Joseph form for numerical stability

        self._predict_count = 0
        self._last_measurement = (cx, cy)

        return (int(round(self._x[0])), int(round(self._x[1])))

    def predict(self) -> Optional[Tuple[int, int]]:
        """仅执行预测步骤（无观测更新），返回预测位置。

        Returns
        -------
        Optional[Tuple[int, int]]
            预测的光斑位置。如果超过最大预测步数，返回 None。
        """
        if not self._initialized:
            LOGGER.warning("KalmanSpotTracker: predict() 调用时滤波器未初始化")
            return None

        self._predict_count += 1
        if self._predict_count > self._max_predict_steps:
            LOGGER.warning(
                "KalmanSpotTracker: 超过最大预测步数 (%d > %d)，返回 None",
                self._predict_count, self._max_predict_steps,
            )
            return None

        Q = self._process_noise_q * np.array([
            [0.25, 0, 0.5, 0],
            [0, 0.25, 0, 0.5],
            [0.5, 0, 1, 0],
            [0, 0.5, 0, 1],
        ], dtype=np.float64)
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + Q

        return (int(round(self._x[0])), int(round(self._x[1])))

    def reset(self) -> None:
        """重置滤波器到未初始化状态。"""
        LOGGER.info("KalmanSpotTracker: 重置滤波器")
        self._x = np.zeros(4, dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 100.0
        self._initialized = False
        self._predict_count = 0
        self._last_measurement = None

    def get_velocity(self) -> Tuple[float, float]:
        """获取当前估计的速度 (vx, vy) 像素/帧。"""
        return (float(self._x[2]), float(self._x[3]))

    def get_position(self) -> Optional[Tuple[int, int]]:
        """获取当前估计的位置。"""
        if not self._initialized:
            return None
        return (int(round(self._x[0])), int(round(self._x[1])))
