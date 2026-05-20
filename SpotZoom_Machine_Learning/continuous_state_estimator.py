"""
连续时间状态估计器 (ContinuousStateEstimator)

灵感来源:
- torchdiffeq (https://github.com/rtqichen/torchdiffeq) — Neural ODE:
  将连续动态系统建模为常微分方程，使用数值积分器求解隐式层
- NeuralOperator (https://github.com/neuraloperator/neuraloperator) — 神经算子:
  学习连续函数空间之间的映射，而非离散点之间的映射
- Extended Kalman Filter (EKF) — 扩展卡尔曼滤波:
  非线性系统的最优递推状态估计
- Ornstein-Uhlenbeck Process — OU 过程:
  均值回复随机过程，广泛用于物理系统的漂移建模

算法原理:
- Ornstein-Uhlenbeck 连续动态: dx/dt = -theta * (x - mu) + sigma * dW
  均值回复特性使光斑漂移趋向平衡位置，物理上合理
- Euler-Maruyama 离散化: SDE 的经典数值积分方法
- Continuous-time EKF: 连续时间预测 + 离散时间测量更新
- Analytical Covariance Propagation: OU 过程的协方差有解析解，
  避免数值积分误差
- Mahalanobis Distance: 基于状态协方差的异常检测统计量

功能:
- 建模光斑位置的连续时间漂移动态 (非仅离散帧)
- 在检测帧之间进行高精度预测和补偿
- 基于物理漂移模型的光滑轨迹插值
- 实时异常检测与置信度评估
- 估计器健康状态监控

依赖: numpy, logging (纯 numpy 实现，无 PyTorch/TensorFlow)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.ContinuousStateEstimator")


# ======================== 数据类 ========================


@dataclass
class StateEstimatorConfig:
    """连续时间状态估计器配置。

    Parameters
    ----------
    drift_rate : float
        Ornstein-Uhlenbeck 均值回复速率 theta。
        值越大，漂移越快回到平衡位置。典型值 0.1 ~ 10.0。
    noise_intensity : float
        OU 过程扩散系数 sigma (噪声强度)。
        值越大，随机漂移越剧烈。典型值 0.01 ~ 5.0。
    process_noise : float
        EKF 过程噪声协方差 Q 的缩放因子。典型值 0.001 ~ 1.0。
    measurement_noise : float
        EKF 测量噪声协方差 R 的缩放因子。典型值 0.1 ~ 10.0。
    equilibrium_position : Tuple[float, float]
        OU 过程的平衡位置 mu = (mu_x, mu_y)。默认 (0, 0)。
    initial_position_uncertainty : float
        初始位置协方差的对角线值。典型值 1.0 ~ 1000.0。
    initial_velocity_uncertainty : float
        初始速度协方差的对角线值。典型值 0.1 ~ 100.0。
    max_prediction_horizon : float
        最大预测时间跨度 (秒)。超过此时间的预测将返回低置信度。
        典型值 1.0 ~ 60.0。
    anomaly_threshold : float
        Mahalanobis 距离异常检测阈值。典型值 3.0 ~ 5.0。
    integration_steps : int
        Euler-Maruyama 积分步数 (用于轨迹预测)。典型值 10 ~ 100。
    velocity_decay_rate : float
        速度分量的附加衰减率。模拟阻尼效应。典型值 0.0 ~ 5.0。
    """

    drift_rate: float = 2.0
    noise_intensity: float = 0.5
    process_noise: float = 0.01
    measurement_noise: float = 1.0
    equilibrium_position: Tuple[float, float] = (0.0, 0.0)
    initial_position_uncertainty: float = 100.0
    initial_velocity_uncertainty: float = 10.0
    max_prediction_horizon: float = 10.0
    anomaly_threshold: float = 4.0
    integration_steps: int = 50
    velocity_decay_rate: float = 1.0


@dataclass
class StateEstimationResult:
    """状态估计结果。

    Parameters
    ----------
    state : np.ndarray
        状态向量 [x, y, vx, vy]。
    velocity : np.ndarray
        速度向量 [vx, vy]。
    covariance : np.ndarray
        状态协方差矩阵 (4x4)。
    confidence : float
        估计置信度 [0, 1]。1.0 = 完全确信。
    is_anomalous : bool
        该估计是否基于异常测量。
    timestamp : float
        估计对应的时间戳。
    dt_since_last_update : float
        距上次测量的时间间隔 (秒)。
    """

    state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4))
    confidence: float = 1.0
    is_anomalous: bool = False
    timestamp: float = 0.0
    dt_since_last_update: float = 0.0


# ======================== 主类 ========================


class ContinuousStateEstimator:
    """连续时间状态估计器。

    基于 Neural ODE 概念和 Ornstein-Uhlenbeck 随机过程，
    对光斑位置漂移进行连续时间建模和状态估计。

    与传统离散帧卡尔曼滤波器不同，本估计器:
    - 使用连续时间动态模型 (OU 过程) 而非离散状态转移矩阵
    - 支持任意时间点的状态预测 (非仅整数帧)
    - 利用解析协方差传播保证数值精度
    - 基于物理漂移模型提供光滑轨迹插值

    状态向量: [x, y, vx, vy] (位置 + 速度)
    连续动态: dx/dt = -theta * (x - mu) + sigma * dW (Ornstein-Uhlenbeck)
    离散化: Euler-Maruyama 方法
    状态估计: Extended Kalman Filter (连续预测 + 离散更新)

    Parameters
    ----------
    config : StateEstimatorConfig, optional
        估计器配置。为 None 时使用默认配置。

    Examples
    --------
    >>> est = ContinuousStateEstimator()
    >>> est.update(t=0.0, measurement=np.array([100.0, 200.0]))
    >>> result = est.predict(t_future=0.5)
    >>> print(f"预测位置: ({result.state[0]:.2f}, {result.state[1]:.2f})")
    >>> print(f"置信度: {result.confidence:.3f}")
    """

    def __init__(self, config: Optional[StateEstimatorConfig] = None):
        """初始化连续时间状态估计器。

        Parameters
        ----------
        config : StateEstimatorConfig, optional
            估计器配置参数。为 None 时使用默认值。
        """
        if config is not None and not isinstance(config, StateEstimatorConfig):
            raise TypeError(
                f"config 必须是 StateEstimatorConfig 类型或 None，"
                f"收到 {type(config).__name__}"
            )

        self._config = config if config is not None else StateEstimatorConfig()

        # --- 状态向量: [x, y, vx, vy] ---
        self._state = np.zeros(4, dtype=np.float64)
        # --- 状态协方差矩阵 (4x4) ---
        self._covariance = np.eye(4, dtype=np.float64)
        self._covariance[0, 0] = self._config.initial_position_uncertainty
        self._covariance[1, 1] = self._config.initial_position_uncertainty
        self._covariance[2, 2] = self._config.initial_velocity_uncertainty
        self._covariance[3, 3] = self._config.initial_velocity_uncertainty

        # --- 时间追踪 ---
        self._last_time: Optional[float] = None
        self._last_measurement_time: Optional[float] = None

        # --- 初始化标志 ---
        self._initialized = False

        # --- 统计信息 ---
        self._total_updates = 0
        self._total_predictions = 0
        self._total_anomalies = 0
        self._max_mahalanobis_seen = 0.0
        self._creation_time = time.time()

        # --- 观测矩阵 H: 观测 [x, y]，状态 [x, y, vx, vy] ---
        self._H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # --- 测量噪声协方差 R ---
        self._R = self._config.measurement_noise * np.eye(2, dtype=np.float64)

        LOGGER.info(
            "连续时间状态估计器已初始化: "
            f"drift_rate={self._config.drift_rate}, "
            f"noise_intensity={self._config.noise_intensity}"
        )

    # ==================== 属性 ====================

    @property
    def is_initialized(self) -> bool:
        """估计器是否已用首次测量初始化。"""
        return self._initialized

    @property
    def state(self) -> np.ndarray:
        """当前状态向量 [x, y, vx, vy] 的副本。"""
        return self._state.copy()

    @property
    def covariance(self) -> np.ndarray:
        """当前状态协方差矩阵 (4x4) 的副本。"""
        return self._covariance.copy()

    @property
    def last_time(self) -> Optional[float]:
        """最后一次状态更新的时间戳。"""
        return self._last_time

    @property
    def config(self) -> StateEstimatorConfig:
        """当前配置的副本。"""
        # 返回一个新的 dataclass 实例
        return StateEstimatorConfig(
            drift_rate=self._config.drift_rate,
            noise_intensity=self._config.noise_intensity,
            process_noise=self._config.process_noise,
            measurement_noise=self._config.measurement_noise,
            equilibrium_position=self._config.equilibrium_position,
            initial_position_uncertainty=self._config.initial_position_uncertainty,
            initial_velocity_uncertainty=self._config.initial_velocity_uncertainty,
            max_prediction_horizon=self._config.max_prediction_horizon,
            anomaly_threshold=self._config.anomaly_threshold,
            integration_steps=self._config.integration_steps,
            velocity_decay_rate=self._config.velocity_decay_rate,
        )

    # ==================== 核心方法 ====================

    def update(self, t: float, measurement: np.ndarray,
               uncertainty: float = 1.0) -> StateEstimationResult:
        """在时间 t 融入新的测量值。

        执行连续时间预测 (从上次更新到当前时刻)，
        然后执行标准 EKF 测量校正。

        Parameters
        ----------
        t : float
            测量时间戳 (秒)。必须 >= 上次更新时间。
        measurement : np.ndarray
            测量向量 [x, y]，形状 (2,)。
        uncertainty : float, optional
            本次测量的不确定性权重。1.0 = 标准噪声，
            >1.0 = 更不确定，<1.0 = 更可信。默认 1.0。

        Returns
        -------
        StateEstimationResult
            更新后的状态估计结果。

        Raises
        ------
        ValueError
            如果 measurement 形状不正确或时间戳无效。
        """
        # --- 输入验证 ---
        measurement = np.asarray(measurement, dtype=np.float64)
        if measurement.shape != (2,):
            raise ValueError(
                f"measurement 形状必须为 (2,)，收到 {measurement.shape}"
            )
        if not np.all(np.isfinite(measurement)):
            raise ValueError("measurement 包含非有限值 (NaN/Inf)")

        if self._initialized and self._last_time is not None and t < self._last_time:
            raise ValueError(
                f"时间戳 t={t} 早于上次更新时间 {self._last_time}，"
                f"不允许时间回退"
            )

        uncertainty = float(uncertainty)
        if uncertainty <= 0:
            raise ValueError(f"uncertainty 必须为正数，收到 {uncertainty}")

        # --- 首次初始化 ---
        if not self._initialized:
            self._state[0] = measurement[0]
            self._state[1] = measurement[1]
            self._state[2] = 0.0  # 初始速度为零
            self._state[3] = 0.0
            self._last_time = t
            self._last_measurement_time = t
            self._initialized = True
            self._total_updates += 1

            LOGGER.info(
                f"状态估计器首次初始化: t={t:.4f}, "
                f"pos=({measurement[0]:.2f}, {measurement[1]:.2f})"
            )

            return StateEstimationResult(
                state=self._state.copy(),
                velocity=self._state[2:4].copy(),
                covariance=self._covariance.copy(),
                confidence=1.0,
                is_anomalous=False,
                timestamp=t,
                dt_since_last_update=0.0,
            )

        # --- 连续时间预测 (从 last_time 到 t) ---
        dt = t - self._last_time
        if dt > 0:
            self._propagate_covariance(dt)
            self._propagate_state(dt)

        # --- 异常检测 (在更新前检查) ---
        anomaly_info = self.detect_anomaly(measurement)
        is_anomalous = anomaly_info["is_anomalous"]

        if is_anomalous:
            self._total_anomalies += 1
            LOGGER.warning(
                f"检测到异常测量: t={t:.4f}, "
                f"mahalanobis={anomaly_info['mahalanobis_distance']:.3f}, "
                f"threshold={self._config.anomaly_threshold:.1f}"
            )

        # --- EKF 测量更新 ---
        # 调整测量噪声
        R_adjusted = self._R * (uncertainty ** 2)

        # 新息 (innovation)
        z = measurement
        z_pred = self._H @ self._state
        y_innovation = z - z_pred

        # 新息协方差
        S = self._H @ self._covariance @ self._H.T + R_adjusted

        # 卡尔曼增益
        try:
            K = self._covariance @ self._H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            LOGGER.error("新息协方差矩阵奇异，使用伪逆")
            K = self._covariance @ self._H.T @ np.linalg.pinv(S)

        # 状态更新
        self._state = self._state + K @ y_innovation

        # 协方差更新 (Joseph form 保证数值稳定性)
        I_KH = np.eye(4) - K @ self._H
        self._covariance = (
            I_KH @ self._covariance @ I_KH.T + K @ R_adjusted @ K.T
        )

        # 确保协方差矩阵对称
        self._covariance = 0.5 * (
            self._covariance + self._covariance.T
        )

        # --- 更新时间戳 ---
        self._last_time = t
        self._last_measurement_time = t
        self._total_updates += 1

        dt_since = 0.0 if self._last_measurement_time is None else 0.0
        confidence = self.compute_confidence(t)

        LOGGER.debug(
            f"状态更新: t={t:.4f}, dt={dt:.4f}, "
            f"pos=({self._state[0]:.2f}, {self._state[1]:.2f}), "
            f"vel=({self._state[2]:.2f}, {self._state[3]:.2f}), "
            f"conf={confidence:.3f}"
        )

        return StateEstimationResult(
            state=self._state.copy(),
            velocity=self._state[2:4].copy(),
            covariance=self._covariance.copy(),
            confidence=confidence,
            is_anomalous=is_anomalous,
            timestamp=t,
            dt_since_last_update=dt_since,
        )

    def predict(self, t_future: float) -> StateEstimationResult:
        """预测未来时刻 t_future 的状态。

        使用连续时间 OU 动态模型进行解析预测，
        并传播协方差矩阵以给出预测不确定性。

        Parameters
        ----------
        t_future : float
            未来时间戳 (秒)。必须 >= 当前时间。

        Returns
        -------
        StateEstimationResult
            预测的状态估计结果。

        Raises
        ------
        ValueError
            如果估计器未初始化或时间戳无效。
        RuntimeError
            如果预测时间超出最大预测范围。
        """
        if not self._initialized:
            raise RuntimeError("估计器尚未初始化，请先调用 update()")

        if self._last_time is None:
            raise RuntimeError("内部时间戳异常 (last_time is None)")

        if t_future < self._last_time:
            raise ValueError(
                f"预测时间 t_future={t_future} 早于当前时间 {self._last_time}"
            )

        dt = t_future - self._last_time

        if dt > self._config.max_prediction_horizon:
            LOGGER.warning(
                f"预测时间 {dt:.2f}s 超出最大范围 "
                f"{self._config.max_prediction_horizon:.1f}s，"
                f"置信度将显著降低"
            )

        # --- 连续时间状态传播 (OU 过程解析解) ---
        theta = self._config.drift_rate
        mu_x, mu_y = self._config.equilibrium_position
        decay_v = self._config.velocity_decay_rate

        exp_neg_theta_dt = np.exp(-theta * dt)
        exp_neg_decay_dt = np.exp(-decay_v * dt)

        # 位置: 均值回复到平衡位置
        x_pred = mu_x + (self._state[0] - mu_x) * exp_neg_theta_dt
        y_pred = mu_y + (self._state[1] - mu_y) * exp_neg_theta_dt

        # 速度: 指数衰减 (阻尼)
        vx_pred = self._state[2] * exp_neg_decay_dt
        vy_pred = self._state[3] * exp_neg_decay_dt

        predicted_state = np.array([
            x_pred, y_pred, vx_pred, vy_pred
        ], dtype=np.float64)

        # --- 协方差传播 (解析解) ---
        predicted_cov = self._propagate_covariance_analytical(
            self._covariance, dt
        )

        # --- 置信度计算 ---
        confidence = self.compute_confidence(t_future)

        self._total_predictions += 1

        dt_since = (
            0.0 if self._last_measurement_time is None
            else t_future - self._last_measurement_time
        )

        LOGGER.debug(
            f"状态预测: t_future={t_future:.4f}, dt={dt:.4f}, "
            f"pos=({x_pred:.2f}, {y_pred:.2f}), "
            f"conf={confidence:.3f}"
        )

        return StateEstimationResult(
            state=predicted_state,
            velocity=predicted_state[2:4].copy(),
            covariance=predicted_cov,
            confidence=confidence,
            is_anomalous=False,
            timestamp=t_future,
            dt_since_last_update=dt_since,
        )

    def predict_trajectory(self, t_start: float, t_end: float,
                           num_points: int = 50) -> Dict[str, Any]:
        """预测时间范围 [t_start, t_end] 内的完整轨迹。

        使用 Euler-Maruyama 方法对 OU 过程进行数值积分，
        生成包含位置、速度和不确定性的完整轨迹。

        Parameters
        ----------
        t_start : float
            轨迹起始时间 (秒)。
        t_end : float
            轨迹结束时间 (秒)。
        num_points : int, optional
            轨迹采样点数。默认 50。

        Returns
        -------
        Dict[str, Any]
            轨迹字典，包含:
            - 'times': 时间数组 (num_points,)
            - 'positions': 位置数组 (num_points, 2)
            - 'velocities': 速度数组 (num_points, 2)
            - 'position_uncertainties': 位置不确定性 (num_points, 2)
            - 'confidences': 置信度数组 (num_points,)
            - 'mean_position': 平均位置 (2,)
            - 'max_displacement': 最大位移

        Raises
        ------
        ValueError
            如果参数无效。
        RuntimeError
            如果估计器未初始化。
        """
        if not self._initialized:
            raise RuntimeError("估计器尚未初始化，请先调用 update()")

        if t_start >= t_end:
            raise ValueError(
                f"t_start ({t_start}) 必须小于 t_end ({t_end})"
            )

        num_points = int(num_points)
        if num_points < 2:
            raise ValueError("num_points 必须至少为 2")

        times = np.linspace(t_start, t_end, num_points)
        dt_total = t_end - t_start

        # 使用 Euler-Maruyama 积分生成轨迹
        theta = self._config.drift_rate
        sigma = self._config.noise_intensity
        mu_x, mu_y = self._config.equilibrium_position
        decay_v = self._config.velocity_decay_rate

        # 起始状态: 从当前状态预测到 t_start
        if self._last_time is not None and t_start > self._last_time:
            start_result = self.predict(t_start)
            current_state = start_result.state.copy()
            current_cov = start_result.covariance.copy()
        elif self._last_time is not None and np.isclose(t_start, self._last_time):
            current_state = self._state.copy()
            current_cov = self._covariance.copy()
        else:
            raise ValueError(
                f"t_start ({t_start}) 早于当前状态时间 ({self._last_time})"
            )

        positions = np.zeros((num_points, 2), dtype=np.float64)
        velocities = np.zeros((num_points, 2), dtype=np.float64)
        position_uncertainties = np.zeros((num_points, 2), dtype=np.float64)
        confidences = np.zeros(num_points, dtype=np.float64)

        # Euler-Maruyama 步长
        dt_step = dt_total / (num_points - 1)

        for i in range(num_points):
            positions[i] = current_state[0:2]
            velocities[i] = current_state[2:4]
            position_uncertainties[i] = np.sqrt(np.diag(current_cov)[0:2])
            confidences[i] = self._compute_confidence_from_cov(current_cov)

            if i < num_points - 1:
                # OU 过程漂移项
                drift_x = -theta * (current_state[0] - mu_x)
                drift_y = -theta * (current_state[1] - mu_y)
                drift_vx = -decay_v * current_state[2]
                drift_vy = -decay_v * current_state[3]

                # Euler-Maruyama 更新 (确定性部分，不含随机噪声)
                # 预测轨迹使用均值路径
                current_state[0] += drift_x * dt_step
                current_state[1] += drift_y * dt_step
                current_state[2] += drift_vx * dt_step
                current_state[3] += drift_vy * dt_step

                # 协方差传播
                current_cov = self._propagate_covariance_analytical(
                    current_cov, dt_step
                )

        # 计算统计摘要
        displacements = np.linalg.norm(
            positions - positions[0], axis=1
        )
        mean_position = np.mean(positions, axis=0)
        max_displacement = float(np.max(displacements))

        LOGGER.info(
            f"轨迹预测: t=[{t_start:.4f}, {t_end:.4f}], "
            f"{num_points} 点, "
            f"mean_pos=({mean_position[0]:.2f}, {mean_position[1]:.2f}), "
            f"max_disp={max_displacement:.2f}"
        )

        return {
            "times": times,
            "positions": positions,
            "velocities": velocities,
            "position_uncertainties": position_uncertainties,
            "confidences": confidences,
            "mean_position": mean_position,
            "max_displacement": max_displacement,
        }

    def set_dynamics_model(self, drift_rate: float,
                           noise_intensity: float) -> None:
        """设置连续动态模型参数 (Ornstein-Uhlenbeck 过程)。

        允许运行时调整漂移速率和噪声强度，
        以适应不同的光斑漂移特性。

        Parameters
        ----------
        drift_rate : float
            OU 均值回复速率 theta。必须 > 0。
        noise_intensity : float
            OU 扩散系数 sigma。必须 >= 0。

        Raises
        ------
        ValueError
            如果参数无效。
        """
        drift_rate = float(drift_rate)
        noise_intensity = float(noise_intensity)

        if drift_rate <= 0:
            raise ValueError(
                f"drift_rate 必须为正数，收到 {drift_rate}"
            )
        if noise_intensity < 0:
            raise ValueError(
                f"noise_intensity 必须非负，收到 {noise_intensity}"
            )

        old_drift = self._config.drift_rate
        old_noise = self._config.noise_intensity

        self._config.drift_rate = drift_rate
        self._config.noise_intensity = noise_intensity

        LOGGER.info(
            f"动态模型参数已更新: "
            f"drift_rate {old_drift:.3f} -> {drift_rate:.3f}, "
            f"noise_intensity {old_noise:.3f} -> {noise_intensity:.3f}"
        )

    def compute_confidence(self, t: float) -> float:
        """计算时刻 t 的估计置信度。

        置信度基于:
        1. 距上次测量的时间间隔 (越远越低)
        2. 当前协方差矩阵的迹 (越大越低)
        3. 是否超出最大预测范围

        Parameters
        ----------
        t : float
            查询时间戳 (秒)。

        Returns
        -------
        float
            置信度值 [0, 1]。1.0 = 完全确信，0.0 = 完全不确定。
        """
        if not self._initialized or self._last_measurement_time is None:
            return 0.0

        # 时间衰减因子
        dt_since_measurement = t - self._last_measurement_time
        if dt_since_measurement < 0:
            dt_since_measurement = 0.0

        # 基于协方差的置信度
        cov_confidence = self._compute_confidence_from_cov(self._covariance)

        # 时间衰减: 指数衰减
        tau = self._config.max_prediction_horizon / 3.0
        time_factor = np.exp(-dt_since_measurement / tau)

        # 超出预测范围的惩罚
        horizon_penalty = 1.0
        if dt_since_measurement > self._config.max_prediction_horizon:
            excess = (
                dt_since_measurement - self._config.max_prediction_horizon
            )
            horizon_penalty = np.exp(-excess / tau)

        confidence = float(
            np.clip(cov_confidence * time_factor * horizon_penalty, 0.0, 1.0)
        )
        return confidence

    def detect_anomaly(self, measurement: np.ndarray) -> Dict[str, Any]:
        """检测测量值是否异常。

        基于 Mahalanobis 距离判断测量值与预测状态的一致性。
        Mahalanobis 距离考虑了状态协方差，因此对不确定性的
        变化具有自适应性。

        Parameters
        ----------
        measurement : np.ndarray
            测量向量 [x, y]，形状 (2,)。

        Returns
        -------
        Dict[str, Any]
            异常检测结果字典:
            - 'is_anomalous': bool — 是否异常
            - 'mahalanobis_distance': float — Mahalanobis 距离
            - 'threshold': float — 检测阈值
            - 'residual': np.ndarray — 测量残差 [dx, dy]
            - 'normalized_residual': np.ndarray — 归一化残差

        Raises
        ------
        ValueError
            如果 measurement 形状不正确。
        RuntimeError
            如果估计器未初始化。
        """
        if not self._initialized:
            raise RuntimeError("估计器尚未初始化，无法进行异常检测")

        measurement = np.asarray(measurement, dtype=np.float64)
        if measurement.shape != (2,):
            raise ValueError(
                f"measurement 形状必须为 (2,)，收到 {measurement.shape}"
            )

        # 预测的测量值
        z_pred = self._H @ self._state

        # 残差
        residual = measurement - z_pred

        # 新息协方差
        S = self._H @ self._covariance @ self._H.T + self._R

        # Mahalanobis 距离: d = sqrt(residual^T @ S^{-1} @ residual)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        mahal_sq = float(residual @ S_inv @ residual)
        mahal_dist = np.sqrt(max(mahal_sq, 0.0))

        # 归一化残差
        try:
            S_sqrt_inv = np.linalg.cholesky(np.linalg.inv(S)).T
            normalized_residual = S_sqrt_inv @ residual
        except np.linalg.LinAlgError:
            normalized_residual = residual / (np.std(residual) + 1e-10)

        is_anomalous = mahal_dist > self._config.anomaly_threshold

        # 更新统计
        if mahal_dist > self._max_mahalanobis_seen:
            self._max_mahalanobis_seen = mahal_dist

        return {
            "is_anomalous": is_anomalous,
            "mahalanobis_distance": mahal_dist,
            "threshold": self._config.anomaly_threshold,
            "residual": residual.copy(),
            "normalized_residual": normalized_residual.copy(),
        }

    def interpolate_between_detections(
        self,
        t1: float,
        state1: np.ndarray,
        t2: float,
        state2: np.ndarray,
        t_query: float,
    ) -> np.ndarray:
        """在两次检测之间进行光滑插值。

        使用 OU 漂移模型提供物理上合理的轨迹插值，
        而非简单的线性插值。插值考虑了均值回复效应和速度衰减。

        Parameters
        ----------
        t1 : float
            第一次检测时间。
        state1 : np.ndarray
            第一次检测状态 [x, y, vx, vy] 或 [x, y]。
        t2 : float
            第二次检测时间。
        state2 : np.ndarray
            第二次检测状态 [x, y, vx, vy] 或 [x, y]。
        t_query : float
            查询时间。必须在 [t1, t2] 范围内。

        Returns
        -------
        np.ndarray
            插值后的状态 [x, y, vx, vy]，形状 (4,)。

        Raises
        ------
        ValueError
            如果参数无效或时间范围不正确。
        """
        # 输入验证
        state1 = np.asarray(state1, dtype=np.float64).ravel()
        state2 = np.asarray(state2, dtype=np.float64).ravel()

        if t1 >= t2:
            raise ValueError(f"t1 ({t1}) 必须小于 t2 ({t2})")

        if t_query < t1 or t_query > t2:
            raise ValueError(
                f"t_query ({t_query}) 必须在 [t1={t1}, t2={t2}] 范围内"
            )

        # 补全状态向量 (如果只提供了位置)
        if state1.shape == (2,):
            s1 = np.array([state1[0], state1[1], 0.0, 0.0])
        elif state1.shape == (4,):
            s1 = state1.copy()
        else:
            raise ValueError(
                f"state1 形状必须为 (2,) 或 (4,)，收到 {state1.shape}"
            )

        if state2.shape == (2,):
            s2 = np.array([state2[0], state2[1], 0.0, 0.0])
        elif state2.shape == (4,):
            s2 = state2.copy()
        else:
            raise ValueError(
                f"state2 形状必须为 (2,) 或 (4,)，收到 {state2.shape}"
            )

        theta = self._config.drift_rate
        mu_x, mu_y = self._config.equilibrium_position
        decay_v = self._config.velocity_decay_rate

        # 总时间跨度
        dt_total = t2 - t1

        # --- 方法: 使用 OU 过程的均值函数进行插值 ---
        # OU 过程条件均值:
        # E[X(t) | X(t1)=s1, X(t2)=s2] 是一个加权组合

        # 前向传播: 从 s1 到 t_query
        dt_fwd = t_query - t1
        exp_fwd = np.exp(-theta * dt_fwd)
        exp_fwd_v = np.exp(-decay_v * dt_fwd)

        x_fwd = mu_x + (s1[0] - mu_x) * exp_fwd
        y_fwd = mu_y + (s1[1] - mu_y) * exp_fwd
        vx_fwd = s1[2] * exp_fwd_v
        vy_fwd = s1[3] * exp_fwd_v

        # 后向传播: 从 s2 回到 t_query
        dt_bwd = t2 - t_query
        exp_bwd = np.exp(-theta * dt_bwd)
        exp_bwd_v = np.exp(-decay_v * dt_bwd)

        x_bwd = mu_x + (s2[0] - mu_x) * exp_bwd
        y_bwd = mu_y + (s2[1] - mu_y) * exp_bwd
        vx_bwd = s2[2] * exp_bwd_v
        vy_bwd = s2[3] * exp_bwd_v

        # 布朗桥加权: 基于时间比例的加权平均
        # 权重与 OU 过程的条件协方差成反比
        if dt_fwd < 1e-12:
            # t_query == t1
            return s1.copy()
        if dt_bwd < 1e-12:
            # t_query == t2
            return s2.copy()

        # 前向和后向的协方差 (仅位置部分)
        sigma_sq = self._config.noise_intensity ** 2
        var_fwd = sigma_sq / (2 * theta) * (1 - np.exp(-2 * theta * dt_fwd))
        var_bwd = sigma_sq / (2 * theta) * (1 - np.exp(-2 * theta * dt_bwd))

        # 加权: 协方差越小权重越大
        w_fwd = 1.0 / (var_fwd + 1e-12)
        w_bwd = 1.0 / (var_bwd + 1e-12)
        w_total = w_fwd + w_bwd

        # 位置插值
        x_interp = (w_fwd * x_fwd + w_bwd * x_bwd) / w_total
        y_interp = (w_fwd * y_fwd + w_bwd * y_bwd) / w_total

        # 速度插值 (简单加权平均)
        vx_interp = (w_fwd * vx_fwd + w_bwd * vx_bwd) / w_total
        vy_interp = (w_fwd * vy_fwd + w_bwd * vy_bwd) / w_total

        result = np.array([
            x_interp, y_interp, vx_interp, vy_interp
        ], dtype=np.float64)

        LOGGER.debug(
            f"轨迹插值: t_query={t_query:.4f}, "
            f"result=({x_interp:.2f}, {y_interp:.2f})"
        )

        return result

    def get_health_report(self) -> Dict[str, Any]:
        """返回估计器健康状态报告。

        Returns
        -------
        Dict[str, Any]
            健康报告字典，包含:
            - 'is_initialized': bool — 是否已初始化
            - 'total_updates': int — 总更新次数
            - 'total_predictions': int — 总预测次数
            - 'total_anomalies': int — 检测到的异常总数
            - 'anomaly_rate': float — 异常率
            - 'max_mahalanobis_seen': float — 最大 Mahalanobis 距离
            - 'current_position': np.ndarray — 当前位置 [x, y]
            - 'current_velocity': np.ndarray — 当前速度 [vx, vy]
            - 'position_uncertainty': float — 位置不确定性 (std)
            - 'velocity_uncertainty': float — 速度不确定性 (std)
            - 'current_confidence': float — 当前置信度
            - 'time_since_last_measurement': float — 距上次测量的时间
            - 'config': dict — 当前配置参数
            - 'uptime_seconds': float — 运行时间
            - 'status': str — 整体状态 ('healthy'/'warning'/'critical')
        """
        if self._initialized:
            pos_unc = float(np.sqrt(self._covariance[0, 0] + self._covariance[1, 1]) / np.sqrt(2))
            vel_unc = float(np.sqrt(self._covariance[2, 2] + self._covariance[3, 3]) / np.sqrt(2))
            current_time = time.time()
            current_conf = self.compute_confidence(
                self._last_time if self._last_time is not None else current_time
            )
            time_since_meas = (
                0.0 if self._last_measurement_time is None
                else current_time - self._last_measurement_time
            )
        else:
            pos_unc = float("inf")
            vel_unc = float("inf")
            current_conf = 0.0
            time_since_meas = float("inf")

        # 异常率
        anomaly_rate = (
            self._total_anomalies / max(self._total_updates, 1)
        )

        # 整体状态判定
        if not self._initialized:
            status = "critical"
        elif current_conf < 0.3 or anomaly_rate > 0.5:
            status = "critical"
        elif current_conf < 0.6 or anomaly_rate > 0.2:
            status = "warning"
        else:
            status = "healthy"

        uptime = time.time() - self._creation_time

        report = {
            "is_initialized": self._initialized,
            "total_updates": self._total_updates,
            "total_predictions": self._total_predictions,
            "total_anomalies": self._total_anomalies,
            "anomaly_rate": anomaly_rate,
            "max_mahalanobis_seen": self._max_mahalanobis_seen,
            "current_position": self._state[0:2].copy() if self._initialized else np.zeros(2),
            "current_velocity": self._state[2:4].copy() if self._initialized else np.zeros(2),
            "position_uncertainty": pos_unc,
            "velocity_uncertainty": vel_unc,
            "current_confidence": current_conf,
            "time_since_last_measurement": time_since_meas,
            "config": {
                "drift_rate": self._config.drift_rate,
                "noise_intensity": self._config.noise_intensity,
                "process_noise": self._config.process_noise,
                "measurement_noise": self._config.measurement_noise,
                "max_prediction_horizon": self._config.max_prediction_horizon,
                "anomaly_threshold": self._config.anomaly_threshold,
            },
            "uptime_seconds": uptime,
            "status": status,
        }

        LOGGER.debug(f"健康报告: status={status}, conf={current_conf:.3f}")
        return report

    def reset(self) -> None:
        """重置估计器到未初始化状态。"""
        self._state = np.zeros(4, dtype=np.float64)
        self._covariance = np.eye(4, dtype=np.float64)
        self._covariance[0, 0] = self._config.initial_position_uncertainty
        self._covariance[1, 1] = self._config.initial_position_uncertainty
        self._covariance[2, 2] = self._config.initial_velocity_uncertainty
        self._covariance[3, 3] = self._config.initial_velocity_uncertainty

        self._last_time = None
        self._last_measurement_time = None
        self._initialized = False

        # 不重置统计信息 (保留历史)
        LOGGER.info("状态估计器已重置")

    # ==================== 内部方法 ====================

    def _propagate_state(self, dt: float) -> None:
        """使用 OU 过程解析解传播状态 (原地修改)。

        Parameters
        ----------
        dt : float
            时间步长 (秒)。
        """
        theta = self._config.drift_rate
        mu_x, mu_y = self._config.equilibrium_position
        decay_v = self._config.velocity_decay_rate

        exp_neg_theta_dt = np.exp(-theta * dt)
        exp_neg_decay_dt = np.exp(-decay_v * dt)

        # 位置: 均值回复
        self._state[0] = mu_x + (self._state[0] - mu_x) * exp_neg_theta_dt
        self._state[1] = mu_y + (self._state[1] - mu_y) * exp_neg_theta_dt

        # 速度: 指数衰减
        self._state[2] *= exp_neg_decay_dt
        self._state[3] *= exp_neg_decay_dt

    def _propagate_covariance(self, dt: float) -> None:
        """传播协方差矩阵 (原地修改)。

        使用解析解进行协方差传播，保证数值精度。

        Parameters
        ----------
        dt : float
            时间步长 (秒)。
        """
        self._covariance = self._propagate_covariance_analytical(
            self._covariance, dt
        )

    def _propagate_covariance_analytical(
        self, P: np.ndarray, dt: float
    ) -> np.ndarray:
        """OU 过程的解析协方差传播。

        对于 OU 过程 dx = -theta*(x-mu)*dt + sigma*dW，
        条件协方差有解析解:
        Var(t+dt) = Var(t)*exp(-2*theta*dt) + sigma^2/(2*theta)*(1-exp(-2*theta*dt))

        对于 4 维状态 [x, y, vx, vy]，位置分量使用 OU 协方差，
        速度分量使用指数衰减模型。

        Parameters
        ----------
        P : np.ndarray
            当前协方差矩阵 (4x4)。
        dt : float
            时间步长 (秒)。

        Returns
        -------
        np.ndarray
            传播后的协方差矩阵 (4x4)。
        """
        theta = self._config.drift_rate
        sigma = self._config.noise_intensity
        q = self._config.process_noise
        decay_v = self._config.velocity_decay_rate

        exp_neg_2theta_dt = np.exp(-2 * theta * dt)
        exp_neg_2decay_dt = np.exp(-2 * decay_v * dt)

        # OU 过程稳态方差
        if theta > 1e-12:
            ou_stationary_var = (sigma ** 2) / (2 * theta)
        else:
            ou_stationary_var = sigma ** 2 * dt  # 退化为 Wiener 过程

        # 位置协方差: OU 过程
        pos_factor = exp_neg_2theta_dt
        pos_add = ou_stationary_var * (1 - pos_factor) + q * dt

        # 速度协方差: 指数衰减 + 过程噪声
        vel_factor = exp_neg_2decay_dt
        vel_add = q * dt * 0.1  # 速度噪声较小

        # 交叉协方差衰减
        cross_factor_pos = np.exp(-theta * dt)
        cross_factor_vel = np.exp(-decay_v * dt)
        cross_decay = cross_factor_pos * cross_factor_vel

        P_new = P.copy()

        # 位置-位置协方差
        P_new[0, 0] = P[0, 0] * pos_factor + pos_add
        P_new[1, 1] = P[1, 1] * pos_factor + pos_add

        # 速度-速度协方差
        P_new[2, 2] = P[2, 2] * vel_factor + vel_add
        P_new[3, 3] = P[3, 3] * vel_factor + vel_add

        # 位置-速度交叉协方差
        P_new[0, 2] = P[0, 2] * cross_decay
        P_new[2, 0] = P_new[0, 2]
        P_new[0, 3] = P[0, 3] * cross_decay
        P_new[3, 0] = P_new[0, 3]
        P_new[1, 2] = P[1, 2] * cross_decay
        P_new[2, 1] = P_new[1, 2]
        P_new[1, 3] = P[1, 3] * cross_decay
        P_new[3, 1] = P_new[1, 3]

        # 位置-位置交叉协方差 (x-y 耦合)
        P_new[0, 1] = P[0, 1] * pos_factor
        P_new[1, 0] = P_new[0, 1]

        # 速度-速度交叉协方差 (vx-vy 耦合)
        P_new[2, 3] = P[2, 3] * vel_factor
        P_new[3, 2] = P_new[2, 3]

        # 确保对称性和正定性
        P_new = 0.5 * (P_new + P_new.T)

        # 对角线最小值保证正定性
        min_diag = 1e-10
        for i in range(4):
            if P_new[i, i] < min_diag:
                P_new[i, i] = min_diag

        return P_new

    def _compute_confidence_from_cov(self, P: np.ndarray) -> float:
        """基于协方差矩阵计算置信度。

        使用位置不确定性的归一化值作为置信度指标。

        Parameters
        ----------
        P : np.ndarray
            状态协方差矩阵 (4x4)。

        Returns
        -------
        float
            置信度 [0, 1]。
        """
        # 位置不确定性 (位置协方差的平均标准差)
        pos_var = 0.5 * (P[0, 0] + P[1, 1])
        pos_std = np.sqrt(max(pos_var, 1e-12))

        # 归一化: 使用初始不确定性作为参考尺度
        ref_std = np.sqrt(self._config.initial_position_uncertainty)
        if ref_std < 1e-12:
            ref_std = 1.0

        # Sigmoid 型映射: 不确定性越小置信度越高
        ratio = pos_std / ref_std
        confidence = 1.0 / (1.0 + ratio ** 2)

        return float(np.clip(confidence, 0.0, 1.0))
