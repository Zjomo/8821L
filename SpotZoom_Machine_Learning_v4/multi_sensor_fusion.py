"""
多模态传感器融合定位器 (Multi-Sensor Fusion Localizer)

灵感来源: robot_localization (https://github.com/cra-ros-pkg/robot_localization)
           扩展卡尔曼滤波 (EKF) 传感器融合框架

核心思想:
──────────────────────────────────────────────────────────────────────
robot_localization 是 ROS 生态中最广泛使用的传感器融合框架，支持
多传感器异步数据融合、状态向量增广、协方差交叉 (CI) 等技术。

本模块将其核心思想适配到光斑检测场景:
- 融合 YOLO 检测、经典算法检测、卡尔曼预测等多种位置估计
- 异步传感器数据的时间对齐与融合
- 协方差交叉算法处理不一致估计
- 自适应噪声估计应对传感器退化

适用场景:
- 多检测器并行运行时的位置融合
- 检测器置信度动态变化时的鲁棒定位
- 高精度亚像素定位的多源信息整合
"""

__version__ = "1.0.0"
__author__ = "SpotZoom Team"

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import time


class SensorType(Enum):
    """传感器类型枚举"""
    YOLO_DETECTOR = "yolo"
    CLASSIC_DETECTOR = "classic"
    KALMAN_PREDICTOR = "kalman"
    SUBPIXEL_CENTROID = "subpixel"
    OPTICAL_FLOW = "optical_flow"
    EXTERNAL = "external"


@dataclass
class SensorReading:
    """单次传感器读数"""
    sensor_type: SensorType
    position: np.ndarray          # [x, y] 像素坐标
    timestamp: float              # 时间戳 (秒)
    confidence: float = 1.0       # 置信度 [0, 1]
    covariance: Optional[np.ndarray] = None  # 2x2 协方差矩阵
    raw_score: float = 1.0        # 原始检测分数

    def __post_init__(self):
        if self.covariance is None:
            # 基于置信度生成默认协方差
            sigma = max(0.5, 10.0 * (1.0 - self.confidence))
            self.covariance = np.eye(2) * sigma ** 2


@dataclass
class FusionConfig:
    """融合配置"""
    # 过程噪声 (像素^2/s^2)
    process_noise: float = 5.0
    # 最大缓冲区大小 (每个传感器)
    max_buffer_size: int = 50
    # 时间对齐窗口 (秒)
    time_alignment_window: float = 0.1
    # 协方差交叉权重 (0=纯EKF, 1=纯CI)
    ci_weight: float = 0.3
    # 最小融合传感器数
    min_sensors: int = 1
    # 异常值拒绝阈值 (马氏距离)
    outlier_threshold: float = 5.0
    # 速度估计窗口
    velocity_window: int = 5
    # 自适应噪声启用
    adaptive_noise: bool = True
    # 传感器可靠性衰减因子
    reliability_decay: float = 0.995
    # 最低可靠性阈值
    min_reliability: float = 0.1


@dataclass
class FusionResult:
    """融合结果"""
    fused_position: np.ndarray       # [x, y]
    fused_velocity: np.ndarray       # [vx, vy]
    fused_covariance: np.ndarray     # 2x2
    timestamp: float
    num_sensors_used: int
    sensor_types_used: List[str]
    innovation: float                # 新息 (马氏距离)
    is_outlier: bool = False
    individual_contributions: Dict[str, float] = field(default_factory=dict)


class MultiSensorFusion:
    """
    多模态传感器融合定位器

    基于 robot_localization 的 EKF 融合框架，支持:
    - 多传感器异步数据融合
    - 协方差交叉 (Covariance Intersection) 处理不一致估计
    - 自适应噪声估计
    - 异常值检测与拒绝
    - 速度估计与预测

    用法示例:
        fusion = MultiSensorFusion(FusionConfig())
        fusion.add_reading(SensorReading(
            sensor_type=SensorType.YOLO_DETECTOR,
            position=np.array([320.0, 240.0]),
            timestamp=time.time(),
            confidence=0.95
        ))
        result = fusion.get_fused_state()
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

        # 状态向量: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.covariance = np.eye(4) * 100.0  # 初始大协方差

        # 传感器缓冲区
        self._buffers: Dict[SensorType, List[SensorReading]] = {
            st: [] for st in SensorType
        }

        # 传感器可靠性跟踪
        self._reliability: Dict[SensorType, float] = {
            st: 1.0 for st in SensorType
        }

        # 历史位置 (用于速度估计)
        self._position_history: List[Tuple[float, np.ndarray]] = []

        # 统计信息
        self._total_fusions = 0
        self._outlier_count = 0
        self._last_update_time: Optional[float] = None

    def add_reading(self, reading: SensorReading) -> None:
        """
        添加传感器读数到缓冲区

        Args:
            reading: 传感器读数
        """
        buffer = self._buffers[reading.sensor_type]
        buffer.append(reading)

        # 限制缓冲区大小
        if len(buffer) > self.config.max_buffer_size:
            buffer.pop(0)

    def get_fused_state(self, current_time: Optional[float] = None) -> FusionResult:
        """
        获取当前融合状态

        Args:
            current_time: 当前时间戳 (None 则使用最新读数时间)

        Returns:
            FusionResult: 融合结果
        """
        if current_time is None:
            current_time = time.time()

        # 收集有效传感器读数
        active_readings = self._collect_active_readings(current_time)

        if len(active_readings) < self.config.min_sensors:
            # 传感器不足，仅做预测
            self._predict(current_time)
            return FusionResult(
                fused_position=self.state[:2].copy(),
                fused_velocity=self.state[2:].copy(),
                fused_covariance=self.covariance[:2, :2].copy(),
                timestamp=current_time,
                num_sensors_used=0,
                sensor_types_used=[],
                innovation=0.0
            )

        # EKF 预测步骤
        self._predict(current_time)

        # 多传感器融合更新
        return self._fuse_measurements(active_readings, current_time)

    def _collect_active_readings(self, current_time: float) -> List[SensorReading]:
        """收集时间窗口内的有效传感器读数"""
        active = []
        window = self.config.time_alignment_window

        for sensor_type, buffer in self._buffers.items():
            if not buffer:
                continue

            # 找到时间窗口内最新的读数
            best = None
            for reading in reversed(buffer):
                if abs(reading.timestamp - current_time) <= window:
                    best = reading
                    break

            if best is not None:
                # 应用可靠性加权
                reliability = self._reliability[sensor_type]
                if reliability >= self.config.min_reliability:
                    active.append(best)

        return active

    def _predict(self, current_time: float) -> None:
        """EKF 预测步骤 (匀速运动模型)"""
        if self._last_update_time is None:
            self._last_update_time = current_time
            return

        dt = current_time - self._last_update_time
        if dt <= 0:
            return

        # 状态转移矩阵 (匀速模型)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 过程噪声矩阵
        q = self.config.process_noise
        Q = np.array([
            [q * dt**3 / 3, 0, q * dt**2 / 2, 0],
            [0, q * dt**3 / 3, 0, q * dt**2 / 2],
            [q * dt**2 / 2, 0, q * dt, 0],
            [0, q * dt**2 / 2, 0, q * dt]
        ])

        # 预测
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
        self._last_update_time = current_time

    def _fuse_measurements(self, readings: List[SensorReading],
                           current_time: float) -> FusionResult:
        """多传感器融合更新"""
        if len(readings) == 1:
            # 单传感器直接 EKF 更新
            return self._ekf_update(readings[0], current_time)

        # 多传感器: 协方差交叉 (CI) + EKF 混合
        return self._covariance_intersection(readings, current_time)

    def _ekf_update(self, reading: SensorReading,
                    current_time: float) -> FusionResult:
        """标准 EKF 更新"""
        # 观测矩阵
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 观测噪声
        R = reading.covariance.copy()
        # 加入可靠性因子
        reliability = self._reliability[reading.sensor_type]
        R /= max(reliability, self.config.min_reliability)

        # 新息
        z = reading.position
        z_pred = H @ self.state
        innovation = z - z_pred

        # 新息协方差
        S = H @ self.covariance @ H.T + R

        # 马氏距离 (异常检测)
        try:
            S_inv = np.linalg.inv(S)
            mahal_dist = float(innovation.T @ S_inv @ innovation)
        except np.linalg.LinAlgError:
            mahal_dist = 0.0

        is_outlier = mahal_dist > self.config.outlier_threshold ** 2

        if not is_outlier:
            # 卡尔曼增益
            K = self.covariance @ H.T @ S_inv

            # 状态更新
            self.state = self.state + K @ innovation

            # Joseph 形式协方差更新 (数值稳定)
            I_KH = np.eye(4) - K @ H
            self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R @ K.T

            # 更新可靠性
            self._update_reliability(reading.sensor_type, mahal_dist, True)
        else:
            self._outlier_count += 1
            self._update_reliability(reading.sensor_type, mahal_dist, False)

        self._total_fusions += 1

        return FusionResult(
            fused_position=self.state[:2].copy(),
            fused_velocity=self.state[2:].copy(),
            fused_covariance=self.covariance[:2, :2].copy(),
            timestamp=current_time,
            num_sensors_used=1,
            sensor_types_used=[reading.sensor_type.value],
            innovation=np.sqrt(mahal_dist),
            is_outlier=is_outlier,
            individual_contributions={reading.sensor_type.value: reading.confidence}
        )

    def _covariance_intersection(self, readings: List[SensorReading],
                                  current_time: float) -> FusionResult:
        """
        协方差交叉 (Covariance Intersection) 融合

        当多个传感器估计不一致时，CI 算法能保证融合结果的一致性，
        不需要知道传感器之间的交叉相关性。

        参考: Julier, S.J., Uhlmann, J.K. (1997)
        "A Non-divergent Estimation Algorithm in the Presence of Unknown Correlations"
        """
        n = len(readings)
        ci_w = self.config.ci_weight

        # 对每个传感器进行 EKF 更新，收集局部估计
        local_estimates = []
        for reading in readings:
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            R = reading.covariance.copy()
            reliability = self._reliability[reading.sensor_type]
            R /= max(reliability, self.config.min_reliability)

            z = reading.position
            z_pred = H @ self.state
            innov = z - z_pred
            S = H @ self.covariance @ H.T + R

            try:
                S_inv = np.linalg.inv(S)
                K = self.covariance @ H.T @ S_inv
                x_local = self.state + K @ innov
                P_local = (np.eye(4) - K @ H) @ self.covariance @ (np.eye(4) - K @ H).T + K @ R @ K.T
                local_estimates.append((x_local, P_local, reading))
            except np.linalg.LinAlgError:
                local_estimates.append((self.state.copy(), self.covariance.copy(), reading))

        if len(local_estimates) == 1:
            x_local, P_local, reading = local_estimates[0]
            self.state = x_local
            self.covariance = P_local
            return FusionResult(
                fused_position=self.state[:2].copy(),
                fused_velocity=self.state[2:].copy(),
                fused_covariance=self.covariance[:2, :2].copy(),
                timestamp=current_time,
                num_sensors_used=1,
                sensor_types_used=[reading.sensor_type.value],
                innovation=0.0,
                individual_contributions={reading.sensor_type.value: reading.confidence}
            )

        # CI 融合: P_ci^{-1} = sum(w_i * P_i^{-1}), x_ci = P_ci * sum(w_i * P_i^{-1} * x_i)
        weights = np.array([r.confidence * self._reliability[r.sensor_type]
                           for _, _, r in local_estimates])
        weights = weights / weights.sum()

        P_ci_inv = np.zeros((4, 4))
        x_ci_weighted = np.zeros(4)

        for i, (x_i, P_i, reading) in enumerate(local_estimates):
            w = weights[i]
            try:
                P_i_inv = np.linalg.inv(P_i)
            except np.linalg.LinAlgError:
                P_i_inv = np.eye(4) * 0.01

            # 混合 EKF 和 CI
            blended_inv = (1 - ci_w) * np.eye(4) / np.trace(self.covariance) + ci_w * P_i_inv
            P_ci_inv += w * blended_inv
            x_ci_weighted += w * blended_inv @ x_i

        try:
            P_ci = np.linalg.inv(P_ci_inv)
            x_ci = P_ci @ x_ci_weighted
        except np.linalg.LinAlgError:
            # 回退到加权平均
            x_ci = sum(w * x for w, (x, _, _) in zip(weights, local_estimates))
            P_ci = self.covariance.copy()

        self.state = x_ci
        self.covariance = P_ci

        # 更新各传感器可靠性
        for _, _, reading in local_estimates:
            self._update_reliability(reading.sensor_type, 0.0, True)

        self._total_fusions += 1

        contributions = {r.sensor_type.value: float(w)
                        for (_, _, r), w in zip(local_estimates, weights)}

        return FusionResult(
            fused_position=self.state[:2].copy(),
            fused_velocity=self.state[2:].copy(),
            fused_covariance=self.covariance[:2, :2].copy(),
            timestamp=current_time,
            num_sensors_used=len(local_estimates),
            sensor_types_used=[r.sensor_type.value for _, _, r in local_estimates],
            innovation=0.0,
            individual_contributions=contributions
        )

    def _update_reliability(self, sensor_type: SensorType,
                            mahal_dist: float, accepted: bool) -> None:
        """更新传感器可靠性估计"""
        if not self.config.adaptive_noise:
            return

        decay = self.config.reliability_decay
        if accepted:
            # 正常更新: 可靠性缓慢恢复
            self._reliability[sensor_type] = min(
                1.0,
                self._reliability[sensor_type] * decay + (1 - decay) * 0.8
            )
        else:
            # 异常: 可靠性快速下降
            self._reliability[sensor_type] = max(
                self.config.min_reliability,
                self._reliability[sensor_type] * decay * 0.5
            )

    def reset(self) -> None:
        """重置融合器状态"""
        self.state = np.zeros(4)
        self.covariance = np.eye(4) * 100.0
        self._buffers = {st: [] for st in SensorType}
        self._reliability = {st: 1.0 for st in SensorType}
        self._position_history = []
        self._total_fusions = 0
        self._outlier_count = 0
        self._last_update_time = None

    def get_statistics(self) -> Dict:
        """获取融合统计信息"""
        return {
            "total_fusions": self._total_fusions,
            "outlier_count": self._outlier_count,
            "outlier_rate": (self._outlier_count / max(1, self._total_fusions)),
            "sensor_reliability": {
                st.value: round(rel, 3)
                for st, rel in self._reliability.items()
            },
            "buffer_sizes": {
                st.value: len(buf)
                for st, buf in self._buffers.items()
            },
            "state": self.state.tolist(),
            "position_uncertainty": float(np.sqrt(np.trace(self.covariance[:2, :2])))
        }
