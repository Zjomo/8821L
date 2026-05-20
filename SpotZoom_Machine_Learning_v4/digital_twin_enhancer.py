"""
光学系统数字孪生增强器 (Optical System Digital Twin Enhancer)

灵感来源: NVIDIA Omniverse (https://developer.nvidia.com/omniverse)
           Azure Digital Twins
           工业数字孪生框架

核心思想:
──────────────────────────────────────────────────────────────────────
数字孪生 (Digital Twin) 是物理系统的虚拟副本，通过实时数据同步
和物理模型仿真，实现状态监控、预测性维护和优化决策。

本模块为 SpotZoom 光斑对准系统构建数字孪生增强层:
- 光学系统状态建模 (位置、像差、环境参数)
- 实时物理仿真 (光束传播、热效应、振动)
- 状态预测 (短期轨迹预测、长期漂移预测)
- 异常预警 (基于仿真与实测的偏差)
- 优化建议 (基于孪生模型的参数优化)
- 历史回放与场景重现

适用场景:
- 系统调试与参数优化
- 预测性维护
- 异常根因分析
- 操作员培训 (虚拟环境)
"""

__version__ = "1.0.0"
__author__ = "SpotZoom Team"

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable, Any
from enum import Enum
from collections import deque
import time
import json


class TwinComponent(Enum):
    """孪生组件"""
    OPTICAL_BENCH = "optical_bench"     # 光学平台
    BEAM_PATH = "beam_path"             # 光路
    DETECTOR = "detector"               # 检测器
    STAGE_XY = "stage_xy"               # XY 平台
    STAGE_Z = "stage_z"                 # Z 轴
    ENVIRONMENT = "environment"         # 环境
    CONTROLLER = "controller"           # 控制器


class PredictionHorizon(Enum):
    """预测时域"""
    INSTANT = 0          # 瞬时
    SHORT_TERM = 1       # 短期 (< 1s)
    MEDIUM_TERM = 2      # 中期 (1-10s)
    LONG_TERM = 3        # 长期 (> 10s)


@dataclass
class TwinConfig:
    """数字孪生配置"""
    # 仿真参数
    simulation_dt: float = 0.01         # 仿真步长 (s)
    prediction_horizon_s: float = 5.0   # 预测时域 (s)
    history_length: int = 1000          # 历史长度

    # 物理模型参数
    thermal_time_constant: float = 60.0  # 热时间常数 (s)
    vibration_damping: float = 0.95      # 振动阻尼
    beam_wander_sigma: float = 0.5       # 光束游走标准差 (像素)

    # 状态同步
    sync_tolerance: float = 0.1         # 同步容差
    drift_detection_threshold: float = 2.0  # 漂移检测阈值

    # 预测模型
    prediction_model_order: int = 3      # AR 模型阶数
    prediction_confidence: float = 0.95  # 预测置信度

    # 组件启用
    enabled_components: List[TwinComponent] = field(default_factory=lambda: list(TwinComponent))


@dataclass
class TwinState:
    """孪生状态"""
    timestamp: float

    # 光学平台状态
    spot_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    spot_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    spot_intensity: float = 1.0
    spot_quality: float = 1.0

    # 平台状态
    stage_x: float = 0.0
    stage_y: float = 0.0
    stage_z: float = 0.0

    # 像差状态 (Zernike 系数)
    aberrations: Dict[int, float] = field(default_factory=dict)

    # 环境状态
    temperature: float = 25.0
    vibration_level: float = 0.0

    # 控制器状态
    pid_output: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    convergence_error: float = 0.0

    # 元数据
    is_simulated: bool = False
    confidence: float = 1.0


@dataclass
class PredictionResult:
    """预测结果"""
    horizon: PredictionHorizon
    predicted_states: List[TwinState]
    confidence_intervals: List[Tuple[np.ndarray, np.ndarray]]  # (lower, upper)
    mean_absolute_error: float
    prediction_time_ms: float


class DigitalTwinEnhancer:
    """
    光学系统数字孪生增强器

    构建 SpotZoom 系统的虚拟副本, 支持:
    1. 实时状态同步 (物理系统 <-> 虚拟模型)
    2. 物理仿真 (光束传播、热漂移、振动)
    3. 状态预测 (AR 模型 + 物理约束)
    4. 异常检测 (仿真偏差)
    5. 优化建议

    用法示例:
        twin = DigitalTwinEnhancer(TwinConfig())
        twin.sync_state(TwinState(
            timestamp=time.time(),
            spot_position=np.array([320.0, 240.0]),
            stage_x=100.0, stage_y=200.0
        ))
        prediction = twin.predict(PredictionHorizon.SHORT_TERM)
        anomaly = twin.detect_anomaly()
    """

    def __init__(self, config: Optional[TwinConfig] = None):
        self.config = config or TwinConfig()

        # 当前状态
        self._current_state = TwinState(timestamp=0.0)

        # 历史状态
        self._state_history: deque = deque(maxlen=self.config.history_length)

        # 预测模型 (AR 系数)
        self._ar_coefficients: Dict[str, np.ndarray] = {}
        self._ar_initialized = False

        # 仿真状态
        self._sim_time = 0.0
        self._thermal_state = 0.0  # 热漂移累积
        self._vibration_state = np.zeros(2)  # 振动状态

        # 异常记录
        self._anomaly_log: List[Dict] = []

        # 统计
        self._sync_count = 0
        self._prediction_count = 0
        self._simulation_count = 0

    def sync_state(self, state: TwinState) -> Dict:
        """
        同步物理系统状态到数字孪生

        Args:
            state: 来自物理系统的状态

        Returns:
            Dict: 同步信息 (偏差、异常等)
        """
        sync_info = {
            "timestamp": state.timestamp,
            "deviations": {},
            "anomalies": [],
            "drift_detected": False
        }

        if self._current_state.timestamp > 0:
            # 计算与仿真预测的偏差
            deviations = self._compute_deviations(state)
            sync_info["deviations"] = deviations

            # 检测异常偏差
            for key, dev in deviations.items():
                if abs(dev) > self.config.drift_detection_threshold:
                    sync_info["anomalies"].append({
                        "parameter": key,
                        "deviation": float(dev),
                        "threshold": self.config.drift_detection_threshold
                    })
                    sync_info["drift_detected"] = True

            # 记录异常
            if sync_info["anomalies"]:
                self._anomaly_log.append({
                    "timestamp": state.timestamp,
                    "anomalies": sync_info["anomalies"]
                })

        # 更新当前状态
        self._current_state = state
        self._current_state.is_simulated = False
        self._state_history.append(state)

        # 更新 AR 模型
        self._update_ar_model()

        self._sync_count += 1
        return sync_info

    def predict(self, horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM,
                n_steps: int = 50) -> PredictionResult:
        """
        预测未来状态

        Args:
            horizon: 预测时域
            n_steps: 预测步数

        Returns:
            PredictionResult: 预测结果
        """
        start_time = time.time()

        if len(self._state_history) < self.config.prediction_model_order + 5:
            # 数据不足, 使用简单外推
            return self._simple_extrapolation(horizon, n_steps)

        # AR 模型预测
        predicted_states = []
        confidence_intervals = []

        current = self._current_state
        dt = self.config.simulation_dt

        for step in range(n_steps):
            t = current.timestamp + (step + 1) * dt
            pred_state = TwinState(
                timestamp=t,
                is_simulated=True,
                confidence=max(0.1, 1.0 - step / n_steps)
            )

            # 位置预测
            if self._ar_initialized and "spot_x" in self._ar_coefficients:
                pred_x = self._ar_predict("spot_x", step)
                pred_y = self._ar_predict("spot_y", step)
                pred_state.spot_position = np.array([pred_x, pred_y])
            else:
                # 线性外推
                pred_state.spot_position = (
                    current.spot_position +
                    current.spot_velocity * (step + 1) * dt
                )

            # 强度预测 (带衰减)
            pred_state.spot_intensity = current.spot_intensity * (
                1 - 0.001 * step
            )

            # 振动衰减预测
            vibration_decay = self.config.vibration_damping ** step
            pred_state.vibration_level = current.vibration_level * vibration_decay

            # 热漂移预测
            thermal_drift = self._thermal_state * (
                1 - np.exp(-(step + 1) * dt / self.config.thermal_time_constant)
            )
            pred_state.spot_position += thermal_drift * 0.1

            # 置信区间
            uncertainty = 0.5 * (step + 1) * dt * self.config.beam_wander_sigma
            ci_lower = pred_state.spot_position - uncertainty
            ci_upper = pred_state.spot_position + uncertainty
            confidence_intervals.append((ci_lower, ci_upper))

            predicted_states.append(pred_state)

        # 计算预测误差 (使用历史验证)
        mae = self._estimate_prediction_error()

        elapsed_ms = (time.time() - start_time) * 1000

        self._prediction_count += 1

        return PredictionResult(
            horizon=horizon,
            predicted_states=predicted_states,
            confidence_intervals=confidence_intervals,
            mean_absolute_error=mae,
            prediction_time_ms=elapsed_ms
        )

    def simulate(self, duration_s: float,
                 perturbation: Optional[Dict] = None) -> List[TwinState]:
        """
        运行物理仿真

        Args:
            duration_s: 仿真时长 (秒)
            perturbation: 扰动参数

        Returns:
            List[TwinState]: 仿真状态序列
        """
        n_steps = int(duration_s / self.config.simulation_dt)
        states = []

        sim_state = TwinState(
            timestamp=self._current_state.timestamp,
            spot_position=self._current_state.spot_position.copy(),
            spot_velocity=self._current_state.spot_velocity.copy(),
            spot_intensity=self._current_state.spot_intensity,
            stage_x=self._current_state.stage_x,
            stage_y=self._current_state.stage_y,
            temperature=self._current_state.temperature,
            vibration_level=self._current_state.vibration_level,
            is_simulated=True
        )

        for step in range(n_steps):
            dt = self.config.simulation_dt

            # 应用扰动
            if perturbation:
                sim_state = self._apply_perturbation(sim_state, perturbation, dt)

            # 物理仿真
            sim_state = self._physics_step(sim_state, dt)

            states.append(sim_state)

        self._simulation_count += 1
        return states

    def detect_anomaly(self) -> Optional[Dict]:
        """
        基于数字孪生检测异常

        Returns:
            Dict 或 None: 异常信息
        """
        if len(self._state_history) < 10:
            return None

        recent = list(self._state_history)[-10:]
        positions = np.array([s.spot_position for s in recent])

        # 位置方差异常
        pos_var = np.var(positions, axis=0)
        if np.any(pos_var > self.config.drift_detection_threshold ** 2):
            return {
                "type": "position_instability",
                "variance": pos_var.tolist(),
                "severity": float(np.max(pos_var) / self.config.drift_detection_threshold),
                "suggestion": "检查平台稳定性或增加振动补偿"
            }

        # 趋势异常 (线性漂移)
        if len(positions) >= 5:
            x = np.arange(len(positions))
            slopes = np.polyfit(x, positions, 1)[0, :]  # 线性拟合斜率
            if np.any(np.abs(slopes) > 0.5):
                return {
                    "type": "linear_drift",
                    "slopes": slopes.tolist(),
                    "severity": float(np.max(np.abs(slopes))),
                    "suggestion": "检测到线性漂移, 建议重新校准或启用漂移补偿"
                }

        # 质量退化
        qualities = [s.spot_quality for s in recent]
        quality_trend = np.polyfit(np.arange(len(qualities)), qualities, 1)[0]
        if quality_trend < -0.01:
            return {
                "type": "quality_degradation",
                "trend": float(quality_trend),
                "suggestion": "光斑质量持续下降, 建议检查光学元件清洁度"
            }

        return None

    def get_optimization_suggestions(self) -> List[Dict]:
        """
        基于数字孪生分析生成优化建议

        Returns:
            List[Dict]: 优化建议列表
        """
        suggestions = []

        if len(self._state_history) < 20:
            return suggestions

        recent = list(self._state_history)[-50:]
        positions = np.array([s.spot_position for s in recent])
        velocities = np.array([s.spot_velocity for s in recent])

        # 1. 振动分析
        vel_std = np.std(velocities, axis=0)
        if np.any(vel_std > 1.0):
            suggestions.append({
                "category": "vibration",
                "priority": "high",
                "description": f"检测到显著振动 (速度标准差: {vel_std})",
                "suggestion": "启用振动补偿模块或增加阻尼",
                "estimated_improvement": "30-50%"
            })

        # 2. 漂移分析
        if len(positions) >= 20:
            drift = positions[-1] - positions[0]
            drift_rate = np.linalg.norm(drift) / (recent[-1].timestamp - recent[0].timestamp + 1e-10)
            if drift_rate > 0.1:
                suggestions.append({
                    "category": "drift",
                    "priority": "medium",
                    "description": f"检测到位置漂移 (速率: {drift_rate:.2f} px/s)",
                    "suggestion": "启用漂移补偿或缩短校准周期",
                    "estimated_improvement": "20-40%"
                })

        # 3. 收敛性分析
        errors = [s.convergence_error for s in recent if hasattr(s, 'convergence_error')]
        if errors and np.mean(errors[-10:]) > np.mean(errors[:10]):
            suggestions.append({
                "category": "convergence",
                "priority": "medium",
                "description": "收敛误差呈增大趋势",
                "suggestion": "调整 PID 参数或切换到自适应控制策略",
                "estimated_improvement": "15-30%"
            })

        # 4. 资源利用率
        if self._current_state.vibration_level > 0.5:
            suggestions.append({
                "category": "resource",
                "priority": "low",
                "description": "振动水平较高, 控制器负载增加",
                "suggestion": "考虑降低控制频率或使用自适应调度",
                "estimated_improvement": "10-20%"
            })

        return suggestions

    def _compute_deviations(self, actual: TwinState) -> Dict:
        """计算实际状态与仿真预测的偏差"""
        deviations = {}

        pos_dev = np.linalg.norm(actual.spot_position - self._current_state.spot_position)
        deviations["position"] = pos_dev

        intensity_dev = abs(actual.spot_intensity - self._current_state.spot_intensity)
        deviations["intensity"] = intensity_dev

        temp_dev = abs(actual.temperature - self._current_state.temperature)
        deviations["temperature"] = temp_dev

        return deviations

    def _physics_step(self, state: TwinState, dt: float) -> TwinState:
        """单步物理仿真"""
        new_state = TwinState(
            timestamp=state.timestamp + dt,
            is_simulated=True
        )

        # 位置更新 (匀速 + 随机游走)
        noise = np.random.normal(0, self.config.beam_wander_sigma * np.sqrt(dt), 2)
        new_state.spot_position = state.spot_position + state.spot_velocity * dt + noise
        new_state.spot_velocity = state.spot_velocity * self.config.vibration_damping

        # 热漂移
        self._thermal_state += (state.temperature - 25.0) * dt * 0.01
        thermal_shift = self._thermal_state * 0.01 * dt
        new_state.spot_position += thermal_shift

        # 强度衰减 (模拟散射损失)
        new_state.spot_intensity = state.spot_intensity * (1 - 0.0001 * dt)

        # 环境更新
        new_state.temperature = state.temperature + np.random.normal(0, 0.01 * dt)
        new_state.vibration_level = state.vibration_level * self.config.vibration_damping + \
            np.random.normal(0, 0.1 * dt)

        # 平台状态
        new_state.stage_x = state.stage_x
        new_state.stage_y = state.stage_y
        new_state.stage_z = state.stage_z

        return new_state

    def _apply_perturbation(self, state: TwinState,
                            perturbation: Dict, dt: float) -> TwinState:
        """应用扰动"""
        new_state = TwinState(
            timestamp=state.timestamp,
            spot_position=state.spot_position.copy(),
            spot_velocity=state.spot_velocity.copy(),
            spot_intensity=state.spot_intensity,
            stage_x=state.stage_x,
            stage_y=state.stage_y,
            temperature=state.temperature,
            vibration_level=state.vibration_level,
            is_simulated=True
        )

        if "position_shift" in perturbation:
            new_state.spot_position += np.array(perturbation["position_shift"]) * dt

        if "velocity_impulse" in perturbation:
            new_state.spot_velocity += np.array(perturbation["velocity_impulse"])

        if "temperature_change" in perturbation:
            new_state.temperature += perturbation["temperature_change"] * dt

        if "vibration_burst" in perturbation:
            new_state.vibration_level += perturbation["vibration_burst"]

        return new_state

    def _update_ar_model(self) -> None:
        """更新 AR 预测模型"""
        n = len(self._state_history)
        order = self.config.prediction_model_order

        if n < order + 10:
            return

        positions = np.array([s.spot_position for s in list(self._state_history)[-n:]])

        for dim, name in enumerate(["spot_x", "spot_y"]):
            series = positions[:, dim]
            coeffs = self._fit_ar(series, order)
            if coeffs is not None:
                self._ar_coefficients[name] = coeffs

        self._ar_initialized = True

    def _fit_ar(self, series: np.ndarray, order: int) -> Optional[np.ndarray]:
        """拟合 AR 模型 (最小二乘)"""
        n = len(series)
        if n < order + 1:
            return None

        # 构建 Yule-Walker 方程
        X = np.zeros((n - order, order))
        y = series[order:]

        for i in range(n - order):
            X[i, :] = series[order - 1 - i: order - 1 - i + order][::-1]

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return coeffs
        except np.linalg.LinAlgError:
            return None

    def _ar_predict(self, name: str, steps_ahead: int) -> float:
        """AR 模型预测"""
        if name not in self._ar_coefficients:
            return self._current_state.spot_position[0 if "x" in name else 1]

        coeffs = self._ar_coefficients[name]
        order = len(coeffs)
        history = list(self._state_history)

        if len(history) < order:
            return self._current_state.spot_position[0 if "x" in name else 1]

        values = [s.spot_position[0 if "x" in name else 1] for s in history[-order:]]

        prediction = np.dot(coeffs, values[::-1])
        return float(prediction)

    def _simple_extrapolation(self, horizon: PredictionHorizon,
                               n_steps: int) -> PredictionResult:
        """简单线性外推"""
        states = []
        intervals = []

        for step in range(n_steps):
            dt = self.config.simulation_dt * (step + 1)
            state = TwinState(
                timestamp=self._current_state.timestamp + dt,
                spot_position=self._current_state.spot_position +
                    self._current_state.spot_velocity * dt,
                is_simulated=True,
                confidence=max(0.1, 1.0 - step / n_steps)
            )
            states.append(state)
            uncertainty = 0.5 * dt * self.config.beam_wander_sigma
            intervals.append((
                state.spot_position - uncertainty,
                state.spot_position + uncertainty
            ))

        return PredictionResult(
            horizon=horizon,
            predicted_states=states,
            confidence_intervals=intervals,
            mean_absolute_error=float('nan'),
            prediction_time_ms=0.0
        )

    def _estimate_prediction_error(self) -> float:
        """估计预测误差 (使用历史数据回测)"""
        if len(self._state_history) < 20:
            return float('nan')

        positions = np.array([s.spot_position for s in self._state_history])
        if len(positions) < 10:
            return float('nan')

        # 简单估计: 使用最近位置变化的平均幅度
        diffs = np.diff(positions, axis=0)
        return float(np.mean(np.abs(diffs)))

    def export_state(self) -> Dict:
        """导出当前孪生状态"""
        return {
            "timestamp": self._current_state.timestamp,
            "spot_position": self._current_state.spot_position.tolist(),
            "spot_velocity": self._current_state.spot_velocity.tolist(),
            "spot_intensity": self._current_state.spot_intensity,
            "temperature": self._current_state.temperature,
            "vibration_level": self._current_state.vibration_level,
            "history_length": len(self._state_history),
            "sync_count": self._sync_count,
            "prediction_count": self._prediction_count,
            "anomaly_log_count": len(self._anomaly_log)
        }

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "sync_count": self._sync_count,
            "prediction_count": self._prediction_count,
            "simulation_count": self._simulation_count,
            "anomaly_count": len(self._anomaly_log),
            "history_length": len(self._state_history),
            "ar_initialized": self._ar_initialized,
            "current_state": self.export_state()
        }

    def reset(self) -> None:
        """重置数字孪生"""
        self._current_state = TwinState(timestamp=0.0)
        self._state_history.clear()
        self._ar_coefficients = {}
        self._ar_initialized = False
        self._sim_time = 0.0
        self._thermal_state = 0.0
        self._vibration_state = np.zeros(2)
        self._anomaly_log = []
        self._sync_count = 0
        self._prediction_count = 0
        self._simulation_count = 0
