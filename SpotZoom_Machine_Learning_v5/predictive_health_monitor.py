"""
预测性健康监控器 (PredictiveHealthMonitor)

基于 OOPAO (https://github.com/cheritier/OOPAO) 和 python-microscope
的系统健康监控理念，为 SpotZoom 提供预测性维护和异常预警能力。

灵感来源:
- OOPAO: AO 系统仿真与监控 (https://github.com/cheritier/OOPAO)
- python-microscope: 设备健康监控 (https://github.com/python-microscope/microscope)
- v1/diagnostic_health_monitor.py: 现有健康监控

算法原理:
  1. 多维健康评估: 从性能、稳定性、资源、安全四个维度评估系统健康
  2. 趋势预测: 使用指数平滑预测关键指标的未来趋势
  3. 异常预警: 基于统计过程控制的异常检测
  4. 剩余使用寿命 (RUL): 预测组件的剩余使用寿命

与现有模块的关系:
  - 增强 v1/diagnostic_health_monitor.py 的预测能力
  - 与 v1/anomaly_detector.py 的异常检测协同
  - 与 v1/safety_manager.py 的安全保护协同

外部依赖: numpy, scipy (可选)
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class HealthDimension(Enum):
    """健康评估维度。"""
    PERFORMANCE = "performance"     # 性能（定位精度、控制质量）
    STABILITY = "stability"         # 稳定性（漂移、振动）
    RESOURCE = "resource"           # 资源（CPU、内存、温度）
    SAFETY = "safety"               # 安全（限位、超时）


class HealthStatus(Enum):
    """健康状态。"""
    EXCELLENT = "excellent"         # 优秀 (>90)
    GOOD = "good"                   # 良好 (70-90)
    WARNING = "warning"             # 警告 (50-70)
    CRITICAL = "critical"           # 严重 (30-50)
    FAULT = "fault"                 # 故障 (<30)


class AlertSeverity(Enum):
    """告警严重程度。"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PredictiveHealthConfig:
    """预测性健康监控配置。"""
    # 历史数据
    history_buffer_size: int = 500   # 历史数据缓冲区大小
    prediction_window: int = 50      # 预测窗口大小

    # 性能阈值
    performance_warning: float = 0.7  # 性能警告阈值
    performance_critical: float = 0.5

    # 稳定性阈值
    stability_warning: float = 0.5   # 漂移速率 (px/s)
    stability_critical: float = 1.0

    # 资源阈值
    cpu_warning: float = 80.0        # CPU 使用率 %
    memory_warning: float = 85.0     # 内存使用率 %
    temperature_warning: float = 70.0 # 温度 °C

    # 预测参数
    trend_smoothing: float = 0.3     # 趋势平滑因子
    anomaly_sigma: float = 3.0       # 异常检测标准差倍数

    # RUL 参数
    rul_warning_days: int = 7        # RUL 预警天数
    rul_critical_days: int = 3       # RUL 严重天数


@dataclass
class HealthMetric:
    """健康指标。"""
    name: str
    value: float
    unit: str = ""
    dimension: HealthDimension = HealthDimension.PERFORMANCE
    timestamp: float = 0.0
    is_anomaly: bool = False
    trend: float = 0.0               # 趋势（正=恶化，负=改善）
    predicted_value: float = 0.0


@dataclass
class PredictiveHealthReport:
    """预测性健康报告。"""
    # 综合健康
    overall_score: float = 100.0
    overall_status: str = HealthStatus.EXCELLENT.value
    timestamp: float = 0.0

    # 维度得分
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    dimension_statuses: Dict[str, str] = field(default_factory=dict)

    # 指标详情
    metrics: List[HealthMetric] = field(default_factory=list)

    # 告警
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    # 预测
    predictions: Dict[str, float] = field(default_factory=dict)

    # RUL
    remaining_useful_life: Dict[str, int] = field(default_factory=dict)

    # 元信息
    processing_time_ms: float = 0.0


class PredictiveHealthMonitor:
    """预测性健康监控器。

    多维度系统健康评估，支持趋势预测、异常检测和 RUL 估计。

    Parameters
    ----------
    config : PredictiveHealthConfig
        监控器配置。
    """

    def __init__(self, config: Optional[PredictiveHealthConfig] = None):
        self.config = config or PredictiveHealthConfig()
        self._history: Dict[str, deque] = {}
        self._baselines: Dict[str, Tuple[float, float]] = {}  # (mean, std)

    def update_metric(
        self,
        name: str,
        value: float,
        dimension: HealthDimension = HealthDimension.PERFORMANCE,
        unit: str = "",
    ):
        """更新健康指标。"""
        if name not in self._history:
            self._history[name] = deque(maxlen=self.config.history_buffer_size)

        self._history[name].append({
            'value': value,
            'timestamp': time.time(),
            'dimension': dimension,
            'unit': unit,
        })

    def assess(self) -> PredictiveHealthReport:
        """执行健康评估。"""
        t0 = time.perf_counter()
        report = PredictiveHealthReport(timestamp=time.time())

        # 更新基线
        self._update_baselines()

        # 评估各指标
        for name, history in self._history.items():
            if len(history) < 2:
                continue

            values = np.array([h['value'] for h in history])
            dimension = history[-1]['dimension']
            unit = history[-1]['unit']

            # 趋势预测
            trend = self._compute_trend(values)
            predicted = self._predict(values)

            # 异常检测
            is_anomaly = self._detect_anomaly(name, values[-1])

            metric = HealthMetric(
                name=name,
                value=float(values[-1]),
                unit=unit,
                dimension=dimension,
                timestamp=history[-1]['timestamp'],
                is_anomaly=is_anomaly,
                trend=trend,
                predicted_value=predicted,
            )
            report.metrics.append(metric)

            # 生成告警
            if is_anomaly:
                report.alerts.append({
                    'metric': name,
                    'severity': AlertSeverity.WARNING.value,
                    'message': f"{name} 检测到异常 (值={values[-1]:.3f})",
                })

            # 趋势告警
            if trend > 0.01:  # 恶化趋势
                report.alerts.append({
                    'metric': name,
                    'severity': AlertSeverity.INFO.value,
                    'message': f"{name} 呈恶化趋势 (斜率={trend:.4f})",
                })

        # 计算维度得分
        self._compute_dimension_scores(report)

        # 计算综合得分
        if report.dimension_scores:
            report.overall_score = float(np.mean(list(report.dimension_scores.values())))

        # 确定状态
        report.overall_status = self._score_to_status(report.overall_score).value

        # RUL 估计
        self._estimate_rul(report)

        report.processing_time_ms = (time.perf_counter() - t0) * 1000
        return report

    def _update_baselines(self):
        """更新统计基线。"""
        for name, history in self._history.items():
            if len(history) < 10:
                continue
            values = np.array([h['value'] for h in history])
            self._baselines[name] = (float(np.mean(values)), float(np.std(values)))

    def _compute_trend(self, values: np.ndarray) -> float:
        """计算趋势斜率（线性回归）。"""
        if len(values) < 3:
            return 0.0

        x = np.arange(len(values), dtype=np.float64)
        # 加权线性回归（近期数据权重更大）
        weights = np.exp(-0.1 * (len(values) - 1 - x))
        w_sum = np.sum(weights)
        wx = np.sum(weights * x)
        wy = np.sum(weights * values)
        wxx = np.sum(weights * x**2)
        wxy = np.sum(weights * x * values)

        denom = w_sum * wxx - wx**2
        if abs(denom) < 1e-10:
            return 0.0

        slope = (w_sum * wxy - wx * wy) / denom
        return float(slope)

    def _predict(self, values: np.ndarray) -> float:
        """预测未来值。"""
        if len(values) < 2:
            return float(values[-1]) if len(values) > 0 else 0.0

        # 指数平滑预测
        alpha = self.config.trend_smoothing
        s = values[0]
        for v in values[1:]:
            s = alpha * v + (1 - alpha) * s

        # 线性外推
        trend = self._compute_trend(values)
        return float(s + trend * self.config.prediction_window)

    def _detect_anomaly(self, name: str, value: float) -> bool:
        """基于统计过程控制的异常检测。"""
        if name not in self._baselines:
            return False

        mean, std = self._baselines[name]
        if std < 1e-10:
            return False

        z_score = abs(value - mean) / std
        return z_score > self.config.anomaly_sigma

    def _compute_dimension_scores(self, report: PredictiveHealthReport):
        """计算各维度健康得分。"""
        dim_metrics: Dict[HealthDimension, List[HealthMetric]] = {}

        for metric in report.metrics:
            dim = metric.dimension
            if dim not in dim_metrics:
                dim_metrics[dim] = []
            dim_metrics[dim].append(metric)

        for dim, metrics in dim_metrics.items():
            scores = []
            for m in metrics:
                # 基于异常和趋势计算得分
                score = 100.0
                if m.is_anomaly:
                    score -= 30
                score -= abs(m.trend) * 100  # 趋势惩罚
                score = max(0, min(100, score))
                scores.append(score)

            avg_score = float(np.mean(scores)) if scores else 100.0
            report.dimension_scores[dim.value] = avg_score
            report.dimension_statuses[dim.value] = self._score_to_status(avg_score).value

    def _estimate_rul(self, report: PredictiveHealthReport):
        """估计剩余使用寿命。"""
        for metric in report.metrics:
            if metric.trend <= 0:
                continue  # 无恶化趋势

            # 估计达到警告阈值的时间
            threshold = 50.0  # 通用阈值
            remaining = (metric.value - threshold) / metric.trend if metric.trend > 0 else float('inf')

            if remaining < 0:
                report.remaining_useful_life[metric.name] = 0
            elif remaining == float('inf'):
                report.remaining_useful_life[metric.name] = 9999
            else:
                # 转换为"周期数"
                report.remaining_useful_life[metric.name] = int(remaining)

    @staticmethod
    def _score_to_status(score: float) -> HealthStatus:
        """将得分转换为状态。"""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 70:
            return HealthStatus.GOOD
        elif score >= 50:
            return HealthStatus.WARNING
        elif score >= 30:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAULT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = PredictiveHealthConfig()
    monitor = PredictiveHealthMonitor(config)

    # 模拟数据
    np.random.seed(42)
    for i in range(100):
        monitor.update_metric("定位精度", 0.1 + np.random.randn() * 0.02 + i * 0.0005,
                              HealthDimension.PERFORMANCE, "px")
        monitor.update_metric("漂移速率", 0.2 + np.random.randn() * 0.05,
                              HealthDimension.STABILITY, "px/s")
        monitor.update_metric("CPU使用率", 45 + np.random.randn() * 5,
                              HealthDimension.RESOURCE, "%")

    report = monitor.assess()
    print(f"综合得分: {report.overall_score:.1f} ({report.overall_status})")
    print(f"维度得分: {report.dimension_scores}")
    print(f"告警数: {len(report.alerts)}")
    for m in report.metrics:
        print(f"  {m.name}: {m.value:.3f}{m.unit}, 趋势={m.trend:.4f}, 异常={m.is_anomaly}")
