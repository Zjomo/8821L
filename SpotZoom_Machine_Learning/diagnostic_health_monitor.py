"""
诊断健康监控器 (DiagnosticHealthMonitor)

灵感来源:
- Prometheus Metrics — 云原生监控系统的指标采集与告警框架，支持多维
  时间序列数据存储、PromQL 查询语言和灵活的告警规则配置
- OpenTelemetry Health Checks — 可观测性标准的健康检查机制，通过结构化
  的健康报告暴露服务的运行状态、资源使用和依赖可用性
- python-control 频域分析 — 控制系统频域分析工具，通过 Bode 图、Nyquist
  图等评估系统的稳定性裕度和频率响应特性

算法原理:
- Exponential Moving Average (EMA) — 指数移动平均，用于平滑指标趋势
- Z-Score Anomaly Scoring — Z 分数异常评分，基于统计偏差检测异常
- Linear Trend Estimation — 线性趋势估计，通过最小二乘法拟合指标趋势
- Weighted Health Score — 加权健康评分，综合各维度指标生成总体健康度

功能:
- 持续采集系统健康指标 (检测率、收敛速度、电机响应、焦距稳定性、振动)
- 生成综合健康评分 (0-100)
- 阈值检查与告警生成
- 趋势分析与异常检测
- 诊断报告生成与建议

依赖: numpy, logging, dataclasses, time, collections (无外部框架)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.DiagnosticHealthMonitor")


class HealthDimension(Enum):
    """健康维度。"""
    DETECTION = "detection"       # 检测性能
    CONVERGENCE = "convergence"   # 收敛性能
    MOTOR = "motor"               # 电机响应
    FOCUS = "focus"               # 焦距稳定性
    VIBRATION = "vibration"       # 振动水平
    THERMAL = "thermal"           # 热稳定性


class AlertSeverity(Enum):
    """告警严重程度。"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HealthAlert:
    """健康告警。"""
    dimension: HealthDimension
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    trend: str                    # "rising", "falling", "stable"


@dataclass
class DiagnosticReport:
    """诊断报告。"""
    overall_health_score: float                                  # 总体健康评分 [0, 100]
    dimension_scores: Dict[str, float]                           # 各维度评分
    alerts: List[HealthAlert]                                    # 告警列表
    recommendations: List[str]                                   # 建议列表
    timestamp: float                                             # 报告时间戳


class MetricHistory:
    """指标历史记录。

    存储单维度指标的时间序列，提供统计计算功能。

    Parameters
    ----------
    retention : int
        最大保留样本数。
    ema_alpha : float
        EMA 平滑因子。
    """

    def __init__(self, retention: int = 500, ema_alpha: float = 0.05):
        self.retention = int(retention)
        self.ema_alpha = float(ema_alpha)

        self._values: Deque[float] = deque(maxlen=retention)
        self._timestamps: Deque[float] = deque(maxlen=retention)
        self._ema: Optional[float] = None
        self._ema_var: float = 1.0

    def record(self, value: float, timestamp: Optional[float] = None) -> None:
        """记录一个指标值。

        Parameters
        ----------
        value : float
            指标值。
        timestamp : float or None
            时间戳。
        """
        if timestamp is None:
            timestamp = time.time()

        self._values.append(float(value))
        self._timestamps.append(float(timestamp))

        # 更新 EMA
        if self._ema is None:
            self._ema = float(value)
        else:
            self._ema = self.ema_alpha * float(value) + (1.0 - self.ema_alpha) * self._ema

        # 更新 EMA 方差
        diff = float(value) - self._ema
        self._ema_var = 0.95 * self._ema_var + 0.05 * diff * diff
        self._ema_var = max(self._ema_var, 1e-12)

    @property
    def count(self) -> int:
        """已记录的样本数。"""
        return len(self._values)

    @property
    def ema(self) -> float:
        """当前 EMA 值。"""
        return self._ema if self._ema is not None else 0.0

    @property
    def ema_std(self) -> float:
        """当前 EMA 标准差。"""
        return float(np.sqrt(self._ema_var))

    def get_statistics(self) -> Dict[str, float]:
        """计算基本统计量。

        Returns
        -------
        Dict[str, float]
            包含 mean, std, min, max, ema, ema_std, trend_slope 的字典。
        """
        if len(self._values) == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'ema': 0.0, 'ema_std': 0.0, 'trend_slope': 0.0,
                'count': 0,
            }

        arr = np.array(list(self._values), dtype=np.float64)

        stats = {
            'mean': round(float(np.mean(arr)), 6),
            'std': round(float(np.std(arr)), 6),
            'min': round(float(np.min(arr)), 6),
            'max': round(float(np.max(arr)), 6),
            'ema': round(self.ema, 6),
            'ema_std': round(self.ema_std, 6),
            'count': len(self._values),
        }

        # 趋势斜率 (线性回归)
        stats['trend_slope'] = round(self._compute_trend_slope(), 6)

        return stats

    def get_window(self, window_s: float) -> List[Tuple[float, float]]:
        """获取指定时间窗口内的数据。

        Parameters
        ----------
        window_s : float
            窗口大小 (秒)。

        Returns
        -------
        List[Tuple[float, float]]
            (timestamp, value) 列表。
        """
        if not self._timestamps:
            return []

        cutoff = self._timestamps[-1] - window_s
        result = []
        for ts, val in zip(self._timestamps, self._values):
            if ts >= cutoff:
                result.append((ts, val))

        return result

    def get_zscore(self, value: float) -> float:
        """计算给定值的 Z 分数。

        Parameters
        ----------
        value : float
            待评估的值。

        Returns
        -------
        float
            Z 分数。数据不足时返回 0.0。
        """
        if len(self._values) < 5:
            return 0.0

        mean = self.ema
        std = self.ema_std

        if std < 1e-9:
            return 0.0

        return (value - mean) / std

    def _compute_trend_slope(self) -> float:
        """计算趋势斜率 (线性最小二乘)。

        Returns
        -------
        float
            趋势斜率 (单位/秒)。数据不足时返回 0.0。
        """
        if len(self._values) < 10:
            return 0.0

        timestamps = np.array(list(self._timestamps), dtype=np.float64)
        values = np.array(list(self._values), dtype=np.float64)

        # 归一化时间
        t_normalized = timestamps - timestamps[0]
        if t_normalized[-1] < 1e-9:
            return 0.0

        # 最小二乘: slope = sum((t - t_mean) * (v - v_mean)) / sum((t - t_mean)^2)
        t_mean = np.mean(t_normalized)
        v_mean = np.mean(values)

        numerator = np.sum((t_normalized - t_mean) * (values - v_mean))
        denominator = np.sum((t_normalized - t_mean) ** 2)

        if abs(denominator) < 1e-12:
            return 0.0

        slope = float(numerator / denominator)
        return slope

    def clear(self) -> None:
        """清除所有历史数据。"""
        self._values.clear()
        self._timestamps.clear()
        self._ema = None
        self._ema_var = 1.0


class DiagnosticHealthMonitor:
    """诊断健康监控器。

    持续监控系统各维度的健康指标，生成综合健康评分和诊断报告。

    Parameters
    ----------
    retention : int
        各维度指标保留样本数。
    ema_alpha : float
        EMA 平滑因子。
    alert_suppression_s : float
        同维度告警抑制时间 (秒)。
    """

    # 各维度的默认阈值和权重
    _DIMENSION_CONFIG = {
        HealthDimension.DETECTION: {
            'warning_threshold': 0.7,     # 检测置信度
            'critical_threshold': 0.4,
            'weight': 0.25,
            'higher_is_better': True,
        },
        HealthDimension.CONVERGENCE: {
            'warning_threshold': 5.0,     # 收敛时间 (秒)
            'critical_threshold': 15.0,
            'weight': 0.20,
            'higher_is_better': False,
        },
        HealthDimension.MOTOR: {
            'warning_threshold': 200.0,   # 电机响应时间 (ms)
            'critical_threshold': 500.0,
            'weight': 0.15,
            'higher_is_better': False,
        },
        HealthDimension.FOCUS: {
            'warning_threshold': 0.6,     # 焦距评分
            'critical_threshold': 0.3,
            'weight': 0.20,
            'higher_is_better': True,
        },
        HealthDimension.VIBRATION: {
            'warning_threshold': 2.0,     # 振动幅度 (像素)
            'critical_threshold': 5.0,
            'weight': 0.10,
            'higher_is_better': False,
        },
        HealthDimension.THERMAL: {
            'warning_threshold': 3.0,     # 温度漂移 (度/分钟)
            'critical_threshold': 8.0,
            'weight': 0.10,
            'higher_is_better': False,
        },
    }

    def __init__(
        self,
        retention: int = 500,
        ema_alpha: float = 0.05,
        alert_suppression_s: float = 30.0,
    ):
        self.retention = int(retention)
        self.ema_alpha = float(ema_alpha)
        self.alert_suppression_s = float(alert_suppression_s)

        # 各维度的指标历史
        self._histories: Dict[HealthDimension, MetricHistory] = {}
        for dim in HealthDimension:
            self._histories[dim] = MetricHistory(
                retention=retention, ema_alpha=ema_alpha,
            )

        # 告警记录
        self._alerts: List[HealthAlert] = []
        self._last_alert_time: Dict[HealthDimension, float] = {}
        self._max_alerts: int = 100

        # 自定义阈值覆盖
        self._custom_thresholds: Dict[HealthDimension, Dict[str, float]] = {}

        LOGGER.info(
            "DiagnosticHealthMonitor: 初始化完成 "
            "(retention=%d, alpha=%.3f, suppression=%.1fs)",
            retention, ema_alpha, alert_suppression_s,
        )

    def record_metric(
        self,
        name: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> Optional[HealthAlert]:
        """记录一个健康指标值。

        Parameters
        ----------
        name : str
            指标名称 (对应 HealthDimension.value)。
        value : float
            指标值。
        timestamp : float or None
            时间戳。

        Returns
        -------
        HealthAlert or None
            如果触发告警则返回告警对象。
        """
        dimension = self._get_dimension_by_name(name)
        if dimension is None:
            LOGGER.debug("未知健康维度: %s", name)
            return None

        self._histories[dimension].record(value, timestamp)

        # 检查阈值
        alert = self._check_dimension_threshold(dimension, value)
        return alert

    def get_health_score(self) -> float:
        """获取综合健康评分。

        Returns
        -------
        float
            健康评分 [0, 100]。
        """
        dimension_scores = self._compute_dimension_scores()

        if not dimension_scores:
            return 100.0

        # 加权平均
        total_weight = 0.0
        weighted_score = 0.0

        for dim, score in dimension_scores.items():
            config = self._DIMENSION_CONFIG[dim]
            weight = config['weight']
            weighted_score += score * weight
            total_weight += weight

        if total_weight < 1e-9:
            return 100.0

        overall = weighted_score / total_weight
        return round(max(0.0, min(100.0, overall)), 2)

    def get_diagnostic_report(self) -> DiagnosticReport:
        """生成诊断报告。

        Returns
        -------
        DiagnosticReport
            诊断报告。
        """
        dimension_scores = self._compute_dimension_scores()
        overall_score = self.get_health_score()
        alerts = self._get_recent_alerts(20)
        recommendations = self._generate_recommendations(dimension_scores, alerts)

        return DiagnosticReport(
            overall_health_score=overall_score,
            dimension_scores={
                dim.value: round(score, 2) for dim, score in dimension_scores.items()
            },
            alerts=alerts,
            recommendations=recommendations,
            timestamp=time.time(),
        )

    def check_thresholds(self) -> List[HealthAlert]:
        """检查所有维度的当前阈值。

        Returns
        -------
        List[HealthAlert]
            触发的告警列表。
        """
        new_alerts = []

        for dim in HealthDimension:
            history = self._histories[dim]
            if history.count == 0:
                continue

            current_value = history.ema
            alert = self._check_dimension_threshold(dim, current_value)
            if alert is not None:
                new_alerts.append(alert)

        return new_alerts

    def get_trend(self, metric_name: str, window_s: float = 60.0) -> Dict[str, float]:
        """获取指定指标的趋势信息。

        Parameters
        ----------
        metric_name : str
            指标名称。
        window_s : float
            时间窗口 (秒)。

        Returns
        -------
        Dict[str, float]
            趋势信息字典 (包含 slope, current, mean, std 等)。
        """
        dimension = self._get_dimension_by_name(metric_name)
        if dimension is None:
            return {}

        history = self._histories[dimension]
        window_data = history.get_window(window_s)

        if len(window_data) < 3:
            return {'current': history.ema, 'slope': 0.0, 'samples': len(window_data)}

        values = np.array([v for _, v in window_data], dtype=np.float64)
        timestamps = np.array([t for t, _ in window_data], dtype=np.float64)

        # 趋势斜率
        t_norm = timestamps - timestamps[0]
        if t_norm[-1] > 1e-9:
            t_mean = np.mean(t_norm)
            v_mean = np.mean(values)
            numerator = np.sum((t_norm - t_mean) * (values - v_mean))
            denominator = np.sum((t_norm - t_mean) ** 2)
            slope = float(numerator / denominator) if abs(denominator) > 1e-12 else 0.0
        else:
            slope = 0.0

        return {
            'current': round(history.ema, 6),
            'mean': round(float(np.mean(values)), 6),
            'std': round(float(np.std(values)), 6),
            'min': round(float(np.min(values)), 6),
            'max': round(float(np.max(values)), 6),
            'slope': round(slope, 6),
            'samples': len(window_data),
            'window_s': window_s,
        }

    def reset(self) -> None:
        """重置监控器，清除所有历史数据和告警。"""
        for dim in HealthDimension:
            self._histories[dim].clear()

        self._alerts.clear()
        self._last_alert_time.clear()

        LOGGER.info("DiagnosticHealthMonitor: 监控器已重置")

    # ======================== 内部方法 ========================

    def _get_dimension_by_name(self, name: str) -> Optional[HealthDimension]:
        """根据名称获取健康维度枚举。"""
        for dim in HealthDimension:
            if dim.value == name:
                return dim
        return None

    def _get_config(self, dimension: HealthDimension) -> Dict[str, float]:
        """获取维度配置 (支持自定义覆盖)。"""
        config = dict(self._DIMENSION_CONFIG[dimension])
        if dimension in self._custom_thresholds:
            config.update(self._custom_thresholds[dimension])
        return config

    def _compute_dimension_scores(self) -> Dict[HealthDimension, float]:
        """计算各维度的健康评分 [0, 100]。

        Returns
        -------
        Dict[HealthDimension, float]
            维度到评分的映射。
        """
        scores: Dict[HealthDimension, float] = {}

        for dim in HealthDimension:
            history = self._histories[dim]
            config = self._get_config(dim)

            if history.count == 0:
                # 无数据时给满分
                scores[dim] = 100.0
                continue

            current = history.ema
            warn_thresh = config['warning_threshold']
            crit_thresh = config['critical_threshold']
            higher_is_better = config['higher_is_better']

            # 计算评分
            if higher_is_better:
                # 越高越好: 评分 = (current - critical) / (warning - critical) * 100
                if warn_thresh > crit_thresh:
                    score = (current - crit_thresh) / (warn_thresh - crit_thresh) * 100.0
                else:
                    score = 100.0
            else:
                # 越低越好: 评分 = (critical - current) / (critical - warning) * 100
                if crit_thresh > warn_thresh:
                    score = (crit_thresh - current) / (crit_thresh - warn_thresh) * 100.0
                else:
                    score = 100.0

            scores[dim] = max(0.0, min(100.0, score))

        return scores

    def _check_dimension_threshold(
        self,
        dimension: HealthDimension,
        value: float,
    ) -> Optional[HealthAlert]:
        """检查单维度阈值。

        Parameters
        ----------
        dimension : HealthDimension
            健康维度。
        value : float
            当前值。

        Returns
        -------
        HealthAlert or None
            触发的告警。
        """
        config = self._get_config(dimension)
        history = self._histories[dimension]

        # 告警抑制
        now = time.time()
        last_alert = self._last_alert_time.get(dimension, 0.0)
        if now - last_alert < self.alert_suppression_s:
            return None

        warn_thresh = config['warning_threshold']
        crit_thresh = config['critical_threshold']
        higher_is_better = config['higher_is_better']

        severity = None
        if higher_is_better:
            if value < crit_thresh:
                severity = AlertSeverity.CRITICAL
            elif value < warn_thresh:
                severity = AlertSeverity.WARNING
        else:
            if value > crit_thresh:
                severity = AlertSeverity.CRITICAL
            elif value > warn_thresh:
                severity = AlertSeverity.WARNING

        if severity is None:
            return None

        # 确定趋势
        stats = history.get_statistics()
        slope = stats.get('trend_slope', 0.0)
        if higher_is_better:
            if slope < -0.01:
                trend = "falling"
            elif slope > 0.01:
                trend = "rising"
            else:
                trend = "stable"
        else:
            if slope > 0.01:
                trend = "rising"
            elif slope < -0.01:
                trend = "falling"
            else:
                trend = "stable"

        threshold_used = crit_thresh if severity == AlertSeverity.CRITICAL else warn_thresh

        alert = HealthAlert(
            dimension=dimension,
            severity=severity,
            message=f"{dimension.value} 指标异常: {value:.3f} "
                    f"(阈值={threshold_used:.3f}, 趋势={trend})",
            value=value,
            threshold=threshold_used,
            trend=trend,
        )

        self._last_alert_time[dimension] = now
        self._alerts.append(alert)

        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]

        LOGGER.warning(
            "DiagnosticHealthMonitor: [%s] %s",
            severity.value, alert.message,
        )

        return alert

    def _get_recent_alerts(self, count: int = 20) -> List[HealthAlert]:
        """获取最近的告警。

        Parameters
        ----------
        count : int
            返回数量。

        Returns
        -------
        List[HealthAlert]
            最近的告警列表。
        """
        return list(self._alerts[-count:])

    def _generate_recommendations(
        self,
        dimension_scores: Dict[HealthDimension, float],
        alerts: List[HealthAlert],
    ) -> List[str]:
        """生成诊断建议。

        Parameters
        ----------
        dimension_scores : Dict[HealthDimension, float]
            各维度评分。
        alerts : List[HealthAlert]
            告警列表。

        Returns
        -------
        List[str]
            建议列表。
        """
        recommendations = []

        # 基于低评分维度生成建议
        for dim, score in dimension_scores.items():
            if score < 30:
                if dim == HealthDimension.DETECTION:
                    recommendations.append(
                        "检测性能严重不足: 检查光源亮度、相机曝光和光斑检测参数"
                    )
                elif dim == HealthDimension.CONVERGENCE:
                    recommendations.append(
                        "收敛速度过慢: 考虑增大 PID 增益或检查机械响应"
                    )
                elif dim == HealthDimension.MOTOR:
                    recommendations.append(
                        "电机响应异常: 检查电机连接、步进驱动和机械负载"
                    )
                elif dim == HealthDimension.FOCUS:
                    recommendations.append(
                        "焦距稳定性差: 检查自动对焦系统和温度漂移"
                    )
                elif dim == HealthDimension.VIBRATION:
                    recommendations.append(
                        "振动水平过高: 检查隔振平台和环境振动源"
                    )
                elif dim == HealthDimension.THERMAL:
                    recommendations.append(
                        "热漂移过大: 等待系统热平衡或启用温度补偿"
                    )
            elif score < 60:
                if dim == HealthDimension.DETECTION:
                    recommendations.append(
                        "检测性能下降: 建议检查图像质量和检测阈值"
                    )
                elif dim == HealthDimension.VIBRATION:
                    recommendations.append(
                        "振动略有增加: 监控趋势，必要时启用振动补偿"
                    )

        # 基于告警趋势生成建议
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if len(critical_alerts) >= 3:
            recommendations.append(
                "多个维度出现严重告警: 建议暂停对准，进行系统全面检查"
            )

        rising_alerts = [a for a in alerts if a.trend == "rising" and a.severity == AlertSeverity.WARNING]
        if len(rising_alerts) >= 2:
            recommendations.append(
                "多个指标呈恶化趋势: 建议预防性维护"
            )

        if not recommendations:
            recommendations.append("系统运行正常，各维度指标健康")

        return recommendations


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    monitor = DiagnosticHealthMonitor(
        retention=200,
        ema_alpha=0.1,
        alert_suppression_s=5.0,
    )

    print("=== 诊断健康监控器测试 ===\n")

    np.random.seed(42)

    # 模拟正常运行
    print("--- 阶段 1: 正常运行 ---")
    for i in range(30):
        t = i * 0.1
        monitor.record_metric("detection", 0.9 + np.random.normal(0, 0.03), t)
        monitor.record_metric("convergence", 2.0 + np.random.normal(0, 0.3), t)
        monitor.record_metric("motor", 100.0 + np.random.normal(0, 10), t)
        monitor.record_metric("focus", 0.85 + np.random.normal(0, 0.02), t)
        monitor.record_metric("vibration", 0.5 + np.random.normal(0, 0.1), t)
        monitor.record_metric("thermal", 1.0 + np.random.normal(0, 0.2), t)

    report = monitor.get_diagnostic_report()
    print(f"  健康评分: {report.overall_health_score:.1f}/100")
    print(f"  维度评分: {report.dimension_scores}")
    print(f"  告警数: {len(report.alerts)}")
    print(f"  建议: {report.recommendations[0]}")

    # 模拟性能下降
    print("\n--- 阶段 2: 性能下降 ---")
    for i in range(30):
        t = 3.0 + i * 0.1
        # 检测性能逐渐下降
        det = max(0.2, 0.9 - i * 0.02 + np.random.normal(0, 0.03))
        monitor.record_metric("detection", det, t)
        monitor.record_metric("convergence", 2.0 + i * 0.3 + np.random.normal(0, 0.3), t)
        monitor.record_metric("motor", 100.0 + np.random.normal(0, 10), t)
        monitor.record_metric("focus", 0.85 + np.random.normal(0, 0.02), t)
        monitor.record_metric("vibration", 0.5 + np.random.normal(0, 0.1), t)
        monitor.record_metric("thermal", 1.0 + np.random.normal(0, 0.2), t)

    report = monitor.get_diagnostic_report()
    print(f"  健康评分: {report.overall_health_score:.1f}/100")
    print(f"  维度评分: {report.dimension_scores}")
    print(f"  告警数: {len(report.alerts)}")
    for alert in report.alerts[-3:]:
        print(f"    [{alert.severity.value}] {alert.message}")
    for rec in report.recommendations:
        print(f"  建议: {rec}")

    # 趋势分析
    print("\n--- 趋势分析 ---")
    trend = monitor.get_trend("detection", window_s=10.0)
    print(f"  检测性能趋势: {trend}")

    trend = monitor.get_trend("convergence", window_s=10.0)
    print(f"  收敛时间趋势: {trend}")

    # 阈值检查
    print("\n--- 阈值检查 ---")
    alerts = monitor.check_thresholds()
    print(f"  触发告警数: {len(alerts)}")
    for alert in alerts:
        print(f"    [{alert.severity.value}] {alert.message}")

    print("\n测试完成")
