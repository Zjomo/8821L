"""
实时诊断仪表盘 (RealtimeDiagnosticsDashboard)

灵感来源:
- Bluesky RunEngine (https://github.com/bluesky/bluesky) — 实验运行监控引擎
- Prometheus — 指标收集与聚合系统
- Grafana — 实时监控仪表盘

算法原理:
- Circular Buffer — 固定大小环形缓冲区高效存储时序数据
- Z-Score Anomaly Detection — 基于标准差的异常检测
- IQR (Interquartile Range) — 四分位距异常检测
- Exponential Moving Average — 指数移动平均趋势分析

功能:
- 实时指标收集与聚合 (CPU、内存、延迟、精度)
- 环形缓冲区高效存储时序数据
- 多种异常检测方法 (Z-Score, IQR)
- 趋势分析与摘要报告生成
- 告警阈值管理

依赖: numpy, scipy (无外部深度学习框架依赖)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.RealtimeDiagnosticsDashboard")


class _CircularBuffer:
    """固定大小环形缓冲区。"""

    def __init__(self, capacity: int):
        self._capacity = int(capacity)
        self._buffer: List[float] = [0.0] * self._capacity
        self._head: int = 0
        self._count: int = 0

    def push(self, value: float):
        """写入一个值。"""
        self._buffer[self._head] = float(value)
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    def get_all(self) -> np.ndarray:
        """获取所有有效数据 (按时间顺序)。"""
        if self._count < self._capacity:
            return np.array(self._buffer[:self._count], dtype=np.float64)
        else:
            return np.array(
                self._buffer[self._head:] + self._buffer[:self._head],
                dtype=np.float64,
            )

    def get_latest(self, n: int = 1) -> np.ndarray:
        """获取最近 n 个值。"""
        n = min(n, self._count)
        if n == 0:
            return np.array([], dtype=np.float64)
        if self._count < self._capacity:
            return np.array(self._buffer[self._count - n:self._count], dtype=np.float64)
        else:
            start = (self._head - n) % self._capacity
            if start + n <= self._capacity:
                return np.array(self._buffer[start:start + n], dtype=np.float64)
            else:
                return np.array(
                    self._buffer[start:] + self._buffer[:n - (self._capacity - start)],
                    dtype=np.float64,
                )

    @property
    def count(self) -> int:
        """当前有效数据量。"""
        return self._count

    @property
    def capacity(self) -> int:
        """缓冲区容量。"""
        return self._capacity

    def clear(self):
        """清空缓冲区。"""
        self._buffer = [0.0] * self._capacity
        self._head = 0
        self._count = 0

    def mean(self) -> float:
        """计算均值。"""
        if self._count == 0:
            return 0.0
        return float(np.mean(self.get_all()))

    def std(self) -> float:
        """计算标准差。"""
        if self._count < 2:
            return 0.0
        return float(np.std(self.get_all()))

    def min(self) -> float:
        """计算最小值。"""
        if self._count == 0:
            return 0.0
        return float(np.min(self.get_all()))

    def max(self) -> float:
        """计算最大值。"""
        if self._count == 0:
            return 0.0
        return float(np.max(self.get_all()))

    def percentile(self, q: float) -> float:
        """计算百分位数。"""
        if self._count == 0:
            return 0.0
        return float(np.percentile(self.get_all(), q))


@dataclass
class DashboardConfig:
    """仪表盘配置参数。"""
    # --- 缓冲区 ---
    buffer_capacity: int = 1000  # 环形缓冲区容量

    # --- 监控指标 ---
    metric_names: List[str] = field(default_factory=lambda: [
        "cpu_usage", "memory_usage", "latency_ms",
        "accuracy", "fps", "wavefront_rms",
    ])

    # --- 异常检测 ---
    anomaly_method: str = "zscore"  # "zscore", "iqr", "both"
    zscore_threshold: float = 3.0  # Z-Score 阈值
    iqr_multiplier: float = 1.5  # IQR 倍数
    anomaly_window: int = 50  # 异常检测窗口大小

    # --- 告警 ---
    alert_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # 例如: {"cpu_usage": (0, 90), "latency_ms": (0, 50)}

    # --- 趋势分析 ---
    trend_window: int = 100  # 趋势分析窗口大小
    ema_alpha: float = 0.1  # 指数移动平均平滑因子

    # --- 报告 ---
    report_interval: int = 100  # 自动报告间隔 (采样数)


@dataclass
class AnomalyEvent:
    """异常事件。"""
    metric_name: str
    value: float
    timestamp: float
    severity: str  # "low", "medium", "high"
    method: str  # "zscore", "iqr"
    details: str = ""


@dataclass
class DiagnosticsReport:
    """诊断报告。"""
    timestamp: float = 0.0
    metrics_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    anomalies: List[AnomalyEvent] = field(default_factory=list)
    trends: Dict[str, str] = field(default_factory=dict)  # "increasing", "decreasing", "stable"
    health_score: float = 1.0  # 0~1 健康评分
    total_samples: int = 0
    alerts: List[str] = field(default_factory=list)


class RealtimeDiagnosticsDashboard:
    """实时诊断仪表盘。

    收集、聚合和分析实时系统指标，
    提供异常检测、趋势分析和健康报告。

    Parameters
    ----------
    config : DashboardConfig, optional
        仪表盘配置参数。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[DashboardConfig] = None):
        self._cfg = config if config is not None else DashboardConfig()
        self._buffers: Dict[str, _CircularBuffer] = {}
        self._anomalies: List[AnomalyEvent] = []
        self._total_samples: int = 0
        self._start_time: float = time.time()
        self._last_report_time: float = 0.0

        # 初始化各指标的缓冲区
        for name in self._cfg.metric_names:
            self._buffers[name] = _CircularBuffer(self._cfg.buffer_capacity)

    def record(self, metric_name: str, value: float):
        """记录一个指标值。

        Parameters
        ----------
        metric_name : str
            指标名称。
        value : float
            指标值。
        """
        if metric_name not in self._buffers:
            self._buffers[metric_name] = _CircularBuffer(self._cfg.buffer_capacity)

        self._buffers[metric_name].push(value)
        self._total_samples += 1

        # 实时异常检测
        anomaly = self._check_anomaly(metric_name, value)
        if anomaly is not None:
            self._anomalies.append(anomaly)
            # 限制异常记录数量
            if len(self._anomalies) > self._cfg.buffer_capacity:
                self._anomalies = self._anomalies[-self._cfg.buffer_capacity:]

    def record_batch(self, metrics: Dict[str, float]):
        """批量记录多个指标值。

        Parameters
        ----------
        metrics : Dict[str, float]
            指标名称到值的映射。
        """
        for name, value in metrics.items():
            self.record(name, value)

    def _check_anomaly(
        self, metric_name: str, value: float
    ) -> Optional[AnomalyEvent]:
        """检查单个值是否异常。

        Parameters
        ----------
        metric_name : str
            指标名称。
        value : float
            指标值。

        Returns
        -------
        AnomalyEvent or None
            如果检测到异常则返回事件，否则返回 None。
        """
        buffer = self._buffers.get(metric_name)
        if buffer is None or buffer.count < self._cfg.anomaly_window:
            return None

        window_data = buffer.get_latest(self._cfg.anomaly_window)
        if len(window_data) == 0:
            return None

        method = self._cfg.anomaly_method
        is_anomaly = False
        used_method = ""

        if method in ("zscore", "both"):
            mean = np.mean(window_data)
            std = np.std(window_data) + 1e-12
            z_score = abs(value - mean) / std
            if z_score > self._cfg.zscore_threshold:
                is_anomaly = True
                used_method = "zscore"

        if not is_anomaly and method in ("iqr", "both"):
            q1 = np.percentile(window_data, 25)
            q3 = np.percentile(window_data, 75)
            iqr = q3 - q1 + 1e-12
            lower = q1 - self._cfg.iqr_multiplier * iqr
            upper = q3 + self._cfg.iqr_multiplier * iqr
            if value < lower or value > upper:
                is_anomaly = True
                used_method = "iqr"

        if is_anomaly:
            # 判断严重程度
            mean = np.mean(window_data)
            std = np.std(window_data) + 1e-12
            deviation = abs(value - mean) / std

            if deviation > 5.0:
                severity = "high"
            elif deviation > 3.0:
                severity = "medium"
            else:
                severity = "low"

            return AnomalyEvent(
                metric_name=metric_name,
                value=value,
                timestamp=time.time(),
                severity=severity,
                method=used_method,
                details=f"值={value:.4f}, 均值={mean:.4f}, 标准差={std:.4f}",
            )

        return None

    def _compute_trend(
        self, metric_name: str
    ) -> str:
        """计算指标趋势。

        Parameters
        ----------
        metric_name : str
            指标名称。

        Returns
        -------
        str
            趋势: "increasing", "decreasing", "stable"
        """
        buffer = self._buffers.get(metric_name)
        if buffer is None or buffer.count < self._cfg.trend_window:
            return "stable"

        data = buffer.get_latest(self._cfg.trend_window)
        if len(data) < 2:
            return "stable"

        # 使用线性回归斜率判断趋势
        x = np.arange(len(data), dtype=np.float64)
        y = data

        # 最小二乘拟合
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return "stable"

        # 归一化斜率
        mean_y = np.mean(y) + 1e-12
        normalized_slope = slope / mean_y

        if normalized_slope > 0.01:
            return "increasing"
        elif normalized_slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _compute_ema(
        self, data: np.ndarray, alpha: float
    ) -> np.ndarray:
        """计算指数移动平均。

        Parameters
        ----------
        data : np.ndarray
            输入数据。
        alpha : float
            平滑因子。

        Returns
        -------
        np.ndarray
            EMA 结果。
        """
        ema = np.zeros_like(data, dtype=np.float64)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _compute_health_score(self) -> float:
        """计算系统健康评分。

        Returns
        -------
        float
            健康评分 [0, 1]。
        """
        if not self._buffers:
            return 1.0

        scores = []

        for name, buffer in self._buffers.items():
            if buffer.count < 10:
                continue

            data = buffer.get_all()

            # 检查告警阈值
            if name in self._cfg.alert_thresholds:
                lo, hi = self._cfg.alert_thresholds[name]
                latest = buffer.get_latest(1)[0] if buffer.count > 0 else 0
                if latest > hi or latest < lo:
                    scores.append(0.3)
                    continue

            # 基于变异系数评分
            mean = np.mean(data) + 1e-12
            cv = np.std(data) / mean
            stability_score = max(0, 1.0 - cv)
            scores.append(stability_score)

        if not scores:
            return 1.0

        return float(np.mean(scores))

    def generate_report(self) -> DiagnosticsReport:
        """生成诊断报告。

        Returns
        -------
        DiagnosticsReport
            诊断报告。
        """
        metrics_summary: Dict[str, Dict[str, float]] = {}
        trends: Dict[str, str] = {}
        alerts: List[str] = []

        for name, buffer in self._buffers.items():
            if buffer.count == 0:
                continue

            data = buffer.get_all()
            latest = buffer.get_latest(1)
            latest_val = float(latest[0]) if len(latest) > 0 else 0.0

            summary = {
                "current": latest_val,
                "mean": buffer.mean(),
                "std": buffer.std(),
                "min": buffer.min(),
                "max": buffer.max(),
                "p25": buffer.percentile(25),
                "p50": buffer.percentile(50),
                "p75": buffer.percentile(75),
                "p95": buffer.percentile(95),
                "count": float(buffer.count),
            }
            metrics_summary[name] = summary

            # 趋势
            trends[name] = self._compute_trend(name)

            # 告警检查
            if name in self._cfg.alert_thresholds:
                lo, hi = self._cfg.alert_thresholds[name]
                if latest_val > hi:
                    alerts.append(
                        f"[告警] {name}={latest_val:.2f} 超过上限 {hi:.2f}"
                    )
                elif latest_val < lo:
                    alerts.append(
                        f"[告警] {name}={latest_val:.2f} 低于下限 {lo:.2f}"
                    )

        # 健康评分
        health = self._compute_health_score()

        # 最近的异常
        recent_anomalies = [
            a for a in self._anomalies
            if time.time() - a.timestamp < 60.0  # 最近60秒
        ]

        report = DiagnosticsReport(
            timestamp=time.time(),
            metrics_summary=metrics_summary,
            anomalies=recent_anomalies,
            trends=trends,
            health_score=health,
            total_samples=self._total_samples,
            alerts=alerts,
        )

        LOGGER.info(
            "诊断报告: 健康评分=%.2f, 指标数=%d, 异常数=%d, 告警数=%d",
            health, len(metrics_summary), len(recent_anomalies), len(alerts),
        )

        return report

    def get_metric_history(
        self, metric_name: str, n: Optional[int] = None
    ) -> np.ndarray:
        """获取指标历史数据。

        Parameters
        ----------
        metric_name : str
            指标名称。
        n : int, optional
            返回最近 n 个值。为 None 时返回全部。

        Returns
        -------
        np.ndarray
            指标值数组。
        """
        buffer = self._buffers.get(metric_name)
        if buffer is None:
            return np.array([], dtype=np.float64)

        if n is not None:
            return buffer.get_latest(n)
        return buffer.get_all()

    def get_anomalies(
        self, since: Optional[float] = None
    ) -> List[AnomalyEvent]:
        """获取异常事件。

        Parameters
        ----------
        since : float, optional
            起始时间戳。为 None 时返回全部。

        Returns
        -------
        List[AnomalyEvent]
            异常事件列表。
        """
        if since is None:
            return list(self._anomalies)
        return [a for a in self._anomalies if a.timestamp >= since]

    def set_alert_threshold(
        self, metric_name: str, low: float, high: float
    ):
        """设置告警阈值。

        Parameters
        ----------
        metric_name : str
            指标名称。
        low : float
            下限。
        high : float
            上限。
        """
        self._cfg.alert_thresholds[metric_name] = (low, high)
        LOGGER.info("设置告警阈值: %s = [%.2f, %.2f]", metric_name, low, high)

    def get_summary_text(self) -> str:
        """获取文本摘要。

        Returns
        -------
        str
            文本格式的诊断摘要。
        """
        report = self.generate_report()
        lines = [
            "=" * 50,
            "实时诊断摘要",
            "=" * 50,
            f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"总采样数: {report.total_samples}",
            f"健康评分: {report.health_score:.2f}",
            "",
            "--- 指标概览 ---",
        ]

        for name, summary in report.metrics_summary.items():
            trend = report.trends.get(name, "stable")
            trend_symbol = {"increasing": "↑", "decreasing": "↓", "stable": "→"}.get(trend, "?")
            lines.append(
                f"  {name}: 当前={summary['current']:.2f}, "
                f"均值={summary['mean']:.2f}±{summary['std']:.2f} "
                f"{trend_symbol}"
            )

        if report.alerts:
            lines.append("")
            lines.append("--- 告警 ---")
            for alert in report.alerts:
                lines.append(f"  {alert}")

        if report.anomalies:
            lines.append("")
            lines.append(f"--- 异常 ({len(report.anomalies)} 个) ---")
            for a in report.anomalies[-5:]:  # 最近5个
                lines.append(
                    f"  [{a.severity}] {a.metric_name}={a.value:.4f} ({a.method})"
                )

        lines.append("=" * 50)
        return "\n".join(lines)

    def reset(self):
        """重置仪表盘状态。"""
        for buffer in self._buffers.values():
            buffer.clear()
        self._anomalies.clear()
        self._total_samples = 0
        self._start_time = time.time()
        self._last_report_time = 0.0
        LOGGER.info("实时诊断仪表盘已重置")
