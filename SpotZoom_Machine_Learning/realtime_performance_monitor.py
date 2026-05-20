"""
实时性能监控器 (RealtimePerformanceMonitor)

灵感来源:
- python-control — 频域分析与控制系统性能评估
- Bluesky — 实验数据采集实时仪表盘 (NSLS-II)
- Prometheus — 时序指标采集与告警系统
- OpenTelemetry — 可观测性框架

算法原理:
- Exponential Moving Average (EMA) — 指数移动平均 (延迟平滑)
- Sliding Window Statistics — 滑动窗口统计 (均值/标准差/百分位)
- Anomaly Detection — 异常检测 (基于 Z-score 和移动标准差)
- FPS Estimation — 帧率估计 (基于帧间隔时间)

功能:
- 实时监控检测延迟、运动延迟、周期时间、FPS 等关键指标
- 基于滑动窗口的统计平滑
- 自动检测性能退化与异常
- 生成结构化性能快照和告警

依赖: numpy, logging
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.PerformanceMonitor")


class AlertSeverity(Enum):
    """告警严重程度。"""
    INFO = "info"          # 信息
    WARNING = "warning"    # 警告
    CRITICAL = "critical"  # 严重


class AlertType(Enum):
    """告警类型。"""
    HIGH_DETECTION_LATENCY = "high_detection_latency"       # 检测延迟过高
    HIGH_MOVE_LATENCY = "high_move_latency"                 # 运动延迟过高
    LOW_FPS = "low_fps"                                     # 帧率过低
    HIGH_CYCLE_TIME = "high_cycle_time"                     # 周期时间过长
    FPS_JITTER = "fps_jitter"                               # 帧率抖动过大
    LATENCY_SPIKE = "latency_spike"                         # 延迟尖峰


@dataclass
class PerformanceAlert:
    """性能告警。"""
    alert_type: AlertType       # 告警类型
    message: str                # 告警消息
    severity: AlertSeverity     # 严重程度
    timestamp: float            # 时间戳


@dataclass
class PerformanceSnapshot:
    """性能快照。"""
    avg_detection_latency_ms: float     # 平均检测延迟 (毫秒)
    avg_move_latency_ms: float          # 平均运动延迟 (毫秒)
    avg_cycle_time_ms: float            # 平均周期时间 (毫秒)
    current_fps: float                  # 当前帧率
    total_cycles: int                   # 总周期数
    uptime_s: float                     # 运行时间 (秒)
    detection_latency_p50_ms: float     # 检测延迟 P50 (毫秒)
    detection_latency_p99_ms: float     # 检测延迟 P99 (毫秒)
    move_latency_p50_ms: float          # 运动延迟 P50 (毫秒)
    move_latency_p99_ms: float          # 运动延迟 P99 (毫秒)
    fps_std: float                      # FPS 标准差
    fps_min: float                      # 最低 FPS


class RealtimePerformanceMonitor:
    """实时性能监控器。

    监控控制循环各阶段的延迟和帧率，自动检测性能异常。

    Parameters
    ----------
    window_size : int
        滑动窗口大小 (样本数)。
    detection_latency_threshold_ms : float
        检测延迟告警阈值 (毫秒)。
    move_latency_threshold_ms : float
        运动延迟告警阈值 (毫秒)。
    min_fps_threshold : float
        最低可接受帧率。
    fps_jitter_threshold : float
        FPS 抖动告警阈值 (标准差)。
    latency_spike_factor : float
        延迟尖峰检测因子 (超过 N 倍标准差视为尖峰)。
    max_alerts : int
        最大保留告警数。
    """

    def __init__(
        self,
        window_size: int = 100,
        detection_latency_threshold_ms: float = 50.0,
        move_latency_threshold_ms: float = 100.0,
        min_fps_threshold: float = 10.0,
        fps_jitter_threshold: float = 5.0,
        latency_spike_factor: float = 3.0,
        max_alerts: int = 50,
    ):
        self.window_size = int(window_size)
        self.detection_latency_threshold_ms = float(detection_latency_threshold_ms)
        self.move_latency_threshold_ms = float(move_latency_threshold_ms)
        self.min_fps_threshold = float(min_fps_threshold)
        self.fps_jitter_threshold = float(fps_jitter_threshold)
        self.latency_spike_factor = float(latency_spike_factor)
        self.max_alerts = int(max_alerts)

        # 延迟记录 (秒)
        self._detection_latencies: List[float] = []
        self._move_latencies: List[float] = []
        self._cycle_times: List[float] = []

        # FPS 追踪
        self._frame_timestamps: List[float] = []
        self._cycle_start_time: Optional[float] = None

        # 告警
        self._alerts: List[PerformanceAlert] = []

        # 统计
        self._total_cycles: int = 0
        self._start_time: float = time.time()

    def record_detection_latency(self, seconds: float) -> None:
        """记录一次检测延迟。

        Parameters
        ----------
        seconds : float
            检测耗时 (秒)。
        """
        latency_ms = float(seconds) * 1000.0
        self._detection_latencies.append(latency_ms)
        if len(self._detection_latencies) > self.window_size:
            self._detection_latencies = self._detection_latencies[-self.window_size:]

        # 检测延迟尖峰
        self._check_latency_spike(latency_ms, self._detection_latencies, AlertType.HIGH_DETECTION_LATENCY)

    def record_move_latency(self, seconds: float) -> None:
        """记录一次运动延迟。

        Parameters
        ----------
        seconds : float
            运动耗时 (秒)。
        """
        latency_ms = float(seconds) * 1000.0
        self._move_latencies.append(latency_ms)
        if len(self._move_latencies) > self.window_size:
            self._move_latencies = self._move_latencies[-self.window_size:]

        # 检测延迟尖峰
        self._check_latency_spike(latency_ms, self._move_latencies, AlertType.HIGH_MOVE_LATENCY)

    def record_cycle(self) -> None:
        """记录一个完整的控制周期。"""
        now = time.time()

        # 计算周期时间
        if self._cycle_start_time is not None:
            cycle_time_ms = (now - self._cycle_start_time) * 1000.0
            self._cycle_times.append(cycle_time_ms)
            if len(self._cycle_times) > self.window_size:
                self._cycle_times = self._cycle_times[-self.window_size:]

        self._cycle_start_time = now
        self._total_cycles += 1

        # 记录帧时间戳 (用于 FPS 计算)
        self._frame_timestamps.append(now)
        if len(self._frame_timestamps) > self.window_size:
            self._frame_timestamps = self._frame_timestamps[-self.window_size:]

        # 检查 FPS 相关告警
        self._check_fps_alerts()

    def _check_latency_spike(
        self,
        current_ms: float,
        history: List[float],
        alert_type: AlertType,
    ) -> None:
        """检查延迟是否出现尖峰异常。

        Parameters
        ----------
        current_ms : float
            当前延迟值 (毫秒)。
        history : List[float]
            延迟历史。
        alert_type : AlertType
            告警类型。
        """
        if len(history) < 5:
            return

        hist_array = np.array(history[:-1])  # 排除当前值
        mean_val = float(np.mean(hist_array))
        std_val = float(np.std(hist_array))

        if std_val < 1e-6:
            return

        z_score = abs(current_ms - mean_val) / std_val

        if z_score > self.latency_spike_factor:
            threshold = self.detection_latency_threshold_ms if alert_type == AlertType.HIGH_DETECTION_LATENCY \
                else self.move_latency_threshold_ms

            severity = AlertSeverity.CRITICAL if current_ms > threshold * 2 else AlertSeverity.WARNING
            self._add_alert(PerformanceAlert(
                alert_type=AlertType.LATENCY_SPIKE,
                message=f"延迟尖峰: {current_ms:.1f}ms (Z-score={z_score:.1f}, "
                        f"均值={mean_val:.1f}ms, 类型={alert_type.value})",
                severity=severity,
                timestamp=time.time(),
            ))

        # 检查是否持续超过阈值
        elif current_ms > (self.detection_latency_threshold_ms if alert_type == AlertType.HIGH_DETECTION_LATENCY
                           else self.move_latency_threshold_ms):
            self._add_alert(PerformanceAlert(
                alert_type=alert_type,
                message=f"{alert_type.value}: {current_ms:.1f}ms 超过阈值 "
                        f"{self.detection_latency_threshold_ms if alert_type == AlertType.HIGH_DETECTION_LATENCY else self.move_latency_threshold_ms:.1f}ms",
                severity=AlertSeverity.WARNING,
                timestamp=time.time(),
            ))

    def _check_fps_alerts(self) -> None:
        """检查 FPS 相关告警。"""
        fps = self._estimate_fps()
        if fps is None:
            return

        # 低帧率告警
        if fps < self.min_fps_threshold:
            self._add_alert(PerformanceAlert(
                alert_type=AlertType.LOW_FPS,
                message=f"帧率过低: {fps:.1f} FPS (阈值={self.min_fps_threshold:.1f})",
                severity=AlertSeverity.CRITICAL if fps < self.min_fps_threshold / 2 else AlertSeverity.WARNING,
                timestamp=time.time(),
            ))

        # FPS 抖动告警
        if len(self._frame_timestamps) >= 10:
            intervals = np.diff(self._frame_timestamps[-20:])
            if len(intervals) > 0:
                interval_fps = 1.0 / np.maximum(intervals, 1e-6)
                fps_std = float(np.std(interval_fps))
                if fps_std > self.fps_jitter_threshold:
                    self._add_alert(PerformanceAlert(
                        alert_type=AlertType.FPS_JITTER,
                        message=f"帧率抖动过大: std={fps_std:.1f} FPS (阈值={self.fps_jitter_threshold:.1f})",
                        severity=AlertSeverity.WARNING,
                        timestamp=time.time(),
                    ))

    def _estimate_fps(self) -> Optional[float]:
        """基于最近帧时间戳估计当前 FPS。

        Returns
        -------
        float or None
            估计的 FPS。如果数据不足返回 None。
        """
        if len(self._frame_timestamps) < 2:
            return None

        recent = self._frame_timestamps[-min(20, len(self._frame_timestamps)):]
        intervals = np.diff(recent)
        if len(intervals) == 0:
            return None

        # 使用中值间隔 (鲁棒估计)
        median_interval = float(np.median(intervals))
        if median_interval < 1e-6:
            return None

        return 1.0 / median_interval

    def _add_alert(self, alert: PerformanceAlert) -> None:
        """添加告警并限制数量。

        Parameters
        ----------
        alert : PerformanceAlert
            告警对象。
        """
        self._alerts.append(alert)
        if len(self._alerts) > self.max_alerts:
            self._alerts = self._alerts[-self.max_alerts:]

        LOGGER.debug("PerformanceAlert [%s]: %s", alert.severity.value, alert.message)

    def get_snapshot(self) -> PerformanceSnapshot:
        """生成当前性能快照。

        Returns
        -------
        PerformanceSnapshot
            性能快照。
        """
        # 平均延迟
        avg_det = float(np.mean(self._detection_latencies)) if self._detection_latencies else 0.0
        avg_move = float(np.mean(self._move_latencies)) if self._move_latencies else 0.0
        avg_cycle = float(np.mean(self._cycle_times)) if self._cycle_times else 0.0

        # 百分位延迟
        det_p50 = float(np.percentile(self._detection_latencies, 50)) if self._detection_latencies else 0.0
        det_p99 = float(np.percentile(self._detection_latencies, 99)) if len(self._detection_latencies) >= 10 else 0.0
        move_p50 = float(np.percentile(self._move_latencies, 50)) if self._move_latencies else 0.0
        move_p99 = float(np.percentile(self._move_latencies, 99)) if len(self._move_latencies) >= 10 else 0.0

        # FPS
        fps = self._estimate_fps() or 0.0
        fps_std = 0.0
        fps_min = 0.0
        if len(self._frame_timestamps) >= 10:
            intervals = np.diff(self._frame_timestamps[-20:])
            if len(intervals) > 0:
                interval_fps = 1.0 / np.maximum(intervals, 1e-6)
                fps_std = float(np.std(interval_fps))
                fps_min = float(np.min(interval_fps))

        uptime = time.time() - self._start_time

        return PerformanceSnapshot(
            avg_detection_latency_ms=round(avg_det, 2),
            avg_move_latency_ms=round(avg_move, 2),
            avg_cycle_time_ms=round(avg_cycle, 2),
            current_fps=round(fps, 1),
            total_cycles=self._total_cycles,
            uptime_s=round(uptime, 1),
            detection_latency_p50_ms=round(det_p50, 2),
            detection_latency_p99_ms=round(det_p99, 2),
            move_latency_p50_ms=round(move_p50, 2),
            move_latency_p99_ms=round(move_p99, 2),
            fps_std=round(fps_std, 1),
            fps_min=round(fps_min, 1),
        )

    def get_alerts(self) -> List[PerformanceAlert]:
        """获取所有告警。

        Returns
        -------
        List[PerformanceAlert]
            告警列表 (按时间排序)。
        """
        return list(self._alerts)

    def get_recent_alerts(self, count: int = 10) -> List[PerformanceAlert]:
        """获取最近 N 条告警。

        Parameters
        ----------
        count : int
            返回的告警数量。

        Returns
        -------
        List[PerformanceAlert]
            最近的告警列表。
        """
        return list(self._alerts[-count:])

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[PerformanceAlert]:
        """按严重程度筛选告警。

        Parameters
        ----------
        severity : AlertSeverity
            告警严重程度。

        Returns
        -------
        List[PerformanceAlert]
            匹配的告警列表。
        """
        return [a for a in self._alerts if a.severity == severity]

    def clear_alerts(self) -> None:
        """清除所有告警。"""
        self._alerts.clear()

    @property
    def total_cycles(self) -> int:
        """总周期数。"""
        return self._total_cycles

    @property
    def uptime(self) -> float:
        """运行时间 (秒)。"""
        return time.time() - self._start_time

    def reset(self) -> None:
        """重置监控器，清除所有数据。"""
        self._detection_latencies.clear()
        self._move_latencies.clear()
        self._cycle_times.clear()
        self._frame_timestamps.clear()
        self._alerts.clear()
        self._total_cycles = 0
        self._cycle_start_time = None
        self._start_time = time.time()
        LOGGER.info("PerformanceMonitor: 监控器已重置")
