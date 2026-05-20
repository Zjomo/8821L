"""
智能异常检测器 (IntelligentAnomalyDetector)

灵感来源:
- Prometheus AlertManager — 时序指标异常检测与告警
- OpenTelemetry — 可观测性框架 (Logs/Metrics/Traces)
- Elastic APM — 应用性能监控与异常检测
- DeepSeek 推理系统监控 — 实时性能异常检测

算法原理:
- Statistical Process Control (SPC) — 统计过程控制
- Z-Score Anomaly Detection — Z分数异常检测
- Exponential Weighted Moving Average (EWMA) — 指数加权移动平均
- Change Point Detection — 变点检测

功能:
- 实时检测控制循环中的异常行为
- 支持多种异常类型检测 (延迟尖峰、精度下降、系统卡顿等)
- 提供告警分级和自动抑制
- 生成异常诊断报告

依赖: numpy, logging, dataclasses
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np

LOGGER = logging.getLogger("SpotZoom.AnomalyDetector")


class AnomalyType(Enum):
    """异常类型。"""
    DETECTION_LATENCY_SPIKE = "detection_latency_spike"       # 检测延迟尖峰
    MOVE_LATENCY_SPIKE = "move_latency_spike"                 # 运动延迟尖峰
    LOW_DETECTION_CONFIDENCE = "low_detection_confidence"     # 检测置信度过低
    ALIGNMENT_DIVERGENCE = "alignment_divergence"             # 对准发散
    EXCESSIVE_OSCILLATION = "excessive_oscillation"           # 过度振荡
    DEADLOCK_DETECTED = "deadlock_detected"                   # 死锁检测
    MEMORY_LEAK_SUSPECTED = "memory_leak_suspected"           # 内存泄漏嫌疑
    FPS_DROP = "fps_drop"                                     # 帧率骤降
    POSITION_DRIFT = "position_drift"                         # 位置漂移
    CONTROL_LOOP_TIMEOUT = "control_loop_timeout"             # 控制循环超时


class AnomalySeverity(Enum):
    """异常严重程度。"""
    INFO = "info"           # 信息级别
    WARNING = "warning"     # 警告
    ERROR = "error"         # 错误
    CRITICAL = "critical"   # 严重


@dataclass
class AnomalyEvent:
    """异常事件。"""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    timestamp: float
    value: float
    threshold: float
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    is_suppressed: bool = False


@dataclass
class AnomalyStatistics:
    """异常统计。"""
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    suppression_count: int
    last_event_time: Optional[float]
    mean_inter_event_interval_s: float


class IntelligentAnomalyDetector:
    """智能异常检测器。

    实时监控控制循环指标，检测异常行为并生成告警。

    Parameters
    ----------
    window_size : int
        滑动窗口大小 (样本数)。
    z_score_threshold : float
        Z分数异常检测阈值。
    ewma_alpha : float
        EWMA 平滑因子。
    suppression_window_s : float
        告警抑制窗口 (秒)。
    max_events : int
        最大保留事件数。
    """

    def __init__(
        self,
        window_size: int = 100,
        z_score_threshold: float = 3.0,
        ewma_alpha: float = 0.1,
        suppression_window_s: float = 60.0,
        max_events: int = 200,
    ):
        self.window_size = int(window_size)
        self.z_score_threshold = float(z_score_threshold)
        self.ewma_alpha = float(ewma_alpha)
        self.suppression_window_s = float(suppression_window_s)
        self.max_events = int(max_events)

        # 指标历史
        self._detection_latencies: List[float] = []
        self._move_latencies: List[float] = []
        self._confidences: List[float] = []
        self._position_errors: List[Tuple[float, float]] = []
        self._fps_history: List[float] = []

        # EWMA 状态
        self._ewma_detection: Optional[float] = None
        self._ewma_move: Optional[float] = None
        self._ewma_fps: Optional[float] = None

        # 异常事件
        self._events: List[AnomalyEvent] = []

        # 告警抑制
        self._last_alert_time: Dict[AnomalyType, float] = {}

        # 阈值配置
        self._thresholds = {
            AnomalyType.DETECTION_LATENCY_SPIKE: 50.0,      # ms
            AnomalyType.MOVE_LATENCY_SPIKE: 100.0,          # ms
            AnomalyType.LOW_DETECTION_CONFIDENCE: 0.3,      # confidence
            AnomalyType.ALIGNMENT_DIVERGENCE: 50.0,         # px error
            AnomalyType.EXCESSIVE_OSCILLATION: 5,           # direction changes
            AnomalyType.FPS_DROP: 10.0,                     # FPS
            AnomalyType.POSITION_DRIFT: 100.0,              # px
            AnomalyType.CONTROL_LOOP_TIMEOUT: 1000.0,       # ms
        }

    def record_detection_latency(self, latency_ms: float) -> Optional[AnomalyEvent]:
        """记录检测延迟并检测异常。

        Parameters
        ----------
        latency_ms : float
            检测延迟 (毫秒)。

        Returns
        -------
        AnomalyEvent or None
            检测到的异常事件，无异常返回 None。
        """
        self._detection_latencies.append(latency_ms)
        if len(self._detection_latencies) > self.window_size:
            self._detection_latencies = self._detection_latencies[-self.window_size:]

        # 更新 EWMA
        if self._ewma_detection is None:
            self._ewma_detection = latency_ms
        else:
            self._ewma_detection = self.ewma_alpha * latency_ms + (1 - self.ewma_alpha) * self._ewma_detection

        # Z-score 异常检测
        event = self._detect_zscore_anomaly(
            latency_ms,
            self._detection_latencies,
            AnomalyType.DETECTION_LATENCY_SPIKE,
            self._thresholds[AnomalyType.DETECTION_LATENCY_SPIKE],
            "检测延迟",
            "ms",
        )

        return event

    def record_move_latency(self, latency_ms: float) -> Optional[AnomalyEvent]:
        """记录运动延迟并检测异常。"""
        self._move_latencies.append(latency_ms)
        if len(self._move_latencies) > self.window_size:
            self._move_latencies = self._move_latencies[-self.window_size:]

        if self._ewma_move is None:
            self._ewma_move = latency_ms
        else:
            self._ewma_move = self.ewma_alpha * latency_ms + (1 - self.ewma_alpha) * self._ewma_move

        event = self._detect_zscore_anomaly(
            latency_ms,
            self._move_latencies,
            AnomalyType.MOVE_LATENCY_SPIKE,
            self._thresholds[AnomalyType.MOVE_LATENCY_SPIKE],
            "运动延迟",
            "ms",
        )

        return event

    def record_detection_confidence(self, confidence: float) -> Optional[AnomalyEvent]:
        """记录检测置信度并检测异常。"""
        self._confidences.append(confidence)
        if len(self._confidences) > self.window_size:
            self._confidences = self._confidences[-self.window_size:]

        # 低置信度检测
        threshold = self._thresholds[AnomalyType.LOW_DETECTION_CONFIDENCE]
        if confidence < threshold:
            event = AnomalyEvent(
                anomaly_type=AnomalyType.LOW_DETECTION_CONFIDENCE,
                severity=AnomalySeverity.WARNING if confidence > threshold * 0.5 else AnomalySeverity.ERROR,
                timestamp=time.time(),
                value=confidence,
                threshold=threshold,
                message=f"检测置信度过低: {confidence:.3f} < {threshold:.3f}",
                context={"confidence": confidence},
            )
            return self._add_event(event)

        return None

    def record_position_error(self, error_x: float, error_y: float) -> Optional[AnomalyEvent]:
        """记录位置误差并检测异常。"""
        self._position_errors.append((error_x, error_y))
        if len(self._position_errors) > self.window_size:
            self._position_errors = self._position_errors[-self.window_size:]

        error_mag = np.sqrt(error_x ** 2 + error_y ** 2)

        # 对准发散检测
        threshold = self._thresholds[AnomalyType.ALIGNMENT_DIVERGENCE]
        if error_mag > threshold:
            event = AnomalyEvent(
                anomaly_type=AnomalyType.ALIGNMENT_DIVERGENCE,
                severity=AnomalySeverity.ERROR,
                timestamp=time.time(),
                value=error_mag,
                threshold=threshold,
                message=f"对准发散: 误差 {error_mag:.1f}px > {threshold:.1f}px",
                context={"error_x": error_x, "error_y": error_y},
            )
            return self._add_event(event)

        # 过度振荡检测
        if len(self._position_errors) >= 10:
            oscillation = self._detect_oscillation()
            if oscillation > self._thresholds[AnomalyType.EXCESSIVE_OSCILLATION]:
                event = AnomalyEvent(
                    anomaly_type=AnomalyType.EXCESSIVE_OSCILLATION,
                    severity=AnomalySeverity.WARNING,
                    timestamp=time.time(),
                    value=oscillation,
                    threshold=self._thresholds[AnomalyType.EXCESSIVE_OSCILLATION],
                    message=f"过度振荡: 方向变化 {oscillation} 次",
                    context={"recent_errors": self._position_errors[-10:]},
                )
                return self._add_event(event)

        return None

    def record_fps(self, fps: float) -> Optional[AnomalyEvent]:
        """记录帧率并检测异常。"""
        self._fps_history.append(fps)
        if len(self._fps_history) > self.window_size:
            self._fps_history = self._fps_history[-self.window_size:]

        if self._ewma_fps is None:
            self._ewma_fps = fps
        else:
            self._ewma_fps = self.ewma_alpha * fps + (1 - self.ewma_alpha) * self._ewma_fps

        # 帧率骤降检测
        threshold = self._thresholds[AnomalyType.FPS_DROP]
        if fps < threshold:
            event = AnomalyEvent(
                anomaly_type=AnomalyType.FPS_DROP,
                severity=AnomalySeverity.CRITICAL if fps < threshold / 2 else AnomalySeverity.WARNING,
                timestamp=time.time(),
                value=fps,
                threshold=threshold,
                message=f"帧率骤降: {fps:.1f} FPS < {threshold:.1f} FPS",
                context={"ewma_fps": self._ewma_fps},
            )
            return self._add_event(event)

        return None

    def _detect_zscore_anomaly(
        self,
        value: float,
        history: List[float],
        anomaly_type: AnomalyType,
        threshold: float,
        name: str,
        unit: str,
    ) -> Optional[AnomalyEvent]:
        """使用 Z-score 检测异常。"""
        if len(history) < 10:
            return None

        hist_array = np.array(history[:-1])  # 排除当前值
        mean_val = float(np.mean(hist_array))
        std_val = float(np.std(hist_array))

        if std_val < 1e-6:
            return None

        z_score = abs(value - mean_val) / std_val

        if z_score > self.z_score_threshold:
            severity = AnomalySeverity.WARNING
            if value > threshold * 2:
                severity = AnomalySeverity.ERROR
            if value > threshold * 3:
                severity = AnomalySeverity.CRITICAL

            event = AnomalyEvent(
                anomaly_type=anomaly_type,
                severity=severity,
                timestamp=time.time(),
                value=value,
                threshold=threshold,
                message=f"{name}异常: {value:.1f}{unit} (Z={z_score:.1f}, 均值={mean_val:.1f}{unit})",
                context={"z_score": z_score, "mean": mean_val, "std": std_val},
            )
            return self._add_event(event)

        return None

    def _detect_oscillation(self) -> int:
        """检测位置振荡 (方向变化次数)。"""
        if len(self._position_errors) < 10:
            return 0

        recent = self._position_errors[-10:]
        direction_changes = 0

        prev_sign_x = None
        prev_sign_y = None

        for ex, ey in recent:
            sign_x = 1 if ex > 0 else -1 if ex < 0 else 0
            sign_y = 1 if ey > 0 else -1 if ey < 0 else 0

            if prev_sign_x is not None and sign_x != 0 and sign_x != prev_sign_x:
                direction_changes += 1
            if prev_sign_y is not None and sign_y != 0 and sign_y != prev_sign_y:
                direction_changes += 1

            prev_sign_x = sign_x
            prev_sign_y = sign_y

        return direction_changes

    def _add_event(self, event: AnomalyEvent) -> Optional[AnomalyEvent]:
        """添加异常事件 (含告警抑制)。"""
        now = time.time()

        # 告警抑制
        last_time = self._last_alert_time.get(event.anomaly_type, 0)
        if now - last_time < self.suppression_window_s:
            event.is_suppressed = True
            LOGGER.debug("Anomaly suppressed: %s", event.message)
            return None

        self._last_alert_time[event.anomaly_type] = now
        self._events.append(event)

        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]

        LOGGER.warning(
            "Anomaly detected [%s]: %s",
            event.severity.value,
            event.message,
        )

        return event

    def get_statistics(self) -> AnomalyStatistics:
        """获取异常统计。"""
        events_by_type: Dict[str, int] = {}
        events_by_severity: Dict[str, int] = {}
        suppression_count = 0
        last_event_time = None
        inter_event_intervals = []

        for i, event in enumerate(self._events):
            # 按类型统计
            type_key = event.anomaly_type.value
            events_by_type[type_key] = events_by_type.get(type_key, 0) + 1

            # 按严重程度统计
            sev_key = event.severity.value
            events_by_severity[sev_key] = events_by_severity.get(sev_key, 0) + 1

            # 抑制计数
            if event.is_suppressed:
                suppression_count += 1

            # 最后事件时间
            if last_event_time is None or event.timestamp > last_event_time:
                last_event_time = event.timestamp

            # 事件间隔
            if i > 0:
                inter_event_intervals.append(event.timestamp - self._events[i - 1].timestamp)

        mean_interval = float(np.mean(inter_event_intervals)) if inter_event_intervals else 0.0

        return AnomalyStatistics(
            total_events=len(self._events),
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            suppression_count=suppression_count,
            last_event_time=last_event_time,
            mean_inter_event_interval_s=round(mean_interval, 2),
        )

    def get_events(
        self,
        severity: Optional[AnomalySeverity] = None,
        anomaly_type: Optional[AnomalyType] = None,
        limit: int = 50,
    ) -> List[AnomalyEvent]:
        """获取异常事件列表。"""
        events = self._events

        if severity is not None:
            events = [e for e in events if e.severity == severity]
        if anomaly_type is not None:
            events = [e for e in events if e.anomaly_type == anomaly_type]

        return events[-limit:]

    def get_recent_events(self, count: int = 10) -> List[AnomalyEvent]:
        """获取最近的异常事件。"""
        return list(self._events[-count:])

    def clear_events(self) -> None:
        """清除所有异常事件。"""
        self._events.clear()
        self._last_alert_time.clear()
        LOGGER.info("Anomaly events cleared")

    def reset(self) -> None:
        """重置检测器。"""
        self._detection_latencies.clear()
        self._move_latencies.clear()
        self._confidences.clear()
        self._position_errors.clear()
        self._fps_history.clear()
        self._ewma_detection = None
        self._ewma_move = None
        self._ewma_fps = None
        self._events.clear()
        self._last_alert_time.clear()
        LOGGER.info("AnomalyDetector reset")

    def set_threshold(self, anomaly_type: AnomalyType, threshold: float) -> None:
        """设置异常阈值。"""
        self._thresholds[anomaly_type] = float(threshold)
        LOGGER.info("Threshold set: %s = %.2f", anomaly_type.value, threshold)

    def get_threshold(self, anomaly_type: AnomalyType) -> float:
        """获取异常阈值。"""
        return self._thresholds.get(anomaly_type, 0.0)
