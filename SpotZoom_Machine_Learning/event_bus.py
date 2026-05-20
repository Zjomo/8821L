"""
事件总线模块 (EventBus)

灵感来源:
- Bluesky RunEngine 事件系统 — 发布-订阅解耦
- python-control 信号路由 — 优先级与过滤
- Prometheus/OpenTelemetry 事件管道架构 — 记录与回放

功能:
- 线程安全的发布-订阅事件总线，实现模块间解耦通信
- 支持事件优先级排序、过滤器筛选、异步回调执行
- 提供事件历史记录与回放功能，便于调试与分析
- 使用环形缓冲区管理事件历史，避免内存无限增长

依赖: numpy, logging, dataclasses, threading, typing (无外部依赖)
"""

import logging
import time
import uuid
import heapq
import threading
import weakref
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

LOGGER = logging.getLogger("SpotZoom.EventBus")


# ---------------------------------------------------------------------------
# 预定义事件类型
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """预定义事件类型常量。"""
    DETECTION_COMPLETED = "detection_completed"      # 光斑检测完成
    ALIGNMENT_STEP = "alignment_step"                # 单步对准执行
    MOTOR_MOVED = "motor_moved"                      # 电机移动完成
    FOCUS_CHANGED = "focus_changed"                  # 焦距发生变化
    ANOMALY_DETECTED = "anomaly_detected"            # 检测到异常
    SAFETY_TRIGGERED = "safety_triggered"            # 安全保护触发
    CALIBRATION_UPDATED = "calibration_updated"      # 标定参数更新
    SYSTEM_STATUS = "system_status"                  # 系统状态变更


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SpotZoomEvent:
    """基础事件数据结构。

    Parameters
    ----------
    event_type : str
        事件类型，建议使用 EventType 枚举值。
    source : str
        事件来源模块名称。
    data : dict
        事件负载数据。
    priority : int
        事件优先级，数值越大优先级越高。默认 0。
    timestamp : float or None
        事件时间戳 (time.time())。为 None 时自动取当前时间。
    event_id : str or None
        事件唯一标识。为 None 时自动生成 UUID。
    """
    event_type: str
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __lt__(self, other: "SpotZoomEvent") -> bool:
        """优先级比较 (用于优先队列，数值越大优先级越高)。"""
        if not isinstance(other, SpotZoomEvent):
            return NotImplemented
        # heapq 是最小堆，因此取负数使大优先级排在前面
        return (-self.priority, self.timestamp) < (-other.priority, other.timestamp)


@dataclass
class EventSubscription:
    """订阅信息数据结构。

    Parameters
    ----------
    sub_id : str
        订阅唯一标识。
    event_type : str
        订阅的事件类型。
    callback : callable
        回调函数，接收 SpotZoomEvent 参数。
    priority : int
        订阅优先级，数值越大越先执行。默认 0。
    filter_fn : callable or None
        过滤函数，接收 SpotZoomEvent，返回 bool。
        仅当返回 True 时才调用回调。
    """
    sub_id: str
    event_type: str
    callback: Callable
    priority: int = 0
    filter_fn: Optional[Callable[[SpotZoomEvent], bool]] = None


# ---------------------------------------------------------------------------
# EventBus 核心实现
# ---------------------------------------------------------------------------

class EventBus:
    """线程安全的发布-订阅事件总线。

    支持事件优先级排序、过滤器筛选、异步回调执行、
    事件历史记录与回放。适用于 SpotZoom 光学对准系统中
    各模块之间的解耦通信。

    设计参考:
    - Bluesky RunEngine: 事件驱动的实验控制流
    - python-control: 信号路由与优先级
    - OpenTelemetry: 事件管道与可观测性

    Parameters
    ----------
    max_history : int
        事件历史最大记录条数，使用环形缓冲区存储。默认 1000。
    worker_threads : int
        异步处理的工作线程数。默认 1。
        设为 0 时，publish_async 将退化为同步执行。
    """

    def __init__(self, max_history: int = 1000, worker_threads: int = 1):
        self._max_history = int(max_history)
        self._worker_threads = int(worker_threads)

        # 订阅表: event_type -> [EventSubscription, ...]
        self._subscriptions: Dict[str, List[EventSubscription]] = {}
        # 全局订阅 (订阅所有事件): "*" -> [EventSubscription, ...]
        self._wildcard_subs: List[EventSubscription] = []

        # 事件历史环形缓冲区
        self._history: Deque[SpotZoomEvent] = deque(maxlen=self._max_history)

        # 线程安全锁
        self._lock = threading.Lock()

        # 异步事件队列
        self._async_queue: Queue[Tuple[int, SpotZoomEvent]] = Queue()
        # 序列号，用于保证同优先级事件的 FIFO 顺序
        self._seq = 0
        self._seq_lock = threading.Lock()

        # 工作线程
        self._workers: List[threading.Thread] = []
        self._shutdown_flag = threading.Event()

        if self._worker_threads > 0:
            self._start_workers()

        LOGGER.info(
            "EventBus 初始化完成: max_history=%d, worker_threads=%d",
            self._max_history,
            self._worker_threads,
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _start_workers(self) -> None:
        """启动异步工作线程。"""
        for i in range(self._worker_threads):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"EventBus-Worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        LOGGER.debug("已启动 %d 个异步工作线程", self._worker_threads)

    def _worker_loop(self) -> None:
        """工作线程主循环，从队列中取出事件并分发。"""
        while not self._shutdown_flag.is_set():
            try:
                seq, event = self._async_queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self._dispatch(event)
            except Exception:
                LOGGER.exception(
                    "异步事件分发异常: event_type=%s, source=%s",
                    event.event_type,
                    event.source,
                )
            finally:
                self._async_queue.task_done()

    def _get_next_seq(self) -> int:
        """获取下一个序列号 (线程安全)。"""
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _dispatch(self, event: SpotZoomEvent) -> int:
        """将事件分发给匹配的订阅者 (内部方法，调用方需持有锁或保证安全)。

        Returns
        -------
        int
            实际通知的订阅者数量。
        """
        notified = 0

        # 收集匹配的订阅者
        matched: List[EventSubscription] = []

        with self._lock:
            # 特定事件类型的订阅者
            subs = self._subscriptions.get(event.event_type, [])
            matched.extend(subs)
            # 通配符订阅者
            matched.extend(self._wildcard_subs)

        # 按优先级排序 (数值大的先执行)
        matched.sort(key=lambda s: s.priority, reverse=True)

        for sub in matched:
            try:
                # 检查回调是否仍然有效
                cb = sub.callback
                if isinstance(cb, weakref.ref):
                    cb = cb()
                    if cb is None:
                        LOGGER.debug("订阅 %s 的弱引用回调已失效，自动清理", sub.sub_id)
                        self.unsubscribe(sub.sub_id)
                        continue

                # 应用过滤器
                if sub.filter_fn is not None:
                    try:
                        if not sub.filter_fn(event):
                            continue
                    except Exception:
                        LOGGER.warning(
                            "订阅 %s 的过滤器异常，跳过该订阅", sub.sub_id
                        )
                        continue

                # 执行回调
                cb(event)
                notified += 1

            except Exception:
                LOGGER.exception(
                    "事件回调异常: sub_id=%s, event_type=%s, source=%s",
                    sub.sub_id,
                    event.event_type,
                    event.source,
                )

        return notified

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[SpotZoomEvent], None],
        priority: int = 0,
        filter_fn: Optional[Callable[[SpotZoomEvent], bool]] = None,
        weak: bool = False,
    ) -> str:
        """注册事件订阅。

        Parameters
        ----------
        event_type : str
            要订阅的事件类型。使用 "*" 可订阅所有事件。
        callback : callable
            回调函数，接收一个 SpotZoomEvent 参数。
        priority : int
            订阅优先级，数值越大越先执行。默认 0。
        filter_fn : callable or None
            过滤函数，接收 SpotZoomEvent，返回 bool。
            仅当返回 True 时才调用回调。默认 None (不过滤)。
        weak : bool
            是否使用弱引用持有回调。设为 True 可避免回调
            阻止对象被垃圾回收，但需注意回调可能提前失效。
            默认 False。

        Returns
        -------
        str
            订阅标识 (sub_id)，可用于取消订阅。
        """
        sub_id = uuid.uuid4().hex[:12]

        # 可选弱引用包装
        actual_callback: Callable = callback
        if weak:
            if not callable(callback):
                raise TypeError("callback 必须是可调用对象")
            actual_callback = weakref.ref(callback)  # type: ignore[assignment]

        sub = EventSubscription(
            sub_id=sub_id,
            event_type=event_type,
            callback=actual_callback,
            priority=int(priority),
            filter_fn=filter_fn,
        )

        with self._lock:
            if event_type == "*":
                self._wildcard_subs.append(sub)
            else:
                if event_type not in self._subscriptions:
                    self._subscriptions[event_type] = []
                self._subscriptions[event_type].append(sub)

        LOGGER.debug(
            "订阅注册: sub_id=%s, event_type=%s, priority=%d, weak=%s",
            sub_id,
            event_type,
            priority,
            weak,
        )
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """取消事件订阅。

        Parameters
        ----------
        sub_id : str
            订阅标识 (由 subscribe 返回)。

        Returns
        -------
        bool
            是否成功移除。若 sub_id 不存在则返回 False。
        """
        removed = False

        with self._lock:
            # 在通配符订阅中查找
            for i, sub in enumerate(self._wildcard_subs):
                if sub.sub_id == sub_id:
                    self._wildcard_subs.pop(i)
                    removed = True
                    break

            # 在特定类型订阅中查找
            if not removed:
                for etype, subs in self._subscriptions.items():
                    for i, sub in enumerate(subs):
                        if sub.sub_id == sub_id:
                            subs.pop(i)
                            removed = True
                            break
                    if removed:
                        break

        if removed:
            LOGGER.debug("订阅已移除: sub_id=%s", sub_id)
        else:
            LOGGER.debug("订阅移除失败 (未找到): sub_id=%s", sub_id)

        return removed

    def publish(self, event: SpotZoomEvent) -> int:
        """同步发布事件，立即分发给所有匹配的订阅者。

        Parameters
        ----------
        event : SpotZoomEvent
            要发布的事件对象。

        Returns
        -------
        int
            实际通知的订阅者数量。
        """
        # 记录到历史
        with self._lock:
            self._history.append(event)

        # 同步分发
        notified = self._dispatch(event)

        LOGGER.debug(
            "事件发布 (同步): type=%s, source=%s, notified=%d",
            event.event_type,
            event.source,
            notified,
        )
        return notified

    def publish_async(self, event: SpotZoomEvent) -> None:
        """异步发布事件，将事件放入队列由工作线程处理。

        若 worker_threads=0，则退化为同步执行。

        Parameters
        ----------
        event : SpotZoomEvent
            要发布的事件对象。
        """
        # 记录到历史
        with self._lock:
            self._history.append(event)

        if self._worker_threads <= 0:
            # 无工作线程，退化为同步
            self._dispatch(event)
            LOGGER.debug(
                "事件发布 (同步退化): type=%s, source=%s",
                event.event_type,
                event.source,
            )
        else:
            seq = self._get_next_seq()
            self._async_queue.put((seq, event))
            LOGGER.debug(
                "事件发布 (异步入队): type=%s, source=%s, queue_size=%d",
                event.event_type,
                event.source,
                self._async_queue.qsize(),
            )

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[SpotZoomEvent]:
        """获取事件历史记录。

        Parameters
        ----------
        event_type : str or None
            事件类型过滤。为 None 时返回所有类型的事件。
        limit : int
            最大返回条数。默认 100。

        Returns
        -------
        list of SpotZoomEvent
            事件列表，按时间正序排列 (最旧的在前)。
        """
        with self._lock:
            if event_type is None:
                events = list(self._history)
            else:
                events = [e for e in self._history if e.event_type == event_type]

        # 返回最近的 limit 条
        return events[-limit:] if len(events) > limit else events

    def clear_history(self) -> None:
        """清空事件历史记录。"""
        with self._lock:
            self._history.clear()
        LOGGER.debug("事件历史已清空")

    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """获取订阅者数量。

        Parameters
        ----------
        event_type : str or None
            事件类型。为 None 时返回所有订阅者总数。

        Returns
        -------
        int
            订阅者数量。
        """
        with self._lock:
            if event_type is None:
                total = len(self._wildcard_subs)
                for subs in self._subscriptions.values():
                    total += len(subs)
                return total
            elif event_type == "*":
                return len(self._wildcard_subs)
            else:
                return len(self._subscriptions.get(event_type, []))

    def replay(self, events: List[SpotZoomEvent]) -> int:
        """回放一组事件，重新分发给当前订阅者。

        回放的事件也会被记录到历史中。常用于调试场景，
        例如重现某个时刻的事件序列。

        Parameters
        ----------
        events : list of SpotZoomEvent
            要回放的事件列表。

        Returns
        -------
        int
            总通知次数。
        """
        if not events:
            return 0

        total_notified = 0
        for event in events:
            # 回放时更新时间戳，标记为回放事件
            replay_event = SpotZoomEvent(
                event_type=event.event_type,
                source=event.source,
                data={**event.data, "_replay": True, "_original_timestamp": event.timestamp},
                priority=event.priority,
                timestamp=time.time(),
                event_id=uuid.uuid4().hex[:12],
            )
            notified = self.publish(replay_event)
            total_notified += notified

        LOGGER.info(
            "事件回放完成: replay_count=%d, total_notified=%d",
            len(events),
            total_notified,
        )
        return total_notified

    def shutdown(self) -> None:
        """关闭事件总线，停止所有工作线程。

        关闭后将不再处理异步事件队列中的剩余事件。
        同步 publish 仍可使用，但通常不建议在关闭后继续操作。
        """
        self._shutdown_flag.set()

        # 等待工作线程结束
        for t in self._workers:
            t.join(timeout=5.0)

        self._workers.clear()

        # 清空异步队列
        while not self._async_queue.empty():
            try:
                self._async_queue.get_nowait()
                self._async_queue.task_done()
            except Empty:
                break

        LOGGER.info("EventBus 已关闭")

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def create_event(
        self,
        event_type: str,
        source: str = "",
        data: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> SpotZoomEvent:
        """创建事件对象的便捷方法。

        Parameters
        ----------
        event_type : str
            事件类型。
        source : str
            事件来源。
        data : dict or None
            事件负载数据。
        priority : int
            事件优先级。

        Returns
        -------
        SpotZoomEvent
            创建的事件对象。
        """
        return SpotZoomEvent(
            event_type=event_type,
            source=source,
            data=data or {},
            priority=priority,
        )

    def get_event_types(self) -> List[str]:
        """获取当前已注册订阅的所有事件类型。

        Returns
        -------
        list of str
            事件类型列表。
        """
        with self._lock:
            types = list(self._subscriptions.keys())
            if self._wildcard_subs:
                types.append("*")
            return sorted(types)

    def __repr__(self) -> str:
        return (
            f"EventBus(subscribers={self.get_subscriber_count()}, "
            f"history={len(self._history)}/{self._max_history}, "
            f"workers={len(self._workers)})"
        )

    def __len__(self) -> int:
        """返回当前历史记录中的事件数量。"""
        return len(self._history)
