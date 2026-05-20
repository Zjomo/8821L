"""
实时性能自适应调度器 (Real-time Performance Adaptive Scheduler)

灵感来源: Apache StreamPipes, adaptive-sampling 框架
           实时系统的反馈调度 (Feedback Scheduling)

核心思想:
──────────────────────────────────────────────────────────────────────
在实时控制系统中，计算资源有限但任务负载动态变化。
自适应调度根据当前系统负载和性能指标，动态调整各模块的
执行频率、计算精度和资源分配。

本模块实现:
- 基于控制理论的反馈调度器
- 多优先级任务队列管理
- 自适应采样率控制 (检测频率、控制频率)
- 计算精度动态调整 (检测分辨率、迭代次数)
- QoS 感知的资源分配
- 过载保护与优雅降级

适用场景:
- CPU/GPU 资源受限的嵌入式平台
- 多模块并行运行时的资源竞争
- 实时性要求高的闭环控制
"""

__version__ = "1.0.0"
__author__ = "SpotZoom Team"

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable, Any
from enum import Enum, auto
from collections import deque
import time
import threading


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 0    # 关键任务 (安全相关)
    HIGH = 1        # 高优先级 (核心控制)
    NORMAL = 2      # 普通优先级 (检测、分析)
    LOW = 3         # 低优先级 (日志、监控)
    BACKGROUND = 4  # 后台任务 (优化、训练)


class TaskState(Enum):
    """任务状态"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    SKIPPED = auto()
    DEGRADED = auto()


class DegradationLevel(Enum):
    """降级级别"""
    FULL = 0          # 全功能
    REDUCED_FREQ = 1  # 降低频率
    REDUCED_PRECISION = 2  # 降低精度
    MINIMAL = 3       # 最小功能
    SUSPENDED = 4     # 挂起


@dataclass
class ScheduledTask:
    """调度任务"""
    name: str
    priority: TaskPriority
    base_period_ms: float       # 基础执行周期 (ms)
    estimated_cost_ms: float    # 估计执行耗时 (ms)
    deadline_ms: float = 0.0    # 截止时间
    min_period_ms: float = 10.0 # 最小执行周期
    max_period_ms: float = 1000.0  # 最大执行周期
    state: TaskState = TaskState.PENDING
    degradation: DegradationLevel = DegradationLevel.FULL
    last_execution_time: float = 0.0
    last_cost_ms: float = 0.0
    skip_count: int = 0
    total_executions: int = 0
    quality_factor: float = 1.0  # 质量因子 [0, 1]


@dataclass
class SchedulerConfig:
    """调度器配置"""
    # 控制参数
    target_cpu_usage: float = 0.7       # 目标 CPU 使用率
    control_period_ms: float = 100.0    # 调度控制周期
    adaptation_rate: float = 0.1        # 适应速率

    # 降级阈值
    overload_threshold: float = 0.9     # 过载阈值
    underload_threshold: float = 0.3    # 欠载阈值
    critical_reserve: float = 0.15      # 关键任务保留资源

    # 采样率调整范围
    min_sample_ratio: float = 0.1       # 最小采样比
    max_sample_ratio: float = 1.0       # 最大采样比

    # 精度调整
    precision_levels: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.25])

    # 监控
    history_length: int = 200
    smoothing_factor: float = 0.3


@dataclass
class ScheduleDecision:
    """调度决策"""
    task_name: str
    action: str                          # execute / skip / degrade
    effective_period_ms: float
    quality_factor: float
    degradation_level: DegradationLevel
    estimated_cost_ms: float
    reason: str


class AdaptiveScheduler:
    """
    实时性能自适应调度器

    基于反馈控制理论，动态调整任务执行频率和精度:
    - PI 控制器调节整体 CPU 使用率
    - 优先级驱动的任务调度
    - 自适应采样率控制
    - 优雅降级策略

    用法示例:
        scheduler = AdaptiveScheduler(SchedulerConfig())
        scheduler.register_task("detection", TaskPriority.HIGH, period_ms=33)
        scheduler.register_task("kalman", TaskPriority.NORMAL, period_ms=33)
        scheduler.register_task("logging", TaskPriority.LOW, period_ms=1000)

        # 在主循环中
        decisions = scheduler.schedule(current_time)
        for d in decisions:
            if d.action == "execute":
                execute_task(d.task_name, d.quality_factor)
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()

        # 注册的任务
        self._tasks: Dict[str, ScheduledTask] = {}

        # PI 控制器状态
        self._integral_error = 0.0
        self._prev_error = 0.0
        self._kp = 0.5   # 比例增益
        self._ki = 0.1   # 积分增益

        # 性能监控
        self._cpu_history: deque = deque(maxlen=self.config.history_length)
        self._latency_history: deque = deque(maxlen=self.config.history_length)
        self._deadline_misses: int = 0
        self._total_scheduled: int = 0

        # 当前系统状态
        self._current_cpu_estimate: float = 0.5
        self._last_schedule_time: float = 0.0

        # 锁
        self._lock = threading.Lock()

    def register_task(self, name: str, priority: TaskPriority,
                      period_ms: float, estimated_cost_ms: float = 10.0,
                      deadline_ms: float = 0.0) -> None:
        """
        注册任务

        Args:
            name: 任务名称
            priority: 优先级
            period_ms: 执行周期 (ms)
            estimated_cost_ms: 估计执行耗时 (ms)
            deadline_ms: 截止时间 (ms), 0 表示等于周期
        """
        with self._lock:
            self._tasks[name] = ScheduledTask(
                name=name,
                priority=priority,
                base_period_ms=period_ms,
                estimated_cost_ms=estimated_cost_ms,
                deadline_ms=deadline_ms if deadline_ms > 0 else period_ms,
                min_period_ms=period_ms * self.config.min_sample_ratio,
                max_period_ms=period_ms * 2.0,
            )

    def unregister_task(self, name: str) -> None:
        """取消注册任务"""
        with self._lock:
            self._tasks.pop(name, None)

    def update_cpu_usage(self, cpu_fraction: float) -> None:
        """更新 CPU 使用率估计"""
        alpha = self.config.smoothing_factor
        self._current_cpu_estimate = (
            alpha * cpu_fraction + (1 - alpha) * self._current_cpu_estimate
        )
        self._cpu_history.append(self._current_cpu_estimate)

    def report_execution(self, task_name: str, actual_cost_ms: float,
                         deadline_met: bool = True) -> None:
        """
        报告任务执行结果

        Args:
            task_name: 任务名称
            actual_cost_ms: 实际执行耗时 (ms)
            deadline_met: 是否满足截止时间
        """
        with self._lock:
            if task_name in self._tasks:
                task = self._tasks[task_name]
                task.last_cost_ms = actual_cost_ms
                task.estimated_cost_ms = (
                    0.7 * task.estimated_cost_ms + 0.3 * actual_cost_ms
                )
                task.total_executions += 1

                if not deadline_met:
                    self._deadline_misses += 1

            self._latency_history.append(actual_cost_ms)

    def schedule(self, current_time: Optional[float] = None) -> List[ScheduleDecision]:
        """
        执行调度决策

        Args:
            current_time: 当前时间戳

        Returns:
            List[ScheduleDecision]: 各任务的调度决策
        """
        if current_time is None:
            current_time = time.time() * 1000  # 转换为 ms

        with self._lock:
            # 更新反馈控制器
            self._update_controller()

            # 计算可用资源预算
            budget = self._compute_budget()

            # 按优先级排序任务
            sorted_tasks = sorted(
                self._tasks.values(),
                key=lambda t: (t.priority.value, t.base_period_ms)
            )

            decisions = []
            remaining_budget = budget

            for task in sorted_tasks:
                decision = self._schedule_task(task, current_time, remaining_budget)
                decisions.append(decision)

                if decision.action == "execute":
                    remaining_budget -= decision.estimated_cost_ms

                self._total_scheduled += 1

            self._last_schedule_time = current_time
            return decisions

    def _update_controller(self) -> None:
        """PI 控制器更新"""
        error = self.config.target_cpu_usage - self._current_cpu_estimate

        self._integral_error += error
        # 积分限幅
        self._integral_error = np.clip(self._integral_error, -1.0, 1.0)

        # PI 输出
        output = self._kp * error + self._ki * self._integral_error

        # 更新全局采样率缩放因子
        self._global_scale = np.clip(
            1.0 + output * self.config.adaptation_rate,
            self.config.min_sample_ratio,
            self.config.max_sample_ratio
        )

        self._prev_error = error

    def _compute_budget(self) -> float:
        """计算当前调度周期的可用时间预算 (ms)"""
        period = self.config.control_period_ms
        # 可用预算 = 控制周期 * (1 - 保留比例)
        available = period * (1.0 - self.config.critical_reserve)
        return available

    def _schedule_task(self, task: ScheduledTask,
                       current_time: float,
                       remaining_budget: float) -> ScheduleDecision:
        """为单个任务做调度决策"""
        # 检查是否需要执行
        time_since_last = current_time - task.last_execution_time
        effective_period = task.base_period_ms / max(self._global_scale, 0.1)

        if time_since_last < effective_period:
            return ScheduleDecision(
                task_name=task.name,
                action="skip",
                effective_period_ms=effective_period,
                quality_factor=task.quality_factor,
                degradation_level=task.degradation,
                estimated_cost_ms=0,
                reason="未到执行时间"
            )

        # 检查资源是否充足
        estimated_cost = task.estimated_cost_ms * task.quality_factor

        if task.priority == TaskPriority.CRITICAL:
            # 关键任务始终执行
            action = "execute"
            reason = "关键任务, 必须执行"
            task.degradation = DegradationLevel.FULL
            task.quality_factor = 1.0

        elif remaining_budget < estimated_cost * 0.5:
            # 资源不足, 降级或跳过
            if task.priority.value <= TaskPriority.NORMAL.value:
                # 可降级
                new_quality = max(0.25, remaining_budget / max(estimated_cost, 1))
                degradation = self._quality_to_degradation(new_quality)
                task.quality_factor = new_quality
                task.degradation = degradation
                action = "degrade"
                reason = f"资源不足, 降级至 {degradation.name}"
            else:
                action = "skip"
                task.skip_count += 1
                reason = "资源不足, 跳过低优先级任务"

        elif self._current_cpu_estimate > self.config.overload_threshold:
            # 过载状态
            if task.priority.value >= TaskPriority.LOW.value:
                action = "skip"
                task.skip_count += 1
                reason = "系统过载, 跳过低优先级任务"
            else:
                # 降低精度
                scale = 1.0 - (self._current_cpu_estimate - self.config.overload_threshold) * 2
                new_quality = max(0.25, scale)
                task.quality_factor = new_quality
                task.degradation = self._quality_to_degradation(new_quality)
                action = "degrade"
                reason = f"系统过载, 降低精度至 {new_quality:.0%}"
        else:
            action = "execute"
            task.quality_factor = 1.0
            task.degradation = DegradationLevel.FULL
            reason = "正常执行"

        if action in ("execute", "degrade"):
            task.last_execution_time = current_time
            task.state = TaskState.RUNNING
        elif action == "skip":
            task.state = TaskState.SKIPPED

        return ScheduleDecision(
            task_name=task.name,
            action=action,
            effective_period_ms=effective_period,
            quality_factor=task.quality_factor,
            degradation_level=task.degradation,
            estimated_cost_ms=estimated_cost * task.quality_factor,
            reason=reason
        )

    def _quality_to_degradation(self, quality: float) -> DegradationLevel:
        """质量因子映射到降级级别"""
        if quality >= 0.9:
            return DegradationLevel.FULL
        elif quality >= 0.6:
            return DegradationLevel.REDUCED_FREQ
        elif quality >= 0.4:
            return DegradationLevel.REDUCED_PRECISION
        elif quality >= 0.2:
            return DegradationLevel.MINIMAL
        else:
            return DegradationLevel.SUSPENDED

    def get_effective_period(self, task_name: str) -> float:
        """获取任务的有效执行周期"""
        with self._lock:
            if task_name in self._tasks:
                task = self._tasks[task_name]
                return task.base_period_ms / max(self._global_scale, 0.1)
            return 0.0

    def get_quality_factor(self, task_name: str) -> float:
        """获取任务的质量因子"""
        with self._lock:
            if task_name in self._tasks:
                return self._tasks[task_name].quality_factor
            return 1.0

    def get_statistics(self) -> Dict:
        """获取调度统计信息"""
        with self._lock:
            task_stats = {}
            for name, task in self._tasks.items():
                task_stats[name] = {
                    "priority": task.priority.name,
                    "state": task.state.name,
                    "degradation": task.degradation.name,
                    "quality_factor": round(task.quality_factor, 3),
                    "skip_count": task.skip_count,
                    "total_executions": task.total_executions,
                    "avg_cost_ms": round(task.last_cost_ms, 2),
                    "effective_period_ms": round(
                        task.base_period_ms / max(self._global_scale, 0.1), 2
                    )
                }

            return {
                "current_cpu_estimate": round(self._current_cpu_estimate, 3),
                "global_scale": round(getattr(self, '_global_scale', 1.0), 3),
                "deadline_misses": self._deadline_misses,
                "total_scheduled": self._total_scheduled,
                "tasks": task_stats,
                "avg_latency_ms": round(
                    np.mean(list(self._latency_history)) if self._latency_history else 0, 2
                )
            }

    def reset(self) -> None:
        """重置调度器"""
        with self._lock:
            self._integral_error = 0.0
            self._prev_error = 0.0
            self._deadline_misses = 0
            self._total_scheduled = 0
            self._cpu_history.clear()
            self._latency_history.clear()
            for task in self._tasks.values():
                task.skip_count = 0
                task.total_executions = 0
                task.quality_factor = 1.0
                task.degradation = DegradationLevel.FULL
                task.state = TaskState.PENDING
