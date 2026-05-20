"""
Runtime Profiler - 运行时性能分析器

Inspired by:
- py-spy: Sampling profiler for Python programs
- cProfile: Python built-in profiler
- TensorBoard: Real-time metrics visualization

Core Innovation:
- 轻量级采样性能分析，不影响实时控制
- 分阶段耗时统计 (采集/检测/控制/通信)
- 性能瓶颈自动识别
- 内存使用监控
- 纯 Python 实现，零外部依赖
"""

from __future__ import annotations

import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ProfilerConfig:
    """性能分析器配置"""
    # 是否启用
    enabled: bool = True
    # 历史记录长度
    history_length: int = 1000
    # 瓶颈检测阈值 (ms)
    bottleneck_threshold_ms: float = 50.0
    # 报告间隔 (秒)
    report_interval_s: float = 10.0
    # 是否启用内存监控
    memory_monitoring: bool = True
    # 采样间隔 (ms)
    sample_interval_ms: float = 10.0


@dataclass
class StageTiming:
    """阶段耗时统计"""
    name: str
    total_time_ms: float = 0.0
    count: int = 0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    recent_times: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(1, self.count)

    @property
    def recent_avg_ms(self) -> float:
        if not self.recent_times:
            return 0.0
        return sum(self.recent_times) / len(self.recent_times)


@dataclass
class ProfilerReport:
    """性能分析报告"""
    # 各阶段耗时
    stage_timings: Dict[str, StageTiming]
    # 总运行时间 (ms)
    total_runtime_ms: float
    # 总帧数
    total_frames: int
    # 平均帧率
    avg_fps: float
    # 瓶颈阶段
    bottleneck_stage: Optional[str]
    # 瓶颈耗时 (ms)
    bottleneck_time_ms: float
    # 内存使用 (MB)
    memory_usage_mb: float
    # 周期时间统计
    cycle_time_avg_ms: float
    cycle_time_p50_ms: float
    cycle_time_p95_ms: float
    cycle_time_p99_ms: float


class RuntimeProfiler:
    """运行时性能分析器

    轻量级性能监控工具，用于识别对准系统中的
    性能瓶颈和优化机会。

    Inspired by py-spy's sampling approach and TensorBoard's
    real-time metrics tracking.
    """

    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self._stages: Dict[str, StageTiming] = {}
        self._cycle_times: Deque[float] = deque(maxlen=self.config.history_length)
        self._start_time: Optional[float] = None
        self._cycle_start: Optional[float] = None
        self._frame_count: int = 0
        self._lock = threading.Lock()
        self._active_stage: Optional[str] = None
        self._stage_start_time: Optional[float] = None
        self._peak_memory_mb: float = 0.0

    def reset(self) -> None:
        """重置分析器"""
        with self._lock:
            self._stages.clear()
            self._cycle_times.clear()
            self._start_time = time.perf_counter()
            self._cycle_start = None
            self._frame_count = 0
            self._peak_memory_mb = 0.0

    def start_cycle(self) -> None:
        """开始一个控制周期"""
        if not self.config.enabled:
            return
        self._cycle_start = time.perf_counter()
        if self._start_time is None:
            self._start_time = time.perf_counter()

    def end_cycle(self) -> None:
        """结束一个控制周期"""
        if not self.config.enabled or self._cycle_start is None:
            return
        cycle_time = (time.perf_counter() - self._cycle_start) * 1000.0
        self._cycle_times.append(cycle_time)
        self._frame_count += 1
        self._cycle_start = None

    def start_stage(self, name: str) -> None:
        """开始一个阶段计时"""
        if not self.config.enabled:
            return
        self._active_stage = name
        self._stage_start_time = time.perf_counter()

    def end_stage(self, name: str) -> float:
        """结束一个阶段计时

        Args:
            name: 阶段名称

        Returns:
            耗时 (ms)
        """
        if not self.config.enabled or self._stage_start_time is None:
            return 0.0

        elapsed = (time.perf_counter() - self._stage_start_time) * 1000.0

        with self._lock:
            if name not in self._stages:
                self._stages[name] = StageTiming(name=name)

            timing = self._stages[name]
            timing.total_time_ms += elapsed
            timing.count += 1
            timing.min_time_ms = min(timing.min_time_ms, elapsed)
            timing.max_time_ms = max(timing.max_time_ms, elapsed)
            timing.recent_times.append(elapsed)

        self._active_stage = None
        self._stage_start_time = None
        return elapsed

    def record_stage(self, name: str, duration_ms: float) -> None:
        """直接记录阶段耗时"""
        if not self.config.enabled:
            return

        with self._lock:
            if name not in self._stages:
                self._stages[name] = StageTiming(name=name)

            timing = self._stages[name]
            timing.total_time_ms += duration_ms
            timing.count += 1
            timing.min_time_ms = min(timing.min_time_ms, duration_ms)
            timing.max_time_ms = max(timing.max_time_ms, duration_ms)
            timing.recent_times.append(duration_ms)

    def _get_memory_usage_mb(self) -> float:
        """获取当前内存使用量 (MB)"""
        if not self.config.memory_monitoring:
            return 0.0
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / (1024 * 1024)
            self._peak_memory_mb = max(self._peak_memory_mb, mem)
            return mem
        except ImportError:
            return 0.0

    def get_report(self) -> ProfilerReport:
        """生成性能分析报告"""
        with self._lock:
            stages = dict(self._stages)

        # 总运行时间
        if self._start_time is not None:
            total_runtime = (time.perf_counter() - self._start_time) * 1000.0
        else:
            total_runtime = 0.0

        # 帧率
        if total_runtime > 0:
            avg_fps = self._frame_count / (total_runtime / 1000.0)
        else:
            avg_fps = 0.0

        # 瓶颈检测
        bottleneck_stage = None
        bottleneck_time = 0.0
        for name, timing in stages.items():
            if timing.recent_avg_ms > self.config.bottleneck_threshold_ms:
                if timing.recent_avg_ms > bottleneck_time:
                    bottleneck_stage = name
                    bottleneck_time = timing.recent_avg_ms

        # 如果没有超过阈值的，取最慢的
        if bottleneck_stage is None and stages:
            bottleneck_stage = max(stages, key=lambda n: stages[n].recent_avg_ms)
            bottleneck_time = stages[bottleneck_stage].recent_avg_ms

        # 周期时间统计
        if self._cycle_times:
            cycle_arr = np.array(list(self._cycle_times))
            cycle_avg = float(np.mean(cycle_arr))
            cycle_p50 = float(np.percentile(cycle_arr, 50))
            cycle_p95 = float(np.percentile(cycle_arr, 95))
            cycle_p99 = float(np.percentile(cycle_arr, 99))
        else:
            cycle_avg = cycle_p50 = cycle_p95 = cycle_p99 = 0.0

        return ProfilerReport(
            stage_timings=stages,
            total_runtime_ms=total_runtime,
            total_frames=self._frame_count,
            avg_fps=avg_fps,
            bottleneck_stage=bottleneck_stage,
            bottleneck_time_ms=bottleneck_time,
            memory_usage_mb=self._get_memory_usage_mb(),
            cycle_time_avg_ms=cycle_avg,
            cycle_time_p50_ms=cycle_p50,
            cycle_time_p95_ms=cycle_p95,
            cycle_time_p99_ms=cycle_p99,
        )

    def get_stage_summary(self) -> Dict[str, dict]:
        """获取各阶段耗时摘要"""
        with self._lock:
            stages = dict(self._stages)

        summary = {}
        for name, timing in stages.items():
            summary[name] = {
                "avg_ms": timing.avg_time_ms,
                "recent_avg_ms": timing.recent_avg_ms,
                "min_ms": timing.min_time_ms if timing.min_time_ms != float("inf") else 0.0,
                "max_ms": timing.max_time_ms,
                "count": timing.count,
                "total_ms": timing.total_time_ms,
                "is_bottleneck": timing.recent_avg_ms > self.config.bottleneck_threshold_ms,
            }

        return summary

    def get_diagnostics(self) -> dict:
        """获取分析器诊断信息"""
        report = self.get_report()
        return {
            "enabled": self.config.enabled,
            "total_frames": report.total_frames,
            "avg_fps": report.avg_fps,
            "bottleneck": report.bottleneck_stage,
            "peak_memory_mb": self._peak_memory_mb,
            "num_stages": len(report.stage_timings),
        }
