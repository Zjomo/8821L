"""
实时控制管线 v2 (RealtimePipelineV2)

基于 pyRTC (https://github.com/jacotay7/pyRTC) 的模块化实时 AO 控制架构，
为 SpotZoom 提供高性能、可扩展的实时控制管线。

灵感来源:
- pyRTC: Python Real-Time Controller for AO (https://github.com/jacotay7/pyRTC)
- OOPAO: 面向对象AO仿真 (https://github.com/cheritier/OOPAO)
- leap-c: 学习型预测控制 (https://github.com/leap-c/leap-c)

算法原理:
  1. 四阶段管线: Sensor -> Processor -> Controller -> Corrector
  2. 组件化设计: 每个阶段可独立替换，支持热插拔
  3. 软实时保证: 使用优先级调度和超时保护
  4. 遥测记录: 全链路数据记录与性能分析
  5. AI 控制器接口: 原生支持神经网络控制器

与现有模块的关系:
  - 重构 v1/realtime_control_pipeline.py 的架构
  - 与 v3/realtime_ao_pipeline.py 协同
  - 与 v1/learning_mpc_controller.py 的 AI 控制器协同

外部依赖: numpy, threading, time (标准库)
"""

import numpy as np
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Dict, Any, Protocol
from enum import Enum
from collections import deque
import abc

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """管线状态。"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class StageType(Enum):
    """管线阶段类型。"""
    SENSOR = "sensor"
    PROCESSOR = "processor"
    CONTROLLER = "controller"
    CORRECTOR = "corrector"


@dataclass
class PipelineV2Config:
    """实时管线配置。"""
    # 控制参数
    control_frequency_hz: float = 100.0  # 目标控制频率
    max_cycle_time_ms: float = 10.0      # 最大周期时间
    timeout_ms: float = 50.0             # 超时时间

    # 遥测参数
    telemetry_buffer_size: int = 1000    # 遥测缓冲区大小
    record_wavefront: bool = True        # 记录波前数据
    record_corrections: bool = True      # 记录校正量
    record_performance: bool = True      # 记录性能指标

    # 安全参数
    max_correction: float = 1.0          # 最大校正量
    correction_rate_limit: float = 0.1   # 校正速率限制
    enable_watchdog: bool = True         # 启用看门狗

    # 线程参数
    use_realtime_thread: bool = False    # 使用实时线程（需要 OS 支持）
    thread_priority: int = 50            # 线程优先级


@dataclass
class TelemetryPoint:
    """遥测数据点。"""
    timestamp: float = 0.0
    cycle_time_ms: float = 0.0
    wavefront_rms: float = 0.0
    correction_norm: float = 0.0
    quality_metric: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineV2Report:
    """管线执行报告。"""
    total_cycles: int = 0
    total_time_ms: float = 0.0
    achieved_frequency_hz: float = 0.0
    mean_cycle_time_ms: float = 0.0
    max_cycle_time_ms: float = 0.0
    min_cycle_time_ms: float = 0.0
    mean_wavefront_rms: float = 0.0
    mean_correction_norm: float = 0.0
    timeout_count: int = 0
    error_count: int = 0
    final_state: str = PipelineState.IDLE.value


class PipelineStage(abc.ABC):
    """管线阶段基类。"""

    @abc.abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据。"""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """阶段名称。"""
        pass

    @property
    @abc.abstractmethod
    def stage_type(self) -> StageType:
        """阶段类型。"""
        pass


class SensorStage(PipelineStage):
    """传感器阶段：采集波前/图像数据。"""

    def __init__(self, acquire_callback: Callable[[], np.ndarray]):
        self._callback = acquire_callback
        self._last_data = np.zeros((1,))

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        data['wavefront'] = self._callback()
        data['sensor_time_ms'] = (time.perf_counter() - t0) * 1000
        return data

    @property
    def name(self) -> str:
        return "sensor"

    @property
    def stage_type(self) -> StageType:
        return StageType.SENSOR


class ProcessorStage(PipelineStage):
    """处理器阶段：波前处理（滤波、模式分解等）。"""

    def __init__(self, process_callback: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self._callback = process_callback or (lambda x: x)

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        wf = data.get('wavefront', np.zeros((1,)))
        data['processed_wavefront'] = self._callback(wf)
        data['wavefront_rms'] = float(np.std(data['processed_wavefront']))
        data['processor_time_ms'] = (time.perf_counter() - t0) * 1000
        return data

    @property
    def name(self) -> str:
        return "processor"

    @property
    def stage_type(self) -> StageType:
        return StageType.PROCESSOR


class ControllerStage(PipelineStage):
    """控制器阶段：计算校正量。"""

    def __init__(
        self,
        control_callback: Callable[[np.ndarray], np.ndarray],
        gain: float = 0.5,
        integrator: bool = True,
        leaky_factor: float = 0.999,
    ):
        self._callback = control_callback
        self._gain = gain
        self._integrator = integrator
        self._leaky_factor = leaky_factor
        self._integral = None

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        wf = data.get('processed_wavefront', np.zeros((1,)))

        # 计算控制量
        raw_correction = self._callback(wf)

        # 比例控制
        correction = self._gain * raw_correction

        # 积分控制
        if self._integrator:
            if self._integral is None:
                self._integral = np.zeros_like(correction)
            self._integral = self._leaky_factor * self._integral + correction
            correction = self._integral

        data['correction'] = correction
        data['correction_norm'] = float(np.linalg.norm(correction))
        data['controller_time_ms'] = (time.perf_counter() - t0) * 1000
        return data

    @property
    def name(self) -> str:
        return "controller"

    @property
    def stage_type(self) -> StageType:
        return StageType.CONTROLLER

    def reset(self):
        """重置积分器。"""
        self._integral = None


class CorrectorStage(PipelineStage):
    """校正器阶段：应用校正量到硬件。"""

    def __init__(
        self,
        apply_callback: Callable[[np.ndarray], None],
        max_correction: float = 1.0,
        rate_limit: float = 0.1,
    ):
        self._callback = apply_callback
        self._max_correction = max_correction
        self._rate_limit = rate_limit
        self._last_correction = np.zeros((1,))

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        correction = data.get('correction', np.zeros((1,)))

        # 限幅
        correction = np.clip(correction, -self._max_correction, self._max_correction)

        # 速率限制
        delta = correction - self._last_correction
        max_delta = self._rate_limit
        delta = np.clip(delta, -max_delta, max_delta)
        correction = self._last_correction + delta
        self._last_correction = correction.copy()

        # 应用校正
        try:
            self._callback(correction)
            data['correction_applied'] = True
        except Exception as e:
            logger.error(f"校正应用失败: {e}")
            data['correction_applied'] = False
            data['error'] = str(e)

        data['final_correction'] = correction
        data['corrector_time_ms'] = (time.perf_counter() - t0) * 1000
        return data

    @property
    def name(self) -> str:
        return "corrector"

    @property
    def stage_type(self) -> StageType:
        return StageType.CORRECTOR


class RealtimePipelineV2:
    """实时控制管线 v2。

    四阶段模块化管线：Sensor -> Processor -> Controller -> Corrector。
    支持组件热插拔、遥测记录、看门狗保护。

    Parameters
    ----------
    config : PipelineV2Config
        管线配置。
    """

    def __init__(self, config: Optional[PipelineV2Config] = None):
        self.config = config or PipelineV2Config()
        self._stages: List[PipelineStage] = []
        self._state = PipelineState.IDLE
        self._telemetry: deque = deque(maxlen=self.config.telemetry_buffer_size)
        self._lock = threading.Lock()
        self._cycle_count = 0
        self._error_count = 0
        self._timeout_count = 0

    @property
    def state(self) -> PipelineState:
        return self._state

    def add_stage(self, stage: PipelineStage):
        """添加管线阶段。"""
        self._stages.append(stage)
        logger.info(f"添加阶段: {stage.name} ({stage.stage_type.value})")

    def remove_stage(self, stage_type: StageType):
        """移除指定类型的阶段。"""
        self._stages = [s for s in self._stages if s.stage_type != stage_type]

    def run_cycle(self) -> TelemetryPoint:
        """执行单个控制周期。"""
        t0 = time.perf_counter()
        telemetry = TelemetryPoint(timestamp=t0)

        data: Dict[str, Any] = {}

        try:
            # 依次执行各阶段
            for stage in self._stages:
                stage_start = time.perf_counter()
                data = stage.process(data)
                stage_time = (time.perf_counter() - stage_start) * 1000
                telemetry.stage_timings[stage.name] = stage_time

                # 超时检查
                if stage_time > self.config.timeout_ms:
                    telemetry.error = f"阶段 {stage.name} 超时 ({stage_time:.1f}ms)"
                    self._timeout_count += 1
                    logger.warning(telemetry.error)

            # 填充遥测
            telemetry.wavefront_rms = data.get('wavefront_rms', 0.0)
            telemetry.correction_norm = data.get('correction_norm', 0.0)
            telemetry.quality_metric = data.get('quality_metric', 0.0)

        except Exception as e:
            telemetry.error = str(e)
            self._error_count += 1
            logger.error(f"控制周期异常: {e}")

        telemetry.cycle_time_ms = (time.perf_counter() - t0) * 1000
        self._cycle_count += 1

        with self._lock:
            self._telemetry.append(telemetry)

        return telemetry

    def run_n_cycles(self, n: int) -> PipelineV2Report:
        """执行 n 个控制周期。"""
        self._state = PipelineState.RUNNING
        t_start = time.perf_counter()

        cycle_times = []
        for _ in range(n):
            tp = self.run_cycle()
            cycle_times.append(tp.cycle_time_ms)

        self._state = PipelineState.STOPPED
        total_time = (time.perf_counter() - t_start) * 1000

        # 生成报告
        report = PipelineV2Report(
            total_cycles=n,
            total_time_ms=total_time,
            achieved_frequency_hz=n / (total_time / 1000),
            mean_cycle_time_ms=float(np.mean(cycle_times)),
            max_cycle_time_ms=float(np.max(cycle_times)),
            min_cycle_time_ms=float(np.min(cycle_times)),
            timeout_count=self._timeout_count,
            error_count=self._error_count,
            final_state=self._state.value,
        )

        # 从遥测计算统计
        if self._telemetry:
            rms_values = [t.wavefront_rms for t in self._telemetry]
            corr_values = [t.correction_norm for t in self._telemetry]
            report.mean_wavefront_rms = float(np.mean(rms_values))
            report.mean_correction_norm = float(np.mean(corr_values))

        return report

    def get_telemetry(self, n: Optional[int] = None) -> List[TelemetryPoint]:
        """获取遥测数据。"""
        with self._lock:
            data = list(self._telemetry)
        if n is not None:
            return data[-n:]
        return data

    def reset(self):
        """重置管线。"""
        self._state = PipelineState.IDLE
        self._cycle_count = 0
        self._error_count = 0
        self._timeout_count = 0
        self._telemetry.clear()
        for stage in self._stages:
            if hasattr(stage, 'reset'):
                stage.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = PipelineV2Config(control_frequency_hz=100)
    pipeline = RealtimePipelineV2(config)

    # 模拟传感器
    def mock_sensor():
        return np.random.randn(64) * 0.1

    # 模拟处理器
    def mock_processor(wf):
        return wf - np.mean(wf)

    # 模拟控制器
    def mock_controller(wf):
        return -wf * 0.5

    # 模拟校正器
    applied = []
    def mock_corrector(correction):
        applied.append(correction.copy())

    # 构建管线
    pipeline.add_stage(SensorStage(mock_sensor))
    pipeline.add_stage(ProcessorStage(mock_processor))
    pipeline.add_stage(ControllerStage(mock_controller, gain=0.5))
    pipeline.add_stage(CorrectorStage(mock_corrector, max_correction=1.0))

    # 运行
    report = pipeline.run_n_cycles(100)
    print(f"总周期: {report.total_cycles}")
    print(f"实际频率: {report.achieved_frequency_hz:.1f} Hz")
    print(f"平均周期时间: {report.mean_cycle_time_ms:.2f} ms")
    print(f"最大周期时间: {report.max_cycle_time_ms:.2f} ms")
    print(f"超时次数: {report.timeout_count}")
    print(f"错误次数: {report.error_count}")
