"""
实时控制管线引擎 (RealtimeControlPipeline)

灵感来源:
- pyRTC (https://github.com/jacotay7/pyRTC) — 组件化自适应光学实时控制框架，
  将 AO 控制循环分解为 Sensor -> Processor -> Controller -> Corrector 四个
  独立组件，支持软/硬实时模式切换
- ARTIQ (M-Labs) — 量子实验实时控制框架，支持确定性时序和内核级实时调度
- CACAO (ESO) — 自适应光学实时控制闭环，多线程流水线架构
- RTC Toolbox (MathWorks) — 实时控制工具箱，支持硬实时仿真与部署

算法原理:
- Component-Based Pipeline — 组件化管线: 将控制循环分解为可独立替换的功能组件
- Soft Real-Time Control — 软实时控制: 单线程顺序执行，依赖 OS 调度
- Hard Real-Time Simulation — 硬实时仿真: 多线程 + 共享内存 IPC 模拟
- PID Control Law — PID 控制律: u(t) = Kp*e + Ki*int(e) + Kd*de/dt
- Integrator Anti-Windup — 积分抗饱和: 限制积分项累积，防止超调
- Bumpless Transfer — 无扰动切换: 控制模式切换时保持输出连续性
- Deadband Filtering — 死区滤波: 小误差不产生控制输出，减少抖动
- Rate Limiting — 速率限制: 限制控制输出变化速率，保护执行机构
- Timing Jitter Analysis — 时序抖动分析: 统计循环周期的方差和百分位

功能:
- 组件化 AO 控制管线 (Sensor -> Processor -> Controller -> Corrector)
- 软实时 (单线程) 和硬实时仿真 (多线程) 两种运行模式
- 可配置管线频率和时序分析
- 管线暂停/恢复/单步执行模式
- 错误恢复与容错机制
- 组件热替换支持
- 完整遥测记录与性能报告

依赖: numpy, opencv-python (仅用于图像处理辅助), logging, threading, time
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

LOGGER = logging.getLogger("SpotZoom.RealtimeControlPipeline")

# 模块默认禁用标志
realtime_control_pipeline_enabled: bool = False


# ======================== 枚举与状态 ========================


class PipelineState(Enum):
    """管线运行状态。"""
    IDLE = "idle"           # 空闲 (未启动)
    RUNNING = "running"     # 运行中
    PAUSED = "paused"       # 已暂停
    ERROR = "error"         # 错误状态
    STOPPED = "stopped"     # 已停止


class ControllerType(Enum):
    """控制器类型。"""
    P = "P"       # 比例控制
    PI = "PI"     # 比例积分控制
    PID = "PID"   # 比例积分微分控制


# ======================== 配置数据类 ========================


@dataclass
class PipelineConfig:
    """管线配置参数。

    Attributes
    ----------
    target_frequency_hz : float
        目标管线循环频率 (Hz)。
    mode : str
        运行模式: 'soft' (单线程软实时) 或 'hard' (多线程硬实时仿真)。
    enable_telemetry : bool
        是否启用遥测记录。
    telemetry_buffer_size : int
        遥测数据缓冲区大小 (样本数)。
    max_frame_delay_s : float
        最大允许帧延迟 (秒)，超过则丢弃。
    error_recovery_retries : int
        错误恢复最大重试次数。
    component_timeout_s : float
        单个组件处理超时时间 (秒)。
    """
    target_frequency_hz: float = 10.0
    mode: str = "soft"  # 'soft', 'hard'
    enable_telemetry: bool = True
    telemetry_buffer_size: int = 1000
    max_frame_delay_s: float = 1.0
    error_recovery_retries: int = 3
    component_timeout_s: float = 5.0


# ======================== 结果数据类 ========================


@dataclass
class ComponentTimingStats:
    """单个组件的时序统计。

    Attributes
    ----------
    name : str
        组件名称。
    call_count : int
        调用次数。
    mean_ms : float
        平均处理时间 (毫秒)。
    min_ms : float
        最小处理时间 (毫秒)。
    max_ms : float
        最大处理时间 (毫秒)。
    std_ms : float
        处理时间标准差 (毫秒)。
    p50_ms : float
        处理时间 P50 (毫秒)。
    p99_ms : float
        处理时间 P99 (毫秒)。
    timeout_count : int
        超时次数。
    error_count : int
        错误次数。
    """
    name: str = ""
    call_count: int = 0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p99_ms: float = 0.0
    timeout_count: int = 0
    error_count: int = 0


@dataclass
class LoopRateStats:
    """循环速率统计。

    Attributes
    ----------
    mean_hz : float
        平均循环频率 (Hz)。
    min_hz : float
        最低循环频率 (Hz)。
    max_hz : float
        最高循环频率 (Hz)。
    std_hz : float
        循环频率标准差 (Hz)。
    p99_hz : float
        循环频率 P99 (Hz)。
    target_hz : float
        目标频率 (Hz)。
    samples : int
        统计样本数。
    """
    mean_hz: float = 0.0
    min_hz: float = 0.0
    max_hz: float = 0.0
    std_hz: float = 0.0
    p99_hz: float = 0.0
    target_hz: float = 0.0
    samples: int = 0


@dataclass
class LatencyBreakdown:
    """管线延迟分解。

    Attributes
    ----------
    sensor_ms : float
        传感器采集延迟 (毫秒)。
    processing_ms : float
        图像处理延迟 (毫秒)。
    control_ms : float
        控制计算延迟 (毫秒)。
    correction_ms : float
        校正执行延迟 (毫秒)。
    total_ms : float
        总延迟 (毫秒)。
    overhead_ms : float
        管线开销 (毫秒)。
    """
    sensor_ms: float = 0.0
    processing_ms: float = 0.0
    control_ms: float = 0.0
    correction_ms: float = 0.0
    total_ms: float = 0.0
    overhead_ms: float = 0.0


@dataclass
class PipelineReport:
    """管线运行报告。

    Attributes
    ----------
    actual_frequency_hz : float
        实际平均循环频率 (Hz)。
    timing_jitter_ms : float
        时序抖动 (毫秒)，即循环周期标准差。
    component_breakdown : Dict[str, ComponentTimingStats]
        各组件时序统计。
    total_frames_processed : int
        总处理帧数。
    error_count : int
        总错误数。
    state : PipelineState
        管线当前状态。
    loop_rate : LoopRateStats
        循环速率统计。
    latency : LatencyBreakdown
        延迟分解。
    uptime_s : float
        总运行时间 (秒)。
    """
    actual_frequency_hz: float = 0.0
    timing_jitter_ms: float = 0.0
    component_breakdown: Dict[str, ComponentTimingStats] = field(default_factory=dict)
    total_frames_processed: int = 0
    error_count: int = 0
    state: PipelineState = PipelineState.IDLE
    loop_rate: LoopRateStats = field(default_factory=LoopRateStats)
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    uptime_s: float = 0.0


# ======================== 组件接口 (Protocol) ========================


@runtime_checkable
class PipelineComponent(Protocol):
    """管线组件基接口 (Protocol)。

    所有管线组件必须实现以下方法:
    - initialize(): 初始化组件
    - process(input_data): 处理输入数据，返回输出数据
    - reset(): 重置组件状态
    - get_status(): 获取组件状态描述

    组件元数据属性:
    - name: str — 组件名称
    - version: str — 组件版本
    - description: str — 组件描述
    """
    name: str
    version: str
    description: str

    def initialize(self) -> None:
        """初始化组件。"""
        ...

    def process(self, input_data: Any) -> Any:
        """处理输入数据。

        Parameters
        ----------
        input_data : Any
            输入数据 (类型由具体组件定义)。

        Returns
        -------
        Any
            处理结果。
        """
        ...

    def reset(self) -> None:
        """重置组件内部状态。"""
        ...

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态。

        Returns
        -------
        Dict[str, Any]
            状态信息字典。
        """
        ...


# ======================== 传感器组件 ========================


@dataclass
class SensorOutput:
    """传感器输出数据。

    Attributes
    ----------
    frame : np.ndarray
        采集的图像帧 (2D, uint8 或 float)。
    timestamp : float
        采集时间戳 (time.time())。
    frame_id : int
        帧序号。
    is_valid : bool
        帧是否有效 (False 表示丢帧)。
    exposure_time_ms : float
        曝光时间 (毫秒)。
    """
    frame: np.ndarray = field(default_factory=lambda: np.zeros((64, 64), dtype=np.uint8))
    timestamp: float = 0.0
    frame_id: int = 0
    is_valid: bool = True
    exposure_time_ms: float = 10.0


class SensorComponent:
    """传感器组件 (模拟/真实)。

    模拟相机帧采集，支持:
    - 可配置帧率和噪声水平
    - 帧缓冲和时间戳
    - 丢帧模拟 (用于鲁棒性测试)
    - 高斯噪声和泊松噪声

    Parameters
    ----------
    frame_width : int
        帧宽度 (像素)。
    frame_height : int
        帧高度 (像素)。
    noise_std : float
        高斯噪声标准差 (0 表示无噪声)。
    dropout_probability : float
        丢帧概率 [0, 1]。
    exposure_time_ms : float
        模拟曝光时间 (毫秒)。
    pixel_depth : str
        像素深度: 'uint8' 或 'float32'。
    """

    def __init__(
        self,
        frame_width: int = 64,
        frame_height: int = 64,
        noise_std: float = 2.0,
        dropout_probability: float = 0.0,
        exposure_time_ms: float = 10.0,
        pixel_depth: str = "uint8",
    ):
        self.name: str = "SensorComponent"
        self.version: str = "1.0.0"
        self.description: str = "传感器组件: 模拟/真实帧采集"

        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.noise_std = float(noise_std)
        self.dropout_probability = float(dropout_probability)
        self.exposure_time_ms = float(exposure_time_ms)
        self.pixel_depth = pixel_depth

        self._frame_id: int = 0
        self._initialized: bool = False
        self._last_frame_time: float = 0.0

        # 帧缓冲 (最近 N 帧)
        self._buffer_size: int = 5
        self._frame_buffer: Deque[SensorOutput] = deque(maxlen=self._buffer_size)

    def initialize(self) -> None:
        """初始化传感器组件。"""
        self._frame_id = 0
        self._initialized = True
        self._last_frame_time = time.time()
        self._frame_buffer.clear()
        LOGGER.info(f"[{self.name}] 初始化完成: {self.frame_width}x{self.frame_height}, "
                     f"噪声={self.noise_std}, 丢帧率={self.dropout_probability:.1%}")

    def process(self, input_data: Any = None) -> SensorOutput:
        """采集一帧图像。

        Parameters
        ----------
        input_data : Any
            可选的外部帧数据。为 None 时生成模拟帧。

        Returns
        -------
        SensorOutput
            传感器输出数据。
        """
        if not self._initialized:
            raise RuntimeError(f"[{self.name}] 组件未初始化，请先调用 initialize()")

        self._frame_id += 1
        now = time.time()

        # 丢帧模拟
        if np.random.random() < self.dropout_probability:
            output = SensorOutput(
                frame=np.zeros((self.frame_height, self.frame_width), dtype=np.uint8),
                timestamp=now,
                frame_id=self._frame_id,
                is_valid=False,
                exposure_time_ms=self.exposure_time_ms,
            )
            self._frame_buffer.append(output)
            self._last_frame_time = now
            LOGGER.debug(f"[{self.name}] 帧 #{self._frame_id} 丢帧")
            return output

        # 生成或使用外部帧
        if input_data is not None and isinstance(input_data, np.ndarray):
            frame = input_data.copy()
            if frame.shape != (self.frame_height, self.frame_width):
                # 尝试调整大小
                if _HAS_CV2:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height),
                                       interpolation=cv2.INTER_LINEAR)
                else:
                    # 简单的最近邻插值
                    y_idx = np.linspace(0, frame.shape[0] - 1, self.frame_height).astype(int)
                    x_idx = np.linspace(0, frame.shape[1] - 1, self.frame_width).astype(int)
                    frame = frame[np.ix_(y_idx, x_idx)]
        else:
            # 生成模拟帧 (中心亮斑 + 背景)
            yy, xx = np.mgrid[0:self.frame_height, 0:self.frame_width]
            cx, cy = self.frame_width / 2.0, self.frame_height / 2.0
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            sigma = min(self.frame_width, self.frame_height) / 8.0
            frame = 200.0 * np.exp(-r2 / (2.0 * sigma ** 2))

        # 添加噪声
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, frame.shape)
            frame = frame + noise

        # 像素深度转换
        if self.pixel_depth == "uint8":
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame = frame.astype(np.float32)

        output = SensorOutput(
            frame=frame,
            timestamp=now,
            frame_id=self._frame_id,
            is_valid=True,
            exposure_time_ms=self.exposure_time_ms,
        )
        self._frame_buffer.append(output)
        self._last_frame_time = now
        return output

    def reset(self) -> None:
        """重置传感器状态。"""
        self._frame_id = 0
        self._frame_buffer.clear()
        self._last_frame_time = 0.0
        LOGGER.info(f"[{self.name}] 已重置")

    def get_status(self) -> Dict[str, Any]:
        """获取传感器状态。

        Returns
        -------
        Dict[str, Any]
            包含帧计数、缓冲区大小、初始化状态等。
        """
        return {
            "name": self.name,
            "version": self.version,
            "initialized": self._initialized,
            "frame_count": self._frame_id,
            "buffer_size": len(self._frame_buffer),
            "last_frame_time": self._last_frame_time,
            "frame_size": f"{self.frame_width}x{self.frame_height}",
        }


# ======================== 图像处理组件 ========================


@dataclass
class ProcessingOutput:
    """图像处理输出数据。

    Attributes
    ----------
    centroid_x : float
        光斑质心 X 坐标 (像素)。
    centroid_y : float
        光斑质心 Y 坐标 (像素)。
    spot_intensity : float
        光斑总强度。
    spot_peak : float
        光斑峰值强度。
    background_mean : float
        背景均值。
    spot_detected : bool
        是否检测到光斑。
    snr : float
        信噪比。
    raw_frame : np.ndarray
        原始帧 (用于可视化)。
    processed_frame : np.ndarray
        处理后的帧 (用于可视化)。
    """
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    spot_intensity: float = 0.0
    spot_peak: float = 0.0
    background_mean: float = 0.0
    spot_detected: bool = False
    snr: float = 0.0
    raw_frame: np.ndarray = field(default_factory=lambda: np.zeros((64, 64), dtype=np.uint8))
    processed_frame: np.ndarray = field(default_factory=lambda: np.zeros((64, 64), dtype=np.uint8))


class ProcessingComponent:
    """图像处理组件。

    对传感器帧进行:
    - 背景减除 (滑动平均或固定阈值)
    - 光斑检测 (阈值法)
    - 质心提取 (加权质心法)
    - 信噪比计算

    Parameters
    ----------
    background_method : str
        背景估计方法: 'median' (中值滤波) 或 'threshold' (阈值法)。
    detection_threshold_sigma : float
        检测阈值 (以背景标准差的倍数表示)。
    enable_background_subtraction : bool
        是否启用背景减除。
    min_spot_pixels : int
        最小光斑像素数。
    """

    def __init__(
        self,
        background_method: str = "median",
        detection_threshold_sigma: float = 3.0,
        enable_background_subtraction: bool = True,
        min_spot_pixels: int = 4,
    ):
        self.name: str = "ProcessingComponent"
        self.version: str = "1.0.0"
        self.description: str = "图像处理组件: 光斑检测与质心提取"

        self.background_method = background_method
        self.detection_threshold_sigma = float(detection_threshold_sigma)
        self.enable_background_subtraction = enable_background_subtraction
        self.min_spot_pixels = int(min_spot_pixels)

        self._initialized: bool = False
        self._background: Optional[np.ndarray] = None
        self._frame_count: int = 0

        # 背景估计缓冲
        self._bg_buffer_size: int = 10
        self._bg_buffer: Deque[np.ndarray] = deque(maxlen=self._bg_buffer_size)

    def initialize(self) -> None:
        """初始化处理组件。"""
        self._initialized = True
        self._background = None
        self._frame_count = 0
        self._bg_buffer.clear()
        LOGGER.info(f"[{self.name}] 初始化完成: 背景方法={self.background_method}, "
                     f"检测阈值={self.detection_threshold_sigma}σ")

    def process(self, input_data: Any) -> ProcessingOutput:
        """处理传感器帧。

        Parameters
        ----------
        input_data : SensorOutput
            传感器输出数据。

        Returns
        -------
        ProcessingOutput
            处理结果 (质心、强度、SNR 等)。
        """
        if not self._initialized:
            raise RuntimeError(f"[{self.name}] 组件未初始化，请先调用 initialize()")

        if not isinstance(input_data, SensorOutput):
            raise TypeError(f"[{self.name}] 输入数据类型错误，期望 SensorOutput，"
                            f" 实际为 {type(input_data).__name__}")

        sensor_output: SensorOutput = input_data
        frame = sensor_output.frame.astype(np.float64)

        # 无效帧直接返回
        if not sensor_output.is_valid:
            return ProcessingOutput(
                raw_frame=sensor_output.frame,
                processed_frame=sensor_output.frame,
            )

        self._frame_count += 1
        h, w = frame.shape

        # ---- 背景估计 ----
        if self.enable_background_subtraction:
            self._bg_buffer.append(frame.copy())
            if len(self._bg_buffer) >= 3:
                # 使用中值作为背景估计
                bg_stack = np.stack(list(self._bg_buffer), axis=0)
                self._background = np.median(bg_stack, axis=0)
            else:
                self._background = np.zeros_like(frame)

            processed = frame - self._background
            bg_mean = float(np.mean(self._background))
            bg_std = float(np.std(self._background)) + 1e-10
        else:
            processed = frame.copy()
            bg_mean = float(np.mean(frame))
            bg_std = float(np.std(frame)) + 1e-10

        # ---- 阈值分割 ----
        threshold = bg_mean + self.detection_threshold_sigma * bg_std
        binary_mask = processed > threshold

        # ---- 质心计算 (加权质心法) ----
        spot_pixels = np.sum(binary_mask)
        if spot_pixels >= self.min_spot_pixels:
            weighted_frame = np.maximum(processed, 0) * binary_mask
            total_weight = np.sum(weighted_frame) + 1e-10
            yy, xx = np.mgrid[0:h, 0:w]
            centroid_x = float(np.sum(xx * weighted_frame) / total_weight)
            centroid_y = float(np.sum(yy * weighted_frame) / total_weight)
            spot_intensity = float(np.sum(processed[binary_mask]))
            spot_peak = float(np.max(processed))
            spot_detected = True
        else:
            centroid_x = w / 2.0
            centroid_y = h / 2.0
            spot_intensity = 0.0
            spot_peak = float(np.max(processed))
            spot_detected = False

        # ---- SNR 计算 ----
        if bg_std > 1e-10:
            snr = spot_peak / bg_std
        else:
            snr = 0.0

        # 处理后帧 (用于可视化)
        vis_frame = np.clip(processed, 0, 255).astype(np.uint8) if self.pixel_depth == "uint8" else processed.astype(np.float32)

        return ProcessingOutput(
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            spot_intensity=spot_intensity,
            spot_peak=spot_peak,
            background_mean=bg_mean,
            spot_detected=spot_detected,
            snr=snr,
            raw_frame=sensor_output.frame,
            processed_frame=vis_frame,
        )

    @property
    def pixel_depth(self) -> str:
        """当前像素深度 (兼容属性)。"""
        return "uint8"

    def reset(self) -> None:
        """重置处理组件状态。"""
        self._background = None
        self._frame_count = 0
        self._bg_buffer.clear()
        LOGGER.info(f"[{self.name}] 已重置")

    def get_status(self) -> Dict[str, Any]:
        """获取处理组件状态。

        Returns
        -------
        Dict[str, Any]
            包含帧计数、背景状态等。
        """
        return {
            "name": self.name,
            "version": self.version,
            "initialized": self._initialized,
            "frames_processed": self._frame_count,
            "background_available": self._background is not None,
            "bg_buffer_size": len(self._bg_buffer),
        }


# ======================== 控制组件 ========================


@dataclass
class ControlOutput:
    """控制输出数据。

    Attributes
    ----------
    command_x : float
        X 方向控制命令。
    command_y : float
        Y 方向控制命令。
    error_x : float
        X 方向当前误差。
    error_y : float
        Y 方向当前误差。
    integral_x : float
        X 方向积分项。
    integral_y : float
        Y 方向积分项。
    is_saturated : bool
        输出是否饱和。
    mode : str
        当前控制器模式。
    """
    command_x: float = 0.0
    command_y: float = 0.0
    error_x: float = 0.0
    error_y: float = 0.0
    integral_x: float = 0.0
    integral_y: float = 0.0
    is_saturated: bool = False
    mode: str = "PID"


class ControlComponent:
    """控制组件 (PID/PI/P)。

    实现 PID 控制律，支持:
    - P / PI / PID 三种控制器类型
    - 积分抗饱和 (Anti-Windup)
    - 输出速率限制
    - 无扰动切换 (Bumpless Transfer)
    - 独立的 X/Y 双轴控制

    PID 控制律:
        u(t) = Kp * e(t) + Ki * integral(e) + Kd * d(e)/dt

    Parameters
    ----------
    controller_type : ControllerType or str
        控制器类型: 'P', 'PI', 'PID'。
    kp : float
        比例增益。
    ki : float
        积分增益。
    kd : float
        微分增益。
    output_limit : float
        输出限幅 (绝对值)。
    integral_limit : float
        积分项限幅 (防止积分饱和)。
    rate_limit : float
        输出速率限制 (每步最大变化量)。
    derivative_filter_alpha : float
        微分项低通滤波系数 (0~1)，越小滤波越强。
    setpoint_x : float
        X 方向目标位置 (像素)。
    setpoint_y : float
        Y 方向目标位置 (像素)。
    """

    def __init__(
        self,
        controller_type: Any = ControllerType.PID,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.01,
        output_limit: float = 100.0,
        integral_limit: float = 50.0,
        rate_limit: float = 20.0,
        derivative_filter_alpha: float = 0.1,
        setpoint_x: float = 32.0,
        setpoint_y: float = 32.0,
    ):
        self.name: str = "ControlComponent"
        self.version: str = "1.0.0"
        self.description: str = "控制组件: PID/PI/P 控制律"

        # 解析控制器类型
        if isinstance(controller_type, str):
            self._controller_type = ControllerType(controller_type.upper())
        else:
            self._controller_type = controller_type

        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = float(output_limit)
        self.integral_limit = float(integral_limit)
        self.rate_limit = float(rate_limit)
        self.derivative_filter_alpha = float(derivative_filter_alpha)
        self.setpoint_x = float(setpoint_x)
        self.setpoint_y = float(setpoint_y)

        self._initialized: bool = False

        # 控制器状态
        self._integral_x: float = 0.0
        self._integral_y: float = 0.0
        self._prev_error_x: float = 0.0
        self._prev_error_y: float = 0.0
        self._prev_output_x: float = 0.0
        self._prev_output_y: float = 0.0
        self._filtered_derivative_x: float = 0.0
        self._filtered_derivative_y: float = 0.0
        self._step_count: int = 0

    def initialize(self) -> None:
        """初始化控制组件。"""
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._prev_output_x = 0.0
        self._prev_output_y = 0.0
        self._filtered_derivative_x = 0.0
        self._filtered_derivative_y = 0.0
        self._step_count = 0
        self._initialized = True
        LOGGER.info(f"[{self.name}] 初始化完成: 类型={self._controller_type.value}, "
                     f"Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")

    def process(self, input_data: Any) -> ControlOutput:
        """执行一步控制计算。

        Parameters
        ----------
        input_data : ProcessingOutput
            处理组件的输出 (包含质心位置)。

        Returns
        -------
        ControlOutput
            控制输出 (X/Y 方向命令)。
        """
        if not self._initialized:
            raise RuntimeError(f"[{self.name}] 组件未初始化，请先调用 initialize()")

        if not isinstance(input_data, ProcessingOutput):
            raise TypeError(f"[{self.name}] 输入数据类型错误，期望 ProcessingOutput，"
                            f" 实际为 {type(input_data).__name__}")

        proc: ProcessingOutput = input_data
        self._step_count += 1

        # 误差计算 (负反馈: 误差 = 设定点 - 测量值)
        error_x = self.setpoint_x - proc.centroid_x
        error_y = self.setpoint_y - proc.centroid_y

        # ---- 比例项 ----
        p_x = self.kp * error_x
        p_y = self.kp * error_y

        # ---- 积分项 (带抗饱和) ----
        if self._controller_type in (ControllerType.PI, ControllerType.PID):
            self._integral_x += error_x
            self._integral_y += error_y

            # 积分限幅 (Anti-Windup)
            self._integral_x = np.clip(self._integral_x,
                                       -self.integral_limit, self.integral_limit)
            self._integral_y = np.clip(self._integral_y,
                                       -self.integral_limit, self.integral_limit)

            i_x = self.ki * self._integral_x
            i_y = self.ki * self._integral_y
        else:
            i_x = 0.0
            i_y = 0.0

        # ---- 微分项 (带低通滤波) ----
        if self._controller_type == ControllerType.PID:
            raw_deriv_x = error_x - self._prev_error_x
            raw_deriv_y = error_y - self._prev_error_y

            # 一阶低通滤波
            alpha = self.derivative_filter_alpha
            self._filtered_derivative_x = alpha * raw_deriv_x + \
                (1.0 - alpha) * self._filtered_derivative_x
            self._filtered_derivative_y = alpha * raw_deriv_y + \
                (1.0 - alpha) * self._filtered_derivative_y

            d_x = self.kd * self._filtered_derivative_x
            d_y = self.kd * self._filtered_derivative_y
        else:
            d_x = 0.0
            d_y = 0.0

        # ---- 合成输出 ----
        raw_output_x = p_x + i_x + d_x
        raw_output_y = p_y + i_y + d_y

        # ---- 输出限幅 ----
        output_x = np.clip(raw_output_x, -self.output_limit, self.output_limit)
        output_y = np.clip(raw_output_y, -self.output_limit, self.output_limit)

        # ---- 速率限制 (Bumpless Transfer) ----
        delta_x = output_x - self._prev_output_x
        delta_y = output_y - self._prev_output_y
        delta_x = np.clip(delta_x, -self.rate_limit, self.rate_limit)
        delta_y = np.clip(delta_y, -self.rate_limit, self.rate_limit)
        output_x = self._prev_output_x + delta_x
        output_y = self._prev_output_y + delta_y

        is_saturated = (abs(raw_output_x) > self.output_limit or
                        abs(raw_output_y) > self.output_limit)

        # 更新状态
        self._prev_error_x = error_x
        self._prev_error_y = error_y
        self._prev_output_x = output_x
        self._prev_output_y = output_y

        return ControlOutput(
            command_x=output_x,
            command_y=output_y,
            error_x=error_x,
            error_y=error_y,
            integral_x=self._integral_x,
            integral_y=self._integral_y,
            is_saturated=is_saturated,
            mode=self._controller_type.value,
        )

    def set_setpoint(self, x: float, y: float) -> None:
        """设置目标位置。

        Parameters
        ----------
        x, y : float
            目标 X/Y 坐标 (像素)。
        """
        self.setpoint_x = float(x)
        self.setpoint_y = float(y)
        LOGGER.info(f"[{self.name}] 目标位置更新: ({self.setpoint_x:.2f}, {self.setpoint_y:.2f})")

    def set_gains(self, kp: Optional[float] = None, ki: Optional[float] = None,
                  kd: Optional[float] = None) -> None:
        """在线调整 PID 增益 (无扰动切换)。

        Parameters
        ----------
        kp, ki, kd : float or None
            新的增益值。None 表示保持不变。
        """
        if kp is not None:
            self.kp = float(kp)
        if ki is not None:
            self.ki = float(ki)
        if kd is not None:
            self.kd = float(kd)
        LOGGER.info(f"[{self.name}] 增益更新: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")

    def switch_controller_type(self, new_type: Any) -> None:
        """切换控制器类型 (无扰动切换)。

        切换时保留当前积分状态，避免输出跳变。

        Parameters
        ----------
        new_type : ControllerType or str
            新的控制器类型。
        """
        if isinstance(new_type, str):
            new_type = ControllerType(new_type.upper())
        old_type = self._controller_type
        self._controller_type = new_type

        # 切换到无积分模式时清零积分
        if new_type == ControllerType.P:
            self._integral_x = 0.0
            self._integral_y = 0.0

        LOGGER.info(f"[{self.name}] 控制器类型切换: {old_type.value} -> {new_type.value}")

    def reset(self) -> None:
        """重置控制器状态。"""
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._prev_output_x = 0.0
        self._prev_output_y = 0.0
        self._filtered_derivative_x = 0.0
        self._filtered_derivative_y = 0.0
        self._step_count = 0
        LOGGER.info(f"[{self.name}] 已重置")

    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态。

        Returns
        -------
        Dict[str, Any]
            包含增益、积分状态、步数等。
        """
        return {
            "name": self.name,
            "version": self.version,
            "initialized": self._initialized,
            "controller_type": self._controller_type.value,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "setpoint": (self.setpoint_x, self.setpoint_y),
            "integral": (self._integral_x, self._integral_y),
            "last_output": (self._prev_output_x, self._prev_output_y),
            "step_count": self._step_count,
        }


# ======================== 校正组件 ========================


@dataclass
class CorrectionOutput:
    """校正输出数据。

    Attributes
    ----------
    motor_command_x : float
        X 方向电机命令 (步数或电压)。
    motor_command_y : float
        Y 方向电机命令 (步数或电压)。
    raw_command_x : float
        X 方向原始控制命令。
    raw_command_y : float
        Y 方向原始控制命令。
    is_applied : bool
        命令是否已应用。
    is_in_deadband : bool
        命令是否在死区内。
    buffer_size : int
        命令缓冲区剩余量。
    """
    motor_command_x: float = 0.0
    motor_command_y: float = 0.0
    raw_command_x: float = 0.0
    raw_command_y: float = 0.0
    is_applied: bool = False
    is_in_deadband: bool = False
    buffer_size: int = 0


class CorrectorComponent:
    """校正组件 (电机命令生成)。

    将控制命令转换为电机执行命令，支持:
    - 命令缩放和符号校正
    - 步长限制
    - 死区滤波 (小命令不执行)
    - 命令缓冲

    Parameters
    ----------
    scale_x : float
        X 方向缩放因子。
    scale_y : float
        Y 方向缩放因子。
    sign_x : float
        X 方向符号校正 (+1 或 -1)。
    sign_y : float
        Y 方向符号校正 (+1 或 -1)。
    max_step : float
        单步最大步数。
    deadband : float
        死区范围 (绝对值小于此值不执行)。
    buffer_size : int
        命令缓冲区大小。
    """

    def __init__(
        self,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        sign_x: float = 1.0,
        sign_y: float = 1.0,
        max_step: float = 50.0,
        deadband: float = 0.5,
        buffer_size: int = 10,
    ):
        self.name: str = "CorrectorComponent"
        self.version: str = "1.0.0"
        self.description: str = "校正组件: 电机命令生成与限幅"

        self.scale_x = float(scale_x)
        self.scale_y = float(scale_y)
        self.sign_x = float(sign_x)
        self.sign_y = float(sign_y)
        self.max_step = float(max_step)
        self.deadband = float(deadband)
        self._buffer_capacity = int(buffer_size)

        self._initialized: bool = False
        self._command_buffer: Deque[Tuple[float, float]] = deque(maxlen=self._buffer_capacity)
        self._total_steps_x: float = 0.0
        self._total_steps_y: float = 0.0
        self._command_count: int = 0

    def initialize(self) -> None:
        """初始化校正组件。"""
        self._initialized = True
        self._command_buffer.clear()
        self._total_steps_x = 0.0
        self._total_steps_y = 0.0
        self._command_count = 0
        LOGGER.info(f"[{self.name}] 初始化完成: 缩放=({self.scale_x}, {self.scale_y}), "
                     f"符号=({self.sign_x}, {self.sign_y}), 最大步长={self.max_step}, "
                     f"死区={self.deadband}")

    def process(self, input_data: Any) -> CorrectionOutput:
        """将控制命令转换为电机命令。

        Parameters
        ----------
        input_data : ControlOutput
            控制组件的输出。

        Returns
        -------
        CorrectionOutput
            校正输出 (电机命令)。
        """
        if not self._initialized:
            raise RuntimeError(f"[{self.name}] 组件未初始化，请先调用 initialize()")

        if not isinstance(input_data, ControlOutput):
            raise TypeError(f"[{self.name}] 输入数据类型错误，期望 ControlOutput，"
                            f" 实际为 {type(input_data).__name__}")

        ctrl: ControlOutput = input_data
        self._command_count += 1

        # ---- 缩放和符号校正 ----
        scaled_x = ctrl.command_x * self.scale_x * self.sign_x
        scaled_y = ctrl.command_y * self.scale_y * self.sign_y

        # ---- 死区滤波 ----
        abs_x = abs(scaled_x)
        abs_y = abs(scaled_y)
        in_deadband = (abs_x < self.deadband) and (abs_y < self.deadband)

        if in_deadband:
            motor_x = 0.0
            motor_y = 0.0
            is_applied = False
        else:
            # ---- 步长限制 ----
            motor_x = np.clip(scaled_x, -self.max_step, self.max_step)
            motor_y = np.clip(scaled_y, -self.max_step, self.max_step)
            is_applied = True

            # 缓冲命令
            self._command_buffer.append((motor_x, motor_y))
            self._total_steps_x += abs(motor_x)
            self._total_steps_y += abs(motor_y)

        return CorrectionOutput(
            motor_command_x=motor_x,
            motor_command_y=motor_y,
            raw_command_x=ctrl.command_x,
            raw_command_y=ctrl.command_y,
            is_applied=is_applied,
            is_in_deadband=in_deadband,
            buffer_size=len(self._command_buffer),
        )

    def reset(self) -> None:
        """重置校正组件状态。"""
        self._command_buffer.clear()
        self._total_steps_x = 0.0
        self._total_steps_y = 0.0
        self._command_count = 0
        LOGGER.info(f"[{self.name}] 已重置")

    def get_status(self) -> Dict[str, Any]:
        """获取校正组件状态。

        Returns
        -------
        Dict[str, Any]
            包含缓冲区大小、累计步数等。
        """
        return {
            "name": self.name,
            "version": self.version,
            "initialized": self._initialized,
            "command_count": self._command_count,
            "buffer_remaining": self._buffer_capacity - len(self._command_buffer),
            "total_steps": (self._total_steps_x, self._total_steps_y),
            "scale": (self.scale_x, self.scale_y),
            "sign": (self.sign_x, self.sign_y),
        }


# ======================== 遥测记录器 ========================


class TelemetryRecorder:
    """管线遥测记录器。

    记录:
    - 各组件处理时间统计
    - 循环速率测量 (均值、最小、最大、标准差、P99)
    - 时序抖动分析
    - 延迟分解

    Parameters
    ----------
    buffer_size : int
        遥测数据缓冲区大小 (样本数)。
    """

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = int(buffer_size)

        # 各组件处理时间记录 (秒)
        self._component_times: Dict[str, Deque[float]] = {}
        # 各组件超时计数
        self._component_timeouts: Dict[str, int] = {}
        # 各组件错误计数
        self._component_errors: Dict[str, int] = {}

        # 循环周期记录 (秒)
        self._loop_periods: Deque[float] = deque(maxlen=self.buffer_size)
        # 循环时间戳记录
        self._loop_timestamps: Deque[float] = deque(maxlen=self.buffer_size)

        # 延迟分解记录
        self._sensor_times: Deque[float] = deque(maxlen=self.buffer_size)
        self._processing_times: Deque[float] = deque(maxlen=self.buffer_size)
        self._control_times: Deque[float] = deque(maxlen=self.buffer_size)
        self._correction_times: Deque[float] = deque(maxlen=self.buffer_size)
        self._overhead_times: Deque[float] = deque(maxlen=self.buffer_size)

        # 总帧计数
        self._total_frames: int = 0
        # 总错误计数
        self._total_errors: int = 0
        # 启动时间
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """启动遥测记录。"""
        self._start_time = time.time()
        self._total_frames = 0
        self._total_errors = 0
        self._loop_periods.clear()
        self._loop_timestamps.clear()
        self._sensor_times.clear()
        self._processing_times.clear()
        self._control_times.clear()
        self._correction_times.clear()
        self._overhead_times.clear()
        self._component_times.clear()
        self._component_timeouts.clear()
        self._component_errors.clear()

    def register_component(self, name: str) -> None:
        """注册一个组件用于时序跟踪。

        Parameters
        ----------
        name : str
            组件名称。
        """
        if name not in self._component_times:
            self._component_times[name] = deque(maxlen=self.buffer_size)
            self._component_timeouts[name] = 0
            self._component_errors[name] = 0

    def record_component_time(self, name: str, elapsed_s: float) -> None:
        """记录组件处理时间。

        Parameters
        ----------
        name : str
            组件名称。
        elapsed_s : float
            处理耗时 (秒)。
        """
        if name not in self._component_times:
            self.register_component(name)
        self._component_times[name].append(elapsed_s)

    def record_component_error(self, name: str) -> None:
        """记录组件错误。

        Parameters
        ----------
        name : str
            组件名称。
        """
        self._component_errors[name] = self._component_errors.get(name, 0) + 1
        self._total_errors += 1

    def record_component_timeout(self, name: str) -> None:
        """记录组件超时。

        Parameters
        ----------
        name : str
            组件名称。
        """
        self._component_timeouts[name] = self._component_timeouts.get(name, 0) + 1

    def record_loop_period(self, period_s: float) -> None:
        """记录一个循环周期。

        Parameters
        ----------
        period_s : float
            循环周期 (秒)。
        """
        self._loop_periods.append(period_s)
        self._loop_timestamps.append(time.time())

    def record_latency_breakdown(
        self,
        sensor_s: float,
        processing_s: float,
        control_s: float,
        correction_s: float,
        overhead_s: float,
    ) -> None:
        """记录延迟分解。

        Parameters
        ----------
        sensor_s, processing_s, control_s, correction_s, overhead_s : float
            各阶段耗时 (秒)。
        """
        self._sensor_times.append(sensor_s)
        self._processing_times.append(processing_s)
        self._control_times.append(control_s)
        self._correction_times.append(correction_s)
        self._overhead_times.append(overhead_s)
        self._total_frames += 1

    def get_component_stats(self, name: str) -> ComponentTimingStats:
        """获取组件时序统计。

        Parameters
        ----------
        name : str
            组件名称。

        Returns
        -------
        ComponentTimingStats
            组件时序统计结果。
        """
        times = self._component_times.get(name, deque())
        if len(times) == 0:
            return ComponentTimingStats(name=name)

        arr = np.array(times) * 1000.0  # 转换为毫秒
        return ComponentTimingStats(
            name=name,
            call_count=len(arr),
            mean_ms=float(np.mean(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            std_ms=float(np.std(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p99_ms=float(np.percentile(arr, 99)),
            timeout_count=self._component_timeouts.get(name, 0),
            error_count=self._component_errors.get(name, 0),
        )

    def get_all_component_stats(self) -> Dict[str, ComponentTimingStats]:
        """获取所有组件的时序统计。

        Returns
        -------
        Dict[str, ComponentTimingStats]
            各组件名称到统计结果的映射。
        """
        return {name: self.get_component_stats(name) for name in self._component_times}

    def get_loop_rate_stats(self, target_hz: float = 10.0) -> LoopRateStats:
        """获取循环速率统计。

        Parameters
        ----------
        target_hz : float
            目标频率 (Hz)。

        Returns
        -------
        LoopRateStats
            循环速率统计结果。
        """
        if len(self._loop_periods) == 0:
            return LoopRateStats(target_hz=target_hz)

        periods = np.array(self._loop_periods)
        frequencies = 1.0 / (periods + 1e-15)  # 避免除零

        return LoopRateStats(
            mean_hz=float(np.mean(frequencies)),
            min_hz=float(np.min(frequencies)),
            max_hz=float(np.max(frequencies)),
            std_hz=float(np.std(frequencies)),
            p99_hz=float(np.percentile(frequencies, 1)),  # P99 最差情况 = 频率最低的 1%
            target_hz=target_hz,
            samples=len(frequencies),
        )

    def get_timing_jitter_ms(self) -> float:
        """获取时序抖动 (循环周期标准差，毫秒)。

        Returns
        -------
        float
            时序抖动 (毫秒)。
        """
        if len(self._loop_periods) < 2:
            return 0.0
        return float(np.std(list(self._loop_periods)) * 1000.0)

    def get_latency_breakdown(self) -> LatencyBreakdown:
        """获取平均延迟分解。

        Returns
        -------
        LatencyBreakdown
            延迟分解结果 (毫秒)。
        """
        def _mean_ms(d: Deque[float]) -> float:
            if len(d) == 0:
                return 0.0
            return float(np.mean(list(d)) * 1000.0)

        sensor = _mean_ms(self._sensor_times)
        processing = _mean_ms(self._processing_times)
        control = _mean_ms(self._control_times)
        correction = _mean_ms(self._correction_times)
        overhead = _mean_ms(self._overhead_times)

        return LatencyBreakdown(
            sensor_ms=sensor,
            processing_ms=processing,
            control_ms=control,
            correction_ms=correction,
            total_ms=sensor + processing + control + correction + overhead,
            overhead_ms=overhead,
        )

    def get_uptime_s(self) -> float:
        """获取运行时间 (秒)。

        Returns
        -------
        float
            运行时间 (秒)。
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_total_frames(self) -> int:
        """获取总处理帧数。

        Returns
        -------
        int
            总帧数。
        """
        return self._total_frames

    def get_total_errors(self) -> int:
        """获取总错误数。

        Returns
        -------
        int
            总错误数。
        """
        return self._total_errors

    def export_data(self) -> Dict[str, Any]:
        """导出遥测数据为字典。

        Returns
        -------
        Dict[str, Any]
            包含所有遥测数据的字典。
        """
        return {
            "total_frames": self._total_frames,
            "total_errors": self._total_errors,
            "uptime_s": self.get_uptime_s(),
            "loop_periods_ms": [p * 1000.0 for p in self._loop_periods],
            "component_times_ms": {
                name: [t * 1000.0 for t in times]
                for name, times in self._component_times.items()
            },
            "latency_breakdown": {
                "sensor_ms": [t * 1000.0 for t in self._sensor_times],
                "processing_ms": [t * 1000.0 for t in self._processing_times],
                "control_ms": [t * 1000.0 for t in self._control_times],
                "correction_ms": [t * 1000.0 for t in self._correction_times],
                "overhead_ms": [t * 1000.0 for t in self._overhead_times],
            },
        }


# ======================== 实时控制管线 ========================


class RealtimeControlPipeline:
    """实时控制管线主编排器。

    组件化 AO 控制管线，架构:
        Sensor -> Processor -> Controller -> Corrector

    支持两种运行模式:
    - 软实时 (soft): 单线程顺序执行，适用于开发/调试
    - 硬实时仿真 (hard): 多线程 + 共享内存 IPC 模拟

    Parameters
    ----------
    config : PipelineConfig or None
        管线配置参数。为 None 时使用默认值。
    sensor : SensorComponent or None
        传感器组件。为 None 时使用默认模拟传感器。
    processor : ProcessingComponent or None
        处理组件。为 None 时使用默认处理组件。
    controller : ControlComponent or None
        控制组件。为 None 时使用默认 PID 控制器。
    corrector : CorrectorComponent or None
        校正组件。为 None 时使用默认校正组件。
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        sensor: Optional[SensorComponent] = None,
        processor: Optional[ProcessingComponent] = None,
        controller: Optional[ControlComponent] = None,
        corrector: Optional[CorrectorComponent] = None,
    ):
        self.config = config or PipelineConfig()
        self._state: PipelineState = PipelineState.IDLE

        # 组件
        self._sensor = sensor or SensorComponent()
        self._processor = processor or ProcessingComponent()
        self._controller = controller or ControlComponent()
        self._corrector = corrector or CorrectorComponent()

        # 遥测
        self._telemetry = TelemetryRecorder(
            buffer_size=self.config.telemetry_buffer_size
        )

        # 线程控制
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._step_event = threading.Event()
        self._lock = threading.Lock()

        # 共享内存模拟 (hard 模式)
        self._shared_sensor_output: Optional[SensorOutput] = None
        self._shared_processing_output: Optional[ProcessingOutput] = None
        self._shared_control_output: Optional[ControlOutput] = None
        self._shared_correction_output: Optional[CorrectionOutput] = None

        # 运行统计
        self._loop_count: int = 0
        self._error_count: int = 0
        self._consecutive_errors: int = 0
        self._start_time: Optional[float] = None
        self._last_loop_time: Optional[float] = None

        # 最新输出 (供外部读取)
        self._latest_sensor: Optional[SensorOutput] = None
        self._latest_processing: Optional[ProcessingOutput] = None
        self._latest_control: Optional[ControlOutput] = None
        self._latest_correction: Optional[CorrectionOutput] = None

        # 回调
        self._on_error: Optional[callable] = None
        self._on_correction: Optional[callable] = None

        LOGGER.info(f"[RealtimeControlPipeline] 创建完成, 模式={self.config.mode}, "
                     f"目标频率={self.config.target_frequency_hz}Hz")

    # ---- 属性 ----

    @property
    def state(self) -> PipelineState:
        """当前管线状态。"""
        return self._state

    @property
    def sensor(self) -> SensorComponent:
        """传感器组件。"""
        return self._sensor

    @property
    def processor(self) -> ProcessingComponent:
        """处理组件。"""
        return self._processor

    @property
    def controller(self) -> ControlComponent:
        """控制组件。"""
        return self._controller

    @property
    def corrector(self) -> CorrectorComponent:
        """校正组件。"""
        return self._corrector

    @property
    def telemetry(self) -> TelemetryRecorder:
        """遥测记录器。"""
        return self._telemetry

    @property
    def latest_correction(self) -> Optional[CorrectionOutput]:
        """最新的校正输出。"""
        return self._latest_correction

    # ---- 组件热替换 ----

    def swap_sensor(self, new_sensor: SensorComponent) -> None:
        """热替换传感器组件。

        Parameters
        ----------
        new_sensor : SensorComponent
            新的传感器组件。

        Raises
        ------
        RuntimeError
            管线正在运行时不允许替换。
        """
        with self._lock:
            if self._state == PipelineState.RUNNING:
                raise RuntimeError("管线运行中不允许替换组件，请先暂停或停止")
            new_sensor.initialize()
            self._sensor = new_sensor
            self._telemetry.register_component(new_sensor.name)
            LOGGER.info(f"[Pipeline] 传感器组件已替换: {new_sensor.name}")

    def swap_processor(self, new_processor: ProcessingComponent) -> None:
        """热替换处理组件。

        Parameters
        ----------
        new_processor : ProcessingComponent
            新的处理组件。

        Raises
        ------
        RuntimeError
            管线正在运行时不允许替换。
        """
        with self._lock:
            if self._state == PipelineState.RUNNING:
                raise RuntimeError("管线运行中不允许替换组件，请先暂停或停止")
            new_processor.initialize()
            self._processor = new_processor
            self._telemetry.register_component(new_processor.name)
            LOGGER.info(f"[Pipeline] 处理组件已替换: {new_processor.name}")

    def swap_controller(self, new_controller: ControlComponent) -> None:
        """热替换控制组件。

        Parameters
        ----------
        new_controller : ControlComponent
            新的控制组件。

        Raises
        ------
        RuntimeError
            管线正在运行时不允许替换。
        """
        with self._lock:
            if self._state == PipelineState.RUNNING:
                raise RuntimeError("管线运行中不允许替换组件，请先暂停或停止")
            new_controller.initialize()
            self._controller = new_controller
            self._telemetry.register_component(new_controller.name)
            LOGGER.info(f"[Pipeline] 控制组件已替换: {new_controller.name}")

    def swap_corrector(self, new_corrector: CorrectorComponent) -> None:
        """热替换校正组件。

        Parameters
        ----------
        new_corrector : CorrectorComponent
            新的校正组件。

        Raises
        ------
        RuntimeError
            管线正在运行时不允许替换。
        """
        with self._lock:
            if self._state == PipelineState.RUNNING:
                raise RuntimeError("管线运行中不允许替换组件，请先暂停或停止")
            new_corrector.initialize()
            self._corrector = new_corrector
            self._telemetry.register_component(new_corrector.name)
            LOGGER.info(f"[Pipeline] 校正组件已替换: {new_corrector.name}")

    # ---- 回调设置 ----

    def set_error_callback(self, callback: Optional[callable]) -> None:
        """设置错误回调。

        Parameters
        ----------
        callback : callable or None
            回调函数，接收 (error: Exception, component_name: str) 参数。
        """
        self._on_error = callback

    def set_correction_callback(self, callback: Optional[callable]) -> None:
        """设置校正输出回调。

        Parameters
        ----------
        callback : callable or None
            回调函数，接收 CorrectionOutput 参数。
        """
        self._on_correction = callback

    # ---- 管线生命周期 ----

    def initialize(self) -> None:
        """初始化管线及所有组件。"""
        LOGGER.info("[Pipeline] 正在初始化管线...")
        self._sensor.initialize()
        self._processor.initialize()
        self._controller.initialize()
        self._corrector.initialize()

        # 注册遥测组件
        self._telemetry.register_component(self._sensor.name)
        self._telemetry.register_component(self._processor.name)
        self._telemetry.register_component(self._controller.name)
        self._telemetry.register_component(self._corrector.name)

        self._state = PipelineState.IDLE
        LOGGER.info("[Pipeline] 管线初始化完成")

    def start(self) -> None:
        """启动管线。

        Raises
        ------
        RuntimeError
            管线未初始化或已在运行。
        """
        if self._state == PipelineState.RUNNING:
            raise RuntimeError("管线已在运行中")
        if self._state != PipelineState.IDLE and self._state != PipelineState.STOPPED:
            raise RuntimeError(f"管线当前状态 {self._state.value} 不允许启动")

        self._stop_event.clear()
        self._pause_event.set()  # 默认不暂停
        self._step_event.clear()

        # 重置统计
        self._loop_count = 0
        self._error_count = 0
        self._consecutive_errors = 0
        self._start_time = time.time()
        self._last_loop_time = None

        # 启动遥测
        if self.config.enable_telemetry:
            self._telemetry.start()

        self._state = PipelineState.RUNNING

        if self.config.mode == "soft":
            # 软实时: 在当前线程中运行
            LOGGER.info("[Pipeline] 软实时模式启动")
        else:
            # 硬实时仿真: 启动后台线程
            self._thread = threading.Thread(
                target=self._hard_rtc_loop,
                name="RTC-Pipeline",
                daemon=True,
            )
            self._thread.start()
            LOGGER.info("[Pipeline] 硬实时仿真模式启动 (后台线程)")

    def stop(self) -> None:
        """停止管线。"""
        LOGGER.info("[Pipeline] 正在停止管线...")
        self._stop_event.set()
        self._pause_event.set()  # 解除暂停以允许线程退出

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                LOGGER.warning("[Pipeline] 管线线程未能在 5 秒内退出")
            self._thread = None

        self._state = PipelineState.STOPPED
        LOGGER.info(f"[Pipeline] 管线已停止, 共处理 {self._loop_count} 帧, "
                     f"错误 {self._error_count} 次")

    def pause(self) -> None:
        """暂停管线。"""
        if self._state != PipelineState.RUNNING:
            LOGGER.warning(f"[Pipeline] 当前状态 {self._state.value} 不允许暂停")
            return
        self._pause_event.clear()
        self._state = PipelineState.PAUSED
        LOGGER.info("[Pipeline] 管线已暂停")

    def resume(self) -> None:
        """恢复管线运行。"""
        if self._state != PipelineState.PAUSED:
            LOGGER.warning(f"[Pipeline] 当前状态 {self._state.value} 不允许恢复")
            return
        self._pause_event.set()
        self._state = PipelineState.RUNNING
        self._last_loop_time = time.time()  # 重置计时避免暂停时间影响
        LOGGER.info("[Pipeline] 管线已恢复")

    def step(self) -> bool:
        """单步执行一帧 (仅在暂停状态下有效)。

        Returns
        -------
        bool
            是否成功执行一步。
        """
        if self._state != PipelineState.PAUSED:
            LOGGER.warning("[Pipeline] 单步执行仅在暂停状态下有效")
            return False
        self._step_event.set()
        time.sleep(0.05)  # 等待一步完成
        self._step_event.clear()
        return True

    def run_one_cycle(self) -> Optional[CorrectionOutput]:
        """执行一个完整的控制循环 (软实时模式手动调用)。

        管线流程:
        1. Sensor: 采集帧
        2. Processor: 处理帧，提取质心
        3. Controller: 计算控制命令
        4. Corrector: 生成电机命令

        Returns
        -------
        CorrectionOutput or None
            校正输出。失败时返回 None。
        """
        if self._state != PipelineState.RUNNING and self._state != PipelineState.PAUSED:
            LOGGER.warning(f"[Pipeline] 当前状态 {self._state.value} 不允许执行循环")
            return None

        cycle_start = time.time()

        try:
            # ---- 1. 传感器采集 ----
            t0 = time.time()
            sensor_output = self._sensor.process()
            t_sensor = time.time() - t0

            # 帧延迟检查
            if self._last_loop_time is not None:
                frame_delay = sensor_output.timestamp - self._last_loop_time
                if frame_delay > self.config.max_frame_delay_s:
                    LOGGER.warning(f"[Pipeline] 帧延迟过大: {frame_delay:.3f}s > "
                                   f"{self.config.max_frame_delay_s:.3f}s")

            self._latest_sensor = sensor_output

            # ---- 2. 图像处理 ----
            t0 = time.time()
            processing_output = self._processor.process(sensor_output)
            t_processing = time.time() - t0

            self._latest_processing = processing_output

            # ---- 3. 控制计算 ----
            t0 = time.time()
            control_output = self._controller.process(processing_output)
            t_control = time.time() - t0

            self._latest_control = control_output

            # ---- 4. 校正执行 ----
            t0 = time.time()
            correction_output = self._corrector.process(control_output)
            t_correction = time.time() - t0

            self._latest_correction = correction_output

            # 共享内存更新 (hard 模式)
            self._shared_sensor_output = sensor_output
            self._shared_processing_output = processing_output
            self._shared_control_output = control_output
            self._shared_correction_output = correction_output

            # ---- 开销计算 ----
            cycle_end = time.time()
            t_total = cycle_end - cycle_start
            t_overhead = t_total - (t_sensor + t_processing + t_control + t_correction)
            t_overhead = max(t_overhead, 0.0)

            # ---- 遥测记录 ----
            if self.config.enable_telemetry:
                self._telemetry.record_component_time(self._sensor.name, t_sensor)
                self._telemetry.record_component_time(self._processor.name, t_processing)
                self._telemetry.record_component_time(self._controller.name, t_control)
                self._telemetry.record_component_time(self._corrector.name, t_correction)
                self._telemetry.record_latency_breakdown(
                    t_sensor, t_processing, t_control, t_correction, t_overhead
                )

            # ---- 循环周期记录 ----
            if self._last_loop_time is not None:
                period = cycle_start - self._last_loop_time
                if self.config.enable_telemetry:
                    self._telemetry.record_loop_period(period)

            self._last_loop_time = cycle_start
            self._loop_count += 1
            self._consecutive_errors = 0

            # 回调
            if self._on_correction is not None:
                try:
                    self._on_correction(correction_output)
                except Exception as cb_err:
                    LOGGER.error(f"[Pipeline] 校正回调异常: {cb_err}")

            return correction_output

        except Exception as e:
            self._error_count += 1
            self._consecutive_errors += 1

            # 记录错误组件
            if self.config.enable_telemetry:
                # 尝试确定哪个组件出错
                for comp_name in [self._sensor.name, self._processor.name,
                                  self._controller.name, self._corrector.name]:
                    self._telemetry.record_component_error(comp_name)

            # 错误回调
            if self._on_error is not None:
                try:
                    self._on_error(e, "pipeline")
                except Exception:
                    pass

            # 错误恢复
            if self._consecutive_errors >= self.config.error_recovery_retries:
                LOGGER.error(f"[Pipeline] 连续错误 {self._consecutive_errors} 次，"
                             f"超过重试上限 {self.config.error_recovery_retries}，"
                             f"管线进入 ERROR 状态")
                self._state = PipelineState.ERROR
                return None

            LOGGER.warning(f"[Pipeline] 循环错误 ({self._consecutive_errors}/"
                           f"{self.config.error_recovery_retries}): {e}")
            return None

    def run_soft(self, num_cycles: Optional[int] = None) -> None:
        """软实时模式: 在当前线程中运行管线。

        Parameters
        ----------
        num_cycles : int or None
            运行循环数。None 表示持续运行直到 stop() 被调用。
        """
        if self._state != PipelineState.RUNNING:
            raise RuntimeError("管线未运行，请先调用 start()")

        target_period = 1.0 / self.config.target_frequency_hz
        cycle_count = 0

        LOGGER.info(f"[Pipeline] 软实时运行开始, 目标周期={target_period*1000:.1f}ms, "
                     f"循环数={'无限' if num_cycles is None else num_cycles}")

        while not self._stop_event.is_set():
            # 暂停检查
            if not self._pause_event.is_set():
                # 暂停状态: 等待恢复或单步
                if self._step_event.is_set():
                    self._step_event.clear()
                    self.run_one_cycle()
                    cycle_count += 1
                time.sleep(0.01)
                continue

            cycle_start = time.time()
            self.run_one_cycle()
            cycle_count += 1

            # 检查循环数限制
            if num_cycles is not None and cycle_count >= num_cycles:
                LOGGER.info(f"[Pipeline] 已完成 {num_cycles} 个循环")
                break

            # 帧率控制
            elapsed = time.time() - cycle_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _hard_rtc_loop(self) -> None:
        """硬实时仿真循环 (在后台线程中运行)。

        模拟多线程流水线架构:
        - 主线程: 管线调度和时序控制
        - 共享内存: 组件间数据传递
        """
        target_period = 1.0 / self.config.target_frequency_hz

        LOGGER.info("[Pipeline] 硬实时仿真循环启动")

        while not self._stop_event.is_set():
            # 暂停检查
            if not self._pause_event.is_set():
                if self._step_event.is_set():
                    self._step_event.clear()
                    with self._lock:
                        self.run_one_cycle()
                time.sleep(0.001)
                continue

            cycle_start = time.time()

            with self._lock:
                self.run_one_cycle()

            # 帧率控制 (更精确的 sleep)
            elapsed = time.time() - cycle_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                # 分段 sleep 以提高响应性
                sleep_end = time.time() + sleep_time
                while time.time() < sleep_end and not self._stop_event.is_set():
                    time.sleep(min(0.001, sleep_end - time.time()))

        LOGGER.info("[Pipeline] 硬实时仿真循环退出")

    # ---- 报告生成 ----

    def get_report(self) -> PipelineReport:
        """生成管线运行报告。

        Returns
        -------
        PipelineReport
            包含频率、抖动、组件统计、延迟分解等。
        """
        if self.config.enable_telemetry:
            component_stats = self._telemetry.get_all_component_stats()
            loop_rate = self._telemetry.get_loop_rate_stats(self.config.target_frequency_hz)
            jitter = self._telemetry.get_timing_jitter_ms()
            latency = self._telemetry.get_latency_breakdown()
            total_frames = self._telemetry.get_total_frames()
            total_errors = self._telemetry.get_total_errors()
            uptime = self._telemetry.get_uptime_s()
        else:
            component_stats = {}
            loop_rate = LoopRateStats(target_hz=self.config.target_frequency_hz)
            jitter = 0.0
            latency = LatencyBreakdown()
            total_frames = self._loop_count
            total_errors = self._error_count
            uptime = (time.time() - self._start_time) if self._start_time else 0.0

        return PipelineReport(
            actual_frequency_hz=loop_rate.mean_hz,
            timing_jitter_ms=jitter,
            component_breakdown=component_stats,
            total_frames_processed=total_frames,
            error_count=total_errors,
            state=self._state,
            loop_rate=loop_rate,
            latency=latency,
            uptime_s=uptime,
        )

    def print_report(self) -> None:
        """打印管线运行报告到日志。"""
        report = self.get_report()
        LOGGER.info("=" * 60)
        LOGGER.info("管线运行报告")
        LOGGER.info("=" * 60)
        LOGGER.info(f"  状态: {report.state.value}")
        LOGGER.info(f"  运行时间: {report.uptime_s:.1f}s")
        LOGGER.info(f"  总帧数: {report.total_frames_processed}")
        LOGGER.info(f"  错误数: {report.error_count}")
        LOGGER.info(f"  实际频率: {report.actual_frequency_hz:.2f} Hz "
                     f"(目标: {report.loop_rate.target_hz:.1f} Hz)")
        LOGGER.info(f"  时序抖动: {report.timing_jitter_ms:.3f} ms")
        LOGGER.info(f"  循环速率: mean={report.loop_rate.mean_hz:.2f}, "
                     f"min={report.loop_rate.min_hz:.2f}, "
                     f"max={report.loop_rate.max_hz:.2f}, "
                     f"std={report.loop_rate.std_hz:.2f} Hz")
        LOGGER.info(f"  延迟分解: sensor={report.latency.sensor_ms:.2f}, "
                     f"processing={report.latency.processing_ms:.2f}, "
                     f"control={report.latency.control_ms:.2f}, "
                     f"correction={report.latency.correction_ms:.2f}, "
                     f"overhead={report.latency.overhead_ms:.2f} ms")
        LOGGER.info(f"  总延迟: {report.latency.total_ms:.2f} ms")
        LOGGER.info("-" * 60)
        LOGGER.info("组件统计:")
        for name, stats in report.component_breakdown.items():
            LOGGER.info(f"  [{name}] 调用={stats.call_count}, "
                         f"均值={stats.mean_ms:.2f}, "
                         f"min={stats.min_ms:.2f}, "
                         f"max={stats.max_ms:.2f}, "
                         f"P99={stats.p99_ms:.2f} ms, "
                         f"超时={stats.timeout_count}, "
                         f"错误={stats.error_count}")
        LOGGER.info("=" * 60)

    # ---- 状态查询 ----

    def get_all_status(self) -> Dict[str, Any]:
        """获取管线及所有组件的完整状态。

        Returns
        -------
        Dict[str, Any]
            包含管线状态和各组件状态的字典。
        """
        return {
            "pipeline_state": self._state.value,
            "config": {
                "mode": self.config.mode,
                "target_frequency_hz": self.config.target_frequency_hz,
                "enable_telemetry": self.config.enable_telemetry,
            },
            "loop_count": self._loop_count,
            "error_count": self._error_count,
            "sensor": self._sensor.get_status(),
            "processor": self._processor.get_status(),
            "controller": self._controller.get_status(),
            "corrector": self._corrector.get_status(),
        }

    def reset(self) -> None:
        """重置管线及所有组件。"""
        if self._state == PipelineState.RUNNING:
            raise RuntimeError("请先停止管线再重置")

        self._sensor.reset()
        self._processor.reset()
        self._controller.reset()
        self._corrector.reset()

        self._loop_count = 0
        self._error_count = 0
        self._consecutive_errors = 0
        self._start_time = None
        self._last_loop_time = None
        self._latest_sensor = None
        self._latest_processing = None
        self._latest_control = None
        self._latest_correction = None
        self._shared_sensor_output = None
        self._shared_processing_output = None
        self._shared_control_output = None
        self._shared_correction_output = None

        self._state = PipelineState.IDLE
        LOGGER.info("[Pipeline] 管线已重置")
