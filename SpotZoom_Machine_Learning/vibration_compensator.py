"""
振动补偿器 (VibrationCompensator)

灵感来源:
- BoT-SORT (2022, Aharon N. et al.) — 相机运动补偿 (Camera Motion Compensation)
- ISO 11670 — 激光光束指向稳定性测量标准
- 机械振动隔离研究 — 主动隔振与被动隔振理论

算法原理:
- FFT Spectrum Analysis — 快速傅里叶变换频谱分析 (识别周期性振动频率)
- IIR Notch Filter — 无限脉冲响应陷波滤波器 (抑制特定频率的周期振动)
- Exponential Moving Average — 指数移动平均 (估计随机振动基线)
- Circular Buffer — 环形缓冲区 (高效存储最近位置样本)
- Feedforward Compensation — 前馈补偿 (生成反向补偿信号)

功能:
- 实时检测光斑位置序列中的振动分量
- 区分周期性振动 (机械/旋转引起) 与随机振动 (环境噪声)
- 对周期性振动进行陷波滤波抑制
- 对随机振动进行 EMA 平滑估计
- 生成前馈补偿信号，用于光束稳定控制

依赖: numpy, logging, dataclasses (无 PyTorch/TensorFlow)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.VibrationCompensator")


class VibrationMode(Enum):
    """振动模式分类。"""
    PERIODIC = "periodic"  # 周期性振动 (机械旋转、风扇等)
    RANDOM = "random"      # 随机振动 (环境噪声、地面震动等)
    MIXED = "mixed"        # 混合振动 (周期性 + 随机)


@dataclass
class VibrationReport:
    """振动分析完整报告。"""
    is_vibrating: bool                # 是否存在显著振动
    rms_x: float                      # X 方向 RMS 振动幅度 (像素)
    rms_y: float                      # Y 方向 RMS 振动幅度 (像素)
    dominant_freq_hz: float           # 主振动频率 (Hz)
    dominant_amplitude_px: float      # 主振动频率对应的幅度 (像素)
    compensation_efficiency: float    # 补偿效率 [0, 1]
    timestamp: float                  # 报告生成时间戳
    vibration_mode: VibrationMode = VibrationMode.MIXED  # 振动模式
    peak_to_peak_x: float = 0.0       # X 方向峰峰值 (像素)
    peak_to_peak_y: float = 0.0       # Y 方向峰峰值 (像素)
    num_dominant_frequencies: int = 0 # 检测到的显著频率分量数


class VibrationCompensator:
    """光学对准系统振动补偿器。

    通过 FFT 频谱分析实时检测光斑位置序列中的振动分量，
    将振动分解为周期性分量和随机分量，并生成前馈补偿信号。

    算法流程:
    1. 环形缓冲区存储最近 N 帧位置数据
    2. FFT 分析识别周期性振动频率
    3. IIR 陷波滤波器抑制周期性振动
    4. EMA 估计随机振动基线漂移
    5. 合成前馈补偿信号

    Parameters
    ----------
    fft_size : int
        FFT 分析窗口大小。必须是 2 的幂次。值越大频率分辨率越高，
        但时间响应越慢。典型值 128 ~ 512。
    vibration_threshold_hz : float
        振动检测频率下限 (Hz)。低于此频率的运动被视为漂移而非振动。
        典型值 0.5 ~ 2.0 Hz。
    damping_ratio : float
        陷波滤波器阻尼比。值越小陷波越窄 (选择性越高)，
        值越大陷波越宽 (鲁棒性越好)。典型值 0.1 ~ 1.0。
    """

    def __init__(
        self,
        fft_size: int = 256,
        vibration_threshold_hz: float = 1.0,
        damping_ratio: float = 0.7,
    ):
        # 参数校验
        if fft_size <= 0 or (fft_size & (fft_size - 1)) != 0:
            raise ValueError(f"fft_size 必须是 2 的幂次，当前值: {fft_size}")
        if vibration_threshold_hz <= 0:
            raise ValueError(f"vibration_threshold_hz 必须为正数，当前值: {vibration_threshold_hz}")
        if not (0.0 < damping_ratio <= 1.0):
            raise ValueError(f"damping_ratio 必须在 (0, 1] 范围内，当前值: {damping_ratio}")

        self.fft_size = int(fft_size)
        self.vibration_threshold_hz = float(vibration_threshold_hz)
        self.damping_ratio = float(damping_ratio)

        # 位置环形缓冲区
        self._buffer_x: Deque[float] = deque(maxlen=self.fft_size)
        self._buffer_y: Deque[float] = deque(maxlen=self.fft_size)
        self._buffer_t: Deque[float] = deque(maxlen=self.fft_size)

        # EMA 平滑状态 (用于随机振动基线估计)
        self._ema_x: float = 0.0
        self._ema_y: float = 0.0
        self._ema_alpha: float = 0.05  # EMA 平滑系数
        self._ema_initialized: bool = False

        # 陷波滤波器状态 (二阶 IIR，每个轴一组)
        self._notch_states_x: List[float] = [0.0, 0.0]
        self._notch_states_y: List[float] = [0.0, 0.0]

        # 当前补偿输出
        self._compensation_x: float = 0.0
        self._compensation_y: float = 0.0

        # 上一次 FFT 分析结果缓存
        self._cached_spectrum: Optional[dict] = None
        self._cached_dominant: Optional[List[Tuple[float, float]]] = None
        self._spectrum_dirty: bool = True  # 标记是否需要重新计算

        # 采样率估计
        self._sample_rate_estimate: float = 30.0  # 默认 30 Hz
        self._prev_timestamp: Optional[float] = None

        # 统计信息
        self._total_samples: int = 0
        self._start_time: Optional[float] = None

        LOGGER.info(
            "VibrationCompensator: 初始化完成 (fft_size=%d, threshold=%.1f Hz, damping=%.2f)",
            self.fft_size, self.vibration_threshold_hz, self.damping_ratio,
        )

    def update(self, position_x: float, position_y: float, timestamp: Optional[float] = None) -> None:
        """添加新的位置采样点。

        Parameters
        ----------
        position_x : float
            X 方向位置 (像素)。
        position_y : float
            Y 方向位置 (像素)。
        timestamp : float or None
            采样时间戳 (秒)。如果为 None，使用系统时间。
        """
        if timestamp is None:
            timestamp = time.time()

        if self._start_time is None:
            self._start_time = timestamp

        # 估计采样率
        if self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp
            if dt > 1e-6:
                # 指数加权移动平均更新采样率估计
                instant_rate = 1.0 / dt
                self._sample_rate_estimate = (
                    0.9 * self._sample_rate_estimate + 0.1 * instant_rate
                )
        self._prev_timestamp = timestamp

        # 写入环形缓冲区
        self._buffer_x.append(float(position_x))
        self._buffer_y.append(float(position_y))
        self._buffer_t.append(float(timestamp))

        # 更新 EMA 基线
        if not self._ema_initialized:
            self._ema_x = float(position_x)
            self._ema_y = float(position_y)
            self._ema_initialized = True
        else:
            alpha = self._ema_alpha
            self._ema_x = alpha * float(position_x) + (1.0 - alpha) * self._ema_x
            self._ema_y = alpha * float(position_y) + (1.0 - alpha) * self._ema_y

        # 标记频谱需要重新计算
        self._spectrum_dirty = True

        # 更新补偿信号
        self._update_compensation()

        self._total_samples += 1

        LOGGER.debug(
            "VibrationCompensator: 更新位置 (%.2f, %.2f), 补偿 (%.4f, %.4f)",
            position_x, position_y, self._compensation_x, self._compensation_y,
        )

    def get_compensation(self) -> Tuple[float, float]:
        """获取当前前馈补偿向量。

        补偿信号方向与振动方向相反，可直接叠加到控制输出上。

        Returns
        -------
        Tuple[float, float]
            (compensation_x, compensation_y) 补偿向量 (像素)。
        """
        return (self._compensation_x, self._compensation_y)

    def get_vibration_spectrum(self) -> dict:
        """获取频谱分析结果。

        对缓冲区中的位置数据进行 FFT 分析，返回频率、幅度信息。

        Returns
        -------
        dict
            频谱分析结果，包含:
            - 'frequencies_hz': 频率数组 (Hz)
            - 'amplitudes_x': X 方向幅度谱
            - 'amplitudes_y': Y 方向幅度谱
            - 'amplitudes_combined': 合成幅度谱
            - 'sample_rate_hz': 估计采样率 (Hz)
            - 'num_samples': 分析使用的样本数
        """
        if not self._spectrum_dirty and self._cached_spectrum is not None:
            return self._cached_spectrum

        n = len(self._buffer_x)
        if n < 4:
            LOGGER.debug("VibrationCompensator: 样本不足 (%d)，无法进行频谱分析", n)
            return {
                "frequencies_hz": np.array([]),
                "amplitudes_x": np.array([]),
                "amplitudes_y": np.array([]),
                "amplitudes_combined": np.array([]),
                "sample_rate_hz": self._sample_rate_estimate,
                "num_samples": n,
            }

        # 构造信号数组 (去均值)
        x_arr = np.array(self._buffer_x, dtype=np.float64)
        y_arr = np.array(self._buffer_y, dtype=np.float64)
        x_arr = x_arr - np.mean(x_arr)
        y_arr = y_arr - np.mean(y_arr)

        # 加窗 (Hann 窗减少频谱泄漏)
        window = np.hanning(n)
        x_windowed = x_arr * window
        y_windowed = y_arr * window

        # FFT
        fft_x = np.fft.rfft(x_windowed)
        fft_y = np.fft.rfft(y_windowed)

        # 幅度谱 (归一化)
        amplitudes_x = 2.0 * np.abs(fft_x) / n
        amplitudes_y = 2.0 * np.abs(fft_y) / n
        amplitudes_combined = np.sqrt(amplitudes_x ** 2 + amplitudes_y ** 2)

        # 频率轴
        freqs = np.fft.rfftfreq(n, d=1.0 / self._sample_rate_estimate)

        self._cached_spectrum = {
            "frequencies_hz": freqs,
            "amplitudes_x": amplitudes_x,
            "amplitudes_y": amplitudes_y,
            "amplitudes_combined": amplitudes_combined,
            "sample_rate_hz": self._sample_rate_estimate,
            "num_samples": n,
        }
        self._spectrum_dirty = False

        return self._cached_spectrum

    def get_dominant_frequencies(self, n: int = 3) -> List[Tuple[float, float]]:
        """获取前 N 个主导振动频率。

        Parameters
        ----------
        n : int
            返回的主导频率数量。

        Returns
        -------
        List[Tuple[float, float]]
            [(frequency_hz, amplitude_px), ...] 按幅度降序排列。
            仅返回高于振动阈值频率的分量。
        """
        if not self._spectrum_dirty and self._cached_dominant is not None:
            return self._cached_dominant[:n]

        spectrum = self.get_vibration_spectrum()
        freqs = spectrum["frequencies_hz"]
        amps = spectrum["amplitudes_combined"]

        if len(freqs) == 0:
            self._cached_dominant = []
            return []

        # 仅考虑高于阈值的频率 (排除 DC 分量和低频漂移)
        mask = freqs >= self.vibration_threshold_hz
        valid_freqs = freqs[mask]
        valid_amps = amps[mask]

        if len(valid_freqs) == 0:
            self._cached_dominant = []
            return []

        # 按幅度降序排列
        sorted_indices = np.argsort(valid_amps)[::-1]
        top_indices = sorted_indices[:n]

        result = [
            (float(valid_freqs[i]), float(valid_amps[i]))
            for i in top_indices
        ]

        self._cached_dominant = result
        return result

    def is_vibrating(self) -> bool:
        """判断当前是否存在显著振动。

        当合成 RMS 振动幅度超过阈值 (0.5 像素) 且存在高于
        振动阈值频率的显著频率分量时，判定为振动状态。

        Returns
        -------
        bool
            True 表示存在显著振动。
        """
        n = len(self._buffer_x)
        if n < 8:
            return False

        # 检查 RMS
        x_arr = np.array(self._buffer_x, dtype=np.float64)
        y_arr = np.array(self._buffer_y, dtype=np.float64)
        rms = float(np.sqrt(np.std(x_arr) ** 2 + np.std(y_arr) ** 2))

        if rms < 0.5:
            return False

        # 检查是否存在显著的高频分量
        dominant = self.get_dominant_frequencies(n=1)
        if len(dominant) == 0:
            return False

        # 主导频率幅度需达到 RMS 的一定比例
        dominant_amp = dominant[0][1]
        if dominant_amp < 0.1 * rms:
            return False

        return True

    def get_vibration_metrics(self) -> dict:
        """获取振动度量指标。

        Returns
        -------
        dict
            振动度量，包含:
            - 'rms_x': X 方向 RMS (像素)
            - 'rms_y': Y 方向 RMS (像素)
            - 'rms_combined': 合成 RMS (像素)
            - 'peak_to_peak_x': X 方向峰峰值 (像素)
            - 'peak_to_peak_y': Y 方向峰峰值 (像素)
            - 'dominant_freq_hz': 主导频率 (Hz)
            - 'dominant_amplitude_px': 主导幅度 (像素)
            - 'is_vibrating': 是否振动
            - 'vibration_mode': 振动模式
            - 'sample_rate_hz': 估计采样率 (Hz)
            - 'total_samples': 总采样数
        """
        n = len(self._buffer_x)
        if n < 2:
            return {
                "rms_x": 0.0, "rms_y": 0.0, "rms_combined": 0.0,
                "peak_to_peak_x": 0.0, "peak_to_peak_y": 0.0,
                "dominant_freq_hz": 0.0, "dominant_amplitude_px": 0.0,
                "is_vibrating": False,
                "vibration_mode": VibrationMode.RANDOM,
                "sample_rate_hz": self._sample_rate_estimate,
                "total_samples": self._total_samples,
            }

        x_arr = np.array(self._buffer_x, dtype=np.float64)
        y_arr = np.array(self._buffer_y, dtype=np.float64)

        rms_x = float(np.std(x_arr))
        rms_y = float(np.std(y_arr))
        rms_combined = float(np.sqrt(rms_x ** 2 + rms_y ** 2))
        ptp_x = float(np.ptp(x_arr))
        ptp_y = float(np.ptp(y_arr))

        dominant = self.get_dominant_frequencies(n=1)
        if len(dominant) > 0:
            dom_freq, dom_amp = dominant[0]
        else:
            dom_freq, dom_amp = 0.0, 0.0

        vibrating = self.is_vibrating()
        mode = self._classify_vibration_mode(rms_combined, dom_freq, dom_amp)

        return {
            "rms_x": round(rms_x, 4),
            "rms_y": round(rms_y, 4),
            "rms_combined": round(rms_combined, 4),
            "peak_to_peak_x": round(ptp_x, 4),
            "peak_to_peak_y": round(ptp_y, 4),
            "dominant_freq_hz": round(dom_freq, 2),
            "dominant_amplitude_px": round(dom_amp, 4),
            "is_vibrating": vibrating,
            "vibration_mode": mode,
            "sample_rate_hz": round(self._sample_rate_estimate, 1),
            "total_samples": self._total_samples,
        }

    def get_report(self) -> VibrationReport:
        """生成完整的振动分析报告。

        Returns
        -------
        VibrationReport
            振动分析报告数据类。
        """
        metrics = self.get_vibration_metrics()
        dominant = self.get_dominant_frequencies(n=3)

        # 计算补偿效率
        compensation = self.get_compensation()
        comp_magnitude = float(np.sqrt(compensation[0] ** 2 + compensation[1] ** 2))
        rms_combined = metrics["rms_combined"]

        if rms_combined > 1e-6:
            # 补偿效率 = 补偿信号覆盖的振动比例
            efficiency = min(1.0, comp_magnitude / rms_combined)
        else:
            efficiency = 1.0  # 无振动时效率为 100%

        report = VibrationReport(
            is_vibrating=metrics["is_vibrating"],
            rms_x=metrics["rms_x"],
            rms_y=metrics["rms_y"],
            dominant_freq_hz=metrics["dominant_freq_hz"],
            dominant_amplitude_px=metrics["dominant_amplitude_px"],
            compensation_efficiency=round(efficiency, 4),
            timestamp=time.time(),
            vibration_mode=metrics["vibration_mode"],
            peak_to_peak_x=metrics["peak_to_peak_x"],
            peak_to_peak_y=metrics["peak_to_peak_y"],
            num_dominant_frequencies=len(dominant),
        )

        LOGGER.debug(
            "VibrationCompensator: 报告 — vibrating=%s, RMS=%.3f, "
            "dom_freq=%.1f Hz, dom_amp=%.3f px, efficiency=%.2f",
            report.is_vibrating, rms_combined,
            report.dominant_freq_hz, report.dominant_amplitude_px,
            report.compensation_efficiency,
        )

        return report

    def reset(self) -> None:
        """重置补偿器，清除所有缓冲区和内部状态。"""
        self._buffer_x.clear()
        self._buffer_y.clear()
        self._buffer_t.clear()

        self._ema_x = 0.0
        self._ema_y = 0.0
        self._ema_initialized = False

        self._notch_states_x = [0.0, 0.0]
        self._notch_states_y = [0.0, 0.0]

        self._compensation_x = 0.0
        self._compensation_y = 0.0

        self._cached_spectrum = None
        self._cached_dominant = None
        self._spectrum_dirty = True

        self._sample_rate_estimate = 30.0
        self._prev_timestamp = None
        self._total_samples = 0
        self._start_time = None

        LOGGER.info("VibrationCompensator: 补偿器已重置")

    # ======================== 内部方法 ========================

    def _update_compensation(self) -> None:
        """更新前馈补偿信号。

        补偿信号 = -(周期性补偿 + 随机补偿)
        - 周期性补偿: 通过陷波滤波器提取的周期振动分量的反向
        - 随机补偿: EMA 基线偏移的反向
        """
        n = len(self._buffer_x)
        if n < 4:
            self._compensation_x = 0.0
            self._compensation_y = 0.0
            return

        current_x = self._buffer_x[-1]
        current_y = self._buffer_y[-1]

        # 1. 随机振动补偿 (EMA 基线偏移)
        random_comp_x = current_x - self._ema_x
        random_comp_y = current_y - self._ema_y

        # 2. 周期性振动补偿 (陷波滤波器)
        periodic_comp_x = self._apply_notch_filter_x(current_x - self._ema_x)
        periodic_comp_y = self._apply_notch_filter_y(current_y - self._ema_y)

        # 如果有主导频率，更新陷波滤波器参数
        if self._spectrum_dirty:
            dominant = self.get_dominant_frequencies(n=1)
            if len(dominant) > 0 and dominant[0][1] > 0.1:
                target_freq = dominant[0][0]
                self._update_notch_coefficients(target_freq)

        # 3. 合成补偿信号 (取反)
        self._compensation_x = -(periodic_comp_x + 0.3 * random_comp_x)
        self._compensation_y = -(periodic_comp_y + 0.3 * random_comp_y)

        # 限幅 (防止补偿信号过大导致控制不稳定)
        max_comp = 5.0  # 最大补偿量 (像素)
        comp_mag = float(np.sqrt(
            self._compensation_x ** 2 + self._compensation_y ** 2
        ))
        if comp_mag > max_comp:
            scale = max_comp / comp_mag
            self._compensation_x *= scale
            self._compensation_y *= scale

    def _apply_notch_filter_x(self, x: float) -> float:
        """对 X 通道应用二阶 IIR 陷波滤波器。

        Parameters
        ----------
        x : float
            输入信号 (去基线后的 X 位置)。

        Returns
        -------
        float
            滤波后输出 (周期性振动分量)。
        """
        return self._apply_notch_filter(x, self._notch_states_x)

    def _apply_notch_filter_y(self, y: float) -> float:
        """对 Y 通道应用二阶 IIR 陷波滤波器。

        Parameters
        ----------
        y : float
            输入信号 (去基线后的 Y 位置)。

        Returns
        -------
        float
            滤波后输出 (周期性振动分量)。
        """
        return self._apply_notch_filter(y, self._notch_states_y)

    def _apply_notch_filter(self, x: float, states: List[float]) -> float:
        """二阶 IIR 陷波滤波器核心实现。

        传递函数:
            H(z) = (1 - 2*cos(w0)*z^-1 + z^-2) / (1 - 2*r*cos(w0)*z^-1 + r^2*z^-2)

        其中 w0 = 2*pi*f0/fs, r = 1 - damping * pi / (2*Q)

        Parameters
        ----------
        x : float
            输入信号。
        states : List[float]
            滤波器状态 [x_prev, x_prev2, y_prev, y_prev2]。

        Returns
        -------
        float
            滤波输出。
        """
        # 获取当前陷波滤波器系数
        b0, b1, b2, a1, a2 = self._get_notch_coefficients()

        # 差分方程 (Direct Form II Transposed)
        # states: [delay_1, delay_2]
        y = b0 * x + states[0]
        states[0] = b1 * x - a1 * y + states[1]
        states[1] = b2 * x - a2 * y

        return y

    # 类级别缓存 (用于陷波滤波器系数)
    _notch_coeffs_cache: Optional[Tuple[float, float, float, float, float]] = None  # type: ignore[assignment]
    _notch_freq_cache: Optional[float] = None  # type: ignore[assignment]

    def _update_notch_coefficients(self, target_freq_hz: float) -> None:
        """根据目标频率更新陷波滤波器系数。

        Parameters
        ----------
        target_freq_hz : float
            目标陷波频率 (Hz)。
        """
        if (VibrationCompensator._notch_freq_cache is not None
                and abs(target_freq_hz - VibrationCompensator._notch_freq_cache) < 0.01):
            return  # 频率变化不大，不更新

        fs = self._sample_rate_estimate
        f0 = target_freq_hz

        # 防止频率超出 Nyquist
        if f0 >= fs / 2:
            f0 = fs / 2 * 0.95

        w0 = 2.0 * np.pi * f0 / fs
        r = 1.0 - self.damping_ratio * np.pi / 2.0

        # 陷波滤波器系数
        b0 = 1.0
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0
        a1 = -2.0 * r * np.cos(w0)
        a2 = r * r

        VibrationCompensator._notch_coeffs_cache = (b0, b1, b2, a1, a2)
        VibrationCompensator._notch_freq_cache = target_freq_hz

        LOGGER.debug(
            "VibrationCompensator: 陷波滤波器更新 — f0=%.1f Hz, r=%.4f",
            target_freq_hz, r,
        )

    def _get_notch_coefficients(self) -> Tuple[float, float, float, float, float]:
        """获取当前陷波滤波器系数。

        Returns
        -------
        Tuple[float, float, float, float, float]
            (b0, b1, b2, a1, a2) 滤波器系数。
        """
        if VibrationCompensator._notch_coeffs_cache is not None:
            return VibrationCompensator._notch_coeffs_cache

        # 默认系数 (全通，不滤波)
        return (1.0, 0.0, 0.0, 0.0, 0.0)

    def _classify_vibration_mode(
        self,
        rms: float,
        dominant_freq: float,
        dominant_amp: float,
    ) -> VibrationMode:
        """分类当前振动模式。

        Parameters
        ----------
        rms : float
            合成 RMS 振动幅度。
        dominant_freq : float
            主导振动频率 (Hz)。
        dominant_amp : float
            主导频率幅度。

        Returns
        -------
        VibrationMode
            振动模式分类。
        """
        if rms < 0.3:
            return VibrationMode.RANDOM

        if dominant_freq <= 0:
            return VibrationMode.RANDOM

        # 计算周期性分量占比
        periodic_ratio = dominant_amp / rms if rms > 1e-6 else 0.0

        if periodic_ratio > 0.7:
            return VibrationMode.PERIODIC
        elif periodic_ratio > 0.3:
            return VibrationMode.MIXED
        else:
            return VibrationMode.RANDOM

    @property
    def sample_count(self) -> int:
        """当前累计采样数。"""
        return self._total_samples

    @property
    def buffer_fill_ratio(self) -> float:
        """缓冲区填充比例 [0, 1]。"""
        return len(self._buffer_x) / self.fft_size if self.fft_size > 0 else 0.0

    @property
    def estimated_sample_rate(self) -> float:
        """估计采样率 (Hz)。"""
        return self._sample_rate_estimate
