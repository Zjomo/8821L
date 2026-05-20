"""
光谱/频率分析器 (SpectralAnalyzer)

基于经典信号处理与谱分析理论的频率域分析算法。

算法原理:
- Welch Method (Welch 1967) — 功率谱密度估计
- Periodogram (Schuster 1898) — 周期图法
- Allan Variance (Allan 1966) — 阿伦方差分析
- Short-Time Fourier Transform (STFT) — 短时傅里叶变换
- Noise Identification via Slope Analysis — 噪声类型斜率识别

功能:
- 功率谱密度 (PSD) 估计 (Welch / Periodogram)
- Allan 方差分析 (用于稳定性评估)
- 振动频率自动识别
- 时频分析 (短时傅里叶变换)
- 噪声类型识别 (白噪声 / 闪烁噪声 / 随机游走)

依赖: numpy (无线性代数库之外的外部依赖)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum

import numpy as np


class NoiseType(Enum):
    """噪声类型枚举。"""
    WHITE = "white"                # 白噪声 (随机角度游走)
    FLICKER = "flicker"            # 闪烁噪声 (1/f 噪声)
    RANDOM_WALK = "random_walk"    # 随机游走
    BIAS_INSTABILITY = "bias_instability"  # 偏置不稳定性
    UNKNOWN = "unknown"            # 未知类型


class PSDMethod(Enum):
    """PSD 估计方法枚举。"""
    WELCH = "welch"          # Welch 方法
    PERIODOGRAM = "periodogram"  # 周期图法


@dataclass
class PSDResult:
    """功率谱密度结果。

    Attributes
    ----------
    frequencies : np.ndarray
        频率轴 (Hz)。
    psd : np.ndarray
        功率谱密度 (单位²/Hz)。
    psd_db : np.ndarray
        功率谱密度 (dB)。
    method : str
        使用的估计方法。
    total_power : float
        总功率 (积分 PSD)。
    peak_frequency : float
        峰值频率 (Hz)。
    peak_power : float
        峰值功率。
    bandwidth_3db : float
        -3dB 带宽 (Hz)。
    rms_amplitude : float
        RMS 振幅。
    """
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    psd: np.ndarray = field(default_factory=lambda: np.array([]))
    psd_db: np.ndarray = field(default_factory=lambda: np.array([]))
    method: str = "welch"
    total_power: float = 0.0
    peak_frequency: float = 0.0
    peak_power: float = 0.0
    bandwidth_3db: float = 0.0
    rms_amplitude: float = 0.0


@dataclass
class AllanVarianceResult:
    """Allan 方差结果。

    Attributes
    ----------
    tau : np.ndarray
        平均时间数组 (秒)。
    allan_var : np.ndarray
        Allan 方差。
    allan_dev : np.ndarray
        Allan 偏差 (标准差)。
    adev_min : float
        最小 Allan 偏差 (最优平均时间处的值)。
    optimal_tau : float
        最优平均时间 (秒)。
    noise_coefficients : Dict[str, float]
        噪声系数 (随机游走 RW, 闪烁噪声 FN, 随机游走 RRW)。
    dominant_noise : NoiseType
        主导噪声类型。
    """
    tau: np.ndarray = field(default_factory=lambda: np.array([]))
    allan_var: np.ndarray = field(default_factory=lambda: np.array([]))
    allan_dev: np.ndarray = field(default_factory=lambda: np.array([]))
    adev_min: float = 0.0
    optimal_tau: float = 0.0
    noise_coefficients: Dict[str, float] = field(default_factory=dict)
    dominant_noise: NoiseType = NoiseType.UNKNOWN


@dataclass
class VibrationSignature:
    """振动特征。

    Attributes
    ----------
    detected_frequencies : List[float]
        检测到的振动频率列表 (Hz)。
    amplitudes : List[float]
        对应的振幅列表。
    snr_values : List[float]
        对应的信噪比列表。
    total_vibration_power : float
        总振动功率。
    dominant_frequency : float
        主导振动频率 (Hz)。
    dominant_amplitude : float
        主导振幅。
    is_periodic : bool
        是否存在周期性振动。
    quality_factor : float
        品质因数 (估计)。
    """
    detected_frequencies: List[float] = field(default_factory=list)
    amplitudes: List[float] = field(default_factory=list)
    snr_values: List[float] = field(default_factory=list)
    total_vibration_power: float = 0.0
    dominant_frequency: float = 0.0
    dominant_amplitude: float = 0.0
    is_periodic: bool = False
    quality_factor: float = 0.0


class SpectralAnalyzer:
    """光谱/频率分析器。

    提供功率谱密度估计、Allan 方差分析、振动频率识别、
    时频分析和噪声类型识别等频率域分析功能。

    Parameters
    ----------
    sample_rate : float
        数据采样率 (Hz)。
    n_fft : int
        FFT 点数。
    window_type : str
        窗函数类型 ('hann', 'hamming', 'blackman', 'rectangular')。
    overlap_ratio : float
        Welch 方法段间重叠比例 [0, 1)。
    """

    def __init__(
        self,
        sample_rate: float = 30.0,
        n_fft: int = 1024,
        window_type: str = "hann",
        overlap_ratio: float = 0.5,
    ):
        self.sample_rate = float(sample_rate)
        self.n_fft = int(n_fft)
        self.window_type = str(window_type).lower()
        self.overlap_ratio = float(overlap_ratio)

        # 预计算窗函数
        self._window = self._create_window(self.n_fft)

    def compute_psd(
        self,
        signal: np.ndarray,
        method: PSDMethod = PSDMethod.WELCH,
        n_per_segment: Optional[int] = None,
    ) -> PSDResult:
        """计算功率谱密度。

        Parameters
        ----------
        signal : np.ndarray
            输入信号。
        method : PSDMethod
            PSD 估计方法。
        n_per_segment : int or None
            每段长度 (仅 Welch 方法)。

        Returns
        -------
        PSDResult
            功率谱密度结果。
        """
        sig = np.asarray(signal, dtype=np.float64).flatten()

        if method == PSDMethod.WELCH:
            return self._welch_psd(sig, n_per_segment)
        else:
            return self._periodogram_psd(sig)

    def compute_allan_variance(
        self,
        signal: np.ndarray,
        min_tau: Optional[float] = None,
        max_tau: Optional[float] = None,
        n_taus: int = 50,
    ) -> AllanVarianceResult:
        """计算 Allan 方差。

        Parameters
        ----------
        signal : np.ndarray
            输入信号。
        min_tau : float or None
            最小平均时间 (秒)。
        max_tau : float or None
            最大平均时间 (秒)。
        n_taus : int
            计算点数。

        Returns
        -------
        AllanVarianceResult
            Allan 方差结果。
        """
        sig = np.asarray(signal, dtype=np.float64).flatten()
        n = len(sig)
        dt = 1.0 / self.sample_rate

        if n < 10:
            return AllanVarianceResult()

        # 生成 tau 数组 (对数间隔)
        if min_tau is None:
            min_tau = dt
        if max_tau is None:
            max_tau = (n // 2) * dt

        tau_min = max(min_tau, dt)
        tau_max = min(max_tau, (n // 2) * dt)

        if tau_max <= tau_min:
            return AllanVarianceResult()

        taus = np.logspace(
            np.log10(tau_min), np.log10(tau_max), n_taus
        )
        taus = np.unique(np.round(taus / dt).astype(int) * dt)

        allan_var = np.zeros(len(taus))
        for i, tau in enumerate(taus):
            m = max(int(round(tau / dt)), 1)
            if m >= n // 2:
                allan_var[i] = 0.0
                continue

            # Allan 方差计算
            # sigma^2(tau) = 0.5 * <(y_{k+2m} - 2*y_{k+m} + y_k)^2>
            n_clusters = n - 2 * m
            if n_clusters < 1:
                allan_var[i] = 0.0
                continue

            cluster_diff = sig[2 * m:] - 2 * sig[m:n - m] + sig[:n - 2 * m]
            allan_var[i] = 0.5 * np.mean(cluster_diff ** 2)

        allan_dev = np.sqrt(np.maximum(allan_var, 0.0))

        # 最优平均时间
        valid_mask = allan_dev > 0
        if np.any(valid_mask):
            min_idx = np.argmin(allan_dev[valid_mask])
            valid_indices = np.where(valid_mask)[0]
            adev_min = float(allan_dev[valid_indices[min_idx]])
            optimal_tau = float(taus[valid_indices[min_idx]])
        else:
            adev_min = 0.0
            optimal_tau = float(taus[0])

        # 噪声系数估计 (通过斜率拟合)
        noise_coeffs = self._estimate_noise_coefficients(taus, allan_dev)
        dominant_noise = self._identify_dominant_noise(taus, allan_dev)

        return AllanVarianceResult(
            tau=taus,
            allan_var=allan_var,
            allan_dev=allan_dev,
            adev_min=adev_min,
            optimal_tau=optimal_tau,
            noise_coefficients=noise_coeffs,
            dominant_noise=dominant_noise,
        )

    def identify_vibrations(
        self,
        signal: np.ndarray,
        min_snr: float = 3.0,
        min_frequency: float = 0.5,
        max_frequency: Optional[float] = None,
    ) -> VibrationSignature:
        """自动识别振动频率。

        Parameters
        ----------
        signal : np.ndarray
            输入信号。
        min_snr : float
            最小信噪比阈值。
        min_frequency : float
            最小检测频率 (Hz)。
        max_frequency : float or None
            最大检测频率 (Hz)。

        Returns
        -------
        VibrationSignature
            振动特征。
        """
        sig = np.asarray(signal, dtype=np.float64).flatten()

        # 计算 PSD
        psd_result = self.compute_psd(sig, PSDMethod.WELCH)

        if len(psd_result.frequencies) == 0:
            return VibrationSignature()

        freqs = psd_result.frequencies
        psd = psd_result.psd

        if max_frequency is None:
            max_frequency = self.sample_rate / 2.0

        # 频率范围掩码
        freq_mask = (freqs >= min_frequency) & (freqs <= max_frequency)
        freqs_masked = freqs[freq_mask]
        psd_masked = psd[freq_mask]

        if len(freqs_masked) == 0:
            return VibrationSignature()

        # 估计噪声底
        noise_floor = float(np.median(psd_masked))
        noise_std = float(np.std(psd_masked[psd_masked < np.percentile(psd_masked, 50)]))
        noise_std = max(noise_std, 1e-10)

        # 寻找峰值
        detected_freqs: List[float] = []
        detected_amps: List[float] = []
        detected_snrs: List[float] = []

        # 简单峰值检测
        threshold = noise_floor + min_snr * noise_std
        peak_mask = psd_masked > threshold

        # 连接相邻峰值
        if np.any(peak_mask):
            peak_indices = np.where(peak_mask)[0]
            groups = self._group_peaks(peak_indices, max_gap=3)

            for group in groups:
                if len(group) == 0:
                    continue
                # 找到组内最大值
                group_psd = psd_masked[group]
                best_in_group = group[np.argmax(group_psd)]
                peak_freq = float(freqs_masked[best_in_group])
                peak_amp = float(psd_masked[best_in_group])
                peak_snr = float((peak_amp - noise_floor) / noise_std)

                detected_freqs.append(peak_freq)
                detected_amps.append(peak_amp)
                detected_snrs.append(peak_snr)

        # 总振动功率
        total_power = float(np.sum(psd_masked[psd_masked > noise_floor]))

        # 主导频率
        if detected_freqs:
            dom_idx = int(np.argmax(detected_amps))
            dominant_freq = detected_freqs[dom_idx]
            dominant_amp = detected_amps[dom_idx]
        else:
            dominant_freq = 0.0
            dominant_amp = 0.0

        # 判断是否周期性
        is_periodic = len(detected_freqs) > 0 and max(detected_snrs) > min_snr * 2

        # 估计品质因数 (基于峰值宽度)
        quality = 0.0
        if detected_freqs and dominant_freq > 0:
            dom_bin = np.argmin(np.abs(freqs_masked - dominant_freq))
            half_power = psd_masked[dom_bin] / 2.0
            above_half = psd_masked > half_power
            bw = float(np.sum(above_half) * (freqs_masked[1] - freqs_masked[0])) if len(freqs_masked) > 1 else 1.0
            quality = dominant_freq / max(bw, 1e-10)

        return VibrationSignature(
            detected_frequencies=detected_freqs,
            amplitudes=detected_amps,
            snr_values=detected_snrs,
            total_vibration_power=total_power,
            dominant_frequency=dominant_freq,
            dominant_amplitude=dominant_amp,
            is_periodic=is_periodic,
            quality_factor=quality,
        )

    def compute_stft(
        self,
        signal: np.ndarray,
        window_size: Optional[int] = None,
        hop_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算短时傅里叶变换 (STFT)。

        Parameters
        ----------
        signal : np.ndarray
            输入信号。
        window_size : int or None
            窗口大小 (默认 n_fft)。
        hop_size : int or None
            跳跃大小 (默认 window_size // 4)。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (时间轴, 频率轴, STFT 幅值矩阵)。
        """
        sig = np.asarray(signal, dtype=np.float64).flatten()
        n = len(sig)

        if window_size is None:
            window_size = min(self.n_fft, n)
        window_size = max(int(window_size), 4)

        if hop_size is None:
            hop_size = max(window_size // 4, 1)

        # 窗函数
        window = self._create_window(window_size)

        # 计算段数
        n_segments = max((n - window_size) // hop_size + 1, 1)

        # 频率轴
        freqs = np.fft.rfftfreq(window_size, d=1.0 / self.sample_rate)

        # 时间轴
        times = np.arange(n_segments) * hop_size / self.sample_rate

        # STFT 计算
        n_freqs = len(freqs)
        stft_matrix = np.zeros((n_segments, n_freqs), dtype=np.float64)

        for i in range(n_segments):
            start = i * hop_size
            end = start + window_size
            if end > n:
                break

            segment = sig[start:end] * window
            spectrum = np.fft.rfft(segment, n=window_size)
            stft_matrix[i, :] = np.abs(spectrum) ** 2

        return times, freqs, stft_matrix

    def identify_noise_type(
        self,
        signal: np.ndarray,
    ) -> Tuple[NoiseType, float]:
        """识别信号中的主导噪声类型。

        通过分析 Allan 方差的双对数斜率来识别噪声类型:
        - 斜率 -1: 白噪声 (随机角度游走)
        - 斜率 0: 闪烁噪声 (1/f)
        - 斜率 +1: 随机游走

        Parameters
        ----------
        signal : np.ndarray
            输入信号。

        Returns
        -------
        Tuple[NoiseType, float]
            (噪声类型, 置信度 [0, 1])。
        """
        sig = np.asarray(signal, dtype=np.float64).flatten()
        if len(sig) < 20:
            return NoiseType.UNKNOWN, 0.0

        allan_result = self.compute_allan_variance(sig)
        if len(allan_result.tau) < 5:
            return NoiseType.UNKNOWN, 0.0

        # 在对数空间拟合斜率
        valid = allan_result.allan_dev > 0
        if np.sum(valid) < 5:
            return NoiseType.UNKNOWN, 0.0

        log_tau = np.log10(allan_result.tau[valid])
        log_adev = np.log10(allan_result.allan_dev[valid])

        # 线性回归
        n_pts = len(log_tau)
        X = np.column_stack([np.ones(n_pts), log_tau])
        y = log_adev
        theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        slope = float(theta[1])

        # 基于斜率判断噪声类型
        if slope < -0.75:
            noise_type = NoiseType.WHITE
            confidence = min(abs(slope + 1.0) / 0.25, 1.0)
        elif slope < -0.25:
            noise_type = NoiseType.FLICKER
            confidence = min(abs(slope) / 0.5, 1.0)
        elif slope < 0.5:
            noise_type = NoiseType.BIAS_INSTABILITY
            confidence = min(abs(slope) / 0.5, 1.0)
        elif slope < 1.25:
            noise_type = NoiseType.RANDOM_WALK
            confidence = min(abs(slope - 1.0) / 0.25, 1.0)
        else:
            noise_type = NoiseType.RANDOM_WALK
            confidence = 0.5

        # 拟合优度
        y_pred = X @ theta
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)
        confidence *= max(r_squared, 0.0)

        return noise_type, float(min(max(confidence, 0.0), 1.0))

    # ---- 内部方法 ----

    def _welch_psd(
        self,
        signal: np.ndarray,
        n_per_segment: Optional[int] = None,
    ) -> PSDResult:
        """Welch 方法功率谱密度估计。"""
        n = len(signal)
        if n < 4:
            return PSDResult(method="welch")

        if n_per_segment is None:
            n_per_segment = min(self.n_fft, n)
        n_per_segment = max(int(n_per_segment), 4)

        noverlap = int(n_per_segment * self.overlap_ratio)
        noverlap = min(noverlap, n_per_segment - 1)

        window = self._create_window(n_per_segment)
        n_fft = max(self.n_fft, n_per_segment)

        # 分段计算
        psd_sum = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        n_segments = 0
        start = 0

        while start + n_per_segment <= n:
            segment = signal[start:start + n_per_segment] * window
            spectrum = np.fft.rfft(segment, n=n_fft)
            psd_sum += np.abs(spectrum) ** 2
            n_segments += 1
            start += n_per_segment - noverlap

        if n_segments == 0:
            return PSDResult(method="welch")

        # 归一化
        window_power = float(np.sum(window ** 2))
        psd = (2.0 / (self.sample_rate * window_power * n_segments)) * psd_sum

        # 频率轴
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        # dB
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20))

        # 总功率
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        total_power = float(np.sum(psd) * df)

        # 峰值
        peak_idx = int(np.argmax(psd))
        peak_freq = float(freqs[peak_idx])
        peak_power = float(psd[peak_idx])

        # -3dB 带宽
        half_power = psd[peak_idx] / 2.0
        above_half = psd > half_power
        if np.any(above_half):
            bw_indices = np.where(above_half)[0]
            bandwidth = float((bw_indices[-1] - bw_indices[0]) * df)
        else:
            bandwidth = 0.0

        # RMS
        rms = float(np.sqrt(total_power))

        return PSDResult(
            frequencies=freqs,
            psd=psd,
            psd_db=psd_db,
            method="welch",
            total_power=total_power,
            peak_frequency=peak_freq,
            peak_power=peak_power,
            bandwidth_3db=bandwidth,
            rms_amplitude=rms,
        )

    def _periodogram_psd(self, signal: np.ndarray) -> PSDResult:
        """周期图法功率谱密度估计。"""
        n = len(signal)
        if n < 4:
            return PSDResult(method="periodogram")

        n_fft = max(self.n_fft, n)
        window = self._create_window(n)

        # 零填充
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[:n] = signal * window

        spectrum = np.fft.rfft(padded, n=n_fft)
        psd = (2.0 / (self.sample_rate * np.sum(window ** 2))) * np.abs(spectrum) ** 2

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20))

        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        total_power = float(np.sum(psd) * df)

        peak_idx = int(np.argmax(psd))
        peak_freq = float(freqs[peak_idx])
        peak_power = float(psd[peak_idx])

        rms = float(np.sqrt(total_power))

        return PSDResult(
            frequencies=freqs,
            psd=psd,
            psd_db=psd_db,
            method="periodogram",
            total_power=total_power,
            peak_frequency=peak_freq,
            peak_power=peak_power,
            rms_amplitude=rms,
        )

    def _estimate_noise_coefficients(
        self,
        tau: np.ndarray,
        allan_dev: np.ndarray,
    ) -> Dict[str, float]:
        """估计噪声系数。

        通过拟合 Allan 偏差曲线的斜率来估计各噪声项系数。
        """
        valid = allan_dev > 0
        if np.sum(valid) < 5:
            return {}

        log_tau = np.log10(tau[valid])
        log_adev = np.log10(allan_dev[valid])

        # 使用三段拟合
        n_pts = len(log_tau)
        coeffs = {}

        # 白噪声 (斜率 -1): 取短 tau 段
        n_white = max(n_pts // 3, 3)
        if n_white >= 2:
            X_white = np.column_stack([np.ones(n_white), log_tau[:n_white]])
            y_white = log_adev[:n_white]
            try:
                theta_w, _, _, _ = np.linalg.lstsq(X_white, y_white, rcond=None)
                slope_w = float(theta_w[1])
                if -1.5 < slope_w < -0.5:
                    # 白噪声系数 N: ADEV(tau) = N / sqrt(tau)
                    # log(ADEV) = log(N) - 0.5 * log(tau)
                    N = float(10 ** theta_w[0])
                    coeffs["white_noise_N"] = N
            except np.linalg.LinAlgError:
                pass

        # 随机游走 (斜率 +1): 取长 tau 段
        n_rw = max(n_pts // 3, 3)
        if n_rw >= 2:
            X_rw = np.column_stack([np.ones(n_rw), log_tau[-n_rw:]])
            y_rw = log_adev[-n_rw:]
            try:
                theta_rw, _, _, _ = np.linalg.lstsq(X_rw, y_rw, rcond=None)
                slope_rw = float(theta_rw[1])
                if 0.5 < slope_rw < 1.5:
                    # 随机游走系数 K: ADEV(tau) = K * sqrt(tau)
                    K = float(10 ** theta_rw[0])
                    coeffs["random_walk_K"] = K
            except np.linalg.LinAlgError:
                pass

        # 闪烁噪声 (斜率 0): 取中间段
        n_mid = max(n_pts // 3, 3)
        mid_start = (n_pts - n_mid) // 2
        if n_mid >= 2:
            X_mid = np.column_stack([np.ones(n_mid), log_tau[mid_start:mid_start + n_mid]])
            y_mid = log_adev[mid_start:mid_start + n_mid]
            try:
                theta_m, _, _, _ = np.linalg.lstsq(X_mid, y_mid, rcond=None)
                slope_m = float(theta_m[1])
                if -0.5 < slope_m < 0.5:
                    B = float(10 ** theta_m[0])
                    coeffs["flicker_noise_B"] = B
            except np.linalg.LinAlgError:
                pass

        return coeffs

    def _identify_dominant_noise(
        self,
        tau: np.ndarray,
        allan_dev: np.ndarray,
    ) -> NoiseType:
        """识别主导噪声类型。"""
        valid = allan_dev > 0
        if np.sum(valid) < 5:
            return NoiseType.UNKNOWN

        log_tau = np.log10(tau[valid])
        log_adev = np.log10(allan_dev[valid])

        # 全局斜率
        n_pts = len(log_tau)
        X = np.column_stack([np.ones(n_pts), log_tau])
        y = log_adev
        try:
            theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            slope = float(theta[1])
        except np.linalg.LinAlgError:
            return NoiseType.UNKNOWN

        if slope < -0.75:
            return NoiseType.WHITE
        elif slope < -0.25:
            return NoiseType.FLICKER
        elif slope < 0.5:
            return NoiseType.BIAS_INSTABILITY
        else:
            return NoiseType.RANDOM_WALK

    def _create_window(self, length: int) -> np.ndarray:
        """创建窗函数。"""
        n = int(length)
        if n <= 0:
            return np.array([1.0])

        if self.window_type == "hann":
            return np.hanning(n)
        elif self.window_type == "hamming":
            return np.hamming(n)
        elif self.window_type == "blackman":
            return np.blackman(n)
        elif self.window_type == "rectangular":
            return np.ones(n)
        else:
            return np.hanning(n)

    @staticmethod
    def _group_peaks(indices: np.ndarray, max_gap: int = 3) -> List[np.ndarray]:
        """将相邻峰值分组。"""
        if len(indices) == 0:
            return []

        groups = []
        current_group = [indices[0]]

        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] <= max_gap:
                current_group.append(indices[i])
            else:
                groups.append(np.array(current_group))
                current_group = [indices[i]]

        groups.append(np.array(current_group))
        return groups
