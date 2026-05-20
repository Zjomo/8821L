"""
光学系统自动辨识器 (OpticalSystemIdentifier)

基于经典系统辨识理论的光学系统参数自动估计算法。

算法原理:
- Step Response Identification — 阶跃响应法辨识传递函数
- Least Squares Estimation — 最小二乘参数估计
- ARX Model — 自回归外生模型
- Cross-Correlation Analysis — 互相关延迟估计
- Frequency Domain Analysis — 频域系统分析

功能:
- 自动辨识光学系统传递函数 (阶跃响应法)
- 建立像素-物理坐标映射模型
- 系统延迟估计 (死区时间测量)
- 频率响应分析
- 模型验证与残差分析

依赖: numpy (无线性代数库之外的外部依赖)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np


@dataclass
class IdentificationConfig:
    """系统辨识配置。

    Attributes
    ----------
    model_order : int
        ARX 模型阶数。
    delay_range : Tuple[int, int]
        搜索的延迟范围 (最小, 最大) 采样点。
    min_step_samples : int
        阶跃响应最少采样点数。
    settling_threshold : float
        稳态判定阈值 (占阶跃幅值的比例)。
    frequency_resolution : int
        频率分析分辨率点数。
    regularization : float
        正则化系数。
    """
    model_order: int = 2
    delay_range: Tuple[int, int] = (0, 50)
    min_step_samples: int = 50
    settling_threshold: float = 0.02
    frequency_resolution: int = 512
    regularization: float = 1e-6


@dataclass
class FrequencyResponseData:
    """频率响应数据。

    Attributes
    ----------
    frequencies : np.ndarray
        频率点 (Hz 或 rad/s)。
    magnitude : np.ndarray
        幅值响应 (dB)。
    phase : np.ndarray
        相位响应 (度)。
    coherence : np.ndarray
        相干函数值 [0, 1]。
    bandwidth : float
        -3dB 带宽。
    resonance_freq : float
        谐振频率 (如存在)。
    phase_margin : float
        相位裕度 (度)。
    gain_margin : float
        增益裕度 (dB)。
    """
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    magnitude: np.ndarray = field(default_factory=lambda: np.array([]))
    phase: np.ndarray = field(default_factory=lambda: np.array([]))
    coherence: np.ndarray = field(default_factory=lambda: np.array([]))
    bandwidth: float = 0.0
    resonance_freq: float = 0.0
    phase_margin: float = 0.0
    gain_margin: float = 0.0


@dataclass
class SystemModel:
    """辨识得到的系统模型。

    Attributes
    ----------
    model_type : str
        模型类型 ('arx', 'first_order', 'second_order')。
    parameters : Dict[str, float]
        模型参数。
    numerator : np.ndarray
        传递函数分子系数。
    denominator : np.ndarray
        传递函数分母系数。
    delay_samples : int
        系统延迟 (采样点数)。
    sample_rate : float
        采样率 (Hz)。
    dc_gain : float
        直流增益。
    rise_time : float
        上升时间 (秒)。
    settling_time : float
        调节时间 (秒)。
    overshoot : float
        超调量 (%)。
    natural_freq : float
        自然频率 (Hz)。
    damping_ratio : float
        阻尼比。
    """
    model_type: str = "unknown"
    parameters: Dict[str, float] = field(default_factory=dict)
    numerator: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    denominator: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    delay_samples: int = 0
    sample_rate: float = 1.0
    dc_gain: float = 1.0
    rise_time: float = 0.0
    settling_time: float = 0.0
    overshoot: float = 0.0
    natural_freq: float = 0.0
    damping_ratio: float = 1.0


@dataclass
class IdentificationResult:
    """系统辨识结果。

    Attributes
    ----------
    success : bool
        辨识是否成功。
    model : SystemModel
        辨识得到的系统模型。
    frequency_response : FrequencyResponseData
        频率响应数据。
    fit_percentage : float
        模型拟合度 (%)。
    residual_std : float
        残差标准差。
    residual_mean : float
        残差均值。
    is_stable : bool
        系统是否稳定。
    model_order_used : int
        实际使用的模型阶数。
    """
    success: bool = False
    model: SystemModel = field(default_factory=SystemModel)
    frequency_response: FrequencyResponseData = field(default_factory=FrequencyResponseData)
    fit_percentage: float = 0.0
    residual_std: float = 0.0
    residual_mean: float = 0.0
    is_stable: bool = True
    model_order_used: int = 0


class OpticalSystemIdentifier:
    """光学系统自动辨识器。

    通过分析光学系统的输入-输出数据，自动辨识系统传递函数模型。
    支持阶跃响应法、ARX 模型辨识、延迟估计和频率响应分析。

    Parameters
    ----------
    config : IdentificationConfig or None
        辨识配置。
    sample_rate : float
        数据采样率 (Hz)。
    """

    def __init__(
        self,
        config: Optional[IdentificationConfig] = None,
        sample_rate: float = 30.0,
    ):
        self.config = config or IdentificationConfig()
        self.sample_rate = float(sample_rate)
        self._identified_model: Optional[SystemModel] = None

    @property
    def identified_model(self) -> Optional[SystemModel]:
        """获取已辨识的系统模型。"""
        return self._identified_model

    def identify_from_step_response(
        self,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
    ) -> IdentificationResult:
        """从阶跃响应数据辨识系统模型。

        Parameters
        ----------
        input_signal : np.ndarray
            输入信号 (如电机控制指令)。
        output_signal : np.ndarray
            输出信号 (如光斑位置)。

        Returns
        -------
        IdentificationResult
            辨识结果。
        """
        inp = np.asarray(input_signal, dtype=np.float64).flatten()
        out = np.asarray(output_signal, dtype=np.float64).flatten()

        n = min(len(inp), len(out))
        inp = inp[:n]
        out = out[:n]

        if n < self.config.min_step_samples:
            return IdentificationResult(success=False)

        # 1. 估计系统延迟
        delay = self._estimate_delay(inp, out)

        # 2. 补偿延迟
        if delay > 0 and delay < n - 1:
            out_compensated = out[delay:]
            inp_compensated = inp[:n - delay]
        else:
            out_compensated = out
            inp_compensated = inp
            delay = 0

        n_eff = len(out_compensated)
        if n_eff < self.config.min_step_samples:
            return IdentificationResult(success=False)

        # 3. 辨识模型参数
        model = self._identify_model(inp_compensated, out_compensated, delay)

        # 4. 频率响应分析
        freq_resp = self._compute_frequency_response(
            inp_compensated, out_compensated
        )

        # 5. 模型验证
        model_output = self._simulate_model(model, inp_compensated)
        fit, res_std, res_mean = self._validate_model(
            out_compensated, model_output
        )

        # 6. 稳定性检查
        is_stable = self._check_stability(model)

        self._identified_model = model

        return IdentificationResult(
            success=True,
            model=model,
            frequency_response=freq_resp,
            fit_percentage=fit,
            residual_std=res_std,
            residual_mean=res_mean,
            is_stable=is_stable,
            model_order_used=self.config.model_order,
        )

    def identify_pixel_to_physical(
        self,
        pixel_positions: np.ndarray,
        physical_positions: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """建立像素-物理坐标映射模型。

        Parameters
        ----------
        pixel_positions : np.ndarray
            像素坐标，形状 (n_samples, 2)。
        physical_positions : np.ndarray
            物理坐标，形状 (n_samples, 2)。

        Returns
        -------
        Tuple[np.ndarray, float]
            (变换矩阵, 映射误差)。
        """
        px = np.asarray(pixel_positions, dtype=np.float64)
        phys = np.asarray(physical_positions, dtype=np.float64)

        if px.ndim == 1:
            px = px.reshape(-1, 1)
        if phys.ndim == 1:
            phys = phys.reshape(-1, 1)

        n = px.shape[0]
        if n < 3:
            return np.eye(max(px.shape[1], phys.shape[1])), float("inf")

        # 构建齐次坐标
        ones = np.ones((n, 1))
        px_h = np.hstack([px, ones])  # (n, 3)

        # 最小二乘求解: phys ≈ px_h @ T
        T, residuals, _, _ = np.linalg.lstsq(px_h, phys, rcond=None)

        # 计算映射误差
        pred = px_h @ T
        error = float(np.sqrt(np.mean((pred - phys) ** 2)))

        return T, error

    def estimate_delay(
        self,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
    ) -> int:
        """估计系统延迟 (死区时间)。

        Parameters
        ----------
        input_signal : np.ndarray
            输入信号。
        output_signal : np.ndarray
            输出信号。

        Returns
        -------
        int
            估计的延迟采样点数。
        """
        inp = np.asarray(input_signal, dtype=np.float64).flatten()
        out = np.asarray(output_signal, dtype=np.float64).flatten()
        return self._estimate_delay(inp, out)

    def compute_frequency_response(
        self,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
    ) -> FrequencyResponseData:
        """计算系统的频率响应。

        Parameters
        ----------
        input_signal : np.ndarray
            输入信号。
        output_signal : np.ndarray
            输出信号。

        Returns
        -------
        FrequencyResponseData
            频率响应数据。
        """
        inp = np.asarray(input_signal, dtype=np.float64).flatten()
        out = np.asarray(output_signal, dtype=np.float64).flatten()
        return self._compute_frequency_response(inp, out)

    def reset(self) -> None:
        """重置辨识器。"""
        self._identified_model = None

    # ---- 内部方法 ----

    def _estimate_delay(
        self,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
    ) -> int:
        """使用互相关估计系统延迟。"""
        min_delay, max_delay = self.config.delay_range
        max_delay = min(max_delay, len(input_signal) - 1)

        # 互相关
        inp_centered = input_signal - np.mean(input_signal)
        out_centered = output_signal - np.mean(output_signal)

        inp_std = np.std(inp_centered)
        out_std = np.std(out_centered)

        if inp_std < 1e-10 or out_std < 1e-10:
            return 0

        inp_norm = inp_centered / inp_std
        out_norm = out_centered / out_std

        best_delay = 0
        best_corr = -float("inf")

        for d in range(min_delay, max_delay + 1):
            if d >= len(out_norm):
                break
            corr = float(np.sum(inp_norm[:len(out_norm) - d] * out_norm[d:]))
            if corr > best_corr:
                best_corr = corr
                best_delay = d

        return best_delay

    def _identify_model(
        self,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
        delay: int,
    ) -> SystemModel:
        """辨识系统模型 (ARX 模型)。"""
        order = self.config.model_order
        n = len(output_signal)

        # 构建 ARX 回归矩阵
        # y(t) = a1*y(t-1) + ... + an*y(t-n) + b0*u(t) + ... + bm*u(t-m)
        n_a = order  # AR 阶数
        n_b = order  # MA 阶数

        # 构建回归矩阵
        rows = []
        targets = []
        for t in range(max(n_a, n_b), n):
            row = []
            # AR 部分: y(t-1), ..., y(t-na)
            for k in range(1, n_a + 1):
                row.append(output_signal[t - k])
            # MA 部分: u(t), u(t-1), ..., u(t-nb)
            for k in range(n_b + 1):
                idx = t - k
                if idx >= 0:
                    row.append(input_signal[idx])
                else:
                    row.append(0.0)
            rows.append(row)
            targets.append(output_signal[t])

        if len(rows) < 10:
            return SystemModel(
                model_type="first_order",
                delay_samples=delay,
                sample_rate=self.sample_rate,
            )

        X = np.array(rows, dtype=np.float64)
        y = np.array(targets, dtype=np.float64)

        # 岭回归求解
        lambda_reg = self.config.regularization
        A = X.T @ X + lambda_reg * np.eye(X.shape[1])
        b = X.T @ y

        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(X, y, rcond=None)[0]

        # 提取 AR 和 MA 系数
        ar_coeffs = theta[:n_a]
        ma_coeffs = theta[n_a:n_a + n_b + 1]

        # 构建传递函数
        # 分子 (MA): [b0, b1, ..., bm]
        numerator = ma_coeffs.copy()
        # 分母 (AR): [1, -a1, -a2, ..., -an]
        denominator = np.concatenate([[1.0], -ar_coeffs])

        # DC 增益
        dc_gain = float(np.sum(numerator) / max(np.sum(denominator), 1e-10))

        # 估计时域参数
        rise_time, settling_time, overshoot = self._estimate_time_domain_params(
            input_signal, output_signal
        )

        # 估计自然频率和阻尼比 (从 AR 系数)
        if n_a >= 2:
            natural_freq, damping_ratio = self._estimate_modal_params(ar_coeffs)
        else:
            natural_freq = 1.0 / max(rise_time, 1e-10)
            damping_ratio = 1.0

        return SystemModel(
            model_type="arx",
            parameters={
                "ar_coeffs": [float(c) for c in ar_coeffs],
                "ma_coeffs": [float(c) for c in ma_coeffs],
            },
            numerator=numerator,
            denominator=denominator,
            delay_samples=delay,
            sample_rate=self.sample_rate,
            dc_gain=dc_gain,
            rise_time=rise_time,
            settling_time=settling_time,
            overshoot=overshoot,
            natural_freq=natural_freq,
            damping_ratio=damping_ratio,
        )

    def _estimate_time_domain_params(
        self,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
    ) -> Tuple[float, float, float]:
        """从阶跃响应估计时域参数。"""
        dt = 1.0 / self.sample_rate

        # 归一化输出
        y = output_signal.copy()
        y_range = y.max() - y.min()
        if y_range < 1e-10:
            return 0.0, 0.0, 0.0
        y_norm = (y - y.min()) / y_range

        # 稳态值
        y_final = float(np.mean(y_norm[-max(int(len(y_norm) * 0.1), 1):]))

        # 上升时间 (10% -> 90%)
        idx_10 = np.searchsorted(y_norm, 0.1)
        idx_90 = np.searchsorted(y_norm, 0.9)
        if idx_10 < len(y_norm) and idx_90 < len(y_norm):
            rise_time = float((idx_90 - idx_10) * dt)
        else:
            rise_time = 0.0

        # 调节时间 (进入 2% 误差带)
        threshold = self.config.settling_threshold
        settled = np.abs(y_norm - y_final) < threshold
        if np.any(settled):
            settle_idx = np.argmax(settled)
            settling_time = float(settle_idx * dt)
        else:
            settling_time = float(len(y_norm) * dt)

        # 超调量
        overshoot = float(max(0.0, (y_norm.max() - y_final) / max(y_final, 1e-10) * 100.0))

        return rise_time, settling_time, overshoot

    def _estimate_modal_params(
        self,
        ar_coeffs: np.ndarray,
    ) -> Tuple[float, float]:
        """从 AR 系数估计自然频率和阻尼比。"""
        if len(ar_coeffs) < 2:
            return 0.0, 1.0

        # 特征方程: z^2 - a1*z - a2 = 0
        a1 = ar_coeffs[0]
        a2 = ar_coeffs[1]

        # z^2 - a1*z - a2 = 0 => z = (a1 +/- sqrt(a1^2 + 4*a2)) / 2
        discriminant = a1 ** 2 + 4 * a2

        if discriminant < 0:
            # 复数极点 -> 欠阻尼
            real_part = a1 / 2.0
            imag_part = np.sqrt(-discriminant) / 2.0
            pole_mag = np.sqrt(real_part ** 2 + imag_part ** 2)

            if pole_mag < 1e-10:
                return 0.0, 1.0

            # 离散到连续
            omega_d = -np.log(pole_mag) * self.sample_rate
            damping = float(-np.log(pole_mag) / np.sqrt(np.log(pole_mag) ** 2 + imag_part ** 2 * 4))
            natural_freq = float(omega_d / (2 * np.pi * max(np.sqrt(max(1 - damping ** 2, 0.01)), 0.01)))
            return natural_freq, max(0.0, min(damping, 2.0))
        else:
            # 实数极点 -> 过阻尼或临界阻尼
            z1 = (a1 + np.sqrt(discriminant)) / 2.0
            z2 = (a1 - np.sqrt(discriminant)) / 2.0

            # 主导极点
            dominant = max(abs(z1), abs(z2))
            if dominant < 1e-10:
                return 0.0, 1.0

            tau = -1.0 / np.log(max(dominant, 1e-10)) / self.sample_rate
            natural_freq = 1.0 / (2 * np.pi * max(tau, 1e-10))

            return natural_freq, 1.0

    def _compute_frequency_response(
        self,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
    ) -> FrequencyResponseData:
        """计算频率响应 (基于 Welch 方法)。"""
        n = len(input_signal)
        n_fft = self.config.frequency_resolution

        # 确保长度一致
        n = min(len(input_signal), len(output_signal))
        inp = input_signal[:n]
        out = output_signal[:n]

        # 分段 (Welch 方法)
        n_per_seg = min(n // 4, 256)
        n_per_seg = max(n_per_seg, 16)
        noverlap = n_per_seg // 2

        # 计算互谱密度和自谱密度
        n_rfft = n_fft // 2 + 1
        S_xy_sum = np.zeros(n_rfft, dtype=np.complex128)
        S_xx_sum = np.zeros(n_rfft, dtype=np.float64)
        S_yy_sum = np.zeros(n_rfft, dtype=np.float64)
        n_segments = 0

        start = 0
        while start + n_per_seg <= n:
            seg_x = inp[start:start + n_per_seg]
            seg_y = out[start:start + n_per_seg]

            # 加窗
            window = np.hanning(n_per_seg)
            seg_x = seg_x * window
            seg_y = seg_y * window

            # FFT
            X = np.fft.rfft(seg_x, n=n_fft)
            Y = np.fft.rfft(seg_y, n=n_fft)

            S_xy_sum += X * np.conj(Y)
            S_xx_sum += np.abs(X) ** 2
            S_yy_sum += np.abs(Y) ** 2
            n_segments += 1
            start += n_per_seg - noverlap

        if n_segments == 0:
            return FrequencyResponseData()

        S_xy = S_xy_sum / n_segments
        S_xx = S_xx_sum / n_segments
        S_yy = S_yy_sum / n_segments

        # 传递函数
        H = S_xy / np.maximum(S_xx, 1e-10)

        # 幅值和相位
        magnitude = 20.0 * np.log10(np.maximum(np.abs(H), 1e-10))
        phase = np.angle(H, deg=True)

        # 相干函数
        coherence = np.abs(S_xy) ** 2 / np.maximum(S_xx * S_yy, 1e-10)
        coherence = np.clip(coherence, 0.0, 1.0)

        # 频率轴
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        # -3dB 带宽
        mag_max = np.max(magnitude)
        idx_3db = np.where(magnitude < mag_max - 3.0)[0]
        bandwidth = float(freqs[idx_3db[0]]) if len(idx_3db) > 0 else float(freqs[-1])

        # 谐振频率
        idx_res = np.argmax(magnitude)
        resonance_freq = float(freqs[idx_res]) if magnitude[idx_res] > mag_max - 1.0 else 0.0

        # 相位裕度和增益裕度 (近似)
        # 找到幅值穿越 0dB 的频率
        idx_0db = np.where(np.diff(np.sign(magnitude)))[0]
        if len(idx_0db) > 0:
            phase_at_0db = phase[idx_0db[0]]
            phase_margin = float(180.0 + phase_at_0db)
        else:
            phase_margin = 0.0

        # 增益裕度
        idx_180 = np.where(np.diff(np.sign(phase + 180.0)))[0]
        if len(idx_180) > 0:
            gain_at_180 = magnitude[idx_180[0]]
            gain_margin = float(-gain_at_180)
        else:
            gain_margin = float("inf")

        return FrequencyResponseData(
            frequencies=freqs,
            magnitude=magnitude,
            phase=phase,
            coherence=coherence,
            bandwidth=bandwidth,
            resonance_freq=resonance_freq,
            phase_margin=phase_margin,
            gain_margin=gain_margin,
        )

    def _simulate_model(
        self,
        model: SystemModel,
        input_signal: np.ndarray,
    ) -> np.ndarray:
        """使用辨识模型仿真输出。"""
        n = len(input_signal)
        output = np.zeros(n, dtype=np.float64)

        ar_coeffs = [-c for c in model.denominator[1:]]  # y(t) = a1*y(t-1) + ...
        ma_coeffs = model.numerator

        n_a = len(ar_coeffs)
        n_b = len(ma_coeffs)

        for t in range(n):
            # AR 部分
            ar_sum = 0.0
            for k in range(n_a):
                if t - k - 1 >= 0:
                    ar_sum += ar_coeffs[k] * output[t - k - 1]

            # MA 部分
            ma_sum = 0.0
            for k in range(n_b):
                if t - k >= 0:
                    ma_sum += ma_coeffs[k] * input_signal[t - k]

            output[t] = ar_sum + ma_sum

        return output

    def _validate_model(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> Tuple[float, float, float]:
        """验证模型拟合质量。"""
        residuals = actual - predicted
        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals))

        # 拟合度 (%)
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        if ss_tot < 1e-10:
            fit = 100.0
        else:
            fit = max(0.0, (1.0 - ss_res / ss_tot) * 100.0)

        return fit, res_std, res_mean

    def _check_stability(self, model: SystemModel) -> bool:
        """检查系统稳定性 (极点是否在单位圆内)。"""
        if len(model.denominator) < 2:
            return True

        # 特征方程的根
        ar_coeffs = model.denominator[1:]
        n_a = len(ar_coeffs)

        if n_a == 0:
            return True

        # 构建伴随矩阵
        companion = np.zeros((n_a, n_a), dtype=np.float64)
        companion[0, :] = ar_coeffs
        for i in range(1, n_a):
            companion[i, i - 1] = 1.0

        try:
            eigenvalues = np.linalg.eigvals(companion)
            max_pole_mag = float(np.max(np.abs(eigenvalues)))
            return max_pole_mag < 1.0
        except np.linalg.LinAlgError:
            return False
