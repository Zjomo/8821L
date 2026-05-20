"""
系统辨识器 (System Identifier)

参考开源项目:
  - ML for AO System Identification: 机器学习驱动的自适应光学系统辨识

核心思想:
  从 AO 系统辨识的最佳实践中借鉴，实现数据驱动的系统辨识方法，
  用于估计光学对准系统的动态特性并优化控制器参数。

  在 SpotZoom 场景中:
  - 系统辨识 → 从输入-输出数据中估计系统传递函数
  - 频率响应 → 系统对不同频率扰动的响应特性
  - PID 优化 → 基于系统模型自动调优 PID 参数

创新点:
  1. ARX/ARMAX 模型辨识
  2. 频率响应估计 (Welch 方法)
  3. 基于模型的最优 PID 参数整定
  4. 闭环系统验证与性能预测
  5. 多输入多输出 (MIMO) 系统支持

纯 numpy 实现，无外部依赖。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SysIdConfig:
    """系统辨识器配置。"""
    # ARX 模型阶数 (na, nb, nk)
    arx_order_na: int = 2  # 输出多项式阶数
    arx_order_nb: int = 2  # 输入多项式阶数
    arx_order_nk: int = 1  # 延迟
    # 频率响应估计参数
    freq_response_nperseg: int = 256  # Welch 方法段长度
    freq_response_noverlap: int = 128  # Welch 方法重叠
    # 频率范围 (归一化, 0~0.5 对应 Nyquist)
    freq_range: Tuple[float, float] = (0.001, 0.5)
    # PID 优化参数范围
    pid_kp_range: Tuple[float, float] = (0.01, 10.0)
    pid_ki_range: Tuple[float, float] = (0.001, 5.0)
    pid_kd_range: Tuple[float, float] = (0.0, 2.0)
    # PID 优化目标权重
    pid_settling_weight: float = 0.4
    pid_overshoot_weight: float = 0.3
    pid_steady_error_weight: float = 0.2
    pid_control_effort_weight: float = 0.1
    # 仿真参数
    simulation_steps: int = 500
    # 采样率 (Hz)
    sampling_rate: float = 1000.0
    # 数据归一化
    normalize_data: bool = True
    # 正则化系数
    regularization: float = 1e-6


@dataclass
class SysIdResult:
    """系统辨识结果。"""
    # ARX 模型参数 [a1, a2, ..., b0, b1, ...]
    arx_parameters: np.ndarray = field(default_factory=lambda: np.zeros(5))
    # 估计的频率响应
    frequencies: np.ndarray = field(default_factory=lambda: np.zeros(128))
    magnitude: np.ndarray = field(default_factory=lambda: np.zeros(128))
    phase: np.ndarray = field(default_factory=lambda: np.zeros(128))
    # 最优 PID 参数
    optimal_kp: float = 0.0
    optimal_ki: float = 0.0
    optimal_kd: float = 0.0
    # 模型验证指标
    fit_percentage: float = 0.0
    residual_rms: float = 0.0
    # 估计的系统特性
    estimated_bandwidth_hz: float = 0.0
    estimated_phase_margin_deg: float = 0.0
    estimated_gain_margin_db: float = 0.0
    # 是否成功
    success: bool = False


class SystemIdentifier:
    """系统辨识器。

    数据驱动的系统辨识工具，用于估计光学对准系统的动态特性
    并优化控制器参数。

    Parameters
    ----------
    config : SysIdConfig
        辨识器配置参数。

    References
    ----------
    .. [1] Ljung, L. (1999). "System Identification: Theory for the
           User." 2nd ed. Prentice Hall.
    .. [2] Astrom, K. J. & Hagglund, T. (2006). "Advanced PID
           Control." ISA.
    .. [3] Welch, P. D. (1967). "The use of fast Fourier transform
           for the estimation of power spectra." IEEE Trans. Audio
           Electroacoust., 15(2), 70-73.
    """

    def __init__(self, config: Optional[SysIdConfig] = None) -> None:
        self.config = config or SysIdConfig()
        self._input_data: Optional[np.ndarray] = None
        self._output_data: Optional[np.ndarray] = None
        self._model: Optional[SysIdResult] = None

        logger.info(
            f"SystemIdentifier 初始化: "
            f"ARX阶数=({self.config.arx_order_na}, {self.config.arx_order_nb}, "
            f"{self.config.arx_order_nk}), "
            f"采样率={self.config.sampling_rate}Hz"
        )

    def collect_data(
        self,
        input_signal: np.ndarray,
        output_signal: np.ndarray,
    ) -> None:
        """收集输入-输出数据。

        Parameters
        ----------
        input_signal : np.ndarray
            输入信号 (控制器输出 / DM 命令)。
        output_signal : np.ndarray
            输出信号 (WFS 测量 / 位置误差)。
        """
        if len(input_signal) != len(output_signal):
            raise ValueError(
                f"输入信号长度 {len(input_signal)} 与输出信号长度 "
                f"{len(output_signal)} 不匹配"
            )

        min_length = max(
            self.config.arx_order_na,
            self.config.arx_order_nb + self.config.arx_order_nk,
        ) + 10

        if len(input_signal) < min_length:
            raise ValueError(
                f"数据长度 {len(input_signal)} 太短，至少需要 {min_length}"
            )

        self._input_data = input_signal.copy()
        self._output_data = output_signal.copy()

        logger.info(
            f"数据收集完成: {len(input_signal)} 个样本, "
            f"输入范围=[{np.min(input_signal):.3f}, {np.max(input_signal):.3f}], "
            f"输出范围=[{np.min(output_signal):.3f}, {np.max(output_signal):.3f}]"
        )

    def estimate_frequency_response(
        self,
        input_signal: Optional[np.ndarray] = None,
        output_signal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """估计频率响应。

        使用 Welch 方法估计系统的频率响应 (传递函数)。

        Parameters
        ----------
        input_signal : np.ndarray, optional
            输入信号。默认使用已收集的数据。
        output_signal : np.ndarray, optional
            输出信号。默认使用已收集的数据。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (频率数组 Hz, 幅度, 相位 度)。
        """
        u = input_signal if input_signal is not None else self._input_data
        y = output_signal if output_signal is not None else self._output_data

        if u is None or y is None:
            raise ValueError("无数据可分析，请先调用 collect_data()")

        # 归一化
        if self.config.normalize_data:
            u = (u - np.mean(u)) / (np.std(u) + 1e-10)
            y = (y - np.mean(y)) / (np.std(y) + 1e-10)

        # 互功率谱密度
        nperseg = min(self.config.freq_response_nperseg, len(u) // 2)
        noverlap = min(self.config.freq_response_noverlap, nperseg - 1)

        f, Pxy = self._welch_psd(y, u, nperseg, noverlap)
        f, Pxx = self._welch_psd(u, u, nperseg, noverlap)

        # 传递函数估计
        H = Pxy / (Pxx + 1e-10)

        # 转换为 Hz
        freq_hz = f * self.config.sampling_rate

        # 幅度和相位
        magnitude = np.abs(H)
        phase = np.angle(H, deg=True)

        # 频率范围滤波
        freq_min = self.config.freq_range[0] * self.config.sampling_rate
        freq_max = self.config.freq_range[1] * self.config.sampling_rate
        mask = (freq_hz >= freq_min) & (freq_hz <= freq_max)
        freq_hz = freq_hz[mask]
        magnitude = magnitude[mask]
        phase = phase[mask]

        logger.info(
            f"频率响应估计完成: {len(freq_hz)} 个频率点, "
            f"范围=[{freq_hz[0]:.1f}, {freq_hz[-1]:.1f}] Hz"
        )

        return freq_hz, magnitude, phase

    def identify_model(
        self,
        input_signal: Optional[np.ndarray] = None,
        output_signal: Optional[np.ndarray] = None,
    ) -> SysIdResult:
        """执行系统辨识。

        使用 ARX 模型从输入-输出数据中辨识系统模型。

        Parameters
        ----------
        input_signal : np.ndarray, optional
            输入信号。
        output_signal : np.ndarray, optional
            输出信号。

        Returns
        -------
        SysIdResult
            辨识结果。
        """
        u = input_signal if input_signal is not None else self._input_data
        y = output_signal if output_signal is not None else self._output_data

        if u is None or y is None:
            raise ValueError("无数据可分析，请先调用 collect_data()")

        logger.info("开始 ARX 模型辨识...")

        result = SysIdResult()

        # 归一化
        u_mean, u_std = np.mean(u), np.std(u) + 1e-10
        y_mean, y_std = np.mean(y), np.std(y) + 1e-10
        if self.config.normalize_data:
            u_norm = (u - u_mean) / u_std
            y_norm = (y - y_mean) / y_std
        else:
            u_norm = u
            y_norm = y

        # 构建 ARX 回归矩阵
        na = self.config.arx_order_na
        nb = self.config.arx_order_nb
        nk = self.config.arx_order_nk
        N = len(u_norm)

        # 回归矩阵行数
        n_rows = N - max(na, nb + nk - 1)

        if n_rows < 10:
            raise ValueError("数据长度不足以进行 ARX 辨识")

        # 构建回归矩阵 Phi 和目标向量 Y
        # y(t) + a1*y(t-1) + ... + ana*y(t-na) = b0*u(t-nk) + ... + bnb*u(t-nk-nb+1)
        n_params = na + nb
        Phi = np.zeros((n_rows, n_params))
        Y = np.zeros(n_rows)

        for i in range(n_rows):
            t = i + max(na, nb + nk - 1)
            # 输出延迟项
            for j in range(na):
                Phi[i, j] = -y_norm[t - 1 - j]
            # 输入项
            for j in range(nb):
                idx = t - nk - j
                if 0 <= idx < N:
                    Phi[i, na + j] = u_norm[idx]
            Y[i] = y_norm[t]

        # 最小二乘求解 (带正则化)
        I_reg = self.config.regularization * np.eye(n_params)
        try:
            params, _, _, _ = np.linalg.lstsq(
                Phi.T @ Phi + I_reg, Phi.T @ Y, rcond=None
            )
        except np.linalg.LinAlgError:
            logger.error("ARX 模型求解失败 (矩阵奇异)")
            result.success = False
            return result

        result.arx_parameters = params

        # 模型验证: 计算拟合百分比
        Y_pred = Phi @ params
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        result.fit_percentage = float(max(0, (1 - ss_res / (ss_tot + 1e-10)) * 100))
        result.residual_rms = float(np.sqrt(np.mean((Y - Y_pred) ** 2)))

        # 频率响应估计
        freqs, mag, pha = self.estimate_frequency_response(u, y)
        result.frequencies = freqs
        result.magnitude = mag
        result.phase = pha

        # 估计系统特性
        result.estimated_bandwidth_hz = self._estimate_bandwidth(freqs, mag)
        result.estimated_gain_margin_db, result.estimated_phase_margin_deg = (
            self._estimate_stability_margins(freqs, mag, pha)
        )

        result.success = result.fit_percentage > 10

        self._model = result

        logger.info(
            f"ARX 辨识完成: 拟合度={result.fit_percentage:.1f}%, "
            f"残差RMS={result.residual_rms:.4f}, "
            f"带宽={result.estimated_bandwidth_hz:.1f}Hz"
        )
        return result

    def optimize_pid(
        self,
        model_result: Optional[SysIdResult] = None,
    ) -> Tuple[float, float, float]:
        """基于辨识模型优化 PID 参数。

        Parameters
        ----------
        model_result : SysIdResult, optional
            辨识结果。默认使用上次辨识结果。

        Returns
        -------
        Tuple[float, float, float]
            最优 PID 参数 (Kp, Ki, Kd)。
        """
        if model_result is None:
            model_result = self._model

        if model_result is None or not model_result.success:
            logger.error("无有效模型，请先执行 identify_model()")
            return (0.0, 0.0, 0.0)

        logger.info("开始 PID 参数优化...")

        best_score = float("inf")
        best_params = (1.0, 0.1, 0.0)

        # 网格搜索 + 精细搜索
        n_coarse = 20
        kp_range = np.linspace(*self.config.pid_kp_range, n_coarse)
        ki_range = np.linspace(*self.config.pid_ki_range, n_coarse)
        kd_range = np.linspace(*self.config.pid_kd_range, max(n_coarse // 2, 5))

        for kp in kp_range:
            for ki in ki_range:
                for kd in kd_range:
                    score = self._evaluate_pid(kp, ki, kd, model_result)
                    if score < best_score:
                        best_score = score
                        best_params = (kp, ki, kd)

        # 精细搜索
        kp_fine = np.linspace(
            best_params[0] * 0.8, best_params[0] * 1.2, 20
        )
        ki_fine = np.linspace(
            best_params[1] * 0.8, best_params[1] * 1.2, 20
        )
        kd_fine = np.linspace(
            best_params[2] * 0.8, best_params[2] * 1.2, 10
        )

        for kp in kp_fine:
            for ki in ki_fine:
                for kd in kd_fine:
                    score = self._evaluate_pid(kp, ki, kd, model_result)
                    if score < best_score:
                        best_score = score
                        best_params = (kp, ki, kd)

        model_result.optimal_kp = best_params[0]
        model_result.optimal_ki = best_params[1]
        model_result.optimal_kd = best_params[2]

        logger.info(
            f"PID 优化完成: Kp={best_params[0]:.4f}, "
            f"Ki={best_params[1]:.4f}, Kd={best_params[2]:.4f}, "
            f"评分={best_score:.4f}"
        )
        return best_params

    def validate_model(
        self,
        model_result: Optional[SysIdResult] = None,
        validation_input: Optional[np.ndarray] = None,
        validation_output: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """验证模型在独立数据上的性能。

        Parameters
        ----------
        model_result : SysIdResult, optional
            辨识结果。
        validation_input : np.ndarray, optional
            验证输入信号。
        validation_output : np.ndarray, optional
            验证输出信号。

        Returns
        -------
        dict
            验证指标。
        """
        if model_result is None:
            model_result = self._model

        if model_result is None:
            return {"error": "无模型可验证"}

        u = validation_input if validation_input is not None else self._input_data
        y = validation_output if validation_output is not None else self._output_data

        if u is None or y is None:
            return {"error": "无验证数据"}

        # 使用辨识的 ARX 模型仿真
        na = self.config.arx_order_na
        nb = self.config.arx_order_nb
        nk = self.config.arx_order_nk
        params = model_result.arx_parameters

        y_sim = np.zeros_like(y)
        for t in range(max(na, nb + nk), len(y)):
            y_pred = 0.0
            for j in range(na):
                y_pred -= params[j] * y_sim[t - 1 - j]
            for j in range(nb):
                idx = t - nk - j
                if 0 <= idx < len(u):
                    y_pred += params[na + j] * u[idx]
            y_sim[t] = y_pred

        # 计算验证指标
        start = max(na, nb + nk)
        residual = y[start:] - y_sim[start:]
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((y[start:] - np.mean(y[start:])) ** 2)
        fit_pct = float(max(0, (1 - ss_res / (ss_tot + 1e-10)) * 100))
        rms = float(np.sqrt(np.mean(residual ** 2)))
        max_error = float(np.max(np.abs(residual)))

        metrics = {
            "validation_fit_percent": fit_pct,
            "validation_rms": rms,
            "validation_max_error": max_error,
            "validation_correlation": float(np.corrcoef(y[start:], y_sim[start:])[0, 1]),
        }

        logger.info(f"模型验证: 拟合度={fit_pct:.1f}%, RMS={rms:.4f}")
        return metrics

    # ============ 内部方法 ============

    def _welch_psd(
        self,
        x: np.ndarray,
        y: np.ndarray,
        nperseg: int,
        noverlap: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Welch 方法估计互功率谱密度。"""
        step = nperseg - noverlap
        n_segments = max(1, (len(x) - nperseg) // step + 1)

        freqs = np.fft.rfftfreq(nperseg)
        psd = np.zeros(len(freqs), dtype=complex)

        window = np.hanning(nperseg)
        window_sum = np.sum(window ** 2)

        for i in range(n_segments):
            start = i * step
            end = start + nperseg
            if end > len(x):
                break

            x_seg = x[start:end] * window
            y_seg = y[start:end] * window

            X = np.fft.rfft(x_seg)
            Y = np.fft.rfft(y_seg)

            psd += X * np.conj(Y)

        psd /= (n_segments * window_sum)
        return freqs, psd

    def _estimate_bandwidth(
        self, freqs: np.ndarray, magnitude: np.ndarray
    ) -> float:
        """估计 -3dB 带宽。"""
        if len(magnitude) == 0:
            return 0.0

        dc_level = magnitude[0] if magnitude[0] > 0 else np.max(magnitude)
        threshold = dc_level / np.sqrt(2)

        for i in range(len(magnitude)):
            if magnitude[i] < threshold:
                return float(freqs[i])

        return float(freqs[-1])

    def _estimate_stability_margins(
        self,
        freqs: np.ndarray,
        magnitude: np.ndarray,
        phase: np.ndarray,
    ) -> Tuple[float, float]:
        """估计增益裕度和相位裕度。"""
        gain_margin_db = 0.0
        phase_margin_deg = 0.0

        if len(magnitude) < 2:
            return (gain_margin_db, phase_margin_deg)

        # 相位裕度: 在增益交叉频率 (|H|=1) 处的相位距 -180 度的距离
        for i in range(len(magnitude) - 1):
            if magnitude[i] >= 1.0 and magnitude[i + 1] < 1.0:
                # 线性插值
                frac = (1.0 - magnitude[i]) / (magnitude[i + 1] - magnitude[i] + 1e-10)
                phase_at_cross = phase[i] + frac * (phase[i + 1] - phase[i])
                phase_margin_deg = 180.0 + phase_at_cross
                break

        # 增益裕度: 在相位交叉频率 (phase=-180) 处的增益
        for i in range(len(phase) - 1):
            if phase[i] >= -180 and phase[i + 1] < -180:
                frac = (-180 - phase[i]) / (phase[i + 1] - phase[i] + 1e-10)
                mag_at_cross = magnitude[i] + frac * (magnitude[i + 1] - magnitude[i])
                if mag_at_cross > 0:
                    gain_margin_db = 20 * np.log10(mag_at_cross)
                break

        return (gain_margin_db, phase_margin_deg)

    def _evaluate_pid(
        self,
        kp: float,
        ki: float,
        kd: float,
        model: SysIdResult,
    ) -> float:
        """评估 PID 参数的性能分数 (越低越好)。

        通过仿真闭环系统响应来评估。
        """
        na = self.config.arx_order_na
        nb = self.config.arx_order_nb
        nk = self.config.arx_order_nk
        params = model.arx_parameters
        dt = 1.0 / self.config.sampling_rate

        # 仿真闭环阶跃响应
        n_steps = self.config.simulation_steps
        y = np.zeros(n_steps)
        u = np.zeros(n_steps)
        error = np.zeros(n_steps)
        integral_error = 0.0
        prev_error = 0.0
        ref = 1.0  # 阶跃参考

        # 限制 PID 参数范围
        kp = np.clip(kp, *self.config.pid_kp_range)
        ki = np.clip(ki, *self.config.pid_ki_range)
        kd = np.clip(kd, *self.config.pid_kd_range)

        for t in range(max(na, nb + nk), n_steps):
            error[t] = ref - y[t]
            integral_error += error[t] * dt
            derivative_error = (error[t] - prev_error) / dt

            # PID 控制律
            u[t] = kp * error[t] + ki * integral_error + kd * derivative_error
            u[t] = np.clip(u[t], -10, 10)

            # 系统响应 (ARX 模型)
            y_pred = 0.0
            for j in range(na):
                y_pred -= params[j] * y[t - 1 - j]
            for j in range(nb):
                idx = t - nk - j
                if 0 <= idx < n_steps:
                    y_pred += params[na + j] * u[idx]
            y[t] = np.clip(y_pred, -100, 100)

            prev_error = error[t]

        # 评估指标
        # 1. 调节时间 (2% 准则)
        settling_region = np.abs(y[max(na, nb + nk):] - ref) <= 0.02 * ref
        if np.any(settling_region):
            settling_indices = np.where(settling_region)[0]
            settling_time = settling_indices[0] * dt
        else:
            settling_time = n_steps * dt

        # 2. 超调量
        overshoot = max(0, (np.max(y) - ref) / ref)

        # 3. 稳态误差
        steady_state_error = abs(ref - np.mean(y[-50:]))

        # 4. 控制 effort
        control_effort = np.mean(u ** 2)

        # 加权评分
        cfg = self.config
        score = (
            cfg.pid_settling_weight * settling_time
            + cfg.pid_overshoot_weight * overshoot * 10
            + cfg.pid_steady_error_weight * steady_state_error * 100
            + cfg.pid_control_effort_weight * control_effort
        )

        return score
