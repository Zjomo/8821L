"""
光学传递函数分析器 (OTFAnalyzer)

灵感来源:
- HCIPy (https://github.com/ehpor/hcipy) — 光学传播与衍射仿真
- Prysm (https://github.com/brandondube/prysm) — OTF/MTF/PSF 分析工具
- Popov (2020) — 实用光学系统 MTF 测量方法

算法原理:
- Optical Transfer Function (OTF) — 光学系统频率响应函数
- Modulation Transfer Function (MTF) — OTF 模值，描述对比度传递
- Phase Transfer Function (PTF) — OTF 相位，描述相位传递
- Wiener-Khinchin Theorem — 自相关函数与功率谱密度的傅里叶变换关系

功能:
- 从 PSF 图像计算 OTF/MTF/PTF
- 截止频率检测
- 从 OTF 形状诊断像差类型
- 各方向 MTF 分析 (子午/弧矢)
- 衍射极限 MTF 参考线计算

依赖: numpy, scipy (无外部深度学习框架依赖)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.OTFAnalyzer")


@dataclass
class OTFConfig:
    """OTF 分析器配置参数。"""
    # --- 图像参数 ---
    pixel_size: float = 0.1  # 像素大小 (微米)
    wavelength: float = 0.55  # 波长 (微米)
    na: float = 0.95  # 数值孔径
    magnification: float = 100.0  # 放大倍率

    # --- 分析参数 ---
    pad_factor: int = 4  # 零填充因子 (减少频谱泄漏)
    window_type: str = "hann"  # 窗函数类型: "hann", "hamming", "blackman", "none"
    normalize: bool = True  # 是否归一化 OTF

    # --- MTF 分析 ---
    mtf_directions: List[str] = field(default_factory=lambda: ["horizontal", "vertical", "radial"])
    frequency_bins: int = 100  # 频率采样点数

    # --- 诊断参数 ---
    aberration_threshold: float = 0.1  # 像差诊断阈值


@dataclass
class OTFResult:
    """OTF 分析结果。"""
    otf: np.ndarray = field(default_factory=lambda: np.array([]))
    mtf: np.ndarray = field(default_factory=lambda: np.array([]))
    ptf: np.ndarray = field(default_factory=lambda: np.array([]))
    frequency_axes: Tuple[np.ndarray, np.ndarray] = (
        field(default_factory=lambda: np.array([])),
        field(default_factory=lambda: np.array([])),
    )
    cutoff_frequency: float = 0.0  # 截止频率 (cycles/微米)
    mtf50_frequency: float = 0.0  # MTF50 频率
    mtf_directional: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    strehl_ratio: float = 0.0
    aberration_diagnosis: Dict[str, float] = field(default_factory=dict)
    psf_centroid: Tuple[float, float] = (0.0, 0.0)
    psf_width: Tuple[float, float] = (0.0, 0.0)


class OTFAnalyzer:
    """光学传递函数分析器。

    从光斑 (PSF) 图像计算光学传递函数 (OTF)、调制传递函数 (MTF)
    和相位传递函数 (PTF)，并进行像差诊断。

    Parameters
    ----------
    config : OTFConfig, optional
        分析器配置参数。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[OTFConfig] = None):
        self._cfg = config if config is not None else OTFConfig()

    def _apply_window(self, psf: np.ndarray) -> np.ndarray:
        """对 PSF 应用窗函数以减少频谱泄漏。"""
        if self._cfg.window_type == "none":
            return psf

        h, w = psf.shape
        if self._cfg.window_type == "hann":
            win_x = np.hanning(w)
            win_y = np.hanning(h)
        elif self._cfg.window_type == "hamming":
            win_x = np.hamming(w)
            win_y = np.hamming(h)
        elif self._cfg.window_type == "blackman":
            win_x = np.blackman(w)
            win_y = np.blackman(h)
        else:
            raise ValueError(f"未知窗函数类型: {self._cfg.window_type}")

        window = np.outer(win_y, win_x)
        return psf * window

    def _compute_cutoff_frequency(self) -> float:
        """计算衍射极限截止频率。

        Returns
        -------
        float
            截止频率 (cycles/微米)。
        """
        # 截止频率 = 2 * NA / lambda
        f_cutoff = 2.0 * self._cfg.na / self._cfg.wavelength
        return f_cutoff

    def _compute_diffraction_limited_mtf(
        self, frequencies: np.ndarray
    ) -> np.ndarray:
        """计算衍射极限 MTF 参考线。

        对于圆形孔径，衍射极限 MTF 为:
        MTF(f) = (2/pi) * [arccos(f/fc) - (f/fc)*sqrt(1-(f/fc)^2)]

        Parameters
        ----------
        frequencies : np.ndarray
            频率数组 (cycles/微米)。

        Returns
        -------
        np.ndarray
            衍射极限 MTF 值。
        """
        f_cutoff = self._compute_cutoff_frequency()
        normalized_f = frequencies / f_cutoff
        mtf = np.zeros_like(frequencies, dtype=np.float64)

        mask = (normalized_f >= 0) & (normalized_f < 1.0)
        fn = normalized_f[mask]
        mtf[mask] = (2.0 / np.pi) * (
            np.arccos(fn) - fn * np.sqrt(1.0 - fn ** 2)
        )

        return mtf

    def _diagnose_aberrations(
        self, mtf: np.ndarray, ptf: np.ndarray
    ) -> Dict[str, float]:
        """从 OTF 形状诊断像差类型。

        Parameters
        ----------
        mtf : np.ndarray
            MTF 二维数组。
        ptf : np.ndarray
            PTF 二维数组。

        Returns
        -------
        Dict[str, float]
            像差诊断结果 (像差名称 -> 置信度)。
        """
        diagnosis: Dict[str, float] = {}
        h, w = mtf.shape
        cy, cx = h // 2, w // 2

        # 提取水平和垂直方向的 MTF 切片
        h_mtf = mtf[cy, cx:]
        v_mtf = mtf[cy:, cx]

        # --- 离焦检测 ---
        # 离焦导致 MTF 在中频段下降，低频保持
        if len(h_mtf) > 10:
            low_freq_mtf = np.mean(h_mtf[:len(h_mtf) // 4])
            mid_freq_mtf = np.mean(h_mtf[len(h_mtf) // 4:len(h_mtf) // 2])
            if low_freq_mtf > 0 and mid_freq_mtf / low_freq_mtf < 0.5:
                diagnosis["离焦 (Defocus)"] = 1.0 - mid_freq_mtf / low_freq_mtf

        # --- 像散检测 ---
        # 像散导致子午和弧矢方向 MTF 不对称
        if len(h_mtf) > 10 and len(v_mtf) > 10:
            min_len = min(len(h_mtf), len(v_mtf))
            h_slice = h_mtf[:min_len]
            v_slice = v_mtf[:min_len]
            asymmetry = np.mean(np.abs(h_slice - v_slice))
            if asymmetry > self._cfg.aberration_threshold:
                diagnosis["像散 (Astigmatism)"] = min(asymmetry, 1.0)

        # --- 彗差检测 ---
        # 彗差导致 PTF 不对称
        ptf_center = ptf[cy, cx:]
        if len(ptf_center) > 10:
            ptf_asymmetry = np.std(np.diff(ptf_center))
            if ptf_asymmetry > self._cfg.aberration_threshold:
                diagnosis["彗差 (Coma)"] = min(ptf_asymmetry, 1.0)

        # --- 球差检测 ---
        # 球差导致高频 MTF 显著下降
        if len(h_mtf) > 10:
            high_freq_mtf = np.mean(h_mtf[3 * len(h_mtf) // 4:])
            if high_freq_mtf < 0.1:
                diagnosis["球差 (Spherical)"] = 1.0 - high_freq_mtf / 0.1

        return diagnosis

    def _extract_directional_mtf(
        self, mtf_2d: np.ndarray, freq_x: np.ndarray, freq_y: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """提取各方向的 MTF 切片。

        Parameters
        ----------
        mtf_2d : np.ndarray
            二维 MTF 数组。
        freq_x : np.ndarray
            X 方向频率轴。
        freq_y : np.ndarray
            Y 方向频率轴。

        Returns
        -------
        Dict[str, Tuple[np.ndarray, np.ndarray]]
            方向名称到 (频率, MTF值) 的映射。
        """
        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        h, w = mtf_2d.shape
        cy, cx = h // 2, w // 2

        # 水平方向 (子午)
        if "horizontal" in self._cfg.mtf_directions:
            result["horizontal"] = (freq_x[cx:], mtf_2d[cy, cx:])

        # 垂直方向 (弧矢)
        if "vertical" in self._cfg.mtf_directions:
            result["vertical"] = (freq_y[cy:], mtf_2d[cy:, cx])

        # 径向平均
        if "radial" in self._cfg.mtf_directions:
            max_radius = min(cx, cy)
            radii = np.arange(1, max_radius)
            radial_mtf = np.zeros(len(radii), dtype=np.float64)
            for i, r in enumerate(radii):
                # 在半径 r 处采样 MTF
                angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                samples = []
                for angle in angles:
                    yi = int(round(cy + r * np.sin(angle)))
                    xi = int(round(cx + r * np.cos(angle)))
                    if 0 <= yi < h and 0 <= xi < w:
                        samples.append(mtf_2d[yi, xi])
                if samples:
                    radial_mtf[i] = np.mean(samples)

            # 将半径转换为频率
            freq_scale = freq_x[1] - freq_x[0] if len(freq_x) > 1 else 1.0
            radial_freq = radii * freq_scale
            result["radial"] = (radial_freq, radial_mtf)

        # 45度对角方向
        if "diagonal" in self._cfg.mtf_directions:
            diag_len = min(cx, cy)
            diag_mtf = np.array([
                mtf_2d[cy + i, cx + i] for i in range(diag_len)
            ])
            diag_freq = np.arange(diag_len) * (freq_x[1] - freq_x[0]) * np.sqrt(2)
            result["diagonal"] = (diag_freq, diag_mtf)

        return result

    def analyze(self, psf_image: np.ndarray) -> OTFResult:
        """分析 PSF 图像，计算 OTF/MTF/PTF。

        Parameters
        ----------
        psf_image : np.ndarray
            PSF 图像 (二维灰度图)。

        Returns
        -------
        OTFResult
            OTF 分析结果。
        """
        psf = np.asarray(psf_image, dtype=np.float64)

        if psf.ndim != 2:
            raise ValueError(f"PSF 图像应为二维数组，实际维度: {psf.ndim}")

        LOGGER.info("开始 OTF 分析: PSF 形状 %s", psf.shape)

        # --- PSF 预处理 ---
        # 减去背景
        psf = psf - np.percentile(psf, 1)
        psf = np.clip(psf, 0, None)

        # 应用窗函数
        psf_windowed = self._apply_window(psf)

        # 零填充
        pad = self._cfg.pad_factor
        h, w = psf.shape
        ph, pw = h * pad, w * pad
        psf_padded = np.zeros((ph, pw), dtype=np.float64)
        psf_padded[:h, :w] = psf_windowed

        # --- 计算 OTF ---
        # OTF = FFT(PSF) / |FFT(PSF)|_max (归一化)
        otf = np.fft.fftshift(np.fft.fft2(psf_padded))

        if self._cfg.normalize:
            otf = otf / (np.abs(otf).max() + 1e-12)

        # --- 计算 MTF 和 PTF ---
        mtf = np.abs(otf)
        ptf = np.angle(otf)

        # --- 频率轴 ---
        # 采样间隔 (物空间频率)
        pixel_size_obj = self._cfg.pixel_size / self._cfg.magnification
        df = 1.0 / (pw * pixel_size_obj)  # 频率间隔 (cycles/微米)

        freq_x = np.fft.fftshift(np.fft.fftfreq(pw, d=pixel_size_obj))
        freq_y = np.fft.fftshift(np.fft.fftfreq(ph, d=pixel_size_obj))

        # --- 截止频率 ---
        f_cutoff = self._compute_cutoff_frequency()

        # --- MTF50 ---
        # 从径向 MTF 中找到 MTF=0.5 的频率
        h_mtf = mtf[ph // 2, pw // 2:]
        h_freq = freq_x[pw // 2:]
        mtf50 = 0.0
        for i in range(1, len(h_mtf)):
            if h_mtf[i] <= 0.5:
                # 线性插值
                if h_mtf[i - 1] - h_mtf[i] > 1e-12:
                    frac = (h_mtf[i - 1] - 0.5) / (h_mtf[i - 1] - h_mtf[i])
                    mtf50 = h_freq[i - 1] + frac * (h_freq[i] - h_freq[i - 1])
                break

        # --- Strehl 比 ---
        strehl = float(np.max(psf) / (np.sum(psf) + 1e-12) * psf.size)

        # --- PSF 质心与宽度 ---
        total = np.sum(psf) + 1e-12
        cy_ps, cx_ps = np.mgrid[:h, :w]
        centroid_y = float(np.sum(cy_ps * psf) / total)
        centroid_x = float(np.sum(cx_ps * psf) / total)

        # 二阶矩宽度
        width_y = float(np.sqrt(np.sum((cy_ps - centroid_y) ** 2 * psf) / total))
        width_x = float(np.sqrt(np.sum((cx_ps - centroid_x) ** 2 * psf) / total))

        # --- 方向 MTF ---
        mtf_directional = self._extract_directional_mtf(mtf, freq_x, freq_y)

        # --- 像差诊断 ---
        aberration_diagnosis = self._diagnose_aberrations(mtf, ptf)

        result = OTFResult(
            otf=otf,
            mtf=mtf,
            ptf=ptf,
            frequency_axes=(freq_x, freq_y),
            cutoff_frequency=f_cutoff,
            mtf50_frequency=mtf50,
            mtf_directional=mtf_directional,
            strehl_ratio=strehl,
            aberration_diagnosis=aberration_diagnosis,
            psf_centroid=(centroid_x, centroid_y),
            psf_width=(width_x, width_y),
        )

        LOGGER.info(
            "OTF 分析完成: 截止频率=%.1f cy/um, MTF50=%.1f cy/um, "
            "Strehl=%.3f, 诊断=%s",
            f_cutoff, mtf50, strehl,
            list(aberration_diagnosis.keys()),
        )

        return result

    def compute_diffraction_limited_mtf(
        self, frequencies: np.ndarray
    ) -> np.ndarray:
        """计算衍射极限 MTF 参考线。

        Parameters
        ----------
        frequencies : np.ndarray
            频率数组 (cycles/微米)。

        Returns
        -------
        np.ndarray
            衍射极限 MTF 值。
        """
        return self._compute_diffraction_limited_mtf(frequencies)

    def compare_mtf(
        self,
        mtf_measured: np.ndarray,
        mtf_reference: np.ndarray,
        frequencies: np.ndarray,
    ) -> Dict[str, float]:
        """比较实测 MTF 与参考 MTF。

        Parameters
        ----------
        mtf_measured : np.ndarray
            实测 MTF 值。
        mtf_reference : np.ndarray
            参考 MTF 值 (如衍射极限)。
        frequencies : np.ndarray
            频率数组。

        Returns
        -------
        Dict[str, float]
            比较指标。
        """
        if len(mtf_measured) != len(mtf_reference):
            min_len = min(len(mtf_measured), len(mtf_reference))
            mtf_measured = mtf_measured[:min_len]
            mtf_reference = mtf_reference[:min_len]
            frequencies = frequencies[:min_len]

        diff = mtf_measured - mtf_reference
        rms_diff = float(np.sqrt(np.mean(diff ** 2)))
        max_diff = float(np.max(np.abs(diff)))
        area_diff = float(np.trapz(mtf_measured - mtf_reference, frequencies))

        return {
            "rms_difference": rms_diff,
            "max_difference": max_diff,
            "area_difference": area_diff,
            "relative_mtf_area": float(
                np.trapz(mtf_measured, frequencies)
                / (np.trapz(mtf_reference, frequencies) + 1e-12)
            ),
        }

    def reset(self):
        """重置分析器状态。"""
        LOGGER.info("OTF 分析器已重置")
