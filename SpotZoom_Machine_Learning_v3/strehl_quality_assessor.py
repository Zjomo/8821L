"""
Strehl 质量评估器 (Strehl Quality Assessor)

参考开源项目:
  - prysm (https://github.com/brandondube/prysm): 光学衍射与像差分析库

核心思想:
  从 prysm 的 PSF 分析引擎中借鉴，实现一套完整的光斑质量评估工具，
  包括 Strehl 比、MTF、包围能量 (EE)、FWHM 等关键指标，
  以及 Zernike 多项式波前拟合。

  在 SpotZoom 场景中:
  - Strehl 比 → 光斑对准质量的综合指标
  - MTF → 系统空间频率响应
  - 包围能量 → 能量集中度
  - FWHM → 光斑尺寸
  - Zernike 拟合 → 像差分解与诊断

创新点:
  1. 噪声鲁棒的 Strehl 比估计 (傅里叶 Ring Test 方法)
  2. 快速 Zernike 多项式拟合 (最小二乘法)
  3. 2D MTF 计算与各向异性分析
  4. 径向包围能量曲线
  5. 综合质量评分 (加权多指标融合)

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
class StrehlConfig:
    """Strehl 质量评估器配置。"""
    # Zernike 多项式最大阶数
    zernike_max_order: int = 6
    # FWHM 拟合方法: "gaussian", "moffat"
    fwhm_fit_method: str = "gaussian"
    # 包围能量计算半径列表 (像素)
    ee_radii: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    # MTF 频率采样点数
    mtf_frequency_points: int = 64
    # 背景估计方法: "median", "ring", "corner"
    background_method: str = "median"
    # 背景估计区域宽度 (像素)
    background_margin: int = 10
    # Strehl 比估计方法: "peak", "energy", "fourier_ring"
    strehl_method: str = "peak"
    # 衍射极限 FWHM (像素，用于 Strehl 比计算)
    diffraction_limit_fwhm: float = 3.0
    # 质量评分权重
    strehl_weight: float = 0.3
    ee_weight: float = 0.25
    fwhm_weight: float = 0.2
    symmetry_weight: float = 0.15
    snr_weight: float = 0.1


@dataclass
class StrehlAssessment:
    """Strehl 质量评估结果。"""
    # Strehl 比 (0~1, 1 = 衍射极限)
    strehl_ratio: float = 0.0
    # 峰值 FWHM (像素)
    fwhm_x: float = 0.0
    fwhm_y: float = 0.0
    fwhm_avg: float = 0.0
    # 包围能量曲线 {radius: fraction}
    encircled_energy: Dict[int, float] = field(default_factory=dict)
    # 50% EE 半径 (像素)
    ee50_radius: float = 0.0
    # MTF 数据 (频率, 值)
    mtf_frequencies: np.ndarray = field(default_factory=lambda: np.zeros(64))
    mtf_values: np.ndarray = field(default_factory=lambda: np.zeros(64))
    # Zernike 系数
    zernike_coefficients: List[float] = field(default_factory=list)
    # 总 RMS 波前误差 (波数)
    rms_wavefront_error: float = 0.0
    # 峰值强度
    peak_intensity: float = 0.0
    # 背景均值
    background_mean: float = 0.0
    # 信噪比
    snr: float = 0.0
    # 对称性分数 (0~1)
    symmetry_score: float = 0.0
    # 综合质量分数 (0~100)
    quality_score: float = 0.0
    # 质心位置
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    # 是否有效
    valid: bool = True


class StrehlQualityAssessor:
    """Strehl 质量评估器。

    提供全面的光斑质量分析，包括 Strehl 比、MTF、包围能量、
    FWHM 和 Zernike 波前拟合。

    Parameters
    ----------
    config : StrehlConfig
        评估器配置参数。

    References
    ----------
    .. [1] prysm documentation:
           https://prysm.readthedocs.io/
    .. [2] Noll, R. J. (1976). "Zernike polynomials and atmospheric
           turbulence." JOSA, 66(3), 207-211.
    .. [3] Tektronix (2018). "Fundamentals of MTF."
    """

    def __init__(self, config: Optional[StrehlConfig] = None) -> None:
        self.config = config or StrehlConfig()
        # 预计算 Zernike 基函数
        self._zernike_basis: Optional[List[np.ndarray]] = None
        logger.info(
            f"StrehlQualityAssessor 初始化: "
            f"Zernike阶数={self.config.zernike_max_order}, "
            f"Strehl方法={self.config.strehl_method}"
        )

    def compute_strehl(self, image: np.ndarray) -> float:
        """计算 Strehl 比。

        Parameters
        ----------
        image : np.ndarray
            光斑图像 (2D)。

        Returns
        -------
        float
            Strehl 比 (0~1)。
        """
        bg = self._estimate_background(image)
        peak = float(np.max(image)) - bg

        if peak <= 0:
            logger.warning("图像峰值 <= 0，Strehl 比设为 0")
            return 0.0

        method = self.config.strehl_method

        if method == "peak":
            # 峰值 Strehl 比: 实际峰值 / 衍射极限峰值
            # 衍射极限峰值近似为 Airy 函数峰值
            diffraction_peak = 1.0  # 归一化
            strehl = peak / (diffraction_peak + bg)
            strehl = min(strehl, 1.0)

        elif method == "energy":
            # 能量 Strehl 比: 核心能量 / 总能量
            h, w = image.shape
            cy, cx = h // 2, w // 2
            core_radius = max(3, int(self.config.diffraction_limit_fwhm))
            yy, xx = np.ogrid[:h, :w]
            core_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= core_radius ** 2
            total_energy = np.sum(np.maximum(image - bg, 0))
            core_energy = np.sum(np.maximum(image[core_mask] - bg, 0))
            strehl = core_energy / (total_energy + 1e-10)
            strehl = min(strehl, 1.0)

        elif method == "fourier_ring":
            # 傅里叶 Ring Test 方法 (更鲁棒)
            strehl = self._fourier_ring_strehl(image)
        else:
            logger.warning(f"未知 Strehl 方法: {method}")
            strehl = 0.0

        return float(np.clip(strehl, 0, 1))

    def compute_mtf(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算调制传递函数 (MTF)。

        Parameters
        ----------
        image : np.ndarray
            光斑图像 (2D)。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (频率数组 cycles/pixel, MTF 值数组)。
        """
        bg = self._estimate_background(image)
        image_clean = np.maximum(image - bg, 0)

        # 2D OTF (光学传递函数)
        psf_centered = self._center_psf(image_clean)
        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf_centered)))
        otf_magnitude = np.abs(otf)
        otf_normalized = otf_magnitude / (np.max(otf_magnitude) + 1e-10)

        # 提取径向平均 MTF
        h, w = otf_normalized.shape
        cy, cx = h // 2, w // 2
        max_freq = min(cx, cy) / w
        frequencies = np.linspace(0, max_freq, self.config.mtf_frequency_points)

        mtf_values = np.zeros_like(frequencies)
        yy, xx = np.ogrid[:h, :w]
        radius_map = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / w

        for i, freq in enumerate(frequencies):
            # 在频率环上取平均
            ring_width = max_freq / self.config.mtf_frequency_points
            mask = np.abs(radius_map - freq) < ring_width
            if np.any(mask):
                mtf_values[i] = float(np.mean(otf_normalized[mask]))

        return frequencies, mtf_values

    def compute_ee(self, image: np.ndarray) -> Dict[int, float]:
        """计算包围能量。

        Parameters
        ----------
        image : np.ndarray
            光斑图像 (2D)。

        Returns
        -------
        Dict[int, float]
            各半径处的包围能量分数 {radius_px: fraction}。
        """
        bg = self._estimate_background(image)
        image_clean = np.maximum(image - bg, 0)

        h, w = image_clean.shape
        cy, cx = h // 2, w // 2
        total_energy = np.sum(image_clean)

        if total_energy < 1e-10:
            return {r: 0.0 for r in self.config.ee_radii}

        yy, xx = np.ogrid[:h, :w]
        dist_map = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        ee_dict: Dict[int, float] = {}
        for radius in self.config.ee_radii:
            mask = dist_map <= radius
            ee = float(np.sum(image_clean[mask]) / total_energy)
            ee_dict[radius] = ee

        return ee_dict

    def compute_fwhm(self, image: np.ndarray) -> Tuple[float, float]:
        """计算 FWHM (半高全宽)。

        Parameters
        ----------
        image : np.ndarray
            光斑图像 (2D)。

        Returns
        -------
        Tuple[float, float]
            (fwhm_x, fwhm_y) 像素。
        """
        bg = self._estimate_background(image)
        image_clean = np.maximum(image - bg, 0)
        peak = np.max(image_clean)

        if peak <= 0:
            return (0.0, 0.0)

        half_max = peak / 2.0

        # X 方向 FWHM
        row_profile = image_clean[image_clean.shape[0] // 2, :]
        fwhm_x = self._compute_1d_fwhm(row_profile, half_max)

        # Y 方向 FWHM
        col_profile = image_clean[:, image_clean.shape[1] // 2]
        fwhm_y = self._compute_1d_fwhm(col_profile, half_max)

        return (fwhm_x, fwhm_y)

    def fit_zernike(self, image: np.ndarray, size: int = 128) -> List[float]:
        """Zernike 多项式波前拟合。

        Parameters
        ----------
        image : np.ndarray
            光斑图像 (2D)。
        size : int
            拟合区域大小 (像素)。

        Returns
        -------
        List[float]
            Zernike 系数列表。
        """
        bg = self._estimate_background(image)
        image_clean = np.maximum(image - bg, 0)

        # 中心裁剪
        h, w = image_clean.shape
        cy, cx = h // 2, w // 2
        half = size // 2
        roi = image_clean[cy - half: cy + half, cx - half: cx + half]

        if roi.shape[0] != size or roi.shape[1] != size:
            logger.warning(f"ROI 尺寸 {roi.shape} 不匹配 {size}x{size}")
            return []

        # 生成 Zernike 基函数
        n_modes = self._zernike_mode_count(self.config.zernike_max_order)
        basis = self._generate_zernike_basis(size, n_modes)

        # 强度 → 振幅 (取平方根)
        amplitude = np.sqrt(roi)

        # 最小二乘拟合
        basis_matrix = np.column_stack([b.flatten() for b in basis])
        coeffs, _, _, _ = np.linalg.lstsq(basis_matrix, amplitude.flatten(), rcond=None)

        return [float(c) for c in coeffs]

    def assess_spot(self, image: np.ndarray) -> StrehlAssessment:
        """综合评估光斑质量。

        Parameters
        ----------
        image : np.ndarray
            光斑图像 (2D)。

        Returns
        -------
        StrehlAssessment
            综合评估结果。
        """
        result = StrehlAssessment()

        # 背景估计
        result.background_mean = float(self._estimate_background(image))
        result.peak_intensity = float(np.max(image))
        result.snr = float(
            (result.peak_intensity - result.background_mean)
            / (np.std(image[image < np.median(image)]) + 1e-10)
        )

        # 质心
        h, w = image.shape
        yy, xx = np.mgrid[:h, :w]
        total = np.sum(image)
        if total > 0:
            result.centroid_x = float(np.sum(xx * image) / total)
            result.centroid_y = float(np.sum(yy * image) / total)

        # Strehl 比
        result.strehl_ratio = self.compute_strehl(image)

        # FWHM
        result.fwhm_x, result.fwhm_y = self.compute_fwhm(image)
        result.fwhm_avg = (result.fwhm_x + result.fwhm_y) / 2.0

        # 包围能量
        result.encircled_energy = self.compute_ee(image)
        # 估计 EE50 半径
        result.ee50_radius = self._estimate_ee50_radius(result.encircled_energy)

        # MTF
        freqs, mtf_vals = self.compute_mtf(image)
        result.mtf_frequencies = freqs
        result.mtf_values = mtf_vals

        # Zernike 拟合
        result.zernike_coefficients = self.fit_zernike(image)
        if result.zernike_coefficients:
            # RMS 波前误差 (从 Zernike 系数估计)
            coeffs_arr = np.array(result.zernike_coefficients[4:])  # 跳过活塞、倾斜
            result.rms_wavefront_error = float(np.sqrt(np.mean(coeffs_arr ** 2)))

        # 对称性
        result.symmetry_score = self._compute_symmetry(image)

        # 综合质量评分 (0~100)
        result.quality_score = self._compute_quality_score(result)

        logger.info(
            f"光斑质量评估: Strehl={result.strehl_ratio:.3f}, "
            f"FWHM={result.fwhm_avg:.2f}px, "
            f"质量分={result.quality_score:.1f}/100"
        )
        return result

    # ============ 内部方法 ============

    def _estimate_background(self, image: np.ndarray) -> float:
        """估计背景强度。"""
        method = self.config.background_method
        margin = self.config.background_margin

        if method == "median":
            # 使用图像边缘中值
            border = np.concatenate([
                image[:margin, :].flatten(),
                image[-margin:, :].flatten(),
                image[:, :margin].flatten(),
                image[:, -margin:].flatten(),
            ])
            return float(np.median(border))

        elif method == "corner":
            # 使用四角区域
            h, w = image.shape
            m = margin
            corners = np.concatenate([
                image[:m, :m].flatten(),
                image[:m, -m:].flatten(),
                image[-m:, :m].flatten(),
                image[-m:, -m:].flatten(),
            ])
            return float(np.median(corners))

        elif method == "ring":
            # 使用环形区域
            h, w = image.shape
            cy, cx = h // 2, w // 2
            yy, xx = np.ogrid[:h, :w]
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            inner_r = min(h, w) * 0.3
            outer_r = min(h, w) * 0.45
            ring_mask = (r >= inner_r) & (r <= outer_r)
            if np.any(ring_mask):
                return float(np.median(image[ring_mask]))
            return float(np.median(image))
        else:
            return float(np.median(image))

    def _center_psf(self, image: np.ndarray) -> np.ndarray:
        """将 PSF 居中。"""
        h, w = image.shape
        cy, cx = h // 2, w // 2
        total = np.sum(image)
        if total <= 0:
            return image
        yy, xx = np.mgrid[:h, :w]
        com_y = float(np.sum(yy * image) / total)
        com_x = float(np.sum(xx * image) / total)
        # 使用 FFT 的循环移位
        shift_y = int(round(cy - com_y))
        shift_x = int(round(cx - com_x))
        return np.roll(np.roll(image, shift_y, axis=0), shift_x, axis=1)

    def _compute_1d_fwhm(self, profile: np.ndarray, half_max: float) -> float:
        """计算 1D 剖面的 FWHM。"""
        above = profile >= half_max
        if not np.any(above):
            return 0.0
        indices = np.where(above)[0]
        return float(indices[-1] - indices[0] + 1)

    def _fourier_ring_strehl(self, image: np.ndarray) -> float:
        """傅里叶 Ring Test 估计 Strehl 比。"""
        bg = self._estimate_background(image)
        image_clean = np.maximum(image - bg, 0)

        # 计算 OTF
        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
            self._center_psf(image_clean)
        )))

        # 在低频区域 (核心) 和中频区域取比值
        h, w = otf.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # 核心区域 (低频)
        core_mask = r <= 3
        # 中频区域
        mid_mask = (r > 5) & (r <= 15)

        if not np.any(core_mask) or not np.any(mid_mask):
            return 0.0

        core_power = np.mean(np.abs(otf[core_mask]) ** 2)
        mid_power = np.mean(np.abs(otf[mid_mask]) ** 2)

        if mid_power < 1e-10:
            return 1.0

        # Strehl 比近似
        strehl = core_power / (core_power + mid_power)
        return float(np.clip(strehl, 0, 1))

    def _compute_symmetry(self, image: np.ndarray) -> float:
        """计算 PSF 对称性分数。"""
        bg = self._estimate_background(image)
        image_clean = np.maximum(image - bg, 0)
        image_clean = self._center_psf(image_clean)

        h, w = image_clean.shape
        cy, cx = h // 2, w // 2

        # 水平对称性
        left = image_clean[:, :cx]
        right = image_clean[:, cx:]
        min_w = min(left.shape[1], right.shape[1])
        left_cropped = left[:, :min_w]
        right_cropped = right[:, -min_w:]
        h_sym = 1.0 - float(np.mean(np.abs(left_cropped - right_cropped[::-1]))) / (
            np.mean(image_clean) + 1e-10
        )

        # 垂直对称性
        top = image_clean[:cy, :]
        bottom = image_clean[cy:, :]
        min_h = min(top.shape[0], bottom.shape[0])
        top_cropped = top[:min_h, :]
        bottom_cropped = bottom[-min_h:, :]
        v_sym = 1.0 - float(np.mean(np.abs(top_cropped - bottom_cropped[::-1]))) / (
            np.mean(image_clean) + 1e-10
        )

        return float(np.clip((h_sym + v_sym) / 2.0, 0, 1))

    def _compute_quality_score(self, result: StrehlAssessment) -> float:
        """计算综合质量评分 (0~100)。"""
        cfg = self.config
        score = 0.0

        # Strehl 比贡献 (0~1 → 0~100)
        score += result.strehl_ratio * 100 * cfg.strehl_weight

        # EE50 贡献 (越小越好)
        if result.ee50_radius > 0:
            ee_score = max(0, 1.0 - result.ee50_radius / 20.0)
            score += ee_score * 100 * cfg.ee_weight

        # FWHM 贡献 (越接近衍射极限越好)
        if result.fwhm_avg > 0:
            fwhm_score = max(0, 1.0 - result.fwhm_avg / (3 * cfg.diffraction_limit_fwhm))
            score += fwhm_score * 100 * cfg.fwhm_weight

        # 对称性贡献
        score += result.symmetry_score * 100 * cfg.symmetry_weight

        # SNR 贡献
        snr_score = min(1.0, result.snr / 50.0)
        score += snr_score * 100 * cfg.snr_weight

        return float(np.clip(score, 0, 100))

    def _estimate_ee50_radius(self, ee_dict: Dict[int, float]) -> float:
        """从包围能量数据估计 50% EE 半径。"""
        radii = sorted(ee_dict.keys())
        values = [ee_dict[r] for r in radii]

        for i in range(len(values) - 1):
            if values[i] <= 0.5 <= values[i + 1]:
                # 线性插值
                r1, r2 = radii[i], radii[i + 1]
                v1, v2 = values[i], values[i + 1]
                if v2 - v1 > 0:
                    return float(r1 + (0.5 - v1) * (r2 - r1) / (v2 - v1))

        return float(radii[-1]) if radii else 0.0

    def _zernike_mode_count(self, max_order: int) -> int:
        """计算 Zernike 模式总数。"""
        n = 0
        j = 0
        while True:
            radial_order = int((-1 + np.sqrt(1 + 8 * j)) / 2)
            if radial_order > max_order:
                break
            n += 1
            j += 1
        return n

    def _generate_zernike_basis(
        self, size: int, n_modes: int
    ) -> List[np.ndarray]:
        """生成 Zernike 基函数。

        使用 Noll 索引约定生成前 n_modes 个 Zernike 多项式。
        """
        basis = []
        yy, xx = np.mgrid[:size, :size]
        x = (xx - size / 2 + 0.5) / (size / 2)
        y = (yy - size / 2 + 0.5) / (size / 2)
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)

        # 单位圆掩膜
        mask = r <= 1.0

        for j in range(n_modes):
            n, m = self._noll_to_nm(j)
            zernike = self._zernike_polynomial(n, m, r, theta) * mask
            # 归一化
            norm = np.sqrt(np.sum(zernike ** 2))
            if norm > 1e-10:
                zernike /= norm
            basis.append(zernike)

        return basis

    def _noll_to_nm(self, j: int) -> Tuple[int, int]:
        """Noll 索引 → (n, m) 径向阶数和角频率。"""
        n = int((-1 + np.sqrt(1 + 8 * j)) / 2)
        m_values = list(range(-n, n + 1, 2))
        k = j - n * (n + 1) // 2
        m = m_values[k] if k < len(m_values) else 0
        return n, m

    def _zernike_polynomial(
        self, n: int, m: int, r: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """计算单个 Zernike 多项式。"""
        if n == 0:
            return np.ones_like(r)
        if n == 1 and m == 1:
            return r * np.cos(theta)
        if n == 1 and m == -1:
            return r * np.sin(theta)

        # 径向多项式
        radial = np.zeros_like(r)
        for s in range((n - abs(m)) // 2 + 1):
            coeff = (
                (-1) ** s
                * math.factorial(n - s)
                / (
                    math.factorial(s)
                    * math.factorial((n + abs(m)) // 2 - s)
                    * math.factorial((n - abs(m)) // 2 - s)
                )
            )
            radial += coeff * r ** (n - 2 * s)

        if m > 0:
            return radial * np.cos(m * theta)
        elif m < 0:
            return radial * np.sin(abs(m) * theta)
        else:
            return radial
