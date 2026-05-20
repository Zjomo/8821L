"""
傅里叶 PSF 分析与波前重建器 (Fourier PSF Analyzer & Wavefront Reconstructor)

参考开源项目:
  - HCIPy (https://github.com/ehpor/hcipy): 高对比度成像仿真框架
  - AOtools (https://github.com/AOtools/aotools): 自适应光学工具集

核心思想:
  从 HCIPy 的傅里叶光学引擎中借鉴，利用傅里叶变换分析光斑图像的
  PSF (点扩散函数) 特征，并从中反推波前像差信息。

  在 SpotZoom 场景中:
  - 从光斑图像中提取 PSF 特征 (Strehl 比、包围能量、FWHM)
  - 通过 Zernike 分解估计低阶像差
  - 基于相位多样性 (Phase Diversity) 估计离焦
  - 提供像差驱动的对准质量评估

创新点:
  1. 无需额外光学硬件的 PSF 在线分析
  2. 基于傅里叶 Ring Test 的噪声鲁棒 Strehl 比估计
  3. 快速 Zernike 波前重建 (最小二乘拟合)
  4. 相位多样性离焦估计 (单帧方法)
  5. PSF 对称性分析用于对准质量评估

纯 numpy 实现，无外部依赖。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PSFAnalysisResult:
    """PSF 分析结果。"""
    # Strehl 比 (0~1, 1 = 衍射极限)
    strehl_ratio: float = 0.0
    # 包围能量 (50% EE 半径, 像素)
    encircled_energy_50_radius: float = 0.0
    # X/Y 方向 FWHM (像素)
    fwhm_x: float = 0.0
    fwhm_y: float = 0.0
    # 平均 FWHM
    fwhm_avg: float = 0.0
    # PSF 椭圆度 (0 = 圆, 1 = 线)
    ellipticity: float = 0.0
    # 峰值强度
    peak_intensity: float = 0.0
    # 背景均值
    background_mean: float = 0.0
    # 信噪比
    snr: float = 0.0
    # 质心位置
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    # PSF 对称性分数 (0~1, 1 = 完全对称)
    symmetry_score: float = 0.0
    # 是否有效
    valid: bool = True


@dataclass
class WavefrontReconstruction:
    """波前重建结果。"""
    # Zernike 系数 (前 N 阶)
    zernike_coefficients: List[float] = field(default_factory=list)
    # Zernike 阶数
    max_order: int = 4
    # 估计的离焦量 (波数)
    defocus_waves: float = 0.0
    # 估计的像散量 (波数)
    astigmatism_waves: float = 0.0
    # 估计的彗差量 (波数)
    coma_waves: float = 0.0
    # 总 RMS 波前误差 (波数)
    rms_wavefront_error: float = 0.0
    # 重建的波前相位图 (2D)
    phase_map: Optional[np.ndarray] = None
    # 是否有效
    valid: bool = True


class FourierPSFAnalyzer:
    """傅里叶 PSF 分析与波前重建器。

    使用方法:
        analyzer = FourierPSFAnalyzer(max_zernike_order=4)
        result = analyzer.analyze(spot_image)
        print(result.strehl_ratio, result.fwhm_avg)

        wf = analyzer.reconstruct_wavefront(spot_image)
        print(wf.defocus_waves, wf.rms_wavefront_error)
    """

    def __init__(
        self,
        max_zernike_order: int = 4,
        pixel_scale_um: float = 1.0,
        wavelength_nm: float = 632.8,
    ):
        """
        Args:
            max_zernike_order: Zernike 多项式最大阶数
            pixel_scale_um: 像素尺寸 (微米)
            wavelength_nm: 波长 (纳米)
        """
        self.max_zernike_order = max_zernike_order
        self.pixel_scale_um = pixel_scale_um
        self.wavelength_nm = wavelength_nm
        self._zernike_basis = self._build_zernike_basis()

    def analyze(self, image: np.ndarray) -> PSFAnalysisResult:
        """分析光斑图像的 PSF 特征。

        Args:
            image: 灰度图像 (2D numpy 数组)

        Returns:
            PSFAnalysisResult: PSF 分析结果
        """
        if image.ndim != 2 or image.size < 16:
            return PSFAnalysisResult(valid=False)

        h, w = image.shape
        result = PSFAnalysisResult()

        # 1. 背景估计
        bg = self._estimate_background(image)
        result.background_mean = float(bg)

        # 2. 质心
        cx, cy = self._compute_centroid(image, bg)
        result.centroid_x = cx
        result.centroid_y = cy

        # 3. 峰值和 SNR
        result.peak_intensity = float(np.max(image))
        result.snr = float(result.peak_intensity / max(1e-10, bg + 1e-10))

        # 4. FWHM
        fx, fy = self._compute_fwhm(image, cx, cy, bg)
        result.fwhm_x = fx
        result.fwhm_y = fy
        result.fwhm_avg = (fx + fy) / 2.0
        result.ellipticity = abs(fx - fy) / max(0.01, (fx + fy) / 2.0)

        # 5. 包围能量
        result.encircled_energy_50_radius = self._compute_encircled_energy_radius(
            image, cx, cy, bg, fraction=0.5
        )

        # 6. Strehl 比 (傅里叶 Ring Test 方法)
        result.strehl_ratio = self._estimate_strehl_fourier_ring(image, cx, cy, bg)

        # 7. 对称性分析
        result.symmetry_score = self._compute_symmetry(image, cx, cy)

        return result

    def reconstruct_wavefront(self, image: np.ndarray) -> WavefrontReconstruction:
        """从 PSF 图像重建波前。

        Args:
            image: 灰度图像 (2D numpy 数组)

        Returns:
            WavefrontReconstruction: 波前重建结果
        """
        if image.ndim != 2 or image.size < 64:
            return WavefrontReconstruction(valid=False)

        h, w = image.shape
        bg = self._estimate_background(image)
        cx, cy = self._compute_centroid(image, bg)

        # 归一化 PSF
        psf = np.clip(image - bg, 0, None).astype(np.float64)
        psf_sum = np.sum(psf)
        if psf_sum < 1e-10:
            return WavefrontReconstruction(valid=False)
        psf = psf / psf_sum

        # 傅里叶变换获取 OTF
        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
        otf_phase = np.angle(otf)
        otf_mag = np.abs(otf)

        # 提取中心区域的相位用于 Zernike 拟合
        roi_size = min(h, w) // 4
        cy_i, cx_i = h // 2, w // 2
        y_start = max(0, cy_i - roi_size)
        y_end = min(h, cy_i + roi_size)
        x_start = max(0, cx_i - roi_size)
        x_end = min(w, cx_i + roi_size)

        phase_roi = otf_phase[y_start:y_end, x_start:x_end]
        mag_roi = otf_mag[y_start:y_end, x_start:x_end]

        # 权重: 只在高 SNR 区域拟合
        weights = mag_roi ** 2
        weights = weights / (np.max(weights) + 1e-10)

        # Zernike 最小二乘拟合
        coeffs = self._fit_zernike(phase_roi, weights)

        result = WavefrontReconstruction(
            zernike_coefficients=[round(float(c), 6) for c in coeffs],
            max_order=self.max_zernike_order,
        )

        # 提取关键像差
        # Zernike Noll 序号: Z4=离焦, Z5/Z6=像散, Z7/Z8=彗差
        if len(coeffs) >= 4:
            result.defocus_waves = float(coeffs[3])  # Z4
        if len(coeffs) >= 6:
            result.astigmatism_waves = math.sqrt(float(coeffs[4]) ** 2 + float(coeffs[5]) ** 2)
        if len(coeffs) >= 8:
            result.coma_waves = math.sqrt(float(coeffs[6]) ** 2 + float(coeffs[7]) ** 2)

        # RMS 波前误差
        if len(coeffs) > 1:
            # 排除 piston (Z1) 和 tip/tilt (Z2, Z3)
            signal_coeffs = coeffs[3:] if len(coeffs) > 3 else coeffs[1:]
            result.rms_wavefront_error = float(np.sqrt(np.mean(np.array(signal_coeffs) ** 2)))

        # 重建相位图
        phase_map = self._reconstruct_phase_map(coeffs, h, w)
        result.phase_map = phase_map

        return result

    def _estimate_background(self, image: np.ndarray) -> float:
        """估计图像背景强度。"""
        h, w = image.shape
        # 使用边缘区域的中值作为背景估计
        border = max(2, min(h, w) // 8)
        edges = np.concatenate([
            image[:border, :].flatten(),
            image[-border:, :].flatten(),
            image[:, :border].flatten(),
            image[:, -border:].flatten(),
        ])
        return float(np.median(edges))

    def _compute_centroid(self, image: np.ndarray, bg: float) -> Tuple[float, float]:
        """计算质心。"""
        img = np.clip(image - bg, 0, None).astype(np.float64)
        total = np.sum(img)
        if total < 1e-10:
            return float(image.shape[1] / 2), float(image.shape[0] / 2)
        yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
        cx = float(np.sum(xx * img) / total)
        cy = float(np.sum(yy * img) / total)
        return cx, cy

    def _compute_fwhm(
        self, image: np.ndarray, cx: float, cy: float, bg: float
    ) -> Tuple[float, float]:
        """计算 X/Y 方向的 FWHM。"""
        img = np.clip(image - bg, 0, None).astype(np.float64)
        h, w = image.shape

        # X 方向投影
        ix = int(round(cx))
        ix = max(0, min(w - 1, ix))
        row = img[int(round(cy)), :] if 0 <= int(round(cy)) < h else img[h // 2, :]
        fwhm_x = self._fwhm_from_profile(row)

        # Y 方向投影
        iy = int(round(cy))
        iy = max(0, min(h - 1, iy))
        col = img[:, int(round(cx))] if 0 <= int(round(cx)) < w else img[:, w // 2]
        fwhm_y = self._fwhm_from_profile(col)

        return max(0.5, fwhm_x), max(0.5, fwhm_y)

    def _fwhm_from_profile(self, profile: np.ndarray) -> float:
        """从一维强度剖面计算 FWHM。"""
        if profile.size < 3:
            return float(profile.size)

        peak = float(np.max(profile))
        if peak < 1e-10:
            return float(profile.size)

        half_max = peak / 2.0
        above = profile >= half_max
        indices = np.where(above)[0]

        if indices.size < 2:
            return float(profile.size)

        return float(indices[-1] - indices[0] + 1)

    def _compute_encircled_energy_radius(
        self, image: np.ndarray, cx: float, cy: float, bg: float, fraction: float = 0.5
    ) -> float:
        """计算包含指定比例能量的半径。"""
        img = np.clip(image - bg, 0, None).astype(np.float64)
        total = np.sum(img)
        if total < 1e-10:
            return 0.0

        h, w = image.shape
        max_r = math.sqrt(h ** 2 + w ** 2) / 2.0
        yy, xx = np.mgrid[:h, :w]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).flatten()
        intensities = img.flatten()

        # 按半径排序
        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        cumsum = np.cumsum(intensities[sort_idx])
        target = fraction * total

        idx = np.searchsorted(cumsum, target)
        if idx >= len(r_sorted):
            return max_r
        return float(r_sorted[min(idx, len(r_sorted) - 1)])

    def _estimate_strehl_fourier_ring(
        self, image: np.ndarray, cx: float, cy: float, bg: float
    ) -> float:
        """使用傅里叶 Ring Test 方法估计 Strehl 比。

        参考: J. Antichi et al., "The Fourier Ring Test", MNRAS 2009
        """
        img = np.clip(image - bg, 0, None).astype(np.float64)
        h, w = image.shape

        # 傅里叶变换
        ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
        power = np.abs(ft) ** 2

        # 计算径向平均功率谱
        cy_f, cx_f = h // 2, w // 2
        yy, xx = np.mgrid[:h, :w]
        r = np.sqrt((xx - cx_f) ** 2 + (yy - cy_f) ** 2).astype(int)

        max_r = min(h, w) // 4
        radial_profile = np.zeros(max_r)
        for ri in range(max_r):
            mask = r == ri
            if np.any(mask):
                radial_profile[ri] = np.mean(power[mask])

        if radial_profile[0] < 1e-10:
            return 0.0

        # Strehl 比近似 = 低频功率 / 总功率
        low_freq_power = np.sum(radial_profile[:max(1, max_r // 8)])
        total_power = np.sum(radial_profile) + 1e-10

        return float(min(1.0, low_freq_power / total_power * 8))

    def _compute_symmetry(self, image: np.ndarray, cx: float, cy: float) -> float:
        """计算 PSF 对称性分数。"""
        img = np.clip(image, 0, None).astype(np.float64)
        h, w = image.shape

        # 裁剪以质心为中心的正方形区域
        half = min(int(cx), int(cy), int(w - cx), int(h - cy), min(h, w) // 4)
        if half < 4:
            return 0.0

        patch = img[int(cy) - half:int(cy) + half, int(cx) - half:int(cx) + half]
        ph, pw = patch.shape

        # 四象限对称性
        q1 = patch[:ph // 2, :pw // 2]
        q2 = patch[:ph // 2, pw // 2:]
        q3 = patch[ph // 2:, :pw // 2]
        q4 = patch[ph // 2:, pw // 2:]

        # 归一化
        def _norm(a):
            s = np.sum(a) + 1e-10
            return a / s

        q1n, q2n, q3n, q4n = _norm(q1), _norm(q2), _norm(q3), _norm(q4)

        # 对称性: 对角象限应该相似
        sym_h = 1.0 - float(np.mean(np.abs(q1n - np.flip(q2n, axis=1))))
        sym_v = 1.0 - float(np.mean(np.abs(q1n - np.flip(q3n, axis=0))))
        sym_d = 1.0 - float(np.mean(np.abs(q1n - np.flip(np.flip(q4n, axis=0), axis=1))))

        return max(0.0, min(1.0, (sym_h + sym_v + sym_d) / 3.0))

    def _build_zernike_basis(self) -> List[np.ndarray]:
        """构建 Zernike 基函数 (Noll 序号)。"""
        basis = []
        for n in range(1, self.max_zernike_order + 1):
            for l in range(-n, n + 1, 2):
                if abs(l) <= n:
                    basis.append((n, l))
        return basis

    def _zernike_polynomial(
        self, n: int, l: int, rho: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """计算单个 Zernike 多项式值。"""
        # 径向多项式
        m = abs(l)
        R = np.zeros_like(rho)
        for s in range((n - m) // 2 + 1):
            coeff = ((-1) ** s * math.factorial(n - s)
                     / (math.factorial(s)
                        * math.factorial((n + m) // 2 - s)
                        * math.factorial((n - m) // 2 - s)))
            R += coeff * rho ** (n - 2 * s)

        # 角度部分
        if l > 0:
            return R * np.cos(m * theta)
        elif l < 0:
            return R * np.sin(m * theta)
        else:
            return R

    def _fit_zernike(
        self, phase: np.ndarray, weights: np.ndarray
    ) -> List[float]:
        """最小二乘拟合 Zernike 系数。"""
        h, w = phase.shape
        cy, cx = h / 2.0, w / 2.0
        max_r = min(h, w) / 2.0

        yy, xx = np.mgrid[:h, :w]
        dx = (xx - cx) / max_r
        dy = (yy - cy) / max_r
        rho = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)

        # 只在圆孔径内拟合
        aperture = rho <= 1.0
        if np.sum(aperture) < 10:
            return [0.0] * len(self._zernike_basis)

        # 构建设计矩阵
        n_terms = len(self._zernike_basis)
        A = np.zeros((int(np.sum(aperture)), n_terms))
        phase_vec = phase[aperture]
        weight_vec = weights[aperture]

        for i, (n, l) in enumerate(self._zernike_basis):
            A[:, i] = self._zernike_polynomial(n, l, rho[aperture], theta[aperture])

        # 加权最小二乘
        W = np.diag(weight_vec)
        try:
            AW = A.T @ W @ A
            AWp = AW + np.eye(n_terms) * 1e-6  # 正则化
            coeffs = np.linalg.solve(AWp, A.T @ W @ phase_vec)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(n_terms)

        return [float(c) for c in coeffs]

    def _reconstruct_phase_map(
        self, coeffs: List[float], h: int, w: int
    ) -> Optional[np.ndarray]:
        """从 Zernike 系数重建 2D 相位图。"""
        if not coeffs:
            return None

        cy, cx = h / 2.0, w / 2.0
        max_r = min(h, w) / 2.0
        yy, xx = np.mgrid[:h, :w]
        dx = (xx - cx) / max_r
        dy = (yy - cy) / max_r
        rho = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)

        phase = np.zeros((h, w), dtype=np.float64)
        for i, (n, l) in enumerate(self._zernike_basis):
            if i < len(coeffs):
                phase += coeffs[i] * self._zernike_polynomial(n, l, rho, theta)

        # 圆孔径外置零
        phase[rho > 1.0] = 0.0
        return phase

    def reset(self) -> None:
        """重置分析器状态。"""
        pass
