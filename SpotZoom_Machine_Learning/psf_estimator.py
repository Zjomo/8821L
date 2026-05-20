"""
PSF 估计器 (PSFEstimator)

灵感来源:
- HCIPy (https://github.com/ehpor/hcipy) — PSF 仿真与质量评估
- AOtools (https://github.com/AOtools/aotools) — 自适应光学 PSF 分析工具
- ISO 11146 — 激光光束参数测量标准

算法原理:
- 2D Gaussian Fitting — 二维高斯函数拟合 (Levenberg-Marquardt 风格，
  纯 numpy 实现，无 scipy 依赖)
- Strehl Ratio — Strehl 比估计 (实测峰值 / 衍射极限峰值)
- Encircled Energy — 环围能量 (特定半径内包含的能量比例)
- Second Moment Width — 二阶矩宽度 (FWHM 估计)
- Symmetry Analysis — 对称性分析 (水平/垂直/对角方向)
- Temporal Tracking — 时间序列趋势跟踪

功能:
- 实时估计 PSF 质量指标 (Strehl 比、FWHM、环围能量)
- PSF 对称性/椭圆度分析
- 时间序列 PSF 趋势跟踪与稳定性评估
- 综合质量评分

依赖: numpy, opencv-python (仅用于图像预处理)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("SpotZoom.PSFEstimator")

# 模块默认禁用标志
psf_estimator_enabled: bool = False


@dataclass
class PSFSymmetry:
    """PSF 对称性分析结果。"""
    horizontal_asymmetry: float  # 水平方向不对称度 [0, 1]
    vertical_asymmetry: float  # 垂直方向不对称度 [0, 1]
    diagonal_asymmetry: float  # 对角方向不对称度 [0, 1]
    symmetry_score: float  # 综合对称性评分 [0, 1] (1=完美对称)


@dataclass
class PSFTrend:
    """PSF 时间趋势分析结果。"""
    strehl_trend: float  # Strehl 比变化趋势 (正=改善, 负=恶化)
    width_trend: float  # PSF 宽度变化趋势 (正=展宽, 负=收窄)
    stability_score: float  # 稳定性评分 [0, 1] (1=完全稳定)
    trend_direction: str  # 趋势方向: "improving", "stable", "degrading"


@dataclass
class PSFReport:
    """PSF 估计报告。"""
    strehl_ratio: float  # Strehl 比
    fwhm_x: float  # X 方向 FWHM (像素)
    fwhm_y: float  # Y 方向 FWHM (像素)
    encircled_energy: Dict[float, float]  # 环围能量 {半径: 比例}
    peak_intensity: float  # 峰值强度
    total_energy: float  # 总能量
    symmetry: PSFSymmetry  # 对称性分析
    ellipticity: float  # 椭圆度 (fwhm_y/fwhm_x, 1=圆形)
    psf_width_trend: PSFTrend  # PSF 宽度趋势
    quality_score: float  # 综合质量评分 [0, 100]


class PSFEstimator:
    """PSF (点扩散函数) 估计器。

    实时分析光斑图像的 PSF 质量指标，包括 Strehl 比、FWHM、
    环围能量、对称性和时间趋势。

    Parameters
    ----------
    diffraction_limit_fwhm : float
        衍射极限 FWHM (像素)，用于 Strehl 比计算。
    encircled_radii : List[float] or None
        计算环围能量的半径列表 (像素)。为 None 时使用默认值。
    history_length : int
        历史数据缓冲区长度 (用于趋势分析)。
    min_samples_for_trend : int
        趋势分析所需的最小样本数。
    """

    def __init__(
        self,
        diffraction_limit_fwhm: float = 3.0,
        encircled_radii: Optional[List[float]] = None,
        history_length: int = 50,
        min_samples_for_trend: int = 10,
    ):
        self.diffraction_limit_fwhm = float(diffraction_limit_fwhm)
        self.encircled_radii = encircled_radii or [1.0, 2.0, 3.0, 5.0, 10.0]
        self.history_length = int(history_length)
        self.min_samples_for_trend = int(min_samples_for_trend)

        # 历史数据缓冲区
        self._strehl_history: Deque[float] = deque(maxlen=self.history_length)
        self._fwhm_x_history: Deque[float] = deque(maxlen=self.history_length)
        self._fwhm_y_history: Deque[float] = deque(maxlen=self.history_length)
        self._peak_history: Deque[float] = deque(maxlen=self.history_length)

        LOGGER.info(
            "PSFEstimator: 初始化完成 (diff_limit=%.2f px, history=%d)",
            self.diffraction_limit_fwhm, self.history_length,
        )

    def estimate(
        self,
        image: np.ndarray,
        region_of_interest: Optional[Tuple[int, int, int, int]] = None,
    ) -> PSFReport:
        """主估计入口: 分析图像的 PSF 质量。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (灰度或 BGR)。
        region_of_interest : Tuple[int, int, int, int] or None
            (x1, y1, x2, y2) 感兴趣区域。为 None 时使用整幅图像。

        Returns
        -------
        PSFReport
            PSF 估计报告。
        """
        try:
            # 预处理
            gray = self._preprocess(image, region_of_interest)
            if gray is None or gray.size < 9:
                return self._empty_report()

            # 计算质心
            total = gray.sum()
            if total < 1e-12:
                return self._empty_report()

            yy, xx = np.mgrid[:gray.shape[0], :gray.shape[1]]
            cx = float(np.sum(xx * gray) / total)
            cy = float(np.sum(yy * gray) / total)
            center = (cx, cy)

            # Strehl 比
            strehl = self.compute_strehl_ratio(gray)

            # FWHM
            fwhm_x, fwhm_y = self.compute_psf_width(gray, center)

            # 环围能量
            encircled = self.compute_encircled_energy(gray, center, self.encircled_radii)

            # 峰值和总能量
            peak_intensity = float(gray.max())
            total_energy = float(total)

            # 对称性
            symmetry = self.analyze_symmetry(gray, center)

            # 椭圆度
            if fwhm_x > 1e-6:
                ellipticity = fwhm_y / fwhm_x
            else:
                ellipticity = 1.0

            # 更新历史
            self._strehl_history.append(strehl)
            self._fwhm_x_history.append(fwhm_x)
            self._fwhm_y_history.append(fwhm_y)
            self._peak_history.append(peak_intensity)

            # 趋势分析
            trend = self.get_trend()

            # 综合质量评分
            quality = self._compute_quality_score(strehl, symmetry, ellipticity, trend)

            LOGGER.debug(
                "PSFEstimator: Strehl=%.4f, FWHM=(%.2f, %.2f), 质量=%.1f",
                strehl, fwhm_x, fwhm_y, quality,
            )

            return PSFReport(
                strehl_ratio=round(strehl, 6),
                fwhm_x=round(fwhm_x, 4),
                fwhm_y=round(fwhm_y, 4),
                encircled_energy=encircled,
                peak_intensity=round(peak_intensity, 4),
                total_energy=round(total_energy, 4),
                symmetry=symmetry,
                ellipticity=round(ellipticity, 4),
                psf_width_trend=trend,
                quality_score=round(quality, 2),
            )

        except Exception as e:
            LOGGER.error("PSFEstimator: 估计失败: %s", e)
            return self._empty_report()

    def compute_strehl_ratio(
        self,
        image: np.ndarray,
        wavelength_px: float = 1.0,
    ) -> float:
        """计算 Strehl 比。

        Strehl 比 = 实测 PSF 峰值 / 衍射极限 PSF 峰值。

        衍射极限峰值通过 Airy 函数峰值估计:
            I_airy_peak = 1.0 (归一化)
            实测峰值 / 理论峰值 = Strehl 比

        使用简化方法: 基于实测 PSF 的归一化峰值与理想高斯 PSF 峰值比较。

        Parameters
        ----------
        image : np.ndarray
            光斑图像 (2D float64)。
        wavelength_px : float
            波长 (像素单位)，用于衍射极限估计。

        Returns
        -------
        float
            Strehl 比 [0, 1]。
        """
        image = image.astype(np.float64)
        total = image.sum()
        if total < 1e-12:
            return 0.0

        # 实测峰值 (归一化)
        measured_peak = float(image.max()) / total

        # 理想衍射极限 PSF 峰值 (Airy 函数中心)
        # 对于圆形孔径: I(0) = (pi * D^2 / (4 * lambda * f))^2
        # 简化: 使用衍射极限 FWHM 对应的高斯峰值
        sigma_diff = self.diffraction_limit_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        h, w = image.shape
        yy, xx = np.mgrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ideal_psf = np.exp(-r2 / (2.0 * sigma_diff ** 2))
        ideal_peak = float(ideal_psf.max()) / ideal_psf.sum()

        # Strehl 比
        if ideal_peak > 1e-12:
            strehl = min(measured_peak / ideal_peak, 1.0)
        else:
            strehl = 0.0

        return max(strehl, 0.0)

    def compute_encircled_energy(
        self,
        image: np.ndarray,
        center: Tuple[float, float],
        radii: List[float],
    ) -> Dict[float, float]:
        """计算环围能量。

        在指定半径的圆内，计算包含的总能量占总能量的比例。

        Parameters
        ----------
        image : np.ndarray
            光斑图像 (2D float64)。
        center : Tuple[float, float]
            光斑中心 (cx, cy)。
        radii : List[float]
            半径列表 (像素)。

        Returns
        -------
        Dict[float, float]
            半径到环围能量比例的映射。
        """
        image = image.astype(np.float64)
        total = image.sum()
        if total < 1e-12:
            return {r: 0.0 for r in radii}

        h, w = image.shape
        yy, xx = np.mgrid[:h, :w]
        r_map = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)

        result: Dict[float, float] = {}
        for radius in radii:
            mask = r_map <= radius
            energy = float(image[mask].sum()) / total
            result[round(float(radius), 4)] = round(min(energy, 1.0), 6)

        return result

    def compute_psf_width(
        self,
        image: np.ndarray,
        center: Tuple[float, float],
    ) -> Tuple[float, float]:
        """计算 PSF 的 FWHM (半高全宽)。

        使用二阶矩方法和 1D 高斯拟合结合估计 FWHM。

        Parameters
        ----------
        image : np.ndarray
            光斑图像 (2D float64)。
        center : Tuple[float, float]
            光斑中心 (cx, cy)。

        Returns
        -------
        Tuple[float, float]
            (fwhm_x, fwhm_y) 像素。
        """
        image = image.astype(np.float64)
        h, w = image.shape
        total = image.sum()
        if total < 1e-12:
            return (0.0, 0.0)

        yy, xx = np.mgrid[:h, :w]
        cx, cy = center

        # 二阶矩方法
        dx = xx - cx
        dy = yy - cy
        sigma_x2 = float(np.sum(dx ** 2 * image) / total)
        sigma_y2 = float(np.sum(dy ** 2 * image) / total)

        sigma_x = np.sqrt(max(sigma_x2, 1e-6))
        sigma_y = np.sqrt(max(sigma_y2, 1e-6))

        # FWHM = 2 * sqrt(2 * ln(2)) * sigma
        fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
        fwhm_x = sigma_x * fwhm_factor
        fwhm_y = sigma_y * fwhm_factor

        # 使用 1D 剖面拟合修正
        fwhm_x = self._refine_fwhm_1d(image, cx, cy, axis='x', initial=fwhm_x)
        fwhm_y = self._refine_fwhm_1d(image, cx, cy, axis='y', initial=fwhm_y)

        return (max(fwhm_x, 0.1), max(fwhm_y, 0.1))

    def analyze_symmetry(
        self,
        image: np.ndarray,
        center: Tuple[float, float],
    ) -> PSFSymmetry:
        """分析 PSF 的对称性。

        通过比较不同方向的半剖面来评估对称性。

        Parameters
        ----------
        image : np.ndarray
            光斑图像 (2D float64)。
        center : Tuple[float, float]
            光斑中心 (cx, cy)。

        Returns
        -------
        PSFSymmetry
            对称性分析结果。
        """
        image = image.astype(np.float64)
        h, w = image.shape
        cx, cy = center

        # 确保中心在图像范围内
        cx = max(0.0, min(float(cx), w - 1.0))
        cy = max(0.0, min(float(cy), h - 1.0))
        icx, icy = int(round(cx)), int(round(cy))

        # 水平对称性: 比较左右半剖面
        left_profile = image[icy, :icx][::-1] if icx > 0 else np.array([])
        right_profile = image[icy, icx:] if icx < w else np.array([])
        h_asym = self._compare_profiles(left_profile, right_profile)

        # 垂直对称性: 比较上下半剖面
        top_profile = image[:icy, icx][::-1] if icy > 0 else np.array([])
        bottom_profile = image[icy:, icx] if icy < h else np.array([])
        v_asym = self._compare_profiles(top_profile, bottom_profile)

        # 对角对称性
        diag_len = min(icx, icy, h - icy - 1, w - icx - 1)
        if diag_len > 2:
            diag1 = np.array([image[icy - i, icx - i] for i in range(1, diag_len + 1)])
            diag2 = np.array([image[icy + i, icx + i] for i in range(1, diag_len + 1)])
            d_asym = self._compare_profiles(diag1, diag2)
        else:
            d_asym = 0.0

        # 综合对称性评分
        symmetry_score = float(np.clip(
            1.0 - (h_asym + v_asym + d_asym) / 3.0, 0.0, 1.0
        ))

        return PSFSymmetry(
            horizontal_asymmetry=round(h_asym, 6),
            vertical_asymmetry=round(v_asym, 6),
            diagonal_asymmetry=round(d_asym, 6),
            symmetry_score=round(symmetry_score, 6),
        )

    def get_trend(self) -> PSFTrend:
        """获取 PSF 时间趋势分析。

        Returns
        -------
        PSFTrend
            趋势分析结果。
        """
        if len(self._strehl_history) < self.min_samples_for_trend:
            return PSFTrend(
                strehl_trend=0.0,
                width_trend=0.0,
                stability_score=1.0,
                trend_direction="stable",
            )

        # Strehl 趋势 (线性回归斜率)
        strehl_list = list(self._strehl_history)
        strehl_trend = self._linear_slope(strehl_list)

        # 宽度趋势 (平均 FWHM)
        fwhm_list = [
            (fx + fy) / 2.0
            for fx, fy in zip(self._fwhm_x_history, self._fwhm_y_history)
        ]
        width_trend = self._linear_slope(fwhm_list)

        # 稳定性评分 (基于 Strehl 和 FWHM 的变异系数)
        strehl_arr = np.array(strehl_list)
        fwhm_arr = np.array(fwhm_list)

        strehl_cv = float(np.std(strehl_arr) / max(np.mean(strehl_arr), 1e-6))
        fwhm_cv = float(np.std(fwhm_arr) / max(np.mean(fwhm_arr), 1e-6))

        stability = float(np.clip(
            1.0 - 0.5 * (strehl_cv + fwhm_cv), 0.0, 1.0
        ))

        # 趋势方向判定
        if abs(strehl_trend) < 0.001 and abs(width_trend) < 0.001:
            direction = "stable"
        elif strehl_trend > 0 and width_trend < 0:
            direction = "improving"
        elif strehl_trend < 0 and width_trend > 0:
            direction = "degrading"
        else:
            direction = "stable"

        return PSFTrend(
            strehl_trend=round(strehl_trend, 6),
            width_trend=round(width_trend, 6),
            stability_score=round(stability, 6),
            trend_direction=direction,
        )

    def reset(self) -> None:
        """重置估计器，清除所有历史数据。"""
        self._strehl_history.clear()
        self._fwhm_x_history.clear()
        self._fwhm_y_history.clear()
        self._peak_history.clear()

        LOGGER.info("PSFEstimator: 估计器已重置")

    # ======================== 内部方法 ========================

    def _preprocess(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        """预处理输入图像。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        roi : Tuple[int, int, int, int] or None
            感兴趣区域。

        Returns
        -------
        np.ndarray or None
            预处理后的灰度图像。
        """
        if image is None or image.size == 0:
            return None

        # 转灰度
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = image.astype(np.float64)

        # ROI 裁剪
        if roi is not None:
            x1, y1, x2, y2 = roi
            h, w = gray.shape
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(int(x2), w)
            y2 = min(int(y2), h)
            if x2 - x1 < 3 or y2 - y1 < 3:
                return None
            gray = gray[y1:y2, x1:x2]

        # 背景扣除
        bg = np.percentile(gray, 10)
        gray = np.maximum(gray - bg, 0.0)

        return gray

    def _refine_fwhm_1d(
        self,
        image: np.ndarray,
        cx: float,
        cy: float,
        axis: str,
        initial: float,
    ) -> float:
        """使用 1D 高斯拟合精化 FWHM 估计。

        Parameters
        ----------
        image : np.ndarray
            光斑图像。
        cx, cy : float
            中心坐标。
        axis : str
            拟合轴 ('x' 或 'y')。
        initial : float
            初始 FWHM 估计值。

        Returns
        -------
        float
            精化后的 FWHM。
        """
        h, w = image.shape

        if axis == 'x':
            icy = int(round(cy))
            if icy < 0 or icy >= h:
                return initial
            profile = image[icy, :].astype(np.float64)
            coords = np.arange(w, dtype=np.float64)
        else:
            icx = int(round(cx))
            if icx < 0 or icx >= w:
                return initial
            profile = image[:, icx].astype(np.float64)
            coords = np.arange(h, dtype=np.float64)

        # 归一化
        peak = profile.max()
        if peak < 1e-12:
            return initial

        profile_norm = profile / peak

        # 初始参数: [amplitude, center, sigma]
        center_est = float(cx) if axis == 'x' else float(cy)
        sigma_init = initial / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # 简单高斯拟合 (迭代最小二乘, 类 LM)
        try:
            params = self._gaussian_fit_1d(coords, profile_norm, center_est, sigma_init)
            if params is not None:
                sigma_fit = abs(params[2])
                fwhm_fit = sigma_fit * 2.0 * np.sqrt(2.0 * np.log(2.0))
                # 仅在拟合结果合理时使用
                if 0.5 < fwhm_fit < max(w, h) * 0.5:
                    return fwhm_fit
        except Exception:
            pass

        return initial

    def _gaussian_fit_1d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        center_init: float,
        sigma_init: float,
        max_iter: int = 20,
    ) -> Optional[np.ndarray]:
        """1D 高斯拟合 (Gauss-Newton 迭代法)。

        模型: y = A * exp(-(x - mu)^2 / (2 * sigma^2))

        Parameters
        ----------
        x : np.ndarray
            坐标数组。
        y : np.ndarray
            数据数组。
        center_init : float
            中心初始值。
        sigma_init : float
            sigma 初始值。
        max_iter : int
            最大迭代次数。

        Returns
        -------
        np.ndarray or None
            拟合参数 [amplitude, center, sigma]。
        """
        n = len(x)
        if n < 5:
            return None

        # 初始参数
        A = float(y.max())
        mu = float(center_init)
        sigma = float(sigma_init)

        if sigma < 0.1:
            sigma = 1.0

        for _ in range(max_iter):
            # 模型预测
            model = A * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))

            # 残差
            residuals = y - model

            # Jacobian
            exp_term = np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))
            dA = exp_term
            dmu = A * exp_term * (x - mu) / (sigma ** 2)
            dsigma = A * exp_term * (x - mu) ** 2 / (sigma ** 3)

            J = np.column_stack([dA, dmu, dsigma])

            # Gauss-Newton 更新
            try:
                JtJ = J.T @ J
                Jtr = J.T @ residuals
                # 正则化
                JtJ += 1e-6 * np.eye(3)
                delta = np.linalg.solve(JtJ, Jtr)
            except np.linalg.LinAlgError:
                break

            # 阻尼更新
            A += delta[0]
            mu += delta[1]
            sigma += delta[2]

            # 约束
            A = max(A, 0.0)
            sigma = max(sigma, 0.1)

            # 收敛检查
            if np.sum(delta ** 2) < 1e-10:
                break

        return np.array([A, mu, sigma])

    def _compare_profiles(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """比较两个剖面曲线的不对称度。

        Parameters
        ----------
        p1, p2 : np.ndarray
            两个剖面数据。

        Returns
        -------
        float
            不对称度 [0, 1] (0=完全对称, 1=完全不对称)。
        """
        if len(p1) < 2 or len(p2) < 2:
            return 0.0

        # 对齐长度
        min_len = min(len(p1), len(p2))
        p1 = p1[:min_len].astype(np.float64)
        p2 = p2[:min_len].astype(np.float64)

        # 归一化
        max_val = max(p1.max(), p2.max(), 1e-12)
        p1 = p1 / max_val
        p2 = p2 / max_val

        # 不对称度 = 归一化平均绝对差
        asymmetry = float(np.mean(np.abs(p1 - p2)))
        return min(asymmetry, 1.0)

    def _linear_slope(self, data: List[float]) -> float:
        """计算数据的线性回归斜率。

        Parameters
        ----------
        data : List[float]
            时间序列数据。

        Returns
        -------
        float
            斜率 (每步变化量)。
        """
        n = len(data)
        if n < 2:
            return 0.0

        x = np.arange(n, dtype=np.float64)
        y = np.array(data, dtype=np.float64)

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = float(np.sum((x - x_mean) * (y - y_mean)))
        denominator = float(np.sum((x - x_mean) ** 2))

        if denominator < 1e-12:
            return 0.0

        return numerator / denominator

    def _compute_quality_score(
        self,
        strehl: float,
        symmetry: PSFSymmetry,
        ellipticity: float,
        trend: PSFTrend,
    ) -> float:
        """计算综合质量评分。

        Parameters
        ----------
        strehl : float
            Strehl 比。
        symmetry : PSFSymmetry
            对称性。
        ellipticity : float
            椭圆度。
        trend : PSFTrend
            趋势。

        Returns
        -------
        float
            质量评分 [0, 100]。
        """
        # Strehl 比评分 (0~40 分)
        strehl_score = min(strehl, 1.0) * 40.0

        # 对称性评分 (0~25 分)
        sym_score = symmetry.symmetry_score * 25.0

        # 椭圆度评分 (0~15 分, 1.0=满分)
        ellipticity_deviation = abs(ellipticity - 1.0)
        ell_score = max(0.0, 1.0 - ellipticity_deviation * 2.0) * 15.0

        # 稳定性评分 (0~20 分)
        stab_score = trend.stability_score * 20.0

        total = strehl_score + sym_score + ell_score + stab_score
        return min(max(total, 0.0), 100.0)

    def _empty_report(self) -> PSFReport:
        """返回空报告。"""
        return PSFReport(
            strehl_ratio=0.0,
            fwhm_x=0.0,
            fwhm_y=0.0,
            encircled_energy={},
            peak_intensity=0.0,
            total_energy=0.0,
            symmetry=PSFSymmetry(0.0, 0.0, 0.0, 0.0),
            ellipticity=1.0,
            psf_width_trend=PSFTrend(0.0, 0.0, 1.0, "stable"),
            quality_score=0.0,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    estimator = PSFEstimator(
        diffraction_limit_fwhm=3.0,
        encircled_radii=[1.0, 2.0, 3.0, 5.0, 10.0],
        history_length=50,
    )

    print("=== PSF 估计器测试 ===\n")

    # 生成理想高斯 PSF
    size = 64
    yy, xx = np.mgrid[:size, :size]
    cx, cy = size / 2.0, size / 2.0
    sigma = 2.0
    psf_ideal = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    psf_ideal = (psf_ideal / psf_ideal.max() * 255.0).astype(np.uint8)

    print("--- 理想高斯 PSF ---")
    report = estimator.estimate(psf_ideal)
    print(f"Strehl 比: {report.strehl_ratio:.4f}")
    print(f"FWHM: ({report.fwhm_x:.2f}, {report.fwhm_y:.2f}) px")
    print(f"椭圆度: {report.ellipticity:.4f}")
    print(f"对称性: {report.symmetry.symmetry_score:.4f}")
    print(f"质量评分: {report.quality_score:.1f}")
    print(f"环围能量: {report.encircled_energy}")

    # 生成含像散的 PSF (椭圆)
    sigma_x, sigma_y = 1.5, 3.5
    psf_astig = np.exp(-((xx - cx) ** 2 / (2.0 * sigma_x ** 2) +
                         (yy - cy) ** 2 / (2.0 * sigma_y ** 2)))
    psf_astig = (psf_astig / psf_astig.max() * 255.0).astype(np.uint8)

    print(f"\n--- 含像散的 PSF (椭圆) ---")
    report2 = estimator.estimate(psf_astig)
    print(f"Strehl 比: {report2.strehl_ratio:.4f}")
    print(f"FWHM: ({report2.fwhm_x:.2f}, {report2.fwhm_y:.2f}) px")
    print(f"椭圆度: {report2.ellipticity:.4f}")
    print(f"对称性: {report2.symmetry.symmetry_score:.4f}")
    print(f"质量评分: {report2.quality_score:.1f}")

    # 时间序列趋势测试
    print(f"\n--- 时间趋势测试 ---")
    for i in range(20):
        # 模拟逐渐改善的 PSF
        sigma_t = 3.0 - i * 0.08
        noise = np.random.normal(0, 2.0, (size, size))
        psf_t = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_t ** 2))
        psf_t = np.clip(psf_t + noise, 0, None)
        psf_t = (psf_t / psf_t.max() * 255.0).astype(np.uint8)
        estimator.estimate(psf_t)

    trend = estimator.get_trend()
    print(f"Strehl 趋势: {trend.strehl_trend:.6f}")
    print(f"宽度趋势: {trend.width_trend:.6f}")
    print(f"稳定性: {trend.stability_score:.4f}")
    print(f"趋势方向: {trend.trend_direction}")

    print("\n测试完成")
