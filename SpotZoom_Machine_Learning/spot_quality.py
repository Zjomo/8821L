"""
光斑质量评估器 (SpotQualityAnalyzer)

基于经典图像处理与光束分析的质量评估算法。

算法原理:
- Laplacian Variance — 清晰度评估 (Pech 2000, 图像模糊度量)
- Contour Circularity — 圆度 (4πA/P², ISO 11146)
- Bilateral Symmetry — 对称性 (互相关度量)
- Gaussian R² Fit — 高斯拟合优度 (最小二乘)
- SNR — 信噪比 (峰值/噪声标准差)

功能:
- 评估光斑的多维质量指标: 圆度、对称性、高斯拟合度、信噪比
- 提供综合质量分数用于对准决策

依赖: numpy, opencv-python (轮廓提取、Laplacian)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class SpotQualityReport:
    """光斑质量评估报告。"""
    overall_score: float  # 综合质量分数 [0, 100]
    sharpness: float  # 清晰度 (Laplacian variance)
    circularity: float  # 圆度 [0, 1]
    symmetry: float  # 对称性 [0, 1]
    gaussian_fit: float  # 高斯拟合度 [0, 1]
    snr: float  # 信噪比
    intensity_uniformity: float  # 强度均匀性 [0, 1]
    radius_px: float  # 等效半径 (像素)
    is_acceptable: bool  # 是否达到可接受质量


class SpotQualityAnalyzer:
    """光斑质量多维度评估器。

    Parameters
    ----------
    min_sharpness : float
        最小可接受清晰度 (Laplacian variance)。
    min_circularity : float
        最小可接受圆度。
    min_snr : float
        最小可接受信噪比。
    weights : dict or None
        各维度权重。默认均等权重。
    """

    def __init__(
        self,
        min_sharpness: float = 20.0,
        min_circularity: float = 0.3,
        min_snr: float = 3.0,
        weights: Optional[dict] = None,
    ):
        self.min_sharpness = float(min_sharpness)
        self.min_circularity = float(min_circularity)
        self.min_snr = float(min_snr)

        self.weights = weights or {
            "sharpness": 0.30,
            "circularity": 0.20,
            "symmetry": 0.15,
            "gaussian_fit": 0.15,
            "snr": 0.10,
            "uniformity": 0.10,
        }

    def analyze(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> SpotQualityReport:
        """对检测到的光斑进行全面质量评估。

        Parameters
        ----------
        frame : np.ndarray
            BGR 图像帧。
        bbox : Tuple[int, int, int, int]
            (x1, y1, x2, y2) 光斑边界框。

        Returns
        -------
        SpotQualityReport
            质量评估报告。
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(int(x2), w)
        y2 = min(int(y2), h)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return SpotQualityReport(
                overall_score=0.0, sharpness=0.0, circularity=0.0,
                symmetry=0.0, gaussian_fit=0.0, snr=0.0,
                intensity_uniformity=0.0, radius_px=0.0, is_acceptable=False,
            )

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # 1. 清晰度 (Laplacian variance)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. 信噪比
        bg_level = np.percentile(gray, 10)
        signal_mask = gray > bg_level
        peak = float(gray.max())
        noise_region = gray[~signal_mask]
        noise_std = float(np.std(noise_region)) if noise_region.size > 0 else 1.0
        snr = peak / max(noise_std, 1e-6)

        # 3. 圆度 (基于轮廓)
        circularity = self._compute_circularity(gray, bg_level)

        # 4. 对称性
        symmetry = self._compute_symmetry(gray)

        # 5. 高斯拟合度
        gaussian_fit = self._compute_gaussian_fit(gray)

        # 6. 强度均匀性
        uniformity = self._compute_uniformity(gray, bg_level)

        # 7. 等效半径
        radius = self._compute_radius(gray, bg_level)

        # 归一化各维度到 [0, 1]
        sharpness_norm = min(sharpness / max(self.min_sharpness * 5, 1.0), 1.0)
        circularity_norm = min(circularity / max(self.min_circularity, 0.01), 1.0)
        snr_norm = min(snr / max(self.min_snr * 5, 1.0), 1.0)

        # 综合分数
        overall = (
            self.weights.get("sharpness", 0.3) * sharpness_norm
            + self.weights.get("circularity", 0.2) * circularity_norm
            + self.weights.get("symmetry", 0.15) * symmetry
            + self.weights.get("gaussian_fit", 0.15) * gaussian_fit
            + self.weights.get("snr", 0.1) * snr_norm
            + self.weights.get("uniformity", 0.1) * uniformity
        )
        overall = min(max(overall * 100, 0.0), 100.0)

        is_acceptable = (
            sharpness >= self.min_sharpness
            and circularity >= self.min_circularity
            and snr >= self.min_snr
        )

        return SpotQualityReport(
            overall_score=overall,
            sharpness=sharpness,
            circularity=circularity,
            symmetry=symmetry,
            gaussian_fit=gaussian_fit,
            snr=snr,
            intensity_uniformity=uniformity,
            radius_px=radius,
            is_acceptable=is_acceptable,
        )

    @staticmethod
    def _compute_circularity(gray: np.ndarray, bg_level: float) -> float:
        """计算光斑圆度 (基于轮廓的面积/周长比)。"""
        binary = (gray > bg_level + 0.2 * (gray.max() - bg_level)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1e-6:
            return 0.0
        # 圆度 = 4π × Area / Perimeter²
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        return float(min(circularity, 1.0))

    @staticmethod
    def _compute_symmetry(gray: np.ndarray) -> float:
        """计算光斑的水平和垂直对称性。"""
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.0

        # 水平对称性
        left = gray[:, :w // 2]
        right = gray[:, w - w // 2:]
        if left.shape != right.shape:
            min_w = min(left.shape[1], right.shape[1])
            left = left[:, :min_w]
            right = right[:, :min_w]
        right_flipped = right[:, ::-1]
        h_sym = 1.0 - float(np.mean(np.abs(left - right_flipped)) / max(gray.max(), 1e-6))

        # 垂直对称性
        top = gray[:h // 2, :]
        bottom = gray[h - h // 2:, :]
        if top.shape != bottom.shape:
            min_h = min(top.shape[0], bottom.shape[0])
            top = top[:min_h, :]
            bottom = bottom[:min_h, :]
        bottom_flipped = bottom[::-1, :]
        v_sym = 1.0 - float(np.mean(np.abs(top - bottom_flipped)) / max(gray.max(), 1e-6))

        return float(min(max((h_sym + v_sym) / 2.0, 0.0), 1.0))

    @staticmethod
    def _compute_gaussian_fit(gray: np.ndarray) -> float:
        """计算光斑与理想高斯分布的拟合度 (R²)。"""
        h, w = gray.shape
        total = gray.sum()
        if total < 1e-6:
            return 0.0

        yy, xx = np.mgrid[:h, :w]
        cx = float(np.sum(xx * gray) / total)
        cy = float(np.sum(yy * gray) / total)
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        sigma2 = float(np.sum(r2 * gray) / total)
        if sigma2 < 1e-6:
            return 0.0

        gaussian = np.exp(-r2 / (2 * sigma2))
        gaussian = gaussian / gaussian.sum() * total

        ss_res = float(np.sum((gray - gaussian) ** 2))
        ss_tot = float(np.sum((gray - gray.mean()) ** 2))
        if ss_tot < 1e-6:
            return 1.0
        r_squared = 1.0 - ss_res / ss_tot
        return float(min(max(r_squared, 0.0), 1.0))

    @staticmethod
    def _compute_uniformity(gray: np.ndarray, bg_level: float) -> float:
        """计算光斑区域强度分布的均匀性。"""
        signal_mask = gray > bg_level
        if not np.any(signal_mask):
            return 0.0
        signal_values = gray[signal_mask]
        if signal_values.size < 2:
            return 1.0
        # 归一化变异系数的逆
        cv_norm = float(np.std(signal_values) / max(np.mean(signal_values), 1e-6))
        uniformity = 1.0 / (1.0 + cv_norm)
        return float(min(max(uniformity, 0.0), 1.0))

    @staticmethod
    def _compute_radius(gray: np.ndarray, bg_level: float) -> float:
        """计算光斑的等效半径。"""
        signal_mask = gray > bg_level
        area = float(np.sum(signal_mask))
        if area < 1:
            return 0.0
        return float(np.sqrt(area / np.pi))
