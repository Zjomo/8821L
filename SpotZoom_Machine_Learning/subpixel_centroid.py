"""
亚像素质心定位器 (SubPixelCentroid)

基于经典光束分析 / 自适应光学质心波前传感器的定位算法。

算法原理:
- Weighted Centroid — 灰度加权质心 (ISO 11146 光束宽度标准)
- Threshold Centroid — 阈值加权质心 (抑制背景噪声)
- Gaussian Fit — 二维高斯迭代拟合 (Newton 法)
- Second Moment Radius — 二阶矩等效半径估计

功能:
- 在检测区域内进行加权质心计算，实现亚像素级定位
- 支持多种质心算法: 灰度加权、阈值加权、高斯拟合
- 提供光斑半径估计与信噪比计算

依赖: numpy, opencv-python (仅用于色彩空间转换)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CentroidResult:
    """亚像素质心计算结果。"""
    cx: float  # 亚像素 x 坐标
    cy: float  # 亚像素 y 坐标
    radius: float  # 估计的光斑半径 (像素)
    peak_intensity: float  # 峰值灰度
    total_intensity: float  # 总灰度
    snr: float  # 信噪比


class SubPixelCentroid:
    """亚像素级光斑质心定位器。

    在检测框裁剪的区域内，使用灰度加权质心算法
    实现亚像素级精度的光斑中心定位。

    Parameters
    ----------
    method : str
        质心算法: 'weighted' (灰度加权), 'threshold' (阈值加权), 'gaussian' (高斯拟合)
    bg_percentile : float
        背景估计百分位数 (0-100)。用于估计背景灰度并扣除。
    min_snr : float
        最小信噪比阈值。低于此值的结果标记为不可靠。
    """

    def __init__(
        self,
        method: str = "weighted",
        bg_percentile: float = 10.0,
        min_snr: float = 3.0,
    ):
        if method not in ("weighted", "threshold", "gaussian"):
            raise ValueError(f"Unknown method: {method}. Use 'weighted', 'threshold', or 'gaussian'.")
        self.method = method
        self.bg_percentile = float(bg_percentile)
        self.min_snr = float(min_snr)

    def compute(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[CentroidResult]:
        """在给定边界框内计算亚像素质心。

        Parameters
        ----------
        frame : np.ndarray
            BGR 格式的图像帧。
        bbox : Tuple[int, int, int, int]
            (x1, y1, x2, y2) 边界框。

        Returns
        -------
        Optional[CentroidResult]
            质心结果。如果计算失败或 SNR 过低，返回 None。
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(int(x2), w)
        y2 = min(int(y2), h)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # 背景估计与扣除
        bg_level = np.percentile(gray, self.bg_percentile)
        signal = gray - bg_level
        signal = np.maximum(signal, 0.0)

        total_intensity = float(signal.sum())
        if total_intensity < 1e-6:
            return None

        peak_intensity = float(signal.max())
        noise_estimate = float(np.std(gray[gray <= bg_level])) if np.any(gray <= bg_level) else 1.0
        snr = peak_intensity / max(noise_estimate, 1e-6)

        if self.method == "weighted":
            cx_local, cy_local = self._weighted_centroid(signal)
        elif self.method == "threshold":
            cx_local, cy_local = self._threshold_centroid(signal, bg_level + 0.3 * peak_intensity)
        elif self.method == "gaussian":
            result = self._gaussian_fit(signal)
            if result is None:
                cx_local, cy_local = self._weighted_centroid(signal)
            else:
                cx_local, cy_local = result

        # 转换为全局坐标
        cx_global = cx_local + x1
        cy_global = cy_local + y1

        # 估计光斑半径 (二阶矩)
        radius = self._estimate_radius(signal, cx_local, cy_local)

        if snr < self.min_snr:
            return None

        return CentroidResult(
            cx=cx_global,
            cy=cy_global,
            radius=radius,
            peak_intensity=peak_intensity,
            total_intensity=total_intensity,
            snr=snr,
        )

    @staticmethod
    def _weighted_centroid(signal: np.ndarray) -> Tuple[float, float]:
        """灰度加权质心。"""
        total = signal.sum()
        if total < 1e-6:
            h, w = signal.shape
            return (w / 2.0, h / 2.0)
        yy, xx = np.mgrid[:signal.shape[0], :signal.shape[1]]
        cx = float(np.sum(xx * signal) / total)
        cy = float(np.sum(yy * signal) / total)
        return (cx, cy)

    @staticmethod
    def _threshold_centroid(signal: np.ndarray, threshold: float) -> Tuple[float, float]:
        """阈值加权质心 — 仅使用高于阈值的像素。"""
        mask = signal > threshold
        if not np.any(mask):
            return SubPixelCentroid._weighted_centroid(signal)
        masked = signal * mask
        total = masked.sum()
        if total < 1e-6:
            h, w = signal.shape
            return (w / 2.0, h / 2.0)
        yy, xx = np.mgrid[:signal.shape[0], :signal.shape[1]]
        cx = float(np.sum(xx * masked) / total)
        cy = float(np.sum(yy * masked) / total)
        return (cx, cy)

    @staticmethod
    def _gaussian_fit(signal: np.ndarray) -> Optional[Tuple[float, float]]:
        """二维高斯拟合求质心。"""
        h, w = signal.shape
        total = signal.sum()
        if total < 1e-6:
            return None

        yy, xx = np.mgrid[:h, :w]
        cx_init = float(np.sum(xx * signal) / total)
        cy_init = float(np.sum(yy * signal) / total)

        # 简单的迭代高斯拟合 (3 次 Newton 迭代)
        cx, cy = cx_init, cy_init
        for _ in range(3):
            dx = xx - cx
            dy = yy - cy
            r2 = dx * dx + dy * dy
            sigma = max(np.sqrt(np.sum(r2 * signal) / total), 1.0)
            gauss = np.exp(-r2 / (2 * sigma * sigma))
            weighted = signal * gauss
            wt = weighted.sum()
            if wt < 1e-6:
                break
            cx = float(np.sum(xx * weighted) / wt)
            cy = float(np.sum(yy * weighted) / wt)

        return (cx, cy)

    @staticmethod
    def _estimate_radius(signal: np.ndarray, cx: float, cy: float) -> float:
        """基于二阶矩估计光斑半径。"""
        total = signal.sum()
        if total < 1e-6:
            return 0.0
        h, w = signal.shape
        yy, xx = np.mgrid[:h, :w]
        dx = xx - cx
        dy = yy - cy
        r2 = dx * dx + dy * dy
        variance = float(np.sum(r2 * signal) / total)
        return float(np.sqrt(max(variance, 0.0)))
