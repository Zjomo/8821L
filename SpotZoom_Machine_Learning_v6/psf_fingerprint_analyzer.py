"""
PSF Fingerprint Analyzer - PSF 指纹分析器

Inspired by:
- prysm (brandondube): Optical diffraction propagation and PSF analysis
- HCIPy (ehpor): High-contrast imaging PSF simulation
- Picasso (jungmannlab): GPU-accelerated PSF fitting for localization

Core Innovation:
- PSF 指纹提取: 从光斑图像中提取独特 PSF 特征
- PSF 异常检测: 识别光学系统状态变化
- PSF 质量评分: 综合评估光斑聚焦质量
- PSF 变化追踪: 监测 PSF 随时间的漂移
- 纯 numpy+cv2 实现，零外部依赖
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class PSFFingerprintConfig:
    """PSF 指纹分析器配置"""
    # 分析区域大小 (像素)
    analysis_size: int = 64
    # Zernike 多项式最大阶数
    zernike_max_order: int = 4
    # PSF 特征维度
    feature_dim: int = 32
    # 异常检测历史窗口
    anomaly_window: int = 50
    # 异常检测阈值 (标准差倍数)
    anomaly_threshold: float = 3.0
    # PSF 漂移检测阈值 (像素)
    drift_threshold_px: float = 0.5
    # 是否启用 Zernike 分解
    zernike_enabled: bool = True
    # 是否启用对称性分析
    symmetry_enabled: bool = True


@dataclass
class PSFFingerprintReport:
    """PSF 指纹分析报告"""
    # PSF 特征向量
    fingerprint: np.ndarray
    # 聚焦质量评分 (0-1)
    focus_score: float
    # 对称性评分 (0-1)
    symmetry_score: float
    # Strehl 比估计
    strehl_estimate: float
    # PSF 宽度 (像素)
    psf_width_x: float
    psf_width_y: float
    # 椭圆度
    ellipticity: float
    # 是否异常
    is_anomaly: bool
    # 异常分数
    anomaly_score: float
    # PSF 漂移量 (像素)
    drift_from_reference: float
    # Zernike 系数 (如果启用)
    zernike_coeffs: Optional[List[float]] = None


class PSFFingerprintAnalyzer:
    """PSF 指纹分析器

    从光斑图像中提取 PSF 特征指纹，用于光学系统
    状态监测、异常检测和质量评估。

    Inspired by prysm's PSF analysis and Picasso's PSF fitting.
    """

    def __init__(self, config: Optional[PSFFingerprintConfig] = None):
        self.config = config or PSFFingerprintConfig()
        self._reference_fingerprint: Optional[np.ndarray] = None
        self._fingerprint_history: Deque[np.ndarray] = deque(maxlen=self.config.anomaly_window)
        self._focus_history: Deque[float] = deque(maxlen=self.config.anomaly_window)

    def reset(self) -> None:
        """重置分析器状态"""
        self._reference_fingerprint = None
        self._fingerprint_history.clear()
        self._focus_history.clear()

    @staticmethod
    def _skew(data: np.ndarray) -> float:
        """计算偏度 (Fisher's skewness)"""
        n = data.size
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def set_reference(self, image: np.ndarray) -> None:
        """设置参考 PSF 指纹"""
        report = self.analyze(image)
        self._reference_fingerprint = report.fingerprint.copy()

    def _extract_psf_region(
        self, image: np.ndarray, center: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """提取 PSF 分析区域"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        size = self.config.analysis_size

        if center is None:
            # 自动寻找最亮区域
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
            cx, cy = max_loc
        else:
            cx, cy = int(center[0]), int(center[1])

        # 提取区域
        half = size // 2
        y_min = max(0, cy - half)
        y_max = min(h, cy + half)
        x_min = max(0, cx - half)
        x_max = min(w, cx + half)

        region = gray[y_min:y_max, x_min:x_max].copy()

        # 如果区域不够大，填充
        if region.shape[0] < size or region.shape[1] < size:
            padded = np.zeros((size, size), dtype=gray.dtype)
            padded[:region.shape[0], :region.shape[1]] = region
            region = padded

        return region

    def _compute_focus_score(self, region: np.ndarray) -> float:
        """计算聚焦评分 (Laplacian 方差法)"""
        laplacian = cv2.Laplacian(region.astype(np.float64), cv2.CV_64F)
        variance = float(np.var(laplacian))

        # 归一化到 [0, 1]
        # 使用经验阈值: 方差 > 500 认为聚焦良好
        score = min(1.0, variance / 500.0)
        return score

    def _compute_symmetry(self, region: np.ndarray) -> float:
        """计算 PSF 对称性评分"""
        h, w = region.shape
        cy, cx = h // 2, w // 2

        # 水平对称性
        left = region[:, :cx]
        right = np.flip(region[:, cx:], axis=1)
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]
        h_sym = 1.0 - float(np.mean(np.abs(left.astype(float) - right.astype(float))) / 256.0)

        # 垂直对称性
        top = region[:cy, :]
        bottom = np.flip(region[cy:, :], axis=0)
        min_h = min(top.shape[0], bottom.shape[0])
        top = top[:min_h, :]
        bottom = bottom[:min_h, :]
        v_sym = 1.0 - float(np.mean(np.abs(top.astype(float) - bottom.astype(float))) / 256.0)

        return float((h_sym + v_sym) / 2.0)

    def _compute_psf_width(self, region: np.ndarray) -> Tuple[float, float]:
        """计算 PSF 宽度 (二阶矩)"""
        h, w = region.shape
        region_float = region.astype(np.float64)
        total = np.sum(region_float)
        if total < 1e-6:
            return float(w / 4), float(h / 4)

        yy, xx = np.mgrid[:h, :w]
        cx = float(np.sum(xx * region_float) / total)
        cy = float(np.sum(yy * region_float) / total)

        var_x = float(np.sum((xx - cx) ** 2 * region_float) / total)
        var_y = float(np.sum((yy - cy) ** 2 * region_float) / total)

        sigma_x = float(np.sqrt(max(0, var_x)))
        sigma_y = float(np.sqrt(max(0, var_y)))

        # FWHM = 2.355 * sigma
        return sigma_x * 2.355, sigma_y * 2.355

    def _estimate_strehl(self, region: np.ndarray) -> float:
        """估计 Strehl 比"""
        region_float = region.astype(np.float64)
        peak = float(np.max(region_float))
        mean = float(np.mean(region_float))

        if mean < 1e-6:
            return 0.0

        # Strehl ≈ peak / (total_energy * ideal_peak_density)
        total = float(np.sum(region_float))
        h, w = region.shape
        area = h * w

        # 理想 PSF 的峰值密度
        ideal_peak_density = total / area * 4.0  # 经验系数

        if ideal_peak_density < 1e-6:
            return 0.0

        strehl = min(1.0, peak / ideal_peak_density)
        return float(strehl)

    def _extract_fingerprint(self, region: np.ndarray) -> np.ndarray:
        """提取 PSF 特征指纹

        使用多尺度特征构建 PSF 指纹向量。
        """
        region_float = region.astype(np.float64)
        h, w = region_float.shape

        # 归一化
        max_val = np.max(region_float)
        if max_val > 0:
            normalized = region_float / max_val
        else:
            normalized = region_float

        features = []

        # 1. 径向强度分布 (8 个环)
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)
        n_rings = 8
        for i in range(n_rings):
            r_inner = i * max_radius / n_rings
            r_outer = (i + 1) * max_radius / n_rings
            yy, xx = np.mgrid[:h, :w]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            mask = (dist >= r_inner) & (dist < r_outer)
            if np.any(mask):
                features.append(float(np.mean(normalized[mask])))
            else:
                features.append(0.0)

        # 2. 角度强度分布 (8 个扇区)
        n_sectors = 8
        for i in range(n_sectors):
            angle_min = i * 2 * np.pi / n_sectors
            angle_max = (i + 1) * 2 * np.pi / n_sectors
            yy, xx = np.mgrid[:h, :w]
            angles = np.arctan2(yy - cy, xx - cx)
            mask = (angles >= angle_min) & (angles < angle_max)
            if np.any(mask):
                features.append(float(np.mean(normalized[mask])))
            else:
                features.append(0.0)

        # 3. 统计特征 (8 个)
        features.extend([
            float(np.mean(normalized)),
            float(np.std(normalized)),
            float(np.max(normalized)),
            float(np.min(normalized)),
            float(np.median(normalized)),
            float(np.percentile(normalized, 90)),
            float(np.percentile(normalized, 10)),
            float(self._skew(normalized.flatten())) if normalized.size > 2 else 0.0,
        ])

        # 4. 频域特征 (8 个)
        fft = np.fft.fft2(normalized)
        fft_shift = np.fft.fftshift(np.abs(fft))
        fft_norm = fft_shift / (np.max(fft_shift) + 1e-10)
        fy, fx = np.mgrid[:h, :w]
        fx = fx - cx
        fy = fy - cy
        freq_dist = np.sqrt(fx ** 2 + fy ** 2).astype(float)
        n_freq = 8
        for i in range(n_freq):
            r_inner = i * max_radius / n_freq
            r_outer = (i + 1) * max_radius / n_freq
            mask = (freq_dist >= r_inner) & (freq_dist < r_outer)
            if np.any(mask):
                features.append(float(np.mean(fft_norm[mask])))
            else:
                features.append(0.0)

        fingerprint = np.array(features[:self.config.feature_dim], dtype=np.float64)

        # 归一化指纹
        norm = np.linalg.norm(fingerprint)
        if norm > 1e-10:
            fingerprint = fingerprint / norm

        return fingerprint

    def _detect_anomaly(self, fingerprint: np.ndarray) -> Tuple[bool, float]:
        """检测 PSF 异常

        Args:
            fingerprint: 当前 PSF 指纹

        Returns:
            (是否异常, 异常分数)
        """
        if len(self._fingerprint_history) < 5:
            return False, 0.0

        # 计算与历史指纹的马氏距离
        history = np.array(list(self._fingerprint_history))
        mean_fp = np.mean(history, axis=0)
        std_fp = np.std(history, axis=0) + 1e-10

        z_score = np.abs(fingerprint - mean_fp) / std_fp
        anomaly_score = float(np.max(z_score))

        is_anomaly = anomaly_score > self.config.anomaly_threshold
        return is_anomaly, anomaly_score

    def analyze(
        self, image: np.ndarray, center: Optional[Tuple[float, float]] = None
    ) -> PSFFingerprintReport:
        """分析光斑 PSF

        Args:
            image: BGR 或灰度图像
            center: 光斑中心 (可选，自动检测)

        Returns:
            PSFFingerprintReport 分析报告
        """
        region = self._extract_psf_region(image, center)

        # 提取特征
        fingerprint = self._extract_fingerprint(region)
        focus_score = self._compute_focus_score(region)
        symmetry_score = self._compute_symmetry(region)
        psf_wx, psf_wy = self._compute_psf_width(region)
        strehl = self._estimate_strehl(region)

        # 椭圆度
        if psf_wx > 1e-6:
            ellipticity = abs(psf_wx - psf_wy) / max(psf_wx, psf_wy)
        else:
            ellipticity = 0.0

        # 异常检测
        is_anomaly, anomaly_score = self._detect_anomaly(fingerprint)

        # PSF 漂移
        drift = 0.0
        if self._reference_fingerprint is not None:
            drift = float(np.linalg.norm(fingerprint - self._reference_fingerprint))

        # 更新历史
        self._fingerprint_history.append(fingerprint)
        self._focus_history.append(focus_score)

        return PSFFingerprintReport(
            fingerprint=fingerprint,
            focus_score=focus_score,
            symmetry_score=symmetry_score,
            strehl_estimate=strehl,
            psf_width_x=psf_wx,
            psf_width_y=psf_wy,
            ellipticity=ellipticity,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            drift_from_reference=drift,
        )

    def get_diagnostics(self) -> dict:
        """获取分析器诊断信息"""
        return {
            "num_fingerprints": len(self._fingerprint_history),
            "has_reference": self._reference_fingerprint is not None,
            "avg_focus_score": float(np.mean(list(self._focus_history))) if self._focus_history else 0.0,
            "recent_anomalies": sum(
                1 for fp in list(self._fingerprint_history)[-10:]
                if self._detect_anomaly(fp)[0]
            ),
        }
