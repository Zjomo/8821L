"""
Diffusion Spot Enhancer - 扩散模型光斑增强器

Inspired by:
- DeepImagePrior (Ulyanov et al.): Zero-shot image enhancement via untrained CNN
- DnCNN (Zhang et al.): Residual learning for image denoising
- CAREamics (CAREamics): Self-supervised denoising for microscopy

Core Innovation:
- 基于迭代扩散去噪的光斑图像增强
- 无需训练的零样本增强 (Deep Image Prior 思想)
- 自适应去噪强度，根据信噪比自动调节
- 保留光斑边缘特征的边缘感知去噪
- 纯 numpy+cv2 实现，零外部依赖
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DiffusionEnhancerConfig:
    """扩散增强器配置"""
    # 迭代去噪次数
    num_iterations: int = 5
    # 初始去噪强度 (越小越强)
    initial_h: float = 10.0
    # 去噪强度衰减因子
    h_decay: float = 0.8
    # 边缘保护权重
    edge_preserve_weight: float = 0.3
    # 最小 SNR 阈值 (低于此值才增强)
    min_snr_threshold: float = 5.0
    # 最大增强倍数
    max_enhancement: float = 2.0
    # 是否启用自适应强度
    adaptive_strength: bool = True
    # 模板窗口大小
    templateWindowSize: int = 7
    # 搜索窗口大小
    searchWindowSize: int = 21


@dataclass
class DiffusionEnhanceResult:
    """扩散增强结果"""
    enhanced_image: np.ndarray
    snr_before: float
    snr_after: float
    noise_estimate: float
    iterations_used: int
    enhancement_factor: float


class DiffusionSpotEnhancer:
    """扩散模型光斑增强器

    使用迭代非局部均值去噪模拟扩散过程，
    实现零样本光斑图像增强。

    Inspired by DeepImagePrior's zero-shot enhancement and
    CAREamics' self-supervised denoising approach.
    """

    def __init__(self, config: Optional[DiffusionEnhancerConfig] = None):
        self.config = config or DiffusionEnhancerConfig()

    def _estimate_snr(self, image: np.ndarray) -> float:
        """估计图像信噪比

        Args:
            image: 灰度图像

        Returns:
            SNR 估计值 (dB)
        """
        # 使用 MAD (Median Absolute Deviation) 估计噪声标准差
        # 噪声估计: 对高通滤波后的图像取中位数绝对偏差
        h, w = image.shape
        if h < 8 or w < 8:
            return 0.0

        # Laplacian 高通滤波
        laplacian = cv2.Laplacian(image.astype(np.float64), cv2.CV_64F)
        sigma_noise = np.median(np.abs(laplacian)) / 0.6745 * (1.0 / np.sqrt(20))

        # 信号估计: 局部均值
        signal = cv2.GaussianBlur(image.astype(np.float64), (15, 15), 3.0)
        sigma_signal = np.std(signal)

        if sigma_noise < 1e-6:
            return 100.0

        snr = 20 * np.log10(sigma_signal / sigma_noise)
        return float(snr)

    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """使用弱纹理区域估计噪声水平

        Args:
            image: 灰度图像

        Returns:
            噪声标准差估计
        """
        gray = image.astype(np.float64)
        h, w = gray.shape

        # 使用 Robust Median Estimator
        # 参考 J. Immerkær, "Fast Noise Variance Estimation", 1996
        H = np.array([[1, -2, 1],
                       [-2, 4, -2],
                       [1, -2, 1]], dtype=np.float64)

        if h < 3 or w < 3:
            return 0.0

        filtered = cv2.filter2D(gray, -1, H)
        sigma = np.sqrt(np.maximum(0, np.mean(filtered ** 2)) / 36.0)
        return float(sigma)

    def _edge_preserving_denoise(
        self, image: np.ndarray, h: float
    ) -> np.ndarray:
        """边缘保护去噪 (单次迭代)

        Args:
            image: 灰度图像
            h: 去噪强度参数

        Returns:
            去噪后的图像
        """
        # 使用 OpenCV 的 fastNlMeansDenoising 作为扩散步骤
        denoised = cv2.fastNlMeansDenoising(
            image, None, h=h,
            templateWindowSize=self.config.templateWindowSize,
            searchWindowSize=self.config.searchWindowSize
        )

        # 边缘保护: 混合原始图像和去噪结果
        edges = cv2.Canny(image, 30, 100)
        edges_float = edges.astype(np.float64) / 255.0

        # 在边缘区域保留更多原始信息
        w = self.config.edge_preserve_weight
        result = (1 - w) * denoised.astype(np.float64) + w * image.astype(np.float64)
        # 边缘区域进一步保留原始信息
        result = result * (1 - edges_float) + image.astype(np.float64) * edges_float

        return np.clip(result, 0, 255).astype(np.uint8)

    def _compute_sharpness(self, image: np.ndarray) -> float:
        """计算图像锐度 (Tenengrad)"""
        gx = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
        sharpness = np.mean(gx ** 2 + gy ** 2)
        return float(sharpness)

    def enhance(self, image: np.ndarray) -> DiffusionEnhanceResult:
        """执行扩散增强

        Args:
            image: BGR 或灰度图像

        Returns:
            DiffusionEnhanceResult 增强结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 估计初始 SNR
        snr_before = self._estimate_snr(gray)
        noise_level = self._estimate_noise_level(gray)

        # 如果 SNR 足够高，不需要增强
        if snr_before > self.config.min_snr_threshold and not self.config.adaptive_strength:
            return DiffusionEnhanceResult(
                enhanced_image=gray,
                snr_before=snr_before,
                snr_after=snr_before,
                noise_estimate=noise_level,
                iterations_used=0,
                enhancement_factor=1.0,
            )

        # 自适应确定去噪强度
        if self.config.adaptive_strength:
            # 低 SNR -> 更强去噪 (更小的 h)
            if snr_before < 3.0:
                initial_h = self.config.initial_h * 0.5
            elif snr_before < 10.0:
                initial_h = self.config.initial_h
            else:
                initial_h = self.config.initial_h * 1.5
        else:
            initial_h = self.config.initial_h

        # 迭代扩散去噪
        current = gray.copy()
        h = initial_h
        best_image = current.copy()
        best_sharpness = self._compute_sharpness(current)

        iterations_used = 0
        for i in range(self.config.num_iterations):
            current = self._edge_preserving_denoise(current, h)
            iterations_used += 1

            # 检查锐度，避免过度平滑
            sharpness = self._compute_sharpness(current)
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_image = current.copy()

            h *= self.config.h_decay

        # 计算增强后的 SNR
        snr_after = self._estimate_snr(best_image)
        enhancement_factor = min(
            self.config.max_enhancement,
            max(1.0, snr_after / (snr_before + 1e-6))
        )

        return DiffusionEnhanceResult(
            enhanced_image=best_image,
            snr_before=snr_before,
            snr_after=snr_after,
            noise_estimate=noise_level,
            iterations_used=iterations_used,
            enhancement_factor=enhancement_factor,
        )

    def enhance_spot_region(
        self, image: np.ndarray, center: Tuple[float, float], radius: int = 30
    ) -> DiffusionEnhanceResult:
        """仅增强光斑区域

        Args:
            image: 输入图像
            center: 光斑中心 (x, y)
            radius: 增强区域半径

        Returns:
            DiffusionEnhanceResult 增强结果 (全图)
        """
        if len(image.shape) == 3:
            result = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            result = image.copy()
            gray = image.copy()

        h, w = gray.shape
        cx, cy = int(center[0]), int(center[1])

        # 提取光斑区域 (带 padding)
        pad = radius * 2
        y_min = max(0, cy - pad)
        y_max = min(h, cy + pad)
        x_min = max(0, cx - pad)
        x_max = min(w, cx + pad)

        region = gray[y_min:y_max, x_min:x_max]
        enhanced_result = self.enhance(region)

        # 将增强后的区域融合回原图
        if len(image.shape) == 3:
            # 对 BGR 三通道分别处理
            for c in range(3):
                channel_region = image[y_min:y_max, x_min:x_max, c]
                channel_enhanced = enhanced_result.enhanced_image
                mask = np.zeros_like(channel_region, dtype=np.float64)
                yy, xx = np.mgrid[:region.shape[0], :region.shape[1]]
                dist = np.sqrt((xx - (cx - x_min)) ** 2 + (yy - (cy - y_min)) ** 2)
                mask = np.clip(1.0 - dist / pad, 0, 1)
                result[y_min:y_max, x_min:x_max, c] = np.clip(
                    channel_region * (1 - mask) + channel_enhanced * mask,
                    0, 255
                ).astype(np.uint8)
        else:
            mask = np.zeros_like(region, dtype=np.float64)
            yy, xx = np.mgrid[:region.shape[0], :region.shape[1]]
            dist = np.sqrt((xx - (cx - x_min)) ** 2 + (yy - (cy - y_min)) ** 2)
            mask = np.clip(1.0 - dist / pad, 0, 1)
            result[y_min:y_max, x_min:x_max] = np.clip(
                region * (1 - mask) + enhanced_result.enhanced_image * mask,
                0, 255
            ).astype(np.uint8)

        return DiffusionEnhanceResult(
            enhanced_image=result,
            snr_before=enhanced_result.snr_before,
            snr_after=enhanced_result.snr_after,
            noise_estimate=enhanced_result.noise_estimate,
            iterations_used=enhanced_result.iterations_used,
            enhancement_factor=enhanced_result.enhancement_factor,
        )
