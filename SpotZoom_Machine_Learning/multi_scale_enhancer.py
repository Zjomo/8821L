"""
MultiScaleEnhancer - 多尺度图像增强器

灵感来源: Kornia, Deep Image Prior, Laplacian Pyramid
功能特点:
- 拉普拉斯金字塔分解
- 各层自适应增强
- 边缘感知滤波
- 细节保留去噪
- 纯numpy/cv2实现

技术路线:
- 多分辨率金字塔分解
- 每层独立增强策略
- 边缘感知平滑
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class EnhancementResult:
    """增强结果。"""
    enhanced_image: np.ndarray
    psnr_gain: float
    ssim_score: float
    processing_time_ms: float
    pyramid_levels: int
    detail_score: float  # 细节保留评分


@dataclass
class MultiScaleEnhancerConfig:
    """增强器配置。"""
    num_levels: int = 4
    # 各层增强参数
    denoise_strength: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    contrast_strength: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.3])
    sharpen_strength: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.2, 0.1])
    # 边缘感知参数
    edge_threshold: float = 0.1
    edge_preserve_sigma: float = 0.5
    # 全局参数
    gamma_correction: float = 1.0
    saturation_boost: float = 1.1


class MultiScaleEnhancer:
    """
    多尺度图像增强器。
    
    使用拉普拉斯金字塔将图像分解为不同频率层，
    对每个层应用针对性的增强策略，最后重建图像。
    
    增强策略:
    - 高频层: 细节增强、边缘锐化
    - 中频层: 对比度增强、噪声抑制
    - 低频层: 平滑去噪、光照均衡
    """
    
    def __init__(self, config: Optional[MultiScaleEnhancerConfig] = None):
        self.config = config or MultiScaleEnhancerConfig()
        self._gaussian_pyramid: List[np.ndarray] = []
        self._laplacian_pyramid: List[np.ndarray] = []
        LOGGER.info("MultiScaleEnhancer初始化完成，金字塔层数=%d", 
                   self.config.num_levels)
    
    def reset(self) -> None:
        """重置增强器状态。"""
        self._gaussian_pyramid.clear()
        self._laplacian_pyramid.clear()
    
    def enhance(self, image: np.ndarray) -> EnhancementResult:
        """
        增强图像。
        
        Args:
            image: 输入图像 (灰度或彩色)
            
        Returns:
            EnhancementResult包含增强结果和指标
        """
        t0 = time.perf_counter()
        
        # 保存原始图像用于计算指标
        original = image.copy()
        
        # 预处理
        if image.ndim == 2:
            gray_original = image.copy()
            working = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            working = image.copy()
            gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 构建金字塔
        self._build_pyramids(working)
        
        # 各层增强
        enhanced_pyramid = self._enhance_pyramid()
        
        # 重建图像
        reconstructed = self._reconstruct(enhanced_pyramid)
        
        # 后处理
        final = self._postprocess(reconstructed)
        
        # 转换为灰度计算指标
        if final.ndim == 3:
            gray_final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        else:
            gray_final = final
        
        # 计算指标
        psnr_gain = self._compute_psnr(gray_original, gray_final)
        ssim_score = self._compute_ssim(gray_original, gray_final)
        detail_score = self._compute_detail_score(gray_original, gray_final)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        LOGGER.debug("增强完成: PSNR=%.2fdB, SSIM=%.3f, 细节评分=%.3f, 耗时=%.2fms",
                    psnr_gain, ssim_score, detail_score, elapsed_ms)
        
        return EnhancementResult(
            enhanced_image=final,
            psnr_gain=psnr_gain,
            ssim_score=ssim_score,
            processing_time_ms=elapsed_ms,
            pyramid_levels=self.config.num_levels,
            detail_score=detail_score
        )
    
    def _build_pyramids(self, image: np.ndarray) -> None:
        """构建高斯和拉普拉斯金字塔。"""
        self._gaussian_pyramid.clear()
        self._laplacian_pyramid.clear()
        
        # 高斯金字塔
        current = image.astype(np.float32)
        self._gaussian_pyramid.append(current)
        
        for _ in range(self.config.num_levels - 1):
            if current.shape[0] < 4 or current.shape[1] < 4:
                break
            current = cv2.pyrDown(current)
            self._gaussian_pyramid.append(current)
        
        # 拉普拉斯金字塔
        for i in range(len(self._gaussian_pyramid) - 1):
            size = (
                self._gaussian_pyramid[i].shape[1],
                self._gaussian_pyramid[i].shape[0]
            )
            upsampled = cv2.pyrUp(self._gaussian_pyramid[i + 1], dstsize=size)
            laplacian = self._gaussian_pyramid[i] - upsampled
            self._laplacian_pyramid.append(laplacian)
        
        # 最顶层
        self._laplacian_pyramid.append(self._gaussian_pyramid[-1])
    
    def _enhance_pyramid(self) -> List[np.ndarray]:
        """对金字塔各层进行增强。"""
        enhanced = []
        
        num_levels = len(self._laplacian_pyramid)
        
        for i, level in enumerate(self._laplacian_pyramid):
            # 确定当前层的增强参数
            idx = min(i, len(self.config.denoise_strength) - 1)
            
            # 根据层数选择不同的增强策略
            if i == 0:
                # 最高频层: 细节增强 + 轻微去噪
                enhanced_level = self._enhance_high_frequency(
                    level, 
                    denoise=self.config.denoise_strength[idx],
                    sharpen=self.config.sharpen_strength[idx]
                )
            elif i < num_levels - 1:
                # 中间层: 对比度增强 + 噪声抑制
                enhanced_level = self._enhance_mid_frequency(
                    level,
                    denoise=self.config.denoise_strength[idx],
                    contrast=self.config.contrast_strength[idx]
                )
            else:
                # 最低频层: 平滑去噪 + 光照均衡
                enhanced_level = self._enhance_low_frequency(
                    level,
                    denoise=self.config.denoise_strength[idx]
                )
            
            enhanced.append(enhanced_level)
        
        return enhanced
    
    def _enhance_high_frequency(
        self, 
        level: np.ndarray,
        denoise: float,
        sharpen: float
    ) -> np.ndarray:
        """增强高频层 (细节层)。"""
        # 边缘感知平滑 (轻微去噪)
        if denoise > 0:
            smoothed = cv2.edgePreservingFilter(
                level.astype(np.uint8),
                flags=1,
                sigma_s=10,
                sigma_r=self.config.edge_preserve_sigma
            ).astype(np.float32)
            level = level * (1 - denoise) + smoothed * denoise
        
        # 细节增强 (拉普拉斯锐化)
        if sharpen > 0:
            kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]], dtype=np.float32)
            sharpened = cv2.filter2D(level, -1, kernel)
            level = level * (1 - sharpen) + sharpened * sharpen
        
        return level
    
    def _enhance_mid_frequency(
        self,
        level: np.ndarray,
        denoise: float,
        contrast: float
    ) -> np.ndarray:
        """增强中频层。"""
        # 对比度增强
        if contrast > 0:
            # CLAHE (对比度受限的自适应直方图均衡)
            if level.ndim == 3:
                lab = cv2.cvtColor(level.astype(np.uint8), cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)
                lab_enhanced = cv2.merge([l_enhanced, a, b])
                level = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR).astype(np.float32)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                level = clahe.apply(level.astype(np.uint8)).astype(np.float32)
        
        # 噪声抑制
        if denoise > 0:
            if level.ndim == 3:
                denoised = cv2.fastNlMeansDenoisingColored(
                    level.astype(np.uint8),
                    None,
                    h=3,
                    hColor=3,
                    templateWindowSize=7,
                    searchWindowSize=21
                ).astype(np.float32)
            else:
                denoised = cv2.fastNlMeansDenoising(
                    level.astype(np.uint8),
                    None,
                    h=3,
                    templateWindowSize=7,
                    searchWindowSize=21
                ).astype(np.float32)
            level = level * (1 - denoise) + denoised * denoise
        
        return level
    
    def _enhance_low_frequency(
        self,
        level: np.ndarray,
        denoise: float
    ) -> np.ndarray:
        """增强低频层。"""
        # 强去噪
        if denoise > 0:
            if level.ndim == 3:
                smoothed = cv2.GaussianBlur(level, (5, 5), 1.5)
            else:
                smoothed = cv2.GaussianBlur(level, (5, 5), 1.5)
            level = level * (1 - denoise) + smoothed * denoise
        
        # 光照均衡
        if level.ndim == 3:
            lab = cv2.cvtColor(level.astype(np.uint8), cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # 应用伽马校正
            l_float = l.astype(np.float32) / 255.0
            l_corrected = np.power(l_float, 1.0 / self.config.gamma_correction) * 255
            l = l_corrected.astype(np.uint8)
            lab = cv2.merge([l, a, b])
            level = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
        
        return level
    
    def _reconstruct(self, enhanced_pyramid: List[np.ndarray]) -> np.ndarray:
        """从增强后的金字塔重建图像。"""
        # 从最顶层开始重建
        current = enhanced_pyramid[-1]
        
        for i in range(len(enhanced_pyramid) - 2, -1, -1):
            # 上采样当前层
            size = (
                enhanced_pyramid[i].shape[1],
                enhanced_pyramid[i].shape[0]
            )
            upsampled = cv2.pyrUp(current, dstsize=size)
            
            # 加上拉普拉斯层
            current = upsampled + enhanced_pyramid[i]
        
        return current
    
    def _postprocess(self, image: np.ndarray) -> np.ndarray:
        """后处理。"""
        # 裁剪到有效范围
        image = np.clip(image, 0, 255)
        
        # 饱和度增强 (彩色图像)
        if image.ndim == 3 and self.config.saturation_boost != 1.0:
            hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.config.saturation_boost, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        
        return image.astype(np.uint8)
    
    def _compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算PSNR。"""
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse < 1e-10:
            return 100.0
        return 10 * np.log10(255.0 ** 2 / mse)
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算SSIM。"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
    
    def _compute_detail_score(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """计算细节保留评分。"""
        # 使用拉普拉斯算子提取细节
        lap_original = cv2.Laplacian(original, cv2.CV_64F)
        lap_enhanced = cv2.Laplacian(enhanced, cv2.CV_64F)
        
        # 计算细节能量的比值
        energy_original = np.mean(lap_original ** 2)
        energy_enhanced = np.mean(lap_enhanced ** 2)
        
        if energy_original < 1e-10:
            return 1.0 if energy_enhanced < 1e-10 else 0.0
        
        ratio = energy_enhanced / energy_original
        # 理想情况下应该有所提升，但不应过度增强
        score = 1.0 - abs(np.log(ratio + 1e-10))
        
        return float(np.clip(score, 0.0, 1.0))


# ============================================================
# 辅助函数
# ============================================================

def create_degraded_image(
    size: Tuple[int, int] = (512, 512),
    noise_level: float = 0.1,
    blur_sigma: float = 1.0,
    low_contrast: bool = True
) -> np.ndarray:
    """创建退化的测试图像。"""
    # 创建基础图像
    image = np.zeros((*size, 3), dtype=np.uint8)
    
    # 添加一些结构
    for i in range(10):
        x = np.random.randint(50, size[1] - 50)
        y = np.random.randint(50, size[0] - 50)
        radius = np.random.randint(10, 30)
        color = tuple(np.random.randint(100, 200, 3).tolist())
        cv2.circle(image, (x, y), radius, color, -1)
    
    # 添加噪声
    noise = np.random.normal(0, noise_level * 255, size).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise[:, :, None], 0, 255).astype(np.uint8)
    
    # 模糊
    if blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    
    # 降低对比度
    if low_contrast:
        image = (image.astype(np.float32) * 0.5 + 64).astype(np.uint8)
    
    return image


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建增强器
    enhancer = MultiScaleEnhancer()
    
    # 创建退化图像
    degraded = create_degraded_image()
    
    # 增强
    result = enhancer.enhance(degraded)
    
    print(f"增强结果:")
    print(f"  PSNR增益: {result.psnr_gain:.2f} dB")
    print(f"  SSIM评分: {result.ssim_score:.3f}")
    print(f"  细节评分: {result.detail_score:.3f}")
    print(f"  金字塔层数: {result.pyramid_levels}")
    print(f"  处理时间: {result.processing_time_ms:.2f} ms")
