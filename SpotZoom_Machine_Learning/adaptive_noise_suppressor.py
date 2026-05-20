"""
自适应噪声抑制器 (v7.0)

实时图像噪声估计与自适应抑制模块，提升光斑检测的鲁棒性。
支持多种降噪策略，根据噪声水平自动选择最优方法。

灵感来源:
- scikit-image restoration (https://scikit-image.org/) — 图像恢复算法
- OpenCV fastNlMeansDenoising — 非局部均值降噪
- BM3D (IEEE TIP 2007) — 块匹配 3D 协同滤波
- Wavelet shrinkage (Donoho 1995) — 小波阈值降噪

算法原理:
  1. 噪声估计: 使用 MAD (Median Absolute Deviation) 估计噪声标准差
  2. 策略选择: 根据噪声水平自动选择降噪方法
     - 低噪声 (< σ=5):   不降噪 (保留原始细节)
     - 中噪声 (5-20):    高斯滤波 + 边缘保护
     - 高噪声 (> 20):    非局部均值 / 小波降噪
  3. 自适应参数: 根据估计的噪声水平调整滤波器参数

外部依赖: numpy, cv2 (可选)
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class DenoiseStrategy(Enum):
    """降噪策略枚举。"""
    NONE = "none"               # 不降噪
    GAUSSIAN = "gaussian"       # 高斯滤波
    MEDIAN = "median"           # 中值滤波
    BILATERAL = "bilateral"     # 双边滤波 (边缘保护)
    NLMEANS = "nlmeans"         # 非局部均值
    WAVELET = "wavelet"         # 小波阈值降噪


@dataclass
class NoiseProfile:
    """噪声分析结果。"""
    estimated_sigma: float = 0.0      # 估计的噪声标准差
    noise_level: str = "low"          # "low", "medium", "high"
    strategy_used: DenoiseStrategy = DenoiseStrategy.NONE
    snr_before: float = 0.0           # 降噪前 SNR
    snr_after: float = 0.0            # 降噪后 SNR
    detail_preservation: float = 1.0  # 细节保留率 (0-1)
    processing_time_ms: float = 0.0   # 处理耗时


@dataclass
class DenoiseConfig:
    """降噪器配置。"""
    # 噪声水平阈值
    low_noise_threshold: float = 5.0     # σ < 5 为低噪声
    high_noise_threshold: float = 20.0   # σ > 20 为高噪声

    # 高斯滤波参数
    gaussian_kernel_size: int = 3

    # 中值滤波参数
    median_kernel_size: int = 3

    # 双边滤波参数
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # 非局部均值参数
    nlmeans_h: float = 10.0
    nlmeans_template_window: int = 7
    nlmeans_search_window: int = 21

    # 小波降噪参数
    wavelet_level: int = 3
    wavelet_threshold_factor: float = 3.0  # σ * factor

    # 自适应模式
    adaptive: bool = True  # 自动选择策略


class AdaptiveNoiseSuppressor:
    """自适应噪声抑制器。

    根据图像噪声水平自动选择最优降噪策略，在噪声抑制和细节保留之间取得平衡。

    使用示例:
        suppressor = AdaptiveNoiseSuppressor()
        denoised, profile = suppressor.suppress(noisy_image)
        print(f"Noise σ={profile.estimated_sigma:.1f}, Strategy={profile.strategy_used.value}")
    """

    def __init__(self, config: Optional[DenoiseConfig] = None):
        self._config = config or DenoiseConfig()
        self._noise_history: list = []

    @property
    def config(self) -> DenoiseConfig:
        return self._config

    def estimate_noise(self, image: np.ndarray) -> float:
        """使用 MAD 方法估计图像噪声标准差。

        基于鲁棒的最小绝对偏差估计器:
            σ = MAD / 0.6745

        对图像进行 Laplacian 滤波后，使用中值绝对偏差估计噪声。

        Args:
            image: 灰度图像 (H, W)

        Returns:
            估计的噪声标准差
        """
        if image.ndim == 3:
            gray = np.mean(image[:, :, :3], axis=2)
        else:
            gray = image.astype(np.float64)

        # Laplacian 核
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        h, w = gray.shape

        # 卷积 (手动实现，避免 cv2 依赖)
        padded = np.pad(gray, 1, mode='reflect')
        laplacian = np.zeros_like(gray)
        for i in range(3):
            for j in range(3):
                laplacian += kernel[i, j] * padded[i:i + h, j:j + w]

        # MAD 估计
        sigma = np.median(np.abs(laplacian)) / 0.6745
        return float(sigma)

    def suppress(
        self,
        image: np.ndarray,
        strategy: Optional[DenoiseStrategy] = None
    ) -> Tuple[np.ndarray, NoiseProfile]:
        """对图像进行自适应降噪。

        Args:
            image: 输入图像 (灰度或彩色)
            strategy: 指定降噪策略 (None=自动选择)

        Returns:
            (denoised_image, noise_profile): 降噪后的图像和噪声分析结果
        """
        import time
        t0 = time.perf_counter()

        # 转换为灰度
        if image.ndim == 3:
            gray = np.mean(image[:, :, :3], axis=2).astype(np.float64)
        else:
            gray = image.astype(np.float64)

        # 估计噪声
        sigma = self.estimate_noise(gray)
        self._noise_history.append(sigma)
        if len(self._noise_history) > 100:
            self._noise_history.pop(0)

        # 确定策略
        if strategy is None and self._config.adaptive:
            strategy = self._select_strategy(sigma)
        elif strategy is None:
            strategy = DenoiseStrategy.NONE

        # 计算 SNR (降噪前)
        snr_before = self._compute_snr(gray)

        # 执行降噪
        if strategy == DenoiseStrategy.NONE:
            denoised = gray.copy()
        elif strategy == DenoiseStrategy.GAUSSIAN:
            denoised = self._gaussian_denoise(gray, sigma)
        elif strategy == DenoiseStrategy.MEDIAN:
            denoised = self._median_denoise(gray, sigma)
        elif strategy == DenoiseStrategy.BILATERAL:
            denoised = self._bilateral_denoise(gray, sigma)
        elif strategy == DenoiseStrategy.NLMEANS:
            denoised = self._nlmeans_denoise(gray, sigma)
        elif strategy == DenoiseStrategy.WAVELET:
            denoised = self._wavelet_denoise(gray, sigma)
        else:
            denoised = gray.copy()

        # 计算 SNR (降噪后)
        snr_after = self._compute_snr(denoised)

        # 细节保留率
        detail_preservation = self._compute_detail_preservation(gray, denoised)

        # 噪声水平分类
        if sigma < self._config.low_noise_threshold:
            noise_level = "low"
        elif sigma < self._config.high_noise_threshold:
            noise_level = "medium"
        else:
            noise_level = "high"

        elapsed_ms = (time.perf_counter() - t0) * 1000

        profile = NoiseProfile(
            estimated_sigma=sigma,
            noise_level=noise_level,
            strategy_used=strategy,
            snr_before=snr_before,
            snr_after=snr_after,
            detail_preservation=detail_preservation,
            processing_time_ms=elapsed_ms,
        )

        logger.debug(
            f"NoiseSuppressor: σ={sigma:.2f}, level={noise_level}, "
            f"strategy={strategy.value}, SNR: {snr_before:.1f}->{snr_after:.1f}, "
            f"detail={detail_preservation:.2f}, time={elapsed_ms:.1f}ms"
        )

        return denoised, profile

    def _select_strategy(self, sigma: float) -> DenoiseStrategy:
        """根据噪声水平选择降噪策略。"""
        cfg = self._config
        if sigma < cfg.low_noise_threshold:
            return DenoiseStrategy.NONE
        elif sigma < cfg.high_noise_threshold:
            # 中等噪声: 优先使用双边滤波 (边缘保护)
            try:
                import cv2
                return DenoiseStrategy.BILATERAL
            except ImportError:
                return DenoiseStrategy.GAUSSIAN
        else:
            # 高噪声: 优先使用非局部均值
            try:
                import cv2
                return DenoiseStrategy.NLMEANS
            except ImportError:
                return DenoiseStrategy.WAVELET

    def _gaussian_denoise(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """高斯滤波降噪。"""
        try:
            import cv2
            ksize = self._config.gaussian_kernel_size
            # 自适应 sigma: 滤波器 sigma 与噪声 sigma 成比例
            blur_sigma = max(0.5, sigma * 0.3)
            return cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)
        except ImportError:
            # 简单均值滤波
            k = self._config.gaussian_kernel_size
            padded = np.pad(img, k // 2, mode='reflect')
            h, w = img.shape
            result = np.zeros_like(img)
            for di in range(k):
                for dj in range(k):
                    result += padded[di:di + h, dj:dj + w]
            return result / (k * k)

    def _median_denoise(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """中值滤波降噪。"""
        try:
            import cv2
            ksize = self._config.median_kernel_size
            return cv2.medianBlur(img.astype(np.uint8), ksize).astype(np.float64)
        except ImportError:
            # 简单中值 (仅 3x3)
            k = 3
            padded = np.pad(img, k // 2, mode='reflect')
            h, w = img.shape
            result = np.zeros_like(img)
            for di in range(k):
                for dj in range(k):
                    result = np.stack([result, padded[di:di + h, dj:dj + w]], axis=-1)
            return np.median(result, axis=-1)

    def _bilateral_denoise(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """双边滤波降噪 (边缘保护)。"""
        try:
            import cv2
            cfg = self._config
            # 自适应参数
            sc = max(10, min(sigma * 5, cfg.bilateral_sigma_color))
            ss = max(10, min(sigma * 3, cfg.bilateral_sigma_space))
            return cv2.bilateralFilter(
                img.astype(np.uint8),
                cfg.bilateral_d, sc, ss
            ).astype(np.float64)
        except ImportError:
            return self._gaussian_denoise(img, sigma)

    def _nlmeans_denoise(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """非局部均值降噪。"""
        try:
            import cv2
            cfg = self._config
            h = max(3, min(sigma * 1.5, cfg.nlmeans_h))
            result = cv2.fastNlMeansDenoising(
                img.astype(np.uint8),
                None,
                h=h,
                templateWindowSize=cfg.nlmeans_template_window,
                searchWindowSize=cfg.nlmeans_search_window,
            )
            return result.astype(np.float64)
        except ImportError:
            return self._wavelet_denoise(img, sigma)

    def _wavelet_denoise(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """小波阈值降噪 (Haar 小波，硬阈值)。"""
        level = self._config.wavelet_level
        threshold = sigma * self._config.wavelet_threshold_factor

        current = img.copy()
        coefficients = []

        # 分解
        for _ in range(level):
            # 行方向
            low_row = (current[:, :-1:2] + current[:, 1::2]) / 2.0
            high_row = (current[:, :-1:2] - current[:, 1::2]) / 2.0
            # 列方向
            ll = (low_row[:-1:2, :] + low_row[1::2, :]) / 2.0
            lh = (low_row[:-1:2, :] - low_row[1::2, :]) / 2.0
            hl = (high_row[:-1:2, :] + high_row[1::2, :]) / 2.0
            hh = (high_row[:-1:2, :] - high_row[1::2, :]) / 2.0
            coefficients.append((lh, hl, hh))
            current = ll

        # 硬阈值
        for lh, hl, hh in coefficients:
            lh[np.abs(lh) < threshold] = 0
            hl[np.abs(hl) < threshold] = 0
            hh[np.abs(hh) < threshold] = 0

        # 重构
        for lh, hl, hh in reversed(coefficients):
            # 列方向逆变换
            h, w = current.shape
            low_row = np.zeros((h * 2, w), dtype=np.float64)
            low_row[:-1:2, :] = current + lh
            low_row[1::2, :] = current - lh

            high_row = np.zeros((h * 2, w), dtype=np.float64)
            high_row[:-1:2, :] = hl + hh
            high_row[1::2, :] = hl - hh

            # 行方向逆变换
            h2, w2 = low_row.shape
            result = np.zeros((h2, w2 * 2), dtype=np.float64)
            result[:, :-1:2] = low_row + high_row
            result[:, 1::2] = low_row - high_row
            current = result

        return current

    def _compute_snr(self, img: np.ndarray) -> float:
        """计算信噪比。"""
        peak = np.max(img)
        noise = np.std(img) if peak > 1e-10 else 1e-10
        return float(peak / noise)

    def _compute_detail_preservation(
        self, original: np.ndarray, denoised: np.ndarray
    ) -> float:
        """计算细节保留率 (基于梯度相似性)。"""
        # Sobel 梯度
        def sobel_grad(img):
            gx = np.diff(img, axis=1)
            gy = np.diff(img, axis=0)
            gx = gx[:, :-1]
            gy = gy[:-1, :]
            return np.sqrt(gx**2 + gy**2)

        grad_orig = sobel_grad(original)
        grad_denoised = sobel_grad(denoised)

        # 确保尺寸一致
        min_h = min(grad_orig.shape[0], grad_denoised.shape[0])
        min_w = min(grad_orig.shape[1], grad_denoised.shape[1])
        grad_orig = grad_orig[:min_h, :min_w]
        grad_denoised = grad_denoised[:min_h, :min_w]

        orig_energy = np.sum(grad_orig**2)
        if orig_energy < 1e-10:
            return 1.0

        # 归一化互相关
        correlation = np.sum(grad_orig * grad_denoised) / (
            np.sqrt(np.sum(grad_orig**2) * np.sum(grad_denoised**2)) + 1e-10
        )
        return float(np.clip(correlation, 0, 1))

    def reset(self):
        """重置降噪器状态。"""
        self._noise_history.clear()
        logger.info("AdaptiveNoiseSuppressor: Reset")

    def get_noise_history(self) -> list:
        """获取噪声估计历史。"""
        return list(self._noise_history)

    def get_average_noise(self) -> float:
        """获取平均噪声水平。"""
        if not self._noise_history:
            return 0.0
        return float(np.mean(self._noise_history))
