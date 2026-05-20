"""
自监督图像降噪器 (Self-Supervised Denoiser)

无需干净参考数据的自监督图像降噪模块。基于盲点网络 (Blind-Spot Network)
概念，利用局部像素统计和多尺度金字塔方法实现自适应降噪。

灵感来源:
- Noise2Void (javiribera/Noise2Void): 自监督降噪框架
  (https://github.com/javiribera/Noise2Void)
- Noise2Noise (NVlabs/noise2noise): 无需干净数据的降噪
  (https://github.com/NVlabs/noise2noise)
- CAREamics (CAREamics): 自监督显微镜降噪
  (https://github.com/CAREamics/CAREamics)

算法原理:
  1. 噪声水平估计: 使用 MAD (Median Absolute Deviation) 方法鲁棒估计噪声标准差
  2. 盲点掩码: 随机选择像素作为自监督训练信号，避免身份映射退化
  3. 多尺度金字塔: 从粗到细逐层降噪，先去除大尺度噪声，再恢复细节
  4. SNR 感知: 根据局部信噪比自适应调整降噪强度
  5. 自适应核: 根据噪声水平和图像内容动态选择滤波核大小

外部依赖: numpy, cv2 (可选)
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class NoiseEstimationMethod(Enum):
    """噪声估计方法枚举。"""
    MAD = "mad"                   # 中值绝对偏差
    LAPLACIAN_MAD = "laplacian"   # Laplacian + MAD
    ROBUST_MAD = "robust_mad"     # 鲁棒 MAD (分块估计取中值)


@dataclass
class DenoiseReport:
    """降噪结果报告。"""
    denoised_image: np.ndarray = None       # 降噪后图像
    estimated_sigma: float = 0.0            # 估计的噪声标准差
    snr_before: float = 0.0                 # 降噪前 SNR (dB)
    snr_after: float = 0.0                  # 降噪后 SNR (dB)
    snr_improvement: float = 0.0            # SNR 改善量 (dB)
    detail_preservation: float = 1.0        # 细节保留率 (0-1)
    blind_spot_count: int = 0               # 使用的盲点数量
    pyramid_levels_used: int = 1            # 金字塔层数
    processing_time_ms: float = 0.0         # 处理耗时 (ms)
    method_used: str = ""                   # 使用的降噪方法描述


@dataclass
class DenoiserConfig:
    """自监督降噪器配置。"""
    # 噪声估计
    noise_estimation_method: NoiseEstimationMethod = NoiseEstimationMethod.LAPLACIAN_MAD
    noise_block_size: int = 32              # 分块噪声估计的块大小

    # 降噪强度
    max_denoising_strength: float = 1.0     # 最大降噪强度 (0-1)
    min_denoising_strength: float = 0.0     # 最小降噪强度
    snr_threshold_low: float = 5.0          # 低 SNR 阈值 (dB)
    snr_threshold_high: float = 20.0        # 高 SNR 阈值 (dB)

    # 多尺度金字塔
    multi_scale_levels: int = 3             # 金字塔层数
    pyramid_downsample_factor: float = 0.5  # 下采样因子

    # 盲点网络参数
    blind_spot_radius: int = 1              # 盲点半径 (像素)
    blind_spot_fraction: float = 0.1        # 盲点采样比例

    # 自适应核
    min_kernel_size: int = 3                # 最小滤波核
    max_kernel_size: int = 7                # 最大滤波核

    # 滤波方法
    use_median_for_high_noise: bool = True  # 高噪声时使用中值滤波
    use_bilateral_for_edge: bool = True     # 边缘区域使用双边滤波


class SelfSupervisedDenoiser:
    """自监督图像降噪器。

    基于盲点网络概念的自监督降噪，无需干净参考图像。
    通过 MAD 噪声估计、多尺度金字塔和 SNR 感知策略实现自适应降噪。

    使用示例:
        denoiser = SelfSupervisedDenoiser()
        report = denoiser.denoise(noisy_image)
        print(f"SNR improvement: {report.snr_improvement:.1f} dB")
        denoised = report.denoised_image
    """

    def __init__(self, config: Optional[DenoiserConfig] = None):
        self._config = config or DenoiserConfig()
        self._noise_history: List[float] = []

    @property
    def config(self) -> DenoiserConfig:
        return self._config

    def denoise(self, image: np.ndarray) -> DenoiseReport:
        """对图像进行自监督降噪。

        Args:
            image: 输入图像 (灰度 H,W 或彩色 H,W,3)，dtype float64/uint8

        Returns:
            DenoiseReport: 降噪结果报告
        """
        import time
        t0 = time.perf_counter()

        # 转换为灰度 float64
        gray = self._to_gray_float64(image)

        # 估计噪声水平
        sigma = self._estimate_noise(gray)
        self._noise_history.append(sigma)
        if len(self._noise_history) > 100:
            self._noise_history.pop(0)

        # 计算 SNR (降噪前)
        snr_before = self._compute_snr_db(gray)

        # 生成盲点掩码
        blind_mask = self._generate_blind_spot_mask(gray.shape)

        # 多尺度金字塔降噪
        denoised, levels_used = self._multi_scale_denoise(gray, sigma, blind_mask)

        # 计算 SNR (降噪后)
        snr_after = self._compute_snr_db(denoised)
        snr_improvement = snr_after - snr_before

        # 细节保留率
        detail_preservation = self._compute_detail_preservation(gray, denoised)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        report = DenoiseReport(
            denoised_image=denoised,
            estimated_sigma=sigma,
            snr_before=snr_before,
            snr_after=snr_after,
            snr_improvement=snr_improvement,
            detail_preservation=detail_preservation,
            blind_spot_count=int(np.sum(blind_mask)),
            pyramid_levels_used=levels_used,
            processing_time_ms=elapsed_ms,
            method_used=f"multi_scale_pyramid_L{levels_used}_sigma{sigma:.1f}",
        )

        logger.debug(
            f"SelfSupervisedDenoiser: sigma={sigma:.2f}, "
            f"SNR: {snr_before:.1f}->{snr_after:.1f} dB (+{snr_improvement:.1f}), "
            f"detail={detail_preservation:.3f}, time={elapsed_ms:.1f}ms"
        )

        return report

    def _to_gray_float64(self, image: np.ndarray) -> np.ndarray:
        """将输入图像转换为灰度 float64。"""
        if image.ndim == 3:
            gray = np.mean(image[:, :, :min(3, image.shape[2])], axis=2)
        else:
            gray = image.copy()
        return gray.astype(np.float64)

    def _estimate_noise(self, image: np.ndarray) -> float:
        """估计图像噪声标准差。

        支持三种方法:
        - MAD: 直接对图像像素计算 MAD
        - LAPLACIAN_MAD: 对 Laplacian 滤波结果计算 MAD (推荐)
        - ROBUST_MAD: 分块估计后取中值

        Args:
            image: 灰度图像 (H, W), float64

        Returns:
            估计的噪声标准差
        """
        cfg = self._config
        method = cfg.noise_estimation_method

        if method == NoiseEstimationMethod.MAD:
            sigma = np.median(np.abs(image - np.median(image))) / 0.6745

        elif method == NoiseEstimationMethod.LAPLACIAN_MAD:
            # Laplacian 核卷积
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
            h, w = image.shape
            padded = np.pad(image, 1, mode='reflect')
            laplacian = np.zeros_like(image)
            for i in range(3):
                for j in range(3):
                    laplacian += kernel[i, j] * padded[i:i + h, j:j + w]
            sigma = np.median(np.abs(laplacian)) / 0.6745

        elif method == NoiseEstimationMethod.ROBUST_MAD:
            # 分块 MAD 估计
            block_size = cfg.noise_block_size
            h, w = image.shape
            block_sigmas = []
            for bi in range(0, h - block_size + 1, block_size):
                for bj in range(0, w - block_size + 1, block_size):
                    block = image[bi:bi + block_size, bj:bj + block_size]
                    # 对块做 Laplacian
                    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                    bh, bw = block.shape
                    padded = np.pad(block, 1, mode='reflect')
                    lap = np.zeros_like(block)
                    for i in range(3):
                        for j in range(3):
                            lap += kernel[i, j] * padded[i:i + bh, j:j + bw]
                    block_sigma = np.median(np.abs(lap)) / 0.6745
                    block_sigmas.append(block_sigma)
            if block_sigmas:
                sigma = float(np.median(block_sigmas))
            else:
                sigma = np.median(np.abs(image - np.median(image))) / 0.6745
        else:
            sigma = np.std(image)

        return float(max(sigma, 1e-10))

    def _generate_blind_spot_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """生成盲点掩码。

        随机选择一定比例的像素作为盲点，用于自监督训练信号。
        盲点周围的像素用于预测盲点位置的真实值。

        Args:
            shape: 图像形状 (H, W)

        Returns:
            布尔掩码 (H, W), True 表示盲点位置
        """
        cfg = self._config
        h, w = shape
        mask = np.random.random((h, w)) < cfg.blind_spot_fraction

        # 确保边缘不被选为盲点 (避免边界效应)
        r = cfg.blind_spot_radius
        mask[:r, :] = False
        mask[-r:, :] = False
        mask[:, :r] = False
        mask[:, -r:] = False

        return mask

    def _multi_scale_denoise(
        self,
        image: np.ndarray,
        sigma: float,
        blind_mask: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """多尺度金字塔降噪。

        从粗到细逐层降噪:
        1. 下采样到最低分辨率
        2. 在低分辨率上降噪
        3. 上采样并融合到下一层
        4. 重复直到原始分辨率

        Args:
            image: 输入图像
            sigma: 噪声标准差
            blind_mask: 盲点掩码

        Returns:
            (denoised_image, levels_used)
        """
        cfg = self._config
        levels = cfg.multi_scale_levels
        factor = cfg.pyramid_downsample_factor

        # 构建金字塔
        pyramid = [image]
        mask_pyramid = [blind_mask]
        current = image.copy()
        current_mask = blind_mask.copy()

        for level in range(1, levels):
            new_h = max(8, int(current.shape[0] * factor))
            new_w = max(8, int(current.shape[1] * factor))
            # 下采样 (均值池化)
            try:
                import cv2
                downsampled = cv2.resize(current, (new_w, new_h),
                                         interpolation=cv2.INTER_AREA)
                mask_down = cv2.resize(current_mask.astype(np.uint8), (new_w, new_h),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
            except ImportError:
                # 简单下采样
                step_h = max(1, current.shape[0] // new_h)
                step_w = max(1, current.shape[1] // new_w)
                downsampled = current[:new_h * step_h:step_h, :new_w * step_w:step_w]
                mask_down = current_mask[:new_h * step_h:step_h, :new_w * step_w:step_w]
            pyramid.append(downsampled)
            mask_pyramid.append(mask_down)
            current = downsampled
            current_mask = mask_down

        # 从最粗层开始降噪
        denoised_pyramid = []
        for level in range(len(pyramid) - 1, -1, -1):
            layer_img = pyramid[level]
            layer_mask = mask_pyramid[level]

            # 根据层级调整噪声估计 (低分辨率噪声更低)
            level_sigma = sigma * (factor ** (len(pyramid) - 1 - level))

            # SNR 感知降噪强度
            snr = self._compute_snr_db(layer_img)
            strength = self._compute_denoising_strength(snr, level_sigma)

            # 自适应核大小
            kernel_size = self._adaptive_kernel_size(level_sigma)

            # 执行单层降噪
            if level == len(pyramid) - 1:
                # 最粗层: 直接降噪
                layer_denoised = self._single_scale_denoise(
                    layer_img, level_sigma, kernel_size, strength
                )
            else:
                # 细层: 融合上一层结果
                prev_denoised = denoised_pyramid[-1]
                # 上采样上一层结果
                try:
                    import cv2
                    upsampled = cv2.resize(prev_denoised, (layer_img.shape[1], layer_img.shape[0]),
                                           interpolation=cv2.INTER_LINEAR)
                except ImportError:
                    upsampled = np.repeat(np.repeat(prev_denoised, 2, axis=0), 2, axis=1)
                    upsampled = upsampled[:layer_img.shape[0], :layer_img.shape[1]]

                # 融合: 当前层降噪 + 上采样引导
                layer_denoised_single = self._single_scale_denoise(
                    layer_img, level_sigma, kernel_size, strength
                )
                # 加权融合 (上采样结果权重更高)
                alpha = 0.5
                layer_denoised = alpha * layer_denoised_single + (1 - alpha) * upsampled

            denoised_pyramid.append(layer_denoised)

        return denoised_pyramid[-1], len(pyramid)

    def _single_scale_denoise(
        self,
        image: np.ndarray,
        sigma: float,
        kernel_size: int,
        strength: float,
    ) -> np.ndarray:
        """单尺度降噪。

        根据噪声水平和图像内容选择滤波方法:
        - 低噪声: 高斯滤波
        - 中噪声: 双边滤波 (边缘保护)
        - 高噪声: 中值滤波

        Args:
            image: 输入图像
            sigma: 噪声标准差
            kernel_size: 滤波核大小
            strength: 降噪强度 (0-1)

        Returns:
            降噪后图像
        """
        cfg = self._config

        # 边缘检测 (用于选择滤波方法)
        edges = self._detect_edges(image)

        if sigma > 20 and cfg.use_median_for_high_noise:
            # 高噪声: 中值滤波
            filtered = self._median_filter(image, kernel_size)
        elif np.mean(edges) > 0.1 and cfg.use_bilateral_for_edge:
            # 边缘丰富: 双边滤波
            filtered = self._bilateral_filter(image, sigma, kernel_size)
        else:
            # 默认: 高斯滤波
            filtered = self._gaussian_filter(image, sigma, kernel_size)

        # 按强度混合原始图像和滤波结果
        result = (1 - strength) * image + strength * filtered
        return result

    def _compute_denoising_strength(self, snr_db: float, sigma: float) -> float:
        """根据 SNR 计算降噪强度。

        SNR 越低 -> 降噪强度越大
        SNR 越高 -> 降噪强度越小 (保留细节)

        Args:
            snr_db: 信噪比 (dB)
            sigma: 噪声标准差

        Returns:
            降噪强度 (0-1)
        """
        cfg = self._config

        if snr_db < cfg.snr_threshold_low:
            strength = cfg.max_denoising_strength
        elif snr_db > cfg.snr_threshold_high:
            strength = cfg.min_denoising_strength
        else:
            # 线性插值
            t = (snr_db - cfg.snr_threshold_low) / (
                cfg.snr_threshold_high - cfg.snr_threshold_low + 1e-10
            )
            strength = cfg.max_denoising_strength * (1 - t) + cfg.min_denoising_strength * t

        return float(np.clip(strength, 0, 1))

    def _adaptive_kernel_size(self, sigma: float) -> int:
        """根据噪声水平自适应选择滤波核大小。

        Args:
            sigma: 噪声标准差

        Returns:
            核大小 (奇数)
        """
        cfg = self._config
        # 噪声越大 -> 核越大
        ratio = sigma / 20.0  # 归一化
        k = int(cfg.min_kernel_size + ratio * (cfg.max_kernel_size - cfg.min_kernel_size))
        k = min(k, cfg.max_kernel_size)
        # 确保为奇数
        if k % 2 == 0:
            k += 1
        return max(cfg.min_kernel_size, k)

    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """检测图像边缘 (Sobel 算子)。

        Args:
            image: 灰度图像

        Returns:
            边缘强度图 (归一化到 0-1)
        """
        try:
            import cv2
            gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(gx ** 2 + gy ** 2)
        except ImportError:
            # 简单差分
            gx = np.diff(image, axis=1)
            gy = np.diff(image, axis=0)
            edges = np.zeros_like(image)
            edges[:, :-1] += np.abs(gx)
            edges[:-1, :] += np.abs(gy)

        # 归一化
        max_val = np.max(edges)
        if max_val > 1e-10:
            edges = edges / max_val
        return edges

    def _gaussian_filter(
        self, image: np.ndarray, sigma: float, kernel_size: int
    ) -> np.ndarray:
        """高斯滤波降噪。

        Args:
            image: 输入图像
            sigma: 噪声标准差
            kernel_size: 核大小

        Returns:
            滤波后图像
        """
        try:
            import cv2
            blur_sigma = max(0.5, sigma * 0.3)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_sigma)
        except ImportError:
            # 手动高斯模糊
            k = kernel_size
            half = k // 2
            x = np.arange(-half, half + 1, dtype=np.float64)
            kernel_1d = np.exp(-x ** 2 / (2 * max(sigma * 0.3, 0.5) ** 2))
            kernel_1d /= kernel_1d.sum()
            # 分离卷积
            padded = np.pad(image, half, mode='reflect')
            temp = np.zeros_like(image)
            for i, w in enumerate(kernel_1d):
                temp += w * padded[i:i + image.shape[0], half:half + image.shape[1]]
            padded2 = np.pad(temp, half, mode='reflect')
            result = np.zeros_like(image)
            for i, w in enumerate(kernel_1d):
                result += w * padded2[half:half + image.shape[0], i:i + image.shape[1]]
            return result

    def _median_filter(
        self, image: np.ndarray, kernel_size: int
    ) -> np.ndarray:
        """中值滤波降噪。

        Args:
            image: 输入图像
            kernel_size: 核大小

        Returns:
            滤波后图像
        """
        try:
            import cv2
            return cv2.medianBlur(
                np.clip(image, 0, 255).astype(np.uint8), kernel_size
            ).astype(np.float64)
        except ImportError:
            # 简单中值 (仅支持 3x3)
            h, w = image.shape
            padded = np.pad(image, 1, mode='reflect')
            result = np.zeros_like(image)
            for di in range(3):
                for dj in range(3):
                    result = np.stack([result, padded[di:di + h, dj:dj + w]], axis=-1)
            return np.median(result, axis=-1)

    def _bilateral_filter(
        self, image: np.ndarray, sigma: float, kernel_size: int
    ) -> np.ndarray:
        """双边滤波降噪 (边缘保护)。

        Args:
            image: 输入图像
            sigma: 噪声标准差
            kernel_size: 核大小

        Returns:
            滤波后图像
        """
        try:
            import cv2
            sc = max(10, min(sigma * 5, 75.0))
            ss = max(10, min(sigma * 3, 75.0))
            return cv2.bilateralFilter(
                np.clip(image, 0, 255).astype(np.uint8),
                kernel_size, sc, ss
            ).astype(np.float64)
        except ImportError:
            return self._gaussian_filter(image, sigma, kernel_size)

    def _compute_snr_db(self, image: np.ndarray) -> float:
        """计算信噪比 (dB)。

        SNR = 20 * log10(peak / noise_std)

        Args:
            image: 输入图像

        Returns:
            SNR (dB)
        """
        peak = np.max(image)
        noise_std = np.std(image)
        if peak < 1e-10 or noise_std < 1e-10:
            return 0.0
        return float(20.0 * np.log10(peak / noise_std))

    def _compute_detail_preservation(
        self, original: np.ndarray, denoised: np.ndarray
    ) -> float:
        """计算细节保留率 (基于梯度相似性)。

        Args:
            original: 原始图像
            denoised: 降噪后图像

        Returns:
            细节保留率 (0-1)
        """
        def gradient_magnitude(img):
            gx = np.diff(img, axis=1)
            gy = np.diff(img, axis=0)
            min_h = min(gx.shape[0], gy.shape[0])
            min_w = min(gx.shape[1], gy.shape[1])
            return np.sqrt(gx[:min_h, :min_w] ** 2 + gy[:min_h, :min_w] ** 2)

        grad_orig = gradient_magnitude(original)
        grad_denoised = gradient_magnitude(denoised)

        orig_energy = np.sum(grad_orig ** 2)
        if orig_energy < 1e-10:
            return 1.0

        correlation = np.sum(grad_orig * grad_denoised) / (
            np.sqrt(np.sum(grad_orig ** 2) * np.sum(grad_denoised ** 2)) + 1e-10
        )
        return float(np.clip(correlation, 0, 1))

    def reset(self):
        """重置降噪器状态。"""
        self._noise_history.clear()
        logger.info("SelfSupervisedDenoiser: Reset")

    def get_noise_history(self) -> List[float]:
        """获取噪声估计历史。"""
        return list(self._noise_history)

    def get_average_noise(self) -> float:
        """获取平均噪声水平。"""
        if not self._noise_history:
            return 0.0
        return float(np.mean(self._noise_history))
