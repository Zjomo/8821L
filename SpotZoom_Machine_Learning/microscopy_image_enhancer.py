"""
显微图像增强器 (MicroscopyImageEnhancer)

灵感来源:
- DeepTrack2 (https://github.com/DeepTrackAI/DeepTrack2) — 模块化图像特征管道
- ZeroCostDL4Mic — 深度学习显微图像增强 (本模块使用经典方法替代)
- TrackPy — 图像预处理与背景扣除

算法原理:
- CLAHE (Contrast Limited Adaptive Histogram Equalization) — 限制对比度自适应直方图均衡
- Wiener Deconvolution — 维纳滤波反卷积
- Richardson-Lucy Deconvolution — 基于泊松噪声模型的迭代反卷积
- Rolling Ball Background Subtraction — 滚动球背景估计与扣除

功能:
- 多尺度对比度增强 (基于 CLAHE)
- 维纳滤波和 Richardson-Lucy 反卷积
- 滚动球背景扣除
- 多种降噪方法 (中值滤波、高斯滤波、双边滤波)
- 图像锐化 (非锐化掩模)

依赖: numpy, cv2, scipy (无外部深度学习框架依赖)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

LOGGER = logging.getLogger("SpotZoom.MicroscopyImageEnhancer")


@dataclass
class EnhancerConfig:
    """增强器配置参数。"""
    # --- CLAHE 参数 ---
    clahe_clip_limit: float = 2.0  # CLAHE 对比度限制
    clahe_grid_size: Tuple[int, int] = (8, 8)  # CLAHE 网格大小
    clahe_enabled: bool = True  # 是否启用 CLAHE

    # --- 多尺度参数 ---
    multi_scale_levels: int = 3  # 多尺度层数
    multi_scale_enabled: bool = True  # 是否启用多尺度增强

    # --- 反卷积参数 ---
    deconvolution_method: str = "wiener"  # "wiener", "richardson_lucy", "none"
    deconvolution_iterations: int = 20  # Richardson-Lucy 迭代次数
    deconvolution_psf_size: int = 15  # PSF 估计大小 (像素)
    deconvolution_psf_sigma: float = 2.0  # PSF 高斯 sigma (像素)
    wiener_noise_power: float = 0.01  # 维纳滤波噪声功率比

    # --- 背景扣除参数 ---
    background_method: str = "rolling_ball"  # "rolling_ball", "morphology", "none"
    rolling_ball_radius: float = 50.0  # 滚动球半径 (像素)
    morphology_kernel_size: int = 15  # 形态学核大小

    # --- 降噪参数 ---
    denoise_method: str = "none"  # "median", "gaussian", "bilateral", "none"
    denoise_kernel_size: int = 3  # 降噪核大小
    bilateral_sigma_color: float = 75.0  # 双边滤波颜色 sigma
    bilateral_sigma_space: float = 75.0  # 双边滤波空间 sigma

    # --- 锐化参数 ---
    sharpen_enabled: bool = False  # 是否启用锐化
    sharpen_amount: float = 1.0  # 锐化强度
    sharpen_sigma: float = 1.0  # 锐化高斯 sigma

    # --- 输出参数 ---
    output_dtype: str = "float64"  # 输出数据类型: "float64", "uint8", "uint16"
    normalize_output: bool = True  # 是否归一化输出


@dataclass
class EnhancementResult:
    """增强结果。"""
    enhanced_image: np.ndarray = field(default_factory=lambda: np.array([]))
    background: np.ndarray = field(default_factory=lambda: np.array([]))
    psf_estimate: np.ndarray = field(default_factory=lambda: np.array([]))
    processing_steps: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class MicroscopyImageEnhancer:
    """显微图像增强器。

    提供多种经典图像处理方法用于增强显微镜光斑图像，
    包括对比度增强、反卷积、背景扣除和降噪。

    Parameters
    ----------
    config : EnhancerConfig, optional
        增强器配置参数。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[EnhancerConfig] = None):
        self._cfg = config if config is not None else EnhancerConfig()

    def _apply_clahe(
        self, image: np.ndarray
    ) -> np.ndarray:
        """应用 CLAHE 对比度增强。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        np.ndarray
            增强后的图像。
        """
        # 转换为 uint8 用于 CLAHE
        img_uint8 = self._to_uint8(image)

        clahe = cv2.createCLAHE(
            clipLimit=self._cfg.clahe_clip_limit,
            tileGridSize=self._cfg.clahe_grid_size,
        )
        result = clahe.apply(img_uint8)

        # 转回浮点
        return result.astype(np.float64) / 255.0

    def _apply_multi_scale_clahe(
        self, image: np.ndarray
    ) -> np.ndarray:
        """多尺度 CLAHE 增强。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        np.ndarray
            多尺度增强后的图像。
        """
        result = np.zeros_like(image, dtype=np.float64)
        n_levels = self._cfg.multi_scale_levels

        for level in range(n_levels):
            # 计算当前尺度的缩放因子
            scale = 2 ** level

            # 下采样
            h, w = image.shape[:2]
            small_h = max(h // scale, 4)
            small_w = max(w // scale, 4)
            small = cv2.resize(image, (small_w, small_h),
                               interpolation=cv2.INTER_AREA)

            # 在当前尺度应用 CLAHE
            enhanced_small = self._apply_clahe(small)

            # 上采样回原始尺寸
            enhanced = cv2.resize(enhanced_small, (w, h),
                                  interpolation=cv2.INTER_LINEAR)

            # 权重随尺度递减
            weight = 1.0 / (level + 1)
            result += weight * enhanced

        # 归一化
        total_weight = sum(1.0 / (l + 1) for l in range(n_levels))
        result /= total_weight

        return result

    def _estimate_psf_gaussian(
        self, size: int, sigma: float
    ) -> np.ndarray:
        """估计高斯 PSF。

        Parameters
        ----------
        size : int
            PSF 大小 (奇数)。
        sigma : float
            高斯 sigma。

        Returns
        -------
        np.ndarray
            归一化 PSF。
        """
        if size % 2 == 0:
            size += 1

        psf = np.zeros((size, size), dtype=np.float64)
        center = size // 2
        y, x = np.mgrid[:size, :size]
        psf = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
        psf /= psf.sum()

        return psf

    def _wiener_deconvolution(
        self,
        image: np.ndarray,
        psf: np.ndarray,
    ) -> np.ndarray:
        """维纳滤波反卷积。

        Parameters
        ----------
        image : np.ndarray
            退化图像。
        psf : np.ndarray
            点扩散函数。

        Returns
        -------
        np.ndarray
            反卷积结果。
        """
        h, w = image.shape
        psf_h, psf_w = psf.shape

        # 零填充 PSF 到图像大小
        psf_padded = np.zeros((h, w), dtype=np.float64)
        py = (h - psf_h) // 2
        px = (w - psf_w) // 2
        psf_padded[py:py + psf_h, px:px + psf_w] = psf

        # FFT
        img_fft = np.fft.fft2(image)
        psf_fft = np.fft.fft2(np.fft.ifftshift(psf_padded))

        # 维纳滤波
        noise_power = self._cfg.wiener_noise_power
        H_conj = np.conj(psf_fft)
        H_sq = np.abs(psf_fft) ** 2
        wiener_filter = H_conj / (H_sq + noise_power)

        result_fft = img_fft * wiener_filter
        result = np.real(np.fft.ifft2(result_fft))

        return result

    def _richardson_lucy_deconvolution(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        n_iterations: int,
    ) -> np.ndarray:
        """Richardson-Lucy 反卷积。

        Parameters
        ----------
        image : np.ndarray
            退化图像。
        psf : np.ndarray
            点扩散函数。
        n_iterations : int
            迭代次数。

        Returns
        -------
        np.ndarray
            反卷积结果。
        """
        image = np.clip(image, 1e-12, None)
        estimate = np.copy(image)

        # PSF 和翻转 PSF 的 FFT
        h, w = image.shape
        psf_h, psf_w = psf.shape
        psf_padded = np.zeros((h, w), dtype=np.float64)
        py = (h - psf_h) // 2
        px = (w - psf_w) // 2
        psf_padded[py:py + psf_h, px:px + psf_w] = psf

        psf_fft = np.fft.fft2(np.fft.ifftshift(psf_padded))
        psf_flip_fft = np.conj(psf_fft)

        eps = 1e-12

        for i in range(n_iterations):
            # 前向: estimate * PSF
            conv = np.real(np.fft.ifft2(np.fft.fft2(estimate) * psf_fft))
            conv = np.clip(conv, eps, None)

            # 比值
            ratio = image / conv

            # 后向: ratio * PSF_flip
            corr = np.real(np.fft.ifft2(np.fft.fft2(ratio) * psf_flip_fft))
            corr = np.clip(corr, eps, None)

            # 更新
            estimate = estimate * corr

            if (i + 1) % 5 == 0:
                LOGGER.debug("Richardson-Lucy 迭代 %d/%d", i + 1, n_iterations)

        return estimate

    def _rolling_ball_background(
        self, image: np.ndarray, radius: float
    ) -> np.ndarray:
        """滚动球背景估计。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        radius : float
            滚动球半径 (像素)。

        Returns
        -------
        np.ndarray
            背景图像。
        """
        # 使用形态学开运算近似滚动球
        kernel_size = int(2 * radius + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 创建椭圆形结构元素
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # 形态学开运算作为背景估计
        background = cv2.morphologyEx(
            self._to_uint8(image), cv2.MORPH_OPEN, kernel
        )

        return background.astype(np.float64) / 255.0

    def _morphology_background(
        self, image: np.ndarray, kernel_size: int
    ) -> np.ndarray:
        """形态学背景估计。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        kernel_size : int
            结构元素大小。

        Returns
        -------
        np.ndarray
            背景图像。
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # 多次膨胀取最大值
        img_uint8 = self._to_uint8(image)
        background = cv2.dilate(img_uint8, kernel, iterations=3)

        return background.astype(np.float64) / 255.0

    def _apply_denoise(self, image: np.ndarray) -> np.ndarray:
        """应用降噪。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        np.ndarray
            降噪后的图像。
        """
        method = self._cfg.denoise_method
        ksize = self._cfg.denoise_kernel_size
        if ksize % 2 == 0:
            ksize += 1

        if method == "median":
            return cv2.medianBlur(self._to_uint8(image), ksize).astype(np.float64) / 255.0
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif method == "bilateral":
            return cv2.bilateralFilter(
                self._to_uint8(image), ksize,
                self._cfg.bilateral_sigma_color,
                self._cfg.bilateral_sigma_space,
            ).astype(np.float64) / 255.0
        elif method == "none":
            return image
        else:
            raise ValueError(f"未知降噪方法: {method}")

    def _apply_sharpen(self, image: np.ndarray) -> np.ndarray:
        """非锐化掩模锐化。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        np.ndarray
            锐化后的图像。
        """
        # 高斯模糊
        blurred = cv2.GaussianBlur(image, (0, 0), self._cfg.sharpen_sigma)

        # 非锐化掩模
        mask = image - blurred
        sharpened = image + self._cfg.sharpen_amount * mask

        return np.clip(sharpened, 0, None)

    def _to_uint8(self, image: np.ndarray) -> np.ndarray:
        """将图像转换为 uint8。"""
        img = np.asarray(image, dtype=np.float64)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        return (img * 255).astype(np.uint8)

    def _compute_metrics(
        self, original: np.ndarray, enhanced: np.ndarray
    ) -> Dict[str, float]:
        """计算增强前后的图像质量指标。

        Parameters
        ----------
        original : np.ndarray
            原始图像。
        enhanced : np.ndarray
            增强后的图像。

        Returns
        -------
        Dict[str, float]
            质量指标。
        """
        metrics: Dict[str, float] = {}

        # 对比度 (标准差)
        metrics["contrast_original"] = float(np.std(original))
        metrics["contrast_enhanced"] = float(np.std(enhanced))
        metrics["contrast_ratio"] = (
            metrics["contrast_enhanced"] / (metrics["contrast_original"] + 1e-12)
        )

        # 信噪比 (简化: 峰值/标准差)
        metrics["snr_original"] = float(
            np.max(original) / (np.std(original) + 1e-12)
        )
        metrics["snr_enhanced"] = float(
            np.max(enhanced) / (np.std(enhanced) + 1e-12)
        )

        # 锐度 (梯度幅值均值)
        gx_o = np.diff(original, axis=1)
        gy_o = np.diff(original, axis=0)
        metrics["sharpness_original"] = float(
            np.mean(np.abs(gx_o)) + np.mean(np.abs(gy_o))
        )

        gx_e = np.diff(enhanced, axis=1)
        gy_e = np.diff(enhanced, axis=0)
        metrics["sharpness_enhanced"] = float(
            np.mean(np.abs(gx_e)) + np.mean(np.abs(gy_e))
        )

        return metrics

    def enhance(self, image: np.ndarray) -> EnhancementResult:
        """增强显微图像。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (二维灰度图)。

        Returns
        -------
        EnhancementResult
            增强结果。
        """
        image = np.asarray(image, dtype=np.float64)

        if image.ndim != 2:
            raise ValueError(f"图像应为二维数组，实际维度: {image.ndim}")

        LOGGER.info("开始图像增强: 形状 %s", image.shape)

        original = image.copy()
        steps: List[str] = []
        background = np.zeros_like(image)
        psf_estimate = np.zeros((self._cfg.deconvolution_psf_size,) * 2)

        # --- 步骤 1: 背景扣除 ---
        if self._cfg.background_method != "none":
            if self._cfg.background_method == "rolling_ball":
                background = self._rolling_ball_background(
                    image, self._cfg.rolling_ball_radius
                )
            elif self._cfg.background_method == "morphology":
                background = self._morphology_background(
                    image, self._cfg.morphology_kernel_size
                )
            image = image - background
            image = np.clip(image, 0, None)
            steps.append(f"背景扣除 ({self._cfg.background_method})")

        # --- 步骤 2: 降噪 ---
        if self._cfg.denoise_method != "none":
            image = self._apply_denoise(image)
            steps.append(f"降噪 ({self._cfg.denoise_method})")

        # --- 步骤 3: 对比度增强 ---
        if self._cfg.clahe_enabled:
            if self._cfg.multi_scale_enabled:
                image = self._apply_multi_scale_clahe(image)
                steps.append("多尺度 CLAHE 增强")
            else:
                image = self._apply_clahe(image)
                steps.append("CLAHE 增强")

        # --- 步骤 4: 反卷积 ---
        if self._cfg.deconvolution_method != "none":
            psf_estimate = self._estimate_psf_gaussian(
                self._cfg.deconvolution_psf_size,
                self._cfg.deconvolution_psf_sigma,
            )

            if self._cfg.deconvolution_method == "wiener":
                image = self._wiener_deconvolution(image, psf_estimate)
                steps.append("维纳反卷积")
            elif self._cfg.deconvolution_method == "richardson_lucy":
                image = self._richardson_lucy_deconvolution(
                    image, psf_estimate, self._cfg.deconvolution_iterations
                )
                steps.append(
                    f"Richardson-Lucy 反卷积 ({self._cfg.deconvolution_iterations} 迭代)"
                )

            image = np.clip(image, 0, None)

        # --- 步骤 5: 锐化 ---
        if self._cfg.sharpen_enabled:
            image = self._apply_sharpen(image)
            steps.append("非锐化掩模锐化")

        # --- 归一化输出 ---
        if self._cfg.normalize_output and image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())

        # --- 输出数据类型转换 ---
        if self._cfg.output_dtype == "uint8":
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        elif self._cfg.output_dtype == "uint16":
            image = (np.clip(image, 0, 1) * 65535).astype(np.uint16)
        else:
            image = image.astype(np.float64)

        # --- 计算质量指标 ---
        metrics = self._compute_metrics(original, image)

        result = EnhancementResult(
            enhanced_image=image,
            background=background,
            psf_estimate=psf_estimate,
            processing_steps=steps,
            metrics=metrics,
        )

        LOGGER.info(
            "图像增强完成: 处理步骤=%s, 对比度提升=%.2fx",
            " -> ".join(steps),
            metrics.get("contrast_ratio", 1.0),
        )

        return result

    def reset(self):
        """重置增强器状态。"""
        LOGGER.info("显微图像增强器已重置")
