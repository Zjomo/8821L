"""
合成数据生成器 (Synthetic Data Generator)

参考开源项目:
  - DeepTrack2 (https://github.com/DeepTrack/DeepTrack2): 深度学习光学粒子追踪

核心思想:
  从 DeepTrack2 的合成数据生成引擎中借鉴，实现光学像差模拟、
  光斑图像生成和域随机化，用于训练鲁棒的光斑检测和对准模型。

  在 SpotZoom 场景中:
  - 合成光斑 → 模拟不同条件下的光斑图像
  - 像差模拟 → 球差、彗差、像散等典型光学像差
  - 域随机化 → 随机化噪声、背景、像差参数

创新点:
  1. 物理准确的像差模拟 (基于 Zernike 多项式)
  2. 多种噪声模型 (泊松、高斯、读出噪声)
  3. 域随机化用于鲁棒训练数据生成
  4. 可配置的光斑 PSF 生成
  5. 批量数据集生成接口

纯 numpy + cv2 实现，无外部依赖。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyntheticDataConfig:
    """合成数据生成器配置。"""
    # 图像尺寸
    image_size: int = 128
    # 光斑中心位置 (相对于图像中心, 像素)
    spot_center: Tuple[float, float] = (0.0, 0.0)
    # 光斑 sigma (像素)
    spot_sigma: float = 3.0
    # 光斑峰值强度
    spot_peak_intensity: float = 1000.0
    # 背景强度
    background_intensity: float = 10.0
    # 泊松噪声使能
    poisson_noise: bool = True
    # 高斯噪声 sigma
    gaussian_noise_sigma: float = 5.0
    # 读出噪声 sigma
    read_noise_sigma: float = 2.0
    # 像差参数: 球差 (波数)
    spherical_aberration: float = 0.0
    # 像差参数: 彗差 (波数)
    coma: float = 0.0
    # 像差参数: 像散 (波数)
    astigmatism: float = 0.0
    # 像差参数: 离焦 (波数)
    defocus: float = 0.0
    # 像差参数: 树叶形 (波数)
    trefoil: float = 0.0
    # 域随机化使能
    domain_randomization: bool = True
    # 随机化范围: sigma
    sigma_range: Tuple[float, float] = (1.5, 8.0)
    # 随机化范围: 峰值强度
    intensity_range: Tuple[float, float] = (100.0, 5000.0)
    # 随机化范围: 背景强度
    bg_range: Tuple[float, float] = (0.0, 50.0)
    # 随机化范围: 偏移量
    offset_range: Tuple[float, float] = (-20.0, 20.0)
    # 随机化范围: 像差
    aberration_range: Tuple[float, float] = (-0.5, 0.5)
    # PSF 生成方法: "gaussian", "airy", "aberrated"
    psf_method: str = "aberrated"
    # 数据类型
    dtype: np.dtype = field(default=np.float64)
    # 种子 (0 = 随机)
    seed: int = 0


@dataclass
class SyntheticSample:
    """单个合成样本。"""
    # 图像数据
    image: np.ndarray = field(default_factory=lambda: np.zeros((128, 128)))
    # 光斑真实中心 X
    true_center_x: float = 0.0
    # 光斑真实中心 Y
    true_center_y: float = 0.0
    # 真实 sigma
    true_sigma: float = 3.0
    # 真实峰值强度
    true_peak: float = 1000.0
    # 使用的像差参数
    aberrations: Dict[str, float] = field(default_factory=dict)
    # 使用的噪声参数
    noise_params: Dict[str, float] = field(default_factory=dict)


class SyntheticDataGenerator:
    """合成数据生成器。

    生成带有光学像差和噪声的合成光斑图像，用于训练和验证
    光斑检测与对准算法。

    Parameters
    ----------
    config : SyntheticDataConfig
        生成器配置参数。

    References
    ----------
    .. [1] DeepTrack2 documentation:
           https://deeptrack.readthedocs.io/
    .. [2] Noll, R. J. (1976). "Zernike polynomials and atmospheric
           turbulence." JOSA, 66(3), 207-211.
    .. [3] Goodman, J. W. (2015). "Statistical Optics." 2nd ed.
    """

    def __init__(self, config: Optional[SyntheticDataConfig] = None) -> None:
        self.config = config or SyntheticDataConfig()
        self._rng = np.random.RandomState(self.config.seed if self.config.seed > 0 else None)
        logger.info(
            f"SyntheticDataGenerator 初始化: "
            f"尺寸={self.config.image_size}, "
            f"PSF方法={self.config.psf_method}, "
            f"域随机化={self.config.domain_randomization}"
        )

    def generate_spot(
        self,
        center: Optional[Tuple[float, float]] = None,
        sigma: Optional[float] = None,
        peak: Optional[float] = None,
        aberrations: Optional[Dict[str, float]] = None,
    ) -> SyntheticSample:
        """生成单个合成光斑图像。

        Parameters
        ----------
        center : Tuple[float, float], optional
            光斑中心 (相对于图像中心)。默认使用配置值。
        sigma : float, optional
            光斑 sigma。默认使用配置值。
        peak : float, optional
            峰值强度。默认使用配置值。
        aberrations : Dict[str, float], optional
            像差参数字典。默认使用配置值。

        Returns
        -------
        SyntheticSample
            合成样本。
        """
        cfg = self.config
        size = cfg.image_size

        # 参数设置
        cx = center[0] if center is not None else cfg.spot_center[0]
        cy = center[1] if center is not None else cfg.spot_center[1]
        sig = sigma if sigma is not None else cfg.spot_sigma
        pk = peak if peak is not None else cfg.spot_peak_intensity

        # 像差参数
        if aberrations is None:
            aberrations = {
                "spherical": cfg.spherical_aberration,
                "coma": cfg.coma,
                "astigmatism": cfg.astigmatism,
                "defocus": cfg.defocus,
                "trefoil": cfg.trefoil,
            }

        # 生成 PSF
        psf = self._generate_psf(size, sig, aberrations)

        # 放置光斑
        image = self._place_spot(size, psf, cx, cy, pk)

        # 添加背景
        image += cfg.background_intensity

        # 添加噪声
        noise_params = self._add_noise(image)

        sample = SyntheticSample(
            image=image,
            true_center_x=cx,
            true_center_y=cy,
            true_sigma=sig,
            true_peak=pk,
            aberrations=aberrations.copy(),
            noise_params=noise_params,
        )

        return sample

    def add_aberration(
        self,
        image: np.ndarray,
        aberration_type: str,
        magnitude: float,
    ) -> np.ndarray:
        """向图像添加像差效果。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        aberration_type : str
            像差类型: "spherical", "coma", "astigmatism", "defocus", "trefoil"。
        magnitude : float
            像差大小 (波数)。

        Returns
        -------
        np.ndarray
            添加像差后的图像。
        """
        aberrations = {
            "spherical": 0.0,
            "coma": 0.0,
            "astigmatism": 0.0,
            "defocus": 0.0,
            "trefoil": 0.0,
        }
        aberrations[aberration_type] = magnitude

        size = image.shape[0]
        sigma = self.config.spot_sigma
        psf = self._generate_psf(size, sigma, aberrations)

        # 使用新 PSF 重新卷积
        # 简化处理: 直接在频域相乘
        image_fft = np.fft.fft2(image)
        psf_fft = np.fft.fft2(psf)
        psf_fft_norm = psf_fft / (np.max(np.abs(psf_fft)) + 1e-10)

        result = np.real(np.fft.ifft2(image_fft * psf_fft_norm))
        return np.maximum(result, 0)

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """向图像添加噪声。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        np.ndarray
            添加噪声后的图像。
        """
        result = image.copy()
        params = self._add_noise(result)
        return result

    def generate_dataset(
        self,
        n_samples: int,
        randomize: bool = True,
    ) -> List[SyntheticSample]:
        """生成合成数据集。

        Parameters
        ----------
        n_samples : int
            样本数量。
        randomize : bool
            是否启用域随机化。

        Returns
        -------
        List[SyntheticSample]
            合成样本列表。
        """
        samples = []
        logger.info(f"开始生成数据集: {n_samples} 个样本, 随机化={randomize}")

        for i in range(n_samples):
            if randomize:
                params = self.randomize_domain()
                sample = self.generate_spot(
                    center=(params["offset_x"], params["offset_y"]),
                    sigma=params["sigma"],
                    peak=params["peak"],
                    aberrations=params["aberrations"],
                )
            else:
                sample = self.generate_spot()

            samples.append(sample)

            if (i + 1) % 100 == 0:
                logger.info(f"已生成 {i + 1}/{n_samples} 个样本")

        logger.info(f"数据集生成完成: {len(samples)} 个样本")
        return samples

    def randomize_domain(self) -> Dict[str, Any]:
        """生成随机化的域参数。

        Returns
        -------
        dict
            随机化参数字典。
        """
        cfg = self.config
        params: Dict[str, Any] = {}

        if cfg.domain_randomization:
            params["sigma"] = float(self._rng.uniform(*cfg.sigma_range))
            params["peak"] = float(self._rng.uniform(*cfg.intensity_range))
            params["bg"] = float(self._rng.uniform(*cfg.bg_range))
            params["offset_x"] = float(self._rng.uniform(*cfg.offset_range))
            params["offset_y"] = float(self._rng.uniform(*cfg.offset_range))

            params["aberrations"] = {
                "spherical": float(self._rng.uniform(*cfg.aberration_range)),
                "coma": float(self._rng.uniform(*cfg.aberration_range)),
                "astigmatism": float(self._rng.uniform(*cfg.aberration_range)),
                "defocus": float(self._rng.uniform(*cfg.aberration_range)),
                "trefoil": float(self._rng.uniform(*cfg.aberration_range)),
            }

            params["noise_sigma"] = float(self._rng.uniform(0, cfg.gaussian_noise_sigma * 2))
        else:
            params["sigma"] = cfg.spot_sigma
            params["peak"] = cfg.spot_peak_intensity
            params["bg"] = cfg.background_intensity
            params["offset_x"] = cfg.spot_center[0]
            params["offset_y"] = cfg.spot_center[1]
            params["aberrations"] = {
                "spherical": cfg.spherical_aberration,
                "coma": cfg.coma,
                "astigmatism": cfg.astigmatism,
                "defocus": cfg.defocus,
                "trefoil": cfg.trefoil,
            }
            params["noise_sigma"] = cfg.gaussian_noise_sigma

        return params

    # ============ 内部方法 ============

    def _generate_psf(
        self,
        size: int,
        sigma: float,
        aberrations: Dict[str, float],
    ) -> np.ndarray:
        """生成 PSF。"""
        if self.config.psf_method == "gaussian":
            return self._gaussian_psf(size, sigma)
        elif self.config.psf_method == "airy":
            return self._airy_psf(size, sigma)
        elif self.config.psf_method == "aberrated":
            return self._aberrated_psf(size, sigma, aberrations)
        else:
            logger.warning(f"未知 PSF 方法: {self.config.psf_method}")
            return self._gaussian_psf(size, sigma)

    def _gaussian_psf(self, size: int, sigma: float) -> np.ndarray:
        """高斯 PSF。"""
        cy, cx = size / 2, size / 2
        yy, xx = np.mgrid[:size, :size]
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        psf = np.exp(-r2 / (2 * sigma ** 2))
        psf /= psf.sum()
        return psf

    def _airy_psf(self, size: int, sigma: float) -> np.ndarray:
        """Airy PSF (简化版)。"""
        cy, cx = size / 2, size / 2
        yy, xx = np.mgrid[:size, :size]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / sigma
        r = np.maximum(r, 1e-10)

        # Airy 函数: (2 * J1(x) / x)^2
        from numpy import sinc
        # J1(x)/x 近似
        x = np.pi * r
        airy = (2 * sinc(x)) ** 2
        airy[r < 1e-10] = 1.0
        airy /= airy.sum()
        return airy

    def _aberrated_psf(
        self,
        size: int,
        sigma: float,
        aberrations: Dict[str, float],
    ) -> np.ndarray:
        """带像差的 PSF (基于 Zernike 相位)。"""
        # 生成理想 PSF
        ideal_psf = self._gaussian_psf(size, sigma)

        # 生成 Zernike 相位屏
        phase = self._generate_aberration_phase(size, aberrations)

        # 相位调制: 在频域应用
        psf_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ideal_psf)))
        phase_modulation = np.exp(1j * phase)
        aberrated_fft = psf_fft * phase_modulation
        aberrated_psf = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(aberrated_fft)))) ** 2

        # 归一化
        total = aberrated_psf.sum()
        if total > 1e-10:
            aberrated_psf /= total

        return aberrated_psf

    def _generate_aberration_phase(
        self,
        size: int,
        aberrations: Dict[str, float],
    ) -> np.ndarray:
        """生成像差相位屏。"""
        cy, cx = size / 2, size / 2
        yy, xx = np.mgrid[:size, :size]
        x = (xx - cx) / cx  # 归一化到 [-1, 1]
        y = (yy - cy) / cy
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)

        # 单位圆掩膜
        mask = r <= 1.0

        phase = np.zeros((size, size))

        # 离焦: Z4 = sqrt(3) * (2r^2 - 1)
        phase += aberrations.get("defocus", 0) * np.sqrt(3) * (2 * r ** 2 - 1)

        # 像散: Z5 = sqrt(6) * r^2 * sin(2*theta)
        phase += aberrations.get("astigmatism", 0) * np.sqrt(6) * r ** 2 * np.sin(2 * theta)

        # 彗差 Y: Z7 = sqrt(8) * (3r^3 - 2r) * sin(theta)
        phase += aberrations.get("coma", 0) * np.sqrt(8) * (3 * r ** 3 - 2 * r) * np.sin(theta)

        # 球差: Z11 = sqrt(5) * (6r^4 - 6r^2 + 1)
        phase += aberrations.get("spherical", 0) * np.sqrt(5) * (6 * r ** 4 - 6 * r ** 2 + 1)

        # 树叶形: Z9 = sqrt(8) * r^3 * sin(3*theta)
        phase += aberrations.get("trefoil", 0) * np.sqrt(8) * r ** 3 * np.sin(3 * theta)

        phase *= mask
        return phase

    def _place_spot(
        self,
        size: int,
        psf: np.ndarray,
        offset_x: float,
        offset_y: float,
        peak: float,
    ) -> np.ndarray:
        """在图像上放置光斑。"""
        image = np.zeros((size, size), dtype=np.float64)
        cy, cx = size / 2, size / 2

        # 亚像素偏移 (使用相位偏移实现)
        psf_h, psf_w = psf.shape
        shift_x = offset_x
        shift_y = offset_y

        # 使用 FFT 实现亚像素位移
        psf_fft = np.fft.fft2(psf)
        h, w = psf.shape
        yy, xx = np.mgrid[:h, :w]
        phase_shift = np.exp(
            -2j * np.pi * (xx * shift_x / w + yy * shift_y / h)
        )
        shifted_psf = np.real(np.fft.ifft2(psf_fft * phase_shift))
        shifted_psf = np.maximum(shifted_psf, 0)
        shifted_psf /= (shifted_psf.sum() + 1e-10)

        # 放置到图像中心
        image = shifted_psf * peak
        return image

    def _add_noise(self, image: np.ndarray) -> Dict[str, float]:
        """向图像添加噪声 (原地修改)。"""
        params: Dict[str, float] = {}

        # 泊松噪声 (信号相关)
        if self.config.poisson_noise:
            # 确保非负
            image_clean = np.maximum(image, 0)
            noisy = self._rng.poisson(image_clean.astype(np.float64)).astype(np.float64)
            image[:] = noisy
            params["poisson_scale"] = 1.0

        # 高斯噪声
        if self.config.gaussian_noise_sigma > 0:
            noise = self._rng.normal(0, self.config.gaussian_noise_sigma, image.shape)
            image[:] += noise
            params["gaussian_sigma"] = self.config.gaussian_noise_sigma

        # 读出噪声
        if self.config.read_noise_sigma > 0:
            read_noise = self._rng.normal(0, self.config.read_noise_sigma, image.shape)
            image[:] += read_noise
            params["read_noise_sigma"] = self.config.read_noise_sigma

        # 确保非负
        np.maximum(image, 0, out=image)

        return params
