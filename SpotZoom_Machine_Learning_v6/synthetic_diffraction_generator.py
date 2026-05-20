"""
Synthetic Diffraction Generator - 合成衍射数据生成器

Inspired by:
- prysm (brandondube): PSF and diffraction pattern generation
- HCIPy (ehpor): Wavefront propagation simulation
- DeepTrack2 (DeepTrackAI): Synthetic particle image generation

Core Innovation:
- 基于傅里叶光学的合成衍射光斑生成
- 支持多种像差模式 (Zernike 多项式)
- 可配置噪声模型 (泊松+高斯)
- 数据增强: 随机像差/偏移/旋转
- 依赖 numpy, cv2
"""

from __future__ import annotations

import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DiffractionGeneratorConfig:
    """衍射数据生成器配置"""
    # 图像大小
    image_size: int = 128
    # 像素大小 (微米)
    pixel_size_um: float = 0.1
    # 波长 (微米)
    wavelength_um: float = 0.633
    # 数值孔径
    numerical_aperture: float = 0.4
    # Zernike 像差系数 (前 15 项)
    zernike_coeffs: Optional[List[float]] = None
    # 泊松噪声水平
    poisson_noise: bool = True
    # 高斯噪声标准差
    gaussian_noise_std: float = 5.0
    # 背景强度
    background_level: float = 10.0
    # 随机偏移范围 (像素)
    random_shift_range: float = 5.0
    # 随机旋转范围 (度)
    random_rotation_range: float = 5.0
    # 随机缩放范围
    random_scale_range: float = 0.1


@dataclass
class DiffractionSample:
    """衍射数据样本"""
    image: np.ndarray
    center: Tuple[float, float]
    zernike_coeffs: List[float]
    snr: float
    psf_width: float


class SyntheticDiffractionGenerator:
    """合成衍射数据生成器

    基于傅里叶光学原理生成合成光斑图像，
    用于训练数据增强和算法验证。

    Inspired by prysm's PSF generation and DeepTrack2's
    synthetic particle image generation.
    """

    def __init__(self, config: Optional[DiffractionGeneratorConfig] = None):
        self.config = config or DiffractionGeneratorConfig()
        if self.config.zernike_coeffs is None:
            self.config.zernike_coeffs = [0.0] * 15

    def _zernike_polynomial(
        self, n: int, m: int, rho: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """计算 Zernike 多项式

        Args:
            n: 径向阶数
            m: 角向频率
            rho: 归一化径向坐标
            theta: 角度坐标

        Returns:
            Zernike 多项式值
        """
        # 简化的 Zernike 多项式实现
        if n == 0 and m == 0:
            return np.ones_like(rho)
        elif n == 1 and m == 1:
            return rho * np.cos(theta)
        elif n == 1 and m == -1:
            return rho * np.sin(theta)
        elif n == 2 and m == 0:
            return 2 * rho ** 2 - 1
        elif n == 2 and m == 2:
            return rho ** 2 * np.cos(2 * theta)
        elif n == 2 and m == -2:
            return rho ** 2 * np.sin(2 * theta)
        elif n == 3 and m == 1:
            return (3 * rho ** 3 - 2 * rho) * np.cos(theta)
        elif n == 3 and m == -1:
            return (3 * rho ** 3 - 2 * rho) * np.sin(theta)
        elif n == 3 and m == 3:
            return rho ** 3 * np.cos(3 * theta)
        elif n == 3 and m == -3:
            return rho ** 3 * np.sin(3 * theta)
        elif n == 4 and m == 0:
            return 6 * rho ** 4 - 6 * rho ** 2 + 1
        elif n == 4 and m == 2:
            return (4 * rho ** 4 - 3 * rho ** 2) * np.cos(2 * theta)
        elif n == 4 and m == -2:
            return (4 * rho ** 4 - 3 * rho ** 2) * np.sin(2 * theta)
        elif n == 4 and m == 4:
            return rho ** 4 * np.cos(4 * theta)
        elif n == 4 and m == -4:
            return rho ** 4 * np.sin(4 * theta)
        else:
            return np.zeros_like(rho)

    def _generate_pupil_function(
        self, zernike_coeffs: List[float], size: int
    ) -> np.ndarray:
        """生成瞳函数

        Args:
            zernike_coeffs: Zernike 系数列表
            size: 图像大小

        Returns:
            复数瞳函数
        """
        # 创建坐标网格
        y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
        rho = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)

        # 圆形光瞳
        pupil_mask = (rho <= 1.0).astype(np.float64)

        # 波前相位
        phase = np.zeros((size, size), dtype=np.float64)

        # Zernike 多项式索引 (Noll 排序)
        zernike_orders = [
            (0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2),
            (3, -1), (3, 1), (3, -3), (3, 3), (4, 0),
            (4, 2), (4, -2), (4, 4), (4, -4),
        ]

        for i, coeff in enumerate(zernike_coeffs[:len(zernike_orders)]):
            if abs(coeff) > 1e-10:
                n, m = zernike_orders[i]
                phase += coeff * self._zernike_polynomial(n, m, rho, theta)

        # 瞳函数 = 光瞳 × exp(i × 相位)
        pupil = pupil_mask * np.exp(1j * phase)
        return pupil

    def generate(
        self,
        zernike_coeffs: Optional[List[float]] = None,
        add_noise: bool = True,
        random_augment: bool = False,
    ) -> DiffractionSample:
        """生成合成衍射光斑图像

        Args:
            zernike_coeffs: 自定义 Zernike 系数
            add_noise: 是否添加噪声
            random_augment: 是否随机增强

        Returns:
            DiffractionSample 衍射数据样本
        """
        size = self.config.image_size
        coeffs = zernike_coeffs or self.config.zernike_coeffs

        # 生成瞳函数
        pupil = self._generate_pupil_function(coeffs, size)

        # PSF = |FFT(pupil)|²
        psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupil)))) ** 2

        # 归一化
        psf = psf / (np.max(psf) + 1e-10)

        # 转换为 uint8 图像
        image = (psf * 255).astype(np.float64)

        # 添加背景
        image += self.config.background_level

        # 随机增强
        center = (size / 2.0, size / 2.0)
        if random_augment:
            # 随机偏移
            shift_x = np.random.uniform(-self.config.random_shift_range, self.config.random_shift_range)
            shift_y = np.random.uniform(-self.config.random_shift_range, self.config.random_shift_range)
            M_shift = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
            image = self._affine_transform(image, M_shift)
            center = (center[0] + shift_x, center[1] + shift_y)

            # 随机旋转
            angle = np.random.uniform(-self.config.random_rotation_range, self.config.random_rotation_range)
            M_rot = cv2_rotation_matrix(size, angle)
            image = self._affine_transform(image, M_rot)

            # 随机缩放
            scale = 1.0 + np.random.uniform(-self.config.random_scale_range, self.config.random_scale_range)
            M_scale = cv2_scale_matrix(size, scale)
            image = self._affine_transform(image, M_scale)

        # 添加噪声
        if add_noise:
            if self.config.poisson_noise:
                image = np.random.poisson(np.maximum(image, 0)).astype(np.float64)
            if self.config.gaussian_noise_std > 0:
                image += np.random.normal(0, self.config.gaussian_noise_std, image.shape)

        image = np.clip(image, 0, 255).astype(np.uint8)

        # 计算 SNR
        peak = float(np.max(image))
        bg_region = image[:size // 4, :size // 4]
        bg_mean = float(np.mean(bg_region))
        bg_std = float(np.std(bg_region))
        snr = (peak - bg_mean) / (bg_std + 1e-6)

        # PSF 宽度
        psf_width = self._compute_psf_width(psf)

        return DiffractionSample(
            image=image,
            center=center,
            zernike_coeffs=coeffs,
            snr=snr,
            psf_width=psf_width,
        )

    def generate_batch(
        self,
        n_samples: int,
        random_zernike: bool = True,
        zernike_std: float = 0.1,
    ) -> List[DiffractionSample]:
        """生成一批合成数据

        Args:
            n_samples: 样本数量
            random_zernike: 是否随机生成 Zernike 系数
            zernike_std: Zernike 系数标准差

        Returns:
            样本列表
        """
        samples = []
        for _ in range(n_samples):
            if random_zernike:
                coeffs = list(np.random.normal(0, zernike_std, 15))
                # 第一项 (piston) 设为 0
                coeffs[0] = 0.0
            else:
                coeffs = self.config.zernike_coeffs or [0.0] * 15

            sample = self.generate(
                zernike_coeffs=coeffs,
                add_noise=True,
                random_augment=True,
            )
            samples.append(sample)

        return samples

    @staticmethod
    def _compute_psf_width(psf: np.ndarray) -> float:
        """计算 PSF 宽度"""
        h, w = psf.shape
        total = np.sum(psf)
        if total < 1e-10:
            return 0.0
        yy, xx = np.mgrid[:h, :w]
        cx = float(np.sum(xx * psf) / total)
        cy = float(np.sum(yy * psf) / total)
        var = float(np.sum(((xx - cx) ** 2 + (yy - cy) ** 2) * psf) / total)
        return float(np.sqrt(max(0, var)))

    @staticmethod
    def _affine_transform(image: np.ndarray, M: np.ndarray) -> np.ndarray:
        """应用仿射变换"""
        h, w = image.shape[:2]
        return cv2.warpAffine(
            image.astype(np.float32), M, (w, h),
            flags=cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )


def cv2_rotation_matrix(size: int, angle: float) -> np.ndarray:
    """生成旋转矩阵"""
    center = (size / 2.0, size / 2.0)
    return cv2.getRotationMatrix2D(center, angle, 1.0)


def cv2_scale_matrix(size: int, scale: float) -> np.ndarray:
    """生成缩放矩阵"""
    center = (size / 2.0, size / 2.0)
    return cv2.getRotationMatrix2D(center, 0, scale)
