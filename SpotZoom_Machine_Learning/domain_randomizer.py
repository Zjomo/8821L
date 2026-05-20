"""
仿真域随机化器 (DomainRandomizer)

灵感来源:
- Domain Randomization (Tobin et al. 2017) — 通过在仿真环境中注入随机扰动，
  使模型在训练时暴露于多样化的视觉条件，从而提升 Sim-to-Real 迁移能力
- NVIDIA Domain Randomization for Robotic Grasping — 针对机器人抓取任务的多维
  域随机化策略 (纹理、光照、相机噪声、几何变换)
- AutoAugment (Zoph et al. 2019) — 自动学习数据增强策略
- Cutout / Random Erasing — 随机擦除增强鲁棒性

算法原理:
- Gaussian Noise Injection — 高斯噪声注入，模拟传感器噪声
- Gaussian Blur — 高斯模糊，模拟散焦和运动模糊
- Brightness / Contrast Jitter — 亮度/对比度抖动，模拟光照变化
- Geometric Transformation — 几何变换 (旋转、缩放、平移)，模拟视角变化
- Uniform / Gaussian / Truncated Normal Distribution — 多种分布支持

功能:
- 为仿真图像注入随机噪声
- 模糊随机化 (高斯模糊)
- 亮度/对比度随机化
- 几何变换随机化 (旋转、缩放、平移)
- 对参数字典应用随机扰动
- 批量生成增强数据集
- 随机化统计与分析

依赖: numpy, cv2, logging, random
"""

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("SpotZoom.DomainRandomizer")


# ======================== 枚举与数据类 ========================


class DistributionType(Enum):
    """随机分布类型。"""
    UNIFORM = "uniform"                   # 均匀分布
    GAUSSIAN = "gaussian"                 # 高斯分布
    TRUNCATED_GAUSSIAN = "truncated_gaussian"  # 截断高斯分布


class RandomizationType(Enum):
    """随机化类型。"""
    NOISE = "noise"           # 噪声
    BLUR = "blur"             # 模糊
    BRIGHTNESS = "brightness"  # 亮度
    GEOMETRIC = "geometric"    # 几何变换


@dataclass
class RandomizationRange:
    """随机化范围定义。

    Parameters
    ----------
    min_value : float
        最小值。
    max_value : float
        最大值。
    distribution : DistributionType
        分布类型。
    mean : float or None
        高斯分布均值 (仅 GAUSSIAN / TRUNCATED_GAUSSIAN)。
    std : float or None
        高斯分布标准差 (仅 GAUSSIAN / TRUNCATED_GAUSSIAN)。
    """
    min_value: float
    max_value: float
    distribution: DistributionType = DistributionType.UNIFORM
    mean: Optional[float] = None
    std: Optional[float] = None

    def sample(self) -> float:
        """从定义的范围中采样一个值。"""
        if self.distribution == DistributionType.UNIFORM:
            return random.uniform(self.min_value, self.max_value)
        elif self.distribution == DistributionType.GAUSSIAN:
            mean = self.mean if self.mean is not None else (self.min_value + self.max_value) / 2.0
            std = self.std if self.std is not None else (self.max_value - self.min_value) / 4.0
            return float(np.clip(random.gauss(mean, std), self.min_value, self.max_value))
        elif self.distribution == DistributionType.TRUNCATED_GAUSSIAN:
            mean = self.mean if self.mean is not None else (self.min_value + self.max_value) / 2.0
            std = self.std if self.std is not None else (self.max_value - self.min_value) / 6.0
            # 截断高斯: 采样直到落在范围内
            for _ in range(100):
                val = random.gauss(mean, std)
                if self.min_value <= val <= self.max_value:
                    return val
            return random.uniform(self.min_value, self.max_value)
        else:
            return random.uniform(self.min_value, self.max_value)


@dataclass
class RandomizationConfig:
    """随机化全局配置。

    Parameters
    ----------
    seed : int or None
        随机种子。为 None 时不设置。
    apply_probabilities : Dict[str, float]
        各类随机化的应用概率。
    preserve_aspect_ratio : bool
        几何变换时是否保持宽高比。
    interpolation : int
        OpenCV 插值方法。
    """
    seed: Optional[int] = None
    apply_probabilities: Dict[str, float] = field(default_factory=lambda: {
        "noise": 0.8,
        "blur": 0.5,
        "brightness": 0.7,
        "geometric": 0.6,
    })
    preserve_aspect_ratio: bool = True
    interpolation: int = cv2.INTER_LINEAR


@dataclass
class AugmentationRecord:
    """单次增强记录。"""
    image_index: int              # 原始图像索引
    augmentation_index: int       # 增强编号
    applied_transforms: List[str]  # 应用的变换列表
    transform_params: Dict[str, float]  # 变换参数
    elapsed_time_ms: float        # 耗时 (毫秒)


@dataclass
class RandomizationStats:
    """随机化统计信息。"""
    total_images_processed: int = 0
    total_augmentations: int = 0
    noise_applied_count: int = 0
    blur_applied_count: int = 0
    brightness_applied_count: int = 0
    geometric_applied_count: int = 0
    avg_noise_std: float = 0.0
    avg_blur_kernel: float = 0.0
    avg_brightness_delta: float = 0.0
    avg_rotation_deg: float = 0.0
    avg_scale_factor: float = 0.0
    total_time_s: float = 0.0


# ======================== 域随机化器 ========================


class DomainRandomizer:
    """仿真域随机化器。

    为仿真环境注入随机扰动 (噪声、模糊、亮度变化、几何变换)，
    增强模型的 Sim-to-Real 迁移鲁棒性。

    Parameters
    ----------
    config : RandomizationConfig or None
        全局配置。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[RandomizationConfig] = None):
        self.config = config or RandomizationConfig()

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        # 随机化注册表
        self._noise_ranges: Dict[str, RandomizationRange] = {}
        self._blur_ranges: Dict[str, RandomizationRange] = {}
        self._brightness_ranges: Dict[str, RandomizationRange] = {}
        self._geometric_ranges: Dict[str, RandomizationRange] = {}

        # 统计
        self._stats = RandomizationStats()
        self._noise_stds: List[float] = []
        self._blur_kernels: List[float] = []
        self._brightness_deltas: List[float] = []
        self._rotation_degs: List[float] = []
        self._scale_factors: List[float] = []

        LOGGER.info(
            "DomainRandomizer: 初始化完成 (seed=%s)",
            self.config.seed,
        )

    # ======================== 注册随机化 ========================

    def add_noise_randomization(
        self,
        key: str,
        range_def: RandomizationRange,
    ) -> None:
        """添加噪声随机化。

        Parameters
        ----------
        key : str
            随机化键名。
        range_def : RandomizationRange
            噪声标准差范围 (高斯噪声的 sigma)。
        """
        self._noise_ranges[key] = range_def
        LOGGER.info(
            "DomainRandomizer: 添加噪声随机化 '%s', 范围=[%.2f, %.2f], 分布=%s",
            key, range_def.min_value, range_def.max_value, range_def.distribution.value,
        )

    def add_blur_randomization(
        self,
        key: str,
        range_def: RandomizationRange,
    ) -> None:
        """添加模糊随机化。

        Parameters
        ----------
        key : str
            随机化键名。
        range_def : RandomizationRange
            高斯模糊核大小范围。
        """
        self._blur_ranges[key] = range_def
        LOGGER.info(
            "DomainRandomizer: 添加模糊随机化 '%s', 范围=[%.1f, %.1f], 分布=%s",
            key, range_def.min_value, range_def.max_value, range_def.distribution.value,
        )

    def add_brightness_randomization(
        self,
        key: str,
        range_def: RandomizationRange,
    ) -> None:
        """添加亮度随机化。

        Parameters
        ----------
        key : str
            随机化键名。
        range_def : RandomizationRange
            亮度偏移范围 (像素值)。
        """
        self._brightness_ranges[key] = range_def
        LOGGER.info(
            "DomainRandomizer: 添加亮度随机化 '%s', 范围=[%.1f, %.1f], 分布=%s",
            key, range_def.min_value, range_def.max_value, range_def.distribution.value,
        )

    def add_geometric_randomization(
        self,
        key: str,
        range_def: RandomizationRange,
    ) -> None:
        """添加几何变换随机化。

        Parameters
        ----------
        key : str
            随机化键名 (支持: 'rotation', 'scale', 'translate_x', 'translate_y')。
        range_def : RandomizationRange
            变换参数范围。
            - rotation: 角度 (度)
            - scale: 缩放因子
            - translate_x/y: 平移像素数
        """
        self._geometric_ranges[key] = range_def
        LOGGER.info(
            "DomainRandomizer: 添加几何随机化 '%s', 范围=[%.2f, %.2f], 分布=%s",
            key, range_def.min_value, range_def.max_value, range_def.distribution.value,
        )

    # ======================== 图像随机化 ========================

    def randomize_image(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """对图像应用所有已注册的随机化。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (BGR 或灰度)。

        Returns
        -------
        Tuple[np.ndarray, Dict[str, float]]
            (随机化后的图像, 使用的参数字典)。
        """
        t0 = time.perf_counter()
        result = image.copy()
        params: Dict[str, float] = {}

        # 1. 噪声
        if self._noise_ranges and self._should_apply("noise"):
            result, noise_params = self._apply_noise(result)
            params.update(noise_params)
            self._stats.noise_applied_count += 1

        # 2. 模糊
        if self._blur_ranges and self._should_apply("blur"):
            result, blur_params = self._apply_blur(result)
            params.update(blur_params)
            self._stats.blur_applied_count += 1

        # 3. 亮度
        if self._brightness_ranges and self._should_apply("brightness"):
            result, bright_params = self._apply_brightness(result)
            params.update(bright_params)
            self._stats.brightness_applied_count += 1

        # 4. 几何变换
        if self._geometric_ranges and self._should_apply("geometric"):
            result, geo_params = self._apply_geometric(result)
            params.update(geo_params)
            self._stats.geometric_applied_count += 1

        self._stats.total_images_processed += 1
        self._stats.total_time_s += time.perf_counter() - t0

        return result, params

    def randomize_parameters(
        self,
        params_dict: Dict[str, float],
    ) -> Dict[str, float]:
        """对参数字典应用随机扰动。

        为每个参数添加一个小的随机偏移量。

        Parameters
        ----------
        params_dict : Dict[str, float]
            原始参数字典。

        Returns
        -------
        Dict[str, float]
            扰动后的参数字典。
        """
        result = {}
        for key, value in params_dict.items():
            # 使用噪声范围中的第一个定义来确定扰动幅度
            if self._noise_ranges:
                range_def = next(iter(self._noise_ranges.values()))
                perturbation = range_def.sample()
                result[key] = value + perturbation * 0.1  # 缩小扰动幅度
            else:
                result[key] = value

        return result

    def generate_augmented_dataset(
        self,
        images: List[np.ndarray],
        count_per_image: int = 5,
    ) -> Tuple[List[np.ndarray], List[AugmentationRecord]]:
        """批量生成增强数据集。

        Parameters
        ----------
        images : List[np.ndarray]
            原始图像列表。
        count_per_image : int
            每张原始图像生成的增强数量。

        Returns
        -------
        Tuple[List[np.ndarray], List[AugmentationRecord]]
            (增强后的图像列表, 增强记录列表)。
        """
        augmented_images: List[np.ndarray] = []
        records: List[AugmentationRecord] = []

        LOGGER.info(
            "DomainRandomizer: 开始批量增强 (原始图像=%d, 每图增强=%d)",
            len(images), count_per_image,
        )

        for img_idx, image in enumerate(images):
            for aug_idx in range(count_per_image):
                t0 = time.perf_counter()
                aug_image, transform_params = self.randomize_image(image)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                augmented_images.append(aug_image)

                applied_transforms = list(transform_params.keys())
                record = AugmentationRecord(
                    image_index=img_idx,
                    augmentation_index=aug_idx,
                    applied_transforms=applied_transforms,
                    transform_params=transform_params,
                    elapsed_time_ms=round(elapsed_ms, 2),
                )
                records.append(record)

                self._stats.total_augmentations += 1

        LOGGER.info(
            "DomainRandomizer: 批量增强完成 (总增强图像=%d, 平均耗时=%.1fms/张)",
            len(augmented_images),
            np.mean([r.elapsed_time_ms for r in records]) if records else 0,
        )

        return augmented_images, records

    def get_randomization_stats(self) -> RandomizationStats:
        """获取随机化统计。

        Returns
        -------
        RandomizationStats
            统计信息。
        """
        stats = RandomizationStats(
            total_images_processed=self._stats.total_images_processed,
            total_augmentations=self._stats.total_augmentations,
            noise_applied_count=self._stats.noise_applied_count,
            blur_applied_count=self._stats.blur_applied_count,
            brightness_applied_count=self._stats.brightness_applied_count,
            geometric_applied_count=self._stats.geometric_applied_count,
            avg_noise_std=float(np.mean(self._noise_stds)) if self._noise_stds else 0.0,
            avg_blur_kernel=float(np.mean(self._blur_kernels)) if self._blur_kernels else 0.0,
            avg_brightness_delta=float(np.mean(self._brightness_deltas)) if self._brightness_deltas else 0.0,
            avg_rotation_deg=float(np.mean(self._rotation_degs)) if self._rotation_degs else 0.0,
            avg_scale_factor=float(np.mean(self._scale_factors)) if self._scale_factors else 0.0,
            total_time_s=round(self._stats.total_time_s, 3),
        )
        return stats

    def reset(self) -> None:
        """重置随机化器和统计。"""
        self._noise_ranges.clear()
        self._blur_ranges.clear()
        self._brightness_ranges.clear()
        self._geometric_ranges.clear()
        self._stats = RandomizationStats()
        self._noise_stds.clear()
        self._blur_kernels.clear()
        self._brightness_deltas.clear()
        self._rotation_degs.clear()
        self._scale_factors.clear()
        LOGGER.info("DomainRandomizer: 已重置")

    # ======================== 内部方法 ========================

    def _should_apply(self, randomization_type: str) -> bool:
        """根据概率决定是否应用某类随机化。"""
        prob = self.config.apply_probabilities.get(randomization_type, 0.5)
        return random.random() < prob

    def _apply_noise(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """应用高斯噪声。"""
        params: Dict[str, float] = {}

        for key, range_def in self._noise_ranges.items():
            sigma = range_def.sample()
            noise = np.random.normal(0, sigma, image.shape).astype(np.float32)

            if image.dtype == np.uint8:
                noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            else:
                noisy = image.astype(np.float32) + noise

            image = noisy
            params[f"noise_{key}_sigma"] = round(sigma, 4)
            self._noise_stds.append(sigma)

        return image, params

    def _apply_blur(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """应用高斯模糊。"""
        params: Dict[str, float] = {}

        for key, range_def in self._blur_ranges.items():
            kernel_size = int(range_def.sample())
            # 确保核大小为奇数且 >= 1
            kernel_size = max(1, kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1

            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            image = blurred
            params[f"blur_{key}_kernel"] = float(kernel_size)
            self._blur_kernels.append(float(kernel_size))

        return image, params

    def _apply_brightness(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """应用亮度/对比度变化。"""
        params: Dict[str, float] = {}

        for key, range_def in self._brightness_ranges.items():
            delta = range_def.sample()

            if image.dtype == np.uint8:
                adjusted = np.clip(
                    image.astype(np.float32) + delta, 0, 255
                ).astype(np.uint8)
            else:
                adjusted = image.astype(np.float32) + delta

            image = adjusted
            params[f"brightness_{key}_delta"] = round(delta, 4)
            self._brightness_deltas.append(delta)

        return image, params

    def _apply_geometric(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """应用几何变换 (旋转、缩放、平移)。"""
        params: Dict[str, float] = {}
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        # 旋转
        if "rotation" in self._geometric_ranges:
            angle = self._geometric_ranges["rotation"].sample()
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h),
                flags=self.config.interpolation,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            params["rotation_deg"] = round(angle, 2)
            self._rotation_degs.append(angle)

        # 缩放
        if "scale" in self._geometric_ranges:
            scale = self._geometric_ranges["scale"].sample()
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            image = cv2.resize(
                image, (new_w, new_h),
                interpolation=self.config.interpolation,
            )
            # 如果尺寸变化了，填充/裁剪回原始尺寸
            if new_w != w or new_h != h:
                if len(image.shape) == 2:
                    canvas = np.zeros((h, w), dtype=image.dtype)
                else:
                    canvas = np.zeros((h, w, image.shape[2]), dtype=image.dtype)

                # 计算居中偏移 (允许负值表示裁剪)
                y_off = (h - new_h) // 2
                x_off = (w - new_w) // 2

                # 源图像和目标画布的有效区域
                src_y_start = max(0, -y_off)
                src_x_start = max(0, -x_off)
                dst_y_start = max(0, y_off)
                dst_x_start = max(0, x_off)

                copy_h = min(new_h - src_y_start, h - dst_y_start)
                copy_w = min(new_w - src_x_start, w - dst_x_start)

                if copy_h > 0 and copy_w > 0:
                    canvas[dst_y_start:dst_y_start + copy_h,
                           dst_x_start:dst_x_start + copy_w] = \
                        image[src_y_start:src_y_start + copy_h,
                              src_x_start:src_x_start + copy_w]

                image = canvas

            params["scale_factor"] = round(scale, 4)
            self._scale_factors.append(scale)

        # 平移 X
        if "translate_x" in self._geometric_ranges:
            tx = int(self._geometric_ranges["translate_x"].sample())
            M = np.float32([[1, 0, tx], [0, 1, 0]])
            image = cv2.warpAffine(
                image, M, (w, h),
                flags=self.config.interpolation,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            params["translate_x"] = float(tx)

        # 平移 Y
        if "translate_y" in self._geometric_ranges:
            ty = int(self._geometric_ranges["translate_y"].sample())
            M = np.float32([[1, 0, 0], [0, 1, ty]])
            image = cv2.warpAffine(
                image, M, (w, h),
                flags=self.config.interpolation,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            params["translate_y"] = float(ty)

        return image, params


# ======================== 测试入口 ========================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=== 仿真域随机化器测试 ===\n")

    np.random.seed(42)
    random.seed(42)

    # 创建测试图像 (100x100 灰度渐变 + 光斑)
    test_image = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            dist = np.sqrt((i - 50) ** 2 + (j - 50) ** 2)
            test_image[i, j] = int(max(0, 255 - dist * 3))

    # 创建随机化器
    randomizer = DomainRandomizer(RandomizationConfig(seed=42))

    # 注册各类随机化
    randomizer.add_noise_randomization(
        "gaussian",
        RandomizationRange(0.0, 30.0, DistributionType.GAUSSIAN, mean=10.0, std=8.0),
    )
    randomizer.add_blur_randomization(
        "gaussian",
        RandomizationRange(1, 7, DistributionType.UNIFORM),
    )
    randomizer.add_brightness_randomization(
        "offset",
        RandomizationRange(-50.0, 50.0, DistributionType.UNIFORM),
    )
    randomizer.add_geometric_randomization(
        "rotation",
        RandomizationRange(-15.0, 15.0, DistributionType.UNIFORM),
    )
    randomizer.add_geometric_randomization(
        "scale",
        RandomizationRange(0.8, 1.2, DistributionType.GAUSSIAN, mean=1.0, std=0.1),
    )
    randomizer.add_geometric_randomization(
        "translate_x",
        RandomizationRange(-10.0, 10.0, DistributionType.UNIFORM),
    )
    randomizer.add_geometric_randomization(
        "translate_y",
        RandomizationRange(-10.0, 10.0, DistributionType.UNIFORM),
    )

    # 测试单张图像随机化
    print("--- 单张图像随机化 ---")
    for i in range(5):
        result, params = randomizer.randomize_image(test_image)
        print(f"  随机化 {i+1}: 变换参数 = {params}")
        print(f"    输出形状: {result.shape}, dtype: {result.dtype}")

    # 测试参数随机化
    print("\n--- 参数随机化 ---")
    original_params = {"focal_length": 50.0, "exposure": 1.0, "gain": 2.0}
    for i in range(3):
        perturbed = randomizer.randomize_parameters(original_params)
        print(f"  原始: {original_params}")
        print(f"  扰动: {perturbed}")
        print()

    # 测试批量增强
    print("--- 批量增强 ---")
    images = [test_image] * 3
    augmented, records = randomizer.generate_augmented_dataset(images, count_per_image=4)
    print(f"  原始图像数: {len(images)}")
    print(f"  增强后总数: {len(augmented)}")
    print(f"  记录数: {len(records)}")

    # 统计信息
    print("\n--- 随机化统计 ---")
    stats = randomizer.get_randomization_stats()
    print(f"  处理图像总数: {stats.total_images_processed}")
    print(f"  增强总数: {stats.total_augmentations}")
    print(f"  噪声应用次数: {stats.noise_applied_count}")
    print(f"  模糊应用次数: {stats.blur_applied_count}")
    print(f"  亮度应用次数: {stats.brightness_applied_count}")
    print(f"  几何变换应用次数: {stats.geometric_applied_count}")
    print(f"  平均噪声标准差: {stats.avg_noise_std:.2f}")
    print(f"  平均模糊核大小: {stats.avg_blur_kernel:.2f}")
    print(f"  平均亮度偏移: {stats.avg_brightness_delta:.2f}")
    print(f"  平均旋转角度: {stats.avg_rotation_deg:.2f}")
    print(f"  平均缩放因子: {stats.avg_scale_factor:.4f}")
    print(f"  总耗时: {stats.total_time_s:.3f}s")

    print("\n测试完成")
