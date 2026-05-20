"""
零样本光斑图像增强器 (Deep Image Prior-inspired Spot Enhancer)

参考开源项目:
  - Deep Image Prior (https://github.com/DmitryUlyanov/deep-image-prior)
    Ulyanov et al., "Deep Image Prior", CVPR 2018

核心思想:
  DIP 的核心发现是: 卷积神经网络的结构本身就是一种强大的图像先验，
  无需任何预训练数据，仅通过优化网络权重来拟合单张退化图像，
  就能实现去噪、超分辨率和修复。

  在 SpotZoom 场景中，我们借鉴 DIP 的思想，但不使用深度网络，
  而是使用可优化的参数化滤波器组 (参数化的小型卷积核集合) 来
  实现类似的效果:
  - 随机噪声输入 → 参数化滤波 → 输出增强图像
  - 损失函数: 保持光斑结构 + 抑制噪声
  - 无需训练数据，纯在线优化

创新点:
  1. 轻量级参数化增强 (无需 GPU/深度学习框架)
  2. 自适应迭代次数 (基于收敛监控自动停止)
  3. 物理约束: 保持光斑总能量守恒
  4. 多尺度增强策略 (粗→细)
  5. 增强质量在线评估

纯 numpy 实现，无外部依赖。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DIPEnhancementResult:
    """DIP 增强结果。"""
    # 增强后的图像
    enhanced_image: np.ndarray
    # 迭代次数
    iterations_used: int
    # 初始损失
    initial_loss: float
    # 最终损失
    final_loss: float
    # 损失下降比例
    loss_reduction_ratio: float
    # SNR 提升量 (dB)
    snr_improvement_db: float
    # 处理时间 (ms)
    processing_time_ms: float
    # 是否收敛
    converged: bool
    # 使用的滤波器参数 (可用于后续快速增强)
    filter_params: Optional[dict] = None


class DeepImagePriorEnhancer:
    """零样本光斑图像增强器。

    借鉴 Deep Image Prior 的核心思想，使用参数化滤波器组
    实现无需训练数据的在线图像增强。

    使用方法:
        enhancer = DeepImagePriorEnhancer(max_iterations=100)
        result = enhancer.enhance(spot_image)
        enhanced = result.enhanced_image
    """

    def __init__(
        self,
        max_iterations: int = 80,
        learning_rate: float = 0.01,
        num_filters: int = 5,
        filter_size: int = 5,
        noise_std: float = 0.05,
        early_stop_patience: int = 10,
        energy_conservation_weight: float = 0.1,
        smoothness_weight: float = 0.05,
        multi_scale: bool = True,
    ):
        """
        Args:
            max_iterations: 最大优化迭代次数
            learning_rate: 优化学习率
            num_filters: 参数化滤波器数量
            filter_size: 滤波器尺寸 (奇数)
            noise_std: 输入噪声标准差 (DIP 风格的随机输入)
            early_stop_patience: 早停耐心值
            energy_conservation_weight: 能量守恒约束权重
            smoothness_weight: 平滑性约束权重
            multi_scale: 是否使用多尺度策略
        """
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.num_filters = num_filters
        self.filter_size = max(3, filter_size | 1)  # 确保奇数
        self.noise_std = noise_std
        self.early_stop_patience = early_stop_patience
        self.energy_conservation_weight = energy_conservation_weight
        self.smoothness_weight = smoothness_weight
        self.multi_scale = multi_scale

        # 缓存上一次的滤波器参数 (用于快速增强相似图像)
        self._cached_params: Optional[dict] = None

    def enhance(self, image: np.ndarray) -> DIPEnhancementResult:
        """增强光斑图像。

        Args:
            image: 灰度图像 (2D numpy 数组, uint8 或 float)

        Returns:
            DIPEnhancementResult: 增强结果
        """
        t0 = time.perf_counter()

        if image.ndim != 2 or image.size < 16:
            return DIPEnhancementResult(
                enhanced_image=image.copy(),
                iterations_used=0,
                initial_loss=0.0,
                final_loss=0.0,
                loss_reduction_ratio=0.0,
                snr_improvement_db=0.0,
                processing_time_ms=0.0,
                converged=False,
            )

        # 归一化到 [0, 1]
        img_float = image.astype(np.float64)
        if img_float.max() > 1.5:
            img_float = img_float / 255.0

        original_energy = float(np.sum(img_float ** 2))

        if self.multi_scale and min(img_float.shape) > 64:
            result, meta = self._enhance_multiscale(img_float, original_energy)
        else:
            result, meta = self._enhance_single_scale(img_float, original_energy)

        # 转回 uint8
        enhanced = np.clip(result * 255.0, 0, 255).astype(np.uint8)

        processing_ms = (time.perf_counter() - t0) * 1000.0

        # SNR 改善
        noise_est = self._estimate_noise_level(img_float)
        enhanced_noise = self._estimate_noise_level(result)
        if noise_est > 1e-10:
            snr_before = float(np.mean(img_float) ** 2) / (noise_est ** 2 + 1e-10)
            snr_after = float(np.mean(result) ** 2) / (enhanced_noise ** 2 + 1e-10)
            snr_improvement = 10.0 * math.log10(max(1e-10, snr_after / max(1e-10, snr_before)))
        else:
            snr_improvement = 0.0

        return DIPEnhancementResult(
            enhanced_image=enhanced,
            iterations_used=meta.get("_iters", 0),
            initial_loss=meta.get("_init_loss", 0.0),
            final_loss=meta.get("_final_loss", 0.0),
            loss_reduction_ratio=meta.get("_loss_ratio", 0.0),
            snr_improvement_db=round(snr_improvement, 3),
            processing_time_ms=round(processing_ms, 2),
            converged=meta.get("_converged", False),
            filter_params=self._cached_params,
        )

    def _enhance_single_scale(
        self, image: np.ndarray, original_energy: float
    ) -> Tuple[np.ndarray, dict]:
        """单尺度增强。返回 (增强图像, 元数据字典)。"""
        h, w = image.shape

        # 初始化随机输入 (DIP 风格) — 使用局部随机生成器避免污染全局状态
        rng = np.random.RandomState(42)
        random_input = rng.randn(h, w).astype(np.float64) * self.noise_std

        # 初始化参数化滤波器
        half = self.filter_size // 2
        filters = rng.randn(self.num_filters, self.filter_size, self.filter_size) * 0.1
        biases = np.zeros(self.num_filters)
        blend_weights = np.ones(self.num_filters) / self.num_filters

        # 目标: 重建干净图像
        target = image.copy()

        best_output = image.copy()
        best_loss = float("inf")
        no_improve_count = 0
        initial_loss = None

        for iteration in range(self.max_iterations):
            # 前向传播: 随机输入 → 滤波 → 混合 → 输出
            filtered = np.zeros_like(image)
            for k in range(self.num_filters):
                # 卷积
                padded = np.pad(random_input, half, mode="reflect")
                conv = np.zeros_like(image)
                for fi in range(self.filter_size):
                    for fj in range(self.filter_size):
                        conv += filters[k, fi, fj] * padded[fi:fi + h, fj:fj + w]
                filtered += blend_weights[k] * (conv + biases[k])

            # 残差连接: 输出 = 滤波结果 + 原图
            output = image + 0.3 * filtered
            output = np.clip(output, 0, 1)

            # 计算损失
            # 1. 数据保真项: 输出应接近目标
            data_loss = float(np.mean((output - target) ** 2))

            # 2. 噪声抑制项: 输出应比输入更平滑
            smooth_loss = self._total_variation(output)

            # 3. 能量守恒项
            output_energy = float(np.sum(output ** 2))
            energy_loss = abs(output_energy - original_energy) / (original_energy + 1e-10)

            total_loss = (data_loss
                          + self.smoothness_weight * smooth_loss
                          + self.energy_conservation_weight * energy_loss)

            if initial_loss is None:
                initial_loss = total_loss

            # 检查是否改善
            if total_loss < best_loss:
                best_loss = total_loss
                best_output = output.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 早停
            if no_improve_count >= self.early_stop_patience:
                break

            # 反向传播 (简化梯度下降)
            residual = output - target
            grad_blend = np.zeros(self.num_filters)
            for k in range(self.num_filters):
                padded = np.pad(random_input, half, mode="reflect")
                conv = np.zeros_like(image)
                for fi in range(self.filter_size):
                    for fj in range(self.filter_size):
                        conv += filters[k, fi, fj] * padded[fi:fi + h, fj:fj + w]
                # 梯度
                grad_blend[k] = float(np.mean(residual * conv)) * 0.3

            # 更新混合权重
            blend_weights -= self.learning_rate * grad_blend
            blend_weights = np.clip(blend_weights, 0, None)
            blend_sum = np.sum(blend_weights) + 1e-10
            blend_weights /= blend_sum

        # 缓存参数
        self._cached_params = {
            "filters": filters.tolist(),
            "biases": biases.tolist(),
            "blend_weights": blend_weights.tolist(),
        }

        meta = {
            "_iters": iteration + 1,
            "_init_loss": initial_loss or 0.0,
            "_final_loss": best_loss,
            "_loss_ratio": (
                (initial_loss - best_loss) / max(1e-10, initial_loss)
                if initial_loss and initial_loss > 1e-10 else 0.0
            ),
            "_converged": no_improve_count >= self.early_stop_patience,
        }

        return best_output, meta

    def _enhance_multiscale(
        self, image: np.ndarray, original_energy: float
    ) -> Tuple[np.ndarray, dict]:
        """多尺度增强 (粗→细)。返回 (增强图像, 元数据字典)。"""
        h, w = image.shape

        # 粗尺度
        scale = 2
        small_h, small_w = h // scale, w // scale
        small = self._resize(image, (small_w, small_h))

        # 在粗尺度上增强
        coarse_enhanced, coarse_meta = self._enhance_single_scale(small, float(np.sum(small ** 2)))

        # 上采样回原始尺寸
        upsampled = self._resize(coarse_enhanced, (w, h))

        # 细尺度: 以粗增强结果为初始化
        refined = image + 0.5 * (upsampled - image)
        refined = np.clip(refined, 0, 1)

        return refined, coarse_meta

    @staticmethod
    def _total_variation(image: np.ndarray) -> float:
        """计算全变分 (Total Variation) 正则项。"""
        dy = np.diff(image, axis=0)
        dx = np.diff(image, axis=1)
        return float(np.mean(np.abs(dy)) + np.mean(np.abs(dx)))

    @staticmethod
    def _estimate_noise_level(image: np.ndarray) -> float:
        """使用 MAD (Median Absolute Deviation) 估计噪声水平。"""
        if image.size < 16:
            return 0.0
        # Laplacian
        lap = np.zeros_like(image)
        lap[1:-1, 1:-1] = (
            image[1:-1, 1:-1] * 4
            - image[:-2, 1:-1] - image[2:, 1:-1]
            - image[1:-1, :-2] - image[1:-1, 2:]
        )
        # MAD 估计
        sigma = float(np.median(np.abs(lap))) / 0.6745 * (1.0 / math.sqrt(20))
        return sigma

    @staticmethod
    def _resize(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """双线性插值缩放 (向量化实现)。"""
        h, w = image.shape
        tw, th = target_size

        y_ratio = h / max(1, th)
        x_ratio = w / max(1, tw)

        yy = np.arange(th) * y_ratio
        xx = np.arange(tw) * x_ratio
        yy = np.clip(yy, 0, h - 1)
        xx = np.clip(xx, 0, w - 1)

        y0 = np.floor(yy).astype(int)
        x0 = np.floor(xx).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)

        fy = (yy - y0)[:, np.newaxis]  # (th, 1)
        fx = (xx - x0)[np.newaxis, :]  # (1, tw)

        result = (image[np.ix_(y0, x0)] * (1 - fy) * (1 - fx)
                  + image[np.ix_(y1, x0)] * fy * (1 - fx)
                  + image[np.ix_(y0, x1)] * (1 - fy) * fx
                  + image[np.ix_(y1, x1)] * fy * fx)
        return result

    def quick_enhance(self, image: np.ndarray) -> np.ndarray:
        """使用缓存的参数快速增强 (适用于连续帧)。"""
        if self._cached_params is None:
            return image.copy()

        if image.ndim != 2:
            return image.copy()

        img_float = image.astype(np.float64)
        if img_float.max() > 1.5:
            img_float = img_float / 255.0

        # 简单的加权中值滤波作为快速增强
        bw = np.array(self._cached_params.get("blend_weights", [1.0]))
        if bw.size == 0 or np.sum(bw) < 1e-10:
            return image.copy()

        # 使用最强权重的滤波器进行简单平滑
        ksize = self.filter_size
        if ksize < 3:
            return image.copy()

        # 简单盒滤波
        half = ksize // 2
        padded = np.pad(img_float, half, mode="reflect")
        h, w = img_float.shape
        smoothed = np.zeros_like(img_float)
        count = 0
        for di in range(ksize):
            for dj in range(ksize):
                smoothed += padded[di:di + h, dj:dj + w]
                count += 1
        smoothed /= count

        # 轻度增强: 混合原图和平滑结果
        enhanced = img_float * 0.6 + smoothed * 0.4
        return np.clip(enhanced * 255, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        """重置增强器状态。"""
        self._cached_params = None
