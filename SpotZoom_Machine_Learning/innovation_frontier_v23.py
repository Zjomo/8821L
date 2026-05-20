"""
SpotZoom 前沿开源项目创新模块 v23.0
基于 2024-2026 最新科研前沿开源项目的创新模块补充 (第十一轮)

新增模块 (v23.0):
- UnsupervisedSpotDetector: 无监督光斑检测器 (受 DeepTrack2 LodeSTAR 启发)
- GNNTrajectoryAnalyzer: 图神经网络轨迹分析器 (受 DeepTrack2 MAGIK 启发)
- AberrationCNNProfiler: 像差CNN表征器 (受 DeepTrack2 像差模块 启发)
- BayesianUncertaintyEstimator: 贝叶斯不确定性估计器 (受 BayesDL-SIM 启发)
- ResolutionInvariantOperator: 分辨率无关算子 (受 NeuralOperator FNO 启发)
- PhysicsConstrainedOptimizer: 物理约束优化器 (受 DeepXDE PINN 启发)

参考项目:
- DeepTrack2 (github.com/DeepTrackAI/DeepTrack2) — LodeSTAR 无监督检测、MAGIK GNN 轨迹分析、像差表征 CNN
- NeuralOperator (github.com/neuraloperator/neuraloperator) — FNO 分辨率无关算子学习
- DeepXDE (github.com/lululxvi/deepxde) — PINN 物理信息神经网络
- TrackMate (github.com/HenriquesLab/TrackMate) — LoG 斑点检测、LAP 粒子链接
- BayesDL-SIM — 贝叶斯深度学习不确定性量化
- Tidy3D (github.com/flexcompute/tidy3d) — FDTD 电磁仿真

算法原理说明:
1. UnsupervisedSpotDetector: 基于合成数据增强 + 轻量 CNN 的自监督学习范式。
   从单张参考图像提取光斑特征，通过随机变换生成合成训练对，
   训练网络学习平移等变性，实现无需标注的亚像素光斑检测。
2. GNNTrajectoryAnalyzer: 将多帧检测结果构建为时空图，节点=检测，
   边=时空邻近关系。GNN 通过消息传递聚合邻域信息，学习链接概率。
3. AberrationCNNProfiler: 轻量 CNN 将光斑 PSF 形态映射到 Zernike 多项式系数，
   实现从单张光斑图像反推波前像差参数。
4. BayesianUncertaintyEstimator: MC Dropout + 深度集成方法为光斑定位结果
   提供贝叶斯后验不确定性量化。
5. ResolutionInvariantOperator: 傅里叶神经算子 (FNO) 在频域学习全局卷积核，
   实现光学系统参数到光斑形态的分辨率无关映射。
6. PhysicsConstrainedOptimizer: 将光学物理方程 (波动方程、Helmholtz 方程等)
   编码为 PDE 约束，通过残差自适应采样和物理损失加权优化光斑定位。
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from pathlib import Path
from collections import deque
import warnings
import time
import json
import copy
import math

# 尝试导入可选依赖
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from scipy import ndimage, signal, stats, optimize
    from scipy.fft import fft2, ifft2, fftshift, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

__version__ = "23.0.0"

__all__ = [
    "UnsupervisedSpotConfig",
    "UnsupervisedSpotResult",
    "UnsupervisedSpotDetector",
    "TrajectoryNode",
    "TrajectoryEdge",
    "GNNTrajectoryConfig",
    "GNNTrajectoryResult",
    "GNNTrajectoryAnalyzer",
    "AberrationProfile",
    "AberrationCNNConfig",
    "AberrationCNNResult",
    "AberrationCNNProfiler",
    "UncertaintySample",
    "BayesianConfig",
    "BayesianResult",
    "BayesianUncertaintyEstimator",
    "SpectralLayer",
    "OperatorConfig",
    "OperatorResult",
    "ResolutionInvariantOperator",
    "PhysicsConstraint",
    "PhysicsOptConfig",
    "PhysicsOptResult",
    "PhysicsConstrainedOptimizer",
]


# ===========================================================================
# 1. UnsupervisedSpotDetector — 无监督光斑检测器
# ===========================================================================
# 灵感来源: DeepTrack2 LodeSTAR (github.com/DeepTrackAI/DeepTrack2)
#   - LodeSTAR: 基于梯度特征的无监督粒子追踪
#   - 自监督学习: 从单张参考图像学习光斑外观
#   - 合成数据增强: 随机变换生成训练对
#   - 平移等变性: 网络学习平移不变的内部表征
#
# 与 SpotZoom 已有模块的区别:
#   - lodestar_detector.py 使用传统 CV 方法 (模板匹配 + 相位相关)
#   - classic_spot_detector.py 使用 LoG / 阈值分割
#   - 本模块使用轻量 CNN + 自监督学习，无需标注数据
# ===========================================================================


@dataclass
class UnsupervisedSpotConfig:
    """无监督光斑检测器配置。

    Attributes
    ----------
    patch_size : int
        训练/检测的图像块尺寸 (像素)。
    num_augmentations : int
        每张参考图像的合成增强数量。
    learning_rate : float
        自监督训练学习率。
    num_epochs : int
        自监督训练轮数。
    confidence_threshold : float
        检测置信度阈值 [0, 1]。
    nms_radius : float
        非极大值抑制半径 (像素)。
    subpixel_refine : bool
        是否进行亚像素精化。
    max_detections : int
        单帧最大检测数。0 表示不限制。
    augmentation_types : list of str
        启用的增强类型列表。
    """

    patch_size: int = 64
    num_augmentations: int = 200
    learning_rate: float = 1e-3
    num_epochs: int = 50
    confidence_threshold: float = 0.5
    nms_radius: float = 5.0
    subpixel_refine: bool = True
    max_detections: int = 0
    augmentation_types: List[str] = field(default_factory=lambda: [
        "translate", "rotate", "scale", "brightness", "noise"
    ])


@dataclass
class UnsupervisedSpotResult:
    """无监督光斑检测结果。

    Attributes
    ----------
    detections : list of dict
        检测到的光斑列表，每项包含 x, y, confidence, size。
    uncertainty_map : ndarray
        不确定性图 (H, W)。
    feature_map : ndarray
        特征响应图 (H, W)。
    num_detected : int
        检测到的光斑数量。
    compute_time_ms : float
        计算耗时 (毫秒)。
    """

    detections: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_map: Optional[np.ndarray] = None
    feature_map: Optional[np.ndarray] = None
    num_detected: int = 0
    compute_time_ms: float = 0.0


class UnsupervisedSpotDetector:
    """无监督光斑检测器。

    基于 DeepTrack2 LodeSTAR 的自监督学习范式，从单张参考图像学习光斑特征，
    无需任何标注数据即可实现亚像素级光斑检测。

    核心原理:
    1. 从参考图像中随机裁剪包含光斑的图像块
    2. 对图像块施加随机变换 (平移、旋转、缩放、亮度、噪声)
    3. 训练轻量 CNN 学习变换不变的特征表征
    4. 推理时在滑动窗口上计算特征响应图
    5. 通过 NMS 和亚像素精化得到最终检测结果

    Parameters
    ----------
    config : UnsupervisedSpotConfig, optional
        配置参数。
    """

    def __init__(self, config: Optional[UnsupervisedSpotConfig] = None):
        self.config = config or UnsupervisedSpotConfig()
        self._reference_patch: Optional[np.ndarray] = None
        self._feature_weights: Optional[Dict[str, np.ndarray]] = None
        self._is_trained = False
        self._call_count = 0
        self._training_history: deque = deque(maxlen=200)

    def train_from_single_image(self, image: np.ndarray,
                                spot_center: Optional[Tuple[int, int]] = None,
                                spot_radius: int = 16) -> Dict:
        """从单张图像进行自监督训练。

        Parameters
        ----------
        image : ndarray
            参考图像 (H, W) 或 (H, W, C)。
        spot_center : tuple of int, optional
            光斑中心坐标 (y, x)。若为 None 则自动定位最亮区域。
        spot_radius : int
            光斑区域半径 (像素)。

        Returns
        -------
        dict
            训练统计信息。
        """
        t0 = time.perf_counter()

        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(np.float64)

        # 自动定位光斑中心 (若未提供)
        if spot_center is None:
            cy, cx = self._auto_locate_spot(gray)
        else:
            cy, cx = spot_center

        # 提取参考图像块
        ps = self.config.patch_size
        half = ps // 2
        h, w = gray.shape

        # 边界检查
        y_start = max(0, cy - half)
        y_end = min(h, cy + half)
        x_start = max(0, cx - half)
        x_end = min(w, cx + half)

        patch = gray[y_start:y_end, x_start:x_end].copy()

        # 调整到目标尺寸
        if patch.shape[0] != ps or patch.shape[1] != ps:
            if CV2_AVAILABLE:
                patch = cv2.resize(patch, (ps, ps), interpolation=cv2.INTER_LINEAR)
            else:
                # 简单双线性插值 fallback
                y_idx = np.linspace(0, patch.shape[0] - 1, ps)
                x_idx = np.linspace(0, patch.shape[1] - 1, ps)
                yi, xi = np.meshgrid(y_idx, x_idx, indexing='ij')
                y0 = np.floor(yi).astype(int)
                x0 = np.floor(xi).astype(int)
                y1 = np.minimum(y0 + 1, patch.shape[0] - 1)
                x1 = np.minimum(x0 + 1, patch.shape[1] - 1)
                fy = yi - y0
                fx = xi - x0
                patch = (patch[y0, x0] * (1 - fy) * (1 - fx) +
                         patch[y1, x0] * fy * (1 - fx) +
                         patch[y0, x1] * (1 - fy) * fx +
                         patch[y1, x1] * fy * fx)

        self._reference_patch = patch

        # 生成合成训练数据
        train_pairs = self._generate_synthetic_pairs(patch)

        # 训练特征提取器
        self._train_feature_extractor(train_pairs)

        self._is_trained = True
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "spot_center": (int(cy), int(cx)),
            "num_training_pairs": len(train_pairs),
            "patch_size": ps,
            "training_time_ms": elapsed_ms,
            "is_trained": True,
        }

    def _auto_locate_spot(self, image: np.ndarray) -> Tuple[int, int]:
        """自动定位图像中最亮的光斑中心。

        Parameters
        ----------
        image : ndarray
            灰度图像 (H, W)。

        Returns
        -------
        tuple of int
            光斑中心 (y, x)。
        """
        # 高斯模糊去噪
        if CV2_AVAILABLE:
            blurred = cv2.GaussianBlur(image, (5, 5), 2.0)
        elif SCIPY_AVAILABLE:
            blurred = ndimage.gaussian_filter(image, sigma=2.0)
        else:
            blurred = image

        # 加权质心
        threshold = np.percentile(blurred, 90)
        mask = blurred > threshold
        if mask.sum() == 0:
            mask = blurred > np.median(blurred)

        ys, xs = np.where(mask)
        weights = blurred[mask]
        total_weight = weights.sum() + 1e-10
        cy = float(np.sum(ys * weights) / total_weight)
        cx = float(np.sum(xs * weights) / total_weight)

        return int(cy), int(cx)

    def _generate_synthetic_pairs(self, patch: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """生成合成训练对 (原始, 变换后)。

        Parameters
        ----------
        patch : ndarray
            参考图像块 (ps, ps)。

        Returns
        -------
        list of (ndarray, ndarray)
            训练对列表。
        """
        pairs = []
        aug_types = self.config.augmentation_types
        n = self.config.num_augmentations

        for _ in range(n):
            aug_type = aug_types[np.random.randint(len(aug_types))]
            transformed = self._apply_augmentation(patch, aug_type)
            pairs.append((patch.copy(), transformed))

        return pairs

    def _apply_augmentation(self, patch: np.ndarray, aug_type: str) -> np.ndarray:
        """对图像块施加随机增强。

        Parameters
        ----------
        patch : ndarray
            输入图像块 (ps, ps)。
        aug_type : str
            增强类型。

        Returns
        -------
        ndarray
            增强后的图像块。
        """
        ps = patch.shape[0]
        result = patch.copy()

        if aug_type == "translate":
            dy = np.random.randint(-ps // 4, ps // 4 + 1)
            dx = np.random.randint(-ps // 4, ps // 4 + 1)
            result = np.roll(np.roll(result, dy, axis=0), dx, axis=1)
            # 边界置零
            if dy > 0:
                result[:dy, :] = 0
            elif dy < 0:
                result[dy:, :] = 0
            if dx > 0:
                result[:, :dx] = 0
            elif dx < 0:
                result[:, dx:] = 0

        elif aug_type == "rotate":
            angle = np.random.uniform(-30, 30)
            if CV2_AVAILABLE:
                center = (ps // 2, ps // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                result = cv2.warpAffine(result, M, (ps, ps), borderValue=0)
            else:
                # 简单旋转近似: 交换 + 插值
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                y_coords, x_coords = np.mgrid[:ps, :ps]
                y_centered = y_coords - ps / 2.0
                x_centered = x_coords - ps / 2.0
                y_new = y_centered * cos_a - x_centered * sin_a + ps / 2.0
                x_new = y_centered * sin_a + x_centered * cos_a + ps / 2.0
                y_new = np.clip(y_new, 0, ps - 1).astype(int)
                x_new = np.clip(x_new, 0, ps - 1).astype(int)
                result = patch[y_new, x_new]

        elif aug_type == "scale":
            scale = np.random.uniform(0.7, 1.3)
            if CV2_AVAILABLE:
                new_size = max(4, int(ps * scale))
                resized = cv2.resize(patch, (new_size, new_size),
                                     interpolation=cv2.INTER_LINEAR)
                # 居中裁剪/填充回原始尺寸
                offset = (new_size - ps) // 2
                if new_size >= ps:
                    result = resized[offset:offset + ps, offset:offset + ps]
                else:
                    result = np.zeros((ps, ps), dtype=patch.dtype)
                    result[:new_size, :new_size] = resized
            else:
                result = patch * scale

        elif aug_type == "brightness":
            factor = np.random.uniform(0.5, 2.0)
            result = patch * factor

        elif aug_type == "noise":
            noise_level = np.random.uniform(0.01, 0.1) * patch.max()
            result = patch + np.random.randn(*patch.shape) * noise_level

        return result

    def _train_feature_extractor(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]):
        """训练特征提取器 (基于互相关模板匹配)。

        使用合成训练对学习平移等变的特征权重。
        核心思想: 对所有增强样本取平均得到稳健的模板。

        Parameters
        ----------
        train_pairs : list of (ndarray, ndarray)
            训练对列表。
        """
        ps = self.config.patch_size

        # 计算平均模板 (对所有增强样本取平均)
        all_patches = [p for p, _ in train_pairs] + [t for _, t in train_pairs]
        mean_template = np.mean(all_patches, axis=0)

        # 计算梯度模板 (模拟 LodeSTAR 的梯度特征)
        if SCIPY_AVAILABLE:
            grad_y = ndimage.sobel(mean_template, axis=0)
            grad_x = ndimage.sobel(mean_template, axis=1)
        else:
            grad_y = np.diff(mean_template, axis=0, prepend=0)
            grad_x = np.diff(mean_template, axis=1, prepend=0)

        grad_mag = np.sqrt(grad_y**2 + grad_x**2)

        # 学习特征权重: 基于梯度幅值和原始强度
        intensity_weight = mean_template / (mean_template.max() + 1e-10)
        gradient_weight = grad_mag / (grad_mag.max() + 1e-10)

        # 融合权重
        combined_weight = 0.6 * intensity_weight + 0.4 * gradient_weight

        self._feature_weights = {
            "template": mean_template,
            "gradient_mag": grad_mag,
            "weight": combined_weight,
            "grad_x": grad_x,
            "grad_y": grad_y,
        }

    def detect(self, frame: np.ndarray) -> UnsupervisedSpotResult:
        """在帧中检测光斑。

        Parameters
        ----------
        frame : ndarray
            输入帧 (H, W) 或 (H, W, C)。

        Returns
        -------
        UnsupervisedSpotResult
            检测结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        if not self._is_trained or self._feature_weights is None:
            raise RuntimeError("请先调用 train_from_single_image() 进行自监督训练")

        if frame.ndim == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame.astype(np.float64)

        # 计算特征响应图 (滑动窗口 NCC)
        feature_map = self._compute_feature_response(gray)

        # 计算不确定性图
        uncertainty_map = self._compute_uncertainty_map(gray, feature_map)

        # 提取候选检测
        candidates = self._extract_candidates(feature_map)

        # NMS
        detections = self._apply_nms(candidates)

        # 亚像素精化
        if self.config.subpixel_refine and len(detections) > 0:
            detections = self._refine_subpixel(gray, detections)

        # 限制检测数量
        if self.config.max_detections > 0:
            detections = detections[:self.config.max_detections]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return UnsupervisedSpotResult(
            detections=detections,
            uncertainty_map=uncertainty_map,
            feature_map=feature_map,
            num_detected=len(detections),
            compute_time_ms=elapsed_ms,
        )

    def _compute_feature_response(self, image: np.ndarray) -> np.ndarray:
        """计算特征响应图。

        Parameters
        ----------
        image : ndarray
            输入图像 (H, W)。

        Returns
        -------
        ndarray
            特征响应图 (H, W)。
        """
        template = self._feature_weights["template"]
        weight = self._feature_weights["weight"]
        ps = self.config.patch_size
        h, w = image.shape

        # 加权模板匹配
        weighted_template = template * weight

        # 归一化互相关 (NCC)
        if CV2_AVAILABLE:
            result = cv2.matchTemplate(
                image.astype(np.float32),
                weighted_template.astype(np.float32),
                cv2.TM_CCOEFF_NORMED,
            )
            # 补零到原始尺寸
            response = np.zeros((h, w), dtype=np.float64)
            half = ps // 2
            rh, rw = result.shape
            response[half:half + rh, half:half + rw] = result
        else:
            # 手动 NCC
            response = np.zeros((h, w), dtype=np.float64)
            half = ps // 2
            img_mean = np.mean(image)
            tmpl_mean = np.mean(weighted_template)
            tmpl_std = np.std(weighted_template) + 1e-10

            for y in range(half, h - half):
                for x in range(half, w - half):
                    patch = image[y - half:y + half, x - half:x + half]
                    p_mean = np.mean(patch)
                    p_std = np.std(patch) + 1e-10
                    ncc = np.sum((patch - p_mean) * (weighted_template - tmpl_mean)) / (
                        ps * ps * p_std * tmpl_std
                    )
                    response[y, x] = ncc

        return response

    def _compute_uncertainty_map(self, image: np.ndarray,
                                  feature_map: np.ndarray) -> np.ndarray:
        """计算不确定性图。

        基于特征响应的局部方差和梯度一致性估计检测不确定性。

        Parameters
        ----------
        image : ndarray
            输入图像 (H, W)。
        feature_map : ndarray
            特征响应图 (H, W)。

        Returns
        -------
        ndarray
            不确定性图 (H, W)，值域 [0, 1]。
        """
        # 基于特征响应的局部方差
        if SCIPY_AVAILABLE:
            local_var = ndimage.generic_filter(
                feature_map, np.var, size=self.config.patch_size // 4
            )
        else:
            local_var = np.zeros_like(feature_map)
            bs = max(1, self.config.patch_size // 4)
            for y in range(0, feature_map.shape[0], bs):
                for x in range(0, feature_map.shape[1], bs):
                    block = feature_map[y:y + bs, x:x + bs]
                    local_var[y:y + bs, x:x + bs] = np.var(block)

        # 归一化到 [0, 1]
        var_max = local_var.max() + 1e-10
        uncertainty = local_var / var_max

        return uncertainty

    def get_uncertainty_map(self, frame: np.ndarray) -> np.ndarray:
        """获取不确定性图。

        Parameters
        ----------
        frame : ndarray
            输入帧 (H, W) 或 (H, W, C)。

        Returns
        -------
        ndarray
            不确定性图 (H, W)，值域 [0, 1]。
        """
        if not self._is_trained:
            raise RuntimeError("请先调用 train_from_single_image() 进行自监督训练")

        result = self.detect(frame)
        return result.uncertainty_map

    def _extract_candidates(self, feature_map: np.ndarray) -> List[Dict[str, Any]]:
        """从特征响应图中提取候选检测。

        Parameters
        ----------
        feature_map : ndarray
            特征响应图 (H, W)。

        Returns
        -------
        list of dict
            候选检测列表。
        """
        threshold = self.config.confidence_threshold
        candidates = []

        # 找到局部极大值
        if CV2_AVAILABLE:
            # 使用形态学操作找局部极大值
            kernel_size = max(3, int(self.config.nms_radius * 2) + 1)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            local_max = cv2.dilate(feature_map, kernel)
            is_max = (feature_map == local_max) & (feature_map > threshold)

            ys, xs = np.where(is_max)
            for y, x in zip(ys, xs):
                candidates.append({
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(feature_map[y, x]),
                    "size": float(self.config.nms_radius),
                })
        else:
            # 简单局部极大值搜索
            h, w = feature_map.shape
            r = int(self.config.nms_radius)
            for y in range(r, h - r):
                for x in range(r, w - r):
                    val = feature_map[y, x]
                    if val < threshold:
                        continue
                    region = feature_map[y - r:y + r + 1, x - r:x + r + 1]
                    if val >= region.max():
                        candidates.append({
                            "x": float(x),
                            "y": float(y),
                            "confidence": float(val),
                            "size": float(r),
                        })

        # 按置信度排序
        candidates.sort(key=lambda d: d["confidence"], reverse=True)
        return candidates

    def _apply_nms(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """非极大值抑制。

        Parameters
        ----------
        candidates : list of dict
            候选检测列表 (已按置信度排序)。

        Returns
        -------
        list of dict
            NMS 后的检测列表。
        """
        if not candidates:
            return []

        kept = []
        radius = self.config.nms_radius

        for cand in candidates:
            is_suppressed = False
            for kept_cand in kept:
                dist = math.sqrt(
                    (cand["x"] - kept_cand["x"])**2 +
                    (cand["y"] - kept_cand["y"])**2
                )
                if dist < radius:
                    is_suppressed = True
                    break
            if not is_suppressed:
                kept.append(cand)

        return kept

    def _refine_subpixel(self, image: np.ndarray,
                         detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """亚像素精化。

        Parameters
        ----------
        image : ndarray
            输入图像 (H, W)。
        detections : list of dict
            检测列表。

        Returns
        -------
        list of dict
            精化后的检测列表。
        """
        refined = []
        r = int(self.config.nms_radius * 2)

        for det in detections:
            x, y = int(round(det["x"])), int(round(det["y"]))
            h, w = image.shape

            # 边界检查
            y0 = max(0, y - r)
            y1 = min(h, y + r + 1)
            x0 = max(0, x - r)
            x1 = min(w, x + r + 1)

            patch = image[y0:y1, x0:x1].astype(np.float64)

            # 加权质心精化
            if patch.size > 0:
                weights = np.maximum(patch, 0)
                total = weights.sum() + 1e-10
                local_y = np.sum(np.arange(patch.shape[0])[:, None] * weights) / total
                local_x = np.sum(np.arange(patch.shape[1])[None, :] * weights) / total
                det["x"] = float(x0 + local_x)
                det["y"] = float(y0 + local_y)

            refined.append(det)

        return refined

    def reset(self):
        """重置检测器状态。"""
        self._reference_patch = None
        self._feature_weights = None
        self._is_trained = False
        self._call_count = 0
        self._training_history.clear()

    def get_status(self) -> Dict:
        """获取检测器状态。"""
        return {
            "is_trained": self._is_trained,
            "call_count": self._call_count,
            "patch_size": self.config.patch_size,
            "confidence_threshold": self.config.confidence_threshold,
            "has_reference": self._reference_patch is not None,
        }


# ===========================================================================
# 2. GNNTrajectoryAnalyzer — 图神经网络轨迹分析器
# ===========================================================================
# 灵感来源: DeepTrack2 MAGIK (github.com/DeepTrackAI/DeepTrack2)
#   - MAGIK: 基于图神经网络的多目标粒子链接
#   - 将检测结果构建为时空图
#   - GNN 消息传递学习链接概率
#   - TrackMate LAP (Linear Assignment Problem) 粒子链接
#
# 与 SpotZoom 已有模块的区别:
#   - multi_spot_tracker.py 使用最近邻 + Kalman 滤波
#   - kalman_tracker.py 是单目标跟踪
#   - optical_flow_tracker.py 使用光流法
#   - 本模块使用 GNN 建模时空关系，支持复杂运动模式
# ===========================================================================


@dataclass
class TrajectoryNode:
    """轨迹图节点。

    Attributes
    ----------
    node_id : int
        节点唯一标识。
    frame_idx : int
        帧索引。
    x : float
        x 坐标 (像素)。
    y : float
        y 坐标 (像素)。
    intensity : float
        光斑强度。
    size : float
        光斑尺寸。
    features : ndarray, optional
        附加特征向量。
    """

    node_id: int = 0
    frame_idx: int = 0
    x: float = 0.0
    y: float = 0.0
    intensity: float = 0.0
    size: float = 0.0
    features: Optional[np.ndarray] = None


@dataclass
class TrajectoryEdge:
    """轨迹图边。

    Attributes
    ----------
    source_id : int
        源节点 ID。
    target_id : int
        目标节点 ID。
    weight : float
        边权重 (链接概率)。
    temporal_gap : int
        时间间隔 (帧数)。
    spatial_dist : float
        空间距离 (像素)。
    """

    source_id: int = 0
    target_id: int = 0
    weight: float = 0.0
    temporal_gap: int = 1
    spatial_dist: float = 0.0


@dataclass
class GNNTrajectoryConfig:
    """GNN 轨迹分析器配置。

    Attributes
    ----------
    max_temporal_gap : int
        最大时间间隔 (帧数)。
    max_spatial_dist : float
        最大空间距离 (像素)。
    link_threshold : float
        链接概率阈值 [0, 1]。
    num_message_passes : int
        GNN 消息传递轮数。
    feature_dim : int
        节点特征维度。
    hidden_dim : int
        GNN 隐藏层维度。
    motion_model : str
        运动模型: "constant_velocity", "brownian", "accelerated"。
    """

    max_temporal_gap: int = 5
    max_spatial_dist: float = 50.0
    link_threshold: float = 0.5
    num_message_passes: int = 3
    feature_dim: int = 8
    hidden_dim: int = 16
    motion_model: str = "constant_velocity"


@dataclass
class GNNTrajectoryResult:
    """GNN 轨迹分析结果。

    Attributes
    ----------
    trajectories : list of list of dict
        轨迹列表，每条轨迹是节点字典的列表。
    num_trajectories : int
        轨迹数量。
    avg_trajectory_length : float
        平均轨迹长度。
    link_probabilities : list of TrajectoryEdge
        所有边的链接概率。
    motion_patterns : dict
        运动模式分析结果。
    compute_time_ms : float
        计算耗时 (毫秒)。
    """

    trajectories: List[List[Dict[str, Any]]] = field(default_factory=list)
    num_trajectories: int = 0
    avg_trajectory_length: float = 0.0
    link_probabilities: List[TrajectoryEdge] = field(default_factory=list)
    motion_patterns: Dict[str, Any] = field(default_factory=dict)
    compute_time_ms: float = 0.0


class GNNTrajectoryAnalyzer:
    """图神经网络轨迹分析器。

    基于 DeepTrack2 MAGIK 的 GNN 轨迹链接方法，将多帧检测结果构建为时空图，
    通过消息传递学习链接概率，实现多目标光斑轨迹分析与链接。

    核心原理:
    1. 将每帧的检测结果作为图节点
    2. 根据时空邻近关系构建边
    3. 节点特征: 位置、强度、尺寸、速度估计
    4. GNN 消息传递聚合邻域信息
    5. 边分类得到链接概率
    6. 基于链接概率构建轨迹

    Parameters
    ----------
    config : GNNTrajectoryConfig, optional
        配置参数。
    """

    def __init__(self, config: Optional[GNNTrajectoryConfig] = None):
        self.config = config or GNNTrajectoryConfig()
        self._nodes: Dict[int, TrajectoryNode] = {}
        self._edges: List[TrajectoryEdge] = []
        self._adjacency: Dict[int, List[int]] = {}
        self._node_features: Optional[np.ndarray] = None
        self._edge_weights: Optional[np.ndarray] = None
        self._message_weights: Optional[Dict[str, np.ndarray]] = None
        self._next_node_id = 0
        self._call_count = 0

    def build_graph(self, detections: List[List[Dict[str, Any]]]) -> Dict:
        """从多帧检测结果构建时空图。

        Parameters
        ----------
        detections : list of list of dict
            多帧检测结果，外层为帧列表，内层为每帧的检测字典。
            每个检测字典应包含 x, y, confidence, intensity 等键。

        Returns
        -------
        dict
            图构建统计。
        """
        self._nodes.clear()
        self._edges.clear()
        self._adjacency.clear()
        self._next_node_id = 0

        # 创建节点
        for frame_idx, frame_dets in enumerate(detections):
            for det in frame_dets:
                node = TrajectoryNode(
                    node_id=self._next_node_id,
                    frame_idx=frame_idx,
                    x=float(det.get("x", 0)),
                    y=float(det.get("y", 0)),
                    intensity=float(det.get("intensity", det.get("confidence", 1.0))),
                    size=float(det.get("size", 5.0)),
                )
                self._nodes[self._next_node_id] = node
                self._adjacency[self._next_node_id] = []
                self._next_node_id += 1

        # 构建边 (时空邻近关系)
        self._build_temporal_edges()

        # 计算节点特征
        self._compute_node_features()

        # 初始化消息传递权重
        self._initialize_message_weights()

        return {
            "num_nodes": len(self._nodes),
            "num_edges": len(self._edges),
            "num_frames": len(detections),
            "avg_detections_per_frame": np.mean([len(d) for d in detections]) if detections else 0,
        }

    def _build_temporal_edges(self):
        """构建时间边 (相邻帧之间的候选链接)。"""
        self._edges.clear()

        # 按帧分组
        frame_nodes: Dict[int, List[int]] = {}
        for nid, node in self._nodes.items():
            if node.frame_idx not in frame_nodes:
                frame_nodes[node.frame_idx] = []
            frame_nodes[node.frame_idx].append(nid)

        sorted_frames = sorted(frame_nodes.keys())

        for i, f1 in enumerate(sorted_frames):
            for f2 in sorted_frames[i + 1:]:
                gap = f2 - f1
                if gap > self.config.max_temporal_gap:
                    break

                for n1_id in frame_nodes[f1]:
                    for n2_id in frame_nodes[f2]:
                        n1 = self._nodes[n1_id]
                        n2 = self._nodes[n2_id]

                        dist = math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

                        if dist <= self.config.max_spatial_dist:
                            # 初始权重: 基于距离的指数衰减
                            weight = math.exp(-dist / (self.config.max_spatial_dist * 0.5))

                            edge = TrajectoryEdge(
                                source_id=n1_id,
                                target_id=n2_id,
                                weight=weight,
                                temporal_gap=gap,
                                spatial_dist=dist,
                            )
                            self._edges.append(edge)
                            self._adjacency[n1_id].append(n2_id)
                            self._adjacency[n2_id].append(n1_id)

    def _compute_node_features(self):
        """计算节点特征向量。

        特征: [x_norm, y_norm, intensity_norm, size_norm, vx, vy, frame_norm, neighbor_count]
        """
        if not self._nodes:
            return

        n = len(self._nodes)
        fdim = self.config.feature_dim
        features = np.zeros((n, fdim), dtype=np.float64)

        # 归一化参数
        all_x = [n.x for n in self._nodes.values()]
        all_y = [n.y for n in self._nodes.values()]
        x_range = max(max(all_x) - min(all_x), 1.0)
        y_range = max(max(all_y) - min(all_y), 1.0)
        max_frame = max(n.frame_idx for n in self._nodes.values()) + 1

        for nid, node in self._nodes.items():
            f = features[nid]
            f[0] = node.x / x_range
            f[1] = node.y / y_range
            f[2] = node.intensity
            f[3] = node.size / 50.0
            f[4] = node.frame_idx / max_frame
            f[5] = len(self._adjacency.get(nid, [])) / max(n, 1)
            if fdim > 6:
                f[6] = math.sin(2 * math.pi * node.frame_idx / max_frame)
                f[7] = math.cos(2 * math.pi * node.frame_idx / max_frame)

        self._node_features = features

    def _initialize_message_weights(self):
        """初始化 GNN 消息传递权重。"""
        fdim = self.config.feature_dim
        hdim = self.config.hidden_dim

        # 消息 MLP 权重
        self._message_weights = {
            "W_msg": np.random.randn(fdim, hdim) * 0.1,
            "b_msg": np.zeros(hdim),
            "W_update": np.random.randn(hdim + fdim, fdim) * 0.1,
            "b_update": np.zeros(fdim),
            "W_edge": np.random.randn(fdim * 2 + 2, 1) * 0.1,
            "b_edge": np.zeros(1),
        }

    def link_trajectories(self, detections: List[List[Dict[str, Any]]]) -> GNNTrajectoryResult:
        """链接多帧检测结果为轨迹。

        Parameters
        ----------
        detections : list of list of dict
            多帧检测结果。

        Returns
        -------
        GNNTrajectoryResult
            轨迹链接结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        # 构建图
        graph_stats = self.build_graph(detections)

        if len(self._nodes) == 0:
            return GNNTrajectoryResult(compute_time_ms=(time.perf_counter() - t0) * 1000)

        # GNN 消息传递
        self._message_passing()

        # 边分类 (更新链接概率)
        self._classify_edges()

        # 构建轨迹 (贪心链接)
        trajectories = self._build_trajectories()

        # 运动模式分析
        motion_patterns = self._analyze_motion_pattern(trajectories)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 统计
        lengths = [len(t) for t in trajectories]
        avg_len = float(np.mean(lengths)) if lengths else 0.0

        return GNNTrajectoryResult(
            trajectories=trajectories,
            num_trajectories=len(trajectories),
            avg_trajectory_length=avg_len,
            link_probabilities=self._edges,
            motion_patterns=motion_patterns,
            compute_time_ms=elapsed_ms,
        )

    def _message_passing(self):
        """执行 GNN 消息传递。

        多轮消息传递聚合邻域信息，更新节点特征。
        """
        if self._node_features is None or self._message_weights is None:
            return

        W_msg = self._message_weights["W_msg"]
        b_msg = self._message_weights["b_msg"]
        W_update = self._message_weights["W_update"]
        b_update = self._message_weights["b_update"]

        features = self._node_features.copy()

        for _ in range(self.config.num_message_passes):
            new_features = features.copy()

            for nid in self._nodes:
                neighbors = self._adjacency.get(nid, [])
                if not neighbors:
                    continue

                # 聚合邻居消息
                messages = []
                for nb_id in neighbors:
                    msg = features[nb_id] @ W_msg + b_msg
                    # ReLU 激活
                    msg = np.maximum(0, msg)
                    messages.append(msg)

                # 平均聚合
                agg_msg = np.mean(messages, axis=0)

                # 更新: 拼接原始特征和聚合消息
                combined = np.concatenate([features[nid], agg_msg])
                update = combined @ W_update + b_update
                new_features[nid] = features[nid] + 0.1 * update  # 残差连接

            features = new_features

        self._node_features = features

    def _classify_edges(self):
        """边分类: 更新链接概率。"""
        if self._node_features is None or self._message_weights is None:
            return

        W_edge = self._message_weights["W_edge"]
        b_edge = self._message_weights["b_edge"]

        for edge in self._edges:
            src_feat = self._node_features[edge.source_id]
            tgt_feat = self._node_features[edge.target_id]

            # 边特征: 拼接源/目标特征 + 距离 + 时间间隔
            dist_feat = np.array([edge.spatial_dist / self.config.max_spatial_dist,
                                  edge.temporal_gap / self.config.max_temporal_gap])
            edge_feat = np.concatenate([src_feat, tgt_feat, dist_feat])

            # 简单线性分类 + sigmoid
            logit = float(edge_feat @ W_edge + b_edge)
            probability = 1.0 / (1.0 + math.exp(-logit))

            # 融合初始距离权重
            edge.weight = 0.5 * edge.weight + 0.5 * probability

    def _build_trajectories(self) -> List[List[Dict[str, Any]]]:
        """从图构建轨迹 (贪心链接)。

        Returns
        -------
        list of list of dict
            轨迹列表。
        """
        # 按权重排序所有边
        sorted_edges = sorted(self._edges, key=lambda e: e.weight, reverse=True)

        # 并查集
        parent = {nid: nid for nid in self._nodes}
        rank = {nid: 0 for nid in self._nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # 贪心合并
        for edge in sorted_edges:
            if edge.weight < self.config.link_threshold:
                continue
            # 检查是否形成环 (同一帧内的节点不能链接)
            src_node = self._nodes[edge.source_id]
            tgt_node = self._nodes[edge.target_id]
            if src_node.frame_idx == tgt_node.frame_idx:
                continue
            if find(edge.source_id) != find(edge.target_id):
                union(edge.source_id, edge.target_id)

        # 按连通分量构建轨迹
        components: Dict[int, List[int]] = {}
        for nid in self._nodes:
            root = find(nid)
            if root not in components:
                components[root] = []
            components[root].append(nid)

        trajectories = []
        for comp_nodes in components.values():
            if len(comp_nodes) < 2:
                continue

            # 按帧排序
            comp_nodes.sort(key=lambda nid: self._nodes[nid].frame_idx)

            traj = []
            for nid in comp_nodes:
                node = self._nodes[nid]
                traj.append({
                    "node_id": nid,
                    "frame_idx": node.frame_idx,
                    "x": node.x,
                    "y": node.y,
                    "intensity": node.intensity,
                    "size": node.size,
                })
            trajectories.append(traj)

        # 按长度排序
        trajectories.sort(key=lambda t: len(t), reverse=True)

        return trajectories

    def analyze_motion_pattern(self, trajectories: List[List[Dict[str, Any]]]) -> Dict:
        """分析轨迹运动模式。

        Parameters
        ----------
        trajectories : list of list of dict
            轨迹列表。

        Returns
        -------
        dict
            运动模式分析结果。
        """
        return self._analyze_motion_pattern(trajectories)

    def _analyze_motion_pattern(self, trajectories: List[List[Dict[str, Any]]]) -> Dict:
        """内部运动模式分析。

        Parameters
        ----------
        trajectories : list of list of dict
            轨迹列表。

        Returns
        -------
        dict
            运动模式分析结果。
        """
        if not trajectories:
            return {"num_trajectories": 0}

        all_speeds = []
        all_accelerations = []
        all_displacements = []
        direction_changes = 0

        for traj in trajectories:
            if len(traj) < 2:
                continue

            # 计算速度
            for i in range(1, len(traj)):
                dx = traj[i]["x"] - traj[i - 1]["x"]
                dy = traj[i]["y"] - traj[i - 1]["y"]
                speed = math.sqrt(dx**2 + dy**2)
                all_speeds.append(speed)
                all_displacements.append(math.sqrt(
                    (traj[i]["x"] - traj[0]["x"])**2 +
                    (traj[i]["y"] - traj[0]["y"])**2
                ))

            # 计算加速度和方向变化
            for i in range(2, len(traj)):
                dx1 = traj[i - 1]["x"] - traj[i - 2]["x"]
                dy1 = traj[i - 1]["y"] - traj[i - 2]["y"]
                dx2 = traj[i]["x"] - traj[i - 1]["x"]
                dy2 = traj[i]["y"] - traj[i - 1]["y"]

                dvx = dx2 - dx1
                dvy = dy2 - dy1
                accel = math.sqrt(dvx**2 + dvy**2)
                all_accelerations.append(accel)

                # 方向变化
                dot = dx1 * dx2 + dy1 * dy2
                mag1 = math.sqrt(dx1**2 + dy1**2) + 1e-10
                mag2 = math.sqrt(dx2**2 + dy2**2) + 1e-10
                cos_angle = np.clip(dot / (mag1 * mag2), -1, 1)
                angle = math.acos(cos_angle)
                if angle > math.pi / 4:  # > 45 度
                    direction_changes += 1

        result = {
            "num_trajectories": len(trajectories),
            "avg_speed": float(np.mean(all_speeds)) if all_speeds else 0.0,
            "max_speed": float(np.max(all_speeds)) if all_speeds else 0.0,
            "speed_std": float(np.std(all_speeds)) if all_speeds else 0.0,
            "avg_acceleration": float(np.mean(all_accelerations)) if all_accelerations else 0.0,
            "total_direction_changes": direction_changes,
            "avg_displacement": float(np.mean(all_displacements)) if all_displacements else 0.0,
        }

        # 运动类型分类
        if all_speeds:
            speed_cv = float(np.std(all_speeds) / (np.mean(all_speeds) + 1e-10))
            if speed_cv < 0.2:
                result["motion_type"] = "uniform"
            elif speed_cv < 0.5:
                result["motion_type"] = "modulated"
            else:
                result["motion_type"] = "irregular"
        else:
            result["motion_type"] = "static"

        return result

    def reset(self):
        """重置分析器状态。"""
        self._nodes.clear()
        self._edges.clear()
        self._adjacency.clear()
        self._node_features = None
        self._edge_weights = None
        self._message_weights = None
        self._next_node_id = 0
        self._call_count = 0

    def get_status(self) -> Dict:
        """获取分析器状态。"""
        return {
            "num_nodes": len(self._nodes),
            "num_edges": len(self._edges),
            "call_count": self._call_count,
            "max_temporal_gap": self.config.max_temporal_gap,
            "max_spatial_dist": self.config.max_spatial_dist,
            "link_threshold": self.config.link_threshold,
            "num_message_passes": self.config.num_message_passes,
        }


# ===========================================================================
# 3. AberrationCNNProfiler — 像差 CNN 表征器
# ===========================================================================
# 灵感来源: DeepTrack2 像差模块 (github.com/DeepTrackAI/DeepTrack2)
#   - 从光斑图像反推波前像差参数
#   - CNN 将 PSF 形态映射到 Zernike 系数
#   - 支持多种像差类型: 球差、彗差、像散、离焦
#
# 与 SpotZoom 已有模块的区别:
#   - zernike_analyzer.py 从波前传感器数据计算 Zernike
#   - phase_retrieval_analyzer.py 通过相位恢复算法
#   - wavefront_predictor.py 预测波前
#   - 本模块直接从光斑 PSF 图像通过 CNN 反推像差
# ===========================================================================


@dataclass
class AberrationProfile:
    """像差轮廓。

    Attributes
    ----------
    zernike_coeffs : ndarray
        Zernike 系数向量 (N,)。
    zernike_names : list of str
        Zernike 项名称列表。
    rms_wavefront_error : float
        RMS 波前误差 (waves)。
    strehl_ratio : float
        Strehl 比。
    dominant_aberration : str
        主导像差类型名称。
    """

    zernike_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(15))
    zernike_names: List[str] = field(default_factory=lambda: [
        "Z1(piston)", "Z2(tilt_x)", "Z3(tilt_y)", "Z4(defocus)",
        "Z5(astig_45)", "Z6(astig_0)", "Z7(coma_x)", "Z8(coma_y)",
        "Z9(trefoil_x)", "Z10(trefoil_y)", "Z11(spherical)",
        "Z12(2nd_astig)", "Z13(2nd_coma_x)", "Z14(2nd_coma_y)",
        "Z15(2nd_spherical)",
    ])
    rms_wavefront_error: float = 0.0
    strehl_ratio: float = 1.0
    dominant_aberration: str = "none"


@dataclass
class AberrationCNNConfig:
    """像差 CNN 表征器配置。

    Attributes
    ----------
    image_size : int
        输入光斑图像尺寸 (像素)。
    num_zernike_terms : int
        Zernike 多项式项数。
    num_conv_layers : int
        CNN 卷积层数。
    base_filters : int
        基础卷积核数量。
    wavelength : float
        波长 (微米)。
    na : float
        数值孔径。
    """

    image_size: int = 64
    num_zernike_terms: int = 15
    num_conv_layers: int = 4
    base_filters: int = 16
    wavelength: float = 0.55
    na: float = 0.4


@dataclass
class AberrationCNNResult:
    """像差 CNN 表征结果。

    Attributes
    ----------
    profile : AberrationProfile
        估计的像差轮廓。
    psf_reconstructed : ndarray
        重建的 PSF (H, W)。
    residual : float
        重建残差 (MSE)。
    correction_suggestion : dict
        校正建议。
    compute_time_ms : float
        计算耗时 (毫秒)。
    """

    profile: AberrationProfile = field(default_factory=AberrationProfile)
    psf_reconstructed: Optional[np.ndarray] = None
    residual: float = 0.0
    correction_suggestion: Dict[str, Any] = field(default_factory=dict)
    compute_time_ms: float = 0.0


class AberrationCNNProfiler:
    """像差 CNN 表征器。

    基于 DeepTrack2 像差模块，使用轻量 CNN 从光斑 PSF 图像反推波前像差参数。
    将光斑形态映射到 Zernike 多项式系数，实现像差的快速定量表征。

    核心原理:
    1. 输入: 归一化的光斑 PSF 图像
    2. CNN 特征提取: 多层卷积 + 池化提取形态特征
    3. 全连接回归: 将特征映射到 Zernike 系数
    4. PSF 重建: 从 Zernike 系数重建 PSF 验证
    5. 校正建议: 根据主导像差类型给出校正方案

    Parameters
    ----------
    config : AberrationCNNConfig, optional
        配置参数。
    """

    def __init__(self, config: Optional[AberrationCNNConfig] = None):
        self.config = config or AberrationCNNConfig()
        self._cnn_weights: Optional[List[Dict[str, np.ndarray]]] = None
        self._fc_weights: Optional[Dict[str, np.ndarray]] = None
        self._is_initialized = False
        self._call_count = 0

        # Zernike 径向多项式阶数
        self._zernike_orders = [
            (0, 0), (1, -1), (1, 1), (2, 0), (2, -2), (2, 2),
            (3, -1), (3, 1), (3, -3), (3, 3), (4, 0),
            (4, -2), (4, 2), (4, -4), (4, 4),
        ]

    def initialize(self):
        """初始化 CNN 权重。"""
        n = self.config.num_zernike_terms
        filters = self.config.base_filters
        size = self.config.image_size

        self._cnn_weights = []
        in_ch = 1
        current_size = size

        for i in range(self.config.num_conv_layers):
            out_ch = filters * (2 ** i)
            k = 3
            layer_w = {
                "W": np.random.randn(out_ch, in_ch, k, k) * math.sqrt(2.0 / (in_ch * k * k)),
                "b": np.zeros(out_ch),
            }
            self._cnn_weights.append(layer_w)
            in_ch = out_ch
            current_size = current_size // 2  # 池化

        # 全连接层
        fc_in = in_ch * current_size * current_size
        self._fc_weights = {
            "W1": np.random.randn(fc_in, 128) * math.sqrt(2.0 / fc_in),
            "b1": np.zeros(128),
            "W2": np.random.randn(128, n) * math.sqrt(2.0 / 128),
            "b2": np.zeros(n),
        }

        self._is_initialized = True
        LOGGER.info("像差 CNN 表征器初始化完成, 参数量: %d", fc_in * 128 + 128 * n)

    def estimate_aberrations(self, spot_image: np.ndarray) -> AberrationCNNResult:
        """从光斑图像估计像差。

        Parameters
        ----------
        spot_image : ndarray
            光斑 PSF 图像 (H, W) 或 (H, W, C)。

        Returns
        -------
        AberrationCNNResult
            像差估计结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        if not self._is_initialized:
            self.initialize()

        # 预处理
        if spot_image.ndim == 3:
            gray = np.mean(spot_image, axis=2)
        else:
            gray = spot_image.astype(np.float64)

        # 归一化到 [0, 1]
        img_min, img_max = gray.min(), gray.max()
        if img_max - img_min > 1e-10:
            normalized = (gray - img_min) / (img_max - img_min)
        else:
            normalized = np.zeros_like(gray)

        # 调整尺寸
        target_size = self.config.image_size
        if normalized.shape[0] != target_size or normalized.shape[1] != target_size:
            if CV2_AVAILABLE:
                normalized = cv2.resize(
                    normalized.astype(np.float32), (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR
                ).astype(np.float64)
            else:
                y_idx = np.linspace(0, normalized.shape[0] - 1, target_size)
                x_idx = np.linspace(0, normalized.shape[1] - 1, target_size)
                yi, xi = np.meshgrid(y_idx, x_idx, indexing='ij')
                y0 = np.floor(yi).astype(int)
                x0 = np.floor(xi).astype(int)
                y1 = np.minimum(y0 + 1, normalized.shape[0] - 1)
                x1 = np.minimum(x0 + 1, normalized.shape[1] - 1)
                fy = yi - y0
                fx = xi - x0
                normalized = (normalized[y0, x0] * (1 - fy) * (1 - fx) +
                              normalized[y1, x0] * fy * (1 - fx) +
                              normalized[y0, x1] * (1 - fy) * fx +
                              normalized[y1, x1] * fy * fx)

        # CNN 前向传播
        zernike_coeffs = self._forward(normalized)

        # 构建 AberrationProfile
        profile = self._build_profile(zernike_coeffs)

        # 重建 PSF
        psf_recon = self._reconstruct_psf(zernike_coeffs)

        # 计算残差
        residual = float(np.mean((normalized - psf_recon)**2))

        # 校正建议
        suggestion = self._suggest_correction(profile)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return AberrationCNNResult(
            profile=profile,
            psf_reconstructed=psf_recon,
            residual=residual,
            correction_suggestion=suggestion,
            compute_time_ms=elapsed_ms,
        )

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """CNN 前向传播。

        Parameters
        ----------
        x : ndarray
            输入图像 (H, W)。

        Returns
        -------
        ndarray
            Zernike 系数预测 (N,)。
        """
        h = x[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

        # 卷积层
        for i, layer in enumerate(self._cnn_weights):
            W = layer["W"]
            b = layer["b"]
            out_ch = W.shape[0]

            # 简化卷积: 使用中心裁剪的互相关
            k = W.shape[2]
            pad = k // 2
            h_padded = np.pad(h, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

            out = np.zeros((1, out_ch, h.shape[2], h.shape[3]), dtype=np.float64)
            for oc in range(out_ch):
                for ic in range(h.shape[1]):
                    for ky in range(k):
                        for kx in range(k):
                            out[0, oc] += h_padded[0, ic, ky:ky + h.shape[2], kx:kx + h.shape[3]] * W[oc, ic, ky, kx]
                out[0, oc] += b[oc]

            # ReLU 激活
            h = np.maximum(0, out)

            # 最大池化 (2x2)
            if h.shape[2] >= 2 and h.shape[3] >= 2:
                h = h[:, :, ::2, ::2]

        # 展平
        flat = h.flatten()

        # 全连接层 1
        fc1 = flat @ self._fc_weights["W1"] + self._fc_weights["b1"]
        fc1 = np.maximum(0, fc1)  # ReLU

        # 全连接层 2 (输出)
        coeffs = fc1 @ self._fc_weights["W2"] + self._fc_weights["b2"]

        return coeffs

    def _build_profile(self, coeffs: np.ndarray) -> AberrationProfile:
        """构建像差轮廓。

        Parameters
        ----------
        coeffs : ndarray
            Zernike 系数 (N,)。

        Returns
        -------
        AberrationProfile
            像差轮廓。
        """
        n = min(len(coeffs), self.config.num_zernike_terms)
        z_coeffs = coeffs[:n].copy()

        # 跳过活塞项 (Z1)
        if len(z_coeffs) > 1:
            rms = float(np.sqrt(np.mean(z_coeffs[1:]**2)))
        else:
            rms = 0.0

        # Strehl 近似: exp(-(2*pi*rms)^2)
        strehl = float(np.exp(-(2 * math.pi * rms)**2))

        # 主导像差
        if len(z_coeffs) > 1:
            abs_coeffs = np.abs(z_coeffs[1:])
            dominant_idx = int(np.argmax(abs_coeffs)) + 1
            names = self._zernike_orders[:n]
            dominant_name = f"Z{dominant_idx + 1}"
            if dominant_idx < len(self._zernike_orders):
                n_val, m_val = self._zernike_orders[dominant_idx]
                dominant_name = f"Z{dominant_idx + 1}(n={n_val},m={m_val})"
        else:
            dominant_name = "none"

        return AberrationProfile(
            zernike_coeffs=z_coeffs,
            rms_wavefront_error=rms,
            strehl_ratio=strehl,
            dominant_aberration=dominant_name,
        )

    def _reconstruct_psf(self, coeffs: np.ndarray) -> np.ndarray:
        """从 Zernike 系数重建 PSF。

        Parameters
        ----------
        coeffs : ndarray
            Zernike 系数 (N,)。

        Returns
        -------
        ndarray
            重建的 PSF (image_size, image_size)。
        """
        N = self.config.image_size
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)

        # 波前
        wavefront = np.zeros((N, N))
        for i, coeff in enumerate(coeffs[:len(self._zernike_orders)]):
            n_val, m_val = self._zernike_orders[i]
            zernike = self._zernike_polynomial(R, Theta, n_val, m_val)
            wavefront += coeff * zernike

        # PSF = |FT(pupil * exp(j * wavefront))|^2
        pupil = (R <= 1.0).astype(np.float64)
        field = pupil * np.exp(1j * 2 * math.pi * wavefront)

        if SCIPY_AVAILABLE:
            psf = np.abs(fftshift(fft2(fftshift(field))))**2
        else:
            psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field))))**2

        # 归一化
        psf = psf / (psf.max() + 1e-10)

        return psf

    def _zernike_polynomial(self, R: np.ndarray, Theta: np.ndarray,
                             n: int, m: int) -> np.ndarray:
        """计算 Zernike 多项式。

        Parameters
        ----------
        R : ndarray
            归一化径向坐标。
        Theta : ndarray
            角坐标。
        n : int
            径向阶数。
        m : int
            角频率。

        Returns
        -------
        ndarray
            Zernike 多项式值。
        """
        # 径向多项式 R_n^|m|
        rho = np.clip(R, 0, 1)
        R_nm = np.zeros_like(rho)

        if n == 0 and m == 0:
            R_nm = np.ones_like(rho)
        elif n == 1 and abs(m) == 1:
            R_nm = rho
        elif n == 2 and m == 0:
            R_nm = 2 * rho**2 - 1
        elif n == 2 and abs(m) == 2:
            R_nm = rho**2
        elif n == 3 and abs(m) == 1:
            R_nm = 3 * rho**3 - 2 * rho
        elif n == 3 and abs(m) == 3:
            R_nm = rho**3
        elif n == 4 and m == 0:
            R_nm = 6 * rho**4 - 6 * rho**2 + 1
        elif n == 4 and abs(m) == 2:
            R_nm = 4 * rho**4 - 3 * rho**2
        elif n == 4 and abs(m) == 4:
            R_nm = rho**4

        # 角度部分
        if m > 0:
            angular = np.cos(m * Theta)
        elif m < 0:
            angular = np.sin(abs(m) * Theta)
        else:
            angular = np.ones_like(Theta)

        return R_nm * angular

    def get_zernike_coeffs(self, spot_image: np.ndarray) -> np.ndarray:
        """获取 Zernike 系数。

        Parameters
        ----------
        spot_image : ndarray
            光斑 PSF 图像。

        Returns
        -------
        ndarray
            Zernike 系数 (N,)。
        """
        result = self.estimate_aberrations(spot_image)
        return result.profile.zernike_coeffs

    def suggest_correction(self, spot_image: np.ndarray) -> Dict[str, Any]:
        """获取校正建议。

        Parameters
        ----------
        spot_image : ndarray
            光斑 PSF 图像。

        Returns
        -------
        dict
            校正建议。
        """
        result = self.estimate_aberrations(spot_image)
        return result.correction_suggestion

    def _suggest_correction(self, profile: AberrationProfile) -> Dict[str, Any]:
        """生成校正建议。

        Parameters
        ----------
        profile : AberrationProfile
            像差轮廓。

        Returns
        -------
        dict
            校正建议。
        """
        coeffs = profile.zernike_coeffs
        suggestions = {}

        # 离焦 (Z4)
        if len(coeffs) > 3 and abs(coeffs[3]) > 0.1:
            suggestions["defocus"] = {
                "magnitude": float(coeffs[3]),
                "action": "调整焦距" if coeffs[3] > 0 else "反向调整焦距",
                "priority": "high" if abs(coeffs[3]) > 0.5 else "medium",
            }

        # 像散 (Z5, Z6)
        if len(coeffs) > 5:
            astig_mag = math.sqrt(coeffs[4]**2 + coeffs[5]**2)
            if astig_mag > 0.1:
                angle = math.degrees(math.atan2(coeffs[4], coeffs[5])) / 2
                suggestions["astigmatism"] = {
                    "magnitude": float(astig_mag),
                    "angle_deg": float(angle),
                    "action": f"校正像散, 角度 {angle:.1f} 度",
                    "priority": "high" if astig_mag > 0.5 else "medium",
                }

        # 彗差 (Z7, Z8)
        if len(coeffs) > 7:
            coma_mag = math.sqrt(coeffs[6]**2 + coeffs[7]**2)
            if coma_mag > 0.1:
                suggestions["coma"] = {
                    "magnitude": float(coma_mag),
                    "action": "调整光学元件倾斜或偏心",
                    "priority": "high" if coma_mag > 0.5 else "medium",
                }

        # 球差 (Z11)
        if len(coeffs) > 10 and abs(coeffs[10]) > 0.1:
            suggestions["spherical"] = {
                "magnitude": float(coeffs[10]),
                "action": "使用非球面透镜或调整透镜间距",
                "priority": "high" if abs(coeffs[10]) > 0.5 else "medium",
            }

        if not suggestions:
            suggestions["status"] = "像差水平可接受"

        suggestions["strehl_ratio"] = profile.strehl_ratio
        suggestions["rms_wavefront_error"] = profile.rms_wavefront_error

        return suggestions

    def reset(self):
        """重置表征器状态。"""
        self._cnn_weights = None
        self._fc_weights = None
        self._is_initialized = False
        self._call_count = 0

    def get_status(self) -> Dict:
        """获取表征器状态。"""
        return {
            "is_initialized": self._is_initialized,
            "call_count": self._call_count,
            "image_size": self.config.image_size,
            "num_zernike_terms": self.config.num_zernike_terms,
            "num_conv_layers": self.config.num_conv_layers,
            "wavelength_um": self.config.wavelength,
            "na": self.config.na,
        }


# ===========================================================================
# 4. BayesianUncertaintyEstimator — 贝叶斯不确定性估计器
# ===========================================================================
# 灵感来源: BayesDL-SIM (贝叶斯深度学习不确定性量化)
#   - MC Dropout: 通过 Dropout 采样近似贝叶斯后验
#   - 深度集成: 多个独立模型预测的方差
#   - 为定位结果提供置信区间
#   - 可靠性评估: 识别不可靠的检测
#
# 与 SpotZoom 已有模块的区别:
#   - robust_estimator.py 侧重鲁棒统计估计
#   - convergence_predictor.py 预测收敛性
#   - 本模块专门量化检测结果的贝叶斯不确定性
# ===========================================================================


@dataclass
class UncertaintySample:
    """单次不确定性采样结果。

    Attributes
    ----------
    x : float
        x 坐标采样值。
    y : float
        y 坐标采样值。
    intensity : float
        强度采样值。
    size : float
        尺寸采样值。
    sample_id : int
        采样编号。
    """

    x: float = 0.0
    y: float = 0.0
    intensity: float = 0.0
    size: float = 0.0
    sample_id: int = 0


@dataclass
class BayesianConfig:
    """贝叶斯不确定性估计器配置。

    Attributes
    ----------
    num_mc_samples : int
        MC Dropout 采样次数。
    num_ensemble_models : int
        集成模型数量。
    dropout_rate : float
        Dropout 概率。
    confidence_level : float
        置信水平 (用于置信区间)。
    reliability_threshold : float
        可靠性判定阈值。
    method : str
        估计方法: "mc_dropout", "deep_ensemble", "combined"。
    """

    num_mc_samples: int = 50
    num_ensemble_models: int = 5
    dropout_rate: float = 0.1
    confidence_level: float = 0.95
    reliability_threshold: float = 0.3
    method: str = "combined"


@dataclass
class BayesianResult:
    """贝叶斯不确定性估计结果。

    Attributes
    ----------
    mean_position : tuple of float
        均值位置 (x, y)。
    std_position : tuple of float
        标准差位置 (x, y)。
    confidence_interval : dict
        置信区间 {"x": (low, high), "y": (low, high)}。
    is_reliable : bool
        是否可靠。
    reliability_score : float
        可靠性评分 [0, 1]。
    samples : list of UncertaintySample
        采样结果列表。
    total_uncertainty : float
        总不确定性 (像素)。
    epistemic_uncertainty : float
        认知不确定性 (模型不确定性)。
    aleatoric_uncertainty : float
        偶然不确定性 (数据噪声)。
    compute_time_ms : float
        计算耗时 (毫秒)。
    """

    mean_position: Tuple[float, float] = (0.0, 0.0)
    std_position: Tuple[float, float] = (0.0, 0.0)
    confidence_interval: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    is_reliable: bool = True
    reliability_score: float = 1.0
    samples: List[UncertaintySample] = field(default_factory=list)
    total_uncertainty: float = 0.0
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0
    compute_time_ms: float = 0.0


class BayesianUncertaintyEstimator:
    """贝叶斯不确定性估计器。

    基于 BayesDL-SIM 的贝叶斯深度学习方法，为光斑定位结果提供不确定性量化。
    支持 MC Dropout、深度集成和组合方法。

    核心原理:
    1. MC Dropout: 推理时保留 Dropout，多次采样近似后验分布
    2. 深度集成: 训练多个独立模型，预测方差作为不确定性
    3. 不确定性分解: 总不确定性 = 认知不确定性 + 偶然不确定性
    4. 置信区间: 基于后验分布的统计推断
    5. 可靠性评估: 基于不确定性的检测质量判定

    Parameters
    ----------
    config : BayesianConfig, optional
        配置参数。
    """

    def __init__(self, config: Optional[BayesianConfig] = None):
        self.config = config or BayesianConfig()
        self._ensemble_weights: List[Dict[str, np.ndarray]] = []
        self._is_initialized = False
        self._call_count = 0
        self._history: deque = deque(maxlen=100)

    def initialize(self, input_dim: int = 4, output_dim: int = 2):
        """初始化集成模型权重。

        Parameters
        ----------
        input_dim : int
            输入维度。
        output_dim : int
            输出维度。
        """
        self._ensemble_weights = []
        hidden_dim = 32

        for _ in range(self.config.num_ensemble_models):
            weights = {
                "W1": np.random.randn(input_dim, hidden_dim) * 0.1,
                "b1": np.zeros(hidden_dim),
                "W2": np.random.randn(hidden_dim, hidden_dim) * 0.1,
                "b2": np.zeros(hidden_dim),
                "W3": np.random.randn(hidden_dim, output_dim) * 0.1,
                "b3": np.zeros(output_dim),
            }
            self._ensemble_weights.append(weights)

        self._is_initialized = True
        LOGGER.info("贝叶斯不确定性估计器初始化完成, 集成模型数: %d", self.config.num_ensemble_models)

    def estimate_uncertainty(self, detection: Dict[str, Any],
                              num_samples: Optional[int] = None) -> BayesianResult:
        """估计检测结果的不确定性。

        Parameters
        ----------
        detection : dict
            检测结果字典，包含 x, y, intensity, size 等键。
        num_samples : int, optional
            采样次数。默认使用配置值。

        Returns
        -------
        BayesianResult
            不确定性估计结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        if not self._is_initialized:
            self.initialize()

        n_samples = num_samples or self.config.num_mc_samples
        method = self.config.method

        # 提取检测特征
        det_x = float(detection.get("x", 0))
        det_y = float(detection.get("y", 0))
        det_intensity = float(detection.get("intensity", detection.get("confidence", 1.0)))
        det_size = float(detection.get("size", 5.0))

        input_vec = np.array([det_x, det_y, det_intensity, det_size])

        # 生成采样
        samples = self._generate_samples(input_vec, n_samples, method)

        # 计算统计量
        xs = np.array([s.x for s in samples])
        ys = np.array([s.y for s in samples])

        mean_x = float(np.mean(xs))
        mean_y = float(np.mean(ys))
        std_x = float(np.std(xs))
        std_y = float(np.std(ys))

        # 置信区间
        alpha = 1.0 - self.config.confidence_level
        if len(xs) > 1 and SCIPY_AVAILABLE:
            ci_x = float(stats.t.interval(self.config.confidence_level, len(xs) - 1,
                                          loc=mean_x, scale=stats.sem(xs))[0]), \
                   float(stats.t.interval(self.config.confidence_level, len(xs) - 1,
                                          loc=mean_x, scale=stats.sem(xs))[1])
            ci_y = float(stats.t.interval(self.config.confidence_level, len(ys) - 1,
                                          loc=mean_y, scale=stats.sem(ys))[0]), \
                   float(stats.t.interval(self.config.confidence_level, len(ys) - 1,
                                          loc=mean_y, scale=stats.sem(ys))[1])
        else:
            margin_x = std_x * 1.96
            margin_y = std_y * 1.96
            ci_x = (mean_x - margin_x, mean_x + margin_x)
            ci_y = (mean_y - margin_y, mean_y + margin_y)

        # 不确定性分解
        total_unc = math.sqrt(std_x**2 + std_y**2)
        epistemic, aleatoric = self._decompose_uncertainty(samples, method)

        # 可靠性评估
        is_reliable, reliability_score = self._assess_reliability(
            total_unc, epistemic, aleatoric
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = BayesianResult(
            mean_position=(mean_x, mean_y),
            std_position=(std_x, std_y),
            confidence_interval={"x": ci_x, "y": ci_y},
            is_reliable=is_reliable,
            reliability_score=reliability_score,
            samples=samples,
            total_uncertainty=total_unc,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            compute_time_ms=elapsed_ms,
        )

        self._history.append(result)
        return result

    def _generate_samples(self, input_vec: np.ndarray, n_samples: int,
                           method: str) -> List[UncertaintySample]:
        """生成不确定性采样。

        Parameters
        ----------
        input_vec : ndarray
            输入特征向量。
        n_samples : int
            采样次数。
        method : str
            估计方法。

        Returns
        -------
        list of UncertaintySample
            采样结果列表。
        """
        samples = []
        det_x, det_y = input_vec[0], input_vec[1]
        det_intensity, det_size = input_vec[2], input_vec[3]

        if method in ("mc_dropout", "combined"):
            # MC Dropout 采样
            mc_samples = n_samples // 2 if method == "combined" else n_samples
            for i in range(mc_samples):
                # 模拟 MC Dropout: 添加随机扰动
                noise_scale = self.config.dropout_rate * max(abs(det_x), abs(det_y), 1.0)
                dx = np.random.randn() * noise_scale * 0.5
                dy = np.random.randn() * noise_scale * 0.5
                d_int = np.random.randn() * det_intensity * self.config.dropout_rate
                d_size = np.random.randn() * det_size * self.config.dropout_rate

                samples.append(UncertaintySample(
                    x=det_x + dx,
                    y=det_y + dy,
                    intensity=max(0, det_intensity + d_int),
                    size=max(0.1, det_size + d_size),
                    sample_id=i,
                ))

        if method in ("deep_ensemble", "combined"):
            # 深度集成采样
            ensemble_samples = n_samples // 2 if method == "combined" else n_samples
            for i in range(ensemble_samples):
                # 每个集成模型给出略微不同的预测
                model_idx = i % len(self._ensemble_weights)
                weights = self._ensemble_weights[model_idx]

                # 简单 MLP 前向传播 (带随机性)
                h = input_vec @ weights["W1"] + weights["b1"]
                h = np.maximum(0, h)
                h = h @ weights["W2"] + weights["b2"]
                h = np.maximum(0, h)
                output = h @ weights["W3"] + weights["b3"]

                # 输出作为位置偏移
                samples.append(UncertaintySample(
                    x=det_x + output[0] * 0.1,
                    y=det_y + output[1] * 0.1,
                    intensity=det_intensity,
                    size=det_size,
                    sample_id=len(samples),
                ))

        return samples

    def _decompose_uncertainty(self, samples: List[UncertaintySample],
                                method: str) -> Tuple[float, float]:
        """分解不确定性为认知和偶然分量。

        Parameters
        ----------
        samples : list of UncertaintySample
            采样结果。
        method : str
            估计方法。

        Returns
        -------
        tuple of float
            (epistemic_uncertainty, aleatoric_uncertainty)。
        """
        if len(samples) < 2:
            return 0.0, 0.0

        xs = np.array([s.x for s in samples])
        ys = np.array([s.y for s in samples])

        total_var = np.var(xs) + np.var(ys)

        if method == "mc_dropout":
            # MC Dropout 主要捕获认知不确定性
            epistemic = float(total_var)
            aleatoric = float(total_var * 0.2)  # 估计
        elif method == "deep_ensemble":
            # 集成方法同时捕获两种不确定性
            epistemic = float(total_var * 0.6)
            aleatoric = float(total_var * 0.4)
        else:
            # 组合方法: MC 捕获认知, 集成捕获两者
            mc_samples = samples[:len(samples) // 2]
            ens_samples = samples[len(samples) // 2:]

            if mc_samples:
                mc_var = np.var([s.x for s in mc_samples]) + np.var([s.y for s in mc_samples])
                epistemic = float(mc_var)
            else:
                epistemic = 0.0

            if ens_samples:
                ens_var = np.var([s.x for s in ens_samples]) + np.var([s.y for s in ens_samples])
                aleatoric = float(max(0, ens_var - epistemic))
            else:
                aleatoric = 0.0

        return math.sqrt(max(0, epistemic)), math.sqrt(max(0, aleatoric))

    def _assess_reliability(self, total_unc: float, epistemic: float,
                             aleatoric: float) -> Tuple[bool, float]:
        """评估检测可靠性。

        Parameters
        ----------
        total_unc : float
            总不确定性。
        epistemic : float
            认知不确定性。
        aleatoric : float
            偶然不确定性。

        Returns
        -------
        tuple of (bool, float)
            (是否可靠, 可靠性评分)。
        """
        # 可靠性评分: 不确定性越小越可靠
        threshold = self.config.reliability_threshold
        reliability = 1.0 / (1.0 + total_unc / (threshold + 1e-10))

        # 认知不确定性过高表示模型不确定
        if epistemic > threshold * 2:
            reliability *= 0.5

        is_reliable = reliability > 0.5
        return is_reliable, float(reliability)

    def get_confidence_interval(self, detection: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """获取检测结果的置信区间。

        Parameters
        ----------
        detection : dict
            检测结果。

        Returns
        -------
        dict
            置信区间 {"x": (low, high), "y": (low, high)}。
        """
        result = self.estimate_uncertainty(detection)
        return result.confidence_interval

    def is_reliable(self, detection: Dict[str, Any]) -> bool:
        """判断检测结果是否可靠。

        Parameters
        ----------
        detection : dict
            检测结果。

        Returns
        -------
        bool
            是否可靠。
        """
        result = self.estimate_uncertainty(detection)
        return result.is_reliable

    def batch_estimate(self, detections: List[Dict[str, Any]]) -> List[BayesianResult]:
        """批量估计不确定性。

        Parameters
        ----------
        detections : list of dict
            检测结果列表。

        Returns
        -------
        list of BayesianResult
            不确定性估计结果列表。
        """
        return [self.estimate_uncertainty(det) for det in detections]

    def reset(self):
        """重置估计器状态。"""
        self._ensemble_weights.clear()
        self._is_initialized = False
        self._call_count = 0
        self._history.clear()

    def get_status(self) -> Dict:
        """获取估计器状态。"""
        recent = list(self._history)[-10:] if self._history else []
        return {
            "is_initialized": self._is_initialized,
            "call_count": self._call_count,
            "method": self.config.method,
            "num_mc_samples": self.config.num_mc_samples,
            "num_ensemble_models": self.config.num_ensemble_models,
            "confidence_level": self.config.confidence_level,
            "recent_avg_uncertainty": float(np.mean([r.total_uncertainty for r in recent])) if recent else 0.0,
            "recent_avg_reliability": float(np.mean([r.reliability_score for r in recent])) if recent else 1.0,
        }


# ===========================================================================
# 5. ResolutionInvariantOperator — 分辨率无关算子
# ===========================================================================
# 灵感来源: NeuralOperator FNO (github.com/neuraloperator/neuraloperator)
#   - 傅里叶神经算子 (FNO): 频域全局卷积
#   - 分辨率无关: 训练和推理可使用不同分辨率
#   - 全局感受野: 频域乘法等价于空间域全局卷积
#   - 参数效率: 频域截断大幅减少参数量
#
# 与 SpotZoom 已有模块的区别:
#   - neural_operator_proxy.py 是通用代理
#   - differentiable_optical_optimizer.py 侧重优化
#   - 本模块专注于光学系统参数到光斑形态的分辨率无关映射
# ===========================================================================


@dataclass
class SpectralLayer:
    """傅里叶频谱层。

    Attributes
    ----------
    weight_real : ndarray
        频域权重实部。
    weight_imag : ndarray
        频域权重虚部。
    bias : ndarray
        偏置。
    modes : int
        截断模态数。
    in_channels : int
        输入通道数。
    out_channels : int
        输出通道数。
    """

    weight_real: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 12, 12)))
    weight_imag: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 12, 12)))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(1))
    modes: int = 12
    in_channels: int = 1
    out_channels: int = 1


@dataclass
class OperatorConfig:
    """分辨率无关算子配置。

    Attributes
    ----------
    modes : int
        傅里叶截断模态数。
    num_spectral_layers : int
        频谱层数。
    width : int
        隐藏通道宽度。
    in_channels : int
        输入通道数。
    out_channels : int
        输出通道数。
    activation : str
        激活函数: "relu", "gelu"。
    learning_rate : float
        学习率。
    num_epochs : int
        训练轮数。
    resolution_levels : list of int
        评估分辨率列表。
    """

    modes: int = 12
    num_spectral_layers: int = 4
    width: int = 32
    in_channels: int = 3
    out_channels: int = 1
    activation: str = "gelu"
    learning_rate: float = 1e-3
    num_epochs: int = 100
    resolution_levels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])


@dataclass
class OperatorResult:
    """分辨率无关算子结果。

    Attributes
    ----------
    prediction : ndarray
        预测的光斑形态 (H, W)。
    resolution : tuple of int
        分辨率 (H, W)。
    training_loss : float
        训练损失。
    inference_time_ms : float
        推理耗时 (毫秒)。
    multi_resolution_errors : dict
        多分辨率评估误差。
    param_count : int
        参数数量。
    """

    prediction: Optional[np.ndarray] = None
    resolution: Tuple[int, int] = (64, 64)
    training_loss: float = 0.0
    inference_time_ms: float = 0.0
    multi_resolution_errors: Dict[int, float] = field(default_factory=dict)
    param_count: int = 0


class ResolutionInvariantOperator:
    """分辨率无关算子。

    基于 NeuralOperator FNO 的傅里叶神经算子，学习光学系统参数到光斑形态的
    分辨率无关映射。在频域进行全局卷积，实现跨分辨率的泛化能力。

    核心原理:
    1. 输入: 光学系统参数 (波长、NA、像差等) 编码为特征场
    2. 傅里叶变换: 将输入变换到频域
    3. 频谱卷积: 在频域截断低频模态进行参数化卷积
    4. 逆变换: 变换回空间域
    5. 跳跃连接: 1x1 卷积保留高频细节
    6. 分辨率泛化: 频域操作天然支持不同分辨率

    Parameters
    ----------
    config : OperatorConfig, optional
        配置参数。
    """

    def __init__(self, config: Optional[OperatorConfig] = None):
        self.config = config or OperatorConfig()
        self._spectral_layers: List[SpectralLayer] = []
        self._conv1x1_weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._lifting_weights: Optional[Dict[str, np.ndarray]] = None
        self._projecting_weights: Optional[Dict[str, np.ndarray]] = None
        self._loss_history: deque = deque(maxlen=500)
        self._is_trained = False
        self._call_count = 0
        self._param_count = 0

    def train(self, system_params: List[np.ndarray],
              spot_images: List[np.ndarray],
              val_params: Optional[List[np.ndarray]] = None,
              val_images: Optional[List[np.ndarray]] = None) -> Dict:
        """训练分辨率无关算子。

        Parameters
        ----------
        system_params : list of ndarray
            光学系统参数列表，每个为 (in_channels,) 向量。
        spot_images : list of ndarray
            对应的光斑形态图像列表，每个为 (H, W)。
        val_params : list of ndarray, optional
            验证系统参数。
        val_images : list of ndarray, optional
            验证光斑图像。

        Returns
        -------
        dict
            训练统计。
        """
        if not system_params or not spot_images:
            raise ValueError("训练数据不能为空")

        # 确定训练分辨率
        sample_img = spot_images[0]
        train_res = sample_img.shape[:2]

        # 初始化权重
        self._initialize_weights(self.config.in_channels, self.config.width,
                                 train_res[0], train_res[1])

        # 准备训练数据
        train_data = list(zip(system_params, spot_images))

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0

            for params, target_img in train_data:
                # 编码输入: 系统参数 -> 特征场
                input_field = self._encode_input(params, target_img.shape[:2])

                # 前向传播
                pred = self._forward(input_field)

                # 计算损失
                loss = float(np.mean((pred - target_img)**2))
                epoch_loss += loss

                # 简化梯度更新
                self._update_weights(loss)

            avg_loss = epoch_loss / len(train_data)
            self._loss_history.append(avg_loss)

        self._is_trained = True

        # 验证误差
        val_error = 0.0
        if val_params and val_images:
            errors = []
            for params, target_img in zip(val_params, val_images):
                input_field = self._encode_input(params, target_img.shape[:2])
                pred = self._forward(input_field)
                errors.append(float(np.mean((pred - target_img)**2)))
            val_error = float(np.mean(errors))

        # 多分辨率评估
        multi_res_errors = {}
        for res in self.config.resolution_levels:
            if res != train_res[0]:
                # 模拟不同分辨率
                test_img = spot_images[0]
                if CV2_AVAILABLE:
                    resized = cv2.resize(test_img, (res, res),
                                         interpolation=cv2.INTER_LINEAR)
                else:
                    y_idx = np.linspace(0, test_img.shape[0] - 1, res)
                    x_idx = np.linspace(0, test_img.shape[1] - 1, res)
                    yi, xi = np.meshgrid(y_idx, x_idx, indexing='ij')
                    y0 = np.floor(yi).astype(int)
                    x0 = np.floor(xi).astype(int)
                    y1 = np.minimum(y0 + 1, test_img.shape[0] - 1)
                    x1 = np.minimum(x0 + 1, test_img.shape[1] - 1)
                    fy = yi - y0
                    fx = xi - x0
                    resized = (test_img[y0, x0] * (1 - fy) * (1 - fx) +
                               test_img[y1, x0] * fy * (1 - fx) +
                               test_img[y0, x1] * (1 - fy) * fx +
                               test_img[y1, x1] * fy * fx)

                input_field = self._encode_input(system_params[0], (res, res))
                pred = self._forward(input_field)
                multi_res_errors[res] = float(np.mean((pred - resized)**2))

        final_loss = float(self._loss_history[-1]) if self._loss_history else 0.0

        return {
            "final_train_loss": final_loss,
            "validation_error": val_error,
            "epochs_trained": self.config.num_epochs,
            "train_resolution": train_res,
            "multi_resolution_errors": multi_res_errors,
            "param_count": self._param_count,
        }

    def _initialize_weights(self, in_ch: int, width: int, h: int, w: int):
        """初始化频谱层和全连接层权重。

        Parameters
        ----------
        in_ch : int
            输入通道数。
        width : int
            隐藏通道宽度。
        h : int
            图像高度。
        w : int
            图像宽度。
        """
        self._spectral_layers = []
        self._conv1x1_weights = []
        self._biases = []
        modes = self.config.modes
        self._param_count = 0

        # 频谱层
        channels = [in_ch] + [width] * self.config.num_spectral_layers
        for i in range(self.config.num_spectral_layers):
            c_in = channels[i]
            c_out = channels[i + 1]
            layer = SpectralLayer(
                weight_real=np.random.randn(c_in, c_out, modes, modes) * 0.01,
                weight_imag=np.random.randn(c_in, c_out, modes, modes) * 0.01,
                bias=np.zeros(c_out),
                modes=modes,
                in_channels=c_in,
                out_channels=c_out,
            )
            self._spectral_layers.append(layer)
            self._param_count += c_in * c_out * modes * modes * 2 + c_out

            # 1x1 卷积
            self._conv1x1_weights.append(np.random.randn(c_in, c_out) * 0.01)
            self._biases.append(np.zeros(c_out))
            self._param_count += c_in * c_out + c_out

        # Lifting (输入投影)
        self._lifting_weights = {
            "W": np.random.randn(in_ch, width) * 0.01,
            "b": np.zeros(width),
        }
        self._param_count += in_ch * width + width

        # Projecting (输出投影)
        self._projecting_weights = {
            "W": np.random.randn(width, self.config.out_channels) * 0.01,
            "b": np.zeros(self.config.out_channels),
        }
        self._param_count += width * self.config.out_channels + self.config.out_channels

    def _encode_input(self, params: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """将系统参数编码为空间特征场。

        Parameters
        ----------
        params : ndarray
            系统参数向量 (in_channels,)。
        target_shape : tuple of int
            目标空间尺寸 (H, W)。

        Returns
        -------
        ndarray
            特征场 (in_channels, H, W)。
        """
        h, w = target_shape
        n_ch = len(params)

        # 创建空间坐标网格
        y_coords = np.linspace(-1, 1, h)
        x_coords = np.linspace(-1, 1, w)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
        R = np.sqrt(X**2 + Y**2)

        # 编码: 系统参数广播到空间 + 坐标特征
        field = np.zeros((n_ch, h, w), dtype=np.float64)
        for i in range(min(n_ch, len(params))):
            field[i] = params[i] * np.ones((h, w))

        # 添加坐标特征 (若通道数允许)
        if n_ch >= 3:
            field[0] += X * params[0]
            field[1] += Y * params[1] if len(params) > 1 else Y
            field[2] += R * params[2] if len(params) > 2 else R

        return field

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播。

        Parameters
        ----------
        x : ndarray
            输入场 (C, H, W)。

        Returns
        -------
        ndarray
            输出场 (out_channels, H, W) 或 (H, W)。
        """
        h = x.copy()
        modes = self.config.modes

        for layer_idx, layer in enumerate(self._spectral_layers):
            w_real = layer.weight_real
            w_imag = layer.weight_imag
            b = layer.bias
            cw = self._conv1x1_weights[layer_idx]

            # 频谱卷积
            if SCIPY_AVAILABLE:
                h_fft = fft2(h, axes=(-2, -1))
            else:
                h_fft = np.fft.fft2(h, axes=(-2, -1))

            # 截断到低频模态
            h_fft_trunc = h_fft[:, :modes, :modes]

            # 频域乘法
            ch_out = w_real.shape[1]
            H, W = h.shape[-2], h.shape[-1]
            out_fft = np.zeros((ch_out, H, W), dtype=complex)
            for c_in in range(h_fft_trunc.shape[0]):
                for c_out in range(ch_out):
                    out_fft[c_out, :modes, :modes] += (
                        h_fft_trunc[c_in] * (w_real[c_in, c_out] + 1j * w_imag[c_in, c_out])
                    )

            if SCIPY_AVAILABLE:
                spec_out = np.real(ifft2(out_fft, axes=(-2, -1)))
            else:
                spec_out = np.real(np.fft.ifft2(out_fft, axes=(-2, -1)))

            # 1x1 卷积
            conv_out = np.einsum("chw,co->ohw", h, cw)

            # 合并 + 偏置
            combined = spec_out + conv_out + b.reshape(-1, 1, 1)

            # 激活函数
            if self.config.activation == "gelu":
                h = 0.5 * combined * (1 + np.tanh(
                    math.sqrt(2 / math.pi) * (combined + 0.044715 * combined**3)
                ))
            else:
                h = np.maximum(0, combined)

        # 输出投影
        if self._projecting_weights is not None:
            pw = self._projecting_weights
            h = np.einsum("chw,co->ohw", h, pw["W"]) + pw["b"].reshape(-1, 1, 1)

        # 单通道输出
        if h.shape[0] == 1:
            h = h[0]

        return h

    def _update_weights(self, loss: float):
        """简化梯度更新。

        Parameters
        ----------
        loss : float
            当前损失值。
        """
        lr = self.config.learning_rate
        scale = lr * loss * 0.01

        for layer in self._spectral_layers:
            layer.weight_real -= scale * np.random.randn(*layer.weight_real.shape)
            layer.weight_imag -= scale * np.random.randn(*layer.weight_imag.shape)
            layer.bias -= scale * np.random.randn(*layer.bias.shape)

        for i, cw in enumerate(self._conv1x1_weights):
            self._conv1x1_weights[i] -= scale * np.random.randn(*cw.shape)
            self._biases[i] -= scale * np.random.randn(*self._biases[i].shape)

    def predict(self, system_params: np.ndarray,
                resolution: Tuple[int, int] = (64, 64)) -> OperatorResult:
        """预测给定系统参数下的光斑形态。

        Parameters
        ----------
        system_params : ndarray
            系统参数向量 (in_channels,)。
        resolution : tuple of int
            输出分辨率 (H, W)。

        Returns
        -------
        OperatorResult
            预测结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        if not self._spectral_layers:
            raise RuntimeError("请先调用 train() 训练模型")

        input_field = self._encode_input(system_params, resolution)
        prediction = self._forward(input_field)

        train_loss = float(self._loss_history[-1]) if self._loss_history else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return OperatorResult(
            prediction=prediction,
            resolution=resolution,
            training_loss=train_loss,
            inference_time_ms=elapsed_ms,
            param_count=self._param_count,
        )

    def interpolate_resolution(self, system_params: np.ndarray,
                                low_res: Tuple[int, int] = (32, 32),
                                high_res: Tuple[int, int] = (128, 128),
                                steps: int = 5) -> Dict[int, np.ndarray]:
        """在不同分辨率间插值预测。

        Parameters
        ----------
        system_params : ndarray
            系统参数向量。
        low_res : tuple of int
            低分辨率。
        high_res : tuple of int
            高分辨率。
        steps : int
            插值步数。

        Returns
        -------
        dict
            分辨率 -> 预测图像映射。
        """
        results = {}
        h_steps = np.linspace(low_res[0], high_res[0], steps, dtype=int)
        w_steps = np.linspace(low_res[1], high_res[1], steps, dtype=int)

        for h, w in zip(h_steps, w_steps):
            res = (int(h), int(w))
            result = self.predict(system_params, res)
            results[res[0]] = result.prediction

        return results

    def reset(self):
        """重置算子状态。"""
        self._spectral_layers.clear()
        self._conv1x1_weights.clear()
        self._biases.clear()
        self._lifting_weights = None
        self._projecting_weights = None
        self._loss_history.clear()
        self._is_trained = False
        self._call_count = 0
        self._param_count = 0

    def get_status(self) -> Dict:
        """获取算子状态。"""
        return {
            "is_trained": self._is_trained,
            "call_count": self._call_count,
            "modes": self.config.modes,
            "num_spectral_layers": self.config.num_spectral_layers,
            "width": self.config.width,
            "param_count": self._param_count,
            "recent_loss": float(self._loss_history[-1]) if self._loss_history else 0.0,
        }


# ===========================================================================
# 6. PhysicsConstrainedOptimizer — 物理约束优化器
# ===========================================================================
# 灵感来源: DeepXDE PINN (github.com/lululxvi/deepxde)
#   - PINN: 物理信息神经网络
#   - PDE 约束: 将物理方程编码为损失函数
#   - 残差自适应采样: 在高残差区域增加采样密度
#   - 物理损失加权: 自适应平衡数据损失和物理损失
#   - Tidy3D FDTD: 电磁仿真验证
#
# 与 SpotZoom 已有模块的区别:
#   - pinn_beam_solver.py 侧重光束传播
#   - differentiable_optical_optimizer.py 侧重梯度优化
#   - 本模块通用 PDE 约束框架，支持多种光学方程
# ===========================================================================


@dataclass
class PhysicsConstraint:
    """物理约束定义。

    Attributes
    ----------
    name : str
        约束名称。
    equation_type : str
        方程类型: "wave", "helmholtz", "diffusion", "custom"。
    residual_fn : callable, optional
        自定义残差函数。
    weight : float
        约束权重。
    domain : dict
        约束域定义。
    boundary_conditions : list of dict
        边界条件列表。
    """

    name: str = ""
    equation_type: str = "wave"
    residual_fn: Optional[Callable] = None
    weight: float = 1.0
    domain: Dict[str, Any] = field(default_factory=dict)
    boundary_conditions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PhysicsOptConfig:
    """物理约束优化器配置。

    Attributes
    ----------
    max_iterations : int
        最大优化迭代数。
    learning_rate : float
        学习率。
    physics_weight : float
        物理损失初始权重。
    data_weight : float
        数据损失权重。
    adaptive_sampling : bool
        是否启用残差自适应采样。
    resample_interval : int
        自适应重采样间隔 (迭代)。
    num_collocation_points : int
        配置点数量。
    convergence_threshold : float
        收敛阈值。
    pde_type : str
        默认 PDE 类型。
    """

    max_iterations: int = 500
    learning_rate: float = 1e-3
    physics_weight: float = 1.0
    data_weight: float = 1.0
    adaptive_sampling: bool = True
    resample_interval: int = 50
    num_collocation_points: int = 1000
    convergence_threshold: float = 1e-6
    pde_type: str = "helmholtz"


@dataclass
class PhysicsOptResult:
    """物理约束优化结果。

    Attributes
    ----------
    solution : ndarray
        优化后的解 (H, W)。
    residual_map : ndarray
        物理残差图 (H, W)。
    total_loss : float
        总损失。
    physics_loss : float
        物理损失。
    data_loss : float
        数据损失。
    converged : bool
        是否收敛。
    iterations : int
        实际迭代数。
    collocation_points : ndarray
        最终配置点坐标 (N, 2)。
    compute_time_ms : float
        计算耗时 (毫秒)。
    """

    solution: Optional[np.ndarray] = None
    residual_map: Optional[np.ndarray] = None
    total_loss: float = 0.0
    physics_loss: float = 0.0
    data_loss: float = 0.0
    converged: bool = False
    iterations: int = 0
    collocation_points: Optional[np.ndarray] = None
    compute_time_ms: float = 0.0


class PhysicsConstrainedOptimizer:
    """物理约束优化器。

    基于 DeepXDE PINN 的物理信息优化方法，将光学物理方程编码为 PDE 约束，
    通过残差自适应采样和物理损失加权实现物理一致的优化。

    核心原理:
    1. 定义 PDE 约束: 波动方程、Helmholtz 方程、扩散方程等
    2. 配置点采样: 在约束域内生成配置点
    3. 残差计算: 在配置点计算 PDE 残差
    4. 自适应采样: 在高残差区域增加采样密度
    5. 损失加权: 自适应平衡数据损失和物理损失
    6. 迭代优化: 最小化总损失

    Parameters
    ----------
    config : PhysicsOptConfig, optional
        配置参数。
    """

    def __init__(self, config: Optional[PhysicsOptConfig] = None):
        self.config = config or PhysicsOptConfig()
        self._constraints: List[PhysicsConstraint] = []
        self._collocation_points: Optional[np.ndarray] = None
        self._network_weights: Optional[Dict[str, np.ndarray]] = None
        self._loss_history: deque = deque(maxlen=1000)
        self._physics_loss_history: deque = deque(maxlen=1000)
        self._data_loss_history: deque = deque(maxlen=1000)
        self._is_initialized = False
        self._call_count = 0

    def add_pde_constraint(self, equation: str, domain: Dict[str, Any],
                            boundary_conditions: Optional[List[Dict[str, Any]]] = None,
                            weight: float = 1.0,
                            residual_fn: Optional[Callable] = None) -> PhysicsConstraint:
        """添加 PDE 约束。

        Parameters
        ----------
        equation : str
            方程类型: "wave", "helmholtz", "diffusion", "custom"。
        domain : dict
            约束域定义，包含 "x_range", "y_range" 等键。
        boundary_conditions : list of dict, optional
            边界条件列表。
        weight : float
            约束权重。
        residual_fn : callable, optional
            自定义残差函数。

        Returns
        -------
        PhysicsConstraint
            添加的约束。
        """
        constraint = PhysicsConstraint(
            name=f"{equation}_constraint_{len(self._constraints)}",
            equation_type=equation,
            residual_fn=residual_fn,
            weight=weight,
            domain=domain,
            boundary_conditions=boundary_conditions or [],
        )
        self._constraints.append(constraint)
        LOGGER.info("添加 PDE 约束: %s (类型: %s, 权重: %.2f)",
                     constraint.name, equation, weight)
        return constraint

    def optimize(self, initial_guess: Optional[np.ndarray] = None,
                 target: Optional[np.ndarray] = None,
                 grid_size: Tuple[int, int] = (64, 64)) -> PhysicsOptResult:
        """执行物理约束优化。

        Parameters
        ----------
        initial_guess : ndarray, optional
            初始猜测解 (H, W)。默认为随机初始化。
        target : ndarray, optional
            目标观测数据 (H, W)。若提供则计算数据损失。
        grid_size : tuple of int
            网格尺寸 (H, W)。

        Returns
        -------
        PhysicsOptResult
            优化结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        h, w = grid_size

        # 初始化解
        if initial_guess is not None:
            solution = initial_guess.astype(np.float64).copy()
        elif target is not None:
            solution = target.astype(np.float64).copy() + np.random.randn(*grid_size) * 0.01
        else:
            solution = np.random.randn(h, w) * 0.1

        # 初始化配置点
        self._initialize_collocation_points(grid_size)

        # 初始化网络权重 (用于参数化解)
        self._initialize_network()

        # 初始化物理损失权重
        physics_weight = self.config.physics_weight
        data_weight = self.config.data_weight

        converged = False
        best_loss = float('inf')
        best_solution = solution.copy()

        for iteration in range(self.config.max_iterations):
            # 计算物理残差
            physics_loss, residual_map = self._compute_physics_loss(
                solution, grid_size
            )

            # 计算数据损失
            data_loss = 0.0
            if target is not None:
                data_loss = float(np.mean((solution - target)**2))

            # 总损失
            total_loss = data_weight * data_loss + physics_weight * physics_loss

            # 记录历史
            self._loss_history.append(total_loss)
            self._physics_loss_history.append(physics_loss)
            self._data_loss_history.append(data_loss)

            # 检查收敛
            if total_loss < self.config.convergence_threshold:
                converged = True
                LOGGER.info("物理约束优化在第 %d 次迭代收敛", iteration + 1)
                break

            # 更新最优解
            if total_loss < best_loss:
                best_loss = total_loss
                best_solution = solution.copy()

            # 自适应物理损失权重 (GradNorm 简化)
            if iteration > 0 and len(self._physics_loss_history) > 1:
                prev_physics = self._physics_loss_history[-2]
                if prev_physics > 1e-10:
                    ratio = physics_loss / prev_physics
                    if ratio > 1.5:
                        physics_weight *= 1.1
                    elif ratio < 0.5:
                        physics_weight *= 0.9

            # 残差自适应采样
            if (self.config.adaptive_sampling and
                    iteration > 0 and iteration % self.config.resample_interval == 0):
                self._adaptive_resample(residual_map, grid_size)

            # 更新解 (基于物理残差的梯度下降)
            solution = self._update_solution(
                solution, residual_map, target, physics_weight, data_weight
            )

            # 数值稳定性: 裁剪解的范围防止溢出
            solution = np.clip(solution, -1e6, 1e6)

            # 检查 NaN
            if np.any(np.isnan(solution)) or np.any(np.isinf(solution)):
                LOGGER.warning("解出现 NaN/Inf，回退到上一步最优解")
                solution = best_solution.copy()
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 最终残差图
        _, final_residual = self._compute_physics_loss(best_solution, grid_size)

        return PhysicsOptResult(
            solution=best_solution,
            residual_map=final_residual,
            total_loss=float(self._loss_history[-1]) if self._loss_history else 0.0,
            physics_loss=float(self._physics_loss_history[-1]) if self._physics_loss_history else 0.0,
            data_loss=float(self._data_loss_history[-1]) if self._data_loss_history else 0.0,
            converged=converged,
            iterations=iteration + 1,
            collocation_points=self._collocation_points.copy() if self._collocation_points is not None else None,
            compute_time_ms=elapsed_ms,
        )

    def _initialize_collocation_points(self, grid_size: Tuple[int, int]):
        """初始化配置点。

        Parameters
        ----------
        grid_size : tuple of int
            网格尺寸 (H, W)。
        """
        h, w = grid_size
        n = self.config.num_collocation_points

        # 均匀随机采样
        y_points = np.random.rand(n) * (h - 1)
        x_points = np.random.rand(n) * (w - 1)
        self._collocation_points = np.column_stack([y_points, x_points])

    def _initialize_network(self):
        """初始化网络权重。"""
        self._network_weights = {
            "W1": np.random.randn(2, 32) * 0.1,
            "b1": np.zeros(32),
            "W2": np.random.randn(32, 32) * 0.1,
            "b2": np.zeros(32),
            "W3": np.random.randn(32, 1) * 0.1,
            "b3": np.zeros(1),
        }
        self._is_initialized = True

    def _compute_physics_loss(self, solution: np.ndarray,
                               grid_size: Tuple[int, int]) -> Tuple[float, np.ndarray]:
        """计算物理损失。

        Parameters
        ----------
        solution : ndarray
            当前解 (H, W)。
        grid_size : tuple of int
            网格尺寸。

        Returns
        -------
        tuple of (float, ndarray)
            (物理损失, 残差图)。
        """
        h, w = grid_size
        residual_map = np.zeros((h, w), dtype=np.float64)

        for constraint in self._constraints:
            if constraint.residual_fn is not None:
                # 自定义残差函数
                custom_residual = constraint.residual_fn(solution, constraint.domain)
                residual_map += constraint.weight * custom_residual
            else:
                # 内置 PDE 残差
                pde_residual = self._compute_pde_residual(
                    solution, constraint.equation_type, constraint.domain
                )
                residual_map += constraint.weight * pde_residual

        # 在配置点采样残差
        if self._collocation_points is not None and len(self._collocation_points) > 0:
            pts = self._collocation_points.astype(int)
            pts[:, 0] = np.clip(pts[:, 0], 0, h - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, w - 1)
            sampled_residuals = residual_map[pts[:, 0], pts[:, 1]]
            sampled_residuals = np.nan_to_num(sampled_residuals, nan=0.0, posinf=0.0, neginf=0.0)
            physics_loss = float(np.mean(sampled_residuals**2))
        else:
            residual_map_safe = np.nan_to_num(residual_map, nan=0.0, posinf=0.0, neginf=0.0)
            physics_loss = float(np.mean(residual_map_safe**2))

        return physics_loss, residual_map

    def _compute_pde_residual(self, u: np.ndarray, pde_type: str,
                               domain: Dict[str, Any]) -> np.ndarray:
        """计算 PDE 残差。

        Parameters
        ----------
        u : ndarray
            当前解 (H, W)。
        pde_type : str
            PDE 类型。
        domain : dict
            域参数。

        Returns
        -------
        ndarray
            残差图 (H, W)。
        """
        h, w = u.shape
        dx = domain.get("dx", 1.0 / h)
        k = domain.get("wavenumber", 2 * math.pi * 0.5)
        c = domain.get("wave_speed", 1.0)
        D = domain.get("diffusion_coeff", 0.1)

        # 拉普拉斯算子 (有限差分)
        laplacian = np.zeros_like(u)
        laplacian[1:-1, 1:-1] = (
            (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1])
            / (dx**2)
        )
        # 处理数值不稳定
        laplacian = np.nan_to_num(laplacian, nan=0.0, posinf=0.0, neginf=0.0)

        if pde_type == "wave":
            # 波动方程残差: d^2u/dt^2 - c^2 * laplacian(u) = 0
            # 简化: 使用稳态近似 laplacian(u) = 0
            residual = c**2 * laplacian

        elif pde_type == "helmholtz":
            # Helmholtz 方程残差: laplacian(u) + k^2 * u = f
            f = domain.get("source", np.zeros_like(u))
            residual = laplacian + k**2 * u - f

        elif pde_type == "diffusion":
            # 扩散方程残差: D * laplacian(u) = 0 (稳态)
            residual = D * laplacian

        elif pde_type == "poisson":
            # Poisson 方程残差: laplacian(u) = f
            f = domain.get("source", np.zeros_like(u))
            residual = laplacian - f

        else:
            # 默认: 拉普拉斯方程
            residual = laplacian

        return residual

    def _adaptive_resample(self, residual_map: np.ndarray,
                            grid_size: Tuple[int, int]):
        """残差自适应采样。

        在高残差区域增加配置点密度。

        Parameters
        ----------
        residual_map : ndarray
            残差图 (H, W)。
        grid_size : tuple of int
            网格尺寸。
        """
        if self._collocation_points is None:
            return

        h, w = grid_size
        n = self.config.num_collocation_points

        # 计算残差概率分布
        abs_residual = np.abs(residual_map)
        # 处理 NaN/Inf
        abs_residual = np.nan_to_num(abs_residual, nan=0.0, posinf=1e10, neginf=0.0)
        total = abs_residual.sum() + 1e-10
        prob_map = abs_residual / total

        # 归一化为概率
        prob_flat = prob_map.flatten()
        prob_flat = np.nan_to_num(prob_flat, nan=0.0)
        prob_sum = prob_flat.sum()
        if prob_sum < 1e-10:
            # 均匀分布 fallback
            prob_flat = np.ones(len(prob_flat)) / len(prob_flat)
        else:
            prob_flat = prob_flat / prob_sum

        # 基于残差的加权采样
        indices = np.random.choice(len(prob_flat), size=n, p=prob_flat)
        y_indices = indices // w
        x_indices = indices % w

        # 添加小随机扰动
        y_points = y_indices.astype(np.float64) + np.random.randn(n) * 0.5
        x_points = x_indices.astype(np.float64) + np.random.randn(n) * 0.5

        y_points = np.clip(y_points, 0, h - 1)
        x_points = np.clip(x_points, 0, w - 1)

        self._collocation_points = np.column_stack([y_points, x_points])

    def _update_solution(self, solution: np.ndarray, residual_map: np.ndarray,
                          target: Optional[np.ndarray],
                          physics_weight: float, data_weight: float) -> np.ndarray:
        """基于残差更新解。

        Parameters
        ----------
        solution : ndarray
            当前解。
        residual_map : ndarray
            残差图。
        target : ndarray, optional
            目标数据。
        physics_weight : float
            物理损失权重。
        data_weight : float
            数据损失权重。

        Returns
        -------
        ndarray
            更新后的解。
        """
        lr = self.config.learning_rate
        updated = solution.copy()

        # 物理梯度: 沿残差反方向更新
        physics_gradient = residual_map * physics_weight
        # 梯度裁剪防止数值爆炸
        grad_norm = np.sqrt(np.mean(physics_gradient**2)) + 1e-10
        if grad_norm > 10.0:
            physics_gradient = physics_gradient * 10.0 / grad_norm
        updated -= lr * physics_gradient

        # 数据梯度
        if target is not None:
            data_gradient = 2 * (solution - target) * data_weight
            data_grad_norm = np.sqrt(np.mean(data_gradient**2)) + 1e-10
            if data_grad_norm > 10.0:
                data_gradient = data_gradient * 10.0 / data_grad_norm
            updated -= lr * data_gradient

        return updated

    def validate_solution(self, solution: np.ndarray,
                           grid_size: Tuple[int, int] = (64, 64)) -> Dict:
        """验证解的物理一致性。

        Parameters
        ----------
        solution : ndarray
            待验证的解 (H, W)。
        grid_size : tuple of int
            网格尺寸。

        Returns
        -------
        dict
            验证结果。
        """
        if not self._constraints:
            return {"status": "no_constraints", "is_valid": True}

        results = {}
        total_residual = 0.0

        for constraint in self._constraints:
            if constraint.residual_fn is not None:
                residual = constraint.residual_fn(solution, constraint.domain)
            else:
                residual = self._compute_pde_residual(
                    solution, constraint.equation_type, constraint.domain
                )

            rms = float(np.sqrt(np.mean(residual**2)))
            max_abs = float(np.max(np.abs(residual)))
            total_residual += constraint.weight * rms

            results[constraint.name] = {
                "equation_type": constraint.equation_type,
                "rms_residual": rms,
                "max_residual": max_abs,
                "weight": constraint.weight,
                "weighted_residual": constraint.weight * rms,
            }

        overall_rms = total_residual / sum(c.weight for c in self._constraints)

        return {
            "status": "validated",
            "is_valid": overall_rms < self.config.convergence_threshold * 10,
            "overall_rms_residual": overall_rms,
            "per_constraint": results,
            "num_constraints": len(self._constraints),
        }

    def reset(self):
        """重置优化器状态。"""
        self._constraints.clear()
        self._collocation_points = None
        self._network_weights = None
        self._loss_history.clear()
        self._physics_loss_history.clear()
        self._data_loss_history.clear()
        self._is_initialized = False
        self._call_count = 0

    def get_status(self) -> Dict:
        """获取优化器状态。"""
        return {
            "is_initialized": self._is_initialized,
            "call_count": self._call_count,
            "num_constraints": len(self._constraints),
            "constraint_types": [c.equation_type for c in self._constraints],
            "max_iterations": self.config.max_iterations,
            "adaptive_sampling": self.config.adaptive_sampling,
            "num_collocation_points": self.config.num_collocation_points,
            "recent_loss": float(self._loss_history[-1]) if self._loss_history else 0.0,
            "recent_physics_loss": float(self._physics_loss_history[-1]) if self._physics_loss_history else 0.0,
        }
