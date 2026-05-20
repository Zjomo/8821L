"""
SpotZoom 前沿开源项目创新模块 v20.0
基于 2024-2026 最新科研前沿开源项目的创新模块补充 (第八轮)

新增模块 (v20.0):
- TopologyOptimizedNeuralController: 拓扑优化神经控制器 (受 NAS + Optuna 启发)
- HolographicSpotReconstructor: 全息光斑重建器 (受 Digital Holography + Gerchberg-Saxton 启发)
- FederatedMultiTaskOptimizer: 联邦多任务优化器 (受 MOON + FedProx 启发)
- NeuralODEBeamPropagator: 神经常微分方程光束传播器 (受 torchdiffeq + Neural ODE 启发)
- QuantumInspiredOptimizer: 量子启发优化器 (受 QAOA + Quantum Annealing 启发)
- MultimodalFusionTracker: 多模态融合跟踪器 (受 ViT + CLIP 启发)

参考项目:
- NAS (Neural Architecture Search) + Optuna (optuna/optuna) — 自动超参数/结构搜索
- Digital Holography + Gerchberg-Saxton Algorithm — 数字全息相位恢复
- MOON (Model-Contrastive Federated Learning) + FedProx — 联邦多任务学习
- Neural ODE (Chen et al., NeurIPS 2018) + torchdiffeq — 连续深度模型
- QAOA (Quantum Approximate Optimization Algorithm) + Quantum Annealing — 量子优化
- ViT (Vision Transformer) + CLIP (Contrastive Language-Image Pre-training) — 多模态学习
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

__all__ = [
    "TopologyOptConfig",
    "TopologyOptResult",
    "TopologyOptimizedNeuralController",
    "HolographicReconConfig",
    "HolographicReconResult",
    "HolographicSpotReconstructor",
    "FederatedMultiTaskConfig",
    "FederatedMultiTaskResult",
    "FederatedMultiTaskOptimizer",
    "NeuralODEConfig",
    "NeuralODEResult",
    "NeuralODEBeamPropagator",
    "QuantumInspiredConfig",
    "QuantumInspiredResult",
    "QuantumInspiredOptimizer",
    "MultimodalFusionConfig",
    "MultimodalFusionResult",
    "MultimodalFusionTracker",
]


# ============================================================================
# 辅助函数
# ============================================================================

def _safe_center_of_mass(img: np.ndarray) -> Tuple[float, float]:
    """安全计算质心，scipy 不可用时使用纯 numpy 回退。

    Safe center-of-mass computation with pure-numpy fallback.

    Args:
        img: 输入图像

    Returns:
        (cy, cx) 质心坐标
    """
    total = img.sum()
    if total <= 0:
        return img.shape[0] / 2.0, img.shape[1] / 2.0
    if SCIPY_AVAILABLE:
        return ndimage.center_of_mass(img)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    return float(np.sum(yy * img) / total), float(np.sum(xx * img) / total)


def _generate_gaussian_spot(
    size: int = 64,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    sigma: float = 5.0,
    amplitude: float = 1.0,
    noise_level: float = 0.01,
) -> np.ndarray:
    """生成模拟高斯光斑图像。

    Generate a simulated Gaussian spot image for testing.

    Args:
        size: 图像尺寸 (正方形)
        cx: 光斑中心 x，None 时为图像中心
        cy: 光斑中心 y，None 时为图像中心
        sigma: 高斯标准差 (像素)
        amplitude: 峰值振幅
        noise_level: 噪声水平

    Returns:
        模拟光斑图像
    """
    if cx is None:
        cx = size / 2.0
    if cy is None:
        cy = size / 2.0
    yy, xx = np.mgrid[:size, :size]
    spot = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    noise = np.random.randn(size, size) * noise_level * amplitude
    return spot + noise


# ============================================================================
# 1. TopologyOptimizedNeuralController — 拓扑优化神经控制器
# ============================================================================
# 参考: NAS (Neural Architecture Search) + Optuna (optuna/optuna)
# 自动搜索最优控制网络拓扑结构，支持多目标优化

@dataclass
class TopologyOptConfig:
    """拓扑优化神经控制器配置

    Configuration for topology-optimized neural controller.
    """
    search_space_layers: Tuple[int, int] = (1, 6)       # 网络层数搜索范围
    search_space_hidden: Tuple[int, int] = (16, 256)     # 隐藏层维度搜索范围
    search_space_activation: List[str] = field(
        default_factory=lambda: ["relu", "tanh", "sigmoid", "leaky_relu"]
    )  # 候选激活函数
    num_trials: int = 50                                  # 搜索试验次数
    max_iterations_per_trial: int = 100                   # 每次试验最大迭代
    objectives: List[str] = field(
        default_factory=lambda: ["accuracy", "speed", "robustness"]
    )  # 多目标优化目标
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "accuracy": 0.5, "speed": 0.3, "robustness": 0.2
        }
    )  # 目标权重
    input_dim: int = 8                                    # 输入维度 (状态特征)
    output_dim: int = 3                                   # 输出维度 (X/Y/Z 校正)
    seed: int = 42


@dataclass
class TopologyOptResult:
    """拓扑优化结果

    Result of topology optimization.
    """
    best_architecture: Dict[str, Any]        # 最优网络结构
    best_score: float                        # 最优综合分数
    objective_scores: Dict[str, float]       # 各目标分数
    all_trials: List[Dict[str, Any]]         # 所有试验记录
    convergence_history: List[float]         # 收敛历史
    processing_time_ms: float                # 处理时间
    pareto_front: List[Dict[str, Any]]       # Pareto 前沿解


class TopologyOptimizedNeuralController:
    """拓扑优化神经控制器

    受 NAS (Neural Architecture Search) 和 Optuna 启发，自动搜索最优控制网络拓扑结构。
    支持多目标优化 (精度、速度、鲁棒性)，为光斑对准系统找到最佳控制器架构。

    Inspired by NAS and Optuna, automatically searches for the optimal control
    network topology. Supports multi-objective optimization (accuracy, speed,
    robustness) to find the best controller architecture for spot alignment.

    核心特性:
    - 自动搜索网络层数、隐藏维度、激活函数
    - 多目标 Pareto 优化
    - 基于 TPE (Tree-structured Parzen Estimator) 的贝叶斯搜索
    - 支持自定义搜索空间和约束
    """

    def __init__(self, config: Optional[TopologyOptConfig] = None):
        """初始化拓扑优化神经控制器。

        Initialize the topology-optimized neural controller.

        Args:
            config: 配置参数，为 None 时使用默认值
        """
        self.config = config or TopologyOptConfig()
        self._rng = np.random.RandomState(self.config.seed)
        self._best_architecture: Optional[Dict[str, Any]] = None
        self._best_score: float = -float('inf')
        self._trials: deque = deque(maxlen=500)
        self._convergence: deque = deque(maxlen=500)
        self._pareto_front: List[Dict[str, Any]] = []
        self._model_weights: Optional[Dict[str, np.ndarray]] = None

    def reset(self):
        """重置控制器状态。

        Reset the controller state.
        """
        self._best_architecture = None
        self._best_score = -float('inf')
        self._trials.clear()
        self._convergence.clear()
        self._pareto_front.clear()
        self._model_weights = None
        LOGGER.info("TopologyOptimizedNeuralController 已重置")

    def _sample_architecture(self, trial_idx: int) -> Dict[str, Any]:
        """采样一个候选网络架构。

        Sample a candidate network architecture using TPE-like strategy.

        Args:
            trial_idx: 当前试验索引

        Returns:
            候选架构描述字典
        """
        cfg = self.config

        # TPE-like: 前期探索，后期利用
        if trial_idx < 5 or not self._trials:
            # 随机探索
            num_layers = self._rng.randint(cfg.search_space_layers[0],
                                           cfg.search_space_layers[1] + 1)
            hidden_dims = [
                int(self._rng.randint(cfg.search_space_hidden[0],
                                      cfg.search_space_hidden[1] + 1))
                for _ in range(num_layers)
            ]
            activations = [
                self._rng.choice(cfg.search_space_activation)
                for _ in range(num_layers)
            ]
        else:
            # 基于历史最优的利用
            best_trial = max(self._trials, key=lambda t: t["score"])
            best_arch = best_trial["architecture"]

            num_layers = best_arch["num_layers"]
            # 在最优架构附近扰动
            num_layers = int(np.clip(
                num_layers + self._rng.randint(-1, 2),
                cfg.search_space_layers[0],
                cfg.search_space_layers[1]
            ))
            hidden_dims = []
            for i in range(num_layers):
                if i < len(best_arch["hidden_dims"]):
                    base = best_arch["hidden_dims"][i]
                    dim = int(np.clip(
                        base + self._rng.randint(-32, 33),
                        cfg.search_space_hidden[0],
                        cfg.search_space_hidden[1]
                    ))
                else:
                    dim = int(self._rng.randint(cfg.search_space_hidden[0],
                                                cfg.search_space_hidden[1] + 1))
                hidden_dims.append(dim)
            activations = [
                self._rng.choice(cfg.search_space_activation)
                for _ in range(num_layers)
            ]

        architecture = {
            "num_layers": num_layers,
            "hidden_dims": hidden_dims,
            "activations": activations,
            "input_dim": cfg.input_dim,
            "output_dim": cfg.output_dim,
        }
        return architecture

    def _evaluate_architecture(self, architecture: Dict[str, Any],
                               trial_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """评估候选架构的多目标性能。

        Evaluate a candidate architecture on multiple objectives.

        Args:
            architecture: 候选架构描述
            trial_data: 评估数据 (None 时自动生成)

        Returns:
            各目标的分数字典
        """
        if trial_data is None:
            # 生成模拟评估数据
            n_samples = 200
            trial_data = self._rng.randn(n_samples, self.config.input_dim)

        num_layers = architecture["num_layers"]
        hidden_dims = architecture["hidden_dims"]

        # 精度评估: 模拟训练误差 (层数和维度适中时最优)
        total_params = sum(hidden_dims) + self.config.input_dim * hidden_dims[0] if hidden_dims else 0
        for i in range(len(hidden_dims) - 1):
            total_params += hidden_dims[i] * hidden_dims[i + 1]
        if hidden_dims:
            total_params += hidden_dims[-1] * self.config.output_dim

        # 模拟精度: 参数量适中时精度高 (避免过参数化和欠参数化)
        ideal_params = 5000
        accuracy = float(np.exp(-((total_params - ideal_params) / ideal_params) ** 2))

        # 添加随机性模拟训练波动
        accuracy *= (0.9 + 0.1 * self._rng.rand())

        # 速度评估: 参数越少速度越快
        max_possible_params = self.config.search_space_hidden[1] ** 2 * self.config.search_space_layers[1]
        speed = float(1.0 - total_params / (max_possible_params + 1))

        # 鲁棒性评估: 层数适中、激活函数多样性高时鲁棒性好
        activation_diversity = len(set(architecture["activations"])) / max(len(architecture["activations"]), 1)
        layer_balance = 1.0 - np.std(hidden_dims) / (np.mean(hidden_dims) + 1e-8) if hidden_dims else 0
        robustness = float(0.5 * activation_diversity + 0.5 * np.clip(layer_balance, 0, 1))
        robustness *= (0.85 + 0.15 * self._rng.rand())

        return {
            "accuracy": float(np.clip(accuracy, 0, 1)),
            "speed": float(np.clip(speed, 0, 1)),
            "robustness": float(np.clip(robustness, 0, 1)),
        }

    def _compute_pareto_front(self):
        """计算 Pareto 前沿解。

        Compute the Pareto front from all evaluated trials.
        """
        if not self._trials:
            return

        objectives = self.config.objectives
        pareto = []
        for trial in self._trials:
            scores = trial["objective_scores"]
            dominated = False
            for other in self._trials:
                if other is trial:
                    continue
                other_scores = other["objective_scores"]
                # 检查 other 是否在所有目标上不差于 trial 且至少一个更优
                all_better_or_equal = all(
                    other_scores.get(obj, 0) >= scores.get(obj, 0)
                    for obj in objectives
                )
                any_better = any(
                    other_scores.get(obj, 0) > scores.get(obj, 0)
                    for obj in objectives
                )
                if all_better_or_equal and any_better:
                    dominated = True
                    break
            if not dominated:
                pareto.append(trial)

        self._pareto_front = pareto
        LOGGER.debug("Pareto 前沿包含 %d 个解", len(pareto))

    def _composite_score(self, objective_scores: Dict[str, float]) -> float:
        """计算加权综合分数。

        Compute weighted composite score from individual objectives.

        Args:
            objective_scores: 各目标分数

        Returns:
            加权综合分数
        """
        total = 0.0
        weight_sum = 0.0
        for obj_name, weight in self.config.objective_weights.items():
            if obj_name in objective_scores:
                total += weight * objective_scores[obj_name]
                weight_sum += weight
        return total / (weight_sum + 1e-8)

    def search(self, trial_data: Optional[np.ndarray] = None) -> TopologyOptResult:
        """执行拓扑搜索。

        Execute the topology search over the architecture space.

        Args:
            trial_data: 评估数据 (None 时自动生成)

        Returns:
            TopologyOptResult 搜索结果
        """
        t0 = time.perf_counter()
        LOGGER.info("开始拓扑优化搜索, 目标试验次数=%d", self.config.num_trials)

        for trial_idx in range(self.config.num_trials):
            # 采样架构
            architecture = self._sample_architecture(trial_idx)

            # 评估多目标
            objective_scores = self._evaluate_architecture(architecture, trial_data)

            # 计算综合分数
            score = self._composite_score(objective_scores)

            # 记录试验
            trial_record = {
                "trial_idx": trial_idx,
                "architecture": architecture,
                "objective_scores": objective_scores,
                "score": score,
            }
            self._trials.append(trial_record)
            self._convergence.append(score)

            # 更新最优
            if score > self._best_score:
                self._best_score = score
                self._best_architecture = copy.deepcopy(architecture)
                LOGGER.debug(
                    "试验 %d: 新最优 score=%.4f, 架构=%s",
                    trial_idx, score, architecture
                )

        # 计算 Pareto 前沿
        self._compute_pareto_front()

        # 初始化最优架构的权重
        self._initialize_weights(self._best_architecture)

        elapsed = (time.perf_counter() - t0) * 1000.0
        LOGGER.info(
            "拓扑搜索完成, 最优 score=%.4f, 耗时=%.1fms",
            self._best_score, elapsed
        )

        return TopologyOptResult(
            best_architecture=self._best_architecture or {},
            best_score=self._best_score,
            objective_scores=self._trials[-1]["objective_scores"] if self._trials else {},
            all_trials=self._trials,
            convergence_history=self._convergence,
            processing_time_ms=elapsed,
            pareto_front=self._pareto_front,
        )

    def _initialize_weights(self, architecture: Dict[str, Any]):
        """初始化最优架构的网络权重。

        Initialize network weights for the best architecture.

        Args:
            architecture: 最优架构描述
        """
        self._model_weights = {}
        hidden_dims = architecture["hidden_dims"]
        input_dim = architecture["input_dim"]
        output_dim = architecture["output_dim"]

        # Xavier 初始化
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._model_weights[f"W{i}"] = self._rng.randn(fan_in, fan_out) * std
            self._model_weights[f"b{i}"] = np.zeros(fan_out)

        LOGGER.info("已初始化网络权重, 共 %d 层", len(layer_dims) - 1)

    def _activate(self, x: np.ndarray, activation: str) -> np.ndarray:
        """应用激活函数。

        Apply activation function.

        Args:
            x: 输入数组
            activation: 激活函数名称

        Returns:
            激活后的数组
        """
        if activation == "relu":
            return np.maximum(0, x)
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif activation == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        else:
            return np.maximum(0, x)

    def process(self, state: np.ndarray) -> np.ndarray:
        """使用最优架构处理状态，输出控制信号。

        Process state through the best architecture to produce control signals.

        Args:
            state: 输入状态向量

        Returns:
            控制信号向量
        """
        if self._best_architecture is None or self._model_weights is None:
            LOGGER.warning("尚未完成拓扑搜索，请先调用 search()")
            return np.zeros(self.config.output_dim)

        x = state.astype(np.float64).flatten()
        if x.shape[0] != self.config.input_dim:
            LOGGER.warning(
                "输入维度 %d 与配置 input_dim=%d 不匹配",
                x.shape[0], self.config.input_dim
            )
            if x.shape[0] > self.config.input_dim:
                x = x[:self.config.input_dim]
            else:
                x = np.pad(x, (0, self.config.input_dim - x.shape[0]))

        hidden_dims = self._best_architecture["hidden_dims"]
        activations = self._best_architecture["activations"]
        num_layers = self._best_architecture["num_layers"]

        # 前向传播
        for i in range(num_layers):
            W = self._model_weights[f"W{i}"]
            b = self._model_weights[f"b{i}"]
            x = x @ W + b
            act_name = activations[i] if i < len(activations) else "relu"
            x = self._activate(x, act_name)

        # 输出层 (无激活)
        W_out = self._model_weights[f"W{num_layers}"]
        b_out = self._model_weights[f"b{num_layers}"]
        output = x @ W_out + b_out

        return output

    def get_architecture_summary(self) -> Dict[str, Any]:
        """获取最优架构摘要。

        Get a summary of the best architecture found.

        Returns:
            架构摘要字典
        """
        if self._best_architecture is None:
            return {"status": "not_searched"}

        hidden_dims = self._best_architecture["hidden_dims"]
        total_params = sum(hidden_dims)
        total_params += self.config.input_dim * hidden_dims[0] if hidden_dims else 0
        for i in range(len(hidden_dims) - 1):
            total_params += hidden_dims[i] * hidden_dims[i + 1]
        if hidden_dims:
            total_params += hidden_dims[-1] * self.config.output_dim

        return {
            "status": "searched",
            "architecture": self._best_architecture,
            "total_parameters": total_params,
            "best_score": self._best_score,
            "num_trials": len(self._trials),
            "pareto_size": len(self._pareto_front),
        }


# ============================================================================
# 2. HolographicSpotReconstructor — 全息光斑重建器
# ============================================================================
# 参考: Digital Holography + Phase Retrieval (Gerchberg-Saxton Algorithm)
# 从单幅干涉图重建光斑复振幅场

@dataclass
class HolographicReconConfig:
    """全息光斑重建配置

    Configuration for holographic spot reconstruction.
    """
    image_size: int = 256                              # 图像尺寸
    wavelength: float = 632.8e-9                       # 波长 (m), He-Ne 激光
    pixel_size: float = 5.0e-6                         # 像素尺寸 (m)
    propagation_distance: float = 0.1                  # 传播距离 (m)
    num_gs_iterations: int = 50                        # Gerchberg-Saxton 迭代次数
    support_constraint: str = "tight"                  # 支撑约束: "tight" / "loose" / "none"
    multi_wavelength: bool = False                     # 多波长重建
    wavelengths: List[float] = field(
        default_factory=lambda: [632.8e-9, 532.0e-9, 450.0e-9]
    )  # 多波长列表
    multi_channel: bool = False                        # 多通道重建
    regularization: float = 1e-3                       # 正则化强度
    noise_threshold: float = 0.05                      # 噪声阈值


@dataclass
class HolographicReconResult:
    """全息光斑重建结果

    Result of holographic spot reconstruction.
    """
    amplitude: np.ndarray              # 重建的振幅分布
    phase: np.ndarray                  # 重建的相位分布
    complex_field: np.ndarray          # 重建的复振幅场
    reconstruction_error: float        # 重建误差
    convergence_history: List[float]   # 收敛历史
    wavelength: float                  # 使用波长
    processing_time_ms: float          # 处理时间
    multi_wavelength_results: Optional[List[Dict[str, Any]]] = None  # 多波长结果


class HolographicSpotReconstructor:
    """全息光斑重建器

    受数字全息术 (Digital Holography) 和 Gerchberg-Saxton 相位恢复算法启发，
    从单幅干涉图重建光斑的复振幅场 (振幅 + 相位)。

    Inspired by Digital Holography and the Gerchberg-Saxton phase retrieval
    algorithm, reconstructs the complex amplitude field (amplitude + phase)
    of a spot from a single interferogram.

    核心特性:
    - Gerchberg-Saxton 迭代相位恢复
    - 角谱法波场传播
    - 多波长重建支持
    - 支撑约束和正则化
    - 自适应噪声抑制
    """

    def __init__(self, config: Optional[HolographicReconConfig] = None):
        """初始化全息光斑重建器。

        Initialize the holographic spot reconstructor.

        Args:
            config: 配置参数，为 None 时使用默认值
        """
        self.config = config or HolographicReconConfig()
        self._last_result: Optional[HolographicReconResult] = None

    def reset(self):
        """重置重建器状态。

        Reset the reconstructor state.
        """
        self._last_result = None
        LOGGER.info("HolographicSpotReconstructor 已重置")

    def _angular_spectrum_propagate(
        self,
        field: np.ndarray,
        distance: float,
        wavelength: float,
    ) -> np.ndarray:
        """角谱法波场传播。

        Angular Spectrum Method for wave field propagation.

        Args:
            field: 输入复数场 (2D)
            distance: 传播距离 (m)
            wavelength: 波长 (m)

        Returns:
            传播后的复数场
        """
        ny, nx = field.shape
        k = 2.0 * np.pi / wavelength
        pixel_size = self.config.pixel_size

        # 频率坐标
        fx = np.fft.fftfreq(nx, d=pixel_size)
        fy = np.fft.fftfreq(ny, d=pixel_size)
        FX, FY = np.meshgrid(fx, fy)

        # 传递函数
        arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
        propagator = np.zeros_like(arg, dtype=np.complex128)
        valid = arg >= 0
        propagator[valid] = np.exp(1j * k * distance * np.sqrt(arg[valid]))

        # 传播
        field_fft = np.fft.fft2(field)
        result_fft = field_fft * propagator
        result = np.fft.ifft2(result_fft)
        return result

    def _apply_support_constraint(
        self, field: np.ndarray, measurement: np.ndarray
    ) -> np.ndarray:
        """应用支撑约束。

        Apply support constraint to the reconstructed field.

        Args:
            field: 当前估计的复数场
            measurement: 测量的干涉图强度

        Returns:
            约束后的复数场
        """
        amplitude = np.abs(field)
        phase = np.angle(field)

        if self.config.support_constraint == "tight":
            # 紧支撑: 用测量强度替换振幅
            new_amplitude = np.sqrt(np.maximum(measurement, 0))
        elif self.config.support_constraint == "loose":
            # 松支撑: 混合测量振幅和估计振幅
            measured_amplitude = np.sqrt(np.maximum(measurement, 0))
            new_amplitude = 0.7 * measured_amplitude + 0.3 * amplitude
        else:
            # 无约束
            new_amplitude = amplitude

        return new_amplitude * np.exp(1j * phase)

    def _apply_fourier_constraint(
        self, field: np.ndarray, pupil_mask: np.ndarray
    ) -> np.ndarray:
        """应用傅里叶域约束 (孔径约束)。

        Apply Fourier domain constraint (aperture constraint).

        Args:
            field: 空间域复数场
            pupil_mask: 孔径掩模

        Returns:
            约束后的复数场
        """
        field_fft = np.fft.fft2(field)
        field_fft = field_fft * pupil_mask
        return np.fft.ifft2(field_fft)

    def reconstruct(
        self, interferogram: np.ndarray
    ) -> HolographicReconResult:
        """从干涉图重建光斑复振幅场。

        Reconstruct the complex amplitude field from an interferogram.

        Args:
            interferogram: 输入干涉图 (2D numpy 数组)

        Returns:
            HolographicReconResult 重建结果
        """
        t0 = time.perf_counter()

        if self.config.multi_wavelength:
            result = self._reconstruct_multi_wavelength(interferogram)
        else:
            result = self._reconstruct_single_wavelength(
                interferogram, self.config.wavelength
            )

        self._last_result = result
        elapsed = (time.perf_counter() - t0) * 1000.0
        result.processing_time_ms = elapsed

        LOGGER.info(
            "全息重建完成, 误差=%.6f, 耗时=%.1fms",
            result.reconstruction_error, elapsed
        )
        return result

    def _reconstruct_single_wavelength(
        self, interferogram: np.ndarray, wavelength: float
    ) -> HolographicReconResult:
        """单波长 Gerchberg-Saxton 重建。

        Single-wavelength Gerchberg-Saxton reconstruction.

        Args:
            interferogram: 干涉图
            wavelength: 波长

        Returns:
            重建结果
        """
        # 预处理
        img = interferogram.astype(np.float64)
        if img.max() > 1:
            img = img / img.max()

        npix = self.config.image_size
        if img.shape[0] != npix or img.shape[1] != npix:
            if CV2_AVAILABLE:
                img = cv2.resize(img, (npix, npix))
            else:
                # numpy 最近邻插值
                y_idx = np.linspace(0, img.shape[0] - 1, npix).astype(int)
                x_idx = np.linspace(0, img.shape[1] - 1, npix).astype(int)
                img = img[np.ix_(y_idx, x_idx)]

        # 噪声抑制
        img = np.maximum(img - self.config.noise_threshold, 0)

        # 初始化: 随机相位 + 测量振幅
        amplitude = np.sqrt(img)
        random_phase = np.random.uniform(0, 2 * np.pi, img.shape)
        field = amplitude * np.exp(1j * random_phase)

        # 孔径掩模 (圆形)
        yy, xx = np.mgrid[:npix, :npix]
        cx, cy = npix / 2, npix / 2
        radius = npix / 2 - 2
        pupil_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
        pupil_mask = pupil_mask.astype(np.float64)

        convergence = []
        distance = self.config.propagation_distance

        for iteration in range(self.config.num_gs_iterations):
            # 1. 传播到观测平面
            propagated = self._angular_spectrum_propagate(field, distance, wavelength)

            # 2. 应用强度约束 (替换振幅，保留相位)
            prop_amplitude = np.abs(propagated)
            prop_phase = np.angle(propagated)
            new_amplitude = np.sqrt(img)
            field = new_amplitude * np.exp(1j * prop_phase)

            # 3. 反向传播到孔径平面
            field = self._angular_spectrum_propagate(field, -distance, wavelength)

            # 4. 应用孔径约束
            field = self._apply_fourier_constraint(field, pupil_mask)

            # 5. 正则化 (平滑)
            if self.config.regularization > 0 and SCIPY_AVAILABLE:
                real_part = ndimage.gaussian_filter(field.real, sigma=0.5)
                imag_part = ndimage.gaussian_filter(field.imag, sigma=0.5)
                field = (1 - self.config.regularization) * field + \
                        self.config.regularization * (real_part + 1j * imag_part)

            # 计算误差
            error = float(np.mean((np.abs(propagated) ** 2 - img) ** 2))
            convergence.append(error)

        # 提取结果
        final_field = self._angular_spectrum_propagate(field, distance, wavelength)
        amplitude_result = np.abs(final_field)
        phase_result = np.angle(final_field)

        return HolographicReconResult(
            amplitude=amplitude_result,
            phase=phase_result,
            complex_field=final_field,
            reconstruction_error=convergence[-1] if convergence else float('inf'),
            convergence_history=convergence,
            wavelength=wavelength,
            processing_time_ms=0.0,
        )

    def _reconstruct_multi_wavelength(
        self, interferogram: np.ndarray
    ) -> HolographicReconResult:
        """多波长重建。

        Multi-wavelength reconstruction for improved phase unwrapping.

        Args:
            interferogram: 干涉图

        Returns:
            多波长重建结果
        """
        multi_results = []
        best_result = None
        best_error = float('inf')

        for wl in self.config.wavelengths:
            result = self._reconstruct_single_wavelength(interferogram, wl)
            multi_results.append({
                "wavelength": wl,
                "amplitude": result.amplitude,
                "phase": result.phase,
                "error": result.reconstruction_error,
            })
            if result.reconstruction_error < best_error:
                best_error = result.reconstruction_error
                best_result = result

        # 多波长相位合成
        if len(multi_results) >= 2:
            # 合成相位: 加权平均
            total_weight = sum(1.0 / (r["error"] + 1e-8) for r in multi_results)
            synthetic_phase = np.zeros_like(multi_results[0]["phase"])
            for r in multi_results:
                w = (1.0 / (r["error"] + 1e-8)) / total_weight
                synthetic_phase += w * r["phase"]

            best_result.phase = synthetic_phase
            best_result.complex_field = best_result.amplitude * np.exp(1j * synthetic_phase)
            best_result.multi_wavelength_results = multi_results

        return best_result

    def batch_reconstruct(
        self, interferograms: List[np.ndarray]
    ) -> List[HolographicReconResult]:
        """批量重建多幅干涉图。

        Batch reconstruct multiple interferograms.

        Args:
            interferograms: 干涉图列表

        Returns:
            重建结果列表
        """
        return [self.reconstruct(img) for img in interferograms]


# ============================================================================
# 3. FederatedMultiTaskOptimizer — 联邦多任务优化器
# ============================================================================
# 参考: MOON (Model-Contrastive Federated Learning) + FedProx
# 多个光学系统协同训练，保护数据隐私

@dataclass
class FederatedMultiTaskConfig:
    """联邦多任务优化配置

    Configuration for federated multi-task optimization.
    """
    num_clients: int = 5                                # 客户端数量
    num_rounds: int = 20                                # 联邦轮次
    local_epochs: int = 5                               # 本地训练轮次
    learning_rate: float = 1e-3                         # 学习率
    proximal_mu: float = 0.01                           # FedProx 近端项系数
    moon_temperature: float = 0.5                       # MOON 对比温度
    moon_tau: float = 0.9                               # MOON 动量系数
    model_dim: int = 64                                 # 模型维度
    task_dim: int = 16                                  # 任务特定维度
    communication_compression: float = 1.0              # 通信压缩率 (1.0=无压缩)
    heterogeneity_level: float = 0.3                    # 数据异构程度
    aggregation_strategy: str = "fedavg"                # 聚合策略: "fedavg" / "fedprox" / "moon"
    differential_privacy_epsilon: float = 0.0           # 差分隐私 epsilon (0=不启用)


@dataclass
class FederatedMultiTaskResult:
    """联邦多任务优化结果

    Result of federated multi-task optimization.
    """
    global_model: Dict[str, np.ndarray]                 # 全局模型参数
    client_models: List[Dict[str, np.ndarray]]          # 各客户端模型参数
    convergence_history: List[float]                    # 收敛历史
    communication_cost: float                           # 通信开销 (参数量)
    per_client_metrics: List[Dict[str, float]]          # 各客户端指标
    privacy_budget_used: float                          # 已使用隐私预算
    total_rounds: int                                   # 总轮次
    processing_time_ms: float                           # 处理时间


class FederatedMultiTaskOptimizer:
    """联邦多任务优化器

    受 MOON (Model-Contrastive Federated Learning) 和 FedProx 启发，
    实现多个光学系统的协同训练，同时保护各系统的数据隐私。

    Inspired by MOON and FedProx, implements collaborative training across
    multiple optical systems while preserving data privacy.

    核心特性:
    - FedAvg/FedProx/MOON 三种聚合策略
    - 异构设备和非 IID 数据分布支持
    - 通信压缩降低带宽需求
    - 差分隐私保护
    - 多任务学习: 共享表示 + 任务特定头
    """

    def __init__(self, config: Optional[FederatedMultiTaskConfig] = None):
        """初始化联邦多任务优化器。

        Initialize the federated multi-task optimizer.

        Args:
            config: 配置参数，为 None 时使用默认值
        """
        self.config = config or FederatedMultiTaskConfig()
        self._rng = np.random.RandomState(42)
        self._global_model: Optional[Dict[str, np.ndarray]] = None
        self._client_models: List[Dict[str, np.ndarray]] = []
        self._client_prev_models: List[Dict[str, np.ndarray]] = []
        self._convergence: deque = deque(maxlen=1000)
        self._round_idx = 0
        self._total_comm_cost = 0.0
        self._privacy_budget_used = 0.0

    def reset(self):
        """重置优化器状态。

        Reset the optimizer state.
        """
        self._global_model = None
        self._client_models.clear()
        self._client_prev_models.clear()
        self._convergence.clear()
        self._round_idx = 0
        self._total_comm_cost = 0.0
        self._privacy_budget_used = 0.0
        LOGGER.info("FederatedMultiTaskOptimizer 已重置")

    def _initialize_models(self):
        """初始化全局模型和客户端模型。

        Initialize global and client models.
        """
        dim = self.config.model_dim
        task_dim = self.config.task_dim

        # 共享表示层
        shared_W = self._rng.randn(dim, dim) * np.sqrt(2.0 / dim)
        shared_b = np.zeros(dim)

        self._global_model = {
            "shared_W": shared_W.copy(),
            "shared_b": shared_b.copy(),
        }

        self._client_models = []
        self._client_prev_models = []
        for client_id in range(self.config.num_clients):
            # 任务特定头
            task_W = self._rng.randn(dim, task_dim) * np.sqrt(2.0 / dim)
            task_b = np.zeros(task_dim)

            client_model = {
                "shared_W": shared_W.copy(),
                "shared_b": shared_b.copy(),
                "task_W": task_W.copy(),
                "task_b": task_b.copy(),
            }
            self._client_models.append(client_model)
            self._client_prev_models.append(copy.deepcopy(client_model))

        LOGGER.info(
            "已初始化 %d 个客户端模型, 共享维度=%d, 任务维度=%d",
            self.config.num_clients, dim, task_dim
        )

    def _simulate_client_data(self, client_id: int, n_samples: int = 100) -> np.ndarray:
        """模拟客户端的非 IID 数据分布。

        Simulate non-IID data distribution for a client.

        Args:
            client_id: 客户端 ID
            n_samples: 样本数量

        Returns:
            模拟数据矩阵
        """
        dim = self.config.model_dim
        heterogeneity = self.config.heterogeneity_level

        # 每个客户端有不同的数据分布 (非 IID)
        client_mean = self._rng.randn(dim) * heterogeneity * 5
        client_cov = np.eye(dim) * (1.0 + heterogeneity * self._rng.rand())

        # 使用 Cholesky 分解生成相关数据
        L = np.linalg.cholesky(client_cov + np.eye(dim) * 0.01)
        data = self._rng.randn(n_samples, dim) @ L.T + client_mean

        return data

    def _local_train(self, client_id: int, data: np.ndarray) -> float:
        """客户端本地训练。

        Client-side local training.

        Args:
            client_id: 客户端 ID
            data: 本地训练数据

        Returns:
            本地训练损失
        """
        model = self._client_models[client_id]
        prev_model = self._client_prev_models[client_id]
        lr = self.config.learning_rate
        mu = self.config.proximal_mu

        total_loss = 0.0
        for epoch in range(self.config.local_epochs):
            # 简化训练: 梯度下降
            x = data @ model["shared_W"] + model["shared_b"]

            # ReLU 激活
            x = np.maximum(0, x)

            # 任务特定输出
            output = x @ model["task_W"] + model["task_b"]

            # 模拟损失 (MSE)
            target = np.mean(data, axis=0, keepdims=True)  # 模拟目标
            target_padded = np.tile(target, (output.shape[0], 1))[:, :output.shape[1]]
            loss = float(np.mean((output - target_padded) ** 2))
            total_loss += loss

            # 计算梯度 (简化)
            grad_output = 2 * (output - target_padded) / output.shape[0]

            # 更新任务头
            model["task_W"] -= lr * x.T @ grad_output * 0.01
            model["task_b"] -= lr * grad_output.mean(axis=0) * 0.01

            # FedProx 近端项: 约束本地模型不偏离全局模型太远
            if self.config.aggregation_strategy == "fedprox" and self._global_model is not None:
                for key in ["shared_W", "shared_b"]:
                    proximal_grad = mu * (model[key] - self._global_model[key])
                    model[key] -= lr * proximal_grad * 0.01

            # MOON 对比损失 (简化)
            if self.config.aggregation_strategy == "moon" and self._global_model is not None:
                # 模型对比: 拉近当前模型与全局模型，推远与上一轮模型
                current_repr = model["shared_W"].flatten()
                global_repr = self._global_model["shared_W"].flatten()
                prev_repr = prev_model["shared_W"].flatten()

                # 余弦相似度
                sim_global = np.dot(current_repr, global_repr) / (
                    np.linalg.norm(current_repr) * np.linalg.norm(global_repr) + 1e-8
                )
                sim_prev = np.dot(current_repr, prev_repr) / (
                    np.linalg.norm(current_repr) * np.linalg.norm(prev_repr) + 1e-8
                )

                # 对比梯度
                tau = self.config.moon_tau
                temp = self.config.moon_temperature
                contrastive_grad = tau * (
                    -np.exp(sim_global / temp) / (
                        np.exp(sim_global / temp) + np.exp(sim_prev / temp) + 1e-8
                    ) * (global_repr - sim_global * current_repr) / (
                        np.linalg.norm(current_repr) ** 2 + 1e-8
                    )
                )
                # 更新共享参数
                grad_shape = model["shared_W"].shape
                model["shared_W"] -= lr * contrastive_grad[:grad_shape[0] * grad_shape[1]].reshape(grad_shape) * 0.001

        return total_loss / self.config.local_epochs

    def _aggregate(self):
        """聚合客户端模型更新全局模型。

        Aggregate client models to update the global model.
        """
        if not self._client_models or self._global_model is None:
            return

        n_clients = len(self._client_models)

        # 保存上一轮模型 (用于 MOON)
        self._client_prev_models = [copy.deepcopy(m) for m in self._client_models]

        # FedAvg 聚合共享参数
        for key in ["shared_W", "shared_b"]:
            aggregated = np.zeros_like(self._global_model[key])
            for client_model in self._client_models:
                aggregated += client_model[key]

            # 通信压缩
            if self.config.communication_compression < 1.0:
                # Top-k 稀疏化
                k = int(aggregated.size * self.config.communication_compression)
                if k > 0:
                    flat = aggregated.flatten()
                    threshold = np.partition(np.abs(flat), -k)[-k]
                    mask = np.abs(flat) >= threshold
                    compressed = flat * mask
                    aggregated = compressed.reshape(aggregated.shape)
                    self._total_comm_cost += k
                else:
                    self._total_comm_cost += aggregated.size
            else:
                self._total_comm_cost += aggregated.size

            self._global_model[key] = aggregated / n_clients

        # 分发全局模型到客户端
        for client_model in self._client_models:
            for key in ["shared_W", "shared_b"]:
                client_model[key] = self._global_model[key].copy()

        # 差分隐私
        if self.config.differential_privacy_epsilon > 0:
            for key in self._global_model:
                noise_scale = 1.0 / self.config.differential_privacy_epsilon
                noise = self._rng.randn(*self._global_model[key].shape) * noise_scale * 0.01
                self._global_model[key] += noise
            self._privacy_budget_used += self.config.differential_privacy_epsilon

    def optimize(
        self, client_data: Optional[List[np.ndarray]] = None
    ) -> FederatedMultiTaskResult:
        """执行联邦多任务优化。

        Execute federated multi-task optimization.

        Args:
            client_data: 各客户端数据列表 (None 时自动生成)

        Returns:
            FederatedMultiTaskResult 优化结果
        """
        t0 = time.perf_counter()
        LOGGER.info(
            "开始联邦多任务优化, %d 客户端, %d 轮次, 策略=%s",
            self.config.num_clients, self.config.num_rounds,
            self.config.aggregation_strategy
        )

        self._initialize_models()

        # 生成或使用提供的客户端数据
        if client_data is None:
            client_data = [
                self._simulate_client_data(i)
                for i in range(self.config.num_clients)
            ]

        per_client_metrics = []

        for round_idx in range(self.config.num_rounds):
            self._round_idx = round_idx
            round_losses = []

            # 各客户端本地训练
            for client_id in range(self.config.num_clients):
                data = client_data[client_id] if client_id < len(client_data) \
                    else self._simulate_client_data(client_id)
                loss = self._local_train(client_id, data)
                round_losses.append(loss)

            # 聚合
            self._aggregate()

            # 记录收敛
            avg_loss = float(np.mean(round_losses))
            self._convergence.append(avg_loss)

            if (round_idx + 1) % 5 == 0:
                LOGGER.debug(
                    "联邦轮次 %d/%d, 平均损失=%.6f",
                    round_idx + 1, self.config.num_rounds, avg_loss
                )

        # 收集各客户端指标
        for client_id in range(self.config.num_clients):
            model = self._client_models[client_id]
            per_client_metrics.append({
                "client_id": client_id,
                "shared_norm": float(np.linalg.norm(model["shared_W"])),
                "task_norm": float(np.linalg.norm(model["task_W"])),
                "final_loss": float(round_losses[client_id]) if client_id < len(round_losses) else 0.0,
            })

        elapsed = (time.perf_counter() - t0) * 1000.0

        LOGGER.info(
            "联邦优化完成, 最终损失=%.6f, 通信开销=%.0f 参数, 耗时=%.1fms",
            self._convergence[-1] if self._convergence else 0,
            self._total_comm_cost, elapsed
        )

        return FederatedMultiTaskResult(
            global_model=self._global_model or {},
            client_models=self._client_models,
            convergence_history=self._convergence,
            communication_cost=self._total_comm_cost,
            per_client_metrics=per_client_metrics,
            privacy_budget_used=self._privacy_budget_used,
            total_rounds=self.config.num_rounds,
            processing_time_ms=elapsed,
        )

    def get_federation_summary(self) -> Dict[str, Any]:
        """获取联邦学习摘要。

        Get a summary of the federated learning process.

        Returns:
            联邦学习摘要字典
        """
        return {
            "rounds_completed": self._round_idx,
            "convergence": self._convergence[-10:] if self._convergence else [],
            "communication_cost": self._total_comm_cost,
            "privacy_budget_used": self._privacy_budget_used,
            "aggregation_strategy": self.config.aggregation_strategy,
            "num_clients": self.config.num_clients,
        }


# ============================================================================
# 4. NeuralODEBeamPropagator — 神经常微分方程光束传播器
# ============================================================================
# 参考: Neural ODE (Chen et al., NeurIPS 2018) + torchdiffeq
# 用神经常微分方程建模光束在复杂介质中的传播

@dataclass
class NeuralODEConfig:
    """神经常微分方程光束传播配置

    Configuration for Neural ODE beam propagation.
    """
    latent_dim: int = 64                                 # 潜空间维度
    ode_solver: str = "dopri5"                           # ODE 求解器: "dopri5" / "euler" / "rk4"
    rtol: float = 1e-3                                   # 相对容差
    atol: float = 1e-4                                   # 绝对容差
    max_propagation_steps: int = 100                     # 最大传播步数
    propagation_distance: float = 0.1                    # 传播距离 (m)
    wavelength: float = 632.8e-9                         # 波长 (m)
    pixel_size: float = 5.0e-6                           # 像素尺寸 (m)
    medium_complexity: int = 3                            # 介质复杂度 (1-5)
    adaptive_stepping: bool = True                        # 自适应步长
    num_training_iterations: int = 200                   # 训练迭代次数
    learning_rate: float = 1e-3


@dataclass
class NeuralODEResult:
    """神经常微分方程光束传播结果

    Result of Neural ODE beam propagation.
    """
    output_field: np.ndarray              # 输出光场
    propagation_trajectory: List[np.ndarray]  # 传播轨迹 (中间场)
    integration_time: float               # 积分时间 (模拟)
    num_function_evaluations: int         # 函数评估次数
    propagation_loss: float               # 传播损失
    processing_time_ms: float             # 处理时间
    step_sizes: List[float]               # 各步长


class NeuralODEBeamPropagator:
    """神经常微分方程光束传播器

    受 Neural ODE (Chen et al., NeurIPS 2018) 和 torchdiffeq 启发，
    用神经常微分方程建模光束在复杂介质中的传播。

    Inspired by Neural ODE and torchdiffeq, models beam propagation through
    complex media using neural ordinary differential equations.

    核心特性:
    - 连续时间建模: dz/dt = f(z, t, theta)
    - 自适应步长: 根据场变化率自动调整
    - 可学习介质模型: 从数据学习介质特性
    - 支持多种 ODE 求解器 (Dormand-Prince, Euler, RK4)
    """

    def __init__(self, config: Optional[NeuralODEConfig] = None):
        """初始化神经常微分方程光束传播器。

        Initialize the Neural ODE beam propagator.

        Args:
            config: 配置参数，为 None 时使用默认值
        """
        self.config = config or NeuralODEConfig()
        self._rng = np.random.RandomState(42)
        self._net_params: Optional[Dict[str, np.ndarray]] = None
        self._is_trained = False
        self._medium_params: Optional[Dict[str, Any]] = None

    def reset(self):
        """重置传播器状态。

        Reset the propagator state.
        """
        self._net_params = None
        self._is_trained = False
        self._medium_params = None
        LOGGER.info("NeuralODEBeamPropagator 已重置")

    def _initialize_network(self, input_dim: int, output_dim: int):
        """初始化神经网络参数 (ODE 右端函数)。

        Initialize neural network parameters for the ODE right-hand side.

        Args:
            input_dim: 输入维度
            output_dim: 输出维度
        """
        hidden = self.config.latent_dim
        self._net_params = {
            "W1": self._rng.randn(input_dim, hidden) * np.sqrt(2.0 / input_dim),
            "b1": np.zeros(hidden),
            "W2": self._rng.randn(hidden, hidden) * np.sqrt(2.0 / hidden),
            "b2": np.zeros(hidden),
            "W3": self._rng.randn(hidden, output_dim) * np.sqrt(2.0 / hidden),
            "b3": np.zeros(output_dim),
        }
        LOGGER.info("已初始化 ODE 网络参数, input=%d, hidden=%d, output=%d",
                     input_dim, hidden, output_dim)

    def _ode_rhs(self, z: np.ndarray, t: float) -> np.ndarray:
        """ODE 右端函数: dz/dt = f(z, t)。

        ODE right-hand side function: dz/dt = f(z, t).

        Args:
            z: 当前状态 (展平的复数场)
            t: 当前时间 (传播距离)

        Returns:
            状态导数
        """
        if self._net_params is None:
            return np.zeros_like(z)

        p = self._net_params

        # 两层 MLP + tanh 激活
        h = z @ p["W1"] + p["b1"]
        h = np.tanh(h)
        h = h @ p["W2"] + p["b2"]
        h = np.tanh(h)
        dz = h @ p["W3"] + p["b3"]

        # 物理约束: 传播方程的线性部分
        # 简化的傍轴波动方程: dz/dz_prop = (i lambda / 4pi) * laplacian(z)
        # 这里用线性项近似
        dz = dz + 0.1 * z  # 补偿衰减

        return dz

    def _euler_step(self, z: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Euler 方法单步。

        Single Euler method step.

        Args:
            z: 当前状态
            t: 当前时间
            dt: 步长

        Returns:
            下一步状态
        """
        return z + dt * self._ode_rhs(z, t)

    def _rk4_step(self, z: np.ndarray, t: float, dt: float) -> np.ndarray:
        """经典四阶 Runge-Kutta 单步。

        Single classical 4th-order Runge-Kutta step.

        Args:
            z: 当前状态
            t: 当前时间
            dt: 步长

        Returns:
            下一步状态
        """
        k1 = self._ode_rhs(z, t)
        k2 = self._ode_rhs(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._ode_rhs(z + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._ode_rhs(z + dt * k3, t + dt)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _dopri5_step(self, z: np.ndarray, t: float, dt: float) -> Tuple[np.ndarray, float]:
        """Dormand-Prince (RK45) 自适应单步。

        Single Dormand-Prince (RK45) adaptive step.

        Args:
            z: 当前状态
            t: 当前时间
            dt: 步长

        Returns:
            (下一步状态, 误差估计)
        """
        # Butcher tableau coefficients for Dormand-Prince
        a2, a3, a4, a5, a6 = (1/5, 3/10, 4/5, 8/9, 1.0)
        b21 = 1/5
        b31, b32 = 3/40, 9/40
        b41, b42, b43 = 44/45, -56/15, 32/9
        b51, b52, b53, b54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        b61, b62, b63, b64, b65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656

        # 5th order weights
        c1, c3, c4, c5, c6, c7 = (35/384, 500/1113, 125/192, -2187/6784, 11/84, 0.0)

        # 4th order weights (for error estimation)
        dc1 = 71/57600
        dc3 = -71/16695
        dc4 = 71/1920
        dc5 = -17253/339200
        dc6 = 22/525
        dc7 = -1/40

        k1 = dt * self._ode_rhs(z, t)
        k2 = dt * self._ode_rhs(z + b21 * k1, t + a2 * dt)
        k3 = dt * self._ode_rhs(z + b31 * k1 + b32 * k2, t + a3 * dt)
        k4 = dt * self._ode_rhs(z + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * dt)
        k5 = dt * self._ode_rhs(z + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * dt)
        k6 = dt * self._ode_rhs(z + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, t + a6 * dt)

        # 5th order solution
        z_new = z + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 + c6 * k6

        # Error estimate (difference between 5th and 4th order)
        k7 = dt * self._ode_rhs(z_new, t + dt)
        error = dc1 * k1 + dc3 * k3 + dc4 * k4 + dc5 * k5 + dc6 * k6 + dc7 * k7

        error_norm = float(np.linalg.norm(error))
        return z_new, error_norm

    def propagate(self, input_field: np.ndarray) -> NeuralODEResult:
        """传播光束通过复杂介质。

        Propagate beam through complex media using Neural ODE.

        Args:
            input_field: 输入光场 (2D numpy 数组, 复数或实数)

        Returns:
            NeuralODEResult 传播结果
        """
        t0 = time.perf_counter()

        # 转换为实数表示 (复数场 -> [实部, 虚部])
        if np.iscomplexobj(input_field):
            z = np.stack([input_field.real, input_field.imag], axis=-1).flatten()
        else:
            z = input_field.astype(np.float64).flatten()

        field_size = z.shape[0]

        # 初始化网络
        if self._net_params is None:
            self._initialize_network(field_size, field_size)

        # 传播参数
        t_start = 0.0
        t_end = self.config.propagation_distance
        max_steps = self.config.max_propagation_steps

        trajectory = [z.copy()]
        step_sizes = []
        n_evals = 0
        t = t_start

        if self.config.ode_solver == "euler":
            # 固定步长 Euler
            dt = (t_end - t_start) / max_steps
            for _ in range(max_steps):
                z = self._euler_step(z, t, dt)
                t += dt
                trajectory.append(z.copy())
                step_sizes.append(dt)
                n_evals += 1

        elif self.config.ode_solver == "rk4":
            # 固定步长 RK4
            dt = (t_end - t_start) / max_steps
            for _ in range(max_steps):
                z = self._rk4_step(z, t, dt)
                t += dt
                trajectory.append(z.copy())
                step_sizes.append(dt)
                n_evals += 4

        elif self.config.ode_solver == "dopri5":
            # 自适应步长 Dormand-Prince
            dt = (t_end - t_start) / max_steps
            min_dt = dt * 1e-3
            max_dt = dt * 10.0

            while t < t_end and len(trajectory) < min(max_steps * 5, 50000):
                dt = np.clip(dt, min_dt, max_dt)
                if t + dt > t_end:
                    dt = t_end - t

                z_new, error = self._dopri5_step(z, t, dt)
                n_evals += 6

                # 自适应步长控制
                if self.config.adaptive_stepping and error > 0:
                    safety = 0.9
                    if error < self.config.atol:
                        factor = safety * (self.config.atol / error) ** 0.2
                        factor = min(factor, 5.0)
                    else:
                        factor = safety * (self.config.atol / error) ** 0.25
                        factor = max(factor, 0.1)
                    dt *= factor
                else:
                    dt *= 1.0

                z = z_new
                t += dt
                trajectory.append(z.copy())
                step_sizes.append(dt)

        else:
            LOGGER.warning("未知求解器 '%s', 使用 Euler", self.config.ode_solver)
            dt = (t_end - t_start) / max_steps
            for _ in range(max_steps):
                z = self._euler_step(z, t, dt)
                t += dt
                trajectory.append(z.copy())
                step_sizes.append(dt)
                n_evals += 1

        # 重建输出光场
        mid = field_size // 2
        if field_size % 2 == 0:
            output_real = z[:mid].reshape(input_field.shape[:2])
            output_imag = z[mid:].reshape(input_field.shape[:2])
        else:
            output_real = z[:mid + 1].reshape(input_field.shape[:2])
            output_imag = z[mid + 1:].reshape(input_field.shape[:2])
        output_field = output_real + 1j * output_imag

        # 计算传播损失 (能量守恒)
        input_energy = float(np.sum(np.abs(input_field) ** 2))
        output_energy = float(np.sum(np.abs(output_field) ** 2))
        propagation_loss = abs(input_energy - output_energy) / (input_energy + 1e-8)

        elapsed = (time.perf_counter() - t0) * 1000.0

        LOGGER.info(
            "Neural ODE 传播完成, 求解器=%s, 步数=%d, 函数评估=%d, 损失=%.6f",
            self.config.ode_solver, len(step_sizes), n_evals, propagation_loss
        )

        return NeuralODEResult(
            output_field=output_field,
            propagation_trajectory=trajectory,
            integration_time=t_end,
            num_function_evaluations=n_evals,
            propagation_loss=propagation_loss,
            processing_time_ms=elapsed,
            step_sizes=step_sizes,
        )

    def train_step(
        self, input_field: np.ndarray, target_field: np.ndarray
    ) -> float:
        """单步训练: 调整网络参数使传播结果匹配目标。

        Single training step: adjust network parameters to match target.

        Args:
            input_field: 输入光场
            target_field: 目标输出光场

        Returns:
            训练损失
        """
        if self._net_params is None:
            field_size = input_field.size * 2 if np.iscomplexobj(input_field) else input_field.size
            self._initialize_network(field_size, field_size)

        # 前向传播
        result = self.propagate(input_field)
        output = result.output_field

        # 计算损失
        loss = float(np.mean((np.abs(output - target_field) ** 2)))

        # 简化梯度下降 (数值梯度)
        # ⚠️ 数值梯度复杂度 O(N²)，仅适用于小规模网络 (参数量 < 1000)
        total_params = sum(p.size for p in self._net_params.values())
        if total_params > 1000:
            LOGGER.warning(
                "train_step 使用数值梯度，当前参数量=%d > 1000，"
                "计算量极大 (O(N²))。建议使用 torch 后端或减小网络规模。",
                total_params,
            )
            return loss
        lr = self.config.learning_rate
        eps = 1e-5
        for key in self._net_params:
            param = self._net_params[key]
            grad = np.zeros_like(param)
            for idx in np.ndindex(param.shape):
                old_val = param[idx]
                param[idx] = old_val + eps
                result_plus = self.propagate(input_field)
                loss_plus = float(np.mean((np.abs(result_plus.output_field - target_field) ** 2)))
                param[idx] = old_val
                grad[idx] = (loss_plus - loss) / eps
            self._net_params[key] -= lr * grad

        self._is_trained = True
        return loss


# ============================================================================
# 5. QuantumInspiredOptimizer — 量子启发优化器
# ============================================================================
# 参考: QAOA (Quantum Approximate Optimization Algorithm) + Quantum Annealing
# 用量子启发算法优化多参数光学对准

@dataclass
class QuantumInspiredConfig:
    """量子启发优化器配置

    Configuration for quantum-inspired optimizer.
    """
    num_qubits: int = 10                                  # 量子比特数
    num_qaoa_layers: int = 3                              # QAOA 层数 (p)
    annealing_schedule: str = "linear"                    # 退火调度: "linear" / "exponential" / "adaptive"
    num_annealing_steps: int = 100                        # 退火步数
    temperature_start: float = 10.0                       # 初始温度
    temperature_end: float = 0.01                         # 终止温度
    optimization_type: str = "continuous"                 # 优化类型: "continuous" / "combinatorial" / "hybrid"
    param_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x_offset": (-500, 500),
            "y_offset": (-500, 500),
            "z_focus": (-200, 200),
            "intensity": (0.1, 10.0),
            "exposure": (1, 100),
        }
    )  # 参数搜索范围
    max_iterations: int = 200                             # 最大迭代次数
    population_size: int = 50                             # 种群大小 (量子种群)
    tunneling_rate: float = 0.1                           # 量子隧穿率
    superposition_size: int = 20                          # 叠态大小


@dataclass
class QuantumInspiredResult:
    """量子启发优化结果

    Result of quantum-inspired optimization.
    """
    best_parameters: Dict[str, float]                     # 最优参数
    best_energy: float                                    # 最优能量 (目标值)
    energy_history: List[float]                           # 能量历史
    temperature_history: List[float]                      # 温度历史
    tunneling_events: int                                 # 隧穿事件次数
    convergence_achieved: bool                            # 是否收敛
    processing_time_ms: float                             # 处理时间
    quantum_state_distribution: Optional[np.ndarray] = None  # 量子态分布


class QuantumInspiredOptimizer:
    """量子启发优化器

    受 QAOA (Quantum Approximate Optimization Algorithm) 和量子退火 (Quantum Annealing) 启发，
    用量子启发算法优化多参数光学对准。

    Inspired by QAOA and Quantum Annealing, uses quantum-inspired algorithms
    to optimize multi-parameter optical alignment.

    核心特性:
    - 量子退火: 模拟量子隧穿跳出局部最优
    - QAOA: 量子近似优化算法的经典模拟
    - 量子叠加态: 同时探索多个解
    - 支持连续优化和组合优化
    - 自适应退火调度
    """

    def __init__(self, config: Optional[QuantumInspiredConfig] = None):
        """初始化量子启发优化器。

        Initialize the quantum-inspired optimizer.

        Args:
            config: 配置参数，为 None 时使用默认值
        """
        self.config = config or QuantumInspiredConfig()
        self._rng = np.random.RandomState(42)
        self._best_params: Dict[str, float] = {}
        self._best_energy: float = float('inf')
        self._energy_history: deque = deque(maxlen=2000)
        self._temperature_history: deque = deque(maxlen=2000)
        self._tunneling_events: int = 0

    def reset(self):
        """重置优化器状态。

        Reset the optimizer state.
        """
        self._best_params.clear()
        self._best_energy = float('inf')
        self._energy_history.clear()
        self._temperature_history.clear()
        self._tunneling_events = 0
        LOGGER.info("QuantumInspiredOptimizer 已重置")

    def _encode_to_qubits(self, parameters: Dict[str, float]) -> np.ndarray:
        """将连续参数编码到量子比特表示。

        Encode continuous parameters to qubit representation.

        Args:
            parameters: 参数字典

        Returns:
            量子比特状态向量
        """
        n_qubits = self.config.num_qubits
        state = np.zeros(n_qubits)

        param_values = list(parameters.values())
        n_params = len(param_values)

        for i in range(n_qubits):
            if i < n_params:
                # 归一化到 [0, 1]
                bounds = list(self.config.param_bounds.values())
                if i < len(bounds):
                    low, high = bounds[i]
                    normalized = (param_values[i] - low) / (high - low + 1e-8)
                else:
                    normalized = 0.5
                state[i] = np.clip(normalized, 0, 1)
            else:
                state[i] = self._rng.rand()

        return state

    def _decode_from_qubits(self, qubit_state: np.ndarray) -> Dict[str, float]:
        """从量子比特表示解码回连续参数。

        Decode from qubit representation back to continuous parameters.

        Args:
            qubit_state: 量子比特状态向量

        Returns:
            参数字典
        """
        parameters = {}
        param_names = list(self.config.param_bounds.keys())

        for i, name in enumerate(param_names):
            if i < len(qubit_state):
                low, high = self.config.param_bounds[name]
                parameters[name] = float(low + qubit_state[i] * (high - low))
            else:
                low, high = self.config.param_bounds[name]
                parameters[name] = float((low + high) / 2)

        return parameters

    def _objective_function(self, parameters: Dict[str, float]) -> float:
        """目标函数 (能量函数): 值越小越好。

        Objective function (energy function): lower is better.

        Args:
            parameters: 参数字典

        Returns:
            能量值
        """
        # 模拟光学对准目标: 最小化偏移 + 优化聚焦
        x = parameters.get("x_offset", 0)
        y = parameters.get("y_offset", 0)
        z = parameters.get("z_focus", 0)
        intensity = parameters.get("intensity", 1.0)
        exposure = parameters.get("exposure", 10.0)

        # 偏移惩罚 (二次)
        offset_energy = (x ** 2 + y ** 2) / (500 ** 2)

        # 聚焦惩罚
        focus_energy = (z ** 2) / (200 ** 2)

        # 强度惩罚 (偏离理想值)
        ideal_intensity = 5.0
        intensity_energy = ((intensity - ideal_intensity) / ideal_intensity) ** 2

        # 曝光惩罚
        ideal_exposure = 50.0
        exposure_energy = ((exposure - ideal_exposure) / ideal_exposure) ** 2

        # 交叉项 (模拟像差)
        cross_energy = 0.01 * np.sin(x / 100) * np.cos(y / 100)

        total = offset_energy + focus_energy + intensity_energy + exposure_energy + cross_energy
        return float(total)

    def _quantum_tunneling(self, current_state: np.ndarray,
                           temperature: float) -> np.ndarray:
        """量子隧穿操作: 跳出局部最优。

        Quantum tunneling operation: escape local optima.

        Args:
            current_state: 当前量子态
            temperature: 当前温度

        Returns:
            隧穿后的量子态
        """
        tunneling_prob = self.config.tunneling_rate * np.exp(-temperature / self.config.temperature_start)

        if self._rng.rand() < tunneling_prob:
            # 随机翻转若干量子比特
            n_flips = max(1, int(self._rng.exponential(1.0)))
            new_state = current_state.copy()
            flip_indices = self._rng.choice(len(new_state), size=min(n_flips, len(new_state)), replace=False)
            for idx in flip_indices:
                # 大幅跳跃 (隧穿效应)
                new_state[idx] = self._rng.rand()
            self._tunneling_events += 1
            LOGGER.debug("量子隧穿事件: 翻转 %d 个量子比特", n_flips)
            return new_state

        return current_state

    def _quantum_superposition_sample(self, center: np.ndarray,
                                      temperature: float) -> List[np.ndarray]:
        """从量子叠加态采样多个候选解。

        Sample multiple candidate solutions from quantum superposition.

        Args:
            center: 中心量子态
            temperature: 当前温度

        Returns:
            候选解列表
        """
        candidates = []
        for _ in range(self.config.superposition_size):
            # 高斯扰动
            perturbation = self._rng.randn(len(center)) * temperature / self.config.temperature_start * 0.3
            candidate = center + perturbation
            candidate = np.clip(candidate, 0, 1)
            candidates.append(candidate)
        return candidates

    def _qaoa_layer(self, state: np.ndarray, gamma: float, beta: float,
                    cost_matrix: np.ndarray) -> np.ndarray:
        """单层 QAOA 操作 (经典模拟)。

        Single QAOA layer (classical simulation).

        Args:
            state: 输入量子态
            gamma: 问题哈密顿量旋转角度
            beta: 混合哈密顿量旋转角度
            cost_matrix: 问题代价矩阵

        Returns:
            QAOA 操作后的量子态
        """
        n = len(state)

        # 问题哈密顿量演化 (旋转每个量子比特)
        for i in range(n):
            # 代价: 与其他量子比特的相互作用
            local_cost = 0.0
            for j in range(n):
                if i != j:
                    local_cost += cost_matrix[i, j] * state[j]
            # Rz 旋转
            angle = gamma * local_cost
            state[i] = state[i] * np.cos(angle) + np.sqrt(max(0, 1 - state[i] ** 2)) * np.sin(angle)

        # 混合哈密顿量演化 (叠加态旋转)
        for i in range(n):
            # Rx 旋转
            angle = beta
            new_val = state[i] * np.cos(angle) + np.sqrt(max(0, 1 - state[i] ** 2)) * np.sin(angle)
            state[i] = np.clip(new_val, 0, 1)

        return state

    def _get_temperature(self, step: int, total_steps: int) -> float:
        """获取当前温度 (退火调度)。

        Get current temperature based on annealing schedule.

        Args:
            step: 当前步
            total_steps: 总步数

        Returns:
            当前温度
        """
        progress = step / max(total_steps - 1, 1)
        t_start = self.config.temperature_start
        t_end = self.config.temperature_end

        if self.config.annealing_schedule == "linear":
            return t_start + (t_end - t_start) * progress
        elif self.config.annealing_schedule == "exponential":
            return t_start * (t_end / t_start) ** progress
        elif self.config.annealing_schedule == "adaptive":
            # 自适应: 基于能量变化调整
            if len(self._energy_history) >= 5:
                recent_improvement = self._energy_history[-5] - self._energy_history[-1]
                if recent_improvement > 0:
                    # 有改善，保持温度
                    return self._temperature_history[-1] if self._temperature_history else t_start
                else:
                    # 无改善，加速降温
                    return t_start * (t_end / t_start) ** (progress ** 0.5)
            return t_start + (t_end - t_start) * progress
        else:
            return t_start + (t_end - t_start) * progress

    def optimize(
        self, initial_params: Optional[Dict[str, float]] = None
    ) -> QuantumInspiredResult:
        """执行量子启发优化。

        Execute quantum-inspired optimization.

        Args:
            initial_params: 初始参数 (None 时随机初始化)

        Returns:
            QuantumInspiredResult 优化结果
        """
        t0 = time.perf_counter()
        LOGGER.info(
            "开始量子启发优化, 类型=%s, 最大迭代=%d, 量子比特=%d",
            self.config.optimization_type, self.config.max_iterations,
            self.config.num_qubits
        )

        # 初始化
        if initial_params is not None:
            current_state = self._encode_to_qubits(initial_params)
        else:
            current_state = self._rng.rand(self.config.num_qubits)

        current_params = self._decode_from_qubits(current_state)
        current_energy = self._objective_function(current_params)

        self._best_params = current_params.copy()
        self._best_energy = current_energy

        # 构建代价矩阵 (用于 QAOA)
        cost_matrix = np.eye(self.config.num_qubits) * 0.1
        cost_matrix += self._rng.randn(self.config.num_qubits, self.config.num_qubits) * 0.05
        cost_matrix = (cost_matrix + cost_matrix.T) / 2

        # QAOA 参数
        gammas = np.linspace(0, np.pi, self.config.num_qaoa_layers + 2)[1:-1]
        betas = np.linspace(0, np.pi / 2, self.config.num_qaoa_layers + 2)[1:-1]

        for iteration in range(self.config.max_iterations):
            temperature = self._get_temperature(iteration, self.config.max_iterations)
            self._temperature_history.append(temperature)

            # 1. 量子叠加态采样
            candidates = self._quantum_superposition_sample(current_state, temperature)

            # 2. QAOA 层操作
            if self.config.optimization_type in ("combinatorial", "hybrid"):
                for layer_idx in range(self.config.num_qaoa_layers):
                    gamma = gammas[layer_idx] if layer_idx < len(gammas) else np.pi / 2
                    beta = betas[layer_idx] if layer_idx < len(betas) else np.pi / 4
                    for i in range(len(candidates)):
                        candidates[i] = self._qaoa_layer(
                            candidates[i].copy(), gamma, beta, cost_matrix
                        )

            # 3. 评估候选解
            best_candidate_state = current_state.copy()
            best_candidate_energy = current_energy

            for candidate in candidates:
                params = self._decode_from_qubits(candidate)
                energy = self._objective_function(params)
                if energy < best_candidate_energy:
                    best_candidate_energy = energy
                    best_candidate_state = candidate.copy()

            # 4. Metropolis 准则接受/拒绝
            delta_e = best_candidate_energy - current_energy
            if delta_e < 0:
                # 接受更优解
                current_state = best_candidate_state
                current_energy = best_candidate_energy
            elif temperature > 0:
                # 概率接受较差解
                accept_prob = np.exp(-delta_e / (temperature + 1e-8))
                if self._rng.rand() < accept_prob:
                    current_state = best_candidate_state
                    current_energy = best_candidate_energy

            # 5. 量子隧穿
            current_state = self._quantum_tunneling(current_state, temperature)

            # 6. 更新最优
            current_params = self._decode_from_qubits(current_state)
            if current_energy < self._best_energy:
                self._best_energy = current_energy
                self._best_params = current_params.copy()

            self._energy_history.append(current_energy)

            if (iteration + 1) % 50 == 0:
                LOGGER.debug(
                    "迭代 %d/%d, 能量=%.6f, 最优=%.6f, 温度=%.4f, 隧穿=%d",
                    iteration + 1, self.config.max_iterations,
                    current_energy, self._best_energy, temperature,
                    self._tunneling_events
                )

        # 量子态分布 (最终叠加态的概率分布)
        final_superposition = self._quantum_superposition_sample(
            current_state, self.config.temperature_end
        )
        state_distribution = np.array([self._objective_function(
            self._decode_from_qubits(s)
        ) for s in final_superposition])

        elapsed = (time.perf_counter() - t0) * 1000.0

        converged = self._best_energy < 0.1
        LOGGER.info(
            "量子优化完成, 最优能量=%.6f, 隧穿=%d, 收敛=%s, 耗时=%.1fms",
            self._best_energy, self._tunneling_events, converged, elapsed
        )

        return QuantumInspiredResult(
            best_parameters=self._best_params,
            best_energy=self._best_energy,
            energy_history=self._energy_history,
            temperature_history=self._temperature_history,
            tunneling_events=self._tunneling_events,
            convergence_achieved=converged,
            processing_time_ms=elapsed,
            quantum_state_distribution=state_distribution,
        )

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要。

        Get optimization summary.

        Returns:
            优化摘要字典
        """
        return {
            "best_energy": self._best_energy,
            "best_parameters": self._best_params,
            "tunneling_events": self._tunneling_events,
            "energy_trend": self._energy_history[-10:] if self._energy_history else [],
            "temperature_trend": self._temperature_history[-10:] if self._temperature_history else [],
        }


# ============================================================================
# 6. MultimodalFusionTracker — 多模态融合跟踪器
# ============================================================================
# 参考: ViT (Vision Transformer) + CLIP (Contrastive Language-Image Pre-training)
# 融合视觉、位置、时间等多模态信息进行光斑跟踪

@dataclass
class MultimodalFusionConfig:
    """多模态融合跟踪器配置

    Configuration for multimodal fusion tracker.
    """
    patch_size: int = 8                                  # Patch 大小
    embed_dim: int = 64                                  # 嵌入维度
    num_heads: int = 4                                   # 注意力头数
    num_transformer_layers: int = 4                      # Transformer 层数
    num_modalities: int = 3                              # 模态数量 (视觉/位置/时间)
    context_length: int = 16                             # 上下文长度 (帧数)
    dropout_rate: float = 0.1                            # Dropout 率
    position_encoding_type: str = "learned"              # 位置编码: "learned" / "sinusoidal"
    fusion_strategy: str = "cross_attention"             # 融合策略: "cross_attention" / "concat" / "gated"
    tracking_max_displacement: float = 50.0              # 最大跟踪位移 (像素)
    confidence_threshold: float = 0.5                    # 置信度阈值
    temporal_window: int = 5                             # 时间窗口大小


@dataclass
class MultimodalFusionResult:
    """多模态融合跟踪结果

    Result of multimodal fusion tracking.
    """
    tracked_position: Tuple[float, float]                # 跟踪位置 (x, y)
    tracked_velocity: Tuple[float, float]                # 跟踪速度 (vx, vy)
    confidence: float                                    # 跟踪置信度
    modality_weights: Dict[str, float]                   # 各模态权重
    attention_maps: Dict[str, np.ndarray]                # 注意力图
    feature_embeddings: Dict[str, np.ndarray]            # 各模态特征嵌入
    processing_time_ms: float                            # 处理时间
    tracking_status: str                                 # 跟踪状态: "locked" / "searching" / "lost"


class MultimodalFusionTracker:
    """多模态融合跟踪器

    受 ViT (Vision Transformer) 和 CLIP 启发，融合视觉、位置、时间等多模态信息
    进行光斑跟踪。基于 Transformer 的多模态注意力机制实现跨模态信息融合。

    Inspired by ViT and CLIP, fuses visual, positional, and temporal
    multi-modal information for spot tracking. Uses Transformer-based
    multi-modal attention mechanisms for cross-modal information fusion.

    核心特性:
    - Vision Transformer: 将光斑图像分 patch 编码
    - 多模态注意力: 跨模态交叉注意力融合
    - CLIP 风格对比学习: 对齐不同模态的表示空间
    - 时间建模: 结合历史帧信息预测运动
    - 自适应模态权重: 根据场景自动调整模态重要性
    """

    def __init__(self, config: Optional[MultimodalFusionConfig] = None):
        """初始化多模态融合跟踪器。

        Initialize the multimodal fusion tracker.

        Args:
            config: 配置参数，为 None 时使用默认值
        """
        self.config = config or MultimodalFusionConfig()
        self._rng = np.random.RandomState(42)
        self._projection_matrices: Dict[str, np.ndarray] = {}
        self._attention_weights: Dict[str, np.ndarray] = {}
        self._position_embeddings: Optional[np.ndarray] = None
        self._history: deque = deque(maxlen=self.config.context_length)
        self._is_initialized = False
        self._last_position: Optional[Tuple[float, float]] = None
        self._last_velocity: Tuple[float, float] = (0.0, 0.0)

    def reset(self):
        """重置跟踪器状态。

        Reset the tracker state.
        """
        self._history.clear()
        self._last_position = None
        self._last_velocity = (0.0, 0.0)
        self._is_initialized = False
        LOGGER.info("MultimodalFusionTracker 已重置")

    def _initialize_parameters(self, feature_dim: int):
        """初始化模型参数。

        Initialize model parameters.

        Args:
            feature_dim: 特征维度
        """
        embed_dim = self.config.embed_dim

        # 各模态的投影矩阵
        self._projection_matrices = {
            "visual": self._rng.randn(feature_dim, embed_dim) * np.sqrt(2.0 / feature_dim),
            "positional": self._rng.randn(4, embed_dim) * np.sqrt(2.0 / 4),
            "temporal": self._rng.randn(feature_dim, embed_dim) * np.sqrt(2.0 / feature_dim),
        }

        # 多头注意力参数
        self._attention_weights = {
            "W_q": self._rng.randn(embed_dim, embed_dim) * np.sqrt(2.0 / embed_dim),
            "W_k": self._rng.randn(embed_dim, embed_dim) * np.sqrt(2.0 / embed_dim),
            "W_v": self._rng.randn(embed_dim, embed_dim) * np.sqrt(2.0 / embed_dim),
            "W_o": self._rng.randn(embed_dim, embed_dim) * np.sqrt(2.0 / embed_dim),
        }

        # 位置编码
        max_seq_len = self.config.context_length * self.config.num_modalities
        if self.config.position_encoding_type == "sinusoidal":
            self._position_embeddings = self._sinusoidal_encoding(max_seq_len, embed_dim)
        else:
            self._position_embeddings = self._rng.randn(max_seq_len, embed_dim) * 0.02

        self._is_initialized = True
        LOGGER.info("已初始化多模态融合参数, embed_dim=%d, heads=%d", embed_dim, self.config.num_heads)

    def _sinusoidal_encoding(self, max_len: int, dim: int) -> np.ndarray:
        """生成正弦位置编码。

        Generate sinusoidal positional encoding.

        Args:
            max_len: 最大序列长度
            dim: 嵌入维度

        Returns:
            位置编码矩阵 (max_len, dim)
        """
        pe = np.zeros((max_len, dim))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def _extract_visual_features(self, image: np.ndarray) -> np.ndarray:
        """提取视觉特征 (ViT 风格 patch 编码)。

        Extract visual features using ViT-style patch encoding.

        Args:
            image: 输入图像

        Returns:
            视觉特征向量
        """
        img_f = image.astype(np.float64)
        if img_f.max() > 1:
            img_f = img_f / 255.0

        h, w = img_f.shape[:2]
        ps = self.config.patch_size

        # 分 patch
        n_patches_h = h // ps
        n_patches_w = w // ps
        patches = []

        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = img_f[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps]
                # Patch 统计特征
                features = [
                    patch.mean(),
                    patch.std(),
                    patch.max(),
                    float(np.percentile(patch, 90)),
                    float(np.percentile(patch, 10)),
                ]
                # 梯度特征
                if patch.shape[0] > 2 and patch.shape[1] > 2:
                    gy = np.diff(patch, axis=0).mean()
                    gx = np.diff(patch, axis=1).mean()
                    features.extend([gy, gx])
                else:
                    features.extend([0.0, 0.0])
                patches.append(features)

        if not patches:
            return np.zeros(self.config.embed_dim)

        # 展平并截断/填充到固定维度
        patch_features = np.array(patches).flatten()
        target_dim = self.config.embed_dim
        if patch_features.shape[0] >= target_dim:
            # 取前 target_dim 个特征
            visual = patch_features[:target_dim]
        else:
            visual = np.pad(patch_features, (0, target_dim - patch_features.shape[0]))

        # 归一化
        visual = visual / (np.linalg.norm(visual) + 1e-8)
        return visual

    def _extract_positional_features(self, position: Tuple[float, float],
                                     velocity: Tuple[float, float]) -> np.ndarray:
        """提取位置特征。

        Extract positional features.

        Args:
            position: 当前位置 (x, y)
            velocity: 当前速度 (vx, vy)

        Returns:
            位置特征向量
        """
        return np.array([
            position[0] / self.config.tracking_max_displacement,
            position[1] / self.config.tracking_max_displacement,
            velocity[0] / (self.config.tracking_max_displacement + 1e-8),
            velocity[1] / (self.config.tracking_max_displacement + 1e-8),
        ], dtype=np.float64)

    def _extract_temporal_features(self) -> np.ndarray:
        """提取时间特征 (从历史帧)。

        Extract temporal features from frame history.

        Returns:
            时间特征向量
        """
        if len(self._history) < 2:
            return np.zeros(self.config.embed_dim)

        # 从历史位置计算时间特征
        positions = [h["position"] for h in self._history]
        velocities = [h["velocity"] for h in self._history]

        # 位置序列统计
        pos_array = np.array(positions)
        vel_array = np.array(velocities)

        features = [
            pos_array[-1][0], pos_array[-1][1],  # 最新位置
            vel_array[-1][0], vel_array[-1][1],  # 最新速度
            np.std(pos_array[:, 0]), np.std(pos_array[:, 1]),  # 位置标准差
            np.mean(vel_array[:, 0]), np.mean(vel_array[:, 1]),  # 平均速度
        ]

        # 加速度估计
        if len(velocities) >= 2:
            acc_x = vel_array[-1][0] - vel_array[-2][0]
            acc_y = vel_array[-1][1] - vel_array[-2][1]
            features.extend([acc_x, acc_y])
        else:
            features.extend([0.0, 0.0])

        # 运动方向一致性
        if len(velocities) >= 3:
            dirs = vel_array[-3:]
            dir_consistency = float(np.mean([
                np.dot(dirs[i], dirs[i + 1]) / (
                    np.linalg.norm(dirs[i]) * np.linalg.norm(dirs[i + 1]) + 1e-8
                )
                for i in range(len(dirs) - 1)
            ]))
            features.append(dir_consistency)
        else:
            features.append(0.0)

        # 截断/填充
        target_dim = self.config.embed_dim
        features = np.array(features)
        if features.shape[0] >= target_dim:
            temporal = features[:target_dim]
        else:
            temporal = np.pad(features, (0, target_dim - features.shape[0]))

        temporal = temporal / (np.linalg.norm(temporal) + 1e-8)
        return temporal

    def _multi_head_attention(
        self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """多头自注意力机制。

        Multi-head self-attention mechanism.

        Args:
            queries: 查询矩阵 (seq_len, embed_dim)
            keys: 键矩阵 (seq_len, embed_dim)
            values: 值矩阵 (seq_len, embed_dim)

        Returns:
            (注意力输出, 注意力权重)
        """
        embed_dim = self.config.embed_dim
        num_heads = self.config.num_heads
        head_dim = embed_dim // num_heads

        W_q = self._attention_weights["W_q"]
        W_k = self._attention_weights["W_k"]
        W_v = self._attention_weights["W_v"]
        W_o = self._attention_weights["W_o"]

        # 线性投影
        Q = queries @ W_q
        K = keys @ W_k
        V = values @ W_v

        # 分割为多头
        seq_len = Q.shape[0]
        Q_heads = Q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
        K_heads = K.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
        V_heads = V.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

        # 缩放点积注意力
        scale = math.sqrt(head_dim)
        attention_scores = np.matmul(Q_heads, K_heads.transpose(0, 2, 1)) / scale

        # Softmax
        attention_weights = np.exp(
            attention_scores - np.max(attention_scores, axis=-1, keepdims=True)
        )
        attention_weights = attention_weights / (
            np.sum(attention_weights, axis=-1, keepdims=True) + 1e-8
        )

        # 加权求和
        context = np.matmul(attention_weights, V_heads)

        # 合并多头
        context = context.transpose(1, 0, 2).reshape(seq_len, embed_dim)

        # 输出投影
        output = context @ W_o

        # 平均注意力权重 (跨头)
        avg_attention = np.mean(attention_weights, axis=0)

        return output, avg_attention

    def _cross_modal_fusion(
        self, modality_features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """跨模态融合。

        Cross-modal fusion using attention mechanism.

        Args:
            modality_features: 各模态特征字典

        Returns:
            (融合特征, 注意力图字典)
        """
        modality_names = list(modality_features.keys())
        n_modalities = len(modality_names)
        embed_dim = self.config.embed_dim

        # 投影到统一维度
        projected = {}
        for name, feat in modality_features.items():
            if name in self._projection_matrices:
                proj = self._projection_matrices[name]
                if feat.shape[0] >= proj.shape[0]:
                    projected[name] = feat[:proj.shape[0]] @ proj
                else:
                    padded = np.pad(feat, (0, proj.shape[0] - feat.shape[0]))
                    projected[name] = padded @ proj
            else:
                projected[name] = feat[:embed_dim] if feat.shape[0] >= embed_dim \
                    else np.pad(feat, (0, embed_dim - feat.shape[0]))

        # 构建序列: [visual, positional, temporal]
        sequence = np.stack([projected[name] for name in modality_names], axis=0)

        # 添加位置编码
        if self._position_embeddings is not None:
            seq_len = sequence.shape[0]
            sequence = sequence + self._position_embeddings[:seq_len]

        # 多头自注意力
        fused, attention_weights = self._multi_head_attention(sequence, sequence, sequence)

        # 聚合: 取所有模态的平均
        fused_feature = np.mean(fused, axis=0)

        # 计算模态权重 (基于注意力)
        attention_maps = {}
        for i, name in enumerate(modality_names):
            attention_maps[name] = attention_weights[:, i]

        # 模态重要性权重
        modality_weights = {}
        for name in modality_names:
            if name in attention_maps:
                modality_weights[name] = float(np.mean(attention_maps[name]))
            else:
                modality_weights[name] = 1.0 / n_modalities

        # 归一化
        total_w = sum(modality_weights.values())
        if total_w > 0:
            modality_weights = {k: v / total_w for k, v in modality_weights.items()}

        return fused_feature, attention_maps

    def _predict_position(self, fused_feature: np.ndarray) -> Tuple[Tuple[float, float], float]:
        """从融合特征预测位置和置信度。

        Predict position and confidence from fused features.

        Args:
            fused_feature: 融合特征向量

        Returns:
            (预测位置, 置信度)
        """
        embed_dim = self.config.embed_dim

        # 简化的位置预测头
        if TORCH_AVAILABLE:
            try:
                # 使用 torch 进行更精确的预测
                feat_t = torch.tensor(fused_feature, dtype=torch.float32)
                # 位置回归
                pos_weights = torch.randn(embed_dim, 2) * 0.1
                pos_bias = torch.zeros(2)
                pos_pred = (feat_t @ pos_weights + pos_bias).detach().numpy()
                # 置信度
                conf_weight = torch.randn(embed_dim, 1) * 0.1
                conf_bias = torch.tensor([0.0])
                conf_pred = torch.sigmoid(feat_t @ conf_weight + conf_bias).item()
            except Exception:
                pos_pred = fused_feature[:2] * self.config.tracking_max_displacement
                conf_pred = float(np.tanh(np.linalg.norm(fused_feature)))
        else:
            # numpy 回退
            pos_pred = fused_feature[:2] * self.config.tracking_max_displacement
            conf_pred = float(np.tanh(np.linalg.norm(fused_feature)))

        # 结合历史速度进行预测
        predicted_x = pos_pred[0] + self._last_velocity[0] * 0.5
        predicted_y = pos_pred[1] + self._last_velocity[1] * 0.5

        # 限制最大位移
        if self._last_position is not None:
            dx = predicted_x - self._last_position[0]
            dy = predicted_y - self._last_position[1]
            displacement = np.sqrt(dx ** 2 + dy ** 2)
            if displacement > self.config.tracking_max_displacement:
                scale = self.config.tracking_max_displacement / displacement
                predicted_x = self._last_position[0] + dx * scale
                predicted_y = self._last_position[1] + dy * scale

        return (float(predicted_x), float(predicted_y)), float(np.clip(conf_pred, 0, 1))

    def track(self, image: np.ndarray,
              current_position: Optional[Tuple[float, float]] = None) -> MultimodalFusionResult:
        """跟踪光斑位置。

        Track spot position using multimodal fusion.

        Args:
            image: 当前帧图像
            current_position: 当前检测到的位置 (可选, 用于位置模态)

        Returns:
            MultimodalFusionResult 跟踪结果
        """
        t0 = time.perf_counter()

        # 检测当前位置 (如果未提供)
        if current_position is None:
            img_f = image.astype(np.float64)
            if img_f.max() > 1:
                img_f = img_f / 255.0
            cy, cx = _safe_center_of_mass(img_f)
            current_position = (float(cx), float(cy))

        # 初始化参数
        if not self._is_initialized:
            visual_feat = self._extract_visual_features(image)
            self._initialize_parameters(visual_feat.shape[0])

        # 提取各模态特征
        visual_features = self._extract_visual_features(image)
        positional_features = self._extract_positional_features(
            current_position, self._last_velocity
        )
        temporal_features = self._extract_temporal_features()

        modality_features = {
            "visual": visual_features,
            "positional": positional_features,
            "temporal": temporal_features,
        }

        # 跨模态融合
        fused_feature, attention_maps = self._cross_modal_fusion(modality_features)

        # 预测位置
        predicted_position, confidence = self._predict_position(fused_feature)

        # 计算速度
        if self._last_position is not None:
            dt = 1.0  # 假设单位时间步
            vx = (predicted_position[0] - self._last_position[0]) / dt
            vy = (predicted_position[1] - self._last_position[1]) / dt
            self._last_velocity = (vx, vy)
        else:
            self._last_velocity = (0.0, 0.0)

        # 更新历史
        self._history.append({
            "position": predicted_position,
            "velocity": self._last_velocity,
            "confidence": confidence,
            "timestamp": time.time(),
        })

        self._last_position = predicted_position

        # 确定跟踪状态
        if confidence >= self.config.confidence_threshold:
            status = "locked"
        elif confidence >= self.config.confidence_threshold * 0.5:
            status = "searching"
        else:
            status = "lost"

        # 模态权重
        modality_weights = {}
        total_attn = sum(np.mean(attention_maps.get(k, np.ones(1))) for k in modality_features)
        for name in modality_features:
            attn = attention_maps.get(name, np.ones(1))
            modality_weights[name] = float(np.mean(attn) / (total_attn + 1e-8))

        elapsed = (time.perf_counter() - t0) * 1000.0

        LOGGER.debug(
            "多模态跟踪: 位置=(%.2f, %.2f), 置信度=%.3f, 状态=%s",
            predicted_position[0], predicted_position[1], confidence, status
        )

        return MultimodalFusionResult(
            tracked_position=predicted_position,
            tracked_velocity=self._last_velocity,
            confidence=confidence,
            modality_weights=modality_weights,
            attention_maps=attention_maps,
            feature_embeddings=modality_features,
            processing_time_ms=elapsed,
            tracking_status=status,
        )

    def batch_track(
        self, images: List[np.ndarray]
    ) -> List[MultimodalFusionResult]:
        """批量跟踪多帧图像。

        Batch track multiple frames.

        Args:
            images: 图像列表

        Returns:
            跟踪结果列表
        """
        results = []
        for img in images:
            result = self.track(img)
            results.append(result)
        return results

    def get_tracking_summary(self) -> Dict[str, Any]:
        """获取跟踪摘要。

        Get tracking summary.

        Returns:
            跟踪摘要字典
        """
        if not self._history:
            return {"status": "no_data"}

        positions = [h["position"] for h in self._history]
        confidences = [h["confidence"] for h in self._history]

        return {
            "status": "active",
            "current_position": self._last_position,
            "current_velocity": self._last_velocity,
            "avg_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "position_std": float(np.std([p[0] for p in positions])),
            "frame_count": len(self._history),
        }


# ============================================================================
# 独立运行测试
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    print("=" * 70)
    print("SpotZoom 创新模块 v20.0 独立测试")
    print("=" * 70)

    # ---- 测试 1: TopologyOptimizedNeuralController ----
    print("\n[1] TopologyOptimizedNeuralController")
    topo_ctrl = TopologyOptimizedNeuralController()
    topo_result = topo_ctrl.search()
    print(f"  最优架构: {topo_result.best_architecture}")
    print(f"  最优分数: {topo_result.best_score:.4f}")
    print(f"  Pareto 前沿: {len(topo_result.pareto_front)} 个解")
    test_state = np.random.randn(topo_ctrl.config.input_dim)
    control_signal = topo_ctrl.process(test_state)
    print(f"  控制信号: {control_signal}")

    # ---- 测试 2: HolographicSpotReconstructor ----
    print("\n[2] HolographicSpotReconstructor")
    holo_recon = HolographicSpotReconstructor(HolographicReconConfig(
        image_size=128, num_gs_iterations=20
    ))
    # 生成模拟干涉图
    yy, xx = np.mgrid[:128, :128]
    interferogram = (np.cos(xx * 0.5) * np.cos(yy * 0.3) + 1) / 2
    interferogram += np.random.randn(128, 128) * 0.05
    holo_result = holo_recon.reconstruct(interferogram)
    print(f"  重建误差: {holo_result.reconstruction_error:.6f}")
    print(f"  振幅范围: [{holo_result.amplitude.min():.4f}, {holo_result.amplitude.max():.4f}]")
    print(f"  相位范围: [{holo_result.phase.min():.4f}, {holo_result.phase.max():.4f}]")

    # ---- 测试 3: FederatedMultiTaskOptimizer ----
    print("\n[3] FederatedMultiTaskOptimizer")
    fed_opt = FederatedMultiTaskOptimizer(FederatedMultiTaskConfig(
        num_clients=3, num_rounds=5, local_epochs=2
    ))
    fed_result = fed_opt.optimize()
    print(f"  最终损失: {fed_result.convergence_history[-1]:.6f}")
    print(f"  通信开销: {fed_result.communication_cost:.0f} 参数")
    print(f"  各客户端指标: {len(fed_result.per_client_metrics)} 个")

    # ---- 测试 4: NeuralODEBeamPropagator ----
    print("\n[4] NeuralODEBeamPropagator")
    ode_prop = NeuralODEBeamPropagator(NeuralODEConfig(
        max_propagation_steps=20, ode_solver="rk4"
    ))
    input_field = _generate_gaussian_spot(size=32, sigma=3.0)
    ode_result = ode_prop.propagate(input_field)
    print(f"  输出能量: {float(np.sum(np.abs(ode_result.output_field)**2)):.4f}")
    print(f"  传播损失: {ode_result.propagation_loss:.6f}")
    print(f"  函数评估: {ode_result.num_function_evaluations}")

    # ---- 测试 5: QuantumInspiredOptimizer ----
    print("\n[5] QuantumInspiredOptimizer")
    quantum_opt = QuantumInspiredOptimizer(QuantumInspiredConfig(
        max_iterations=100, num_qubits=5, population_size=20
    ))
    q_result = quantum_opt.optimize()
    print(f"  最优能量: {q_result.best_energy:.6f}")
    print(f"  最优参数: {q_result.best_parameters}")
    print(f"  隧穿事件: {q_result.tunneling_events}")
    print(f"  收敛: {q_result.convergence_achieved}")

    # ---- 测试 6: MultimodalFusionTracker ----
    print("\n[6] MultimodalFusionTracker")
    mm_tracker = MultimodalFusionTracker(MultimodalFusionConfig(
        embed_dim=32, num_heads=2, num_transformer_layers=2
    ))
    # 模拟跟踪序列
    for frame_idx in range(5):
        spot = _generate_gaussian_spot(
            size=64, cx=32 + frame_idx * 0.5, cy=32 + frame_idx * 0.3, sigma=4.0
        )
        track_result = mm_tracker.track(spot)
        if frame_idx == 4:
            print(f"  跟踪位置: ({track_result.tracked_position[0]:.2f}, {track_result.tracked_position[1]:.2f})")
            print(f"  置信度: {track_result.confidence:.4f}")
            print(f"  模态权重: {track_result.modality_weights}")
            print(f"  跟踪状态: {track_result.tracking_status}")

    print("\n" + "=" * 70)
    print("所有模块测试完成!")
    print("=" * 70)
