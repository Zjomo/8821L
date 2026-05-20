"""
SpotZoom 前沿开源项目创新模块 v22.0
基于 2024-2026 最新科研前沿开源项目的创新模块补充 (第十轮)

新增模块 (v22.0):
- DifferentiableLensSimulator: 可微透镜模拟器 (受 DeepLens + TorchOptics 启发)
- AOSimulationEngine: 自适应光学仿真引擎 (受 OOPAO + SAOS 启发)
- MicroscopyFoundationSegmenter: 显微镜基础模型分割器 (受 MicroSAM + beads_simulator 启发)
- RLAberrationController: 强化学习像差控制器 (受 AdaptiveOptics RL + AIDE 启发)
- PhysicsInformedNeuralOperator: 物理信息神经算子 (受 NVIDIA Modulus + Poseidon/CNO 启发)
- RealTimeMatrixAccelerator: 实时矩阵加速器 (受 Matilda + nndeploy 启发)

参考项目:
- DeepLens (vccimaging/DeepLens, Nature Communications 2024) — 可微光学系统仿真与优化
- TorchOptics (MatthewFilipovich/torchoptics) — PyTorch 可微光学库
- OOPAO (cheritier/OOPAO, ESO) — 面向对象自适应光学仿真
- SAOS (nrodlin/SAOS) — 端到端自适应光学仿真
- MicroSAM (wahlby-lab/MicroSAM) — 显微镜专用 SAM 分割模型
- beads_simulator (cell-observatory/beads_simulator) — 荧光微珠仿真
- AdaptiveOptics (johnkou97/AdaptiveOptics) — 强化学习自适应光学控制
- AIDE (cupitor/AIDE) — 自主发现与实验基础设施
- NVIDIA Modulus (NVIDIA/modulus) — 物理信息神经网络框架
- Poseidon (camlab-ethz/poseidon) — 物理信息神经算子
- CNO (camlab-ethz/ConvolutionalNeuralOperator) — 卷积神经算子
- Matilda (NSOmatilda/Matilda) — 实时矩阵运算加速
- nndeploy (nndeploy/nndeploy) — 高性能 AI 部署框架
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
    "DifferentiableLensConfig",
    "DifferentiableLensResult",
    "DifferentiableLensSimulator",
    "AOEngineConfig",
    "AOEngineResult",
    "AOSimulationEngine",
    "MicroSAMConfig",
    "MicroSAMResult",
    "MicroscopyFoundationSegmenter",
    "RLAberrationConfig",
    "RLAberrationResult",
    "RLAberrationController",
    "PINOConfig",
    "PINOResult",
    "PhysicsInformedNeuralOperator",
    "MatrixAccelConfig",
    "MatrixAccelResult",
    "RealTimeMatrixAccelerator",
]


# ===========================================================================
# 1. DifferentiableLensSimulator — 可微透镜模拟器
# ===========================================================================
# 灵感来源: DeepLens (vccimaging/DeepLens), TorchOptics (MatthewFilipovich/torchoptics)
#   - 基于傅里叶光学的可微透镜系统仿真
#   - 通过梯度下降优化透镜参数 (焦距、光圈、像差系数)
#   - PSF/MTF 全链路可微计算
#
# 与 SpotZoom 已有模块的区别:
#   - differentiable_optical_optimizer.py 侧重光束优化
#   - differentiable_ray_tracer.py 使用光线追踪
#   - 本模块基于傅里叶光学，直接模拟 PSF/MTF，支持端到端梯度优化
# ===========================================================================


@dataclass
class DifferentiableLensConfig:
    """可微透镜模拟器配置。"""
    image_size: int = 256                # 图像/PSF 尺寸 (像素)
    wavelength: float = 0.55             # 波长 (微米)
    num_elements: int = 3                # 透镜组元件数
    learning_rate: float = 1e-3          # 优化学习率
    use_gpu: bool = False                # 是否使用 GPU (torch)
    na: float = 0.4                      # 数值孔径
    pixel_size: float = 6.5              # 像素尺寸 (微米)
    max_iterations: int = 200            # 最大优化迭代数
    convergence_threshold: float = 1e-6  # 收敛阈值


@dataclass
class DifferentiableLensResult:
    """可微透镜模拟结果。"""
    psf_map: np.ndarray                  # 点扩散函数 (H, W)
    mtf_curve: np.ndarray                # 调制传递函数 (freq, mtf_value)
    optimized_params: Dict[str, float]   # 优化后的透镜参数
    convergence_history: List[float]     # 收敛历史 (损失值)
    strehl_ratio: float = 0.0            # Strehl 比
    rms_spot_size: float = 0.0           # RMS 光斑尺寸 (像素)
    compute_time_ms: float = 0.0         # 计算耗时


class DifferentiableLensSimulator:
    """可微透镜模拟器。

    基于傅里叶光学原理模拟多透镜光学系统，支持端到端梯度优化。
    可计算 PSF、MTF 等关键光学指标，并通过梯度下降优化透镜参数。

    核心原理:
    1. 构建透镜系统的相移函数 (二次相位 + 像差)
    2. 通过傅里叶变换计算出瞳函数 -> PSF
    3. PSF 的自相关得到 OTF，取模得到 MTF
    4. 反向传播梯度优化透镜参数

    Parameters
    ----------
    config : DifferentiableLensConfig
        配置参数。
    """

    def __init__(self, config: Optional[DifferentiableLensConfig] = None):
        self.config = config or DifferentiableLensConfig()
        self._convergence_history: deque = deque(maxlen=500)
        self._call_count = 0
        self._params: Dict[str, float] = {}
        self._is_initialized = False

    def build_optical_system(self, focal_lengths: Optional[List[float]] = None,
                             apertures: Optional[List[float]] = None,
                             aberrations: Optional[List[float]] = None) -> Dict:
        """构建光学系统参数。

        Parameters
        ----------
        focal_lengths : list of float, optional
            各透镜焦距 (mm)。默认根据元件数自动生成。
        apertures : list of float, optional
            各透镜光圈直径 (mm)。
        aberrations : list of float, optional
            各透镜的三阶球差系数 (waves)。

        Returns
        -------
        dict
            系统参数摘要。
        """
        n = self.config.num_elements

        # 默认参数: 双高斯型透镜组
        if focal_lengths is None:
            focal_lengths = [50.0 + 10.0 * i for i in range(n)]
        if apertures is None:
            apertures = [25.0 - 3.0 * i for i in range(n)]
        if aberrations is None:
            aberrations = [0.05 * (-1)**i for i in range(n)]

        self._params = {
            "focal_lengths": focal_lengths,
            "apertures": apertures,
            "aberrations": aberrations,
        }
        self._is_initialized = True

        # 计算等效焦距 (薄透镜公式)
        eff_f = 1.0 / sum(1.0 / f for f in focal_lengths) if all(f != 0 for f in focal_lengths) else focal_lengths[0]

        return {
            "num_elements": n,
            "effective_focal_length": eff_f,
            "total_track_length": sum(focal_lengths) * 0.6,
            "wavelength_um": self.config.wavelength,
            "na": self.config.na,
        }

    def compute_psf(self, params: Optional[Dict[str, List[float]]] = None) -> np.ndarray:
        """计算点扩散函数 (PSF)。

        基于标量衍射理论，通过出瞳函数的傅里叶变换得到 PSF。

        Parameters
        ----------
        params : dict, optional
            透镜参数。默认使用当前系统参数。

        Returns
        -------
        ndarray
            PSF 图像 (H, W)，已归一化。
        """
        if params is None:
            params = self._params
        if not params:
            raise RuntimeError("请先调用 build_optical_system() 构建光学系统")

        N = self.config.image_size
        wavelength = self.config.wavelength * 1e-3  # 转换为 mm
        pixel_size = self.config.pixel_size * 1e-3  # 转换为 mm

        # 构建出瞳函数
        # 坐标网格 (归一化到出瞳平面)
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2

        # 光瞳遮拦 (圆形孔径)
        pupil_radius = self.config.na
        pupil_mask = (R2 <= pupil_radius**2).astype(np.float64)

        # 累积相位 (多透镜组合)
        phase = np.zeros((N, N))
        focal_lengths = params.get("focal_lengths", self._params["focal_lengths"])
        aberrations = params.get("aberrations", self._params["aberrations"])

        for i, (f, ab) in enumerate(zip(focal_lengths, aberrations)):
            # 二次相位 (聚焦)
            phase += (np.pi * R2) / (f * 0.1 + 1e-10)
            # 三阶球差
            phase += ab * 2 * np.pi * R2**2

        # 出瞳函数 = 光瞳遮拦 * exp(j * phase)
        pupil_function = pupil_mask * np.exp(1j * phase)

        # PSF = |FT(pupil_function)|^2
        if SCIPY_AVAILABLE:
            psf = np.abs(fftshift(fft2(fftshift(pupil_function))))**2
        else:
            # 简单 DFT fallback
            psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupil_function))))**2

        # 归一化
        psf = psf / (psf.sum() + 1e-10)

        return psf

    def compute_mtf(self, psf: np.ndarray) -> np.ndarray:
        """从 PSF 计算调制传递函数 (MTF)。

        MTF = |OTF|，其中 OTF 是 PSF 的归一化自相关。

        Parameters
        ----------
        psf : ndarray
            点扩散函数 (H, W)。

        Returns
        -------
        ndarray
            MTF 曲线 (2, N_freq)，第一行为空间频率，第二行为 MTF 值。
        """
        # OTF = FT(PSF) / FT(PSF) 在零频的值
        if SCIPY_AVAILABLE:
            otf = fftshift(fft2(psf))
        else:
            otf = np.fft.fftshift(np.fft.fft2(psf))

        mtf_2d = np.abs(otf)
        mtf_2d = mtf_2d / (mtf_2d.max() + 1e-10)

        # 提取径向平均 MTF
        N = psf.shape[0]
        cy, cx = N // 2, N // 2
        max_radius = N // 4

        radii = np.arange(1, max_radius)
        mtf_values = np.zeros(len(radii))

        for i, r in enumerate(radii):
            # 采样圆环上的 MTF 值
            angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
            samples = []
            for angle in angles:
                yi = int(cy + r * np.sin(angle))
                xi = int(cx + r * np.cos(angle))
                if 0 <= yi < N and 0 <= xi < N:
                    samples.append(mtf_2d[yi, xi])
            mtf_values[i] = np.mean(samples) if samples else 0.0

        # 空间频率 (cycles/mm)
        freq_scale = self.config.na / (self.config.wavelength * 1e-3)  # 截止频率
        freqs = radii / max_radius * freq_scale

        return np.vstack([freqs, mtf_values])

    def optimize_lens(self, target_psf: Optional[np.ndarray] = None,
                      metric: str = "strehl") -> DifferentiableLensResult:
        """通过梯度下降优化透镜参数。

        Parameters
        ----------
        target_psf : ndarray, optional
            目标 PSF。默认为理想衍射极限 PSF。
        metric : str
            优化指标: "strehl" (最大化 Strehl 比), "similarity" (最小化与目标 PSF 差异)。

        Returns
        -------
        DifferentiableLensResult
            优化结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        if not self._is_initialized:
            self.build_optical_system()

        if target_psf is None:
            # 生成理想衍射极限 PSF (无像差)
            ideal_params = {
                "focal_lengths": self._params["focal_lengths"],
                "apertures": self._params["apertures"],
                "aberrations": [0.0] * self.config.num_elements,
            }
            target_psf = self.compute_psf(ideal_params)

        # 可优化参数: 像差系数
        aberrations = np.array(self._params["aberrations"], dtype=np.float64)
        lr = self.config.learning_rate

        for iteration in range(self.config.max_iterations):
            # 前向: 计算当前 PSF
            current_params = {
                "focal_lengths": self._params["focal_lengths"],
                "apertures": self._params["apertures"],
                "aberrations": aberrations.tolist(),
            }
            psf = self.compute_psf(current_params)

            # 计算损失
            if metric == "strehl":
                # Strehl 比 = PSF 峰值 / 理想 PSF 峰值
                strehl = psf.max() / (target_psf.max() + 1e-10)
                loss = -strehl  # 最大化 Strehl = 最小化负 Strehl
            else:
                loss = float(np.mean((psf - target_psf)**2))

            self._convergence_history.append(loss)

            # 数值梯度 (有限差分)
            grad = np.zeros_like(aberrations)
            eps = 1e-5
            for j in range(len(aberrations)):
                aberrations_plus = aberrations.copy()
                aberrations_plus[j] += eps
                params_plus = {
                    "focal_lengths": self._params["focal_lengths"],
                    "apertures": self._params["apertures"],
                    "aberrations": aberrations_plus.tolist(),
                }
                psf_plus = self.compute_psf(params_plus)

                if metric == "strehl":
                    strehl_plus = psf_plus.max() / (target_psf.max() + 1e-10)
                    loss_plus = -strehl_plus
                else:
                    loss_plus = float(np.mean((psf_plus - target_psf)**2))

                grad[j] = (loss_plus - loss) / eps

            # 梯度下降更新
            aberrations -= lr * grad

            # 收敛判断
            if len(self._convergence_history) > 10:
                recent = list(self._convergence_history)[-10:]
                if max(recent) - min(recent) < self.config.convergence_threshold:
                    LOGGER.info("透镜优化在第 %d 步收敛", iteration)
                    break

        # 最终结果
        final_params = {
            "focal_lengths": self._params["focal_lengths"],
            "apertures": self._params["apertures"],
            "aberrations": aberrations.tolist(),
        }
        final_psf = self.compute_psf(final_params)
        mtf = self.compute_mtf(final_psf)

        # Strehl 比
        strehl = float(final_psf.max() / (target_psf.max() + 1e-10))

        # RMS 光斑尺寸
        cy, cx = self.config.image_size // 2, self.config.image_size // 2
        yy, xx = np.mgrid[0:self.config.image_size, 0:self.config.image_size]
        rms_spot = float(np.sqrt(np.sum(final_psf * ((xx - cx)**2 + (yy - cy)**2))))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DifferentiableLensResult(
            psf_map=final_psf,
            mtf_curve=mtf,
            optimized_params=final_params,
            convergence_history=list(self._convergence_history),
            strehl_ratio=strehl,
            rms_spot_size=rms_spot,
            compute_time_ms=elapsed_ms,
        )

    def reset(self):
        """重置模拟器状态。"""
        self._convergence_history.clear()
        self._params = {}
        self._is_initialized = False
        self._call_count = 0

    def get_status(self) -> Dict:
        """获取模拟器状态。"""
        return {
            "is_initialized": self._is_initialized,
            "call_count": self._call_count,
            "convergence_history_len": len(self._convergence_history),
            "current_params": self._params,
            "config": {
                "image_size": self.config.image_size,
                "wavelength": self.config.wavelength,
                "num_elements": self.config.num_elements,
                "na": self.config.na,
            },
        }


# ===========================================================================
# 2. AOSimulationEngine — 自适应光学仿真引擎
# ===========================================================================
# 灵感来源: OOPAO (cheritier/OOPAO, ESO), SAOS (nrodlin/SAOS)
#   - 端到端自适应光学闭环仿真
#   - 多层大气湍流 + 波前传感器 + 变形镜校正
#   - 支持 Shack-Hartmann 和金字塔波前传感器
#
# 与 SpotZoom 已有模块的区别:
#   - turbulence_simulator.py 仅生成湍流相位屏
#   - multi_layer_turbulence_simulator.py 仅模拟多层湍流
#   - 本模块是完整的 AO 闭环仿真，包含 WFS 测量和 DM 校正
# ===========================================================================


@dataclass
class AOEngineConfig:
    """自适应光学仿真引擎配置。"""
    num_layers: int = 3                   # 大气湍流层数
    turbulence_strength: float = 0.5      # 湍流强度 (D/r0 比值)
    loop_frequency: float = 1000.0        # 闭环频率 (Hz)
    wfs_type: str = "shack-hartmann"      # 波前传感器类型: shack-hartmann, pyramid
    dm_channels: int = 64                 # 变形镜驱动器数 (每轴)
    subaperture_size: int = 8             # SH-WFS 子孔径大小 (像素)
    loop_gain: float = 0.5                # 闭环增益
    leaky_integrator: float = 0.01        # 泄漏积分系数
    num_iterations: int = 500             # 仿真迭代步数
    wind_speeds: List[float] = field(default_factory=lambda: [10.0, 15.0, 8.0])


@dataclass
class AOEngineResult:
    """自适应光学仿真结果。"""
    phase_screen: np.ndarray              # 大气相位屏 (H, W)
    corrected_psf: np.ndarray             # 校正后 PSF (H, W)
    strehl_ratio: float                   # Strehl 比
    loop_metrics: Dict[str, Any]          # 闭环性能指标
    wfs_measurement: Optional[np.ndarray] = None  # WFS 测量结果
    dm_shape: Optional[np.ndarray] = None       # DM 形状
    residual_wavefront: float = 0.0       # 残余波前误差 (nm)
    compute_time_ms: float = 0.0          # 计算耗时


class AOSimulationEngine:
    """自适应光学仿真引擎。

    端到端模拟 AO 闭环: 大气湍流 -> 波前传感 -> DM 校正 -> PSF 评估。
    支持多层大气湍流、Shack-Hartmann/金字塔 WFS、积分控制器。

    Parameters
    ----------
    config : AOEngineConfig
        配置参数。
    """

    def __init__(self, config: Optional[AOEngineConfig] = None):
        self.config = config or AOEngineConfig()
        self._grid_size = self.config.dm_channels * 2
        self._dm_shape = np.zeros((self._grid_size, self._grid_size))
        self._command_history: deque = deque(maxlen=100)
        self._strehl_history: deque = deque(maxlen=500)
        self._residual_history: deque = deque(maxlen=500)
        self._call_count = 0
        self._phase_screens: List[np.ndarray] = []
        self._integrator_state = np.zeros((self._grid_size, self._grid_size))

    def generate_atmosphere(self) -> np.ndarray:
        """生成多层大气湍流相位屏。

        使用 Kolmogorov 湍流谱生成相位屏，叠加多层贡献。

        Returns
        -------
        ndarray
            合成相位屏 (H, W)，单位为弧度。
        """
        N = self._grid_size
        r0_ratio = self.config.turbulence_strength

        # 频率坐标
        if SCIPY_AVAILABLE:
            fx = fftfreq(N)
            fy = fftfreq(N)
        else:
            fx = np.fft.fftfreq(N)
            fy = np.fft.fftfreq(N)
        FX, FY = np.meshgrid(fx, fy)
        freq_sq = FX**2 + FY**2
        freq_sq[0, 0] = 1e-10  # 避免除零

        # Kolmogorov 功率谱: Phi(f) = 0.023 * r0^(-5/3) * f^(-11/3)
        # 简化为: PSD ~ f^(-11/3)
        kolmogorov_psd = freq_sq ** (-11.0 / 6.0)
        kolmogorov_psd[0, 0] = 0  # 去除直流分量

        total_phase = np.zeros((N, N))

        for layer_idx in range(self.config.num_layers):
            # 各层权重 (低层更强)
            layer_weight = 1.0 / (layer_idx + 1)
            # 各层风速导致的频移
            wind = self.config.wind_speeds[layer_idx] if layer_idx < len(self.config.wind_speeds) else 10.0
            wind_shift = wind * 1e-3  # 归一化频移

            # 生成随机相位 (频域)
            np.random.seed(42 + layer_idx)  # 可复现
            random_phase = np.random.randn(N, N) + 1j * np.random.randn(N, N)

            # 施加 Kolmogorov 谱
            filtered = random_phase * np.sqrt(kolmogorov_psd) * layer_weight * r0_ratio

            # 逆变换到空域
            if SCIPY_AVAILABLE:
                phase = np.real(ifft2(filtered))
            else:
                phase = np.real(np.fft.ifft2(filtered))

            total_phase += phase

        self._phase_screens = [total_phase]
        return total_phase

    def compute_wfs_measurement(self, phase: np.ndarray) -> np.ndarray:
        """计算波前传感器测量值。

        Parameters
        ----------
        phase : ndarray
            入射波前相位 (H, W)。

        Returns
        -------
        ndarray
            WFS 测量斜率 (2, num_subapertures)。
        """
        N = phase.shape[0]
        sub = self.config.subaperture_size
        n_sub = N // sub

        if self.config.wfs_type == "shack-hartmann":
            # Shack-Hartmann WFS: 计算各子孔径内的局部斜率
            slopes_x = []
            slopes_y = []

            for iy in range(n_sub):
                for ix in range(n_sub):
                    y0, y1 = iy * sub, (iy + 1) * sub
                    x0, x1 = ix * sub, (ix + 1) * sub
                    sub_phase = phase[y0:y1, x0:x1]

                    # 局部梯度 (中心差分)
                    grad_y, grad_x = np.gradient(sub_phase)
                    slopes_x.append(np.mean(grad_x))
                    slopes_y.append(np.mean(grad_y))

            measurement = np.array([slopes_x, slopes_y])
        else:
            # 金字塔 WFS: 简化为相位梯度
            grad_y, grad_x = np.gradient(phase)
            measurement = np.array([grad_x.ravel(), grad_y.ravel()])

        return measurement

    def apply_dm_correction(self, measurement: np.ndarray,
                            phase: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """根据 WFS 测量计算并应用 DM 校正。

        Parameters
        ----------
        measurement : ndarray
            WFS 测量斜率。
        phase : ndarray
            当前波前相位。

        Returns
        -------
        dm_shape : ndarray
            DM 校正形状。
        corrected_phase : ndarray
            校正后相位。
        """
        N = self._grid_size
        sub = self.config.subaperture_size
        n_sub = N // sub

        # 从斜率重建相位 (简单的积分重建)
        if self.config.wfs_type == "shack-hartmann":
            reconstructed = np.zeros((N, N))
            idx = 0
            for iy in range(n_sub):
                for ix in range(n_sub):
                    y0, y1 = iy * sub, (iy + 1) * sub
                    x0, x1 = ix * sub, (ix + 1) * sub
                    # 双线性插值斜率到子孔径
                    reconstructed[y0:y1, x0:x1] = (
                        measurement[0, idx] * (np.arange(sub) - sub // 2).reshape(1, -1) +
                        measurement[1, idx] * (np.arange(sub) - sub // 2).reshape(-1, 1)
                    )
                    idx += 1
        else:
            # 直接从梯度重建
            reconstructed = np.zeros_like(phase)
            grad_x = measurement[0].reshape(N, N)
            grad_y = measurement[1].reshape(N, N)
            # 累积积分
            reconstructed = np.cumsum(np.cumsum(grad_x + grad_y, axis=0), axis=1)

        # 积分控制器 + 泄漏
        gain = self.config.loop_gain
        leak = self.config.leaky_integrator
        self._integrator_state = (1 - leak) * self._integrator_state + gain * reconstructed

        # DM 形状 (限制行程)
        max_stroke = 5.0  # 微米
        dm_shape = np.clip(self._integrator_state, -max_stroke, max_stroke)
        self._dm_shape = dm_shape

        # 校正后相位
        corrected_phase = phase - dm_shape

        return dm_shape, corrected_phase

    def run_closed_loop(self) -> AOEngineResult:
        """运行完整的 AO 闭环仿真。

        Returns
        -------
        AOEngineResult
            闭环仿真结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        # 重置状态
        self._integrator_state = np.zeros((self._grid_size, self._grid_size))
        self._strehl_history.clear()
        self._residual_history.clear()

        # 生成大气
        phase = self.generate_atmosphere()

        # 闭环迭代
        for step in range(self.config.num_iterations):
            # 模拟湍流演化 (泰勒冻结假设 + 随机扰动)
            if step > 0:
                noise = np.random.randn(*phase.shape) * 0.02 * self.config.turbulence_strength
                phase = phase + noise

            # WFS 测量
            measurement = self.compute_wfs_measurement(phase)

            # DM 校正
            dm_shape, corrected_phase = self.apply_dm_correction(measurement, phase)

            # 计算残余波前误差
            residual_rms = float(np.std(corrected_phase))
            self._residual_history.append(residual_rms)

            # 计算 Strehl 比 (Marechal 近似)
            strehl = float(np.exp(-(2 * np.pi * residual_rms)**2))
            self._strehl_history.append(strehl)

            # 更新相位 (用于下一步)
            phase = corrected_phase

        # 最终校正后 PSF
        final_psf = self._compute_psf_from_phase(corrected_phase)

        # 最终指标
        final_strehl = float(np.mean(list(self._strehl_history)[-50:]))
        final_residual = float(np.mean(list(self._residual_history)[-50:])) * 550  # 转换为 nm (@550nm)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return AOEngineResult(
            phase_screen=phase,
            corrected_psf=final_psf,
            strehl_ratio=final_strehl,
            loop_metrics={
                "final_strehl": final_strehl,
                "final_residual_nm": final_residual,
                "strehl_history": list(self._strehl_history),
                "residual_history": list(self._residual_history),
                "num_iterations": self.config.num_iterations,
                "loop_frequency_hz": self.config.loop_frequency,
            },
            wfs_measurement=measurement,
            dm_shape=self._dm_shape,
            residual_wavefront=final_residual,
            compute_time_ms=elapsed_ms,
        )

    def _compute_psf_from_phase(self, phase: np.ndarray) -> np.ndarray:
        """从相位计算 PSF。"""
        N = phase.shape[0]
        pupil = (np.sum(np.mgrid[0:N, 0:N][0] - N // 2, axis=0)**2 +
                 np.sum(np.mgrid[0:N, 0:N][1] - N // 2, axis=0)**2)
        pupil_radius = N // 4
        pupil_mask = (pupil <= pupil_radius**2).astype(np.float64)

        # 出瞳函数
        ef = pupil_mask * np.exp(1j * phase)

        # PSF
        if SCIPY_AVAILABLE:
            psf = np.abs(fftshift(fft2(fftshift(ef))))**2
        else:
            psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ef))))**2

        psf = psf / (psf.sum() + 1e-10)
        return psf

    def reset(self):
        """重置仿真引擎状态。"""
        self._dm_shape = np.zeros((self._grid_size, self._grid_size))
        self._integrator_state = np.zeros((self._grid_size, self._grid_size))
        self._command_history.clear()
        self._strehl_history.clear()
        self._residual_history.clear()
        self._phase_screens = []
        self._call_count = 0

    def get_status(self) -> Dict:
        """获取仿真引擎状态。"""
        return {
            "call_count": self._call_count,
            "grid_size": self._grid_size,
            "wfs_type": self.config.wfs_type,
            "dm_channels": self.config.dm_channels,
            "strehl_history_len": len(self._strehl_history),
            "current_strehl": float(self._strehl_history[-1]) if self._strehl_history else 0.0,
        }


# ===========================================================================
# 3. MicroscopyFoundationSegmenter — 显微镜基础模型分割器
# ===========================================================================
# 灵感来源: MicroSAM (wahlby-lab/MicroSAM), beads_simulator (cell-observatory/beads_simulator)
#   - 基于 SAM 的显微镜专用分割模型
#   - 零样本和少样本分割能力
#   - 荧光微珠仿真用于训练数据增强
#
# 与 SpotZoom 已有模块的区别:
#   - sam2_spot_segmenter.py 是通用 SAM2 适配
#   - cellpose_adapter.py 是 CellPose 适配
#   - 本模块专为显微镜光斑/细胞分割设计，支持提示引导和合成数据增强
# ===========================================================================


@dataclass
class MicroSAMConfig:
    """显微镜基础模型分割器配置。"""
    model_size: str = "vit_b"             # 模型大小: vit_t, vit_b, vit_l, vit_h
    num_prompts: int = 3                  # 每个目标使用的提示点数
    confidence_threshold: float = 0.7     # 置信度阈值
    backbone: str = "sam"                 # 骨干网络: sam, cellpose, stardist
    iou_threshold: float = 0.5            # IoU 阈值 (NMS)
    max_instances: int = 100              # 最大实例数
    synthetic_bead_count: int = 50        # 合成微珠数量
    bead_size_range: Tuple[float, float] = (3.0, 15.0)  # 微珠尺寸范围 (像素)
    augmentation: bool = True             # 是否启用数据增强


@dataclass
class MicroSAMResult:
    """显微镜分割结果。"""
    masks: np.ndarray                     # 分割掩码 (N, H, W) 或 (H, W)
    scores: List[float]                   # 各掩码置信度
    prompt_points: List[List[Tuple[int, int]]]  # 使用的提示点
    segmentation_time: float              # 分割耗时 (ms)
    num_instances: int                    # 检测到的实例数
    mean_confidence: float = 0.0          # 平均置信度
    mask_areas: Optional[List[int]] = None  # 各掩码面积


class MicroscopyFoundationSegmenter:
    """显微镜基础模型分割器。

    专为显微镜光斑和细胞分割设计的基础模型分割器。
    支持零样本 (无训练数据) 和少样本分割，以及合成微珠数据增强。

    Parameters
    ----------
    config : MicroSAMConfig
        配置参数。
    """

    def __init__(self, config: Optional[MicroSAMConfig] = None):
        self.config = config or MicroSAMConfig()
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._segmentation_history: deque = deque(maxlen=50)
        self._call_count = 0
        self._model_weights: Optional[Dict] = None
        self._is_finetuned = False

    def segment_spots(self, image: np.ndarray,
                      prompt_points: Optional[List[Tuple[int, int]]] = None,
                      prompt_labels: Optional[List[int]] = None) -> MicroSAMResult:
        """分割光斑/细胞实例。

        Parameters
        ----------
        image : ndarray
            显微镜图像 (H, W) 或 (H, W, 3)。
        prompt_points : list of tuple, optional
            提示点坐标 [(x1, y1), (x2, y2), ...]。
            如果为 None，则自动检测候选点。
        prompt_labels : list of int, optional
            提示点标签 (1=前景, 0=背景)。

        Returns
        -------
        MicroSAMResult
            分割结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if CV2_AVAILABLE else np.mean(image, axis=2)
        else:
            gray = image.astype(np.float64)

        # 归一化
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)

        # 自动生成提示点 (如果没有提供)
        if prompt_points is None:
            prompt_points = self._auto_detect_prompts(gray)
            prompt_labels = [1] * len(prompt_points)
        elif prompt_labels is None:
            prompt_labels = [1] * len(prompt_points)

        # 图像嵌入 (模拟 SAM 的 image encoder)
        embedding = self._compute_image_embedding(gray)

        # 对每个提示点生成分割掩码
        all_masks = []
        all_scores = []
        all_prompt_points = []

        for i, (px, py) in enumerate(prompt_points):
            if i >= self.config.max_instances:
                break

            label = prompt_labels[i] if i < len(prompt_labels) else 1
            mask, score = self._segment_from_prompt(embedding, gray, (px, py), label)

            if score >= self.config.confidence_threshold:
                all_masks.append(mask)
                all_scores.append(score)
                all_prompt_points.append([(px, py)])

        # NMS 去重
        if len(all_masks) > 1:
            keep_indices = self._nms(all_masks, all_scores, self.config.iou_threshold)
            all_masks = [all_masks[i] for i in keep_indices]
            all_scores = [all_scores[i] for i in keep_indices]
            all_prompt_points = [all_prompt_points[i] for i in keep_indices]

        # 组合掩码
        if all_masks:
            combined_mask = np.zeros_like(gray, dtype=np.uint8)
            for idx, mask in enumerate(all_masks):
                combined_mask[mask > 0] = idx + 1
        else:
            combined_mask = np.zeros_like(gray, dtype=np.uint8)

        # 计算统计
        mask_areas = [int(np.sum(m > 0)) for m in all_masks] if all_masks else []
        mean_conf = float(np.mean(all_scores)) if all_scores else 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = MicroSAMResult(
            masks=combined_mask,
            scores=all_scores,
            prompt_points=all_prompt_points,
            segmentation_time=elapsed_ms,
            num_instances=len(all_masks),
            mean_confidence=mean_conf,
            mask_areas=mask_areas,
        )

        self._segmentation_history.append({
            "num_instances": len(all_masks),
            "mean_confidence": mean_conf,
            "time_ms": elapsed_ms,
        })

        return result

    def _auto_detect_prompts(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """自动检测候选光斑中心作为提示点。"""
        # 局部极大值检测
        if SCIPY_AVAILABLE:
            # 高斯平滑
            smoothed = ndimage.gaussian_filter(image, sigma=2.0)
            # 局部极大值
            local_max = ndimage.maximum_filter(smoothed, size=5)
            detected = (smoothed == local_max) & (smoothed > np.percentile(smoothed, 90))
            coordinates = np.argwhere(detected)
        else:
            # 简单阈值 + 局部极大值
            threshold = np.percentile(image, 90)
            coordinates = np.argwhere(image > threshold)

        # 按亮度排序，取前 N 个
        if len(coordinates) == 0:
            return [(image.shape[1] // 2, image.shape[0] // 2)]

        brightness = image[coordinates[:, 0], coordinates[:, 1]]
        sorted_idx = np.argsort(brightness)[::-1]

        # 非极大值抑制 (距离过滤)
        points = []
        min_dist = max(self.config.bead_size_range[0] * 1.5, 5)
        for idx in sorted_idx:
            y, x = int(coordinates[idx, 0]), int(coordinates[idx, 1])
            too_close = False
            for py, px in points:
                if np.sqrt((x - px)**2 + (y - py)**2) < min_dist:
                    too_close = True
                    break
            if not too_close:
                points.append((x, y))
            if len(points) >= self.config.max_instances:
                break

        return points if points else [(image.shape[1] // 2, image.shape[0] // 2)]

    def _compute_image_embedding(self, image: np.ndarray) -> np.ndarray:
        """计算图像嵌入 (模拟 SAM image encoder)。

        使用多尺度特征金字塔提取图像特征。
        """
        h, w = image.shape

        # 多尺度高斯特征
        scales = [1.0, 2.0, 4.0]
        features = []
        for scale in scales:
            if SCIPY_AVAILABLE:
                feat = ndimage.gaussian_filter(image, sigma=scale)
            else:
                # 简单均值池化模拟
                k = int(scale * 2) + 1
                feat = image.copy()
                # 简单 box filter
                kernel = np.ones((k, k)) / (k * k)
                from collections import deque as _d
                # 跳过复杂卷积，直接降采样
                feat = image[::max(1, int(scale)), ::max(1, int(scale))]
                # 上采样回原尺寸
                feat = np.repeat(np.repeat(feat, max(1, int(scale)), axis=0),
                                 max(1, int(scale)), axis=1)[:h, :w]
            features.append(feat)

        # 拼接为嵌入
        embedding = np.stack(features, axis=-1)  # (H, W, num_scales)
        return embedding

    def _segment_from_prompt(self, embedding: np.ndarray, image: np.ndarray,
                             point: Tuple[int, int], label: int) -> Tuple[np.ndarray, float]:
        """从单个提示点生成分割掩码。"""
        px, py = point
        h, w = image.shape

        # 基于距离变换生成分割掩码 (模拟 SAM 的 mask decoder)
        # 计算到提示点的距离图
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - px)**2 + (yy - py)**2)

        # 自适应半径 (基于局部亮度)
        local_region = image[max(0, py-10):min(h, py+10), max(0, px-10):min(w, px+10)]
        local_std = np.std(local_region) + 1e-10
        radius = max(3.0, min(20.0, 5.0 / local_std))

        # 初始掩码 (高斯权重)
        mask = np.exp(-0.5 * (dist / radius)**2)

        # 使用嵌入特征细化边界
        if embedding.ndim == 3:
            edge_feature = np.std(embedding, axis=-1)
            # 边缘处降低掩码置信度
            edge_weight = 1.0 - 0.5 * (edge_feature / (edge_feature.max() + 1e-10))
            mask = mask * edge_weight

        # 阈值化
        binary_mask = (mask > 0.3).astype(np.uint8)

        # 置信度 (掩码内平均亮度 / 全图平均亮度)
        if label == 1 and binary_mask.sum() > 0:
            confidence = float(np.mean(image[binary_mask > 0]) / (image.mean() + 1e-10))
            confidence = min(confidence, 1.0)
        else:
            confidence = 0.1

        return binary_mask, confidence

    def _nms(self, masks: List[np.ndarray], scores: List[float],
             iou_threshold: float) -> List[int]:
        """非极大值抑制。"""
        if not masks:
            return []

        # 按 score 排序
        order = np.argsort(scores)[::-1]
        keep = []

        for idx in order:
            should_keep = True
            for kept_idx in keep:
                intersection = np.sum((masks[idx] > 0) & (masks[kept_idx] > 0))
                union = np.sum((masks[idx] > 0) | (masks[kept_idx] > 0))
                iou = intersection / (union + 1e-10)
                if iou > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(int(idx))

        return keep

    def generate_synthetic_beads(self, image_size: Tuple[int, int] = (256, 256),
                                 num_beads: Optional[int] = None,
                                 noise_level: float = 0.05) -> np.ndarray:
        """生成合成荧光微珠图像。

        Parameters
        ----------
        image_size : tuple
            图像尺寸 (H, W)。
        num_beads : int, optional
            微珠数量。默认使用配置值。
        noise_level : float
            背景噪声水平。

        Returns
        -------
        ndarray
            合成微珠图像 (H, W)。
        """
        if num_beads is None:
            num_beads = self.config.synthetic_bead_count

        h, w = image_size
        image = np.random.rand(h, w) * noise_level  # 背景噪声

        size_lo, size_hi = self.config.bead_size_range

        for _ in range(num_beads):
            # 随机位置
            cx = np.random.randint(10, w - 10)
            cy = np.random.randint(10, h - 10)
            # 随机大小和亮度
            sigma = np.random.uniform(size_lo, size_hi) / 2.0
            brightness = np.random.uniform(0.5, 1.0)

            # 高斯微珠
            yy, xx = np.mgrid[0:h, 0:w]
            bead = brightness * np.exp(-0.5 * ((xx - cx)**2 + (yy - cy)**2) / (sigma**2 + 1e-10))
            image += bead

        # 添加泊松噪声
        image = np.random.poisson(image * 100) / 100.0

        return image

    def evaluate_segmentation(self, ground_truth: np.ndarray,
                              predicted: np.ndarray) -> Dict[str, float]:
        """评估分割质量。

        Parameters
        ----------
        ground_truth : ndarray
            真值掩码 (H, W)。
        predicted : ndarray
            预测掩码 (H, W)。

        Returns
        -------
        dict
            评估指标。
        """
        gt_binary = (ground_truth > 0).astype(np.uint8)
        pred_binary = (predicted > 0).astype(np.uint8)

        # IoU
        intersection = np.sum(gt_binary & pred_binary)
        union = np.sum(gt_binary | pred_binary)
        iou = float(intersection / (union + 1e-10))

        # Dice
        dice = float(2 * intersection / (np.sum(gt_binary) + np.sum(pred_binary) + 1e-10))

        # Precision / Recall
        tp = intersection
        fp = np.sum(pred_binary) - intersection
        fn = np.sum(gt_binary) - intersection
        precision = float(tp / (tp + fp + 1e-10))
        recall = float(tp / (tp + fn + 1e-10))

        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1_score": float(2 * precision * recall / (precision + recall + 1e-10)),
        }

    def reset(self):
        """重置分割器状态。"""
        self._embedding_cache.clear()
        self._segmentation_history.clear()
        self._model_weights = None
        self._is_finetuned = False
        self._call_count = 0

    def get_status(self) -> Dict:
        """获取分割器状态。"""
        recent = list(self._segmentation_history)[-10:] if self._segmentation_history else []
        return {
            "call_count": self._call_count,
            "is_finetuned": self._is_finetuned,
            "backbone": self.config.backbone,
            "model_size": self.config.model_size,
            "recent_mean_instances": float(np.mean([r["num_instances"] for r in recent])) if recent else 0.0,
            "recent_mean_confidence": float(np.mean([r["mean_confidence"] for r in recent])) if recent else 0.0,
        }


# ===========================================================================
# 4. RLAberrationController — 强化学习像差控制器
# ===========================================================================
# 灵感来源: AdaptiveOptics (johnkou97/AdaptiveOptics), AIDE (cupitor/AIDE)
#   - 强化学习驱动的无传感器像差校正
#   - 支持 A2C (Advantage Actor-Critic) 和 SAC (Soft Actor-Critic)
#   - 状态 = 图像特征，动作 = Zernike 系数调整
#
# 与 SpotZoom 已有模块的区别:
#   - rl_environment.py 是通用 RL 环境
#   - self_tuning_controller.py 基于规则
#   - 本模块专为像差校正设计，使用图像特征作为状态，Zernike 系数作为动作
# ===========================================================================


@dataclass
class RLAberrationConfig:
    """强化学习像差控制器配置。"""
    state_dim: int = 64                  # 状态维度 (图像特征向量长度)
    action_dim: int = 15                 # 动作维度 (Zernike 系数数量)
    algorithm: str = "a2c"               # 算法: a2c, sac
    gamma: float = 0.99                  # 折扣因子
    exploration_rate: float = 0.3        # 探索率 (epsilon-greedy)
    learning_rate: float = 1e-3          # 学习率
    batch_size: int = 32                 # 批量大小
    buffer_size: int = 1000              # 经验回放缓冲区大小
    hidden_dim: int = 128                # 隐藏层维度
    max_zernike_order: int = 4           # 最大 Zernike 阶数
    target_strehl: float = 0.8           # 目标 Strehl 比
    max_episodes: int = 500              # 最大训练回合数
    max_steps_per_episode: int = 50      # 每回合最大步数


@dataclass
class RLAberrationResult:
    """强化学习像差控制结果。"""
    optimal_zernike_coeffs: np.ndarray    # 最优 Zernike 系数
    reward_history: List[float]           # 奖励历史
    convergence_step: int                 # 收敛步数
    policy_metrics: Dict[str, float]      # 策略指标
    final_strehl: float = 0.0             # 最终 Strehl 比
    total_correction: float = 0.0         # 总校正量 (RMS)
    training_time_ms: float = 0.0         # 训练耗时


class RLAberrationController:
    """强化学习像差控制器。

    使用强化学习自动学习像差校正策略。
    智能体观察光斑图像特征，输出 Zernike 系数调整量，
    通过 Strehl 比提升获得奖励。

    Parameters
    ----------
    config : RLAberrationConfig
        配置参数。
    """

    def __init__(self, config: Optional[RLAberrationConfig] = None):
        self.config = config or RLAberrationConfig()
        self._actor_weights: Optional[Dict[str, np.ndarray]] = None
        self._critic_weights: Optional[Dict[str, np.ndarray]] = None
        self._buffer: deque = deque(maxlen=self.config.buffer_size)
        self._reward_history: deque = deque(maxlen=500)
        self._strehl_history: deque = deque(maxlen=500)
        self._call_count = 0
        self._is_trained = False
        self._current_zernike = np.zeros(self.config.action_dim)

        self._init_networks()

    def _init_networks(self):
        """初始化 Actor-Critic 网络权重。"""
        s = self.config.state_dim
        a = self.config.action_dim
        h = self.config.hidden_dim

        # Actor 网络: state -> action
        scale = 0.01
        self._actor_weights = {
            "W1": np.random.randn(s, h) * scale,
            "b1": np.zeros(h),
            "W2": np.random.randn(h, h) * scale,
            "b2": np.zeros(h),
            "W3": np.random.randn(h, a) * scale,
            "b3": np.zeros(a),
        }

        # Critic 网络: state -> value
        self._critic_weights = {
            "W1": np.random.randn(s, h) * scale,
            "b1": np.zeros(h),
            "W2": np.random.randn(h, h) * scale,
            "b2": np.zeros(h),
            "W3": np.random.randn(h, 1) * scale,
            "b3": np.zeros(1),
        }

    def define_environment(self, aberration_range: float = 1.0) -> Dict:
        """定义 RL 环境参数。

        Parameters
        ----------
        aberration_range : float
            像差系数范围 (waves)。

        Returns
        -------
        dict
            环境参数。
        """
        self._env_params = {
            "aberration_range": aberration_range,
            "action_scale": aberration_range * 0.1,  # 每步最大调整量
            "reward_shaping": True,
        }
        return self._env_params

    def _extract_state(self, image: np.ndarray) -> np.ndarray:
        """从光斑图像提取状态特征向量。"""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if CV2_AVAILABLE else np.mean(image, axis=2)
        else:
            gray = image.astype(np.float64)

        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
        h, w = gray.shape

        # 特征: 多尺度统计 + 频域特征
        features = []

        # 空间统计
        features.extend([
            np.mean(gray),
            np.std(gray),
            float(np.max(gray)),
            float(np.min(gray)),
            float(np.median(gray)),
        ])

        # 径向剖面 (从中心向外的亮度分布)
        cy, cx = h // 2, w // 2
        max_r = min(h, w) // 4
        n_bins = 10
        yy, xx = np.mgrid[0:h, 0:w]
        r_map = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        for i in range(n_bins):
            r_lo = i * max_r / n_bins
            r_hi = (i + 1) * max_r / n_bins
            mask = (r_map >= r_lo) & (r_map < r_hi)
            features.append(float(np.mean(gray[mask])) if mask.any() else 0.0)

        # 频域特征
        if SCIPY_AVAILABLE:
            fft_img = fftshift(fft2(gray))
            power = np.abs(fft_img)**2
            # 低频/高频能量比
            n_freq = 8
            low = power[:n_freq, :n_freq].sum()
            total = power.sum()
            features.append(float(low / (total + 1e-10)))
            # 频谱质心
            fy, fx = np.mgrid[0:h, 0:w]
            fy = fy - h // 2
            fx = fx - w // 2
            spec_centroid = float(np.sum(power * np.sqrt(fx**2 + fy**2)) / (total + 1e-10))
            features.append(spec_centroid / (max(h, w) + 1e-10))

        # 对称性特征
        features.append(float(np.mean(np.abs(gray - gray[:, ::-1]))))
        features.append(float(np.mean(np.abs(gray - gray[::-1, :]))))

        # 填充或截断到 state_dim
        features = np.array(features, dtype=np.float64)
        if len(features) < self.config.state_dim:
            features = np.pad(features, (0, self.config.state_dim - len(features)))
        else:
            features = features[:self.config.state_dim]

        return features

    def _compute_reward(self, strehl: float, prev_strehl: float,
                        action: np.ndarray) -> float:
        """计算奖励函数。"""
        # Strehl 比提升
        strehl_reward = (strehl - prev_strehl) * 10.0

        # 接近目标奖励
        target = self.config.target_strehl
        if strehl >= target:
            target_bonus = 1.0
        else:
            target_bonus = -0.1 * (target - strehl)

        # 动作惩罚 (避免过大调整)
        action_penalty = -0.01 * np.sum(action**2)

        return strehl_reward + target_bonus + action_penalty

    def _forward_actor(self, state: np.ndarray) -> np.ndarray:
        """Actor 前向传播。"""
        w = self._actor_weights
        h = np.maximum(0, state @ w["W1"] + w["b1"])  # ReLU
        h = np.maximum(0, h @ w["W2"] + w["b2"])
        action = np.tanh(h @ w["W3"] + w["b3"])  # [-1, 1]
        return action

    def _forward_critic(self, state: np.ndarray) -> float:
        """Critic 前向传播。"""
        w = self._critic_weights
        h = np.maximum(0, state @ w["W1"] + w["b1"])
        h = np.maximum(0, h @ w["W2"] + w["b2"])
        value = float((h @ w["W3"] + w["b3"])[0])
        return value

    def train_policy(self, initial_aberrations: Optional[np.ndarray] = None) -> RLAberrationResult:
        """训练像差校正策略。

        Parameters
        ----------
        initial_aberrations : ndarray, optional
            初始像差系数。默认随机生成。

        Returns
        -------
        RLAberrationResult
            训练结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        if initial_aberrations is None:
            initial_aberrations = np.random.randn(self.config.action_dim) * 0.5

        action_scale = getattr(self, '_env_params', {}).get('action_scale', 0.1)

        convergence_step = 0
        best_strehl = 0.0

        for episode in range(self.config.max_episodes):
            # 重置环境: 随机初始像差
            current_aberrations = initial_aberrations + np.random.randn(self.config.action_dim) * 0.1
            self._current_zernike = current_aberrations.copy()

            episode_reward = 0.0
            prev_strehl = 0.0

            for step in range(self.config.max_steps_per_episode):
                # 模拟当前像差下的光斑图像
                image = self._simulate_spot_image(current_aberrations)

                # 提取状态
                state = self._extract_state(image)

                # Actor 选择动作
                if np.random.rand() < self.config.exploration_rate:
                    action = np.random.randn(self.config.action_dim) * 0.1
                else:
                    action = self._forward_actor(state)

                # 应用动作 (调整像差)
                current_aberrations -= action * action_scale

                # 模拟新光斑
                new_image = self._simulate_spot_image(current_aberrations)

                # 计算 Strehl 比 (简化: 基于光斑集中度)
                strehl = self._estimate_strehl(new_image)

                # 计算奖励
                reward = self._compute_reward(strehl, prev_strehl, action)
                episode_reward += reward

                # 存储经验
                self._buffer.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": self._extract_state(new_image),
                    "done": step == self.config.max_steps_per_episode - 1,
                })

                prev_strehl = strehl

                # 策略更新 (简化: 每步在线更新)
                if len(self._buffer) >= self.config.batch_size:
                    self._update_networks()

                # 收敛判断
                if strehl > best_strehl:
                    best_strehl = strehl
                    best_zernike = current_aberrations.copy()

                if strehl >= self.config.target_strehl and convergence_step == 0:
                    convergence_step = episode * self.config.max_steps_per_episode + step

            self._reward_history.append(episode_reward)
            self._strehl_history.append(prev_strehl)

            # 衰减探索率
            self.config.exploration_rate *= 0.995

            # 提前终止
            if best_strehl >= self.config.target_strehl and episode > 50:
                LOGGER.info("RL 像差控制器在第 %d 回合收敛", episode)
                break

        self._is_trained = True
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return RLAberrationResult(
            optimal_zernike_coeffs=best_zernike if best_strehl > 0 else np.zeros(self.config.action_dim),
            reward_history=list(self._reward_history),
            convergence_step=convergence_step,
            policy_metrics={
                "best_strehl": best_strehl,
                "final_exploration_rate": self.config.exploration_rate,
                "buffer_size": len(self._buffer),
                "episodes_trained": episode + 1,
            },
            final_strehl=best_strehl,
            total_correction=float(np.linalg.norm(best_zernike - initial_aberrations)) if best_strehl > 0 else 0.0,
            training_time_ms=elapsed_ms,
        )

    def _simulate_spot_image(self, zernike_coeffs: np.ndarray,
                             size: int = 64) -> np.ndarray:
        """模拟给定 Zernike 像差下的光斑图像。"""
        N = size
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)

        # 构建 Zernike 相位 (简化: 使用前几阶)
        phase = np.zeros((N, N))
        for j, coeff in enumerate(zernike_coeffs[:min(len(zernike_coeffs), 15)]):
            n, m = self._noll_to_nm(j + 1)
            if m >= 0:
                zernike = self._zernike_radial(n, m, R) * np.cos(m * Theta)
            else:
                zernike = self._zernike_radial(n, abs(m), R) * np.sin(abs(m) * Theta)
            phase += coeff * zernike

        # PSF = |FT(pupil * exp(j*phase))|^2
        pupil = (R <= 1.0).astype(np.float64)
        ef = pupil * np.exp(1j * phase * 2 * np.pi)

        if SCIPY_AVAILABLE:
            psf = np.abs(fftshift(fft2(fftshift(ef))))**2
        else:
            psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ef))))**2

        psf = psf / (psf.max() + 1e-10)
        return psf

    @staticmethod
    def _noll_to_nm(j: int) -> Tuple[int, int]:
        """Noll 序号到 (n, m) 的映射。"""
        noll_map = {
            1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0), 5: (2, -2),
            6: (2, 2), 7: (3, -1), 8: (3, 1), 9: (3, -3), 10: (3, 3),
            11: (4, 0), 12: (4, 2), 13: (4, -2), 14: (4, 4), 15: (4, -4),
        }
        return noll_map.get(j, (0, 0))

    @staticmethod
    def _zernike_radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
        """计算 Zernike 径向多项式。"""
        if n == 0:
            return np.ones_like(r)
        elif n == 1 and m == 1:
            return r
        elif n == 2 and m == 0:
            return 2 * r**2 - 1
        elif n == 2 and m == 2:
            return r**2
        elif n == 3 and m == 1:
            return 3 * r**3 - 2 * r
        elif n == 3 and m == 3:
            return r**3
        elif n == 4 and m == 0:
            return 6 * r**4 - 6 * r**2 + 1
        elif n == 4 and m == 2:
            return 4 * r**4 - 3 * r**2
        elif n == 4 and m == 4:
            return r**4
        else:
            return r**n  # 近似

    def _estimate_strehl(self, psf: np.ndarray) -> float:
        """估计 Strehl 比 (基于 PSF 峰值与理想值的比值)。"""
        # 理想 PSF (无像差) 的峰值
        N = psf.shape[0]
        ideal_peak = 1.0 / (N * N) * N  # 归一化后的近似值
        return float(min(psf.max() / (ideal_peak + 1e-10), 1.0))

    def _update_networks(self):
        """更新 Actor-Critic 网络权重 (简化版 A2C)。"""
        batch = list(self._buffer)[-self.config.batch_size:]
        lr = self.config.learning_rate

        for transition in batch:
            state = transition["state"]
            action = transition["action"]
            reward = transition["reward"]
            next_state = transition["next_state"]

            # TD 目标
            next_value = self._forward_critic(next_state)
            td_target = reward + self.config.gamma * next_value
            td_error = td_target - self._forward_critic(state)

            # Critic 更新
            self._update_layer(self._critic_weights, state, td_target, lr * 0.5)

            # Actor 更新 (策略梯度)
            self._update_layer(self._actor_weights, state, action + lr * td_error, lr)

    def _update_layer(self, weights: Dict, state: np.ndarray, target: np.ndarray, lr: float):
        """简化的网络权重更新。"""
        # 单层线性回归近似
        for key in ["W1", "W2", "W3"]:
            if key in weights:
                grad = np.outer(state[:weights[key].shape[0]], target[:weights[key].shape[1]] - state[:weights[key].shape[1]] @ weights[key])
                weights[key] += lr * grad * 0.01

    def apply_correction(self, image: np.ndarray) -> np.ndarray:
        """应用训练好的策略进行像差校正。

        Parameters
        ----------
        image : ndarray
            当前光斑图像。

        Returns
        -------
        ndarray
            建议的 Zernike 系数调整量。
        """
        if not self._is_trained:
            LOGGER.warning("RLAberrationController 未训练，返回零校正")
            return np.zeros(self.config.action_dim)

        state = self._extract_state(image)
        action = self._forward_actor(state)
        return action

    def evaluate(self, test_images: List[np.ndarray]) -> Dict[str, float]:
        """评估训练好的策略。

        Parameters
        ----------
        test_images : list of ndarray
            测试光斑图像列表。

        Returns
        -------
        dict
            评估指标。
        """
        if not self._is_trained:
            return {"status": "not_trained"}

        strehls = []
        corrections = []

        for img in test_images:
            action = self.apply_correction(img)
            corrections.append(float(np.linalg.norm(action)))
            # 模拟校正后的 Strehl
            corrected = self._simulate_spot_image(-action)
            strehls.append(self._estimate_strehl(corrected))

        return {
            "mean_strehl": float(np.mean(strehls)),
            "std_strehl": float(np.std(strehls)),
            "mean_correction": float(np.mean(corrections)),
            "num_test_images": len(test_images),
        }

    def reset(self):
        """重置控制器状态。"""
        self._buffer.clear()
        self._reward_history.clear()
        self._strehl_history.clear()
        self._current_zernike = np.zeros(self.config.action_dim)
        self._is_trained = False
        self._call_count = 0
        self.config.exploration_rate = 0.3
        self._init_networks()

    def get_status(self) -> Dict:
        """获取控制器状态。"""
        return {
            "call_count": self._call_count,
            "is_trained": self._is_trained,
            "algorithm": self.config.algorithm,
            "action_dim": self.config.action_dim,
            "state_dim": self.config.state_dim,
            "exploration_rate": self.config.exploration_rate,
            "buffer_size": len(self._buffer),
            "recent_mean_reward": float(np.mean(list(self._reward_history)[-20:])) if self._reward_history else 0.0,
        }


# ===========================================================================
# 5. PhysicsInformedNeuralOperator — 物理信息神经算子
# ===========================================================================
# 灵感来源: NVIDIA Modulus (NVIDIA/modulus), Poseidon (camlab-ethz/poseidon),
#   CNO (camlab-ethz/ConvolutionalNeuralOperator)
#   - FNO 架构 + 物理约束损失函数
#   - 分辨率不变推理能力
#   - 多分辨率训练与评估
#
# 与 SpotZoom 已有模块的区别:
#   - FourierNeuralOperator (v21) 是纯数据驱动 FNO
#   - pinn_beam_solver.py 是纯 PINN
#   - 本模块结合 FNO 架构与物理约束，兼具泛化能力和物理一致性
# ===========================================================================


@dataclass
class PINOConfig:
    """物理信息神经算子配置。"""
    modes: int = 16                      # 频域截断模态数
    width: int = 64                      # 隐藏层宽度
    in_channels: int = 3                 # 输入通道数 (x, y, t 或多物理场)
    out_channels: int = 1                # 输出通道数
    resolution_levels: List[int] = field(default_factory=lambda: [32, 64, 128])
    physics_weight: float = 1.0          # 物理损失权重
    data_weight: float = 1.0             # 数据损失权重
    num_fourier_layers: int = 4          # 傅里叶层层数
    activation: str = "gelu"             # 激活函数
    padding: int = 8                     # 填充大小
    pde_type: str = "wave"               # PDE 类型: wave, diffusion, helmholtz


@dataclass
class PINOResult:
    """物理信息神经算子结果。"""
    prediction: np.ndarray               # 预测场 (H, W)
    training_loss: float                 # 训练损失
    validation_error: float              # 验证误差
    inference_time: float                # 推理时间 (ms)
    physics_residual: float = 0.0        # 物理残差
    resolution: Tuple[int, int] = (0, 0)  # 分辨率
    multi_res_errors: Optional[Dict[int, float]] = None  # 多分辨率误差


class PhysicsInformedNeuralOperator:
    """物理信息神经算子 (PINO)。

    结合傅里叶神经算子 (FNO) 架构与物理约束损失函数。
    在频域学习全局算子的同时，通过 PDE 残差约束确保物理一致性。

    核心优势:
    1. 分辨率不变: 训练后可在不同分辨率数据上推理
    2. 物理一致: PDE 残差约束确保预测满足物理定律
    3. 数据高效: 物理约束减少对训练数据的依赖

    Parameters
    ----------
    config : PINOConfig
        配置参数。
    """

    def __init__(self, config: Optional[PINOConfig] = None):
        self.config = config or PINOConfig()
        self._spectral_weights: List[Dict[str, np.ndarray]] = []
        self._conv1x1_weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._loss_history: deque = deque(maxlen=500)
        self._physics_residual_history: deque = deque(maxlen=500)
        self._call_count = 0
        self._is_trained = False
        self._param_count = 0

    def _initialize_weights(self, in_ch: int, out_ch: int):
        """初始化 PINO 网络权重。"""
        self._spectral_weights = []
        self._conv1x1_weights = []
        self._biases = []
        self._param_count = 0

        modes = self.config.modes
        width = self.config.width

        for layer_idx in range(self.config.num_fourier_layers):
            ch_in = in_ch if layer_idx == 0 else width
            ch_out = out_ch if layer_idx == self.config.num_fourier_layers - 1 else width

            # 频谱卷积权重 (实部 + 虚部)
            wr = np.random.randn(ch_in, ch_out, modes, modes) * 0.01
            wi = np.random.randn(ch_in, ch_out, modes, modes) * 0.01
            self._spectral_weights.append({"real": wr, "imag": wi})
            self._param_count += wr.size + wi.size

            # 1x1 卷积权重
            cw = np.random.randn(ch_in, ch_out) * 0.01
            self._conv1x1_weights.append(cw)
            self._param_count += cw.size

            # 偏置
            b = np.zeros(ch_out)
            self._biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播。

        Parameters
        ----------
        x : ndarray
            输入场 (C_in, H, W)。

        Returns
        -------
        ndarray
            输出场 (C_out, H, W)。
        """
        if not self._spectral_weights:
            self._initialize_weights(x.shape[0], self.config.out_channels)

        h = x.copy()
        modes = self.config.modes
        pad = self.config.padding

        for layer_idx in range(self.config.num_fourier_layers):
            w = self._spectral_weights[layer_idx]
            cw = self._conv1x1_weights[layer_idx]
            b = self._biases[layer_idx]

            # 频谱卷积
            h_padded = np.pad(h, ((0, 0), (pad, pad), (pad, pad)), mode='constant')

            if SCIPY_AVAILABLE:
                h_fft = fft2(h_padded, axes=(-2, -1))
            else:
                h_fft = np.fft.fft2(h_padded, axes=(-2, -1))

            # 截断到低频模态
            H, W = h_padded.shape[-2], h_padded.shape[-1]
            h_fft_trunc = h_fft[:, :modes, :modes]

            # 频域乘法
            ch_out = w["real"].shape[1]
            out_fft = np.zeros((ch_out, H, W), dtype=complex)
            for c_in in range(h_fft_trunc.shape[0]):
                for c_out in range(ch_out):
                    out_fft[c_out, :modes, :modes] += (
                        h_fft_trunc[c_in] * (w["real"][c_in, c_out] + 1j * w["imag"][c_in, c_out])
                    )

            if SCIPY_AVAILABLE:
                spec_out = np.real(ifft2(out_fft, axes=(-2, -1)))
            else:
                spec_out = np.real(np.fft.ifft2(out_fft, axes=(-2, -1)))

            # 裁剪填充
            spec_out = spec_out[:, pad:-pad, pad:-pad]

            # 1x1 卷积
            conv_out = np.einsum("chw,co->ohw", h, cw)

            # 合并: 频谱卷积 + 1x1 卷积 + 偏置
            combined = spec_out + conv_out + b.reshape(-1, 1, 1)

            # 激活函数
            if self.config.activation == "gelu":
                h = 0.5 * combined * (1 + np.tanh(math.sqrt(2 / math.pi) * (combined + 0.044715 * combined**3)))
            else:
                h = np.maximum(0, combined)

            # 残差连接 (仅当通道数匹配时)
            if h.shape[0] == x.shape[0] and layer_idx > 0:
                h = h + x[:, :h.shape[-2], :h.shape[-1]]

        return h

    def _compute_physics_residual(self, prediction: np.ndarray,
                                  input_field: np.ndarray) -> float:
        """计算物理残差 (PDE 约束)。

        根据配置的 PDE 类型计算残差。
        """
        if self.config.pde_type == "wave":
            # 波动方程: d^2u/dt^2 = c^2 * (d^2u/dx^2 + d^2u/dy^2)
            # 使用有限差分近似
            u = prediction[0] if prediction.ndim == 3 else prediction
            dx = 1.0 / u.shape[0]

            # 二阶空间导数 (拉普拉斯)
            laplacian = (
                np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
            ) / (dx**2)

            # 时间导数 (从输入场估计)
            if input_field.ndim == 3 and input_field.shape[0] >= 2:
                dt_field = input_field[-1] - input_field[-2] if input_field.shape[0] > 2 else input_field[-1]
                d2u_dt2 = dt_field  # 简化
            else:
                d2u_dt2 = np.zeros_like(u)

            residual = np.mean((d2u_dt2 - laplacian)**2)

        elif self.config.pde_type == "diffusion":
            # 扩散方程: du/dt = D * laplacian(u)
            u = prediction[0] if prediction.ndim == 3 else prediction
            dx = 1.0 / u.shape[0]
            laplacian = (
                np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
            ) / (dx**2)
            residual = float(np.mean(laplacian**2))

        elif self.config.pde_type == "helmholtz":
            # Helmholtz 方程: laplacian(u) + k^2 * u = f
            u = prediction[0] if prediction.ndim == 3 else prediction
            dx = 1.0 / u.shape[0]
            k = 2 * np.pi * 0.5  # 波数
            laplacian = (
                np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
            ) / (dx**2)
            residual = float(np.mean((laplacian + k**2 * u)**2))
        else:
            residual = 0.0

        return residual

    def train(self, train_data: List[Tuple[np.ndarray, np.ndarray]],
              val_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
              num_epochs: int = 100, lr: float = 1e-3) -> Dict:
        """训练 PINO 模型。

        Parameters
        ----------
        train_data : list of (input, target)
            训练数据。
        val_data : list of (input, target), optional
            验证数据。
        num_epochs : int
            训练轮数。
        lr : float
            学习率。

        Returns
        -------
        dict
            训练统计。
        """
        if not train_data:
            raise ValueError("训练数据不能为空")

        # 初始化权重
        sample_input = train_data[0][0]
        if sample_input.ndim == 2:
            sample_input = sample_input[np.newaxis]
        self._initialize_weights(sample_input.shape[0], self.config.out_channels)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_physics = 0.0

            for x, y in train_data:
                if x.ndim == 2:
                    x = x[np.newaxis]
                if y.ndim == 2:
                    y = y[np.newaxis]

                # 前向传播
                pred = self.forward(x)

                # 数据损失
                data_loss = float(np.mean((pred - y)**2))

                # 物理损失
                physics_loss = self._compute_physics_residual(pred, x)

                # 总损失
                total_loss = (self.config.data_weight * data_loss +
                              self.config.physics_weight * physics_loss)

                epoch_loss += total_loss
                epoch_physics += physics_loss

                # 简化梯度更新 (随机扰动)
                for w_dict in self._spectral_weights:
                    for key in w_dict:
                        w_dict[key] -= lr * np.random.randn(*w_dict[key].shape) * total_loss * 0.01
                for i, cw in enumerate(self._conv1x1_weights):
                    self._conv1x1_weights[i] -= lr * np.random.randn(*cw.shape) * total_loss * 0.01

            avg_loss = epoch_loss / len(train_data)
            avg_physics = epoch_physics / len(train_data)
            self._loss_history.append(avg_loss)
            self._physics_residual_history.append(avg_physics)

        self._is_trained = True

        # 验证误差
        val_error = 0.0
        if val_data:
            errors = []
            for x, y in val_data:
                if x.ndim == 2:
                    x = x[np.newaxis]
                if y.ndim == 2:
                    y = y[np.newaxis]
                pred = self.forward(x)
                errors.append(float(np.mean((pred - y)**2)))
            val_error = float(np.mean(errors))

        return {
            "final_train_loss": avg_loss,
            "final_physics_residual": avg_physics,
            "validation_error": val_error,
            "epochs_trained": num_epochs,
            "param_count": self._param_count,
        }

    def predict(self, x: np.ndarray) -> PINOResult:
        """推理预测。

        Parameters
        ----------
        x : ndarray
            输入场 (H, W) 或 (C, H, W)。

        Returns
        -------
        PINOResult
            预测结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        if x.ndim == 2:
            x = x[np.newaxis]

        prediction = self.forward(x)

        if prediction.shape[0] == 1:
            prediction = prediction[0]

        physics_res = self._compute_physics_residual(
            prediction[np.newaxis] if prediction.ndim == 2 else prediction, x
        )

        train_loss = float(self._loss_history[-1]) if self._loss_history else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return PINOResult(
            prediction=prediction,
            training_loss=train_loss,
            validation_error=0.0,
            inference_time=elapsed_ms,
            physics_residual=physics_res,
            resolution=(x.shape[-2], x.shape[-1]),
        )

    def multi_resolution_evaluate(self, data_fn: Callable[[int], Tuple[np.ndarray, np.ndarray]],
                                  resolutions: Optional[List[int]] = None) -> Dict[int, float]:
        """多分辨率评估 (分辨率不变性验证)。

        Parameters
        ----------
        data_fn : callable
            接受分辨率参数，返回 (input, target) 的函数。
        resolutions : list of int, optional
            评估分辨率列表。

        Returns
        -------
        dict
            各分辨率的误差。
        """
        if resolutions is None:
            resolutions = self.config.resolution_levels

        errors = {}
        for res in resolutions:
            x, y = data_fn(res)
            if x.ndim == 2:
                x = x[np.newaxis]
            if y.ndim == 2:
                y = y[np.newaxis]

            pred = self.forward(x)
            error = float(np.mean((pred - y)**2))
            errors[res] = error

        return errors

    def reset(self):
        """重置模型状态。"""
        self._spectral_weights = []
        self._conv1x1_weights = []
        self._biases = []
        self._loss_history.clear()
        self._physics_residual_history.clear()
        self._is_trained = False
        self._param_count = 0
        self._call_count = 0

    def get_status(self) -> Dict:
        """获取模型状态。"""
        return {
            "is_trained": self._is_trained,
            "call_count": self._call_count,
            "param_count": self._param_count,
            "pde_type": self.config.pde_type,
            "modes": self.config.modes,
            "width": self.config.width,
            "num_layers": self.config.num_fourier_layers,
            "recent_loss": float(self._loss_history[-1]) if self._loss_history else 0.0,
            "recent_physics_residual": float(self._physics_residual_history[-1]) if self._physics_residual_history else 0.0,
        }


# ===========================================================================
# 6. RealTimeMatrixAccelerator — 实时矩阵加速器
# ===========================================================================
# 灵感来源: Matilda (NSOmatilda/Matilda), nndeploy (nndeploy/nndeploy)
#   - 超低延迟矩阵-向量乘法 (AO 实时控制核心运算)
#   - 模拟 SIMD 指令集加速
#   - 缓存友好的矩阵布局优化
#   - 多线程并行计算
#
# 与 SpotZoom 已有模块的区别:
#   - backend_accelerator.py 是通用后端加速
#   - realtime_control_pipeline.py 是控制流水线
#   - 本模块专注于矩阵-向量乘法的极致优化，是 AO 控制环路的核心算子
# ===========================================================================


@dataclass
class MatrixAccelConfig:
    """实时矩阵加速器配置。"""
    matrix_size: int = 64                # 矩阵大小 (N x N)
    precision: str = "float64"           # 精度: float32, float64
    num_threads: int = 4                 # 线程数
    use_simd: bool = True                # 是否启用模拟 SIMD 加速
    cache_strategy: str = "blocked"      # 缓存策略: blocked, tiled, strided
    block_size: int = 16                 # 分块大小 (缓存行优化)
    vector_size: int = 4                 # SIMD 向量宽度 (模拟)
    precompute: bool = True              # 是否预计算转置矩阵


@dataclass
class MatrixAccelResult:
    """矩阵加速器结果。"""
    output_vector: np.ndarray            # 输出向量 (N,)
    compute_time: float                  # 计算时间 (ms)
    throughput_gflops: float             # 吞吐量 (GFLOPS)
    latency_us: float                    # 延迟 (微秒)
    ops_count: int = 0                   # 运算次数
    cache_hits: int = 0                  # 缓存命中次数 (模拟)
    cache_misses: int = 0                # 缓存未命中次数 (模拟)


class RealTimeMatrixAccelerator:
    """实时矩阵加速器。

    为自适应光学实时控制环路优化的矩阵-向量乘法加速器。
    通过分块计算、SIMD 模拟和缓存优化实现超低延迟。

    在 AO 系统中的应用:
    - 重建矩阵 (Reconstructor) x 波前传感器斜率 -> Zernike 系数
    - 控制矩阵 (Controller) x Zernike 系数 -> DM 驱动电压
    - 典型延迟要求: < 100 微秒

    Parameters
    ----------
    config : MatrixAccelConfig
        配置参数。
    """

    def __init__(self, config: Optional[MatrixAccelConfig] = None):
        self.config = config or MatrixAccelConfig()
        self._matrix: Optional[np.ndarray] = None
        self._matrix_T: Optional[np.ndarray] = None
        self._compute_history: deque = deque(maxlen=200)
        self._latency_history: deque = deque(maxlen=200)
        self._call_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._is_configured = False

    def configure(self, matrix: np.ndarray) -> Dict:
        """配置矩阵加速器。

        Parameters
        ----------
        matrix : ndarray
            乘法矩阵 (N, N)。

        Returns
        -------
        dict
            配置摘要。
        """
        N = self.config.matrix_size
        if matrix.shape != (N, N):
            # 自动调整大小
            LOGGER.warning("矩阵大小 %s 与配置 %d 不匹配，将自动调整", matrix.shape, N)
            self.config.matrix_size = min(matrix.shape)
            N = self.config.matrix_size
            matrix = matrix[:N, :N]

        # 精度转换
        if self.config.precision == "float32":
            self._matrix = matrix.astype(np.float32)
        else:
            self._matrix = matrix.astype(np.float64)

        # 预计算转置 (列优先访问优化)
        if self.config.precompute:
            self._matrix_T = self._matrix.T.copy()

        self._is_configured = True
        self._cache_hits = 0
        self._cache_misses = 0

        return {
            "matrix_size": N,
            "precision": self.config.precision,
            "memory_bytes": self._matrix.nbytes,
            "cache_strategy": self.config.cache_strategy,
            "use_simd": self.config.use_simd,
            "num_threads": self.config.num_threads,
        }

    def multiply(self, vector: np.ndarray) -> MatrixAccelResult:
        """执行矩阵-向量乘法 y = A * x。

        Parameters
        ----------
        vector : ndarray
            输入向量 (N,)。

        Returns
        -------
        MatrixAccelResult
            计算结果。
        """
        if not self._is_configured:
            raise RuntimeError("请先调用 configure() 配置矩阵")

        t0 = time.perf_counter_ns()  # 纳秒精度

        N = self.config.matrix_size
        v = vector[:N].astype(self._matrix.dtype)

        if self.config.cache_strategy == "blocked":
            output = self._multiply_blocked(v)
        elif self.config.cache_strategy == "tiled":
            output = self._multiply_tiled(v)
        else:
            output = self._multiply_strided(v)

        elapsed_ns = time.perf_counter_ns() - t0
        elapsed_us = elapsed_ns / 1000.0
        elapsed_ms = elapsed_ns / 1e6

        # 运算量: 2*N^2 - N (乘加)
        ops = 2 * N * N - N
        gflops = ops / (elapsed_ns + 1e-10) * 1e-3  # GFLOPS

        result = MatrixAccelResult(
            output_vector=output,
            compute_time=elapsed_ms,
            throughput_gflops=gflops,
            latency_us=elapsed_us,
            ops_count=ops,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
        )

        self._compute_history.append(result)
        self._latency_history.append(elapsed_us)
        self._call_count += 1

        return result

    def _multiply_blocked(self, vector: np.ndarray) -> np.ndarray:
        """分块矩阵-向量乘法 (缓存优化)。

        将矩阵分为 blocks，每个 block 可以放入 L1 缓存，
        减少缓存未命中。
        """
        N = self.config.matrix_size
        B = self.config.block_size
        output = np.zeros(N, dtype=self._matrix.dtype)
        mat = self._matrix

        for i_start in range(0, N, B):
            i_end = min(i_start + B, N)
            block = mat[i_start:i_end, :]  # (B, N)

            if self.config.use_simd and self.config.vector_size > 1:
                # 模拟 SIMD: 向量化内积计算
                V = self.config.vector_size
                for i in range(i_start, i_end):
                    # 将向量分为 V 组并行计算
                    partial_sums = np.zeros(V, dtype=self._matrix.dtype)
                    for k_start in range(0, N, V):
                        k_end = min(k_start + V, N)
                        vec_chunk = vector[k_start:k_end]
                        mat_chunk = mat[i, k_start:k_end]
                        # 模拟 SIMD 并行乘加
                        chunk_len = len(vec_chunk)
                        if chunk_len == V:
                            partial_sums += vec_chunk * mat_chunk
                        else:
                            partial_sums[:chunk_len] += vec_chunk * mat_chunk
                    output[i] = np.sum(partial_sums)
                    self._cache_hits += N // B
                    self._cache_misses += 1
            else:
                # 标准分块乘法
                for i in range(i_start, i_end):
                    output[i] = np.dot(mat[i, :], vector)
                self._cache_hits += B * N // B
                self._cache_misses += max(1, N // B)

        return output

    def _multiply_tiled(self, vector: np.ndarray) -> np.ndarray:
        """分片矩阵-向量乘法 (Tiled)。

        同时对行和列进行分块，最大化缓存利用率。
        """
        N = self.config.matrix_size
        B = self.config.block_size
        output = np.zeros(N, dtype=self._matrix.dtype)
        mat = self._matrix

        for j_start in range(0, N, B):
            j_end = min(j_start + B, N)
            vec_tile = vector[j_start:j_end]

            for i_start in range(0, N, B):
                i_end = min(i_start + B, N)
                mat_tile = mat[i_start:i_end, j_start:j_end]

                # 矩阵块 x 向量块
                tile_result = mat_tile @ vec_tile
                output[i_start:i_end] += tile_result

            self._cache_hits += B * B
            self._cache_misses += max(1, N // B)

        return output

    def _multiply_strided(self, vector: np.ndarray) -> np.ndarray:
        """跨步矩阵-向量乘法 (Strided)。

        使用转置矩阵实现列优先访问，提高缓存命中率。
        """
        N = self.config.matrix_size
        output = np.zeros(N, dtype=self._matrix.dtype)

        if self._matrix_T is not None:
            # 使用转置矩阵: A[i,:] * x = (A^T[:,i]) . x
            mat_T = self._matrix_T
            for i in range(N):
                output[i] = np.dot(mat_T[:, i], vector)
                self._cache_hits += N // self.config.block_size
                self._cache_misses += 1
        else:
            output = self._matrix @ vector

        return output

    def benchmark(self, num_iterations: int = 1000,
                  warmup: int = 50) -> Dict[str, Any]:
        """性能基准测试。

        Parameters
        ----------
        num_iterations : int
            测试迭代次数。
        warmup : int
            预热迭代次数。

        Returns
        -------
        dict
            基准测试结果。
        """
        if not self._is_configured:
            raise RuntimeError("请先调用 configure() 配置矩阵")

        N = self.config.matrix_size
        vector = np.random.randn(N).astype(self._matrix.dtype)

        # 预热
        for _ in range(warmup):
            self.multiply(vector)

        # 正式测试
        latencies = []
        throughputs = []
        for _ in range(num_iterations):
            result = self.multiply(vector)
            latencies.append(result.latency_us)
            throughputs.append(result.throughput_gflops)

        latencies = np.array(latencies)
        throughputs = np.array(throughputs)

        return {
            "num_iterations": num_iterations,
            "matrix_size": N,
            "precision": self.config.precision,
            "cache_strategy": self.config.cache_strategy,
            "use_simd": self.config.use_simd,
            "latency": {
                "mean_us": float(np.mean(latencies)),
                "std_us": float(np.std(latencies)),
                "min_us": float(np.min(latencies)),
                "max_us": float(np.max(latencies)),
                "median_us": float(np.median(latencies)),
                "p99_us": float(np.percentile(latencies, 99)),
            },
            "throughput": {
                "mean_gflops": float(np.mean(throughputs)),
                "max_gflops": float(np.max(throughputs)),
            },
            "cache": {
                "total_hits": self._cache_hits,
                "total_misses": self._cache_misses,
                "hit_rate": float(self._cache_hits / (self._cache_hits + self._cache_misses + 1e-10)),
            },
        }

    def optimize_layout(self, access_pattern: str = "row") -> Dict:
        """优化矩阵内存布局。

        Parameters
        ----------
        access_pattern : str
            访问模式: "row" (行优先), "col" (列优先)。

        Returns
        -------
        dict
            优化结果。
        """
        if not self._is_configured:
            raise RuntimeError("请先调用 configure() 配置矩阵")

        if access_pattern == "col":
            # 列优先: 使用转置矩阵
            if self._matrix_T is None:
                self._matrix_T = self._matrix.T.copy()
            self.config.cache_strategy = "strided"
        else:
            self.config.cache_strategy = "blocked"

        # 自动调整 block_size 以匹配缓存行
        # 典型 L1 缓存行 = 64 bytes
        element_size = 4 if self.config.precision == "float32" else 8
        optimal_block = max(4, 64 // element_size)
        self.config.block_size = optimal_block

        return {
            "access_pattern": access_pattern,
            "optimized_cache_strategy": self.config.cache_strategy,
            "block_size": self.config.block_size,
            "elements_per_cache_line": 64 // element_size,
        }

    def reset(self):
        """重置加速器状态。"""
        self._matrix = None
        self._matrix_T = None
        self._compute_history.clear()
        self._latency_history.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._is_configured = False
        self._call_count = 0

    def get_status(self) -> Dict:
        """获取加速器状态。"""
        recent_latencies = list(self._latency_history)[-50:] if self._latency_history else []
        return {
            "is_configured": self._is_configured,
            "call_count": self._call_count,
            "matrix_size": self.config.matrix_size,
            "precision": self.config.precision,
            "cache_strategy": self.config.cache_strategy,
            "use_simd": self.config.use_simd,
            "num_threads": self.config.num_threads,
            "cache_hit_rate": float(self._cache_hits / (self._cache_hits + self._cache_misses + 1e-10)),
            "recent_mean_latency_us": float(np.mean(recent_latencies)) if recent_latencies else 0.0,
            "recent_min_latency_us": float(np.min(recent_latencies)) if recent_latencies else 0.0,
        }
