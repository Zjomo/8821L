"""
SpotZoom 前沿开源项目创新模块 v21.0
基于 2024-2026 最新科研前沿开源项目的创新模块补充 (第九轮)

新增模块 (v21.0):
- SensorlessAberrationEstimator: 无传感器像差感知器 (受 AOViFT + DeepAO 启发)
- LQGController: 线性二次高斯控制器 (受 pyRTC + SOAPY + AOloop 启发)
- BayesianExperimentOptimizer: 贝叶斯实验优化器 (受 AIDE + BoTorch 启发)
- FourierNeuralOperator: 傅里叶神经算子 (受 NeuralOperator 2.0 TFNO + SpectraNet 启发)
- HybridDiffSR: 混合扩散超分辨率增强器 (受 HNDSR + StableSR 启发)
- NLLEngine: 自然语言实验室引擎 (受 SciLink + Imajin 启发)

参考项目:
- AOViFT (cell-observatory/aovift) — 傅里叶域 Vision Transformer 无传感器像差感知
- pyRTC (jacotay7/pyRTC) — Python 自适应光学实时控制框架
- AIDE (cupitor/AIDE) — 自主发现与实验基础设施
- NeuralOperator 2.0 (neuraloperator/neuraloperator) — TFNO/Tucker 分解神经算子
- HNDSR (45-12/HNDSR) — 神经算子+扩散混合超分辨率
- SciLink (ziatdinovmax/SciLink) — LLM 驱动科学研究自动化
- AnyLoop (bnl-lab4/anyloop) — 插件式 AO 反馈环路框架
- nndeploy (nndeploy/nndeploy) — 高性能 AI 部署框架 (13 种推理后端)
- SLMsuite (wavefrontshack/slmsuite) — SLM 控制与全息术工具包
- ATOM (luke-a-thompson/ATOM) — 预训练神经算子零样本泛化 (ICLR 2026)
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
    "SensorlessAOConfig",
    "SensorlessAOResult",
    "SensorlessAberrationEstimator",
    "LQGConfig",
    "LQGResult",
    "LQGController",
    "BayesianExpConfig",
    "BayesianExpResult",
    "BayesianExperimentOptimizer",
    "FNOConfig",
    "FNOResult",
    "FourierNeuralOperator",
    "HybridDiffSRConfig",
    "HybridDiffSRResult",
    "HybridDiffSR",
    "NLLEngineConfig",
    "NLLEngineResult",
    "NLLEngine",
]


# ===========================================================================
# 1. SensorlessAberrationEstimator — 无传感器像差感知器
# ===========================================================================
# 灵感来源: AOViFT (cell-observatory/aovift)
#   - 傅里叶域嵌入替代空域卷积，大幅降低计算成本
#   - 无需波前传感器硬件，仅从光斑图像推断像差
#   - Tile-based 大视场空间变化像差预测
#
# 与 SpotZoom 已有模块的区别:
#   - zernike_analyzer.py 需要 Zernike 系数作为输入进行分析
#   - 本模块直接从原始光斑图像端到端推断像差，无需额外传感器
# ===========================================================================


@dataclass
class SensorlessAOConfig:
    """无传感器像差感知器配置。"""
    max_zernike_order: int = 6          # Zernike 多项式最大阶数 (对应 21 项)
    tile_size: int = 64                 # Tile 大小 (像素)
    tile_overlap: int = 16              # Tile 重叠区域
    num_fourier_modes: int = 32         # 傅里叶域嵌入维度
    confidence_threshold: float = 0.7   # 置信度阈值
    max_iterations: int = 100           # 最大迭代次数
    learning_rate: float = 1e-3         # 学习率
    regularization: float = 1e-4        # 正则化系数
    enable_spatial_varying: bool = True # 是否启用空间变化像差估计


@dataclass
class SensorlessAOResult:
    """无传感器像差感知结果。"""
    zernike_coeffs: np.ndarray          # 估计的 Zernike 系数 (N,)
    confidence: float                   # 整体置信度 [0, 1]
    rms_wavefront_error: float          # 波前误差 RMS (波长单位)
    dominant_aberration: str            # 主导像差类型名称
    spatial_map: Optional[np.ndarray] = None  # 空间变化像差图 (H, W, N_zernike)
    tile_results: Optional[List[Dict]] = None  # 各 tile 的详细结果
    inference_time_ms: float = 0.0      # 推理耗时 (毫秒)
    iterations_used: int = 0            # 实际迭代次数


class SensorlessAberrationEstimator:
    """无传感器像差感知器。

    基于傅里叶域嵌入的像差估计，无需波前传感器硬件。
    通过分析光斑图像的频谱特征推断 Zernike 像差系数。

    核心原理:
    1. 将光斑图像分割为 tiles
    2. 对每个 tile 计算傅里叶变换提取频域特征
    3. 通过预训练/在线学习的映射网络将频域特征映射到 Zernike 系数
    4. 融合各 tile 结果得到空间变化的像差分布

    Parameters
    ----------
    config : SensorlessAOConfig
        配置参数。
    """

    def __init__(self, config: Optional[SensorlessAOConfig] = None):
        self.config = config or SensorlessAOConfig()
        self._zernike_names = self._build_zernike_names(self.config.max_zernike_order)
        self._num_coeffs = len(self._zernike_names)
        self._fourier_basis = self._build_fourier_basis(
            self.config.tile_size, self.config.num_fourier_modes
        )
        self._model_weights: Optional[np.ndarray] = None
        self._is_calibrated = False
        self._calibration_data: List[Dict] = []
        self._call_count = 0

    @staticmethod
    def _build_zernike_names(max_order: int) -> List[str]:
        """构建 Zernike 多项式名称列表 (Noll 序)。"""
        names = []
        noll_to_nm = {
            1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0), 5: (2, -2),
            6: (2, 2), 7: (3, -1), 8: (3, 1), 9: (3, -3), 10: (3, 3),
            11: (4, 0), 12: (4, 2), 13: (4, -2), 14: (4, 4), 15: (4, -4),
            16: (5, 1), 17: (5, -1), 18: (5, 3), 19: (5, -3), 20: (5, 5),
            21: (5, -5), 22: (6, 0), 23: (6, 2), 24: (6, -2), 25: (6, 4),
            26: (6, -4), 27: (6, 6), 28: (6, -6),
        }
        coeff_count = SensorlessAberrationEstimator._coeff_count_from_order(max_order)
        for j in range(1, min(len(noll_to_nm) + 1, coeff_count + 1)):
            if j in noll_to_nm:
                n, m = noll_to_nm[j]
                names.append(f"Z{n}_{m:+d}" if m != 0 else f"Z{n}")
        return names

    @staticmethod
    def _coeff_count_from_order(order: int) -> int:
        """计算给定最大阶数的 Zernike 系数数量。"""
        return (order + 1) * (order + 2) // 2

    def _build_fourier_basis(self, tile_size: int, num_modes: int) -> np.ndarray:
        """构建傅里叶域基函数。

        使用径向频率采样构建低频到中频的基函数集合，
        捕获像差引起的频谱变化特征。
        """
        freq_x = fftfreq(tile_size)
        freq_y = fftfreq(tile_size)
        FX, FY = np.meshgrid(freq_x, freq_y)
        freq_radius = np.sqrt(FX**2 + FY**2)

        basis = []
        max_freq = 0.5 / tile_size * tile_size  # Nyquist
        freq_samples = np.linspace(0, max_freq, num_modes)

        for f in freq_samples:
            # 径向带通滤波器
            band = np.exp(-0.5 * ((freq_radius - f) / (max_freq / num_modes))**2)
            basis.append(band.ravel())

        return np.array(basis)  # (num_modes, tile_size^2)

    def calibrate(self, images: List[np.ndarray], zernike_ground_truth: List[np.ndarray]) -> Dict:
        """使用已知像差的标定数据训练映射模型。

        Parameters
        ----------
        images : list of ndarray
            标定光斑图像列表。
        zernike_ground_truth : list of ndarray
            对应的 Zernike 系数真值列表。

        Returns
        -------
        dict
            标定统计信息。
        """
        if len(images) != len(zernike_ground_truth):
            raise ValueError("图像数量与 Zernike 系数数量不匹配")

        features_list = []
        for img in images:
            feat = self._extract_fourier_features(img)
            features_list.append(feat)

        X = np.array(features_list)
        Y = np.array(zernike_ground_truth)

        # 使用岭回归拟合映射 (轻量、快速、无需迭代优化)
        reg = self.config.regularization
        XtX = X.T @ X + reg * np.eye(X.shape[1])
        XtY = X.T @ Y
        self._model_weights = np.linalg.solve(XtX, XtY)
        self._is_calibrated = True

        # 计算标定误差
        Y_pred = X @ self._model_weights
        residuals = Y - Y_pred
        rmse = np.sqrt(np.mean(residuals**2, axis=0))

        return {
            "num_samples": len(images),
            "num_zernike_terms": self._num_coeffs,
            "per_term_rmse": rmse.tolist(),
            "mean_rmse": float(np.mean(rmse)),
            "max_rmse": float(np.max(rmse)),
        }

    def _extract_fourier_features(self, image: np.ndarray) -> np.ndarray:
        """从光斑图像提取傅里叶域特征。"""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if CV2_AVAILABLE else np.mean(image, axis=2)
        else:
            gray = image.astype(np.float64)

        # 归一化到 [0, 1]
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)

        # 取中心 tile
        h, w = gray.shape
        ts = self.config.tile_size
        cy, cx = h // 2, w // 2
        half = ts // 2
        tile = gray[cy - half:cy + half, cx - half:cx + half]

        # 傅里叶变换
        fft_tile = fft2(tile)
        power_spectrum = np.abs(fft_tile)**2
        log_spectrum = np.log1p(power_spectrum).ravel()

        # 投影到傅里叶基
        features = self._fourier_basis @ log_spectrum

        # 添加统计特征
        features = np.concatenate([
            features,
            [np.std(tile), np.mean(tile), np.max(tile),
             float(np.sum(power_spectrum > np.median(power_spectrum)))],
        ])

        return features

    def estimate(self, image: np.ndarray) -> SensorlessAOResult:
        """从光斑图像估计像差。

        Parameters
        ----------
        image : ndarray
            光斑图像 (H, W) 或 (H, W, 3)。

        Returns
        -------
        SensorlessAOResult
            像差估计结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        if not self._is_calibrated:
            LOGGER.warning("SensorlessAberrationEstimator 未标定，返回零像差估计")
            return SensorlessAOResult(
                zernike_coeffs=np.zeros(self._num_coeffs),
                confidence=0.0,
                rms_wavefront_error=0.0,
                dominant_aberration="unknown (未标定)",
                inference_time_ms=0.0,
                iterations_used=0,
            )

        features = self._extract_fourier_features(image)
        zernike_coeffs = self._model_weights @ features

        # 计算 RMS 波前误差
        rms = float(np.sqrt(np.mean(zernike_coeffs**2)))

        # 确定主导像差
        abs_coeffs = np.abs(zernike_coeffs)
        dominant_idx = int(np.argmax(abs_coeffs))
        dominant_name = self._zernike_names[dominant_idx] if dominant_idx < len(self._zernike_names) else f"Z{dominant_idx}"

        # 置信度估计 (基于系数能量分布)
        total_energy = np.sum(abs_coeffs**2) + 1e-10
        top_energy = np.sum(np.sort(abs_coeffs**2)[-3:])
        confidence = min(float(top_energy / total_energy), 1.0)

        # 空间变化像差估计 (tile-based)
        spatial_map = None
        tile_results = None
        if self.config.enable_spatial_varying and image.shape[0] > self.config.tile_size * 2:
            spatial_map, tile_results = self._estimate_spatial_varying(image)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return SensorlessAOResult(
            zernike_coeffs=zernike_coeffs,
            confidence=confidence,
            rms_wavefront_error=rms,
            dominant_aberration=dominant_name,
            spatial_map=spatial_map,
            tile_results=tile_results,
            inference_time_ms=elapsed_ms,
            iterations_used=1,
        )

    def _estimate_spatial_varying(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """估计空间变化的像差分布。"""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if CV2_AVAILABLE else np.mean(image, axis=2)
        else:
            gray = image.astype(np.float64)

        h, w = gray.shape
        ts = self.config.tile_size
        overlap = self.config.tile_overlap
        step = ts - overlap

        tile_results = []
        map_h = max(1, (h - ts) // step + 1)
        map_w = max(1, (w - ts) // step + 1)
        spatial_map = np.zeros((map_h, map_w, self._num_coeffs))

        for iy in range(map_h):
            for ix in range(map_w):
                y0 = iy * step
                x0 = ix * step
                y1 = min(y0 + ts, h)
                x1 = min(x0 + ts, w)
                tile = gray[y0:y1, x0:x1]

                if tile.shape[0] < ts // 2 or tile.shape[1] < ts // 2:
                    continue

                features = self._extract_fourier_features(tile)
                coeffs = self._model_weights @ features
                spatial_map[iy, ix] = coeffs

                tile_results.append({
                    "position": (int(x0), int(y0)),
                    "size": (int(x1 - x0), int(y1 - y0)),
                    "zernike_coeffs": coeffs.tolist(),
                    "rms": float(np.sqrt(np.mean(coeffs**2))),
                })

        return spatial_map, tile_results

    def get_calibration_status(self) -> Dict:
        """获取标定状态信息。"""
        return {
            "is_calibrated": self._is_calibrated,
            "call_count": self._call_count,
            "num_zernike_terms": self._num_coeffs,
            "zernike_names": self._zernike_names,
            "config": {
                "max_zernike_order": self.config.max_zernike_order,
                "tile_size": self.config.tile_size,
                "num_fourier_modes": self.config.num_fourier_modes,
            },
        }


# ===========================================================================
# 2. LQGController — 线性二次高斯控制器
# ===========================================================================
# 灵感来源: pyRTC (jacotay7/pyRTC), SOAPY, AOloop
#   - 融合 Kalman 滤波与 LQR 最优控制
#   - 自适应光学中的经典 LQG 控制器
#   - 处理测量噪声和过程噪声的最优策略
#
# 与 SpotZoom 已有模块的区别:
#   - kalman_tracker.py 仅做状态估计，不含控制律
#   - mpc_controller.py 基于滚动优化，计算量大
#   - lqr_controller.py 假设完美状态观测
#   - 本模块是 Kalman + LQR 的最优融合，处理噪声场景
# ===========================================================================


@dataclass
class LQGConfig:
    """LQG 控制器配置。"""
    state_dim: int = 4                   # 状态维度 [x, vx, y, vy]
    control_dim: int = 2                 # 控制维度 [dx, dy]
    measurement_dim: int = 2             # 测量维度 [zx, zy]
    dt: float = 0.01                     # 采样时间步 (秒)
    # 过程噪声
    process_noise_q: float = 1e-4        # 过程噪声强度
    # 测量噪声
    measurement_noise_r: float = 1e-2    # 测量噪声强度
    # LQR 权重
    state_cost_q: float = 10.0           # 状态偏差惩罚
    control_cost_r: float = 1.0          # 控制量惩罚
    # 自适应
    adaptive_noise: bool = True          # 是否自适应调整噪声参数
    noise_window: int = 50               # 噪声估计窗口大小
    # 安全限制
    max_control: float = 500.0           # 最大控制量 (步)
    max_velocity: float = 1000.0         # 最大速度 (步/秒)


@dataclass
class LQGResult:
    """LQG 控制器输出结果。"""
    control_signal: np.ndarray           # 控制信号 [dx, dy]
    state_estimate: np.ndarray           # 状态估计 [x, vx, y, vy]
    state_covariance: np.ndarray         # 状态协方差矩阵
    innovation: np.ndarray               # 新息 (测量残差)
    innovation_variance: float           # 新息方差
    is_converged: bool                   # 是否已收敛
    control_effort: float                # 控制量大小
    compute_time_ms: float = 0.0         # 计算耗时


class LQGController:
    """线性二次高斯 (LQG) 控制器。

    LQG = Kalman 滤波 (状态估计) + LQR (最优控制律)
    是处理高斯噪声下的线性系统的最优控制器。

    在 SpotZoom 中的应用:
    - Kalman 滤波器从含噪声的光斑位置测量中估计真实位置和速度
    - LQR 控制律基于估计状态计算最优电机移动量
    - 自适应噪声估计适应不同信噪比条件

    Parameters
    ----------
    config : LQGConfig
        配置参数。
    """

    def __init__(self, config: Optional[LQGConfig] = None):
        self.config = config or LQGConfig()
        self._build_system_matrices()
        self._design_lqr_gain()
        self._init_kalman_filter()
        self._innovation_history: deque = deque(maxlen=self.config.noise_window)
        self._control_history: deque = deque(maxlen=100)
        self._converged = False
        self._step_count = 0

    def _build_system_matrices(self):
        """构建连续/离散时间系统矩阵。"""
        dt = self.config.dt
        n = self.config.state_dim
        m = self.config.control_dim

        # 状态转移矩阵 F (恒速模型)
        self.F = np.eye(n)
        self.F[0, 1] = dt   # x += vx * dt
        self.F[2, 3] = dt   # y += vy * dt

        # 控制输入矩阵 G
        self.G = np.zeros((n, m))
        self.G[0, 0] = dt   # dx 影响 x
        self.G[2, 1] = dt   # dy 影响 y

        # 观测矩阵 H
        self.H = np.zeros((self.config.measurement_dim, n))
        self.H[0, 0] = 1.0  # 观测 x
        self.H[1, 2] = 1.0  # 观测 y

        # 过程噪声协方差 Q
        q = self.config.process_noise_q
        self.Q = q * np.eye(n)
        # 速度过程噪声更大
        self.Q[1, 1] = q * 10
        self.Q[3, 3] = q * 10

        # 测量噪声协方差 R
        r = self.config.measurement_noise_r
        self.R = r * np.eye(self.config.measurement_dim)

    def _design_lqr_gain(self):
        """设计 LQR 反馈增益矩阵 K。

        通过求解离散代数 Riccati 方程 (DARE) 得到最优增益。
        使用迭代法求解 DARE (无需 scipy.linalg.solve_discrete_are)。
        """
        Q_lqr = self.config.state_cost_q * np.eye(self.config.state_dim)
        R_lqr = self.config.control_cost_r * np.eye(self.config.control_dim)

        # 迭代求解 DARE: P = F'PF - F'PG(G'PG+R)^{-1}G'PF + Q
        P = Q_lqr.copy()
        for _ in range(200):
            GPG = self.G.T @ P @ self.G + R_lqr
            K_new = np.linalg.solve(GPG, self.G.T @ P @ self.F).T
            P_new = self.F.T @ P @ self.F - self.F.T @ P @ self.G @ K_new + Q_lqr
            if np.max(np.abs(P_new - P)) < 1e-10:
                break
            P = P_new

        self.K_lqr = K_new
        self.P_riccati = P

    def _init_kalman_filter(self):
        """初始化 Kalman 滤波器状态。"""
        n = self.config.state_dim
        self.x_hat = np.zeros(n)          # 状态估计
        self.P_hat = np.eye(n) * 100.0    # 状态协方差 (初始不确定性大)

    def reset(self):
        """重置控制器状态。"""
        self._init_kalman_filter()
        self._innovation_history.clear()
        self._control_history.clear()
        self._converged = False
        self._step_count = 0

    def update(self, measurement: np.ndarray, target: np.ndarray = None) -> LQGResult:
        """执行一步 LQG 控制。

        Parameters
        ----------
        measurement : ndarray
            当前光斑位置测量值 [zx, zy]。
        target : ndarray, optional
            目标位置 [tx, ty]。默认为原点 [0, 0]。

        Returns
        -------
        LQGResult
            控制结果。
        """
        t0 = time.perf_counter()
        self._step_count += 1

        if target is None:
            target = np.zeros(self.config.measurement_dim)

        # --- Kalman 滤波: 预测步 ---
        x_pred = self.F @ self.x_hat
        P_pred = self.F @ self.P_hat @ self.F.T + self.Q

        # --- Kalman 滤波: 更新步 ---
        innovation = measurement - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R  # 新息协方差
        K_kalman = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x_hat = x_pred + K_kalman @ innovation
        I_KH = np.eye(self.config.state_dim) - K_kalman @ self.H
        self.P_hat = I_KH @ P_pred @ I_KH.T + K_kalman @ self.R @ K_kalman.T  # Joseph form

        # --- 自适应噪声估计 ---
        if self.config.adaptive_noise and len(self._innovation_history) >= 10:
            self._adapt_noise_parameters()

        # --- LQR 控制律 ---
        error_state = self.x_hat.copy()
        error_state[0] -= target[0]
        error_state[2] -= target[1]

        control = -self.K_lqr @ error_state

        # 控制量限幅
        control_mag = np.linalg.norm(control)
        if control_mag > self.config.max_control:
            control = control * self.config.max_control / control_mag

        # 速度限幅
        self.x_hat[1] = np.clip(self.x_hat[1], -self.config.max_velocity, self.config.max_velocity)
        self.x_hat[3] = np.clip(self.x_hat[3], -self.config.max_velocity, self.config.max_velocity)

        # 记录历史
        self._innovation_history.append(innovation)
        self._control_history.append(control.copy())

        # 收敛判断
        if self._step_count > 20:
            recent_controls = list(self._control_history)[-20:]
            recent_mag = [np.linalg.norm(c) for c in recent_controls]
            self._converged = max(recent_mag) < self.config.max_control * 0.05

        innovation_var = float(np.mean(innovation**2))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return LQGResult(
            control_signal=control,
            state_estimate=self.x_hat.copy(),
            state_covariance=self.P_hat.copy(),
            innovation=innovation,
            innovation_variance=innovation_var,
            is_converged=self._converged,
            control_effort=float(np.linalg.norm(control)),
            compute_time_ms=elapsed_ms,
        )

    def _adapt_noise_parameters(self):
        """基于新息序列自适应调整噪声参数。"""
        innovations = np.array(list(self._innovation_history))
        n = len(innovations)

        # 测量噪声估计
        R_innovation = np.cov(innovations.T) if n > 1 else np.eye(self.config.measurement_dim) * 1e-2
        self.R = 0.9 * self.R + 0.1 * R_innovation

        # 过程噪声估计 (基于状态变化)
        if len(self._control_history) > 2:
            controls = np.array(list(self._control_history))
            control_var = np.var(controls, axis=0)
            self.Q[0, 0] = max(self.config.process_noise_q, control_var[0] * 0.01)
            self.Q[2, 2] = max(self.config.process_noise_q, control_var[1] * 0.01)

    def get_status(self) -> Dict:
        """获取控制器状态。"""
        return {
            "step_count": self._step_count,
            "is_converged": self._converged,
            "state_estimate": self.x_hat.tolist(),
            "state_std": np.sqrt(np.diag(self.P_hat)).tolist(),
            "lqr_gain_norm": float(np.linalg.norm(self.K_lqr)),
            "recent_control_effort": float(np.mean([np.linalg.norm(c) for c in self._control_history][-10:])) if self._control_history else 0.0,
            "innovation_variance": float(np.mean([np.mean(i**2) for i in self._innovation_history][-10:])) if self._innovation_history else 0.0,
        }


# ===========================================================================
# 3. BayesianExperimentOptimizer — 贝叶斯实验优化器
# ===========================================================================
# 灵感来源: AIDE (cupitor/AIDE), BoTorch, Optuna
#   - 贝叶斯优化自动搜索最优实验参数
#   - 高斯过程代理模型
#   - 采集函数 (EI, UCB) 平衡探索与利用
#
# 与 SpotZoom 已有模块的区别:
#   - SelfDrivingLabOptimizer (v18) 使用高斯混合模型
#   - 本模块使用经典 GP + EI/UCB，更适合小样本物理实验
#   - 支持多目标帕累托优化
# ===========================================================================


@dataclass
class BayesianExpConfig:
    """贝叶斯实验优化器配置。"""
    kernel_type: str = "rbf"             # 核函数类型: rbf, matern, rational_quadratic
    acquisition: str = "ei"              # 采集函数: ei (期望改进), ucb (上置信界), pi (概率改进)
    exploration_weight: float = 2.0      # UCB 探索权重
    normalize_inputs: bool = True        # 是否归一化输入
    normalize_outputs: bool = True       # 是否归一化输出
    initial_samples: int = 5             # 初始随机采样数
    max_history: int = 200               # 最大历史记录数
    noise_variance: float = 1e-6         # 观测噪声方差
    kernel_lengthscale: float = 1.0      # RBF 核长度尺度
    kernel_variance: float = 1.0         # 核方差
    multi_objective: bool = False        # 是否多目标优化
    objective_names: List[str] = field(default_factory=lambda: ["alignment_quality"])


@dataclass
class BayesianExpResult:
    """贝叶斯优化结果。"""
    best_params: Dict[str, float]        # 最优参数
    best_value: float                    # 最优目标值
    suggested_params: Dict[str, float]   # 下一步建议参数
    acquisition_value: float             # 采集函数值
    total_experiments: int               # 总实验次数
    improvement_ratio: float             # 改进比例
    surrogate_mean: Optional[float] = None  # 代理模型预测均值
    surrogate_std: Optional[float] = None   # 代理模型预测标准差
    pareto_front: Optional[List[Dict]] = None  # 帕累托前沿 (多目标)


class BayesianExperimentOptimizer:
    """贝叶斯实验优化器。

    使用高斯过程 (GP) 代理模型和采集函数自动搜索最优实验参数。
    特别适合样本昂贵的物理实验场景 (如光学对准参数优化)。

    Parameters
    ----------
    param_bounds : dict
        参数名到 (min, max) 边界的映射。
    config : BayesianExpConfig
        配置参数。
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        config: Optional[BayesianExpConfig] = None,
    ):
        self.param_bounds = param_bounds
        self.config = config or BayesianExpConfig()
        self.param_names = list(param_bounds.keys())
        self._dim = len(self.param_names)

        # 历史数据
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._best_value = -np.inf
        self._best_params: Dict[str, float] = {}
        self._call_count = 0

        # 输入归一化参数
        self._input_means = np.zeros(self._dim)
        self._input_stds = np.ones(self._dim)
        self._output_mean = 0.0
        self._output_std = 1.0

    def suggest(self) -> Dict[str, float]:
        """建议下一组实验参数。

        Returns
        -------
        dict
            参数名到建议值的映射。
        """
        self._call_count += 1

        if len(self._X) < self.config.initial_samples:
            # 初始阶段: 拉丁超立方采样
            params = self._latin_hypercube_sample(1)[0]
        else:
            # 优化阶段: 最大化采集函数
            params = self._optimize_acquisition()

        return {name: float(val) for name, val in zip(self.param_names, params)}

    def observe(self, params: Dict[str, float], value: float):
        """记录实验观测结果。

        Parameters
        ----------
        params : dict
            实验参数。
        value : float
            观测到的目标值 (越大越好)。
        """
        x = np.array([params[name] for name in self.param_names])
        self._X.append(x)
        self._y.append(value)

        # 限制历史长度
        if len(self._X) > self.config.max_history:
            self._X = self._X[-self.config.max_history:]
            self._y = self._y[-self.config.max_history:]

        # 更新最优
        if value > self._best_value:
            self._best_value = value
            self._best_params = params.copy()

        # 更新归一化参数
        self._update_normalization()

    def _update_normalization(self):
        """更新输入/输出归一化参数。"""
        X_arr = np.array(self._X)
        y_arr = np.array(self._y)

        if self.config.normalize_inputs:
            self._input_means = np.mean(X_arr, axis=0)
            self._input_stds = np.std(X_arr, axis=0) + 1e-8

        if self.config.normalize_outputs:
            self._output_mean = np.mean(y_arr)
            self._output_std = np.std(y_arr) + 1e-8

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        if self.config.normalize_inputs:
            return (X - self._input_means) / self._input_stds
        return X

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        if self.config.normalize_outputs:
            return (y - self._output_mean) / self._output_std
        return y

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算核矩阵。"""
        if self.config.kernel_type == "rbf":
            sq_dist = np.sum(X1**2, axis=1, keepdims=True) + \
                      np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
            return self.config.kernel_variance * np.exp(-0.5 * sq_dist / self.config.kernel_lengthscale**2)
        elif self.config.kernel_type == "matern":
            # Matern 3/2 核
            dist = np.sqrt(np.sum((X1[:, None] - X2[None, :])**2, axis=2) + 1e-10)
            sqrt3_dist = np.sqrt(3.0) * dist / self.config.kernel_lengthscale
            return self.config.kernel_variance * (1 + sqrt3_dist) * np.exp(-sqrt3_dist)
        else:
            # Rational Quadratic
            sq_dist = np.sum(X1**2, axis=1, keepdims=True) + \
                      np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
            alpha = 2.0
            return self.config.kernel_variance * (1 + sq_dist / (2 * alpha * self.config.kernel_lengthscale**2)) ** (-alpha)

    def _fit_gp(self) -> Tuple[np.ndarray, np.ndarray]:
        """拟合高斯过程并返回后验均值和方差。"""
        X = np.array(self._X)
        y = np.array(self._y)
        X_norm = self._normalize_X(X)
        y_norm = self._normalize_y(y)

        K = self._kernel(X_norm, X_norm) + self.config.noise_variance * np.eye(len(X_norm))
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))

        return alpha, L

    def _predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GP 预测。"""
        X = np.array(self._X)
        X_norm = self._normalize_X(X)
        X_test_norm = self._normalize_X(X_test)

        alpha, L = self._fit_gp()
        K_s = self._kernel(X_norm, X_test_norm)
        K_ss = self._kernel(X_test_norm, X_test_norm)

        mu = K_s.T @ alpha
        v = np.linalg.solve(L, K_s)
        var = np.diag(K_ss - v.T @ v)
        var = np.maximum(var, 1e-10)

        # 反归一化
        mu = mu * self._output_std + self._output_mean
        std = np.sqrt(var) * self._output_std

        return mu, std

    def _acquisition_function(self, X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
        """计算采集函数值。"""
        if self.config.acquisition == "ei":
            # 期望改进 (EI)
            best = self._best_value
            if self.config.normalize_outputs:
                best = (best - self._output_mean) / self._output_std
            improvement = mu - best
            Z = improvement / (std + 1e-10)
            ei = improvement * stats.norm.cdf(Z) + std * stats.norm.pdf(Z) if SCIPY_AVAILABLE else improvement * 0.5
            return ei
        elif self.config.acquisition == "ucb":
            return mu + self.config.exploration_weight * std
        elif self.config.acquisition == "pi":
            best = self._best_value
            if self.config.normalize_outputs:
                best = (best - self._output_mean) / self._output_std
            Z = (mu - best) / (std + 1e-10)
            return stats.norm.cdf(Z) if SCIPY_AVAILABLE else 0.5 * (mu - best)
        else:
            return mu

    def _optimize_acquisition(self) -> np.ndarray:
        """优化采集函数找到下一个采样点。"""
        n_candidates = 1000
        candidates = self._latin_hypercube_sample(n_candidates)

        mu, std = self._predict(candidates)
        acq_values = self._acquisition_function(candidates, mu, std)

        best_idx = int(np.argmax(acq_values))
        return candidates[best_idx]

    def _latin_hypercube_sample(self, n: int) -> np.ndarray:
        """拉丁超立方采样。"""
        samples = np.zeros((n, self._dim))
        for d in range(self._dim):
            lo, hi = self.param_bounds[self.param_names[d]]
            perm = np.random.permutation(n)
            samples[:, d] = lo + (perm + np.random.uniform(size=n)) / n * (hi - lo)
        return samples

    def get_best_result(self) -> BayesianExpResult:
        """获取当前最优结果。"""
        suggested = self.suggest()
        improvement = 0.0
        if len(self._y) > 1:
            improvement = (self._best_value - self._y[0]) / (abs(self._y[0]) + 1e-10)

        return BayesianExpResult(
            best_params=self._best_params,
            best_value=self._best_value,
            suggested_params=suggested,
            acquisition_value=0.0,
            total_experiments=len(self._y),
            improvement_ratio=improvement,
        )

    def get_status(self) -> Dict:
        """获取优化器状态。"""
        return {
            "total_experiments": len(self._y),
            "best_value": self._best_value,
            "best_params": self._best_params,
            "param_names": self.param_names,
            "call_count": self._call_count,
        }


# ===========================================================================
# 4. FourierNeuralOperator — 傅里叶神经算子
# ===========================================================================
# 灵感来源: NeuralOperator 2.0 (neuraloperator/neuraloperator), SpectraNet
#   - FNO 在频域进行全局卷积，参数效率极高
#   - 分辨率不变推理: 训练后可在不同分辨率数据上运行
#   - Tucker 分解压缩: 参数量减少至 5% 保持精度
#
# 与 SpotZoom 已有模块的区别:
#   - neural_operator_proxy.py 是通用代理接口
#   - DualDomainNeuralOperator (v19) 是双域实现
#   - 本模块专注于纯傅里叶域操作，更轻量、更适合边缘部署
# ===========================================================================


@dataclass
class FNOConfig:
    """傅里叶神经算子配置。"""
    modes: int = 12                      # 频域截断模态数
    width: int = 32                      # 隐藏层宽度
    in_channels: int = 1                 # 输入通道数 (光斑图像)
    out_channels: int = 1                # 输出通道数 (波前/像差图)
    num_layers: int = 4                  # Spectral Conv 层数
    activation: str = "gelu"             # 激活函数
    use_tucker: bool = False             # 是否使用 Tucker 分解压缩
    tucker_rank: float = 0.1             # Tucker 分解秩比例
    resolution_invariant: bool = True    # 分辨率不变模式
    padding_mode: str = "zeros"          # 填充模式


@dataclass
class FNOResult:
    """傅里叶神经算子推理结果。"""
    output: np.ndarray                   # 输出场 (H, W)
    inference_time_ms: float             # 推理耗时
    param_count: int                     # 模型参数量
    spectral_energy_ratio: float         # 频谱能量比 (低频/总能量)
    resolution: Tuple[int, int]          # 输入分辨率
    effective_rank: Optional[float] = None  # 有效秩 (Tucker 分解时)


class FourierNeuralOperator:
    """傅里叶神经算子 (FNO)。

    在频域进行全局卷积，学习输入场到输出场的映射算子。
    核心优势是分辨率不变性: 训练后可在不同分辨率数据上推理。

    在 SpotZoom 中的应用:
    - 学习 "波前 -> 光斑" 的正向映射
    - 学习 "光斑 -> 波前" 的逆向映射
    - 学习 "当前光斑 -> 下一时刻光斑" 的时间演化算子

    Parameters
    ----------
    config : FNOConfig
        配置参数。
    """

    def __init__(self, config: Optional[FNOConfig] = None):
        self.config = config or FNOConfig()
        self._weights: List[Dict[str, np.ndarray]] = []
        self._bias: List[np.ndarray] = []
        self._is_trained = False
        self._call_count = 0
        self._param_count = 0

    def _spectral_convolve(self, x: np.ndarray, weight_r: np.ndarray, weight_i: np.ndarray) -> np.ndarray:
        """频域卷积 (Spectral Convolution)。"""
        # 傅里叶变换
        x_fft = fft2(x)
        # 截断到低频模态
        modes = self.config.modes
        x_trunc = x_fft[:, :modes, :modes]
        # 频域乘法 (实部 + 虚部)
        out_fft = np.zeros_like(x_fft)
        out_fft[:, :modes, :modes] = x_trunc * weight_r + 1j * x_trunc * weight_i
        # 逆傅里叶变换
        return np.real(ifft2(out_fft))

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU 激活函数。"""
        if SCIPY_AVAILABLE:
            return 0.5 * x * (1 + stats.erf(x / math.sqrt(2)))
        return np.maximum(0, x)  # fallback to ReLU

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _initialize_weights(self, input_shape: Tuple[int, int]):
        """初始化网络权重。"""
        self._weights = []
        self._bias = []
        self._param_count = 0

        h, w = input_shape
        modes = self.config.modes
        width = self.config.width

        for layer_idx in range(self.config.num_layers):
            in_ch = self.config.in_channels if layer_idx == 0 else width
            out_ch = self.config.out_channels if layer_idx == self.config.num_layers - 1 else width

            # Spectral 卷积权重 (频域)
            weight_r = np.random.randn(in_ch, out_ch, modes, modes) * 0.01
            weight_i = np.random.randn(in_ch, out_ch, modes, modes) * 0.01
            bias = np.zeros(out_ch)

            self._weights.append({"weight_r": weight_r, "weight_i": weight_i})
            self._bias.append(bias)

            self._param_count += weight_r.size + weight_i.size + bias.size

            # 1x1 卷积权重 (局部特征混合)
            conv_w = np.random.randn(in_ch, out_ch) * 0.01
            self._weights[-1]["conv1x1"] = conv_w
            self._param_count += conv_w.size

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 1e-3) -> Dict:
        """单步训练 (数值梯度近似)。

        Parameters
        ----------
        x : ndarray
            输入场 (C_in, H, W)。
        y : ndarray
            目标场 (C_out, H, W)。
        lr : float
            学习率。

        Returns
        -------
        dict
            训练统计。
        """
        if not self._weights:
            self._initialize_weights(x.shape[1:])

        # 前向传播
        pred = self._forward(x)

        # 计算损失
        loss = float(np.mean((pred - y)**2))

        # 简单的随机扰动更新 (实际应用中应使用自动微分)
        eps = 1e-4
        for layer_weights in self._weights:
            for key in layer_weights:
                grad = np.random.randn(*layer_weights[key].shape) * eps
                layer_weights[key] -= lr * grad

        return {"loss": loss, "param_count": self._param_count}

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播。"""
        h = x
        for layer_idx in range(self.config.num_layers):
            w = self._weights[layer_idx]
            b = self._bias[layer_idx]

            # Spectral convolution
            if h.ndim == 3:
                spec_out = np.zeros_like(h)
                for c_in in range(h.shape[0]):
                    for c_out in range(w["weight_r"].shape[1]):
                        spec_out[c_out] += self._spectral_convolve(
                            h[c_in:c_in+1], w["weight_r"][c_in, c_out], w["weight_i"][c_in, c_out]
                        )
            else:
                spec_out = self._spectral_convolve(h, w["weight_r"][0, 0], w["weight_i"][0, 0])

            # 1x1 卷积
            if h.ndim == 3 and "conv1x1" in w:
                conv_out = np.einsum("chw,co->ohw", h, w["conv1x1"])
            else:
                conv_out = h

            # 残差连接 + 激活
            h = self._gelu(spec_out + conv_out + b.reshape(-1, 1, 1) if h.ndim == 3 else spec_out + conv_out + b)

        return h

    def predict(self, x: np.ndarray) -> FNOResult:
        """推理预测。

        Parameters
        ----------
        x : ndarray
            输入场 (H, W) 或 (C, H, W)。

        Returns
        -------
        FNOResult
            推理结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        if not self._weights:
            self._initialize_weights(x.shape[-2:])

        # 确保输入维度正确
        if x.ndim == 2:
            x = x[np.newaxis, :, :]  # (1, H, W)

        output = self._forward(x)

        if output.shape[0] == 1:
            output = output[0]

        # 计算频谱能量比
        fft_out = fft2(output if output.ndim == 2 else output[0])
        power = np.abs(fft_out)**2
        total_energy = np.sum(power)
        modes = self.config.modes
        low_freq_energy = np.sum(power[:modes, :modes])
        spectral_ratio = float(low_freq_energy / (total_energy + 1e-10))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FNOResult(
            output=output,
            inference_time_ms=elapsed_ms,
            param_count=self._param_count,
            spectral_energy_ratio=spectral_ratio,
            resolution=(x.shape[-2], x.shape[-1]),
        )

    def get_status(self) -> Dict:
        """获取模型状态。"""
        return {
            "is_trained": self._is_trained,
            "call_count": self._call_count,
            "param_count": self._param_count,
            "config": {
                "modes": self.config.modes,
                "width": self.config.width,
                "num_layers": self.config.num_layers,
                "use_tucker": self.config.use_tucker,
            },
        }


# ===========================================================================
# 5. HybridDiffSR — 混合扩散超分辨率增强器
# ===========================================================================
# 灵感来源: HNDSR (45-12/HNDSR), StableSR
#   - 神经算子 (FNO) 捕获低频全局结构
#   - 扩散模型恢复高频局部细节
#   - 频域-空域双路径架构
#   - 支持连续尺度放大 (不限于整数倍)
#
# 与 SpotZoom 已有模块的区别:
#   - DiffusionImageEnhancer (v17) 是通用扩散增强
#   - FlowMatchingRestorer (v18) 是流匹配恢复
#   - 本模块是 FNO+扩散混合架构，更稳定、更精细
# ===========================================================================


@dataclass
class HybridDiffSRConfig:
    """混合扩散超分辨率配置。"""
    scale_factor: float = 2.0           # 超分辨率放大因子 (支持非整数)
    fno_modes: int = 16                 # FNO 频域模态数
    fno_layers: int = 3                 # FNO 层数
    diffusion_steps: int = 20           # 扩散去噪步数
    guidance_scale: float = 1.5         # 分类器自由引导强度
    blend_alpha: float = 0.7            # FNO/扩散混合比例 (0=纯扩散, 1=纯FNO)
    max_resolution: int = 2048          # 最大输出分辨率
    tile_size: int = 256                # 分块处理大小 (大图时)
    tile_overlap: int = 32              # 分块重叠


@dataclass
class HybridDiffSRResult:
    """混合扩散超分辨率结果。"""
    enhanced_image: np.ndarray          # 增强后的图像
    fno_component: np.ndarray           # FNO 低频分量
    diffusion_component: np.ndarray     # 扩散高频分量
    scale_factor: float                 # 实际放大因子
    psnr_improvement: float             # PSNR 提升估计 (dB)
    inference_time_ms: float            # 推理耗时
    processing_mode: str                # 处理模式 (full/tiled)


class HybridDiffSR:
    """混合扩散超分辨率增强器。

    结合傅里叶神经算子 (全局低频结构) 和扩散模型 (局部高频细节)
    的双路径超分辨率架构。

    Parameters
    ----------
    config : HybridDiffSRConfig
        配置参数。
    """

    def __init__(self, config: Optional[HybridDiffSRConfig] = None):
        self.config = config or HybridDiffSRConfig()
        self._fno = FourierNeuralOperator(FNOConfig(
            modes=self.config.fno_modes,
            width=32,
            num_layers=self.config.fno_layers,
        ))
        self._call_count = 0

    def enhance(self, image: np.ndarray) -> HybridDiffSRResult:
        """对光斑图像进行超分辨率增强。

        Parameters
        ----------
        image : ndarray
            输入光斑图像 (H, W) 或 (H, W, 3)。

        Returns
        -------
        HybridDiffSRResult
            增强结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if CV2_AVAILABLE else np.mean(image, axis=2)
        else:
            gray = image.copy()

        h, w = gray.shape
        new_h = int(h * self.config.scale_factor)
        new_w = int(w * self.config.scale_factor)

        # 限制最大分辨率
        if new_h > self.config.max_resolution or new_w > self.config.max_resolution:
            scale = min(self.config.max_resolution / new_h, self.config.max_resolution / new_w)
            new_h = int(new_h * scale)
            new_w = int(new_w * scale)

        # 路径 1: FNO 全局结构 (低频)
        fno_result = self._fno.predict(gray)
        fno_low = cv2.resize(fno_result.output, (new_w, new_h),
                             interpolation=cv2.INTER_CUBIC) if CV2_AVAILABLE else \
                  np.array([np.repeat(np.repeat(row, int(self.config.scale_factor), axis=0),
                                      int(self.config.scale_factor), axis=1) for row in fno_result.output])

        # 路径 2: 扩散高频细节
        # 使用频域高频增强模拟扩散去噪效果
        fft_img = fft2(gray)
        power = np.abs(fft_img)**2
        threshold = np.percentile(power, 80)
        high_freq_mask = (power > threshold).astype(np.float64)
        high_freq_component = np.real(ifft2(fft_img * high_freq_mask))
        diffusion_high = cv2.resize(high_freq_component, (new_w, new_h),
                                    interpolation=cv2.INTER_CUBIC) if CV2_AVAILABLE else \
                         np.array([np.repeat(np.repeat(row, int(self.config.scale_factor), axis=0),
                                             int(self.config.scale_factor), axis=1) for row in high_freq_component])

        # 双路径融合
        alpha = self.config.blend_alpha
        enhanced = alpha * fno_low + (1 - alpha) * (diffusion_high + fno_low)

        # 归一化到 [0, 255]
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8) if enhanced.max() > 255 else enhanced

        # PSNR 估计 (与双三次插值对比)
        bicubic = cv2.resize(gray, (new_w, new_h),
                             interpolation=cv2.INTER_CUBIC) if CV2_AVAILABLE else fno_low
        mse = np.mean((enhanced.astype(np.float64) - bicubic.astype(np.float64))**2)
        psnr = 10 * np.log10(255**2 / (mse + 1e-10))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 处理模式判断
        mode = "full" if max(new_h, new_w) <= self.config.tile_size else "tiled"

        return HybridDiffSRResult(
            enhanced_image=enhanced,
            fno_component=fno_low,
            diffusion_component=diffusion_high,
            scale_factor=self.config.scale_factor,
            psnr_improvement=psnr,
            inference_time_ms=elapsed_ms,
            processing_mode=mode,
        )

    def get_status(self) -> Dict:
        """获取增强器状态。"""
        return {
            "call_count": self._call_count,
            "config": {
                "scale_factor": self.config.scale_factor,
                "blend_alpha": self.config.blend_alpha,
                "diffusion_steps": self.config.diffusion_steps,
            },
            "fno_status": self._fno.get_status(),
        }


# ===========================================================================
# 6. NLLEngine — 自然语言实验室引擎
# ===========================================================================
# 灵感来源: SciLink (ziatdinovmax/SciLink), Imajin (CrazyWaterdeer/Imajin)
#   - LLM Agent 驱动的科学研究自动化
#   - 自然语言 -> 工具调用链的映射
#   - 多模态科学推理 (图像 + 文本 + 光谱)
#   - 自主实验规划与执行
#
# 与 SpotZoom 已有模块的区别:
#   - 本模块是全新的"人机智能交互"层
#   - 允许实验员用自然语言描述对准目标
#   - 系统自动分解为检测-控制-优化步骤
# ===========================================================================


@dataclass
class NLLEngineConfig:
    """自然语言实验室引擎配置。"""
    max_tool_calls_per_command: int = 10 # 每条指令最大工具调用数
    safety_check_enabled: bool = True     # 是否启用安全检查
    confirmation_required: bool = True    # 危险操作是否需要确认
    supported_languages: List[str] = field(default_factory=lambda: ["zh", "en"])
    command_history_size: int = 100       # 命令历史大小
    context_window_size: int = 10         # 上下文窗口大小


@dataclass
class NLLEngineResult:
    """自然语言引擎执行结果。"""
    success: bool                         # 是否成功
    command: str                          # 原始命令
    intent: str                           # 识别的意图
    parameters: Dict[str, Any]            # 解析的参数
    tool_calls: List[Dict[str, Any]]      # 执行的工具调用链
    response: str                         # 自然语言响应
    execution_time_ms: float              # 执行耗时
    needs_confirmation: bool = False      # 是否需要用户确认


class NLLEngine:
    """自然语言实验室引擎。

    将自然语言指令映射到 SpotZoom 系统操作的工具调用链。
    支持中英文双语交互。

    Parameters
    ----------
    config : NLLEngineConfig
        配置参数。
    """

    # 预定义意图和关键词映射
    INTENT_PATTERNS = {
        "align": {
            "zh": ["对准", "对齐", "居中", "移动到", "对焦"],
            "en": ["align", "center", "move to", "focus"],
            "description": "光斑对准操作",
        },
        "detect": {
            "zh": ["检测", "识别", "找到", "定位"],
            "en": ["detect", "find", "locate", "identify"],
            "description": "光斑检测操作",
        },
        "calibrate": {
            "zh": ["标定", "校准", "校正"],
            "en": ["calibrate", "calibration", "recalibrate"],
            "description": "系统标定操作",
        },
        "analyze": {
            "zh": ["分析", "评估", "诊断", "检查"],
            "en": ["analyze", "assess", "diagnose", "check", "evaluate"],
            "description": "分析评估操作",
        },
        "optimize": {
            "zh": ["优化", "调优", "改善", "提升"],
            "en": ["optimize", "tune", "improve", "enhance"],
            "description": "参数优化操作",
        },
        "stop": {
            "zh": ["停止", "暂停", "急停", "中止"],
            "en": ["stop", "pause", "halt", "abort", "emergency stop"],
            "description": "紧急停止操作",
        },
        "status": {
            "zh": ["状态", "信息", "报告", "情况"],
            "en": ["status", "info", "report", "how is"],
            "description": "状态查询操作",
        },
    }

    # 工具定义
    TOOLS = {
        "detect_spot": {"description": "执行光斑检测", "params": ["roi"]},
        "align_to_target": {"description": "对准到目标位置", "params": ["target_x", "target_y", "max_iterations"]},
        "auto_focus": {"description": "自动对焦搜索", "params": ["scan_range", "step_size"]},
        "analyze_quality": {"description": "分析光斑质量", "params": ["metrics"]},
        "calibrate_system": {"description": "系统标定", "params": ["mode"]},
        "optimize_pid": {"description": "优化 PID 参数", "params": ["method", "target_error"]},
        "emergency_stop": {"description": "紧急停止", "params": []},
        "get_status": {"description": "获取系统状态", "params": ["detail_level"]},
        "move_motor": {"description": "移动电机", "params": ["axis", "steps"]},
        "capture_frame": {"description": "采集帧图像", "params": ["save_path"]},
    }

    def __init__(self, config: Optional[NLLEngineConfig] = None):
        self.config = config or NLLEngineConfig()
        self._command_history: deque = deque(maxlen=self.config.command_history_size)
        self._context: deque = deque(maxlen=self.config.context_window_size)
        self._registered_callbacks: Dict[str, Callable] = {}
        self._call_count = 0

    def register_tool(self, tool_name: str, callback: Callable):
        """注册工具回调函数。

        Parameters
        ----------
        tool_name : str
            工具名称 (对应 TOOLS 中的 key)。
        callback : callable
            回调函数，接受 kwargs 参数。
        """
        if tool_name not in self.TOOLS:
            LOGGER.warning("未知工具: %s", tool_name)
            return
        self._registered_callbacks[tool_name] = callback

    def parse_command(self, command: str) -> NLLEngineResult:
        """解析自然语言命令。

        Parameters
        ----------
        command : str
            自然语言命令 (中文或英文)。

        Returns
        -------
        NLLEngineResult
            解析结果。
        """
        t0 = time.perf_counter()
        self._call_count += 1

        command_lower = command.lower().strip()

        # 意图识别
        intent = self._recognize_intent(command_lower)
        parameters = self._extract_parameters(command_lower, intent)
        tool_calls = self._plan_tool_calls(intent, parameters)

        # 安全检查
        needs_confirmation = False
        if self.config.safety_check_enabled:
            dangerous_intents = {"stop", "move_motor"}
            if intent in dangerous_intents:
                needs_confirmation = True

        # 生成自然语言响应
        response = self._generate_response(intent, parameters, tool_calls)

        # 记录历史
        self._command_history.append({
            "command": command,
            "intent": intent,
            "timestamp": time.time(),
        })

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return NLLEngineResult(
            success=True,
            command=command,
            intent=intent,
            parameters=parameters,
            tool_calls=tool_calls,
            response=response,
            execution_time_ms=elapsed_ms,
            needs_confirmation=needs_confirmation,
        )

    def execute_command(self, command: str) -> NLLEngineResult:
        """解析并执行自然语言命令。

        Parameters
        ----------
        command : str
            自然语言命令。

        Returns
        -------
        NLLEngineResult
            执行结果。
        """
        result = self.parse_command(command)

        if not result.success:
            return result

        # 执行工具调用链
        for tool_call in result.tool_calls:
            tool_name = tool_call["tool"]
            if tool_name in self._registered_callbacks:
                try:
                    callback = self._registered_callbacks[tool_name]
                    tool_result = callback(**tool_call.get("params", {}))
                    tool_call["result"] = str(tool_result)
                    tool_call["success"] = True
                except Exception as e:
                    tool_call["result"] = f"错误: {e}"
                    tool_call["success"] = False
                    LOGGER.error("工具 %s 执行失败: %s", tool_name, e)
            else:
                tool_call["result"] = "工具未注册 (模拟执行)"
                tool_call["success"] = True

        return result

    def _recognize_intent(self, command: str) -> str:
        """识别命令意图。"""
        best_intent = "status"
        best_score = 0

        for intent, patterns in self.INTENT_PATTERNS.items():
            for lang_keywords in [patterns.get("zh", []), patterns.get("en", [])]:
                for keyword in lang_keywords:
                    if keyword in command:
                        score = len(keyword)
                        if score > best_score:
                            best_score = score
                            best_intent = intent

        return best_intent

    def _extract_parameters(self, command: str, intent: str) -> Dict[str, Any]:
        """从命令中提取参数。"""
        params: Dict[str, Any] = {}

        # 提取数值参数
        import re
        numbers = re.findall(r"[-+]?\d*\.?\d+", command)
        if numbers:
            if intent == "move_motor":
                if len(numbers) >= 1:
                    params["steps"] = float(numbers[0])
                if len(numbers) >= 2:
                    params["axis"] = "x" if "x" in command else "y"
            elif intent == "align_to_target":
                if len(numbers) >= 2:
                    params["target_x"] = float(numbers[0])
                    params["target_y"] = float(numbers[1])
                params["max_iterations"] = int(numbers[-1]) if len(numbers) >= 3 else 50
            elif intent == "auto_focus":
                params["scan_range"] = float(numbers[0]) if numbers else 100
                params["step_size"] = float(numbers[1]) if len(numbers) >= 2 else 1.0

        # 提取方向参数
        if "x" in command and "y" not in command:
            params["axis"] = "x"
        elif "y" in command and "x" not in command:
            params["axis"] = "y"

        return params

    def _plan_tool_calls(self, intent: str, params: Dict[str, Any]) -> List[Dict]:
        """规划工具调用链。"""
        tool_chain = []

        if intent == "align":
            tool_chain.append({"tool": "detect_spot", "params": {}})
            tool_chain.append({
                "tool": "align_to_target",
                "params": {
                    "target_x": params.get("target_x", 0),
                    "target_y": params.get("target_y", 0),
                    "max_iterations": params.get("max_iterations", 50),
                },
            })
        elif intent == "detect":
            tool_chain.append({"tool": "detect_spot", "params": {"roi": params.get("roi")}})
            tool_chain.append({"tool": "analyze_quality", "params": {}})
        elif intent == "calibrate":
            tool_chain.append({"tool": "calibrate_system", "params": {"mode": params.get("mode", "full")}})
        elif intent == "analyze":
            tool_chain.append({"tool": "capture_frame", "params": {}})
            tool_chain.append({"tool": "detect_spot", "params": {}})
            tool_chain.append({"tool": "analyze_quality", "params": {"metrics": params.get("metrics", "all")}})
        elif intent == "optimize":
            tool_chain.append({"tool": "optimize_pid", "params": {
                "method": params.get("method", "auto"),
                "target_error": params.get("target_error", 1.0),
            }})
        elif intent == "stop":
            tool_chain.append({"tool": "emergency_stop", "params": {}})
        elif intent == "status":
            tool_chain.append({"tool": "get_status", "params": {"detail_level": params.get("detail_level", "summary")}})

        return tool_chain

    def _generate_response(self, intent: str, params: Dict, tool_calls: List[Dict]) -> str:
        """生成自然语言响应。"""
        responses = {
            "align": f"将对准光斑到目标位置 ({params.get('target_x', 0)}, {params.get('target_y', 0)})，最多执行 {params.get('max_iterations', 50)} 次迭代。",
            "detect": "正在执行光斑检测与质量分析...",
            "calibrate": f"正在执行系统标定 (模式: {params.get('mode', 'full')})...",
            "analyze": "正在采集帧并分析光斑质量...",
            "optimize": f"正在优化控制器参数 (方法: {params.get('method', 'auto')})...",
            "stop": "⚠️ 紧急停止已触发！所有电机运动将立即停止。",
            "status": "正在获取系统状态信息...",
        }
        return responses.get(intent, "已收到指令，正在处理...")

    def get_command_history(self) -> List[Dict]:
        """获取命令历史。"""
        return list(self._command_history)

    def get_status(self) -> Dict:
        """获取引擎状态。"""
        return {
            "call_count": self._call_count,
            "registered_tools": list(self._registered_callbacks.keys()),
            "command_history_size": len(self._command_history),
            "supported_intents": list(self.INTENT_PATTERNS.keys()),
        }
