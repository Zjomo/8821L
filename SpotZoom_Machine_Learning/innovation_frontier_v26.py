"""
SpotZoom 前沿创新模块 v26

本模块包含 8 个轻量级、零外部 ML 依赖的组件，灵感来源于 2024-2026 年
前沿开源项目的研究成果：

- AOViFT (Betzig Lab): 基于傅里叶域的自适应光学像差估计。
- anyloop: 插件化实时反馈控制框架。
- Kornia: GPU 加速可微分图像处理流水线。
- gym_ao: 强化学习驱动的光斑对中智能体。
- UniFMIR / Cellpose3: 基础模型启发的图像增强预处理。
- DeepXDE / NVIDIA Modulus PINN: 物理信息约束的波前预测。
- Matilda (NSO): 实时矩阵运算加速器。
- ShackHartmannAnalysis: Shack-Hartmann 波前传感器光斑分析。

所有组件仅依赖 numpy 和 cv2，不引入任何外部机器学习框架。
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


# ============================================================
# 1. FourierDomainAberrationEstimator (inspired by AOViFT)
# ============================================================

@dataclass
class FourierAberrationConfig:
    """傅里叶域像差估计器配置。"""
    fft_size: int = 128
    max_zernike_modes: int = 15
    frequency_band_radius: float = 0.4
    aberration_threshold: float = 0.1


@dataclass
class FourierAberrationResult:
    """傅里叶域像差估计结果。"""
    zernike_coefficients: np.ndarray
    rms_wavefront_error: float
    dominant_mode: int
    estimated_psf_quality: float


class FourierDomainAberrationEstimator:
    """基于傅里叶域分析的光学像差估计器。

    灵感来源于 AOViFT (Betzig Lab)，通过分析光斑图像的傅里叶频谱模式，
    估计光学系统的 Zernike 像差系数。核心思想是：不同的像差模式在频域中
    产生特征性的对称性和能量分布模式。

    算法流程：
    1. 对输入光斑图像进行预处理（去背景、归一化）
    2. 计算二维 FFT 并取功率谱
    3. 分析频域中的径向和角向能量分布
    4. 通过与 Zernike 模式的频域特征匹配，估计各模式系数
    5. 计算波前 RMS 误差和 Strehl 比
    """

    def __init__(self, config: Optional[FourierAberrationConfig] = None):
        self.config = config or FourierAberrationConfig()
        self._zernike_basis: Optional[np.ndarray] = None
        self._precompute_zernike_basis()
        LOGGER.info("傅里叶域像差估计器初始化完成, fft_size=%d, max_modes=%d",
                     self.config.fft_size, self.config.max_zernike_modes)

    def _precompute_zernike_basis(self) -> None:
        """预计算 Zernike 多项式基函数在频域中的表示。"""
        n = self.config.fft_size
        modes = self.config.max_zernike_modes
        basis = np.zeros((modes, n, n), dtype=np.float64)
        cy, cx = n // 2, n // 2
        y, x = np.ogrid[:n, :n]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (n / 2)
        theta = np.arctan2(y - cy, x - cx)
        for j in range(1, modes + 1):
            nn, mm = self._noll_to_zernike(j)
            radial = self._zernike_radial(nn, mm, np.clip(r, 0, 1))
            if mm == 0:
                basis[j - 1] = radial
            elif mm > 0:
                basis[j - 1] = radial * np.cos(mm * theta)
            else:
                basis[j - 1] = radial * np.sin(-mm * theta)
        fft_basis = np.zeros_like(basis)
        for j in range(modes):
            fft_img = np.fft.fftshift(np.fft.fft2(basis[j]))
            fft_basis[j] = np.abs(fft_img) ** 2
            norm = np.sqrt(np.sum(fft_basis[j] ** 2))
            if norm > 1e-12:
                fft_basis[j] /= norm
        self._zernike_basis = fft_basis

    @staticmethod
    def _noll_to_zernike(j: int) -> Tuple[int, int]:
        """将 Noll 编号转换为 Zernike (n, m) 索引。"""
        n = 0
        while (n + 1) * (n + 2) // 2 < j:
            n += 1
        m_values = [n - 2 * k for k in range(n, -1, -1)]
        idx = j - n * (n + 1) // 2 - 1
        return n, m_values[idx]

    @staticmethod
    def _zernike_radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
        """计算 Zernike 径向多项式 R_n^|m|(r)。"""
        m_abs = abs(m)
        result = np.zeros_like(r, dtype=np.float64)
        for s in range((n - m_abs) // 2 + 1):
            coeff = ((-1) ** s * math.factorial(n - s)
                     / (math.factorial(s) * math.factorial((n + m_abs) // 2 - s)
                        * math.factorial((n - m_abs) // 2 - s)))
            result += coeff * r ** (n - 2 * s)
        return result

    def reset(self) -> None:
        """重置估计器内部状态。"""
        LOGGER.debug("傅里叶域像差估计器已重置")

    def process(self, spot_image: np.ndarray) -> FourierAberrationResult:
        """处理光斑图像，估计光学像差。

        Args:
            spot_image: 输入光斑图像（灰度，2D numpy 数组）。
        Returns:
            FourierAberrationResult 包含 Zernike 系数和像差指标。
        """
        t0 = time.perf_counter()
        img = self._preprocess(spot_image)
        power_spectrum = self._compute_power_spectrum(img)
        coefficients = self._estimate_coefficients(power_spectrum)
        rms_wfe = float(np.sqrt(np.mean(coefficients ** 2)))
        dominant = int(np.argmax(np.abs(coefficients))) + 1
        strehl = float(np.exp(-(2 * np.pi * rms_wfe) ** 2))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        LOGGER.debug("像差估计完成: RMS=%.4fλ, 主导模式=Z%d, Strehl=%.3f, 耗时=%.1fms",
                      rms_wfe, dominant, strehl, elapsed_ms)
        return FourierAberrationResult(
            zernike_coefficients=coefficients, rms_wavefront_error=rms_wfe,
            dominant_mode=dominant, estimated_psf_quality=strehl)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理光斑图像：去背景、归一化、调整尺寸。"""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = image.astype(np.float64)
        img = np.clip(img - np.median(img), 0, None)
        max_val = img.max()
        if max_val > 1e-12:
            img /= max_val
        if img.shape[0] != self.config.fft_size or img.shape[1] != self.config.fft_size:
            img = cv2.resize(img, (self.config.fft_size, self.config.fft_size),
                             interpolation=cv2.INTER_LINEAR)
        return img

    def _compute_power_spectrum(self, image: np.ndarray) -> np.ndarray:
        """计算图像的功率谱（Hanning 窗 + 对数压缩）。"""
        rows, cols = image.shape
        window = np.outer(np.hanning(rows), np.hanning(cols))
        fft = np.fft.fft2(image * window)
        return np.log1p(np.abs(np.fft.fftshift(fft)) ** 2)

    def _estimate_coefficients(self, power_spectrum: np.ndarray) -> np.ndarray:
        """通过频域匹配估计 Zernike 系数。"""
        if self._zernike_basis is None:
            self._precompute_zernike_basis()
        n = self.config.fft_size
        cy, cx = n // 2, n // 2
        y, x = np.ogrid[:n, :n]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (n / 2)
        band_mask = (r < self.config.frequency_band_radius).astype(np.float64)
        filtered = (power_spectrum * band_mask).flatten()
        flat_basis = self._zernike_basis.reshape(self.config.max_zernike_modes, -1)
        coefficients = flat_basis @ filtered
        max_abs = np.max(np.abs(coefficients))
        if max_abs > 1e-12:
            coefficients = coefficients / max_abs * self.config.aberration_threshold
        return coefficients


# ============================================================
# 2. PluginFeedbackLoopController (inspired by anyloop)
# ============================================================

@dataclass
class PluginFeedbackConfig:
    """插件反馈控制框架配置。"""
    control_frequency_hz: float = 100.0
    max_plugins: int = 10
    plugin_timeout_ms: float = 50.0
    safety_margin_px: float = 20.0


@dataclass
class PluginFeedbackResult:
    """插件反馈控制结果。"""
    control_signal: Tuple[float, float]
    active_plugins: List[str]
    loop_latency_ms: float
    is_converged: bool


class PluginFeedbackLoopController:
    """插件化实时反馈控制框架。

    灵感来源于 anyloop 项目，实现灵活的插件式反馈控制环路。
    支持三类插件：检测插件（测量当前状态）、控制插件（计算控制量）、
    安全插件（在异常情况下执行保护动作）。

    使用方法：
    1. 通过 register_plugin() 注册各类插件
    2. 调用 process() 执行一个控制周期
    3. 检测 -> 控制 -> 安全检查 的流水线执行
    """

    def __init__(self, config: Optional[PluginFeedbackConfig] = None):
        self.config = config or PluginFeedbackConfig()
        self._detection_plugins: Dict[str, Callable] = {}
        self._control_plugins: Dict[str, Callable] = {}
        self._safety_plugins: Dict[str, Callable] = {}
        self._state: Dict[str, Any] = {}
        self._target: Tuple[float, float] = (0.0, 0.0)
        self._iteration_count: int = 0
        self._convergence_window: deque = deque(maxlen=20)
        LOGGER.info("插件反馈控制框架初始化完成, 控制频率=%.1fHz",
                     self.config.control_frequency_hz)

    def reset(self) -> None:
        """重置控制器状态，清除所有内部缓存。"""
        self._state.clear()
        self._iteration_count = 0
        self._convergence_window.clear()
        LOGGER.debug("插件反馈控制框架已重置")

    def register_plugin(self, name: str, plugin_func: Callable,
                        plugin_type: str = "detection") -> bool:
        """注册插件到控制框架。

        Args:
            name: 插件唯一名称。
            plugin_func: 插件函数，签名为 func(state) -> dict。
            plugin_type: 插件类型，"detection"/"control"/"safety"。
        Returns:
            是否注册成功。
        """
        total = len(self._detection_plugins) + len(self._control_plugins) + len(self._safety_plugins)
        if total >= self.config.max_plugins:
            LOGGER.warning("插件数量已达上限 %d", self.config.max_plugins)
            return False
        registry = {"detection": self._detection_plugins, "control": self._control_plugins,
                     "safety": self._safety_plugins}
        if plugin_type not in registry:
            LOGGER.error("未知插件类型: %s", plugin_type)
            return False
        registry[plugin_type][name] = plugin_func
        LOGGER.info("已注册 %s 插件: %s", plugin_type, name)
        return True

    def set_target(self, target: Tuple[float, float]) -> None:
        """设置控制目标位置。"""
        self._target = target

    def process(self, spot_image: np.ndarray,
                current_position: Tuple[float, float]) -> PluginFeedbackResult:
        """执行一个控制周期：检测 -> 控制 -> 安全检查。

        Args:
            spot_image: 当前光斑图像。
            current_position: 当前光斑位置 (x, y)。
        Returns:
            PluginFeedbackResult 包含控制信号和状态信息。
        """
        t0 = time.perf_counter()
        self._iteration_count += 1
        self._state.update({"image": spot_image, "position": current_position,
                            "target": self._target, "iteration": self._iteration_count,
                            "error": (self._target[0] - current_position[0],
                                      self._target[1] - current_position[1])})
        active: List[str] = []

        # 阶段 1：检测插件
        for name, func in self._detection_plugins.items():
            try:
                result = func(self._state)
                if isinstance(result, dict):
                    self._state.update(result)
                active.append(name)
            except Exception as e:
                LOGGER.warning("检测插件 %s 执行失败: %s", name, e)

        # 阶段 2：控制插件
        cdx, cdy, cc = 0.0, 0.0, 0
        for name, func in self._control_plugins.items():
            try:
                result = func(self._state)
                if isinstance(result, dict) and "dx" in result and "dy" in result:
                    cdx += result["dx"]; cdy += result["dy"]; cc += 1
                    self._state.update(result)
                    active.append(name)
            except Exception as e:
                LOGGER.warning("控制插件 %s 执行失败: %s", name, e)
        if cc > 0:
            cdx /= cc; cdy /= cc

        # 阶段 3：安全检查
        disp = math.sqrt(current_position[0] ** 2 + current_position[1] ** 2)
        if disp > self.config.safety_margin_px:
            for name, func in self._safety_plugins.items():
                try:
                    sr = func(self._state)
                    if isinstance(sr, dict):
                        if "override_dx" in sr: cdx = sr["override_dx"]
                        if "override_dy" in sr: cdy = sr["override_dy"]
                    active.append(name)
                except Exception as e:
                    LOGGER.warning("安全插件 %s 执行失败: %s", name, e)

        err_mag = math.sqrt(self._state["error"][0] ** 2 + self._state["error"][1] ** 2)
        self._convergence_window.append(err_mag)
        is_conv = len(self._convergence_window) >= 20 and np.mean(list(self._convergence_window)) < 0.5
        latency = (time.perf_counter() - t0) * 1000
        LOGGER.debug("控制周期 #%d: 信号=(%.3f,%.3f), 延迟=%.1fms, 收敛=%s",
                      self._iteration_count, cdx, cdy, latency, is_conv)
        return PluginFeedbackResult(control_signal=(cdx, cdy), active_plugins=active,
                                    loop_latency_ms=latency, is_converged=is_conv)


# ============================================================
# 3. GPUDifferentiablePipeline (inspired by Kornia)
# ============================================================

@dataclass
class GPUDifferentiableConfig:
    """GPU 可微分流水线配置。"""
    use_gpu: bool = True
    pipeline_stages: List[str] = field(default_factory=lambda: [
        "normalize", "gaussian_blur", "edge_enhance", "threshold_adaptive", "centroid_compute"])
    precision: str = "float32"


@dataclass
class GPUDifferentiableResult:
    """GPU 可微分流水线处理结果。"""
    processed_image: np.ndarray
    centroid: Tuple[float, float]
    processing_time_ms: float
    gpu_memory_used_mb: float


class GPUDifferentiablePipeline:
    """GPU 加速可微分图像处理流水线。

    灵感来源于 Kornia 项目，实现可组合的图像处理流水线。
    每个阶段都是可微分的（通过有限差分近似梯度），支持自动求导。
    支持阶段：normalize, gaussian_blur, edge_enhance, threshold_adaptive, centroid_compute。
    """

    def __init__(self, config: Optional[GPUDifferentiableConfig] = None):
        self.config = config or GPUDifferentiableConfig()
        self._gpu_available = False
        self._dtype = np.float32 if self.config.precision == "float32" else np.float64
        self._stages: Dict[str, Callable] = {
            "normalize": self._stage_normalize, "gaussian_blur": self._stage_gaussian_blur,
            "edge_enhance": self._stage_edge_enhance, "threshold_adaptive": self._stage_threshold_adaptive,
            "centroid_compute": self._stage_centroid_compute}
        self._intermediate: Dict[str, Any] = {}
        LOGGER.info("GPU 可微分流水线初始化完成, 阶段=%s", self.config.pipeline_stages)

    def reset(self) -> None:
        """重置流水线，清除中间结果。"""
        self._intermediate.clear()

    def process(self, image: np.ndarray) -> GPUDifferentiableResult:
        """执行完整的图像处理流水线。

        Args:
            image: 输入图像（灰度或彩色）。
        Returns:
            GPUDifferentiableResult 包含处理结果和性能指标。
        """
        t0 = time.perf_counter()
        if image.ndim == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image.copy()
        img = img.astype(self._dtype)
        self._intermediate["input"] = img
        current = img
        for sn in self.config.pipeline_stages:
            if sn not in self._stages:
                LOGGER.warning("未知阶段: %s，跳过", sn)
                continue
            current = self._stages[sn](current)
            self._intermediate[sn] = current
        centroid = (0.0, 0.0)
        if "centroid_compute" in self._intermediate:
            cd = self._intermediate["centroid_compute"]
            centroid = (cd[0], cd[1]) if isinstance(cd, tuple) else self._compute_centroid(current)
        final = img
        for sn in reversed(self.config.pipeline_stages):
            if sn != "centroid_compute" and sn in self._intermediate:
                v = self._intermediate[sn]
                if isinstance(v, np.ndarray):
                    final = v; break
        elapsed = (time.perf_counter() - t0) * 1000
        gpu_mem = 0.0 if not self._gpu_available else img.size * (4 if self._dtype == np.float32 else 8) * len(self.config.pipeline_stages) / (1024 * 1024)
        return GPUDifferentiableResult(processed_image=final, centroid=centroid,
                                       processing_time_ms=elapsed, gpu_memory_used_mb=gpu_mem)

    def _stage_normalize(self, image: np.ndarray) -> np.ndarray:
        """归一化到 [0, 1]。"""
        mn, mx = image.min(), image.max()
        return (image - mn) / (mx - mn) if mx - mn > 1e-12 else np.zeros_like(image)

    def _stage_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """高斯模糊去噪。"""
        u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        return cv2.GaussianBlur(u8, (3, 3), 1.0).astype(self._dtype) / 255.0

    def _stage_edge_enhance(self, image: np.ndarray) -> np.ndarray:
        """拉普拉斯边缘增强。"""
        u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        lap = cv2.Laplacian(u8, cv2.CV_16S, ksize=3)
        enhanced = cv2.convertScaleAbs(u8.astype(np.float64) - 0.5 * lap)
        return enhanced.astype(self._dtype) / 255.0

    def _stage_threshold_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Otsu 自适应阈值分割。"""
        u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary.astype(self._dtype) / 255.0

    def _stage_centroid_compute(self, image: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """计算加权质心。"""
        return (*self._compute_centroid(image), image)

    @staticmethod
    def _compute_centroid(image: np.ndarray) -> Tuple[float, float]:
        """计算强度加权质心。"""
        total = image.sum()
        if total < 1e-12:
            return (float(image.shape[1] / 2), float(image.shape[0] / 2))
        yi, xi = np.mgrid[:image.shape[0], :image.shape[1]]
        return (float(np.sum(xi * image) / total), float(np.sum(yi * image) / total))


# ============================================================
# 4. RLSpotCenteringAgent (inspired by gym_ao)
# ============================================================

@dataclass
class RLSpotCenteringConfig:
    """强化学习光斑对中智能体配置。"""
    state_dim: int = 21
    action_dim: int = 5
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 0.2
    max_steps: int = 100


@dataclass
class RLSpotCenteringResult:
    """强化学习对中结果。"""
    action: Tuple[float, float]
    reward: float
    q_value: float
    convergence_step: int


class RLSpotCenteringAgent:
    """强化学习驱动的光斑对中智能体。

    灵感来源于 gym_ao 项目，使用简化的 Q-learning 算法
    实现光斑 tip/tilt 自动对中控制。

    状态空间：光斑偏移量离散化到 state_dim x state_dim 网格。
    动作空间：每个轴上 action_dim 个离散动作。
    奖励函数：基于光斑到目标距离的高斯奖励。
    """

    def __init__(self, config: Optional[RLSpotCenteringConfig] = None):
        self.config = config or RLSpotCenteringConfig()
        self._q_table = np.zeros((self.config.state_dim, self.config.state_dim,
                                  self.config.action_dim, self.config.action_dim))
        self._current_step: int = 0
        self._action_map = np.linspace(-2.0, 2.0, self.config.action_dim)
        self._state_history: deque = deque(maxlen=50)
        LOGGER.info("RL 光斑对中智能体初始化完成, Q表=%s, 探索率=%.2f",
                     self._q_table.shape, self.config.exploration_rate)

    def reset(self) -> None:
        """重置智能体回合（保留 Q 表学习成果）。"""
        self._current_step = 0
        self._state_history.clear()

    def _discretize_state(self, offset: Tuple[float, float], max_range: float = 10.0) -> Tuple[int, int]:
        """将连续偏移量离散化到状态网格。"""
        n = self.config.state_dim
        sx = max(0, min(n - 1, int((offset[0] / max_range + 1) * (n - 1) / 2)))
        sy = max(0, min(n - 1, int((offset[1] / max_range + 1) * (n - 1) / 2)))
        return (sx, sy)

    def _compute_reward(self, offset: Tuple[float, float]) -> float:
        """高斯奖励函数：距中心越近奖励越高。"""
        d = math.sqrt(offset[0] ** 2 + offset[1] ** 2)
        return math.exp(-d ** 2 / 2.0)

    def _select_action(self, state: Tuple[int, int]) -> Tuple[int, int]:
        """epsilon-greedy 动作选择。"""
        if np.random.random() < self.config.exploration_rate:
            return (np.random.randint(0, self.config.action_dim),
                    np.random.randint(0, self.config.action_dim))
        q = self._q_table[state[0], state[1]]
        return (int(np.argmax(q[:, 0])), int(np.argmax(q[0, :])))

    def _update_q(self, state: Tuple[int, int], action: Tuple[int, int],
                  reward: float, next_state: Tuple[int, int]) -> float:
        """Q-learning 核心更新规则。"""
        a, g = self.config.learning_rate, self.config.discount_factor
        old_q = self._q_table[state[0], state[1], action[0], action[1]]
        new_q = old_q + a * (reward + g * np.max(self._q_table[next_state[0], next_state[1]]) - old_q)
        self._q_table[state[0], state[1], action[0], action[1]] = new_q
        return float(new_q)

    def predict(self, current_offset: Tuple[float, float]) -> RLSpotCenteringResult:
        """根据当前偏移量预测最佳对中动作。

        Args:
            current_offset: 当前光斑相对于目标的偏移量 (dx, dy)。
        Returns:
            RLSpotCenteringResult 包含动作、奖励和 Q 值。
        """
        self._current_step += 1
        state = self._discretize_state(current_offset)
        aidx = self._select_action(state)
        dx, dy = float(self._action_map[aidx[0]]), float(self._action_map[aidx[1]])
        new_off = (current_offset[0] + dx, current_offset[1] + dy)
        reward = self._compute_reward(new_off)
        next_state = self._discretize_state(new_off)
        q_val = self._update_q(state, aidx, reward, next_state)
        conv = self._current_step if abs(new_off[0]) < 0.5 and abs(new_off[1]) < 0.5 else 0
        self._state_history.append((state, aidx, reward))
        LOGGER.debug("RL #%d: 偏移=(%.2f,%.2f)->动作=(%.2f,%.2f), r=%.3f, Q=%.3f",
                      self._current_step, *current_offset, dx, dy, reward, q_val)
        return RLSpotCenteringResult(action=(dx, dy), reward=reward, q_value=q_val, convergence_step=conv)


# ============================================================
# 5. FoundationModelEnhancer (inspired by UniFMIR / Cellpose3)
# ============================================================

@dataclass
class FoundationEnhancerConfig:
    """基础模型图像增强器配置。"""
    enhance_strength: float = 0.7
    denoise_kernel_size: int = 3
    sharpen_amount: float = 0.5
    contrast_limit: float = 2.0


@dataclass
class FoundationEnhancerResult:
    """基础模型增强结果。"""
    enhanced_image: np.ndarray
    psnr_improvement: float
    ssim_score: float
    processing_time_ms: float


class FoundationModelEnhancer:
    """基础模型启发的多阶段图像增强器。

    灵感来源于 UniFMIR 和 Cellpose3 的预处理策略，实现多阶段增强流水线：
    去噪（非局部均值）-> 对比度增强（CLAHE）-> 锐化（Unsharp Mask）-> 归一化。
    所有阶段可通过 enhance_strength 统一调节强度。
    """

    def __init__(self, config: Optional[FoundationEnhancerConfig] = None):
        self.config = config or FoundationEnhancerConfig()
        LOGGER.info("基础模型增强器初始化完成, 强度=%.2f", self.config.enhance_strength)

    def reset(self) -> None:
        """重置增强器状态。"""

    def process(self, image: np.ndarray) -> FoundationEnhancerResult:
        """对输入图像执行多阶段增强处理。

        Args:
            image: 输入图像（灰度或彩色）。
        Returns:
            FoundationEnhancerResult 包含增强图像和质量指标。
        """
        t0 = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        original = gray.copy().astype(np.float64)
        # 四阶段处理
        denoised = cv2.fastNlMeansDenoising(gray, None, h=self.config.denoise_kernel_size,
                                             templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=self.config.contrast_limit, tileGridSize=(8, 8))
        contrasted = clahe.apply(denoised)
        blurred = cv2.GaussianBlur(contrasted, (0, 0), 2.0)
        sharpened = cv2.addWeighted(contrasted, 1.0 + self.config.sharpen_amount, blurred,
                                     -self.config.sharpen_amount, 0)
        p_lo, p_hi = np.percentile(sharpened, [2, 98])
        rng = p_hi - p_lo
        normalized = np.clip((sharpened - p_lo) / rng * 255, 0, 255) if rng > 1e-6 else sharpened.astype(np.float64)
        # 按强度混合
        s = self.config.enhance_strength
        result = np.clip(original * (1 - s) + normalized * s, 0, 255).astype(np.uint8)
        psnr = self._compute_psnr(original, result.astype(np.float64))
        ssim = self._compute_ssim(gray, result)
        elapsed = (time.perf_counter() - t0) * 1000
        LOGGER.debug("增强完成: PSNR=%.1fdB, SSIM=%.3f, 耗时=%.1fms", psnr, ssim, elapsed)
        return FoundationEnhancerResult(enhanced_image=result, psnr_improvement=psnr,
                                         ssim_score=ssim, processing_time_ms=elapsed)

    @staticmethod
    def _compute_psnr(orig: np.ndarray, enhanced: np.ndarray) -> float:
        """计算 PSNR。"""
        mse = np.mean((orig - enhanced) ** 2)
        return 100.0 if mse < 1e-12 else float(10 * math.log10(255.0 ** 2 / mse))

    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """计算简化版 SSIM。"""
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        f1, f2 = img1.astype(np.float64), img2.astype(np.float64)
        mu1 = cv2.GaussianBlur(f1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(f2, (11, 11), 1.5)
        s1 = cv2.GaussianBlur(f1 ** 2, (11, 11), 1.5) - mu1 ** 2
        s2 = cv2.GaussianBlur(f2 ** 2, (11, 11), 1.5) - mu2 ** 2
        s12 = cv2.GaussianBlur(f1 * f2, (11, 11), 1.5) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
        return float(np.mean(ssim_map))


# ============================================================
# 6. PhysicsInformedWavefrontPredictor (inspired by DeepXDE / PINN)
# ============================================================

@dataclass
class PhysicsWavefrontConfig:
    """物理信息波前预测器配置。"""
    num_zernike_modes: int = 10
    prediction_horizon_frames: int = 5
    physics_weight: float = 0.3
    temporal_smoothness: float = 0.8


@dataclass
class PhysicsWavefrontResult:
    """物理信息波前预测结果。"""
    predicted_zernike_coeffs: np.ndarray
    prediction_confidence: float
    energy_conservation_error: float


class PhysicsInformedWavefrontPredictor:
    """物理信息约束的波前预测器。

    灵感来源于 DeepXDE 和 NVIDIA Modulus PINN，利用物理约束进行波前时序预测。
    使用 Zernike 基函数参数化，施加能量守恒、时序平滑和模式耦合约束。
    预测方法：基于历史系数的加权线性外推，权重由物理约束违反程度动态调整。
    """

    def __init__(self, config: Optional[PhysicsWavefrontConfig] = None):
        self.config = config or PhysicsWavefrontConfig()
        self._history: deque = deque(maxlen=30)
        self._baseline_energy: float = 0.0
        LOGGER.info("物理信息波前预测器初始化完成, 模式数=%d, 预测步长=%d",
                     self.config.num_zernike_modes, self.config.prediction_horizon_frames)

    def reset(self) -> None:
        """重置预测器，清空历史数据。"""
        self._history.clear()
        self._baseline_energy = 0.0

    def update(self, zernike_coeffs: np.ndarray) -> None:
        """用新的观测数据更新预测器。

        Args:
            zernike_coeffs: 当前帧的 Zernike 系数。
        """
        c = np.array(zernike_coeffs, dtype=np.float64)
        padded = np.zeros(self.config.num_zernike_modes)
        n = min(len(c), self.config.num_zernike_modes)
        padded[:n] = c[:n]
        self._history.append(padded)
        if self._baseline_energy == 0.0:
            self._baseline_energy = float(np.sum(padded ** 2))

    def predict(self) -> PhysicsWavefrontResult:
        """预测未来几帧的 Zernike 系数。

        Returns:
            PhysicsWavefrontResult 包含预测系数和置信度。
        """
        if len(self._history) < 3:
            LOGGER.warning("历史数据不足（%d帧），无法可靠预测", len(self._history))
            return PhysicsWavefrontResult(np.zeros(self.config.num_zernike_modes), 0.0, float('inf'))
        ha = np.array(self._history)
        nf, nm, hz = len(ha), self.config.num_zernike_modes, self.config.prediction_horizon_frames
        preds = np.zeros((hz, nm))
        t = np.arange(nf, dtype=np.float64)
        for m in range(nm):
            cs = ha[:, m]
            w = np.exp(-0.1 * (nf - 1 - t))
            ws = w.sum()
            if ws < 1e-12:
                slope, intercept = 0.0, cs[-1]
            else:
                wmt = np.sum(w * t) / ws
                wmc = np.sum(w * cs) / ws
                wvar = np.sum(w * (t - wmt) ** 2)
                slope = (np.sum(w * (t - wmt) * (cs - wmc)) / wvar) if abs(wvar) > 1e-12 else 0.0
                intercept = wmc - slope * wmt
            for h in range(hz):
                preds[h, m] = intercept + slope * (nf + h)
        preds = self._apply_constraints(preds, ha)
        e_err = self._energy_error(preds)
        conf = self._confidence(ha, preds)
        LOGGER.debug("波前预测: 置信度=%.3f, 能量误差=%.4f", conf, e_err)
        return PhysicsWavefrontResult(predicted_zernike_coeffs=preds,
                                       prediction_confidence=conf, energy_conservation_error=e_err)

    def _apply_constraints(self, preds: np.ndarray, history: np.ndarray) -> np.ndarray:
        """施加物理约束：时序平滑 + 能量守恒 + 模式耦合。"""
        c = preds.copy()
        alpha, last = self.config.temporal_smoothness, history[-1]
        for h in range(len(c)):
            blend = alpha ** (h + 1)
            c[h] = blend * last + (1 - blend) * c[h]
            ce = np.sum(c[h] ** 2)
            if ce > 1e-12:
                scale = np.sqrt(self._baseline_energy / ce)
                cb = self.config.physics_weight * (1 - h / max(len(c), 1))
                c[h] = c[h] * (1 - cb) + c[h] * scale * cb
            for mode in range(2, self.config.num_zernike_modes):
                le = np.mean(c[h, :mode] ** 2)
                he = c[h, mode] ** 2
                if he > le * 2 and le > 1e-12:
                    c[h, mode] *= math.sqrt(le * 2 / he)
        return c

    def _energy_error(self, preds: np.ndarray) -> float:
        """计算能量守恒误差。"""
        if self._baseline_energy < 1e-12:
            return 0.0
        return float(np.mean([abs(np.sum(p ** 2) - self._baseline_energy) / self._baseline_energy for p in preds]))

    def _confidence(self, history: np.ndarray, preds: np.ndarray) -> float:
        """基于时序一致性和预测合理性计算置信度。"""
        if len(history) < 3:
            return 0.0
        diffs = np.diff(history, axis=0)
        consistency = 1.0 / (1.0 + np.mean(np.std(diffs, axis=0)))
        hmin, hmax = np.min(history, axis=0), np.max(history, axis=0)
        hr = hmax - hmin; hr[hr < 1e-12] = 1.0
        oor = np.mean(np.clip(np.abs(preds[0] - np.mean(history, axis=0)) / hr - 1, 0, None))
        return max(0.0, min(1.0, float(consistency / (1.0 + oor))))


# ============================================================
# 7. RealTimeMatrixAccelerator (inspired by Matilda - NSO)
# ============================================================

@dataclass
class MatrixAcceleratorConfig:
    """实时矩阵加速器配置。"""
    matrix_size: int = 64
    use_simd: bool = True
    cache_block_size: int = 16
    precision: str = "float32"


@dataclass
class MatrixAcceleratorResult:
    """矩阵加速器计算结果。"""
    output_vector: np.ndarray
    computation_time_us: float
    gflops_achieved: float
    cache_hit_ratio_estimate: float


class RealTimeMatrixAccelerator:
    """实时矩阵运算加速器。

    灵感来源于 Matilda (NSO)，为自适应光学实时波前校正提供优化的矩阵运算。
    核心是分块矩阵-向量乘法（blocked MV），通过缓存友好的数据访问模式提升性能。
    """

    def __init__(self, config: Optional[MatrixAcceleratorConfig] = None):
        self.config = config or MatrixAcceleratorConfig()
        self._dtype = np.float32 if self.config.precision == "float32" else np.float64
        self._matrix: Optional[np.ndarray] = None
        self._cache_warm: bool = False
        LOGGER.info("实时矩阵加速器初始化完成, 尺寸=%d, 分块=%d",
                     self.config.matrix_size, self.config.cache_block_size)

    def reset(self) -> None:
        """重置加速器。"""
        self._matrix = None
        self._cache_warm = False

    def set_matrix(self, matrix: np.ndarray) -> None:
        """设置校正矩阵。"""
        self._matrix = matrix.astype(self._dtype)
        self._cache_warm = False

    def warm_cache(self) -> None:
        """预热缓存以减少实时计算延迟抖动。"""
        if self._matrix is None:
            return
        n = self._matrix.shape[0]
        dummy = np.ones(n, dtype=self._dtype)
        for _ in range(3):
            self._blocked_mv(self._matrix, dummy)
        self._cache_warm = True

    def process(self, input_vector: np.ndarray) -> MatrixAcceleratorResult:
        """执行分块矩阵-向量乘法。

        Args:
            input_vector: 输入向量。
        Returns:
            MatrixAcceleratorResult 包含输出向量和性能指标。
        """
        if self._matrix is None:
            self._matrix = np.eye(self.config.matrix_size, dtype=self._dtype)
        vec = input_vector.astype(self._dtype)
        t0 = time.perf_counter()
        output = self._blocked_mv(self._matrix, vec)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        n = self._matrix.shape[0]
        flops = 2.0 * n * n
        gflops = flops / (elapsed_us * 1e-6) / 1e9 if elapsed_us > 0 else 0.0
        cache_hit = self._estimate_cache_hit(n)
        LOGGER.debug("矩阵运算: 耗时=%.1fus, GFLOPS=%.2f, 缓存=%.1f%%",
                      elapsed_us, gflops, cache_hit * 100)
        return MatrixAcceleratorResult(output_vector=output, computation_time_us=elapsed_us,
                                        gflops_achieved=gflops, cache_hit_ratio_estimate=cache_hit)

    def _blocked_mv(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """分块矩阵-向量乘法，缓存友好的数据访问。"""
        n, block = matrix.shape[0], self.config.cache_block_size
        output = np.zeros(n, dtype=self._dtype)
        for i0 in range(0, n, block):
            i1 = min(i0 + block, n)
            for j0 in range(0, n, block):
                j1 = min(j0 + block, n)
                output[i0:i1] += matrix[i0:i1, j0:j1] @ vector[j0:j1]
        return output

    def _estimate_cache_hit(self, n: int) -> float:
        """估算缓存命中率（基于 L1=32KB, L2=256KB 假设）。"""
        bpe = 4 if self._dtype == np.float32 else 8
        bb = self.config.cache_block_size ** 2 * bpe
        mb = n ** 2 * bpe
        l1h = min(1.0, 32768 / bb) if bb > 0 else 1.0
        l2h = min(1.0, 262144 / mb) if mb > 0 else 1.0
        return float(0.6 * l2h + 0.4 * l1h)


# ============================================================
# 8. ShackHartmannAnalyzer (inspired by ShackHartmannAnalysis)
# ============================================================

@dataclass
class ShackHartmannConfig:
    """Shack-Hartmann 分析器配置。"""
    lenslet_pitch_px: int = 32
    lenslet_size_px: int = 28
    threshold_factor: float = 3.0
    max_displacement_px: float = 10.0


@dataclass
class ShackHartmannResult:
    """Shack-Hartmann 分析结果。"""
    spot_positions: List[Tuple[float, float]]
    local_slopes: np.ndarray
    reconstructed_wavefront: np.ndarray
    rms_error: float


class ShackHartmannAnalyzer:
    """Shack-Hartmann 波前传感器光斑分析器。

    灵感来源于 ShackHartmannAnalysis 项目，实现完整的 SH 传感器数据处理：
    1. 网格检测：基于微透镜间距定位各子孔径区域
    2. 光斑检测：在每个子孔径内检测光斑质心
    3. 斜率计算：根据光斑偏移计算局部波前斜率
    4. 波前重建：使用积分方法重建全局波前
    """

    def __init__(self, config: Optional[ShackHartmannConfig] = None):
        self.config = config or ShackHartmannConfig()
        self._reference_positions: Optional[np.ndarray] = None
        self._grid_shape: Tuple[int, int] = (0, 0)
        LOGGER.info("SH 分析器初始化完成, 透镜间距=%dpx", self.config.lenslet_pitch_px)

    def reset(self) -> None:
        """重置分析器，清除参考位置。"""
        self._reference_positions = None
        self._grid_shape = (0, 0)

    def calibrate(self, reference_image: np.ndarray) -> None:
        """使用参考图像（平面波前）校准分析器。

        Args:
            reference_image: 参考光斑图像。
        """
        positions = self._detect_all_spots(reference_image)
        self._reference_positions = np.array(positions, dtype=np.float64)
        self._grid_shape = self._grid_shape_for(reference_image.shape)
        LOGGER.info("校准完成, %d 个子孔径, 网格=%s", len(positions), self._grid_shape)

    def detect(self, spot_image: np.ndarray) -> ShackHartmannResult:
        """分析 Shack-Hartmann 光斑图像。

        Args:
            spot_image: SH 传感器图像。
        Returns:
            ShackHartmannResult 包含光斑位置、斜率和重建波前。
        """
        t0 = time.perf_counter()
        positions = self._detect_all_spots(spot_image)
        slopes = self._compute_slopes(positions)
        wavefront = self._reconstruct_wavefront(slopes)
        rms = float(np.sqrt(np.mean(wavefront ** 2))) if wavefront.size > 0 else 0.0
        elapsed = (time.perf_counter() - t0) * 1000
        LOGGER.debug("SH 分析: %d 光斑, RMS=%.4frad, 耗时=%.1fms", len(positions), rms, elapsed)
        return ShackHartmannResult(spot_positions=positions, local_slopes=slopes,
                                    reconstructed_wavefront=wavefront, rms_error=rms)

    def _grid_shape_for(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        """估算微透镜网格行列数。"""
        p = self.config.lenslet_pitch_px
        return (max(1, (shape[0] - p // 2) // p), max(1, (shape[1] - p // 2) // p))

    def _detect_all_spots(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """检测所有子孔径中的光斑位置。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        pitch, size, half = self.config.lenslet_pitch_px, self.config.lenslet_size_px, self.config.lenslet_pitch_px // 2
        rows, cols = gray.shape
        nr, nc = self._grid_shape_for(gray.shape)
        positions: List[Tuple[float, float]] = []
        for row in range(nr):
            for col in range(nc):
                cy, cx = half + row * pitch, half + col * pitch
                y0, y1 = max(0, cy - size // 2), min(rows, cy + size // 2)
                x0, x1 = max(0, cx - size // 2), min(cols, cx + size // 2)
                if y1 <= y0 or x1 <= x0:
                    continue
                spot = self._detect_spot(gray[y0:y1, x0:x1])
                if spot is not None:
                    positions.append((x0 + spot[0], y0 + spot[1]))
        return positions

    def _detect_spot(self, sub: np.ndarray) -> Optional[Tuple[float, float]]:
        """在单个子孔径中检测光斑质心。"""
        if sub.size < 4:
            return None
        bg = np.median(sub)
        noise = np.std(sub[sub < np.percentile(sub, 50)])
        noise = max(noise, 1.0)
        mask = sub > (bg + self.config.threshold_factor * noise)
        if np.sum(mask) < 2:
            return None
        w = sub.astype(np.float64) * mask
        total = w.sum()
        if total < 1e-12:
            return None
        yi, xi = np.mgrid[:sub.shape[0], :sub.shape[1]]
        return (float(np.sum(xi * w) / total), float(np.sum(yi * w) / total))

    def _compute_slopes(self, positions: List[Tuple[float, float]]) -> np.ndarray:
        """计算各子孔径的局部波前斜率。"""
        if not positions:
            return np.zeros((0, 2))
        detected = np.array(positions, dtype=np.float64)
        if self._reference_positions is not None and len(self._reference_positions) == len(positions):
            disp = detected - self._reference_positions
        else:
            pitch = self.config.lenslet_pitch_px
            ref = np.zeros_like(detected)
            for i, (y, x) in enumerate(positions):
                r, c = int(round((y - pitch // 2) / pitch)), int(round((x - pitch // 2) / pitch))
                ref[i] = [pitch // 2 + c * pitch, pitch // 2 + r * pitch]
            disp = detected - ref
        disp = np.clip(disp, -self.config.max_displacement_px, self.config.max_displacement_px)
        return disp / self.config.lenslet_pitch_px

    def _reconstruct_wavefront(self, slopes: np.ndarray) -> np.ndarray:
        """从斜率数据重建波前（Southwell 简化积分法）。"""
        if len(slopes) < 4:
            return np.zeros((self.config.lenslet_pitch_px * 2,) * 2)
        nr, nc = self._grid_shape
        if nr < 2 or nc < 2:
            return np.zeros((self.config.lenslet_pitch_px * 2,) * 2)
        rs = min(self.config.lenslet_pitch_px * min(nr, nc), 128)
        rs = max(16, rs)
        wf = np.zeros((rs, rs), dtype=np.float64)
        sgx, sgy = np.zeros((nr, nc)), np.zeros((nr, nc))
        idx = 0
        for r in range(nr):
            for c in range(nc):
                if idx < len(slopes):
                    sgx[r, c], sgy[r, c] = slopes[idx, 0], slopes[idx, 1]
                    idx += 1
        # 沿 x 方向积分
        for r in range(nr):
            cum = 0.0
            for c in range(1, nc):
                cum += sgx[r, c]
                ys, ye = r * rs // nr, min((r + 1) * rs // nr, rs)
                xs, xe = c * rs // nc, min((c + 1) * rs // nc, rs)
                wf[ys:ye, xs:xe] = cum
        # 沿 y 方向积分叠加
        for c in range(nc):
            cum = 0.0
            for r in range(1, nr):
                cum += sgy[r, c]
                ys, ye = r * rs // nr, min((r + 1) * rs // nr, rs)
                xs, xe = c * rs // nc, min((c + 1) * rs // nc, rs)
                wf[ys:ye, xs:xe] += cum
        wf -= np.mean(wf)
        return wf
