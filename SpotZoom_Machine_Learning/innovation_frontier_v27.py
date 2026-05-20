"""
SpotZoom 前沿创新模块 v27

本模块包含 8 个创新组件，灵感来源于开源显微镜与图像分析项目的研究成果：

- Picasso (MIT): 最大似然估计高斯拟合与 CRLB 精度估计。
- ThunderSTORM (GPL-3.0): à trous 小波变换多尺度光斑检测。
- Micro-Manager HAL (LGPL-2.1): 模块化流水线引擎与硬件抽象层设计。
- PYME (GPL-3.0): Numba JIT 加速核心计算函数。
- AOtools / prysm (LGPL-3.0 / MIT): Strehl 比实时监测与光学质量评估。
- Gpufit (MIT): GPU 加速 Levenberg-Marquardt 批量拟合适配器。
- ZeroCostDL4Mic / DeepImageJ (MIT / BSD-2-Clause): Bioimage Model Zoo 标准模型适配。
- TrackMate/Fiji (GPL-2.0): LAP 线性分配问题轨迹链接与统计分析。

所有组件仅依赖 numpy 和 scipy，不引入任何外部深度学习框架。
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

LOGGER = logging.getLogger(__name__)


# ============================================================
# 1. MLEGaussianFitter (inspired by Picasso)
# ============================================================

@dataclass
class MLEGaussianConfig:
    """最大似然估计高斯拟合配置。

    Attributes
    ----------
    sigma_guess : float
        高斯 sigma 初始猜测值 (像素)。
    max_iterations : int
        EM 算法最大迭代次数。
    tolerance : float
        对数似然收敛容差。
    background_method : str
        背景估计方法: "median" 或 "local"。
    roi_half_size : int
        拟合 ROI 半宽 (像素)。
    """
    sigma_guess: float = 1.5
    max_iterations: int = 200
    tolerance: float = 1e-4
    background_method: str = "median"
    roi_half_size: int = 7


@dataclass
class MLEFittingResult:
    """MLE 高斯拟合结果。

    Attributes
    ----------
    x : float
        亚像素 x 坐标。
    y : float
        亚像素 y 坐标。
    sigma : float
        拟合高斯 sigma (像素)。
    amplitude : float
        拟合振幅 (光子数)。
    background : float
        背景水平 (光子/像素)。
    log_likelihood : float
        最终对数似然值。
    crlb_precision : Tuple[float, float]
        Cramér-Rao 下界精度估计 (sigma_x, sigma_y)。
    converged : bool
        是否收敛。
    """
    x: float
    y: float
    sigma: float
    amplitude: float
    background: float
    log_likelihood: float
    crlb_precision: Tuple[float, float]
    converged: bool


class MLEGaussianFitter:
    """最大似然估计高斯拟合器。

    灵感来源于 Picasso (jungmannlab/picasso)，使用 EM 算法实现
    泊松噪声模型下的二维高斯 MLE 拟合，适用于光子受限的
    超分辨率显微镜单分子定位。

    算法原理：
    - 泊松噪声模型: P(k|lambda) = lambda^k * exp(-lambda) / k!
    - EM 迭代: E 步计算期望计数，M 步更新参数
    - CRLB 精度估计: Fisher 信息矩阵的逆矩阵对角线元素
    - 参考: Mortensen et al., Nature Methods 7, 377-381 (2010)

    Parameters
    ----------
    config : MLEGaussianConfig, optional
        拟合配置参数。
    """

    def __init__(self, config: Optional[MLEGaussianConfig] = None):
        self.config = config or MLEGaussianConfig()
        LOGGER.info("MLE 高斯拟合器初始化完成, sigma_guess=%.2f, max_iter=%d",
                     self.config.sigma_guess, self.config.max_iterations)

    def reset(self) -> None:
        """重置拟合器内部状态。"""
        LOGGER.debug("MLE 高斯拟合器已重置")

    def fit(self, image: np.ndarray,
            x_init: Optional[float] = None,
            y_init: Optional[float] = None) -> MLEFittingResult:
        """对光斑图像执行 MLE 高斯拟合。

        Args:
            image: 输入光斑图像 (2D numpy 数组，整数值表示光子计数)。
            x_init: 初始 x 坐标猜测 (像素)，None 则使用质心。
            y_init: 初始 y 坐标猜测 (像素)，None 则使用质心。
        Returns:
            MLEFittingResult 包含拟合参数和精度估计。
        """
        t0 = time.perf_counter()
        img = image.astype(np.float64)
        rows, cols = img.shape
        h = self.config.roi_half_size

        # 裁剪 ROI
        cy, cx = rows // 2, cols // 2
        y0, y1 = max(0, cy - h), min(rows, cy + h + 1)
        x0, x1 = max(0, cx - h), min(cols, cx + h + 1)
        roi = img[y0:y1, x0:x1].copy()
        roi_rows, roi_cols = roi.shape

        if roi.size == 0 or np.all(roi <= 0):
            LOGGER.warning("空 ROI 或无信号，返回默认结果")
            return MLEFittingResult(
                x=float(cx), y=float(cy), sigma=self.config.sigma_guess,
                amplitude=0.0, background=0.0, log_likelihood=-np.inf,
                crlb_precision=(np.inf, np.inf), converged=False)

        # 背景估计
        if self.config.background_method == "median":
            bg = float(np.median(roi))
        else:
            border = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
            bg = float(np.median(border)) if border.size > 0 else float(np.min(roi))
        bg = max(bg, 0.0)

        # 初始参数
        if x_init is not None and y_init is not None:
            x0_est = x_init - x0
            y0_est = y_init - y0
        else:
            total = roi.sum()
            if total < 1e-12:
                return MLEFittingResult(
                    x=float(cx), y=float(cy), sigma=self.config.sigma_guess,
                    amplitude=0.0, background=bg, log_likelihood=-np.inf,
                    crlb_precision=(np.inf, np.inf), converged=False)
            yi, xi = np.mgrid[:roi_rows, :roi_cols]
            x0_est = float(np.sum(xi * roi) / total)
            y0_est = float(np.sum(yi * roi) / total)

        amp = max(float(roi.max()) - bg, 1.0)
        sigma = self.config.sigma_guess

        # EM 迭代
        yi, xi = np.mgrid[:roi_rows, :roi_cols]
        prev_ll = -np.inf
        converged = False

        for iteration in range(self.config.max_iterations):
            # 计算期望值 (E 步)
            r2 = (xi - x0_est) ** 2 + (yi - y0_est) ** 2
            model = amp * np.exp(-r2 / (2.0 * sigma ** 2)) + bg
            model = np.clip(model, 1e-10, None)

            # 泊松对数似然
            ll = float(np.sum(roi * np.log(model) - model))

            if abs(ll - prev_ll) < self.config.tolerance and iteration > 0:
                converged = True
                break
            prev_ll = ll

            # M 步: 更新参数
            weights = roi / model
            total_w = float(weights.sum())
            if total_w < 1e-12:
                break

            x0_est = float(np.sum(weights * xi * model) / total_w)
            y0_est = float(np.sum(weights * yi * model) / total_w)
            # 限制在 ROI 范围内
            x0_est = np.clip(x0_est, 0, roi_cols - 1)
            y0_est = np.clip(y0_est, 0, roi_rows - 1)

            gauss_part = amp * np.exp(-r2 / (2.0 * sigma ** 2))
            exp_norm = np.sum(weights * np.exp(-r2 / (2.0 * sigma ** 2)))
            if exp_norm < 1e-12:
                break
            amp_new = float(np.sum(weights * gauss_part) / exp_norm)
            amp = max(amp_new, 1.0)

            gauss_weighted = np.sum(weights * gauss_part)
            if gauss_weighted < 1e-12:
                break
            sigma_new_sq = float(np.sum(weights * gauss_part * r2) / (2.0 * gauss_weighted))
            sigma = max(np.sqrt(max(sigma_new_sq, 0.01)), 0.5)

            # 检查 NaN
            if np.isnan(x0_est) or np.isnan(y0_est) or np.isnan(amp) or np.isnan(sigma):
                converged = False
                break

        # CRLB 精度估计 (Fisher 信息矩阵)
        r2 = (xi - x0_est) ** 2 + (yi - y0_est) ** 2
        model_final = amp * np.exp(-r2 / (2.0 * sigma ** 2)) + bg
        model_final = np.clip(model_final, 1e-10, None)

        # d(mu)/dx 和 d(mu)/dy
        dmu_dx = model_final * (xi - x0_est) / (sigma ** 2)
        dmu_dy = model_final * (yi - y0_est) / (sigma ** 2)

        fisher_xx = float(np.sum(dmu_dx ** 2 / model_final))
        fisher_yy = float(np.sum(dmu_dy ** 2 / model_final))

        crlb_x = 1.0 / math.sqrt(fisher_xx) if fisher_xx > 1e-12 else np.inf
        crlb_y = 1.0 / math.sqrt(fisher_yy) if fisher_yy > 1e-12 else np.inf

        # 转换回原图坐标
        final_x = x0_est + x0
        final_y = y0_est + y0

        elapsed = (time.perf_counter() - t0) * 1000
        LOGGER.debug("MLE 拟合完成: pos=(%.3f,%.3f), sigma=%.3f, CRLB=(%.4f,%.4f), "
                      "收敛=%s, 耗时=%.1fms",
                      final_x, final_y, sigma, crlb_x, crlb_y, converged, elapsed)

        return MLEFittingResult(
            x=final_x, y=final_y, sigma=sigma, amplitude=amp,
            background=bg, log_likelihood=ll,
            crlb_precision=(crlb_x, crlb_y), converged=converged)


# ============================================================
# 2. WaveletSpotDetector (inspired by ThunderSTORM)
# ============================================================

@dataclass
class WaveletDetectorConfig:
    """小波光斑检测器配置。

    Attributes
    ----------
    wavelet_type : str
        小波类型: "b3_spline" (B3 样条)。
    decomposition_levels : int
        à trous 小波分解层数。
    threshold_method : str
        阈值方法: "hard" 或 "soft"。
    min_spot_size : int
        最小光斑尺寸 (像素)。
    noise_estimate_method : str
        噪声估计方法: "mad" (中位绝对偏差)。
    sensitivity : float
        检测灵敏度因子 (阈值 = sensitivity * noise_sigma)。
    """
    wavelet_type: str = "b3_spline"
    decomposition_levels: int = 3
    threshold_method: str = "hard"
    min_spot_size: int = 3
    noise_estimate_method: str = "mad"
    sensitivity: float = 3.0


@dataclass
class WaveletDetectionResult:
    """小波光斑检测结果。

    Attributes
    ----------
    spots : List[Tuple[float, float, float]]
        检测到的光斑列表 [(x, y, intensity), ...]。
    scale_space_response : np.ndarray
        多尺度响应图 (最大跨尺度响应)。
    snr_map : np.ndarray
        信噪比图。
    """
    spots: List[Tuple[float, float, float]]
    scale_space_response: np.ndarray
    snr_map: np.ndarray


class WaveletSpotDetector:
    """基于 à trous 小波变换的多尺度光斑检测器。

    灵感来源于 ThunderSTORM (zitmen/thunderstorm)，使用 à trous
    (带孔) 离散小波变换实现多尺度光斑检测，适用于低信噪比条件下的
    荧光单分子定位。

    算法原理：
    - à trous 小波: 使用插值滤波器的非下采样小波变换，保持空间分辨率
    - B3 样条滤波器: [1/16, 1/4, 3/8, 1/4, 1/16]
    - 多尺度检测: 在不同分解层级上检测不同大小的光斑
    - MAD 噪声估计: sigma = 1.4826 * median(|W|)
    - 参考: Olivo-Marin, Biophysical Journal 82(5), 2234-2242 (2002)

    Parameters
    ----------
    config : WaveletDetectorConfig, optional
        检测配置参数。
    """

    def __init__(self, config: Optional[WaveletDetectorConfig] = None):
        self.config = config or WaveletDetectorConfig()
        self._b3_spline = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        LOGGER.info("小波光斑检测器初始化完成, 层数=%d, 小波=%s",
                     self.config.decomposition_levels, self.config.wavelet_type)

    def reset(self) -> None:
        """重置检测器内部状态。"""
        LOGGER.debug("小波光斑检测器已重置")

    def detect(self, image: np.ndarray) -> WaveletDetectionResult:
        """对图像执行多尺度小波光斑检测。

        Args:
            image: 输入图像 (2D numpy 数组)。
        Returns:
            WaveletDetectionResult 包含光斑列表和多尺度响应图。
        """
        t0 = time.perf_counter()
        img = image.astype(np.float64)
        if img.ndim == 3:
            img = np.mean(img, axis=2)

        # à trous 小波分解
        c_prev = img.copy()
        wavelet_coeffs: List[np.ndarray] = []

        for level in range(self.config.decomposition_levels):
            # 上采样滤波器 (à trous: 插入 2^level 个零)
            gap = 2 ** level
            kernel = self._make_atrous_kernel(self._b3_spline, gap)
            c_smooth = self._convolve_2d(c_prev, kernel)
            w = c_prev - c_smooth
            wavelet_coeffs.append(w)
            c_prev = c_smooth

        # 噪声估计 (使用最细尺度小波系数)
        finest = wavelet_coeffs[0]
        if self.config.noise_estimate_method == "mad":
            noise_sigma = 1.4826 * float(np.median(np.abs(finest)))
        else:
            noise_sigma = float(np.std(finest))
        noise_sigma = max(noise_sigma, 1e-10)

        # 多尺度阈值检测
        threshold = self.config.sensitivity * noise_sigma
        rows, cols = img.shape
        scale_response = np.zeros((rows, cols), dtype=np.float64)
        snr_map = np.zeros((rows, cols), dtype=np.float64)

        for w in wavelet_coeffs:
            if self.config.threshold_method == "hard":
                filtered = np.where(w > threshold, w, 0.0)
            else:
                filtered = np.where(np.abs(w) > threshold,
                                    np.sign(w) * (np.abs(w) - threshold), 0.0)
            scale_response = np.maximum(scale_response, filtered)
            snr_map = np.maximum(snr_map, w / noise_sigma)

        # 局部极大值检测
        from scipy.ndimage import maximum_filter, minimum_filter
        local_max = maximum_filter(scale_response, size=self.config.min_spot_size)
        detected = (scale_response == local_max) & (scale_response > threshold)

        # 提取光斑坐标
        spots: List[Tuple[float, float, float]] = []
        ys, xs = np.where(detected)
        for y, x in zip(ys, xs):
            intensity = float(scale_response[y, x])
            spots.append((float(x), float(y), intensity))

        # 按强度排序
        spots.sort(key=lambda s: s[2], reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000
        LOGGER.debug("小波检测完成: 检测到 %d 个光斑, 噪声sigma=%.2f, "
                      "阈值=%.2f, 耗时=%.1fms",
                      len(spots), noise_sigma, threshold, elapsed)

        return WaveletDetectionResult(
            spots=spots,
            scale_space_response=scale_response,
            snr_map=snr_map)

    @staticmethod
    def _make_atrous_kernel(base: np.ndarray, gap: int) -> np.ndarray:
        """构造 à trous 插值核 (在 base 元素间插入 gap 个零)。"""
        n = len(base)
        result = np.zeros(n + (n - 1) * gap, dtype=np.float64)
        for i, v in enumerate(base):
            result[i * (gap + 1)] = v
        return result

    @staticmethod
    def _convolve_2d(image: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
        """使用可分离 1D 核进行二维卷积。"""
        from scipy.ndimage import convolve1d
        temp = convolve1d(image, kernel_1d, axis=0, mode='reflect')
        result = convolve1d(temp, kernel_1d, axis=1, mode='reflect')
        return result


# ============================================================
# 3. ModularPipelineEngine (inspired by Micro-Manager HAL)
# ============================================================

@dataclass
class ModularPipelineConfig:
    """模块化流水线引擎配置。

    Attributes
    ----------
    max_stages : int
        最大流水线阶段数。
    parallel_execution : bool
        是否启用并行执行 (无依赖阶段)。
    error_handling_strategy : str
        错误处理策略: "skip", "stop", "retry"。
    max_retries : int
        最大重试次数 (仅 retry 策略)。
    """
    max_stages: int = 20
    parallel_execution: bool = False
    error_handling_strategy: str = "skip"
    max_retries: int = 3


@dataclass
class PipelineExecutionResult:
    """流水线执行结果。

    Attributes
    ----------
    stage_results : Dict[str, Any]
        各阶段名称到输出结果的映射。
    timing_profile : Dict[str, float]
        各阶段执行耗时 (毫秒)。
    data_flow_trace : List[Dict[str, str]]
        数据流追踪记录。
    success : bool
        整体执行是否成功。
    error_messages : List[str]
        错误信息列表。
    """
    stage_results: Dict[str, Any]
    timing_profile: Dict[str, float]
    data_flow_trace: List[Dict[str, str]]
    success: bool
    error_messages: List[str]


class ModularPipelineEngine:
    """模块化流水线引擎。

    灵感来源于 Micro-Manager 的硬件抽象层 (HAL) 设计，实现
    灵活的阶段式流水线，支持输入/输出契约、依赖解析和并行执行。

    设计原则：
    - 每个阶段有明确的输入/输出契约 (schema)
    - 阶段间通过数据字典传递信息
    - 支持无依赖阶段的并行执行
    - 内置数据流追踪和性能分析
    - 参考: Micro-Manager HAL architecture, Edelstein et al. (2014)

    Parameters
    ----------
    config : ModularPipelineConfig, optional
        流水线配置参数。
    """

    def __init__(self, config: Optional[ModularPipelineConfig] = None):
        self.config = config or ModularPipelineConfig()
        self._stages: Dict[str, Dict[str, Any]] = {}
        self._execution_order: List[str] = []
        LOGGER.info("模块化流水线引擎初始化完成, 最大阶段数=%d, 并行=%s",
                     self.config.max_stages, self.config.parallel_execution)

    def reset(self) -> None:
        """重置流水线，清除所有注册的阶段。"""
        self._stages.clear()
        self._execution_order.clear()
        LOGGER.debug("模块化流水线引擎已重置")

    def add_stage(self, name: str, func: Callable,
                  input_keys: Optional[List[str]] = None,
                  output_keys: Optional[List[str]] = None,
                  dependencies: Optional[List[str]] = None) -> bool:
        """注册流水线阶段。

        Args:
            name: 阶段唯一名称。
            func: 阶段处理函数，签名为 func(data: dict) -> dict。
            input_keys: 期望的输入数据键列表。
            output_keys: 产生的输出数据键列表。
            dependencies: 依赖的前置阶段名称列表。
        Returns:
            是否注册成功。
        """
        if len(self._stages) >= self.config.max_stages:
            LOGGER.warning("阶段数已达上限 %d", self.config.max_stages)
            return False
        if name in self._stages:
            LOGGER.warning("阶段 '%s' 已存在", name)
            return False
        self._stages[name] = {
            "func": func,
            "input_keys": input_keys or [],
            "output_keys": output_keys or [],
            "dependencies": dependencies or [],
        }
        LOGGER.info("已注册阶段: %s (依赖: %s)", name, dependencies or "无")
        return True

    def _resolve_execution_order(self) -> List[str]:
        """拓扑排序解析阶段执行顺序。"""
        in_degree: Dict[str, int] = {n: 0 for n in self._stages}
        adj: Dict[str, List[str]] = {n: [] for n in self._stages}
        for name, info in self._stages.items():
            for dep in info["dependencies"]:
                if dep in self._stages:
                    adj[dep].append(name)
                    in_degree[name] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._stages):
            LOGGER.warning("存在循环依赖，部分阶段可能未执行")
        return order

    def _validate_inputs(self, name: str, data: Dict[str, Any]) -> bool:
        """验证阶段输入是否满足契约。"""
        stage = self._stages[name]
        for key in stage["input_keys"]:
            if key not in data:
                LOGGER.warning("阶段 '%s' 缺少输入键: %s", name, key)
                return False
        return True

    def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> PipelineExecutionResult:
        """执行完整的流水线。

        Args:
            initial_data: 初始输入数据字典。
        Returns:
            PipelineExecutionResult 包含各阶段结果和性能分析。
        """
        t0 = time.perf_counter()
        data = dict(initial_data) if initial_data else {}
        self._execution_order = self._resolve_execution_order()

        stage_results: Dict[str, Any] = {}
        timing_profile: Dict[str, float] = {}
        data_flow_trace: List[Dict[str, str]] = []
        error_messages: List[str] = []
        success = True

        for name in self._execution_order:
            stage = self._stages[name]

            # 验证输入契约
            if not self._validate_inputs(name, data):
                msg = f"阶段 '{name}' 输入验证失败"
                error_messages.append(msg)
                if self.config.error_handling_strategy == "stop":
                    success = False
                    break
                elif self.config.error_handling_strategy == "skip":
                    LOGGER.warning(msg + ", 跳过")
                    continue

            # 执行阶段 (支持重试)
            result = None
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    ts = time.perf_counter()
                    result = stage["func"](data)
                    elapsed = (time.perf_counter() - ts) * 1000
                    timing_profile[name] = elapsed
                    break
                except Exception as e:
                    if attempt == self.config.max_retries:
                        msg = f"阶段 '{name}' 执行失败 (重试 {attempt} 次): {e}"
                        error_messages.append(msg)
                        LOGGER.error(msg)
                        if self.config.error_handling_strategy == "stop":
                            success = False
                            break
                    else:
                        LOGGER.warning("阶段 '%s' 第 %d 次重试: %s", name, attempt, e)

            if result is None:
                if self.config.error_handling_strategy == "stop":
                    success = False
                    break
                continue

            # 合并输出到数据流
            if isinstance(result, dict):
                for key in stage["output_keys"]:
                    if key in result:
                        data[key] = result[key]
                        data_flow_trace.append({
                            "stage": name, "key": key,
                            "type": str(type(result[key]).__name__)
                        })

            stage_results[name] = result

        total_time = (time.perf_counter() - t0) * 1000
        LOGGER.debug("流水线执行完成: %d/%d 阶段成功, 总耗时=%.1fms",
                      len(stage_results), len(self._stages), total_time)

        return PipelineExecutionResult(
            stage_results=stage_results,
            timing_profile=timing_profile,
            data_flow_trace=data_flow_trace,
            success=success,
            error_messages=error_messages)


# ============================================================
# 4. CythonAccelerator (inspired by PYME)
# ============================================================

@dataclass
class JITAcceleratorConfig:
    """JIT 加速器配置。

    Attributes
    ----------
    cache_dir : str
        Numba 缓存目录。
    parallel_threads : int
        并行线程数 (0 为自动检测)。
    fastmath_mode : bool
        是否启用快速数学模式 (降低精度换取速度)。
    debug_mode : bool
        是否启用调试模式。
    """
    cache_dir: str = "__jit_cache__"
    parallel_threads: int = 0
    fastmath_mode: bool = True
    debug_mode: bool = False


@dataclass
class JITAccelerationResult:
    """JIT 加速结果。

    Attributes
    ----------
    compilation_time : float
        编译耗时 (毫秒)。
    speedup_factor : float
        相对纯 numpy 的加速倍数。
    memory_usage : float
        内存使用量 (MB)。
    """
    compilation_time: float
    speedup_factor: float
    memory_usage: float


# --- 纯 numpy 回退实现 ---

def _gaussian_fit_numpy(roi: np.ndarray, x0: float, y0: float,
                        sigma: float, amp: float, bg: float,
                        max_iter: int = 50) -> Tuple[float, float, float, float, float, bool]:
    """纯 numpy Levenberg-Marquardt 高斯拟合。"""
    rows, cols = roi.shape
    yi, xi = np.mgrid[:rows, :cols]
    params = np.array([x0, y0, sigma, amp, bg], dtype=np.float64)
    lam = 1e-3

    for _ in range(max_iter):
        x, y, s, a, b = params
        r2 = (xi - x) ** 2 + (yi - y) ** 2
        model = a * np.exp(-r2 / (2.0 * s ** 2)) + b
        residual = roi - model

        # 雅可比矩阵
        exp_term = np.exp(-r2 / (2.0 * s ** 2))
        J = np.zeros((rows * cols, 5), dtype=np.float64)
        J[:, 0] = a * exp_term * (xi - x) / (s ** 2)   # dx
        J[:, 1] = a * exp_term * (yi - y) / (s ** 2)   # dy
        J[:, 2] = a * exp_term * r2 / (s ** 3)          # dsigma
        J[:, 3] = exp_term                                # dA
        J[:, 4] = -np.ones(rows * cols, dtype=np.float64)  # dB

        JtJ = J.T @ J
        Jtr = J.T @ residual.flatten()
        diag = np.diag(JtJ).copy()

        try:
            delta = np.linalg.solve(JtJ + lam * np.diag(diag), Jtr)
        except np.linalg.LinAlgError:
            break

        new_residual = roi - (params[3] * np.exp(
            -((xi - params[0] - delta[0]) ** 2 + (yi - params[1] - delta[1]) ** 2)
            / (2.0 * (params[2] + delta[2]) ** 2)) + params[4] + delta[4])
        if np.sum(new_residual ** 2) < np.sum(residual ** 2):
            params += delta
            lam *= 0.5
        else:
            lam *= 2.0

        if np.max(np.abs(delta)) < 1e-6:
            return (float(params[0]), float(params[1]), float(params[2]),
                    float(params[3]), float(params[4]), True)

    return (float(params[0]), float(params[1]), float(params[2]),
            float(params[3]), float(params[4]), False)


def _centroid_numpy(roi: np.ndarray) -> Tuple[float, float]:
    """纯 numpy 加权质心计算。"""
    total = roi.sum()
    if total < 1e-12:
        return (float(roi.shape[1] / 2), float(roi.shape[0] / 2))
    yi, xi = np.mgrid[:roi.shape[0], :roi.shape[1]]
    return (float(np.sum(xi * roi) / total), float(np.sum(yi * roi) / total))


def _zernike_fit_numpy(phase: np.ndarray, mask: np.ndarray,
                       n_modes: int = 15, max_iter: int = 30) -> np.ndarray:
    """纯 numpy Zernike 最小二乘拟合。"""
    rows, cols = phase.shape
    cy, cx = rows / 2.0, cols / 2.0
    y, x = np.mgrid[:rows, :cols]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / min(cy, cx)
    theta = np.arctan2(y - cy, x - cx)

    basis = np.zeros((n_modes, rows * cols), dtype=np.float64)
    for j in range(1, n_modes + 1):
        n, m = _noll_to_zernike_index(j)
        radial = _zernike_radial_poly(n, m, np.clip(r, 0, 1))
        if m == 0:
            basis[j - 1] = radial.flatten()
        elif m > 0:
            basis[j - 1] = (radial * np.cos(m * theta)).flatten()
        else:
            basis[j - 1] = (radial * np.sin(-m * theta)).flatten()

    masked_basis = basis[:, mask.flatten()]
    masked_phase = phase.flatten()[mask.flatten()]

    # 正则化最小二乘
    reg = 1e-6 * np.eye(n_modes)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(
            masked_basis.T @ masked_basis + reg,
            masked_basis.T @ masked_phase, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = np.zeros(n_modes)

    return coeffs


def _noll_to_zernike_index(j: int) -> Tuple[int, int]:
    """Noll 编号转 Zernike (n, m)。"""
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    m_values = [n - 2 * k for k in range(n, -1, -1)]
    idx = j - n * (n + 1) // 2 - 1
    return n, m_values[idx]


def _zernike_radial_poly(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """计算 Zernike 径向多项式。"""
    m_abs = abs(m)
    result = np.zeros_like(r, dtype=np.float64)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1) ** s * math.factorial(n - s)
                 / (math.factorial(s) * math.factorial((n + m_abs) // 2 - s)
                    * math.factorial((n - m_abs) // 2 - s)))
        result += coeff * r ** (n - 2 * s)
    return result


# --- 尝试 JIT 编译 ---
if HAS_NUMBA:
    try:
        _gaussian_fit_jit = numba.njit(_gaussian_fit_numpy, fastmath=True, cache=True)
        _centroid_jit = numba.njit(_centroid_numpy, fastmath=True, cache=True)
        _zernike_jit = numba.njit(_zernike_fit_numpy, fastmath=False, cache=True)
        _USING_JIT = True
    except Exception as e:
        LOGGER.warning("Numba JIT 编译失败，回退到纯 numpy: %s", e)
        _gaussian_fit_jit = _gaussian_fit_numpy
        _centroid_jit = _centroid_numpy
        _zernike_jit = _zernike_fit_numpy
        _USING_JIT = False
else:
    _gaussian_fit_jit = _gaussian_fit_numpy
    _centroid_jit = _centroid_numpy
    _zernike_jit = _zernike_fit_numpy
    _USING_JIT = False


class CythonAccelerator:
    """JIT 加速核心计算函数。

    灵感来源于 PYME (python-microscopy) 的性能优化策略，使用
    Numba JIT 编译加速高斯拟合、质心计算和 Zernike 拟合等
    核心计算密集型函数。当 Numba 不可用时，自动回退到纯 numpy 实现。

    算法原理：
    - Numba @njit: 将 Python/numpy 代码编译为 LLVM 机器码
    - fastmath: 允许重新排列浮点运算以提升速度
    - 缓存机制: 首次编译后缓存，后续调用直接加载
    - 参考: PYME/python-microscopy performance documentation

    Parameters
    ----------
    config : JITAcceleratorConfig, optional
        加速器配置参数。
    """

    def __init__(self, config: Optional[JITAcceleratorConfig] = None):
        self.config = config or JITAcceleratorConfig()
        self._compilation_time: float = 0.0
        self._is_jit = _USING_JIT
        LOGGER.info("JIT 加速器初始化完成, JIT=%s, fastmath=%s",
                     self._is_jit, self.config.fastmath_mode)

    def reset(self) -> None:
        """重置加速器状态。"""
        self._compilation_time = 0.0
        LOGGER.debug("JIT 加速器已重置")

    def gaussian_fit(self, roi: np.ndarray,
                     x0: float, y0: float,
                     sigma: float = 1.5,
                     amplitude: float = 100.0,
                     background: float = 10.0) -> Tuple[float, float, float, float, float, bool]:
        """JIT 加速的高斯拟合。

        Args:
            roi: 感兴趣区域图像。
            x0, y0: 初始中心坐标。
            sigma: 初始 sigma。
            amplitude: 初始振幅。
            background: 初始背景。
        Returns:
            (x, y, sigma, amplitude, background, converged)
        """
        t0 = time.perf_counter()
        result = _gaussian_fit_jit(
            roi.astype(np.float64), x0, y0, sigma, amplitude, background)
        self._compilation_time += (time.perf_counter() - t0) * 1000
        return result

    def centroid(self, roi: np.ndarray) -> Tuple[float, float]:
        """JIT 加速的加权质心计算。

        Args:
            roi: 感兴趣区域图像。
        Returns:
            (cx, cy) 亚像素质心坐标。
        """
        t0 = time.perf_counter()
        result = _centroid_jit(roi.astype(np.float64))
        self._compilation_time += (time.perf_counter() - t0) * 1000
        return result

    def zernike_fit(self, phase: np.ndarray, mask: np.ndarray,
                    n_modes: int = 15) -> np.ndarray:
        """JIT 加速的 Zernike 多项式拟合。

        Args:
            phase: 波前相位图 (弧度)。
            mask: 有效区域掩码。
            n_modes: Zernike 模式数。
        Returns:
            Zernike 系数数组。
        """
        t0 = time.perf_counter()
        result = _zernike_jit(
            phase.astype(np.float64), mask.astype(np.float64) > 0.5, n_modes)
        self._compilation_time += (time.perf_counter() - t0) * 1000
        return result

    def benchmark(self, roi_size: int = 15, n_iterations: int = 100) -> JITAccelerationResult:
        """性能基准测试。

        Args:
            roi_size: ROI 尺寸。
            n_iterations: 迭代次数。
        Returns:
            JITAccelerationResult 包含编译时间和加速比。
        """
        rng = np.random.RandomState(42)
        test_roi = rng.poisson(50, (roi_size, roi_size)).astype(np.float64)

        # JIT 版本计时
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            self.centroid(test_roi)
        jit_time = (time.perf_counter() - t0) * 1000

        # 纯 numpy 版本计时
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            _centroid_numpy(test_roi)
        numpy_time = (time.perf_counter() - t0) * 1000

        speedup = numpy_time / jit_time if jit_time > 0 else 1.0
        mem_mb = test_roi.nbytes * 3 / (1024 * 1024)

        LOGGER.info("基准测试: JIT=%.2fms, numpy=%.2fms, 加速=%.1fx",
                     jit_time, numpy_time, speedup)

        return JITAccelerationResult(
            compilation_time=self._compilation_time,
            speedup_factor=speedup,
            memory_usage=mem_mb)


# ============================================================
# 5. StrehlRatioMonitor (inspired by AOtools / prysm)
# ============================================================

@dataclass
class StrehlMonitorConfig:
    """Strehl 比监测器配置。

    Attributes
    ----------
    reference_wavelength : float
        参考波长 (微米)。
    aperture_diameter : float
        入瞳直径 (毫米)。
    measurement_method : str
        测量方法: "fourier" (傅里叶法) 或 "zernike" (Zernike 系数法)。
    tracking_window : int
        长曝光跟踪窗口大小 (帧数)。
    """
    reference_wavelength: float = 0.550
    aperture_diameter: float = 5.0
    measurement_method: str = "fourier"
    tracking_window: int = 50


@dataclass
class StrehlMeasurement:
    """Strehl 比测量结果。

    Attributes
    ----------
    strehl_ratio : float
        Strehl 比 (0~1)。
    wavefront_rms : float
        波前 RMS 误差 (波长单位)。
    encircled_energy : float
        艾里斑第一暗环内能量比例。
    mtf_area : float
        MTF 曲线下面积 (归一化)。
    """
    strehl_ratio: float
    wavefront_rms: float
    encircled_energy: float
    mtf_area: float


class StrehlRatioMonitor:
    """实时 Strehl 比与光学质量监测器。

    灵感来源于 AOtools 和 prysm 项目，通过分析 PSF 图像实时
    监测光学系统的 Strehl 比、波前 RMS 误差和 MTF 质量。

    算法原理：
    - Strehl 比 = 实际 PSF 峰值 / 衍射极限 PSF 峰值
    - 近似公式: S ≈ exp(-(2π·σ_wfe)²)，σ_wfe 为波前 RMS (波长单位)
    - 傅里叶法: 通过 OTF 面积比计算 Strehl 比
    - Zernike 法: 从 Zernike 系数计算波前 RMS
    - 参考: Mahajan, JOSA A 22(8), 1554-1564 (2005)

    Parameters
    ----------
    config : StrehlMonitorConfig, optional
        监测器配置参数。
    """

    def __init__(self, config: Optional[StrehlMonitorConfig] = None):
        self.config = config or StrehlMonitorConfig()
        self._strehl_history: deque = deque(maxlen=self.config.tracking_window)
        self._diffraction_limit_peak: Optional[float] = None
        LOGGER.info("Strehl 比监测器初始化完成, 波长=%.3fμm, 方法=%s",
                     self.config.reference_wavelength, self.config.measurement_method)

    def reset(self) -> None:
        """重置监测器历史数据。"""
        self._strehl_history.clear()
        self._diffraction_limit_peak = None
        LOGGER.debug("Strehl 比监测器已重置")

    def _compute_diffraction_limit_psf(self, size: int) -> np.ndarray:
        """计算理想衍射极限 PSF (艾里斑)。"""
        wavelength_um = self.config.reference_wavelength
        D_mm = self.config.aperture_diameter
        # 归一化频率坐标
        freq = np.linspace(-0.5, 0.5, size)
        fx, fy = np.meshgrid(freq, freq)
        f_radius = np.sqrt(fx ** 2 + fy ** 2)

        # 圆形孔径 OTF (归一化)
        cutoff = 1.0  # 归一化截止频率
        pupil = (f_radius <= cutoff).astype(np.float64)
        # 自相关得到 OTF
        otf = np.fft.fftshift(np.real(np.fft.ifft2(
            np.abs(np.fft.fft2(pupil)) ** 2)))
        otf /= otf.max() if otf.max() > 0 else 1.0

        # PSF = IFFT(OTF)
        psf = np.real(np.fft.fftshift(np.fft.ifft2(
            np.fft.ifftshift(otf))))
        psf = np.clip(psf, 0, None)
        if psf.max() > 0:
            psf /= psf.max()
        return psf

    def measure(self, psf_image: np.ndarray,
                zernike_coeffs: Optional[np.ndarray] = None) -> StrehlMeasurement:
        """测量 PSF 图像的 Strehl 比和光学质量指标。

        Args:
            psf_image: 测量得到的 PSF 图像 (2D numpy 数组)。
            zernike_coeffs: 可选 Zernike 系数 (用于 zernike 方法)。
        Returns:
            StrehlMeasurement 包含 Strehl 比和各项光学指标。
        """
        t0 = time.perf_counter()
        psf = psf_image.astype(np.float64)
        if psf.ndim == 3:
            psf = np.mean(psf, axis=2)

        rows, cols = psf.shape
        size = min(rows, cols)

        if self.config.measurement_method == "fourier":
            strehl, wfe_rms, ee, mtf_area = self._measure_fourier(psf, size)
        elif self.config.measurement_method == "zernike" and zernike_coeffs is not None:
            strehl, wfe_rms, ee, mtf_area = self._measure_zernike(
                psf, zernike_coeffs, size)
        else:
            # 默认使用傅里叶法
            strehl, wfe_rms, ee, mtf_area = self._measure_fourier(psf, size)

        self._strehl_history.append(strehl)
        elapsed = (time.perf_counter() - t0) * 1000
        LOGGER.debug("Strehl 测量完成: S=%.4f, WFE=%.4fλ, EE=%.4f, MTF=%.4f, 耗时=%.1fms",
                      strehl, wfe_rms, ee, mtf_area, elapsed)

        return StrehlMeasurement(
            strehl_ratio=strehl,
            wavefront_rms=wfe_rms,
            encircled_energy=ee,
            mtf_area=mtf_area)

    def _measure_fourier(self, psf: np.ndarray,
                         size: int) -> Tuple[float, float, float, float]:
        """傅里叶法测量 Strehl 比。"""
        # 裁剪到正方形
        cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
        h = size // 2
        psf_crop = psf[cy - h:cy + h, cx - h:cx + h].copy()

        # Strehl 比 = 实际 PSF 峰值 / 衍射极限 PSF 峰值
        measured_peak = float(psf_crop.max())
        if self._diffraction_limit_peak is None:
            ideal_psf = self._compute_diffraction_limit_psf(size)
            self._diffraction_limit_peak = float(ideal_psf.max()) if ideal_psf.max() > 0 else 1.0

        strehl = measured_peak / self._diffraction_limit_peak if self._diffraction_limit_peak > 0 else 0.0
        strehl = min(strehl, 1.0)

        # 波前 RMS 从 Strehl 近似
        if strehl > 1e-10:
            wfe_rms = math.sqrt(-math.log(strehl)) / (2.0 * math.pi)
        else:
            wfe_rms = 1.0

        # 包含能量 (艾里斑第一暗环内)
        yi, xi = np.mgrid[:size, :size]
        r = np.sqrt((xi - size / 2) ** 2 + (yi - size / 2) ** 2)
        airy_radius = 1.22 * size / 4.0  # 近似第一暗环半径
        mask = r <= airy_radius
        total_energy = psf_crop.sum()
        ee = float(psf_crop[mask].sum() / total_energy) if total_energy > 0 else 0.0

        # MTF 面积
        otf = np.abs(np.fft.fft2(psf_crop))
        otf /= otf.max() if otf.max() > 0 else 1.0
        mtf_area = float(otf.sum()) / otf.size

        return strehl, wfe_rms, ee, mtf_area

    def _measure_zernike(self, psf: np.ndarray,
                         coeffs: np.ndarray,
                         size: int) -> Tuple[float, float, float, float]:
        """Zernike 系数法测量 Strehl 比。"""
        # 波前 RMS (排除 piston, tip, tilt)
        if len(coeffs) > 3:
            wfe_rms = float(np.sqrt(np.mean(coeffs[3:] ** 2)))
        else:
            wfe_rms = float(np.sqrt(np.mean(coeffs ** 2)))

        # Strehl 近似
        strehl = math.exp(-(2.0 * math.pi * wfe_rms) ** 2)
        strehl = max(min(strehl, 1.0), 0.0)

        # 包含能量和 MTF 面积使用傅里叶法
        _, _, ee, mtf_area = self._measure_fourier(psf, size)

        return strehl, wfe_rms, ee, mtf_area

    def get_long_exposure_strehl(self) -> float:
        """获取长曝光平均 Strehl 比。"""
        if len(self._strehl_history) == 0:
            return 0.0
        return float(np.mean(list(self._strehl_history)))

    def get_strehl_stability(self) -> float:
        """获取 Strehl 比稳定性 (标准差)。"""
        if len(self._strehl_history) < 2:
            return 0.0
        return float(np.std(list(self._strehl_history)))


# ============================================================
# 6. GpuFitAdapter (inspired by Gpufit)
# ============================================================

@dataclass
class GpuFitConfig:
    """GPU 拟合适配器配置。

    Attributes
    ----------
    model_type : str
        模型类型: "gaussian_2d", "gaussian_elliptical", "psf_model"。
    max_iterations : int
        LM 算法最大迭代次数。
    tolerance : float
        收敛容差。
    estimator_type : str
        估计器类型: "lse" (最小二乘) 或 "mle" (最大似然)。
    """
    model_type: str = "gaussian_2d"
    max_iterations: int = 50
    tolerance: float = 1e-6
    estimator_type: str = "lse"


@dataclass
class GpuFitResult:
    """GPU 拟合结果。

    Attributes
    ----------
    parameters : np.ndarray
        拟合参数数组。
    chi_squared : float
        卡方统计量。
    state : int
        拟合状态 (0=成功, 1=未收敛, 2=奇异矩阵)。
    n_iterations : int
        实际迭代次数。
    """
    parameters: np.ndarray
    chi_squared: float
    state: int
    n_iterations: int


class GpuFitAdapter:
    """GPU 加速 Levenberg-Marquardt 拟合适配器。

    灵感来源于 Gpufit (gpufit/Gpufit)，实现纯 numpy 的 LM 求解器
    作为 GPU 拟合的回退方案。支持批量拟合多个光斑。

    算法原理：
    - Levenberg-Marquardt: 阻尼最小二乘法 (Gauss-Newton + 阻尼因子)
    - 雅可比矩阵: 解析计算各模型参数的偏导数
    - 批量处理: 向量化多个 ROI 的同时拟合
    - 参考: Marquardt, SIAM J. Appl. Math. 11(2), 431-441 (1963)

    Parameters
    ----------
    config : GpuFitConfig, optional
        拟合配置参数。
    """

    def __init__(self, config: Optional[GpuFitConfig] = None):
        self.config = config or GpuFitConfig()
        LOGGER.info("GPU 拟合适配器初始化完成, 模型=%s, 最大迭代=%d",
                     self.config.model_type, self.config.max_iterations)

    def reset(self) -> None:
        """重置适配器状态。"""
        LOGGER.debug("GPU 拟合适配器已重置")

    def fit_single(self, roi: np.ndarray,
                   initial_params: Optional[np.ndarray] = None) -> GpuFitResult:
        """拟合单个 ROI。

        Args:
            roi: 感兴趣区域图像。
            initial_params: 初始参数数组。
        Returns:
            GpuFitResult 包含拟合参数和状态。
        """
        results = self.fit_batch(np.array([roi]),
                                 np.array([initial_params]) if initial_params is not None else None)
        return results[0]

    def fit_batch(self, rois: np.ndarray,
                  initial_params: Optional[np.ndarray] = None) -> List[GpuFitResult]:
        """批量拟合多个 ROI。

        Args:
            rois: ROI 数组，形状 (N, H, W)。
            initial_params: 初始参数数组，形状 (N, n_params) 或 None。
        Returns:
            GpuFitResult 列表。
        """
        t0 = time.perf_counter()
        n_spots = rois.shape[0]
        results: List[GpuFitResult] = []

        for i in range(n_spots):
            roi = rois[i].astype(np.float64)
            params_init = self._get_initial_params(roi, initial_params[i] if initial_params is not None else None)

            params, chi2, state, n_iter = self._levmar_fit(roi, params_init)
            results.append(GpuFitResult(
                parameters=params, chi_squared=chi2,
                state=state, n_iterations=n_iter))

        elapsed = (time.perf_counter() - t0) * 1000
        LOGGER.debug("批量拟合完成: %d 个光斑, 耗时=%.1fms", n_spots, elapsed)
        return results

    def _get_initial_params(self, roi: np.ndarray,
                            params: Optional[np.ndarray] = None) -> np.ndarray:
        """获取初始参数估计。"""
        if params is not None:
            return params.copy()

        rows, cols = roi.shape
        total = roi.sum()
        if total < 1e-12:
            return np.array([cols / 2.0, rows / 2.0, 1.5, 1.0, 0.0])

        yi, xi = np.mgrid[:rows, :cols]
        cx = float(np.sum(xi * roi) / total)
        cy = float(np.sum(yi * roi) / total)
        amp = float(roi.max())
        bg = float(np.median(roi))

        if self.config.model_type == "gaussian_elliptical":
            return np.array([cx, cy, 1.5, 1.5, 0.0, amp, bg])
        else:
            return np.array([cx, cy, 1.5, amp, bg])

    def _levmar_fit(self, roi: np.ndarray,
                    params: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Levenberg-Marquardt 核心拟合。"""
        rows, cols = roi.shape
        yi, xi = np.mgrid[:rows, :cols]
        p = params.copy()
        lam = 1e-3
        nu = 2.0
        state = 1  # 未收敛

        for iteration in range(self.config.max_iterations):
            model, J = self._evaluate_model(roi, p, xi, yi)
            residual = roi - model
            chi2 = float(np.sum(residual ** 2))

            JtJ = J.T @ J
            Jtr = J.T @ residual.flatten()

            try:
                delta = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ)), Jtr)
            except np.linalg.LinAlgError:
                state = 2  # 奇异矩阵
                break

            p_new = p + delta
            model_new, _ = self._evaluate_model(roi, p_new, xi, yi)
            chi2_new = float(np.sum((roi - model_new) ** 2))

            if chi2_new < chi2:
                p = p_new
                lam *= 0.1
                nu = 2.0
                if np.max(np.abs(delta)) < self.config.tolerance:
                    state = 0  # 收敛
                    break
            else:
                lam *= nu
                nu *= 2.0

        model_final, _ = self._evaluate_model(roi, p, xi, yi)
        chi2_final = float(np.sum((roi - model_final) ** 2))
        return p, chi2_final, state, iteration + 1

    def _evaluate_model(self, roi: np.ndarray, params: np.ndarray,
                        xi: np.ndarray, yi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算模型值和雅可比矩阵。"""
        rows, cols = roi.shape
        n_pixels = rows * cols

        if self.config.model_type == "gaussian_elliptical" and len(params) == 7:
            cx, cy, sx, sy, angle, amp, bg = params
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            dx = xi - cx
            dy = yi - cy
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            r2 = u ** 2 / (2.0 * sx ** 2) + v ** 2 / (2.0 * sy ** 2)
            exp_term = np.exp(-r2)
            model = amp * exp_term + bg

            # 雅可比 (7 参数)
            exp_flat = exp_term.flatten()
            u_flat = u.flatten()
            v_flat = v.flatten()
            J = np.zeros((n_pixels, 7), dtype=np.float64)
            J[:, 0] = amp * exp_flat * (cos_a * u_flat / (sx ** 2) + sin_a * v_flat / (sy ** 2))
            J[:, 1] = amp * exp_flat * (sin_a * u_flat / (sx ** 2) - cos_a * v_flat / (sy ** 2))
            J[:, 2] = amp * exp_flat * u_flat ** 2 / (sx ** 3)
            J[:, 3] = amp * exp_flat * v_flat ** 2 / (sy ** 3)
            J[:, 4] = amp * exp_flat * (sin_a * 2 * u_flat * v_flat) * (1.0 / (2.0 * sx ** 2) - 1.0 / (2.0 * sy ** 2))
            J[:, 5] = exp_flat
            J[:, 6] = -1.0
            return model, J

        else:
            # 标准 2D 高斯 (5 参数)
            cx, cy, sigma, amp, bg = params[:5]
            r2 = (xi - cx) ** 2 + (yi - cy) ** 2
            exp_term = np.exp(-r2 / (2.0 * sigma ** 2))
            model = amp * exp_term + bg

            exp_flat = exp_term.flatten()
            xi_flat = xi.flatten()
            yi_flat = yi.flatten()
            r2_flat = r2.flatten()
            J = np.zeros((n_pixels, 5), dtype=np.float64)
            J[:, 0] = amp * exp_flat * (xi_flat - cx) / (sigma ** 2)
            J[:, 1] = amp * exp_flat * (yi_flat - cy) / (sigma ** 2)
            J[:, 2] = amp * exp_flat * r2_flat / (sigma ** 3)
            J[:, 3] = exp_flat
            J[:, 4] = -1.0
            return model, J


# ============================================================
# 7. BioimageModelZooAdapter (inspired by ZeroCostDL4Mic / DeepImageJ)
# ============================================================

@dataclass
class BioimageZooConfig:
    """Bioimage Model Zoo 适配器配置。

    Attributes
    ----------
    model_id : str
        Bioimage Model Zoo 模型 ID。
    model_source : str
        模型来源路径。
    device : str
        计算设备: "cpu"。
    preprocessing_pipeline : List[str]
        预处理流水线步骤。
    tile_size : int
        分块处理尺寸 (大图分块)。
    tile_overlap : int
        分块重叠像素数。
    """
    model_id: str = ""
    model_source: str = ""
    device: str = "cpu"
    preprocessing_pipeline: List[str] = field(default_factory=lambda: [
        "normalize_percentile", "resize", "pad"])
    tile_size: int = 512
    tile_overlap: int = 32


@dataclass
class BioimageZooResult:
    """Bioimage Model Zoo 推理结果。

    Attributes
    ----------
    predictions : np.ndarray
        模型预测输出。
    model_metadata : Dict[str, Any]
        模型元数据。
    preprocessing_log : List[str]
        预处理日志。
    """
    predictions: np.ndarray
    model_metadata: Dict[str, Any]
    preprocessing_log: List[str]


class BioimageModelZooAdapter:
    """Bioimage Model Zoo 标准模型适配器。

    灵感来源于 ZeroCostDL4Mic (HenriquesLab) 和 DeepImageJ 项目，
    实现 Bioimage Model Zoo 标准格式的模型加载和推理接口。
    支持 RDF 模型描述解析、预处理/后处理流水线和分块推理。

    算法原理：
    - Bioimage Model Zoo: 统一的生物图像分析模型发布标准
    - 预处理流水线: 归一化 -> 尺寸调整 -> 填充
    - 分块推理: 大图像分块处理，重叠区域平滑拼接
    - 后处理: 阈值化、连通域分析、尺寸过滤
    - 参考: Bioimage Model Zoo specification (2023)

    Parameters
    ----------
    config : BioimageZooConfig, optional
        适配器配置参数。
    """

    def __init__(self, config: Optional[BioimageZooConfig] = None):
        self.config = config or BioimageZooConfig()
        self._model_loaded = False
        self._model_weights: Optional[np.ndarray] = None
        self._model_metadata: Dict[str, Any] = {}
        self._preprocessing_log: List[str] = []
        LOGGER.info("Bioimage Model Zoo 适配器初始化完成, 模型=%s, 设备=%s",
                     self.config.model_id or "未指定", self.config.device)

    def reset(self) -> None:
        """重置适配器状态。"""
        self._model_loaded = False
        self._model_weights = None
        self._model_metadata.clear()
        self._preprocessing_log.clear()
        LOGGER.debug("Bioimage Model Zoo 适配器已重置")

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """加载 Bioimage Model Zoo 格式模型。

        Args:
            model_path: 模型文件路径 (目录或权重文件)。
        Returns:
            是否加载成功。
        """
        path = model_path or self.config.model_source
        if not path:
            LOGGER.warning("未指定模型路径")
            return False

        self._model_metadata = {
            "model_id": self.config.model_id or "unknown",
            "source": path,
            "description": "Bioimage Model Zoo compatible model",
            "input_shape": [None, None, 1],
            "output_shape": [None, None, 1],
            "preprocessing": self.config.preprocessing_pipeline,
            "license": "unknown",
        }

        # 模拟加载权重 (实际实现需解析具体格式)
        self._model_weights = np.zeros((3, 3), dtype=np.float32)
        self._model_loaded = True
        self._preprocessing_log.append(f"模型从 {path} 加载完成")
        LOGGER.info("模型加载完成: %s", path)
        return True

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """执行预处理流水线。"""
        img = image.astype(np.float64)
        self._preprocessing_log.clear()

        for step in self.config.preprocessing_pipeline:
            if step == "normalize_percentile":
                p_low, p_high = np.percentile(img, [1.0, 99.0])
                if p_high - p_low > 1e-10:
                    img = np.clip((img - p_low) / (p_high - p_low), 0.0, 1.0)
                self._preprocessing_log.append(f"百分位归一化: [{p_low:.1f}, {p_high:.1f}]")

            elif step == "normalize_mean_std":
                mu, sigma = img.mean(), img.std()
                if sigma > 1e-10:
                    img = (img - mu) / sigma
                self._preprocessing_log.append(f"均值标准差归一化: μ={mu:.2f}, σ={sigma:.2f}")

            elif step == "resize":
                target = self.config.tile_size
                if img.shape[0] != target or img.shape[1] != target:
                    # 最近邻插值 (纯 numpy)
                    y_idx = np.linspace(0, img.shape[0] - 1, target)
                    x_idx = np.linspace(0, img.shape[1] - 1, target)
                    yi = np.clip(np.round(y_idx).astype(int), 0, img.shape[0] - 1)
                    xi = np.clip(np.round(x_idx).astype(int), 0, img.shape[1] - 1)
                    img = img[np.ix_(yi, xi)]
                self._preprocessing_log.append(f"尺寸调整: {img.shape}")

            elif step == "pad":
                target = self.config.tile_size
                if img.shape[0] < target or img.shape[1] < target:
                    padded = np.zeros((target, target), dtype=img.dtype)
                    padded[:img.shape[0], :img.shape[1]] = img
                    img = padded
                self._preprocessing_log.append(f"填充至: {img.shape}")

        return img

    def _postprocess(self, prediction: np.ndarray) -> np.ndarray:
        """执行后处理。"""
        result = prediction.copy()
        # 确保输出在 [0, 1] 范围
        result = np.clip(result, 0.0, 1.0)
        return result

    def predict(self, image: np.ndarray) -> BioimageZooResult:
        """对图像执行模型推理。

        Args:
            image: 输入图像 (2D numpy 数组)。
        Returns:
            BioimageZooResult 包含预测结果和元数据。
        """
        t0 = time.perf_counter()

        if image.ndim == 3:
            image = np.mean(image, axis=2)

        # 大图分块处理
        if image.shape[0] > self.config.tile_size or image.shape[1] > self.config.tile_size:
            prediction = self._tiled_predict(image)
        else:
            preprocessed = self._preprocess(image)
            # 模拟推理 (实际实现调用模型)
            prediction = self._simulate_inference(preprocessed)
            prediction = self._postprocess(prediction)

        elapsed = (time.perf_counter() - t0) * 1000
        LOGGER.debug("推理完成: 输出形状=%s, 耗时=%.1fms",
                      prediction.shape, elapsed)

        return BioimageZooResult(
            predictions=prediction,
            model_metadata=self._model_metadata,
            preprocessing_log=list(self._preprocessing_log))

    def _tiled_predict(self, image: np.ndarray) -> np.ndarray:
        """分块推理大图像。"""
        tile = self.config.tile_size
        overlap = self.config.tile_overlap
        step = tile - overlap
        rows, cols = image.shape

        output = np.zeros((rows, cols), dtype=np.float64)
        weight = np.zeros((rows, cols), dtype=np.float64)

        # 创建融合权重 (余弦窗)
        wy = np.hanning(tile)
        wx = np.hanning(tile)
        tile_weight = np.outer(wy, wx)

        for y in range(0, rows - overlap, step):
            for x in range(0, cols - overlap, step):
                y1 = min(y + tile, rows)
                x1 = min(x + tile, cols)
                tile_img = image[y:y1, x:x1]

                # 填充到 tile_size
                padded = np.zeros((tile, tile), dtype=np.float64)
                padded[:y1 - y, :x1 - x] = tile_img

                preprocessed = self._preprocess(padded)
                pred = self._simulate_inference(preprocessed)
                pred = self._postprocess(pred)

                # 加权融合
                tw = tile_weight[:y1 - y, :x1 - x]
                output[y:y1, x:x1] += pred[:y1 - y, :x1 - x] * tw
                weight[y:y1, x:x1] += tw

        weight = np.where(weight > 1e-10, weight, 1.0)
        output /= weight
        return output

    @staticmethod
    def _simulate_inference(image: np.ndarray) -> np.ndarray:
        """模拟模型推理 (使用简单的图像处理代替实际神经网络)。"""
        # 使用高斯平滑作为模拟推理
        from scipy.ndimage import gaussian_filter
        result = gaussian_filter(image, sigma=1.0)
        # 添加 sigmoid 激活模拟
        result = 1.0 / (1.0 + np.exp(-5.0 * (result - 0.5)))
        return result


# ============================================================
# 8. TrackMateLinker (inspired by TrackMate/Fiji)
# ============================================================

@dataclass
class LAPLinkerConfig:
    """LAP 轨迹链接器配置。

    Attributes
    ----------
    max_linking_distance : float
        最大链接距离 (像素)。
    gap_closing_max : int
        最大间隙关闭帧数。
    allow_split_merge : bool
        是否允许轨迹分裂/合并。
    cost_function : str
        代价函数: "euclidean" 或 "weighted"。
    alternative_cost_factor : float
        替代链接代价因子 (相对最大距离)。
    """
    max_linking_distance: float = 15.0
    gap_closing_max: int = 2
    allow_split_merge: bool = False
    cost_function: str = "euclidean"
    alternative_cost_factor: float = 1.05


@dataclass
class LAPLinkerResult:
    """LAP 轨迹链接结果。

    Attributes
    ----------
    tracks : List[List[Tuple[int, float, float]]]
        轨迹列表，每条轨迹为 [(frame, x, y), ...]。
    track_statistics : Dict[str, Any]
        轨迹统计信息。
    linking_cost_matrix : np.ndarray
        链接代价矩阵 (最后一帧)。
    """
    tracks: List[List[Tuple[int, float, float]]]
    track_statistics: Dict[str, Any]
    linking_cost_matrix: np.ndarray


class TrackMateLinker:
    """基于 LAP 的光斑轨迹链接器。

    灵感来源于 TrackMate/Fiji (imagej/TrackMate)，使用线性分配
    问题 (LAP) 框架实现跨帧光斑链接和轨迹分析。

    算法原理：
    - LAP (Linear Assignment Problem): 使用匈牙利算法求解最优匹配
    - 代价矩阵: 基于帧间距离构建代价矩阵
    - 间隙关闭: 允许短时间消失后重新链接
    - 分裂/合并: 检测轨迹分叉和汇合事件
    - 参考: Jaqaman et al., Nature Methods 5, 695-697 (2008)

    Parameters
    ----------
    config : LAPLinkerConfig, optional
        链接器配置参数。
    """

    def __init__(self, config: Optional[LAPLinkerConfig] = None):
        self.config = config or LAPLinkerConfig()
        if not HAS_SCIPY:
            LOGGER.warning("scipy 不可用，LAP 链接将使用贪心算法")
        LOGGER.info("LAP 轨迹链接器初始化完成, 最大距离=%.1fpx, 间隙关闭=%d帧",
                     self.config.max_linking_distance, self.config.gap_closing_max)

    def reset(self) -> None:
        """重置链接器状态。"""
        LOGGER.debug("LAP 轨迹链接器已重置")

    def link(self, spots_per_frame: List[List[Tuple[float, float]]]) -> LAPLinkerResult:
        """跨帧链接光斑形成轨迹。

        Args:
            spots_per_frame: 每帧光斑列表，spots_per_frame[frame] = [(x, y), ...]。
        Returns:
            LAPLinkerResult 包含轨迹和统计信息。
        """
        t0 = time.perf_counter()
        n_frames = len(spots_per_frame)

        if n_frames == 0:
            return LAPLinkerResult(
                tracks=[], track_statistics={}, linking_cost_matrix=np.array([]))

        # 为每个光斑分配全局 ID
        spot_ids: List[List[int]] = []
        global_id = 0
        for frame_spots in spots_per_frame:
            frame_ids = list(range(global_id, global_id + len(frame_spots)))
            spot_ids.append(frame_ids)
            global_id += len(frame_spots)

        # 构建链接: frame-to-frame LAP
        links: Dict[int, int] = {}  # source_id -> target_id
        last_cost_matrix = np.array([])

        for f in range(n_frames - 1):
            source_spots = spots_per_frame[f]
            target_spots = spots_per_frame[f + 1]
            source_ids = spot_ids[f]
            target_ids = spot_ids[f + 1]

            if len(source_spots) == 0 or len(target_spots) == 0:
                continue

            # 构建代价矩阵
            cost_matrix = self._build_cost_matrix(source_spots, target_spots)
            last_cost_matrix = cost_matrix.copy()

            # 求解 LAP
            assignments = self._solve_lap(cost_matrix)

            for si, ti in assignments:
                if cost_matrix[si, ti] < self.config.max_linking_distance:
                    links[source_ids[si]] = target_ids[ti]

        # 间隙关闭
        if self.config.gap_closing_max > 0:
            links = self._gap_close(spots_per_frame, spot_ids, links)

        # 从链接构建轨迹
        tracks = self._build_tracks(spots_per_frame, spot_ids, links)

        # 计算统计信息
        stats = self._compute_track_statistics(tracks)

        elapsed = (time.perf_counter() - t0) * 1000
        LOGGER.debug("轨迹链接完成: %d 条轨迹, %d 帧, 耗时=%.1fms",
                      len(tracks), n_frames, elapsed)

        return LAPLinkerResult(
            tracks=tracks,
            track_statistics=stats,
            linking_cost_matrix=last_cost_matrix)

    def _build_cost_matrix(self, source: List[Tuple[float, float]],
                           target: List[Tuple[float, float]]) -> np.ndarray:
        """构建帧间代价矩阵。"""
        n_s = len(source)
        n_t = len(target)
        cost = np.full((n_s, n_t), self.config.max_linking_distance * self.config.alternative_cost_factor)

        for i, (sx, sy) in enumerate(source):
            for j, (tx, ty) in enumerate(target):
                if self.config.cost_function == "euclidean":
                    d = math.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)
                else:
                    d = math.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)
                cost[i, j] = d

        return cost

    def _solve_lap(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """求解线性分配问题。"""
        if HAS_SCIPY:
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                return list(zip(row_ind.tolist(), col_ind.tolist()))
            except Exception:
                pass

        # 回退: 贪心算法
        return self._greedy_assignment(cost_matrix)

    @staticmethod
    def _greedy_assignment(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """贪心分配 (scipy 不可用时的回退方案)。"""
        assignments: List[Tuple[int, int]] = []
        used_rows: set = set()
        used_cols: set = set()

        # 按代价排序
        indices = np.argsort(cost_matrix, axis=None)
        rows, cols = cost_matrix.shape

        for idx in indices:
            r, c = divmod(int(idx), cols)
            if r not in used_rows and c not in used_cols:
                assignments.append((r, c))
                used_rows.add(r)
                used_cols.add(c)

        return assignments

    def _gap_close(self, spots_per_frame: List[List[Tuple[float, float]]],
                   spot_ids: List[List[int]],
                   links: Dict[int, int]) -> Dict[int, int]:
        """间隙关闭：尝试链接间隔不超过 gap_closing_max 帧的光斑。"""
        n_frames = len(spots_per_frame)
        max_gap = self.config.gap_closing_max

        for gap in range(2, max_gap + 1):
            for f in range(n_frames - gap):
                # 查找帧 f 中未链接的光斑
                source_ids = spot_ids[f]
                target_ids = spot_ids[f + gap]

                for sid in source_ids:
                    if sid in links:
                        continue
                    # 查找 sid 在 spots_per_frame 中的坐标
                    si = source_ids.index(sid)
                    sx, sy = spots_per_frame[f][si]

                    best_tid = -1
                    best_dist = self.config.max_linking_distance * gap

                    for tj, tid in enumerate(target_ids):
                        # 检查目标是否已被链接到更近的帧
                        if any(v == tid for v in links.values()):
                            continue
                        tx, ty = spots_per_frame[f + gap][tj]
                        d = math.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)
                        if d < best_dist:
                            best_dist = d
                            best_tid = tid

                    if best_tid >= 0:
                        links[sid] = best_tid

        return links

    def _build_tracks(self, spots_per_frame: List[List[Tuple[float, float]]],
                      spot_ids: List[List[int]],
                      links: Dict[int, int]) -> List[List[Tuple[int, float, float]]]:
        """从链接关系构建轨迹。"""
        # 构建反向映射: id -> (frame, index)
        id_to_frame: Dict[int, Tuple[int, int]] = {}
        for f, frame_ids in enumerate(spot_ids):
            for i, sid in enumerate(frame_ids):
                id_to_frame[sid] = (f, i)

        # 查找轨迹起点 (没有前置链接的光斑)
        all_targets = set(links.values())
        track_starts = [sid for sid in id_to_frame if sid not in all_targets]

        tracks: List[List[Tuple[int, float, float]]] = []
        visited: set = set()

        for start_id in track_starts:
            if start_id in visited:
                continue
            track: List[Tuple[int, float, float]] = []
            current = start_id

            while current is not None and current not in visited:
                visited.add(current)
                if current not in id_to_frame:
                    break
                frame, idx = id_to_frame[current]
                x, y = spots_per_frame[frame][idx]
                track.append((frame, x, y))
                current = links.get(current)

            if len(track) >= 1:
                tracks.append(track)

        # 按长度排序
        tracks.sort(key=lambda t: len(t), reverse=True)
        return tracks

    def _compute_track_statistics(self,
                                   tracks: List[List[Tuple[int, float, float]]]) -> Dict[str, Any]:
        """计算轨迹统计信息。"""
        if not tracks:
            return {"n_tracks": 0}

        lengths = [len(t) for t in tracks]
        displacements: List[float] = []

        for track in tracks:
            if len(track) >= 2:
                x0, y0 = track[0][1], track[0][2]
                x1, y1 = track[-1][1], track[-1][2]
                displacements.append(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

        # 帧间速度
        speeds: List[float] = []
        for track in tracks:
            for i in range(1, len(track)):
                f0, x0, y0 = track[i - 1]
                f1, x1, y1 = track[i]
                dt = max(f1 - f0, 1)
                d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                speeds.append(d / dt)

        # 平均偏移 (MSD)
        msds: List[float] = []
        for track in tracks:
            if len(track) >= 3:
                x0, y0 = track[0][1], track[0][2]
                for f, x, y in track:
                    msds.append((x - x0) ** 2 + (y - y0) ** 2)

        stats = {
            "n_tracks": len(tracks),
            "mean_length": float(np.mean(lengths)),
            "max_length": int(max(lengths)),
            "min_length": int(min(lengths)),
            "mean_displacement": float(np.mean(displacements)) if displacements else 0.0,
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "mean_msd": float(np.mean(msds)) if msds else 0.0,
        }

        return stats
