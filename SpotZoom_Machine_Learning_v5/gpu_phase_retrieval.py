"""
GPU 加速相位恢复引擎 (GPUPhaseRetrieval)

基于 slmsuite (https://github.com/holodyne/slmsuite) 的 GPU 加速迭代相位恢复算法，
为 SpotZoom 的 SLM 全息光斑生成提供高性能计算能力。

灵感来源:
- slmsuite: GPU加速SLM控制与相位恢复 (https://github.com/holodyne/slmsuite)
- GS Algorithm: Gerchberg-Saxton 迭代相位恢复 (Gerchberg & Saxton, 1972)
- WGS: 加权 Gerchberg-Saxton (Fienup, 1993)
- MRAF: Mixed Region Amplitude Freedom (Mansuripur, 1989)

算法原理:
  1. Gerchberg-Saxton (GS): 在远场约束和近场约束之间迭代，
     逐步收敛到目标相位分布
  2. 加权 GS (WGS): 对远场振幅约束引入权重因子，加速收敛
  3. 混合区域振幅自由 (MRAF): 在远场中忽略未使用区域，
     仅优化感兴趣区域的性能
  4. 相位驻定 WGS: 在迭代过程中逐步收紧振幅约束

与现有模块的关系:
  - 增强 v3/slm_holographic_spot_generator.py 的相位恢复能力
  - 增强 v1/phase_retrieval_analyzer.py 的 GS 算法性能
  - 与 v1/differentiable_ray_tracer.py 协同实现端到端可微光学设计

外部依赖: numpy, scipy (可选), cupy (可选 GPU加速)
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class PhaseAlgorithm(Enum):
    """相位恢复算法。"""
    GS = "gs"                     # Gerchberg-Saxton
    WGS = "wgs"                   # 加权 Gerchberg-Saxton
    MRAF = "mraf"                 # Mixed Region Amplitude Freedom
    ER = "er"                     # Error Reduction
    HIO = "hio"                   # Hybrid Input-Output
    RAAR = "raar"                 # Relaxed Averaged Alternating Reflections


class DeviceType(Enum):
    """计算设备类型。"""
    CPU = "cpu"
    GPU = "gpu"                   # 需要 CuPy


@dataclass
class PhaseRetrievalConfig:
    """相位恢复配置。"""
    # 算法选择
    algorithm: PhaseAlgorithm = PhaseAlgorithm.GS

    # 计算设备
    device: DeviceType = DeviceType.CPU

    # 迭代参数
    max_iterations: int = 100
    convergence_threshold: float = 1e-6   # 收敛阈值（MSE 变化量）
    early_stop_patience: int = 10         # 早停耐心值

    # GS/WGS 参数
    wgs_beta: float = 0.9                # WGS 权重因子 [0, 1]
    wgs_beta_schedule: str = "constant"  # constant / linear / exponential

    # HIO 参数
    hio_beta: float = 0.8                # HIO 反馈参数

    # RAAR 参数
    raar_beta: float = 0.9               # RAAR 混合参数

    # MRAF 参数
    mraf_signal_region: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2)

    # 初始相位
    initial_phase: str = "random"        # random / zeros / linear

    # 输出参数
    compute_strehl: bool = True          # 计算 Strehl 比
    compute_efficiency: bool = True      # 计算衍射效率
    return_history: bool = False         # 返回迭代历史


@dataclass
class PhaseRetrievalResult:
    """相位恢复结果。"""
    # 核心输出
    phase: np.ndarray = field(default_factory=lambda: np.array([]))
    amplitude: np.ndarray = field(default_factory=lambda: np.array([]))

    # 质量指标
    strehl_ratio: float = 0.0
    diffraction_efficiency: float = 0.0
    rms_phase_error: float = 0.0
    final_mse: float = 0.0

    # 收敛信息
    iterations_used: int = 0
    converged: bool = False
    convergence_iteration: int = -1

    # 性能信息
    processing_time_ms: float = 0.0
    device_used: str = "cpu"
    algorithm_used: str = ""

    # 迭代历史
    mse_history: List[float] = field(default_factory=list)
    strehl_history: List[float] = field(default_factory=list)


class GPUPhaseRetrieval:
    """GPU 加速相位恢复引擎。

    支持多种迭代相位恢复算法，可自动利用 CuPy 进行 GPU 加速。
    当 CuPy 不可用时，自动回退到 NumPy CPU 计算。

    Parameters
    ----------
    config : PhaseRetrievalConfig
        相位恢复配置。
    """

    def __init__(self, config: Optional[PhaseRetrievalConfig] = None):
        self.config = config or PhaseRetrievalConfig()
        self._xp = np  # 默认 NumPy，GPU 可用时切换为 CuPy
        self._use_gpu = False

        if self.config.device == DeviceType.GPU:
            self._init_gpu()

    def _init_gpu(self):
        """初始化 GPU 计算。"""
        try:
            import cupy as cp
            self._xp = cp
            self._use_gpu = True
            logger.info("GPU 加速已启用 (CuPy)")
        except ImportError:
            logger.warning("CuPy 不可用，回退到 NumPy CPU 计算")
            self._xp = np
            self._use_gpu = False

    def retrieve(
        self,
        target_amplitude: np.ndarray,
        input_amplitude: Optional[np.ndarray] = None,
    ) -> PhaseRetrievalResult:
        """执行相位恢复。

        Parameters
        ----------
        target_amplitude : np.ndarray
            目标远场振幅分布（期望的光斑图案）。
        input_amplitude : np.ndarray, optional
            输入近场振幅分布（SLM 孔径函数）。默认均匀照明。

        Returns
        -------
        PhaseRetrievalResult
            相位恢复结果。
        """
        result = PhaseRetrievalResult(
            algorithm_used=self.config.algorithm.value,
            device_used="gpu" if self._use_gpu else "cpu",
        )

        t0 = time.perf_counter()
        xp = self._xp

        # 转移数据到计算设备
        target = xp.asarray(target_amplitude, dtype=xp.float64)
        if input_amplitude is not None:
            input_amp = xp.asarray(input_amplitude, dtype=xp.float64)
        else:
            input_amp = xp.ones_like(target)

        # 初始相位
        phase = self._init_phase(target.shape)

        # 执行算法
        if self.config.algorithm == PhaseAlgorithm.GS:
            phase, mse_hist = self._gerchberg_saxton(phase, input_amp, target)
        elif self.config.algorithm == PhaseAlgorithm.WGS:
            phase, mse_hist = self._weighted_gs(phase, input_amp, target)
        elif self.config.algorithm == PhaseAlgorithm.MRAF:
            phase, mse_hist = self._mraf(phase, input_amp, target)
        elif self.config.algorithm == PhaseAlgorithm.HIO:
            phase, mse_hist = self._hio(phase, input_amp, target)
        elif self.config.algorithm == PhaseAlgorithm.RAAR:
            phase, mse_hist = self._raar(phase, input_amp, target)
        else:
            phase, mse_hist = self._gerchberg_saxton(phase, input_amp, target)

        # 计算最终远场
        far_field = xp.fft.fftshift(xp.fft.fft2(input_amp * xp.exp(1j * phase)))
        far_amplitude = xp.abs(far_field)

        # 计算质量指标
        if self.config.compute_strehl:
            result.strehl_ratio = float(self._compute_strehl(far_amplitude, target))
        if self.config.compute_efficiency:
            result.diffraction_efficiency = float(
                self._compute_efficiency(far_amplitude, target)
            )

        result.final_mse = float(mse_hist[-1]) if mse_hist else 0.0
        result.iterations_used = len(mse_hist)
        result.mse_history = [float(m) for m in mse_hist]

        # 检查收敛
        if len(mse_hist) > self.config.early_stop_patience:
            recent = mse_hist[-self.config.early_stop_patience:]
            if max(recent) - min(recent) < self.config.convergence_threshold:
                result.converged = True
                result.convergence_iteration = len(mse_hist) - self.config.early_stop_patience

        # 转移结果回 CPU
        result.phase = xp.asnumpy(phase) if self._use_gpu else phase
        result.amplitude = xp.asnumpy(far_amplitude) if self._use_gpu else far_amplitude
        result.processing_time_ms = (time.perf_counter() - t0) * 1000

        return result

    def _init_phase(self, shape: Tuple[int, ...]) -> "np.ndarray":
        """初始化相位分布。"""
        xp = self._xp
        if self.config.initial_phase == "random":
            return xp.random.uniform(0, 2 * np.pi, shape)
        elif self.config.initial_phase == "zeros":
            return xp.zeros(shape)
        elif self.config.initial_phase == "linear":
            y, x = xp.mgrid[:shape[0], :shape[1]]
            return xp.linspace(0, 2 * np.pi, shape[0]).reshape(-1, 1) * xp.ones((1, shape[1]))
        return xp.zeros(shape)

    def _gerchberg_saxton(
        self, phase: "np.ndarray", input_amp: "np.ndarray", target: "np.ndarray"
    ) -> Tuple["np.ndarray", List[float]]:
        """Gerchberg-Saxton 算法。"""
        xp = self._xp
        mse_history = []
        prev_mse = float('inf')

        for i in range(self.config.max_iterations):
            # 近场 -> 远场
            field = input_amp * xp.exp(1j * phase)
            far_field = xp.fft.fftshift(xp.fft.fft2(field))
            far_amp = xp.abs(far_field)

            # 远场约束：替换振幅为目标振幅
            mse = float(xp.mean((far_amp - target) ** 2))
            mse_history.append(mse)

            # 早停检查
            if abs(prev_mse - mse) < self.config.convergence_threshold:
                break
            prev_mse = mse

            far_field_constrained = target * xp.exp(1j * xp.angle(far_field))

            # 远场 -> 近场
            near_field = xp.fft.ifft2(xp.fft.ifftshift(far_field_constrained))
            phase = xp.angle(near_field)

        return phase, mse_history

    def _weighted_gs(
        self, phase: "np.ndarray", input_amp: "np.ndarray", target: "np.ndarray"
    ) -> Tuple["np.ndarray", List[float]]:
        """加权 Gerchberg-Saxton 算法。"""
        xp = self._xp
        mse_history = []
        beta = self.config.wgs_beta
        prev_mse = float('inf')

        for i in range(self.config.max_iterations):
            # 权重调度
            if self.config.wgs_beta_schedule == "linear":
                b = beta * (1 - i / self.config.max_iterations)
            elif self.config.wgs_beta_schedule == "exponential":
                b = beta * np.exp(-3 * i / self.config.max_iterations)
            else:
                b = beta

            field = input_amp * xp.exp(1j * phase)
            far_field = xp.fft.fftshift(xp.fft.fft2(field))
            far_amp = xp.abs(far_field)

            mse = float(xp.mean((far_amp - target) ** 2))
            mse_history.append(mse)

            if abs(prev_mse - mse) < self.config.convergence_threshold:
                break
            prev_mse = mse

            # 加权约束：部分保留原始振幅
            constrained_amp = b * target + (1 - b) * far_amp
            far_field_constrained = constrained_amp * xp.exp(1j * xp.angle(far_field))

            near_field = xp.fft.ifft2(xp.fft.ifftshift(far_field_constrained))
            phase = xp.angle(near_field)

        return phase, mse_history

    def _mraf(
        self, phase: "np.ndarray", input_amp: "np.ndarray", target: "np.ndarray"
    ) -> Tuple["np.ndarray", List[float]]:
        """Mixed Region Amplitude Freedom 算法。"""
        xp = self._xp
        mse_history = []
        prev_mse = float('inf')

        # 创建信号区域掩模
        signal_mask = xp.ones_like(target)
        if self.config.mraf_signal_region is not None:
            x1, y1, x2, y2 = self.config.mraf_signal_region
            signal_mask[:] = 0
            signal_mask[y1:y2, x1:x2] = 1

        for i in range(self.config.max_iterations):
            field = input_amp * xp.exp(1j * phase)
            far_field = xp.fft.fftshift(xp.fft.fft2(field))
            far_amp = xp.abs(far_field)

            # 仅在信号区域计算 MSE
            signal_mse = float(xp.mean(
                (far_amp * signal_mask - target * signal_mask) ** 2
            ))
            mse_history.append(signal_mse)

            if abs(prev_mse - signal_mse) < self.config.convergence_threshold:
                break
            prev_mse = signal_mse

            # MRAF 约束：信号区域强制振幅，非信号区域自由
            constrained_amp = signal_mask * target + (1 - signal_mask) * far_amp
            far_field_constrained = constrained_amp * xp.exp(1j * xp.angle(far_field))

            near_field = xp.fft.ifft2(xp.fft.ifftshift(far_field_constrained))
            phase = xp.angle(near_field)

        return phase, mse_history

    def _hio(
        self, phase: "np.ndarray", input_amp: "np.ndarray", target: "np.ndarray"
    ) -> Tuple["np.ndarray", List[float]]:
        """Hybrid Input-Output 算法。"""
        xp = self._xp
        mse_history = []
        prev_mse = float('inf')
        beta = self.config.hio_beta
        prev_phase = phase.copy()

        for i in range(self.config.max_iterations):
            field = input_amp * xp.exp(1j * phase)
            far_field = xp.fft.fftshift(xp.fft.fft2(field))
            far_amp = xp.abs(far_field)

            mse = float(xp.mean((far_amp - target) ** 2))
            mse_history.append(mse)

            if abs(prev_mse - mse) < self.config.convergence_threshold:
                break
            prev_mse = mse

            far_field_constrained = target * xp.exp(1j * xp.angle(far_field))
            near_field = xp.fft.ifft2(xp.fft.ifftshift(far_field_constrained))

            # HIO 更新：违反约束时使用反馈
            violation = xp.abs(near_field) > input_amp
            new_phase = xp.where(
                violation,
                xp.angle(near_field) - beta * (xp.angle(near_field) - prev_phase),
                xp.angle(near_field)
            )
            prev_phase = phase.copy()
            phase = new_phase

        return phase, mse_history

    def _raar(
        self, phase: "np.ndarray", input_amp: "np.ndarray", target: "np.ndarray"
    ) -> Tuple["np.ndarray", List[float]]:
        """Relaxed Averaged Alternating Reflections 算法。"""
        xp = self._xp
        mse_history = []
        prev_mse = float('inf')
        beta = self.config.raar_beta

        for i in range(self.config.max_iterations):
            field = input_amp * xp.exp(1j * phase)
            far_field = xp.fft.fftshift(xp.fft.fft2(field))
            far_amp = xp.abs(far_field)

            mse = float(xp.mean((far_amp - target) ** 2))
            mse_history.append(mse)

            if abs(prev_mse - mse) < self.config.convergence_threshold:
                break
            prev_mse = mse

            # RAAR 混合
            far_field_constrained = target * xp.exp(1j * xp.angle(far_field))
            near_field = xp.fft.ifft2(xp.fft.ifftshift(far_field_constrained))

            # 反射操作
            reflected = 2 * input_amp * xp.exp(1j * xp.angle(near_field)) - near_field
            # 平均
            mixed = beta * reflected + (1 - beta) * near_field
            phase = xp.angle(mixed)

        return phase, mse_history

    @staticmethod
    def _compute_strehl(computed: "np.ndarray", target: "np.ndarray") -> float:
        """计算 Strehl 比。"""
        peak_computed = float(np.max(computed ** 2))
        peak_target = float(np.max(target ** 2))
        if peak_target < 1e-10:
            return 0.0
        return peak_computed / peak_target

    @staticmethod
    def _compute_efficiency(computed: "np.ndarray", target: "np.ndarray") -> float:
        """计算衍射效率。"""
        signal_region = target > 0.1 * np.max(target)
        if not np.any(signal_region):
            return 0.0
        power_in_signal = float(np.sum(computed[signal_region] ** 2))
        total_power = float(np.sum(computed ** 2))
        if total_power < 1e-10:
            return 0.0
        return power_in_signal / total_power


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试 GS 算法
    config = PhaseRetrievalConfig(
        algorithm=PhaseAlgorithm.GS,
        max_iterations=50,
    )
    engine = GPUPhaseRetrieval(config)

    # 创建目标光斑图案
    target = np.zeros((128, 128))
    target[60:68, 60:68] = 1.0  # 单光斑

    result = engine.retrieve(target)
    print(f"GS: Strehl={result.strehl_ratio:.4f}, 效率={result.diffraction_efficiency:.4f}, "
          f"迭代={result.iterations_used}, 时间={result.processing_time_ms:.1f}ms")

    # 测试 WGS 算法
    config_wgs = PhaseRetrievalConfig(
        algorithm=PhaseAlgorithm.WGS,
        max_iterations=50,
        wgs_beta=0.9,
    )
    engine_wgs = GPUPhaseRetrieval(config_wgs)
    result_wgs = engine_wgs.retrieve(target)
    print(f"WGS: Strehl={result_wgs.strehl_ratio:.4f}, 效率={result_wgs.diffraction_efficiency:.4f}, "
          f"迭代={result_wgs.iterations_used}, 时间={result_wgs.processing_time_ms:.1f}ms")
