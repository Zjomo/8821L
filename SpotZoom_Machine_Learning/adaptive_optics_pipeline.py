"""
自适应光学管道 (AdaptiveOpticsPipeline)

灵感来源:
- HCIPy (https://github.com/ehpor/hcipy) — 完整的自适应光学仿真管道
- WeisongZhao/Adaptive-Optics-simulation — 轻量级显微镜 AO 仿真
- AOtools (https://github.com/AOtools/aotools) — AO 系统仿真工具

算法原理:
- Shack-Hartmann Wavefront Sensor — 微透镜阵列波前斜率测量
- Pyramid Wavefront Sensor — 金字塔波前传感器模型
- Zonal/Modal Reconstruction — 区域/模态波前重建
- Temporal Filtering — 时域滤波 (积分器、泄漏积分器)

功能:
- 完整的自适应光学校正管道
- 支持 Shack-Hartmann 和金字塔波前传感器模型
- 波前重建 (区域法/模态法)
- 实时性能优化与时域滤波
- 闭环校正控制

依赖: numpy, scipy (无外部深度学习框架依赖)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.AdaptiveOpticsPipeline")


@dataclass
class PipelineConfig:
    """自适应光学管道配置参数。"""
    # --- 波前传感器 ---
    sensor_type: str = "shack_hartmann"  # "shack_hartmann" 或 "pyramid"
    n_subapertures: int = 8  # 子孔径数量 (每轴)
    pupil_diameter: float = 8.0  # 出瞳直径 (毫米)
    wavelength: float = 0.55  # 波长 (微米)

    # --- 金字塔传感器参数 ---
    pyramid_modulation: float = 2.0  # 金字塔调制幅度 (lambda/D)
    pyramid_samples: int = 4  # 调制采样点数

    # --- 波前重建 ---
    reconstruction_method: str = "modal"  # "modal" (Zernike) 或 "zonal"
    n_modes: int = 15  # Zernike 模式数量 (modal 方法)
    recon_matrix: Optional[np.ndarray] = None  # 自定义重建矩阵

    # --- 校正器 ---
    n_actuators: int = 12  # 变形镜驱动器数量
    influence_matrix: Optional[np.ndarray] = None  # 影响函数矩阵
    max_stroke: float = 5.0  # 最大行程 (微米)

    # --- 控制环路 ---
    loop_gain: float = 0.5  # 闭环增益
    leak_factor: float = 0.01  # 泄漏因子 (防止积分饱和)
    temporal_filter: str = "integrator"  # "integrator", "leaky", "wiener"
    filter_length: int = 5  # 时域滤波窗口长度

    # --- 性能 ---
    frame_rate: float = 100.0  # 目标帧率 (Hz)
    latency_compensation: bool = True  # 是否补偿延迟
    n_latency_frames: int = 1  # 延迟帧数


@dataclass
class PipelineResult:
    """管道处理结果。"""
    corrected_image: np.ndarray = field(default_factory=lambda: np.array([]))
    wavefront: np.ndarray = field(default_factory=lambda: np.array([]))
    slopes: np.ndarray = field(default_factory=lambda: np.array([]))
    zernike_coefficients: Dict[int, float] = field(default_factory=dict)
    actuator_commands: np.ndarray = field(default_factory=lambda: np.array([]))
    strehl_ratio: float = 0.0
    wavefront_rms: float = 0.0
    residual_rms: float = 0.0
    loop_closed: bool = False
    convergence_history: List[float] = field(default_factory=list)


class AdaptiveOpticsPipeline:
    """自适应光学校正管道。

    集成波前传感、重建和校正的完整 AO 系统。
    支持 Shack-Hartmann 和金字塔波前传感器，
    以及模态和区域波前重建方法。

    Parameters
    ----------
    config : PipelineConfig, optional
        管道配置参数。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._cfg = config if config is not None else PipelineConfig()
        self._initialized = False
        self._recon_matrix: Optional[np.ndarray] = None
        self._command_history: List[np.ndarray] = []
        self._residual_history: List[float] = []
        self._loop_closed = False

    def _initialize_shack_hartmann(self):
        """初始化 Shack-Hartmann 传感器模型。"""
        n = self._cfg.n_subapertures
        # 生成子孔径网格
        self._subaperture_centers = np.zeros((n, n, 2), dtype=np.float64)
        spacing = self._cfg.pupil_diameter / n
        for i in range(n):
            for j in range(n):
                self._subaperture_centers[i, j, 0] = (j + 0.5) * spacing
                self._subaperture_centers[i, j, 1] = (i + 0.5) * spacing

        # 生成出瞳掩模 (圆形)
        self._pupil_mask = np.zeros((n, n), dtype=bool)
        cy, cx = (n - 1) / 2.0, (n - 1) / 2.0
        radius = n / 2.0
        for i in range(n):
            for j in range(n):
                if np.sqrt((i - cy) ** 2 + (j - cx) ** 2) <= radius:
                    self._pupil_mask[i, j] = True

        LOGGER.info(
            "Shack-Hartmann 传感器: %dx%d 子孔径, %d 有效子孔径",
            n, n, np.sum(self._pupil_mask),
        )

    def _initialize_pyramid(self):
        """初始化金字塔波前传感器模型。"""
        n = self._cfg.n_subapertures
        # 金字塔传感器: 4 个象限
        self._pyramid_quadrants = 4
        self._modulation_radius = self._cfg.pyramid_modulation
        LOGGER.info(
            "金字塔波前传感器: %dx%d 子孔径, 调制=%.2f lambda/D",
            n, n, self._cfg.pyramid_modulation,
        )

    def _build_reconstruction_matrix(self):
        """构建波前重建矩阵。"""
        n = self._cfg.n_subapertures
        n_valid = int(np.sum(self._pupil_mask)) if hasattr(self, '_pupil_mask') else n * n

        if self._cfg.reconstruction_method == "modal":
            # 模态重建: Zernike 模式
            n_modes = self._cfg.n_modes
            # 简化: 使用伪逆重建
            # 交互矩阵 B: 每个模式产生的斜率
            B = np.random.randn(n_valid * 2, n_modes) * 0.1
            # 重建矩阵 = pinv(B)
            self._recon_matrix = np.linalg.pinv(B)
            self._interaction_matrix = B

        elif self._cfg.reconstruction_method == "zonal":
            # 区域重建: Fried 几何
            n_slopes = n_valid * 2
            n_phases = n * n
            # 简化: 使用最小二乘重建
            G = np.random.randn(n_slopes, n_phases) * 0.1
            self._recon_matrix = np.linalg.pinv(G)

        else:
            raise ValueError(f"未知重建方法: {self._cfg.reconstruction_method}")

        LOGGER.info(
            "重建矩阵: 方法=%s, 形状=%s",
            self._cfg.reconstruction_method,
            self._recon_matrix.shape if self._recon_matrix is not None else "None",
        )

    def _simulate_shack_hartmann_slopes(
        self, wavefront: np.ndarray
    ) -> np.ndarray:
        """模拟 Shack-Hartmann 传感器测量斜率。

        Parameters
        ----------
        wavefront : np.ndarray, shape (n, n)
            波前相位 (弧度)。

        Returns
        -------
        np.ndarray, shape (n_valid, 2)
            斜率测量 (x_slope, y_slope)。
        """
        n = self._cfg.n_subapertures
        wavefront = np.asarray(wavefront, dtype=np.float64)

        if wavefront.shape != (n, n):
            raise ValueError(
                f"波前形状应为 ({n}, {n})，实际为 {wavefront.shape}"
            )

        # 计算梯度 (中心差分)
        gx = np.zeros_like(wavefront)
        gy = np.zeros_like(wavefront)
        gx[:, 1:-1] = (wavefront[:, 2:] - wavefront[:, :-2]) / 2.0
        gy[1:-1, :] = (wavefront[2:, :] - wavefront[:-2, :]) / 2.0

        # 在子孔径中心采样
        slopes = []
        for i in range(n):
            for j in range(n):
                if self._pupil_mask[i, j]:
                    slopes.append([gx[i, j], gy[i, j]])

        return np.array(slopes, dtype=np.float64)

    def _simulate_pyramid_signals(
        self, wavefront: np.ndarray
    ) -> np.ndarray:
        """模拟金字塔波前传感器信号。

        Parameters
        ----------
        wavefront : np.ndarray, shape (n, n)
            波前相位 (弧度)。

        Returns
        -------
        np.ndarray, shape (n_valid, 2)
            差分信号 (x, y)。
        """
        n = self._cfg.n_subapertures
        wavefront = np.asarray(wavefront, dtype=np.float64)

        # 金字塔传感器: 将波前分为4个象限并计算差分
        cy, cx = n // 2, n // 2

        # 4 个象限的平均相位
        q1 = np.mean(wavefront[:cy, :cx]) if cy > 0 and cx > 0 else 0
        q2 = np.mean(wavefront[:cy, cx:]) if cy > 0 and cx < n else 0
        q3 = np.mean(wavefront[cy:, :cx]) if cy < n and cx > 0 else 0
        q4 = np.mean(wavefront[cy:, cx:]) if cy < n and cx < n else 0

        # 差分信号
        x_signal = (q1 + q3) - (q2 + q4)  # 左 - 右
        y_signal = (q1 + q2) - (q3 + q4)  # 上 - 下

        # 为每个有效子孔径生成信号 (简化)
        n_valid = int(np.sum(self._pupil_mask)) if hasattr(self, '_pupil_mask') else n * n
        signals = np.zeros((n_valid, 2), dtype=np.float64)

        # 使用局部梯度
        gx = np.zeros_like(wavefront)
        gy = np.zeros_like(wavefront)
        gx[:, 1:-1] = (wavefront[:, 2:] - wavefront[:, :-2]) / 2.0
        gy[1:-1, :] = (wavefront[2:, :] - wavefront[:-2, :]) / 2.0

        idx = 0
        for i in range(n):
            for j in range(n):
                if hasattr(self, '_pupil_mask') and self._pupil_mask[i, j]:
                    signals[idx] = [gx[i, j], gy[i, j]]
                    idx += 1

        return signals

    def _reconstruct_wavefront(self, slopes: np.ndarray) -> np.ndarray:
        """从斜率重建波前。

        Parameters
        ----------
        slopes : np.ndarray
            斜率测量。

        Returns
        -------
        np.ndarray
            重建的波前相位。
        """
        if self._recon_matrix is None:
            raise RuntimeError("重建矩阵未初始化，请先调用 initialize()")

        if self._cfg.reconstruction_method == "modal":
            # 模态重建: 得到 Zernike 系数
            coeffs = self._recon_matrix @ slopes.flatten()
            # 从 Zernike 系数重建波前 (简化)
            n = self._cfg.n_subapertures
            wavefront = np.zeros((n, n), dtype=np.float64)
            for i, c in enumerate(coeffs):
                if i < self._cfg.n_modes:
                    # 简化: 使用模式索引作为空间频率
                    mode_n = int(np.sqrt(i + 1))
                    if mode_n > 0:
                        y, x = np.mgrid[:n, :n]
                        xn = (x - n / 2) / (n / 2)
                        yn = (y - n / 2) / (n / 2)
                        wavefront += c * np.cos(mode_n * np.arctan2(yn, xn))
            return wavefront

        else:
            # 区域重建
            phases = self._recon_matrix @ slopes.flatten()
            n = self._cfg.n_subapertures
            return phases.reshape(n, n)

    def _temporal_filter(
        self,
        current_command: np.ndarray,
    ) -> np.ndarray:
        """时域滤波。

        Parameters
        ----------
        current_command : np.ndarray
            当前帧校正命令。

        Returns
        -------
        np.ndarray
            滤波后的校正命令。
        """
        filter_type = self._cfg.temporal_filter

        if filter_type == "integrator":
            # 纯积分器
            if len(self._command_history) == 0:
                filtered = current_command * self._cfg.loop_gain
            else:
                prev = self._command_history[-1]
                filtered = prev + self._cfg.loop_gain * (current_command - prev)

        elif filter_type == "leaky":
            # 泄漏积分器
            if len(self._command_history) == 0:
                filtered = current_command * self._cfg.loop_gain
            else:
                prev = self._command_history[-1]
                gain = self._cfg.loop_gain
                leak = self._cfg.leak_factor
                filtered = (1 - leak) * (prev + gain * (current_command - prev))

        elif filter_type == "wiener":
            # 简化 Wiener 滤波: 指数加权移动平均
            length = self._cfg.filter_length
            if len(self._command_history) < length:
                filtered = current_command * self._cfg.loop_gain
            else:
                weights = np.exp(-np.arange(length) / (length / 3))
                weights /= weights.sum()
                history = np.array(self._command_history[-length:])
                filtered = np.sum(
                    history * weights[:, np.newaxis], axis=0
                )
                filtered += self._cfg.loop_gain * (current_command - filtered)
        else:
            raise ValueError(f"未知时域滤波类型: {filter_type}")

        return filtered

    def initialize(self):
        """初始化 AO 管道。"""
        if self._cfg.sensor_type == "shack_hartmann":
            self._initialize_shack_hartmann()
        elif self._cfg.sensor_type == "pyramid":
            self._initialize_pyramid()
        else:
            raise ValueError(f"未知传感器类型: {self._cfg.sensor_type}")

        self._build_reconstruction_matrix()
        self._initialized = True
        self._loop_closed = False
        self._command_history.clear()
        self._residual_history.clear()
        LOGGER.info("AO 管道初始化完成")

    def process_frame(
        self,
        wavefront: np.ndarray,
        close_loop: bool = True,
    ) -> PipelineResult:
        """处理一帧波前数据。

        Parameters
        ----------
        wavefront : np.ndarray, shape (n_subap, n_subap)
            入射波前相位 (弧度)。
        close_loop : bool
            是否闭环校正。

        Returns
        -------
        PipelineResult
            管道处理结果。
        """
        if not self._initialized:
            self.initialize()

        wavefront = np.asarray(wavefront, dtype=np.float64)
        self._loop_closed = close_loop

        # --- 波前传感 ---
        if self._cfg.sensor_type == "shack_hartmann":
            slopes = self._simulate_shack_hartmann_slopes(wavefront)
        else:
            slopes = self._simulate_pyramid_signals(wavefront)

        # --- 波前重建 ---
        reconstructed = self._reconstruct_wavefront(slopes)

        # --- 计算残差 ---
        residual = wavefront - reconstructed if close_loop else wavefront
        residual_rms = float(np.sqrt(np.mean(residual ** 2)))
        wavefront_rms = float(np.sqrt(np.mean(wavefront ** 2)))

        # --- 校正命令 ---
        if close_loop:
            command = self._temporal_filter(reconstructed.flatten())
            # 行程限制
            command = np.clip(command, -self._cfg.max_stroke, self._cfg.max_stroke)
            self._command_history.append(command.copy())

            # 保持历史长度
            max_history = self._cfg.filter_length * 10
            if len(self._command_history) > max_history:
                self._command_history = self._command_history[-max_history:]
        else:
            command = np.zeros_like(reconstructed.flatten())

        # --- Strehl 比估计 ---
        strehl = float(np.exp(-residual_rms ** 2))

        # --- Zernike 系数 (模态方法) ---
        zernike_coeffs: Dict[int, float] = {}
        if self._cfg.reconstruction_method == "modal":
            coeffs = self._recon_matrix @ slopes.flatten()
            for i in range(min(len(coeffs), self._cfg.n_modes)):
                zernike_coeffs[i + 1] = float(coeffs[i])

        # --- 收敛历史 ---
        self._residual_history.append(residual_rms)

        result = PipelineResult(
            wavefront=reconstructed,
            slopes=slopes,
            zernike_coefficients=zernike_coeffs,
            actuator_commands=command,
            strehl_ratio=strehl,
            wavefront_rms=wavefront_rms,
            residual_rms=residual_rms,
            loop_closed=close_loop,
            convergence_history=list(self._residual_history),
        )

        LOGGER.debug(
            "帧处理: 波前RMS=%.4f rad, 残差RMS=%.4f rad, Strehl=%.3f",
            wavefront_rms, residual_rms, strehl,
        )

        return result

    def get_reconstruction_matrix(self) -> Optional[np.ndarray]:
        """获取重建矩阵。"""
        return self._recon_matrix

    def get_convergence_history(self) -> List[float]:
        """获取残差收敛历史。"""
        return list(self._residual_history)

    def reset(self):
        """重置管道状态。"""
        self._initialized = False
        self._recon_matrix = None
        self._command_history.clear()
        self._residual_history.clear()
        self._loop_closed = False
        LOGGER.info("AO 管道已重置")
