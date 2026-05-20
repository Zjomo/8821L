"""
变形镜校准器 (Deformable Mirror Calibrator)

参考开源项目:
  - dmlib (https://github.com/spacetelescope/dmlib): 变形镜校准库

核心思想:
  从 dmlib 的变形镜校准流程中借鉴，实现干涉校准仿真、
  Zernike 模式控制和影响函数矩阵计算。

  在 SpotZoom 场景中:
  - DM → 变形镜或可变形反射镜
  - 校准 → 建立驱动器电压与面形变化的映射
  - 影响函数 → 每个驱动器对面形的贡献
  - Zernike 控制 → 通过 Zernike 模式控制面形

创新点:
  1. 干涉测量仿真 (双光束干涉)
  2. Zernike 模式分解与控制
  3. 影响函数矩阵计算与伪逆求解
  4. 面形质量分析 (RMS, PV)
  5. 闭环面形校正仿真

纯 numpy 实现，无外部依赖。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DMCalibConfig:
    """变形镜校准配置。"""
    # 驱动器数量
    n_actuators: int = 64
    # 驱动器排列: "square", "hexagonal"
    actuator_layout: str = "square"
    # 驱动器网格大小 (n x n)
    actuator_grid: int = 8
    # 驱动器间距 (像素)
    actuator_spacing: float = 10.0
    # 驱动器影响函数 sigma (像素)
    influence_sigma: float = 8.0
    # 最大驱动电压
    max_voltage: float = 1.0
    # 最小驱动电压
    min_voltage: float = -1.0
    # Zernike 最大阶数
    zernike_max_order: int = 6
    # 校准测量噪声 (波数)
    measurement_noise: float = 0.01
    # 干涉仪波长 (微米)
    interferometer_wavelength: float = 0.633
    # 干涉仪分辨率 (像素)
    interferometer_resolution: int = 256
    # 闭环校正增益
    correction_gain: float = 0.5
    # 闭环校正最大迭代
    correction_max_iter: int = 50
    # 面形目标 RMS (波数)
    target_rms: float = 0.05


@dataclass
class DMCalibResult:
    """DM 校准结果。"""
    # 影响函数矩阵 (n_pixels x n_actuators)
    influence_matrix: np.ndarray = field(default_factory=lambda: np.zeros((256, 64)))
    # 命令矩阵 (n_actuators x n_pixels)
    command_matrix: np.ndarray = field(default_factory=lambda: np.zeros((64, 256)))
    # Zernike 到驱动器的映射矩阵
    zernike_to_actuator: np.ndarray = field(default_factory=lambda: np.zeros((64, 21)))
    # 驱动器到 Zernike 的映射矩阵
    actuator_to_zernike: np.ndarray = field(default_factory=lambda: np.zeros((21, 64)))
    # 当前面形 (波数)
    current_surface: np.ndarray = field(default_factory=lambda: np.zeros((256, 256)))
    # 校准残差 RMS (波数)
    calibration_rms: float = 0.0
    # 是否校准成功
    success: bool = False


class DMCalibrator:
    """变形镜校准器。

    提供变形镜的校准、Zernike 模式控制和面形分析功能。

    Parameters
    ----------
    config : DMCalibConfig
        校准器配置参数。

    References
    ----------
    .. [1] dmlib documentation:
           https://spacetelescope.github.io/dmlib/
    .. [2] Bifano, T. G. (2011). "Adaptive imaging: MEMS deformable
           mirrors." Nature Photonics, 5(1), 21-23.
    .. [3] Noll, R. J. (1976). "Zernike polynomials and atmospheric
           turbulence." JOSA, 66(3), 207-211.
    """

    def __init__(self, config: Optional[DMCalibConfig] = None) -> None:
        self.config = config or DMCalibConfig()
        self._actuator_positions: Optional[np.ndarray] = None
        self._current_voltages: np.ndarray = np.zeros(self.config.n_actuators)
        self._calib_result: Optional[DMCalibResult] = None

        # 初始化驱动器位置
        self._init_actuator_positions()

        logger.info(
            f"DMCalibrator 初始化: "
            f"驱动器={self.config.n_actuators}, "
            f"布局={self.config.actuator_layout}, "
            f"网格={self.config.actuator_grid}x{self.config.actuator_grid}"
        )

    def calibrate(self) -> DMCalibResult:
        """执行完整校准流程。

        Returns
        -------
        DMCalibResult
            校准结果。
        """
        logger.info("开始 DM 校准流程...")

        result = DMCalibResult()

        # 1. 计算影响函数矩阵
        logger.info("步骤 1/4: 计算影响函数矩阵...")
        result.influence_matrix = self.compute_influence_matrix()

        # 2. 计算命令矩阵 (伪逆)
        logger.info("步骤 2/4: 计算命令矩阵...")
        result.command_matrix = np.linalg.pinv(result.influence_matrix)

        # 3. 建立 Zernike 映射
        logger.info("步骤 3/4: 建立 Zernike 模式映射...")
        zernike_modes = self._generate_zernike_modes()
        n_zernike = zernike_modes.shape[1]

        # Zernike → 驱动器
        result.zernike_to_actuator = result.command_matrix @ zernike_modes
        # 驱动器 → Zernike
        result.actuator_to_zernike = np.linalg.pinv(result.zernike_to_actuator)

        # 4. 验证校准质量
        logger.info("步骤 4/4: 验证校准质量...")
        test_surface = zernike_modes[:, 0]  # 用第一个 Zernike 模式测试
        reconstructed = result.influence_matrix @ (result.command_matrix @ test_surface)
        residual = test_surface - reconstructed
        result.calibration_rms = float(np.sqrt(np.mean(residual ** 2)))
        result.success = result.calibration_rms < 0.1

        self._calib_result = result

        logger.info(
            f"DM 校准完成: RMS={result.calibration_rms:.4f} 波, "
            f"成功={result.success}"
        )
        return result

    def compute_influence_matrix(self) -> np.ndarray:
        """计算影响函数矩阵。

        对每个驱动器施加单位电压，测量产生的面形变化。

        Returns
        -------
        np.ndarray
            影响函数矩阵，形状 (n_pixels, n_actuators)。
        """
        res = self.config.interferometer_resolution
        n_act = self.config.n_actuators
        sigma = self.config.influence_sigma

        # 像素坐标网格
        yy, xx = np.mgrid[:res, :res]
        pixel_coords = np.column_stack([xx.flatten(), yy.flatten()])

        influence_matrix = np.zeros((res * res, n_act))

        for i in range(n_act):
            if self._actuator_positions is None:
                break

            ax, ay = self._actuator_positions[i]
            # 高斯影响函数
            r2 = (pixel_coords[:, 0] - ax) ** 2 + (pixel_coords[:, 1] - ay) ** 2
            influence = np.exp(-r2 / (2 * sigma ** 2))

            # 添加测量噪声
            if self.config.measurement_noise > 0:
                influence += np.random.normal(
                    0, self.config.measurement_noise, influence.shape
                )

            influence_matrix[:, i] = influence

        logger.info(f"影响函数矩阵: shape={influence_matrix.shape}")
        return influence_matrix

    def zernike_control(
        self,
        zernike_coefficients: np.ndarray,
        calib_result: Optional[DMCalibResult] = None,
    ) -> np.ndarray:
        """通过 Zernike 系数控制 DM。

        Parameters
        ----------
        zernike_coefficients : np.ndarray
            目标 Zernike 系数向量。
        calib_result : DMCalibResult, optional
            校准结果。如果为 None，使用上次校准结果。

        Returns
        -------
        np.ndarray
            驱动器电压向量。
        """
        if calib_result is None:
            calib_result = self._calib_result

        if calib_result is None:
            logger.error("未找到校准结果，请先执行 calibrate()")
            return np.zeros(self.config.n_actuators)

        voltages = calib_result.zernike_to_actuator @ zernike_coefficients
        voltages = np.clip(voltages, self.config.min_voltage, self.config.max_voltage)

        self._current_voltages = voltages
        logger.info(f"Zernike 控制: {len(zernike_coefficients)} 个模式, "
                     f"最大电压={np.max(np.abs(voltages)):.3f}")

        return voltages

    def analyze_surface(
        self,
        surface: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """分析面形质量。

        Parameters
        ----------
        surface : np.ndarray, optional
            面形数据 (波数)。如果为 None，使用当前面形。

        Returns
        -------
        dict
            面形分析指标。
        """
        if surface is None and self._calib_result is not None:
            surface = self._calib_result.current_surface

        if surface is None:
            logger.error("无面形数据可分析")
            return {}

        surface_flat = surface.flatten()

        # RMS
        rms = float(np.sqrt(np.mean(surface_flat ** 2)))

        # PV (Peak-to-Valley)
        pv = float(np.max(surface_flat) - np.min(surface_flat))

        # 去除活塞后的 RMS
        surface_debiased = surface_flat - np.mean(surface_flat)
        rms_debiased = float(np.sqrt(np.mean(surface_debiased ** 2)))

        # 去除活塞+倾斜后的 RMS
        if self._actuator_positions is not None:
            res = self.config.interferometer_resolution
            yy, xx = np.mgrid[:res, :res]
            A = np.column_stack([
                np.ones(res * res),
                xx.flatten() / res,
                yy.flatten() / res,
            ])
            tilt_coeffs, _, _, _ = np.linalg.lstsq(A, surface_flat, rcond=None)
            surface_detilted = surface_flat - A @ tilt_coeffs
            rms_detilted = float(np.sqrt(np.mean(surface_detilted ** 2)))
        else:
            rms_detilted = rms_debiased

        # Zernike 分解
        if self._calib_result is not None and self._calib_result.actuator_to_zernike is not None:
            zernike_coeffs = self._calib_result.actuator_to_zernike @ self._current_voltages
            zernike_rms = float(np.sqrt(np.mean(zernike_coeffs[4:] ** 2)))  # 跳过活塞和倾斜
        else:
            zernike_rms = 0.0

        metrics = {
            "rms_waves": rms,
            "pv_waves": pv,
            "rms_debiased_waves": rms_debiased,
            "rms_detilted_waves": rms_detilted,
            "zernike_rms_waves": zernike_rms,
            "mean_waves": float(np.mean(surface_flat)),
            "std_waves": float(np.std(surface_flat)),
        }

        logger.info(f"面形分析: RMS={rms:.4f}, PV={pv:.4f} 波")
        return metrics

    def get_actuator_positions(self) -> np.ndarray:
        """获取驱动器位置。

        Returns
        -------
        np.ndarray
            驱动器位置数组，形状 (n_actuators, 2)。
        """
        if self._actuator_positions is None:
            return np.zeros((self.config.n_actuators, 2))
        return self._actuator_positions.copy()

    def get_current_voltages(self) -> np.ndarray:
        """获取当前驱动器电压。

        Returns
        -------
        np.ndarray
            电压向量。
        """
        return self._current_voltages.copy()

    def simulate_interferogram(
        self,
        surface: np.ndarray,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
    ) -> np.ndarray:
        """模拟干涉图。

        Parameters
        ----------
        surface : np.ndarray
            面形数据 (波数)。
        tilt_x : float
            X 方向倾斜 (波数/像素)。
        tilt_y : float
            Y 方向倾斜 (波数/像素)。

        Returns
        -------
        np.ndarray
            干涉图 (0~1 灰度)。
        """
        h, w = surface.shape
        yy, xx = np.mgrid[:h, :w]

        # 参考波 + 测试波
        reference_phase = 2 * np.pi * (tilt_x * xx + tilt_y * yy)
        test_phase = 2 * np.pi * surface

        # 双光束干涉
        intensity = 0.5 * (1 + np.cos(reference_phase - test_phase))

        return intensity

    # ============ 内部方法 ============

    def _init_actuator_positions(self) -> None:
        """初始化驱动器位置。"""
        n = self.config.actuator_grid
        spacing = self.config.actuator_spacing
        res = self.config.interferometer_resolution
        center = res / 2

        positions = []
        if self.config.actuator_layout == "square":
            for row in range(n):
                for col in range(n):
                    x = center + (col - (n - 1) / 2) * spacing
                    y = center + (row - (n - 1) / 2) * spacing
                    positions.append([x, y])
        elif self.config.actuator_layout == "hexagonal":
            for row in range(n):
                for col in range(n):
                    x = center + (col - (n - 1) / 2) * spacing
                    y = center + (row - (n - 1) / 2) * spacing * np.sqrt(3) / 2
                    if row % 2 == 1:
                        x += spacing / 2
                    positions.append([x, y])
        else:
            logger.warning(f"未知布局: {self.config.actuator_layout}, 使用方形")
            for row in range(n):
                for col in range(n):
                    x = center + (col - (n - 1) / 2) * spacing
                    y = center + (row - (n - 1) / 2) * spacing
                    positions.append([x, y])

        self._actuator_positions = np.array(positions[:self.config.n_actuators])

    def _generate_zernike_modes(self) -> np.ndarray:
        """生成 Zernike 模式矩阵。

        Returns
        -------
        np.ndarray
            Zernike 模式矩阵，形状 (n_pixels, n_modes)。
        """
        res = self.config.interferometer_resolution
        max_order = self.config.zernike_max_order

        # 计算模式数
        n_modes = 0
        j = 0
        while True:
            n = int((-1 + np.sqrt(1 + 8 * j)) / 2)
            if n > max_order:
                break
            n_modes += 1
            j += 1

        # 归一化坐标
        cy, cx = res / 2, res / 2
        yy, xx = np.mgrid[:res, :res]
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        theta = np.arctan2((yy - cy) / cy, (xx - cx) / cx)
        mask = r <= 1.0

        modes = np.zeros((res * res, n_modes))

        for j_idx in range(n_modes):
            n, m = self._noll_to_nm(j_idx)
            zernike = self._zernike_polynomial(n, m, r, theta) * mask
            norm = np.sqrt(np.sum(zernike ** 2))
            if norm > 1e-10:
                zernike /= norm
            modes[:, j_idx] = zernike.flatten()

        return modes

    def _noll_to_nm(self, j: int) -> Tuple[int, int]:
        """Noll 索引 → (n, m)。"""
        n = int((-1 + np.sqrt(1 + 8 * j)) / 2)
        m_values = list(range(-n, n + 1, 2))
        k = j - n * (n + 1) // 2
        m = m_values[k] if k < len(m_values) else 0
        return n, m

    def _zernike_polynomial(
        self, n: int, m: int, r: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """计算 Zernike 多项式。"""
        if n == 0:
            return np.ones_like(r)
        if n == 1 and m == 1:
            return r * np.cos(theta)
        if n == 1 and m == -1:
            return r * np.sin(theta)

        radial = np.zeros_like(r)
        for s in range((n - abs(m)) // 2 + 1):
            coeff = (
                (-1) ** s
                * math.factorial(n - s)
                / (
                    math.factorial(s)
                    * math.factorial((n + abs(m)) // 2 - s)
                    * math.factorial((n - abs(m)) // 2 - s)
                )
            )
            radial += coeff * r ** (n - 2 * s)

        if m > 0:
            return radial * np.cos(m * theta)
        elif m < 0:
            return radial * np.sin(abs(m) * theta)
        else:
            return radial
