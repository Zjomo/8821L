"""
数字孪生仿真器 (DigitalTwinSimulator)

灵感来源:
- OOPAO (https://github.com/cheritier/OOPAO) — 端到端自适应光学仿真框架，
  提供从光源到探测器的完整光学链路建模
- lowfssim (https://github.com/nasa-jpl/lowfssim) — NASA JPL 光学仿真模型，
  用于低频波前传感与校正系统的仿真验证
- HCIPy (https://github.com/ehpor/hcipy) — 高对比度成像光学传播仿真
- POPPY (NASA/STScI) — 光学传播与 PSF 仿真

算法原理:
- Gaussian Beam Propagation — 高斯光束传播，基于 ABCD 矩阵法:
    q_out = (A*q_in + B) / (C*q_in + D)
    其中 q = z + i*z_R 为复光束参数
- ABCD Matrix Method — 光学系统的光线传输矩阵方法:
    自由空间: [[1, d], [0, 1]]
    薄透镜:   [[1, 0], [-1/f, 1]]
- Fraunhofer Diffraction — 夫琅禾费衍射: 远场 PSF = |FT(孔径函数)|^2
- Airy Pattern — 圆孔衍射 PSF: I(r) = [2*J1(x)/x]^2
- Detector Noise Model — 探测器噪声模型:
    泊松散粒噪声 + 高斯读出噪声 + 暗电流 + ADC 量化
- Motor Backlash Model — 电机回程差模型:
    死区 + 滞后效应，方向反转时存在定位误差
- Closed-Loop Control Simulation — 闭环控制仿真:
    探测 -> 计算误差 -> 驱动电机 -> 更新场景
- Environmental Disturbance — 环境扰动模型:
    多频振动 (正弦叠加) + 布朗运动漂移 + 热漂移 + 大气湍流

功能:
- 完整光学链路仿真: 光源 -> 光学元件 -> 探测器 -> 图像
- 高斯光束 ABCD 矩阵传播
- 薄透镜、圆孔光阑、遮挡等光学元件建模
- 探测器噪声模型 (散粒噪声、读出噪声、暗电流、量化)
- 电机平台模型 (步进、回程差、死区、速度限制)
- 环境扰动模型 (振动、湍流、热漂移)
- 闭环对准仿真
- 数字孪生验证与参数估计

依赖: numpy, opencv-python (仅用于图像处理辅助)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import cv2
import numpy as np

LOGGER = logging.getLogger("SpotZoom.DigitalTwinSimulator")

# 模块默认禁用标志
digital_twin_enabled: bool = False


# ======================== 物理常数 ========================

SPEED_OF_LIGHT = 2.998e8          # 光速 (m/s)
PLANCK_CONSTANT = 6.626e-34       # 普朗克常数 (J*s)
BOLTZMANN_CONSTANT = 1.381e-23    # 玻尔兹曼常数 (J/K)
ELECTRON_CHARGE = 1.602e-19       # 电子电荷 (C)


# ======================== 数据类 ========================


@dataclass
class SimulationConfig:
    """仿真全局配置。

    Attributes
    ----------
    beam_wavelength_nm : float
        光束波长 (nm)，默认 632.8nm (HeNe 激光)。
    beam_waist_um : float
        光束束腰半径 (um)。
    beam_power_mw : float
        光束功率 (mW)。
    lens_focal_length_mm : float
        透镜焦距 (mm)。
    aperture_diameter_mm : float
        孔径直径 (mm)。
    detector_pixel_size_um : float
        探测器像素尺寸 (um)。
    detector_resolution : Tuple[int, int]
        探测器分辨率 (宽, 高)。
    detector_read_noise_e : float
        探测器读出噪声 (电子)。
    detector_dark_current_e_per_s : float
        暗电流 (电子/秒/像素)。
    detector_quantization_bits : int
        ADC 量化位数。
    detector_quantum_efficiency : float
        量子效率 [0, 1]。
    detector_fill_factor : float
        像素填充因子 [0, 1]。
    detector_full_well_e : float
        满阱电子数。
    motor_step_size_um : float
        电机步进尺寸 (um)。
    motor_backlash_steps : int
        回程差 (步数)。
    motor_max_velocity_steps_s : float
        电机最大速度 (步/秒)。
    motor_max_acceleration_steps_s2 : float
        电机最大加速度 (步/秒^2)。
    motor_direction_sign : Tuple[int, int]
        电机方向符号 (X, Y)，+1 或 -1。
    motor_command_delay_steps : int
        命令响应延迟 (步数)。
    motor_position_noise_std_um : float
        位置噪声标准差 (um)。
    vibration_amplitude_px : float
        振动幅度 (像素)。
    vibration_frequencies_hz : List[float]
        振动频率列表 (Hz)。
    vibration_phases_rad : Optional[List[float]]
        振动初始相位 (弧度)，为 None 时随机生成。
    drift_rate_px_per_s : float
        漂移速率 (像素/秒)。
    thermal_drift_amplitude_px : float
        热漂移幅度 (像素)。
    thermal_drift_time_constant_s : float
        热漂移时间常数 (秒)。
    enable_turbulence : bool
        是否启用大气湍流。
    turbulence_strength : float
        湍流强度 (归一化)。
    simulation_dt_s : float
        仿真时间步长 (秒)。
    seed : Optional[int]
        随机种子。
    """

    # 光束参数
    beam_wavelength_nm: float = 632.8
    beam_waist_um: float = 50.0
    beam_power_mw: float = 1.0

    # 光学参数
    lens_focal_length_mm: float = 100.0
    aperture_diameter_mm: float = 25.0

    # 探测器参数
    detector_pixel_size_um: float = 5.0
    detector_resolution: Tuple[int, int] = (640, 480)
    detector_read_noise_e: float = 2.0
    detector_dark_current_e_per_s: float = 0.1
    detector_quantization_bits: int = 8
    detector_quantum_efficiency: float = 0.8
    detector_fill_factor: float = 0.95
    detector_full_well_e: float = 20000.0

    # 电机参数
    motor_step_size_um: float = 0.5
    motor_backlash_steps: int = 2
    motor_max_velocity_steps_s: float = 5000.0
    motor_max_acceleration_steps_s2: float = 50000.0
    motor_direction_sign: Tuple[int, int] = (1, 1)
    motor_command_delay_steps: int = 0
    motor_position_noise_std_um: float = 0.05

    # 振动参数
    vibration_amplitude_px: float = 0.5
    vibration_frequencies_hz: List[float] = field(
        default_factory=lambda: [1.0, 10.0, 50.0]
    )
    vibration_phases_rad: Optional[List[float]] = None

    # 漂移参数
    drift_rate_px_per_s: float = 0.1
    thermal_drift_amplitude_px: float = 2.0
    thermal_drift_time_constant_s: float = 300.0

    # 湍流参数
    enable_turbulence: bool = False
    turbulence_strength: float = 0.1

    # 仿真参数
    simulation_dt_s: float = 0.01
    seed: Optional[int] = None


@dataclass
class SimulationState:
    """仿真状态快照。

    Attributes
    ----------
    beam_position_x : float
        光束 X 位置 (像素)。
    beam_position_y : float
        光束 Y 位置 (像素)。
    motor_position_x : float
        电机 X 位置 (步数)。
    motor_position_y : float
        电机 Y 位置 (步数)。
    target_position_x : float
        目标 X 位置 (像素)。
    target_position_y : float
        目标 Y 位置 (像素)。
    current_frame : np.ndarray
        当前帧图像。
    time_elapsed_s : float
        已用时间 (秒)。
    iteration_count : int
        迭代次数。
    """

    beam_position_x: float = 0.0
    beam_position_y: float = 0.0
    motor_position_x: float = 0.0
    motor_position_y: float = 0.0
    target_position_x: float = 0.0
    target_position_y: float = 0.0
    current_frame: np.ndarray = field(default_factory=lambda: np.zeros((480, 640), dtype=np.float64))
    time_elapsed_s: float = 0.0
    iteration_count: int = 0


@dataclass
class SimulationResult:
    """仿真结果。

    Attributes
    ----------
    trajectory_x : List[float]
        X 方向轨迹 (像素)。
    trajectory_y : List[float]
        Y 方向轨迹 (像素)。
    error_trajectory : List[float]
        误差轨迹 (像素)。
    convergence_time_s : float
        收敛时间 (秒)，未收敛时为 -1。
    final_error_px : float
        最终误差 (像素)。
    total_motor_steps_x : int
        X 方向总电机步数。
    total_motor_steps_y : int
        Y 方向总电机步数。
    frames_recorded : List[np.ndarray]
        记录的关键帧。
    metrics_summary : Dict[str, float]
        指标汇总。
    success : bool
        是否成功收敛。
    """

    trajectory_x: List[float] = field(default_factory=list)
    trajectory_y: List[float] = field(default_factory=list)
    error_trajectory: List[float] = field(default_factory=list)
    convergence_time_s: float = -1.0
    final_error_px: float = float("inf")
    total_motor_steps_x: int = 0
    total_motor_steps_y: int = 0
    frames_recorded: List[np.ndarray] = field(default_factory=list)
    metrics_summary: Dict[str, float] = field(default_factory=dict)
    success: bool = False


@dataclass
class TwinValidationReport:
    """数字孪生验证报告。

    Attributes
    ----------
    model_fidelity_score : float
        模型保真度评分 [0, 1]，1 = 完美匹配。
    parameter_estimates : Dict[str, float]
        估计的真实系统参数。
    prediction_accuracy : Dict[str, float]
        预测精度指标。
    recommendations : List[str]
        改进建议。
    validation_timestamp : str
        验证时间戳。
    """

    model_fidelity_score: float = 0.0
    parameter_estimates: Dict[str, float] = field(default_factory=dict)
    prediction_accuracy: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    validation_timestamp: str = ""


@dataclass
class BeamParameters:
    """光束参数计算结果。

    Attributes
    ----------
    waist : float
        束腰半径 (m)。
    rayleigh_range : float
        瑞利距离 (m)。
    divergence_half_angle : float
        远场发散半角 (rad)。
    wavelength : float
        波长 (m)。
    q_parameter : complex
        复光束参数 q。
    """

    waist: float = 0.0
    rayleigh_range: float = 0.0
    divergence_half_angle: float = 0.0
    wavelength: float = 0.0
    q_parameter: complex = 0.0 + 0.0j


# ======================== 光学元件协议 ========================


@runtime_checkable
class OpticalElement(Protocol):
    """光学元件协议接口。

    所有光学元件必须实现 propagate_beam 和 get_psf 方法。
    """

    def propagate_beam(self, input_beam: "GaussianBeam") -> "GaussianBeam":
        """传播光束通过该元件。

        Parameters
        ----------
        input_beam : GaussianBeam
            输入高斯光束。

        Returns
        -------
        GaussianBeam
            输出高斯光束。
        """
        ...

    def get_psf(self, grid_size: int = 256, pixel_scale: float = 1e-6) -> np.ndarray:
        """计算该元件的 PSF。

        Parameters
        ----------
        grid_size : int
            计算网格尺寸。
        pixel_scale : float
            像素尺度 (m/pixel)。

        Returns
        -------
        np.ndarray
            2D PSF 数组。
        """
        ...


# ======================== 高斯光束模型 ========================


class GaussianBeam:
    """高斯光束模型。

    基于复光束参数 q 的高斯光束传播模型，支持 ABCD 矩阵传播。

    Parameters
    ----------
    wavelength : float
        波长 (m)。
    waist : float
        束腰半径 w0 (m)。
    waist_position : float
        束腰位置 z0 (m)，默认为 0。
    power : float
        光束功率 (W)。

    Examples
    --------
    >>> beam = GaussianBeam(wavelength=632.8e-9, waist=50e-6)
    >>> beam.waist
    5e-05
    >>> beam.rayleigh_range()
    12.41...
    """

    def __init__(
        self,
        wavelength: float,
        waist: float,
        waist_position: float = 0.0,
        power: float = 1e-3,
    ):
        if wavelength <= 0:
            raise ValueError(f"波长必须 > 0，当前值: {wavelength}")
        if waist <= 0:
            raise ValueError(f"束腰半径必须 > 0，当前值: {waist}")
        if power < 0:
            raise ValueError(f"功率必须 >= 0，当前值: {power}")

        self.wavelength = float(wavelength)
        self.waist = float(waist)
        self.waist_position = float(waist_position)
        self.power = float(power)

    @property
    def k(self) -> float:
        """波数 k = 2*pi/lambda。"""
        return 2.0 * np.pi / self.wavelength

    def rayleigh_range(self) -> float:
        """计算瑞利距离 z_R = pi * w0^2 / lambda。

        Returns
        -------
        float
            瑞利距离 (m)。
        """
        return np.pi * self.waist ** 2 / self.wavelength

    def divergence_half_angle(self) -> float:
        """计算远场发散半角 theta = lambda / (pi * w0)。

        Returns
        -------
        float
            发散半角 (rad)。
        """
        return self.wavelength / (np.pi * self.waist)

    def q_parameter(self, z: float = 0.0) -> complex:
        """计算位置 z 处的复光束参数 q。

        q(z) = (z - z0) + i * z_R

        Parameters
        ----------
        z : float
            传播距离 (m)。

        Returns
        -------
        complex
            复光束参数 q。
        """
        z_r = self.rayleigh_range()
        return complex(z - self.waist_position, z_r)

    def beam_radius(self, z: float) -> float:
        """计算位置 z 处的光束半径 w(z)。

        w(z) = w0 * sqrt(1 + ((z - z0) / z_R)^2)

        Parameters
        ----------
        z : float
            传播距离 (m)。

        Returns
        -------
        float
            光束半径 (m)。
        """
        z_r = self.rayleigh_range()
        dz = z - self.waist_position
        return self.waist * np.sqrt(1.0 + (dz / z_r) ** 2)

    def radius_of_curvature(self, z: float) -> float:
        """计算位置 z 处的波前曲率半径 R(z)。

        R(z) = (z - z0) * (1 + (z_R / (z - z0))^2)

        Parameters
        ----------
        z : float
            传播距离 (m)。

        Returns
        -------
        float
            曲率半径 (m)，束腰处返回 inf。
        """
        z_r = self.rayleigh_range()
        dz = z - self.waist_position
        if abs(dz) < 1e-15:
            return float("inf")
        return dz * (1.0 + (z_r / dz) ** 2)

    def gouy_phase(self, z: float) -> float:
        """计算 Gouy 相位 psi(z)。

        psi(z) = arctan((z - z0) / z_R)

        Parameters
        ----------
        z : float
            传播距离 (m)。

        Returns
        -------
        float
            Gouy 相位 (rad)。
        """
        z_r = self.rayleigh_range()
        return np.arctan2(z - self.waist_position, z_r)

    def intensity_profile(self, z: float, r: np.ndarray) -> np.ndarray:
        """计算位置 z 处的径向强度分布。

        I(r, z) = (2*P / (pi*w(z)^2)) * exp(-2*r^2 / w(z)^2)

        Parameters
        ----------
        z : float
            传播距离 (m)。
        r : np.ndarray
            径向坐标数组 (m)。

        Returns
        -------
        np.ndarray
            强度分布 (W/m^2)。
        """
        w = self.beam_radius(z)
        i0 = 2.0 * self.power / (np.pi * w ** 2)
        return i0 * np.exp(-2.0 * r ** 2 / w ** 2)

    def propagate_abcd(self, z: float, abcd: np.ndarray) -> "GaussianBeam":
        """通过 ABCD 矩阵传播光束。

        q_out = (A*q_in + B) / (C*q_in + D)
        从 q_out 反算新的束腰和束腰位置。

        Parameters
        ----------
        z : float
            当前观测位置 (m)。
        abcd : np.ndarray
            2x2 ABCD 矩阵。

        Returns
        -------
        GaussianBeam
            传播后的新光束。
        """
        if abcd.shape != (2, 2):
            raise ValueError(f"ABCD 矩阵必须为 2x2，当前形状: {abcd.shape}")

        q_in = self.q_parameter(z)
        a, b = abcd[0, 0], abcd[0, 1]
        c, d = abcd[1, 0], abcd[1, 1]

        denom = c * q_in + d
        if abs(denom) < 1e-30:
            raise ValueError("ABCD 传播导致 q 参数发散 (焦点处)")

        q_out = (a * q_in + b) / denom

        # 从 q_out 反算光束参数
        # 1/q = 1/R - i*lambda/(pi*w^2)
        inv_q = 1.0 / q_out
        new_wavelength = self.wavelength

        # 虚部: Im(1/q) = -lambda / (pi * w^2)
        im_inv_q = inv_q.imag
        if abs(im_inv_q) < 1e-30:
            new_waist = 1e-3  # 默认值，避免除零
        else:
            w_squared = -new_wavelength / (np.pi * im_inv_q)
            if w_squared <= 0:
                new_waist = 1e-3
            else:
                new_waist = np.sqrt(w_squared)

        # 实部: Re(1/q) = 1/R
        re_inv_q = inv_q.real
        if abs(re_inv_q) < 1e-30:
            new_waist_pos = 0.0
        else:
            new_R = 1.0 / re_inv_q
            z_r_new = np.pi * new_waist ** 2 / new_wavelength
            # R = z * (1 + (z_R/z)^2) => 近似: z ≈ R / (1 + (z_R/R)^2)
            if abs(new_R) > 1e10:
                new_waist_pos = 0.0
            else:
                # 数值求解: R = z*(1 + (z_R/z)^2)
                # 使用近似: z ≈ R - z_R^2/R (当 |z| >> z_R 时)
                new_waist_pos = new_R - z_r_new ** 2 / new_R if abs(new_R) > 1e-15 else 0.0

        return GaussianBeam(
            wavelength=new_wavelength,
            waist=new_waist,
            waist_position=new_waist_pos,
            power=self.power,
        )

    def to_dict(self) -> Dict[str, float]:
        """导出光束参数为字典。

        Returns
        -------
        Dict[str, float]
            光束参数字典。
        """
        return {
            "wavelength_m": self.wavelength,
            "waist_m": self.waist,
            "waist_position_m": self.waist_position,
            "power_W": self.power,
            "rayleigh_range_m": self.rayleigh_range(),
            "divergence_half_angle_rad": self.divergence_half_angle(),
        }


# ======================== 具体光学元件 ========================


class ThinLens:
    """薄透镜模型。

    Parameters
    ----------
    focal_length : float
        焦距 (m)，正值为会聚透镜。
    diameter : float or None
        透镜直径 (m)，为 None 时无孔径限制。
    name : str
        元件名称。
    """

    def __init__(
        self,
        focal_length: float,
        diameter: Optional[float] = None,
        name: str = "thin_lens",
    ):
        if focal_length == 0:
            raise ValueError("焦距不能为零")
        self.focal_length = float(focal_length)
        self.diameter = float(diameter) if diameter is not None else None
        self.name = name

    @property
    def abcd_matrix(self) -> np.ndarray:
        """薄透镜 ABCD 矩阵: [[1, 0], [-1/f, 1]]。"""
        return np.array([
            [1.0, 0.0],
            [-1.0 / self.focal_length, 1.0],
        ])

    def propagate_beam(self, input_beam: GaussianBeam) -> GaussianBeam:
        """传播光束通过薄透镜。

        Parameters
        ----------
        input_beam : GaussianBeam
            输入光束。

        Returns
        -------
        GaussianBeam
            输出光束。
        """
        output = input_beam.propagate_abcd(0.0, self.abcd_matrix)

        # 如果有孔径限制，检查光束是否被裁剪
        if self.diameter is not None:
            w = input_beam.beam_radius(0.0)
            if w > self.diameter / 2.0:
                LOGGER.debug(
                    f"透镜 '{self.name}': 光束半径 {w*1e6:.1f}um "
                    f"超过孔径半径 {self.diameter/2*1e6:.1f}um"
                )

        return output

    def get_psf(self, grid_size: int = 256, pixel_scale: float = 1e-6) -> np.ndarray:
        """计算薄透镜的衍射极限 PSF (Airy 斑)。

        Parameters
        ----------
        grid_size : int
            网格尺寸。
        pixel_scale : float
            像素尺度 (m/pixel)。

        Returns
        -------
        np.ndarray
            2D PSF 数组 (归一化)。
        """
        if self.diameter is None:
            LOGGER.warning("透镜无孔径限制，使用默认 PSF")
            psf = np.zeros((grid_size, grid_size), dtype=np.float64)
            center = grid_size // 2
            psf[center, center] = 1.0
            return psf

        # Airy 斑: I(r) = [2*J1(x)/x]^2, x = pi*D*r/(lambda*f)
        coords = np.arange(grid_size, dtype=np.float64) - grid_size // 2
        xx, yy = np.meshgrid(coords, coords)
        r = np.sqrt(xx ** 2 + yy ** 2) * pixel_scale

        # 归一化半径参数
        x = np.pi * self.diameter * r / (632.8e-9 * self.focal_length)
        x = np.where(x < 1e-10, 1e-10, x)

        # 使用近似: 2*J1(x)/x ≈ sinc(x/(2*pi)) * cos(x/2 - pi/4) * sqrt(8/(pi*x))
        # 简化: 使用 numpy 的 sinc 函数近似 Airy 函数
        airy = self._airy_function(x)
        psf = airy ** 2
        psf /= psf.max() + 1e-30

        return psf

    @staticmethod
    def _airy_function(x: np.ndarray) -> np.ndarray:
        """计算 Airy 函数 Ai(x) 的近似值。

        使用 J1 贝塞尔函数的近似: 2*J1(x)/x。

        Parameters
        ----------
        x : np.ndarray
            输入数组。

        Returns
        -------
        np.ndarray
            Airy 函数值。
        """
        # 对于小 x: J1(x) ≈ x/2 - x^3/16
        # 2*J1(x)/x ≈ 1 - x^2/8
        small = x < 0.5
        result = np.zeros_like(x)

        # 小 x 近似
        result[small] = 1.0 - x[small] ** 2 / 8.0

        # 大 x: 使用渐近展开
        large = ~small
        xl = x[large]
        # J1(x) ≈ sqrt(2/(pi*x)) * cos(x - 3*pi/4)
        j1_approx = np.sqrt(2.0 / (np.pi * xl)) * np.cos(xl - 3.0 * np.pi / 4.0)
        result[large] = 2.0 * j1_approx / xl

        return result


class FreeSpace:
    """自由空间传播。

    Parameters
    ----------
    distance : float
        传播距离 (m)。
    name : str
        元件名称。
    """

    def __init__(self, distance: float, name: str = "free_space"):
        if distance < 0:
            raise ValueError(f"传播距离必须 >= 0，当前值: {distance}")
        self.distance = float(distance)
        self.name = name

    @property
    def abcd_matrix(self) -> np.ndarray:
        """自由空间 ABCD 矩阵: [[1, d], [0, 1]]。"""
        return np.array([
            [1.0, self.distance],
            [0.0, 1.0],
        ])

    def propagate_beam(self, input_beam: GaussianBeam) -> GaussianBeam:
        """传播光束通过自由空间。

        Parameters
        ----------
        input_beam : GaussianBeam
            输入光束。

        Returns
        -------
        GaussianBeam
            输出光束。
        """
        return input_beam.propagate_abcd(0.0, self.abcd_matrix)

    def get_psf(self, grid_size: int = 256, pixel_scale: float = 1e-6) -> np.ndarray:
        """自由空间不改变 PSF，返回 delta 函数。

        Parameters
        ----------
        grid_size : int
            网格尺寸。
        pixel_scale : float
            像素尺度。

        Returns
        -------
        np.ndarray
            2D PSF 数组。
        """
        psf = np.zeros((grid_size, grid_size), dtype=np.float64)
        center = grid_size // 2
        psf[center, center] = 1.0
        return psf


class CircularAperture:
    """圆孔光阑。

    Parameters
    ----------
    diameter : float
        孔径直径 (m)。
    name : str
        元件名称。
    """

    def __init__(self, diameter: float, name: str = "circular_aperture"):
        if diameter <= 0:
            raise ValueError(f"孔径直径必须 > 0，当前值: {diameter}")
        self.diameter = float(diameter)
        self.name = name

    def propagate_beam(self, input_beam: GaussianBeam) -> GaussianBeam:
        """传播光束通过圆孔光阑。

        光阑不改变高斯光束参数，但会截断光束边缘。
        这里通过减小等效束腰来近似截断效应。

        Parameters
        ----------
        input_beam : GaussianBeam
            输入光束。

        Returns
        -------
        GaussianBeam
            输出光束 (近似)。
        """
        w = input_beam.beam_radius(0.0)
        r_aperture = self.diameter / 2.0

        # 如果光束远小于孔径，几乎无截断
        if w < 0.5 * r_aperture:
            return GaussianBeam(
                wavelength=input_beam.wavelength,
                waist=input_beam.waist,
                waist_position=input_beam.waist_position,
                power=input_beam.power,
            )

        # 近似: 截断后等效束腰减小
        # 使用 1/e^2 功率截断处的等效半径
        truncation_ratio = r_aperture / w
        if truncation_ratio < 1.0:
            # 截断因子: 功率通过比例
            power_fraction = 1.0 - np.exp(-2.0 * truncation_ratio ** 2)
            effective_waist = input_beam.waist * min(truncation_ratio, 0.99)
            return GaussianBeam(
                wavelength=input_beam.wavelength,
                waist=effective_waist,
                waist_position=input_beam.waist_position,
                power=input_beam.power * power_fraction,
            )

        return GaussianBeam(
            wavelength=input_beam.wavelength,
            waist=input_beam.waist,
            waist_position=input_beam.waist_position,
            power=input_beam.power,
        )

    def get_psf(self, grid_size: int = 256, pixel_scale: float = 1e-6) -> np.ndarray:
        """计算圆孔衍射 PSF (Airy 斑)。

        Parameters
        ----------
        grid_size : int
            网格尺寸。
        pixel_scale : float
            像素尺度 (m/pixel)。

        Returns
        -------
        np.ndarray
            2D PSF 数组 (归一化)。
        """
        coords = np.arange(grid_size, dtype=np.float64) - grid_size // 2
        xx, yy = np.meshgrid(coords, coords)
        r = np.sqrt(xx ** 2 + yy ** 2) * pixel_scale

        # Airy 斑参数
        x = np.pi * self.diameter * r / (632.8e-9 * 0.1)  # 假设 f=0.1m
        x = np.where(x < 1e-10, 1e-10, x)

        airy = ThinLens._airy_function(x)
        psf = airy ** 2
        psf /= psf.max() + 1e-30

        return psf


class Obstruction:
    """圆形遮挡 (中心遮拦)。

    Parameters
    ----------
    diameter : float
        遮挡直径 (m)。
    name : str
        元件名称。
    """

    def __init__(self, diameter: float, name: str = "obstruction"):
        if diameter < 0:
            raise ValueError(f"遮挡直径必须 >= 0，当前值: {diameter}")
        self.diameter = float(diameter)
        self.name = name

    def propagate_beam(self, input_beam: GaussianBeam) -> GaussianBeam:
        """传播光束通过遮挡。

        遮挡中心区域，减少通过的光功率。

        Parameters
        ----------
        input_beam : GaussianBeam
            输入光束。

        Returns
        -------
        GaussianBeam
            输出光束。
        """
        w = input_beam.beam_radius(0.0)
        r_obstruction = self.diameter / 2.0

        if r_obstruction <= 0:
            return GaussianBeam(
                wavelength=input_beam.wavelength,
                waist=input_beam.waist,
                waist_position=input_beam.waist_position,
                power=input_beam.power,
            )

        # 被遮挡的功率比例
        blocked_fraction = 1.0 - np.exp(-2.0 * r_obstruction ** 2 / w ** 2)
        transmitted_power = input_beam.power * (1.0 - blocked_fraction)

        return GaussianBeam(
            wavelength=input_beam.wavelength,
            waist=input_beam.waist,
            waist_position=input_beam.waist_position,
            power=max(transmitted_power, 0.0),
        )

    def get_psf(self, grid_size: int = 256, pixel_scale: float = 1e-6) -> np.ndarray:
        """计算遮挡的传递函数 (中心不透光)。

        Parameters
        ----------
        grid_size : int
            网格尺寸。
        pixel_scale : float
            像素尺度。

        Returns
        -------
        np.ndarray
            2D 传递函数。
        """
        coords = np.arange(grid_size, dtype=np.float64) - grid_size // 2
        xx, yy = np.meshgrid(coords, coords)
        r = np.sqrt(xx ** 2 + yy ** 2) * pixel_scale

        # 透射函数: 中心遮挡
        transmission = np.ones((grid_size, grid_size), dtype=np.float64)
        transmission[r < self.diameter / 2.0] = 0.0

        return transmission


# ======================== 探测器模型 ========================


class DetectorModel:
    """探测器/相机模型。

    模拟 CCD/CMOS 探测器的完整成像过程，包括:
    - 光子收集 (量子效率)
    - 散粒噪声 (泊松分布)
    - 暗电流
    - 读出噪声 (高斯分布)
    - ADC 量化
    - 像素填充因子

    Parameters
    ----------
    pixel_size : float
        像素尺寸 (m)。
    resolution : Tuple[int, int]
        分辨率 (宽, 高)。
    read_noise_e : float
        读出噪声 (电子 RMS)。
    dark_current_e_per_s : float
        暗电流 (电子/秒/像素)。
    quantization_bits : int
        ADC 量化位数。
    quantum_efficiency : float
        量子效率 [0, 1]。
    fill_factor : float
        填充因子 [0, 1]。
    full_well_e : float
        满阱容量 (电子)。
    exposure_time : float
        曝光时间 (秒)。
    """

    def __init__(
        self,
        pixel_size: float = 5.0e-6,
        resolution: Tuple[int, int] = (640, 480),
        read_noise_e: float = 2.0,
        dark_current_e_per_s: float = 0.1,
        quantization_bits: int = 8,
        quantum_efficiency: float = 0.8,
        fill_factor: float = 0.95,
        full_well_e: float = 20000.0,
        exposure_time: float = 0.01,
    ):
        if pixel_size <= 0:
            raise ValueError(f"像素尺寸必须 > 0，当前值: {pixel_size}")
        if read_noise_e < 0:
            raise ValueError(f"读出噪声必须 >= 0，当前值: {read_noise_e}")
        if not (0 <= quantum_efficiency <= 1):
            raise ValueError(f"量子效率必须在 [0, 1] 范围内，当前值: {quantum_efficiency}")

        self.pixel_size = float(pixel_size)
        self.resolution = resolution
        self.read_noise_e = float(read_noise_e)
        self.dark_current_e_per_s = float(dark_current_e_per_s)
        self.quantization_bits = int(quantization_bits)
        self.quantum_efficiency = float(quantum_efficiency)
        self.fill_factor = float(fill_factor)
        self.full_well_e = float(full_well_e)
        self.exposure_time = float(exposure_time)

        self._rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        """设置随机种子。

        Parameters
        ----------
        seed : int
            随机种子。
        """
        self._rng = np.random.default_rng(seed)

    def photons_to_electrons(
        self,
        photon_count: np.ndarray,
    ) -> np.ndarray:
        """光子数转电子数 (考虑量子效率)。

        Parameters
        ----------
        photon_count : np.ndarray
            光子数数组。

        Returns
        -------
        np.ndarray
            电子数数组。
        """
        # 量子效率: 每个光子产生电子的概率
        # 使用泊松分布模拟量子效率的统计特性
        mean_electrons = photon_count * self.quantum_efficiency
        electrons = self._rng.poisson(np.maximum(mean_electrons, 0).astype(np.float64))
        return electrons.astype(np.float64)

    def add_dark_current(self, shape: Tuple[int, int]) -> np.ndarray:
        """生成暗电流噪声。

        Parameters
        ----------
        shape : Tuple[int, int]
            图像形状 (高, 宽)。

        Returns
        -------
        np.ndarray
            暗电流电子数。
        """
        mean_dark = self.dark_current_e_per_s * self.exposure_time
        dark = self._rng.poisson(mean_dark, size=shape).astype(np.float64)
        return dark

    def add_read_noise(self, shape: Tuple[int, int]) -> np.ndarray:
        """生成读出噪声。

        Parameters
        ----------
        shape : Tuple[int, int]
            图像形状 (高, 宽)。

        Returns
        -------
        np.ndarray
            读出噪声 (电子)。
        """
        return self._rng.normal(0.0, self.read_noise_e, size=shape)

    def quantize(self, signal_e: np.ndarray) -> np.ndarray:
        """ADC 量化。

        Parameters
        ----------
        signal_e : np.ndarray
            信号 (电子)。

        Returns
        -------
        np.ndarray
            量化后的数字值 (DN)。
        """
        max_dn = 2 ** self.quantization_bits - 1
        gain = self.full_well_e / max_dn  # 电子/DN
        dn = np.round(signal_e / gain).astype(np.float64)
        return np.clip(dn, 0, max_dn)

    def apply_fill_factor(self, image: np.ndarray) -> np.ndarray:
        """应用像素填充因子。

        通过轻微模糊模拟填充因子 < 1 的效果。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        np.ndarray
            填充因子调制后的图像。
        """
        if self.fill_factor >= 0.99:
            return image.copy()

        # 填充因子效应: 有效面积减小 -> 信号减小 + 微小串扰
        effective_signal = image * self.fill_factor

        # 微小串扰 (相邻像素间的电荷扩散)
        kernel_size = 3
        blur_amount = (1.0 - self.fill_factor) * 0.3
        if blur_amount > 0.01:
            effective_signal = effective_signal + blur_amount * cv2.GaussianBlur(
                effective_signal, (kernel_size, kernel_size), 0.5
            )

        return effective_signal

    def capture(
        self,
        psf_image: np.ndarray,
        beam_offset_x: float = 0.0,
        beam_offset_y: float = 0.0,
    ) -> np.ndarray:
        """捕获一帧图像。

        完整模拟: PSF 采样 -> 量子效率 -> 暗电流 -> 读出噪声 -> 量化。

        Parameters
        ----------
        psf_image : np.ndarray
            归一化 PSF 图像 [0, 1]。
        beam_offset_x : float
            光束 X 偏移 (像素)。
        beam_offset_y : float
            光束 Y 偏移 (像素)。

        Returns
        -------
        np.ndarray
            捕获的图像 (DN)。
        """
        h, w = self.resolution

        # 将 PSF 放置到探测器上
        # 计算总光子数 (基于功率和曝光时间)
        total_photons = self._estimate_total_photons(psf_image)

        # 创建探测器图像
        detector_frame = np.zeros((h, w), dtype=np.float64)

        # 将 PSF 图像偏移后放到探测器上
        psf_h, psf_w = psf_image.shape
        # 亚像素偏移使用仿射变换
        offset_matrix = np.float32([
            [1, 0, beam_offset_x],
            [0, 1, beam_offset_y],
        ])
        shifted_psf = cv2.warpAffine(
            psf_image, offset_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # 光子数
        photon_map = shifted_psf * total_photons

        # 量子效率 + 散粒噪声
        electrons = self.photons_to_electrons(photon_map)

        # 暗电流
        electrons += self.add_dark_current((h, w))

        # 读出噪声
        electrons += self.add_read_noise((h, w))

        # 填充因子
        electrons = self.apply_fill_factor(electrons)

        # 饱和截断
        electrons = np.clip(electrons, 0, self.full_well_e)

        # ADC 量化
        digital_image = self.quantize(electrons)

        return digital_image

    def _estimate_total_photons(self, psf_image: np.ndarray) -> float:
        """估计总光子数。

        基于 PSF 图像和典型参数估计每个像素的光子数。

        Parameters
        ----------
        psf_image : np.ndarray
            归一化 PSF 图像。

        Returns
        -------
        float
            总光子数。
        """
        # 典型参数: 1mW HeNe 激光, 10ms 曝光, 5um 像素
        # 峰值光子数约 10000-50000 电子 (满阱的 50%-250%)
        # 这里使用一个合理的默认值
        peak_electrons = self.full_well_e * 0.5
        psf_sum = psf_image.sum()
        if psf_sum < 1e-30:
            return 0.0

        # 总电子数 = 峰值电子数 * (总PSF能量 / 峰值PSF)
        peak_psf = psf_image.max()
        if peak_psf < 1e-30:
            return 0.0

        total_electrons = peak_electrons * psf_sum / peak_psf
        # 转换为光子数 (除以量子效率)
        total_photons = total_electrons / self.quantum_efficiency

        return total_photons

    def generate_spot_image(
        self,
        center_x: float,
        center_y: float,
        sigma: float = 2.0,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """生成带噪声的光斑图像。

        Parameters
        ----------
        center_x : float
        center_y : float
            光斑中心 (像素)。
        sigma : float
            高斯宽度 (像素)。
        amplitude : float
            峰值幅度 (归一化)。

        Returns
        -------
        np.ndarray
            生成的图像 (DN)。
        """
        h, w = self.resolution
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

        # 高斯 PSF
        psf = amplitude * np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma ** 2)
        )

        # 归一化
        if psf.max() > 0:
            psf /= psf.max()

        return self.capture(psf, beam_offset_x=0.0, beam_offset_y=0.0)


# ======================== 电机平台模型 ========================


class MotorStageModel:
    """电机平台模型。

    模拟步进/直流电机驱动的精密位移台，包括:
    - 步进分辨率
    - 回程差 (死区 + 滞后)
    - 速度和加速度限制
    - 位置量化
    - 命令响应延迟
    - 位置噪声 (振动耦合)

    Parameters
    ----------
    step_size : float
        步进尺寸 (um/步)。
    backlash_steps : int
        回程差 (步数)。
    max_velocity : float
        最大速度 (步/秒)。
    max_acceleration : float
        最大加速度 (步/秒^2)。
    direction_sign : Tuple[int, int]
        方向符号 (X, Y)。
    command_delay_steps : int
        命令延迟 (步数)。
    position_noise_std : float
        位置噪声标准差 (um)。
    """

    def __init__(
        self,
        step_size: float = 0.5,
        backlash_steps: int = 2,
        max_velocity: float = 5000.0,
        max_acceleration: float = 50000.0,
        direction_sign: Tuple[int, int] = (1, 1),
        command_delay_steps: int = 0,
        position_noise_std: float = 0.05,
    ):
        if step_size <= 0:
            raise ValueError(f"步进尺寸必须 > 0，当前值: {step_size}")
        if max_velocity <= 0:
            raise ValueError(f"最大速度必须 > 0，当前值: {max_velocity}")

        self.step_size = float(step_size)
        self.backlash_steps = int(backlash_steps)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.direction_sign = direction_sign
        self.command_delay_steps = int(command_delay_steps)
        self.position_noise_std = float(position_noise_std)

        # 状态
        self._position_x: float = 0.0  # 当前 X 位置 (步)
        self._position_y: float = 0.0  # 当前 Y 位置 (步)
        self._velocity_x: float = 0.0  # 当前 X 速度 (步/s)
        self._velocity_y: float = 0.0  # 当前 Y 速度 (步/s)
        self._last_direction_x: int = 0  # 上次 X 运动方向
        self._last_direction_y: int = 0  # 上次 Y 运动方向
        self._command_queue_x: List[int] = []  # X 命令队列
        self._command_queue_y: List[int] = []  # Y 命令队列
        self._total_steps_x: int = 0
        self._total_steps_y: int = 0

        self._rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        """设置随机种子。

        Parameters
        ----------
        seed : int
            随机种子。
        """
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """重置电机状态。"""
        self._position_x = 0.0
        self._position_y = 0.0
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._last_direction_x = 0
        self._last_direction_y = 0
        self._command_queue_x.clear()
        self._command_queue_y.clear()
        self._total_steps_x = 0
        self._total_steps_y = 0

    @property
    def position_x(self) -> float:
        """当前 X 位置 (步)。"""
        return self._position_x

    @property
    def position_y(self) -> float:
        """当前 Y 位置 (步)。"""
        return self._position_y

    @property
    def position_um(self) -> Tuple[float, float]:
        """当前位置 (um)。"""
        return (
            self._position_x * self.step_size,
            self._position_y * self.step_size,
        )

    @property
    def total_steps(self) -> Tuple[int, int]:
        """总步数 (X, Y)。"""
        return (self._total_steps_x, self._total_steps_y)

    def move(
        self,
        steps_x: int,
        steps_y: int,
        dt: float = 0.01,
    ) -> Tuple[float, float]:
        """执行移动命令。

        Parameters
        ----------
        steps_x : int
            X 方向步数 (正=正向)。
        steps_y : int
            Y 方向步数 (正=正向)。
        dt : float
            时间步长 (秒)。

        Returns
        -------
        Tuple[float, float]
            实际移动后的位置 (步)。
        """
        # 应用方向符号
        actual_steps_x = int(steps_x * self.direction_sign[0])
        actual_steps_y = int(steps_y * self.direction_sign[1])

        # 速度限制
        max_steps_per_dt = self.max_velocity * dt
        actual_steps_x = int(np.clip(actual_steps_x, -max_steps_per_dt, max_steps_per_dt))
        actual_steps_y = int(np.clip(actual_steps_y, -max_steps_per_dt, max_steps_per_dt))

        # 回程差模型
        actual_steps_x = self._apply_backlash_x(actual_steps_x)
        actual_steps_y = self._apply_backlash_y(actual_steps_y)

        # 位置量化
        actual_steps_x = int(np.round(actual_steps_x))
        actual_steps_y = int(np.round(actual_steps_y))

        # 更新位置
        self._position_x += actual_steps_x
        self._position_y += actual_steps_y

        # 更新速度 (用于加速度限制)
        if dt > 0:
            self._velocity_x = actual_steps_x / dt
            self._velocity_y = actual_steps_y / dt

        # 累计步数
        self._total_steps_x += abs(actual_steps_x)
        self._total_steps_y += abs(actual_steps_y)

        # 添加位置噪声
        noise_x = self._rng.normal(0, self.position_noise_std / self.step_size)
        noise_y = self._rng.normal(0, self.position_noise_std / self.step_size)

        return (
            self._position_x + noise_x,
            self._position_y + noise_y,
        )

    def _apply_backlash_x(self, steps: int) -> int:
        """应用 X 方向回程差。

        Parameters
        ----------
        steps : int
            目标步数。

        Returns
        -------
        int
        考虑回程差后的实际步数。
        """
        if self.backlash_steps <= 0:
            return steps

        current_direction = 1 if steps > 0 else (-1 if steps < 0 else 0)

        if current_direction != 0 and current_direction != self._last_direction_x:
            # 方向反转: 损失回程差步数
            lost = min(abs(steps), self.backlash_steps)
            effective_steps = steps - current_direction * lost
            self._last_direction_x = current_direction
            return effective_steps

        if current_direction != 0:
            self._last_direction_x = current_direction

        return steps

    def _apply_backlash_y(self, steps: int) -> int:
        """应用 Y 方向回程差。

        Parameters
        ----------
        steps : int
            目标步数。

        Returns
        -------
        int
            考虑回程差后的实际步数。
        """
        if self.backlash_steps <= 0:
            return steps

        current_direction = 1 if steps > 0 else (-1 if steps < 0 else 0)

        if current_direction != 0 and current_direction != self._last_direction_y:
            lost = min(abs(steps), self.backlash_steps)
            effective_steps = steps - current_direction * lost
            self._last_direction_y = current_direction
            return effective_steps

        if current_direction != 0:
            self._last_direction_y = current_direction

        return steps

    def move_to_position(
        self,
        target_x: float,
        target_y: float,
        dt: float = 0.01,
    ) -> Tuple[float, float]:
        """移动到目标位置。

        Parameters
        ----------
        target_x : float
            目标 X 位置 (步)。
        target_y : float
            目标 Y 位置 (步)。
        dt : float
            时间步长 (秒)。

        Returns
        -------
        Tuple[float, float]
            实际位置 (步)。
        """
        steps_x = int(round(target_x - self._position_x))
        steps_y = int(round(target_y - self._position_y))
        return self.move(steps_x, steps_y, dt)


# ======================== 环境扰动模型 ========================


class EnvironmentModel:
    """环境扰动模型。

    模拟多种环境因素对光斑位置的影响:
    - 多频振动 (正弦叠加)
    - 布朗运动漂移 (随机游走)
    - 热漂移 (慢指数衰减)
    - 大气湍流 (简化 Kolmogorov 相位屏)

    Parameters
    ----------
    vibration_amplitude : float
        振动总幅度 (像素)。
    vibration_frequencies : List[float]
        振动频率列表 (Hz)。
    vibration_phases : Optional[List[float]]
        振动初始相位 (弧度)。
    drift_rate : float
        漂移速率 (像素/秒)。
    thermal_amplitude : float
        热漂移幅度 (像素)。
    thermal_time_constant : float
        热漂移时间常数 (秒)。
    enable_turbulence : bool
        是否启用湍流。
    turbulence_strength : float
        湍流强度 [0, 1]。
    """

    def __init__(
        self,
        vibration_amplitude: float = 0.5,
        vibration_frequencies: Optional[List[float]] = None,
        vibration_phases: Optional[List[float]] = None,
        drift_rate: float = 0.1,
        thermal_amplitude: float = 2.0,
        thermal_time_constant: float = 300.0,
        enable_turbulence: bool = False,
        turbulence_strength: float = 0.1,
    ):
        self.vibration_amplitude = float(vibration_amplitude)
        self.vibration_frequencies = vibration_frequencies or [1.0, 10.0, 50.0]
        self.drift_rate = float(drift_rate)
        self.thermal_amplitude = float(thermal_amplitude)
        self.thermal_time_constant = float(thermal_time_constant)
        self.enable_turbulence = enable_turbulence
        self.turbulence_strength = float(turbulence_strength)

        # 初始化振动相位
        if vibration_phases is not None and len(vibration_phases) == len(self.vibration_frequencies):
            self.vibration_phases = [float(p) for p in vibration_phases]
        else:
            self._rng = np.random.default_rng()
            self.vibration_phases = [
                float(self._rng.uniform(0, 2 * np.pi))
                for _ in self.vibration_frequencies
            ]

        # 振动幅度分配: 总幅度均分给各频率分量
        n_freq = len(self.vibration_frequencies)
        self.vibration_component_amplitudes = [
            self.vibration_amplitude / n_freq for _ in range(n_freq)
        ]

        # 状态变量
        self._drift_x: float = 0.0
        self._drift_y: float = 0.0
        self._thermal_x: float = 0.0
        self._thermal_y: float = 0.0
        self._thermal_target_x: float = 0.0
        self._thermal_target_y: float = 0.0
        self._turbulence_phase_screen: Optional[np.ndarray] = None

        # 内部随机数生成器
        self._rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        """设置随机种子。

        Parameters
        ----------
        seed : int
            随机种子。
        """
        self._rng = np.random.default_rng(seed)
        self.vibration_phases = [
            float(self._rng.uniform(0, 2 * np.pi))
            for _ in self.vibration_frequencies
        ]

    def reset(self) -> None:
        """重置扰动状态。"""
        self._drift_x = 0.0
        self._drift_y = 0.0
        self._thermal_x = 0.0
        self._thermal_y = 0.0
        self._thermal_target_x = 0.0
        self._thermal_target_y = 0.0
        self._turbulence_phase_screen = None

    def get_vibration(self, t: float) -> Tuple[float, float]:
        """计算 t 时刻的振动位移。

        Parameters
        ----------
        t : float
            时间 (秒)。

        Returns
        -------
        Tuple[float, float]
            (x, y) 振动位移 (像素)。
        """
        vib_x = 0.0
        vib_y = 0.0

        for i, freq in enumerate(self.vibration_frequencies):
            amp = self.vibration_component_amplitudes[i]
            phase = self.vibration_phases[i]
            # X 和 Y 使用不同相位偏移 (模拟椭圆振动)
            vib_x += amp * np.sin(2.0 * np.pi * freq * t + phase)
            vib_y += amp * np.sin(2.0 * np.pi * freq * t + phase + np.pi / 3.0)

        return vib_x, vib_y

    def get_drift(self, t: float, dt: float) -> Tuple[float, float]:
        """计算 t 时刻的漂移位移 (布朗运动)。

        Parameters
        ----------
        t : float
            时间 (秒)。
        dt : float
            时间步长 (秒)。

        Returns
        -------
        Tuple[float, float]
            (x, y) 漂移位移 (像素)。
        """
        # 布朗运动: dx = drift_rate * sqrt(dt) * N(0,1)
        noise_x = self._rng.normal(0, 1)
        noise_y = self._rng.normal(0, 1)

        self._drift_x += self.drift_rate * np.sqrt(dt) * noise_x
        self._drift_y += self.drift_rate * np.sqrt(dt) * noise_y

        return self._drift_x, self._drift_y

    def get_thermal_drift(self, t: float, dt: float) -> Tuple[float, float]:
        """计算 t 时刻的热漂移。

        使用 Ornstein-Uhlenbeck 过程模拟热漂移:
        dx = -x/tau * dt + sigma * sqrt(dt) * N(0,1)

        Parameters
        ----------
        t : float
            时间 (秒)。
        dt : float
            时间步长 (秒)。

        Returns
        -------
        Tuple[float, float]
            (x, y) 热漂移位移 (像素)。
        """
        tau = self.thermal_time_constant
        sigma = self.thermal_amplitude / np.sqrt(tau)

        # Ornstein-Uhlenbeck 过程
        self._thermal_x += (-self._thermal_x / tau) * dt + sigma * np.sqrt(dt) * self._rng.normal(0, 1)
        self._thermal_y += (-self._thermal_y / tau) * dt + sigma * np.sqrt(dt) * self._rng.normal(0, 1)

        return self._thermal_x, self._thermal_y

    def get_turbulence(self, t: float) -> Tuple[float, float]:
        """计算 t 时刻的湍流引起的位移。

        使用简化的 Kolmogorov 湍流模型。

        Parameters
        ----------
        t : float
            时间 (秒)。

        Returns
        -------
        Tuple[float, float]
            (x, y) 湍流位移 (像素)。
        """
        if not self.enable_turbulence:
            return 0.0, 0.0

        # 简化模型: 多尺度随机波动
        turb_x = self.turbulence_strength * self._rng.normal(0, 1)
        turb_y = self.turbulence_strength * self._rng.normal(0, 1)

        return turb_x, turb_y

    def get_total_disturbance(
        self,
        t: float,
        dt: float,
    ) -> Tuple[float, float]:
        """计算 t 时刻的总扰动。

        Parameters
        ----------
        t : float
            时间 (秒)。
        dt : float
            时间步长 (秒)。

        Returns
        -------
        Tuple[float, float]
            (x, y) 总扰动位移 (像素)。
        """
        vib_x, vib_y = self.get_vibration(t)
        drift_x, drift_y = self.get_drift(t, dt)
        thermal_x, thermal_y = self.get_thermal_drift(t, dt)
        turb_x, turb_y = self.get_turbulence(t)

        total_x = vib_x + drift_x + thermal_x + turb_x
        total_y = vib_y + drift_y + thermal_y + turb_y

        return total_x, total_y

    def generate_phase_screen(
        self,
        size: int = 256,
        fried_parameter: float = 0.1,
        wavelength: float = 632.8e-9,
    ) -> np.ndarray:
        """生成简化的大气湍流相位屏。

        使用 Kolmogorov 功率谱密度的 FFT 方法。

        Parameters
        ----------
        size : int
            相位屏尺寸。
        fried_parameter : float
            Fried 参数 r0 (m)。
        wavelength : float
            波长 (m)。

        Returns
        -------
        np.ndarray
            相位屏 (弧度)。
        """
        # 频率网格
        freq = np.fft.fftfreq(size)
        fx, fy = np.meshgrid(freq, freq)
        f = np.sqrt(fx ** 2 + fy ** 2)
        f[0, 0] = 1e-10  # 避免除零

        # Kolmogorov 功率谱密度: PSD(f) = 0.023 * r0^(-5/3) * f^(-11/3)
        psd = 0.023 * fried_parameter ** (-5.0 / 3.0) * f ** (-11.0 / 3.0)
        psd[0, 0] = 0.0  # 去除直流分量

        # 生成随机相位
        random_phase = np.exp(2j * np.pi * self._rng.random((size, size)))

        # 相位屏 = IFFT(sqrt(PSD) * random_phase)
        phase_screen = np.real(np.fft.ifft2(np.sqrt(psd) * random_phase))

        # 归一化到波长
        phase_screen *= 2.0 * np.pi / wavelength * self.turbulence_strength

        return phase_screen


# ======================== 数字孪生仿真器 ========================


class DigitalTwinSimulator:
    """数字孪生仿真器。

    完整的光学对准系统数字孪生仿真环境，模拟从光源到探测器的
    全链路光学传播，包括环境扰动和电机控制。

    灵感来源:
    - OOPAO: 端到端自适应光学仿真
    - lowfssim: NASA JPL 光学模型
    - HCIPy: 光学传播框架

    Parameters
    ----------
    config : SimulationConfig or None
        仿真配置。为 None 时使用默认配置。

    Examples
    --------
    >>> config = SimulationConfig(beam_wavelength_nm=632.8)
    >>> sim = DigitalTwinSimulator(config)
    >>> result = sim.run_closed_loop(max_iterations=100)
    >>> print(f"最终误差: {result.final_error_px:.2f} 像素")
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()

        # 随机种子
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # 初始化子模型
        self._init_submodels()

        # 仿真状态
        self.state = SimulationState()
        self.state.target_position_x = self.config.detector_resolution[0] / 2.0
        self.state.target_position_y = self.config.detector_resolution[1] / 2.0

        # 光学链路
        self._optical_elements: List[OpticalElement] = []
        self._setup_optical_path()

        # PSF 缓存
        self._psf_cache: Optional[np.ndarray] = None
        self._psf_dirty: bool = True

        LOGGER.info("数字孪生仿真器初始化完成")

    def _init_submodels(self) -> None:
        """初始化子模型。"""
        cfg = self.config

        # 高斯光束
        self.beam = GaussianBeam(
            wavelength=cfg.beam_wavelength_nm * 1e-9,
            waist=cfg.beam_waist_um * 1e-6,
            power=cfg.beam_power_mw * 1e-3,
        )

        # 探测器
        self.detector = DetectorModel(
            pixel_size=cfg.detector_pixel_size_um * 1e-6,
            resolution=cfg.detector_resolution,
            read_noise_e=cfg.detector_read_noise_e,
            dark_current_e_per_s=cfg.detector_dark_current_e_per_s,
            quantization_bits=cfg.detector_quantization_bits,
            quantum_efficiency=cfg.detector_quantum_efficiency,
            fill_factor=cfg.detector_fill_factor,
            full_well_e=cfg.detector_full_well_e,
            exposure_time=cfg.simulation_dt_s,
        )

        # 电机
        self.motor = MotorStageModel(
            step_size=cfg.motor_step_size_um,
            backlash_steps=cfg.motor_backlash_steps,
            max_velocity=cfg.motor_max_velocity_steps_s,
            max_acceleration=cfg.motor_max_acceleration_steps_s2,
            direction_sign=cfg.motor_direction_sign,
            command_delay_steps=cfg.motor_command_delay_steps,
            position_noise_std=cfg.motor_position_noise_std_um,
        )

        # 环境
        self.environment = EnvironmentModel(
            vibration_amplitude=cfg.vibration_amplitude_px,
            vibration_frequencies=cfg.vibration_frequencies_hz,
            vibration_phases=cfg.vibration_phases_rad,
            drift_rate=cfg.drift_rate_px_per_s,
            thermal_amplitude=cfg.thermal_drift_amplitude_px,
            thermal_time_constant=cfg.thermal_drift_time_constant_s,
            enable_turbulence=cfg.enable_turbulence,
            turbulence_strength=cfg.turbulence_strength,
        )

        # 设置随机种子
        if cfg.seed is not None:
            self.detector.seed(cfg.seed)
            self.motor.seed(cfg.seed)
            self.environment.seed(cfg.seed)

    def _setup_optical_path(self) -> None:
        """设置默认光学路径。"""
        cfg = self.config
        self._optical_elements = [
            FreeSpace(distance=0.05, name="source_to_lens"),  # 光源到透镜 50mm
            ThinLens(
                focal_length=cfg.lens_focal_length_mm * 1e-3,
                diameter=cfg.aperture_diameter_mm * 1e-3,
                name="focusing_lens",
            ),
            FreeSpace(distance=cfg.lens_focal_length_mm * 1e-3, name="lens_to_detector"),
        ]
        self._psf_dirty = True

    def set_optical_elements(self, elements: List[OpticalElement]) -> None:
        """自定义光学路径。

        Parameters
        ----------
        elements : List[OpticalElement]
            光学元件列表 (按传播顺序)。
        """
        self._optical_elements = list(elements)
        self._psf_dirty = True
        LOGGER.info(f"光学路径已更新: {len(elements)} 个元件")

    def compute_psf(
        self,
        grid_size: int = 128,
        pixel_scale: Optional[float] = None,
    ) -> np.ndarray:
        """计算系统 PSF。

        Parameters
        ----------
        grid_size : int
            计算网格尺寸。
        pixel_scale : float or None
            像素尺度 (m/pixel)。为 None 时自动计算。

        Returns
        -------
        np.ndarray
            2D PSF 数组 (归一化)。
        """
        if pixel_scale is None:
            pixel_scale = self.config.detector_pixel_size_um * 1e-6

        # 传播光束通过所有光学元件
        current_beam = self.beam
        for element in self._optical_elements:
            current_beam = element.propagate_beam(current_beam)

        # 计算探测器上的光斑大小
        # 使用最后一个透镜的衍射极限 PSF
        psf = self._compute_diffraction_psf(grid_size, pixel_scale, current_beam)

        self._psf_cache = psf
        self._psf_dirty = False

        return psf

    def _compute_diffraction_psf(
        self,
        grid_size: int,
        pixel_scale: float,
        beam: GaussianBeam,
    ) -> np.ndarray:
        """计算衍射极限 PSF。

        Parameters
        ----------
        grid_size : int
            网格尺寸。
        pixel_scale : float
            像素尺度 (m/pixel)。
        beam : GaussianBeam
            传播后的光束。

        Returns
        -------
        np.ndarray
            2D PSF 数组 (归一化)。
        """
        coords = np.arange(grid_size, dtype=np.float64) - grid_size // 2
        xx, yy = np.meshgrid(coords, coords)
        r_px = np.sqrt(xx ** 2 + yy ** 2)
        r_m = r_px * pixel_scale

        # 高斯光束在探测器上的光斑大小
        # w_det = beam.waist (已通过传播计算)
        w_det = beam.beam_radius(0.0)
        sigma_px = w_det / pixel_scale

        if sigma_px < 0.1:
            sigma_px = 0.5  # 最小光斑尺寸

        # 高斯 PSF
        psf = np.exp(-r_px ** 2 / (2.0 * sigma_px ** 2))

        # 叠加衍射效应 (如果有孔径限制)
        aperture = self.config.aperture_diameter_mm * 1e-3
        if aperture > 0:
            # Airy 斑宽度: 1.22 * lambda * f / D
            airy_radius_px = 1.22 * self.config.beam_wavelength_nm * 1e-9 * \
                self.config.lens_focal_length_mm * 1e-3 / aperture / pixel_scale
            if airy_radius_px > 0.1:
                airy = self._airy_pattern(r_px, airy_radius_px)
                # 混合高斯和 Airy (权重取决于截断比)
                truncation = aperture / (2.0 * w_det) if w_det > 0 else 1.0
                if truncation < 2.0:
                    blend = np.clip(1.0 - truncation / 2.0, 0.0, 1.0)
                    psf = (1.0 - blend) * psf + blend * airy

        # 归一化
        psf_sum = psf.sum()
        if psf_sum > 0:
            psf /= psf_sum

        return psf

    @staticmethod
    def _airy_pattern(r: np.ndarray, scale: float) -> np.ndarray:
        """计算 Airy 斑图案。

        Parameters
        ----------
        r : np.ndarray
            径向距离 (像素)。
        scale : float
            Airy 斑特征半径 (像素)。

        Returns
        -------
        np.ndarray
            Airy 斑图案 (归一化)。
        """
        x = np.pi * r / scale
        x = np.where(x < 1e-10, 1e-10, x)
        airy = ThinLens._airy_function(x)
        pattern = airy ** 2
        pmax = pattern.max()
        if pmax > 0:
            pattern /= pmax
        return pattern

    def generate_frame(
        self,
        beam_offset_x: float = 0.0,
        beam_offset_y: float = 0.0,
    ) -> np.ndarray:
        """生成一帧仿真图像。

        Parameters
        ----------
        beam_offset_x : float
            光束 X 偏移 (像素)。
        beam_offset_y : float
            光束 Y 偏移 (像素)。

        Returns
        -------
        np.ndarray
            仿真图像 (DN)。
        """
        # 获取或计算 PSF
        if self._psf_cache is None or self._psf_dirty:
            self.compute_psf()

        psf = self._psf_cache
        if psf is None:
            raise RuntimeError("PSF 计算失败")

        # 使用探测器模型捕获图像
        frame = self.detector.capture(
            psf,
            beam_offset_x=beam_offset_x,
            beam_offset_y=beam_offset_y,
        )

        return frame

    def step(
        self,
        motor_steps_x: int = 0,
        motor_steps_y: int = 0,
    ) -> SimulationState:
        """执行一步仿真。

        Parameters
        ----------
        motor_steps_x : int
            X 方向电机步数。
        motor_steps_y : int
            Y 方向电机步数。

        Returns
        -------
        SimulationState
            更新后的仿真状态。
        """
        dt = self.config.simulation_dt_s
        t = self.state.time_elapsed_s

        # 1. 电机移动
        self.motor.move(motor_steps_x, motor_steps_y, dt)
        motor_pos_x, motor_pos_y = self.motor.position_um

        # 2. 环境扰动
        disturb_x, disturb_y = self.environment.get_total_disturbance(t, dt)

        # 3. 计算光束在探测器上的位置
        # 光束位置 = 目标位置 + 电机偏移 + 扰动
        # 电机移动是负反馈: 电机正方向移动使光斑向目标靠近
        pixel_per_um = 1.0 / self.config.detector_pixel_size_um  # 像素/um (简化)
        motor_offset_x = -motor_pos_x * pixel_per_um * self.config.motor_direction_sign[0]
        motor_offset_y = -motor_pos_y * pixel_per_um * self.config.motor_direction_sign[1]

        beam_x = self.state.target_position_x + motor_offset_x + disturb_x
        beam_y = self.state.target_position_y + motor_offset_y + disturb_y

        # 4. 生成图像
        frame = self.generate_frame(
            beam_offset_x=beam_x - self.state.target_position_x,
            beam_offset_y=beam_y - self.state.target_position_y,
        )

        # 5. 更新状态
        self.state.beam_position_x = beam_x
        self.state.beam_position_y = beam_y
        self.state.motor_position_x = self.motor.position_x
        self.state.motor_position_y = self.motor.position_y
        self.state.current_frame = frame
        self.state.time_elapsed_s += dt
        self.state.iteration_count += 1

        return SimulationState(
            beam_position_x=self.state.beam_position_x,
            beam_position_y=self.state.beam_position_y,
            motor_position_x=self.state.motor_position_x,
            motor_position_y=self.state.motor_position_y,
            target_position_x=self.state.target_position_x,
            target_position_y=self.state.target_position_y,
            current_frame=self.state.current_frame.copy(),
            time_elapsed_s=self.state.time_elapsed_s,
            iteration_count=self.state.iteration_count,
        )

    def detect_spot(self, frame: np.ndarray) -> Tuple[float, float]:
        """从图像中检测光斑位置 (质心法)。

        Parameters
        ----------
        frame : np.ndarray
            输入图像。

        Returns
        -------
        Tuple[float, float]
            (x, y) 光斑质心位置 (像素)。
        """
        # 阈值分割
        threshold = np.max(frame) * 0.1
        binary = frame > threshold

        if binary.sum() < 4:
            LOGGER.warning("光斑检测失败: 未找到足够亮的区域")
            return self.config.detector_resolution[0] / 2.0, self.config.detector_resolution[1] / 2.0

        # 质心计算
        yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]].astype(np.float64)
        total = binary.sum()
        cx = np.sum(xx[binary] * frame[binary]) / (np.sum(frame[binary]) + 1e-30)
        cy = np.sum(yy[binary] * frame[binary]) / (np.sum(frame[binary]) + 1e-30)

        return float(cx), float(cy)

    def compute_error(
        self,
        spot_x: float,
        spot_y: float,
    ) -> Tuple[float, float]:
        """计算对准误差。

        Parameters
        ----------
        spot_x : float
            光斑 X 位置 (像素)。
        spot_y : float
            光斑 Y 位置 (像素)。

        Returns
        -------
        Tuple[float, float]
            (error_x, error_y) 误差 (像素)。
        """
        error_x = self.state.target_position_x - spot_x
        error_y = self.state.target_position_y - spot_y
        return error_x, error_y

    def error_to_motor_steps(
        self,
        error_x: float,
        error_y: float,
        gain: float = 1.0,
    ) -> Tuple[int, int]:
        """将像素误差转换为电机步数。

        Parameters
        ----------
        error_x : float
            X 方向误差 (像素)。
        error_y : float
            Y 方向误差 (像素)。
        gain : float
            控制增益。

        Returns
        -------
        Tuple[int, int]
            (steps_x, steps_y) 电机步数。
        """
        pixel_per_um = 1.0 / self.config.detector_pixel_size_um
        steps_per_pixel = pixel_per_um / self.config.motor_step_size_um

        steps_x = int(round(error_x * steps_per_pixel * gain))
        steps_y = int(round(error_y * steps_per_pixel * gain))

        return steps_x, steps_y

    def run_closed_loop(
        self,
        max_iterations: int = 1000,
        success_threshold_px: float = 1.0,
        gain: float = 1.0,
        gain_schedule: Optional[str] = None,
        record_frames: bool = True,
        frame_record_interval: int = 10,
    ) -> SimulationResult:
        """运行闭环对准仿真。

        仿真循环: 探测 -> 计算误差 -> 驱动电机 -> 更新场景

        Parameters
        ----------
        max_iterations : int
            最大迭代次数。
        success_threshold_px : float
            成功阈值 (像素)。
        gain : float
            初始控制增益。
        gain_schedule : str or None
            增益调度策略: 'constant', 'aggressive', 'conservative', 'adaptive'。
        record_frames : bool
            是否记录帧。
        frame_record_interval : int
            帧记录间隔。

        Returns
        -------
        SimulationResult
            仿真结果。
        """
        self.reset()

        trajectory_x: List[float] = []
        trajectory_y: List[float] = []
        error_trajectory: List[float] = []
        frames: List[np.ndarray] = []
        current_gain = gain
        converged = False
        convergence_iteration = -1

        for i in range(max_iterations):
            # 增益调度
            if gain_schedule == "aggressive":
                current_gain = gain * (1.0 + 0.5 * i / max(1, max_iterations))
            elif gain_schedule == "conservative":
                current_gain = gain * max(0.1, 1.0 - 0.5 * i / max(1, max_iterations))
            elif gain_schedule == "adaptive":
                # 自适应增益: 误差大时用大增益，误差小时用小增益
                if i > 0 and len(error_trajectory) > 0:
                    last_error = error_trajectory[-1]
                    current_gain = gain * min(2.0, max(0.1, last_error / 10.0))

            # 执行一步仿真 (无电机移动，先获取当前状态)
            state = self.step(0, 0)

            # 检测光斑
            spot_x, spot_y = self.detect_spot(state.current_frame)

            # 计算误差
            error_x, error_y = self.compute_error(spot_x, spot_y)
            error_mag = np.sqrt(error_x ** 2 + error_y ** 2)

            # 记录
            trajectory_x.append(state.beam_position_x)
            trajectory_y.append(state.beam_position_y)
            error_trajectory.append(error_mag)

            # 记录帧
            if record_frames and (i % frame_record_interval == 0 or i == max_iterations - 1):
                frames.append(state.current_frame.copy())

            # 检查收敛
            if error_mag < success_threshold_px:
                converged = True
                convergence_iteration = i
                LOGGER.info(f"在第 {i} 步收敛，误差: {error_mag:.3f} 像素")
                break

            # 计算电机步数
            steps_x, steps_y = self.error_to_motor_steps(error_x, error_y, current_gain)

            # 执行电机移动 (下一步生效)
            self.motor.move(steps_x, steps_y, self.config.simulation_dt_s)

        # 构建结果
        convergence_time = (
            convergence_iteration * self.config.simulation_dt_s
            if converged
            else -1.0
        )
        final_error = error_trajectory[-1] if error_trajectory else float("inf")
        total_steps_x, total_steps_y = self.motor.total_steps

        # 计算指标
        metrics = self._compute_metrics(
            error_trajectory,
            converged,
            convergence_iteration,
            max_iterations,
        )

        result = SimulationResult(
            trajectory_x=trajectory_x,
            trajectory_y=trajectory_y,
            error_trajectory=error_trajectory,
            convergence_time_s=convergence_time,
            final_error_px=final_error,
            total_motor_steps_x=total_steps_x,
            total_motor_steps_y=total_steps_y,
            frames_recorded=frames,
            metrics_summary=metrics,
            success=converged,
        )

        LOGGER.info(
            f"仿真完成: {'成功' if converged else '未收敛'}, "
            f"最终误差: {final_error:.3f}px, "
            f"迭代: {len(error_trajectory)}"
        )

        return result

    @staticmethod
    def _compute_metrics(
        error_trajectory: List[float],
        converged: bool,
        convergence_iteration: int,
        max_iterations: int,
    ) -> Dict[str, float]:
        """计算仿真指标。

        Parameters
        ----------
        error_trajectory : List[float]
            误差轨迹。
        converged : bool
            是否收敛。
        convergence_iteration : int
            收敛迭代步。
        max_iterations : int
            最大迭代次数。

        Returns
        -------
        Dict[str, float]
            指标字典。
        """
        if not error_trajectory:
            return {}

        errors = np.array(error_trajectory)
        metrics: Dict[str, float] = {
            "initial_error_px": float(errors[0]),
            "final_error_px": float(errors[-1]),
            "mean_error_px": float(np.mean(errors)),
            "std_error_px": float(np.std(errors)),
            "max_error_px": float(np.max(errors)),
            "min_error_px": float(np.min(errors)),
            "error_reduction_ratio": float(errors[0] / (errors[-1] + 1e-30)),
            "convergence_rate": float(
                (errors[0] - errors[-1]) / (len(errors) + 1)
            ),
            "success": float(converged),
        }

        if converged and convergence_iteration > 0:
            # 收敛后的稳态误差
            steady_state = errors[convergence_iteration:]
            if len(steady_state) > 0:
                metrics["steady_state_error_mean"] = float(np.mean(steady_state))
                metrics["steady_state_error_std"] = float(np.std(steady_state))

        # 误差下降到 10% 的时间
        threshold_10pct = errors[0] * 0.1
        idx_10pct = np.where(errors < threshold_10pct)[0]
        if len(idx_10pct) > 0:
            metrics["time_to_10pct_error"] = float(idx_10pct[0])
        else:
            metrics["time_to_10pct_error"] = -1.0

        return metrics

    def run_open_loop(
        self,
        motor_commands: List[Tuple[int, int]],
    ) -> SimulationResult:
        """运行开环仿真。

        Parameters
        ----------
        motor_commands : List[Tuple[int, int]]
            电机命令序列 [(steps_x, steps_y), ...]。

        Returns
        -------
        SimulationResult
            仿真结果。
        """
        self.reset()

        trajectory_x: List[float] = []
        trajectory_y: List[float] = []
        error_trajectory: List[float] = []
        frames: List[np.ndarray] = []

        for steps_x, steps_y in motor_commands:
            state = self.step(steps_x, steps_y)

            error_x = self.state.target_position_x - state.beam_position_x
            error_y = self.state.target_position_y - state.beam_position_y
            error_mag = np.sqrt(error_x ** 2 + error_y ** 2)

            trajectory_x.append(state.beam_position_x)
            trajectory_y.append(state.beam_position_y)
            error_trajectory.append(error_mag)
            frames.append(state.current_frame.copy())

        final_error = error_trajectory[-1] if error_trajectory else float("inf")
        total_steps_x, total_steps_y = self.motor.total_steps

        return SimulationResult(
            trajectory_x=trajectory_x,
            trajectory_y=trajectory_y,
            error_trajectory=error_trajectory,
            final_error_px=final_error,
            total_motor_steps_x=total_steps_x,
            total_motor_steps_y=total_steps_y,
            frames_recorded=frames,
            success=final_error < 1.0,
        )

    def reset(self) -> None:
        """重置仿真状态。"""
        self.state = SimulationState()
        self.state.target_position_x = self.config.detector_resolution[0] / 2.0
        self.state.target_position_y = self.config.detector_resolution[1] / 2.0

        self.motor.reset()
        self.environment.reset()
        self._psf_dirty = True

        LOGGER.debug("仿真状态已重置")

    def set_initial_offset(
        self,
        offset_x: float,
        offset_y: float,
    ) -> None:
        """设置初始光束偏移。

        Parameters
        ----------
        offset_x : float
            X 方向初始偏移 (像素)。
        offset_y : float
            Y 方向初始偏移 (像素)。
        """
        self.state.beam_position_x = self.state.target_position_x + offset_x
        self.state.beam_position_y = self.state.target_position_y + offset_y
        LOGGER.debug(f"初始偏移设置: ({offset_x:.1f}, {offset_y:.1f}) 像素")

    def set_target_position(
        self,
        target_x: float,
        target_y: float,
    ) -> None:
        """设置目标位置。

        Parameters
        ----------
        target_x : float
            目标 X 位置 (像素)。
        target_y : float
            目标 Y 位置 (像素)。
        """
        self.state.target_position_x = target_x
        self.state.target_position_y = target_y
        LOGGER.debug(f"目标位置设置: ({target_x:.1f}, {target_y:.1f}) 像素")

    def validate_against_real(
        self,
        real_trajectory: List[Tuple[float, float]],
        real_params: Optional[Dict[str, float]] = None,
    ) -> TwinValidationReport:
        """将数字孪生与真实系统数据对比验证。

        Parameters
        ----------
        real_trajectory : List[Tuple[float, float]]
            真实系统轨迹 [(x, y), ...]。
        real_params : Dict[str, float] or None
            真实系统已知参数。

        Returns
        -------
        TwinValidationReport
            验证报告。
        """
        import datetime

        report = TwinValidationReport(
            validation_timestamp=datetime.datetime.now().isoformat(),
        )

        if len(real_trajectory) < 2:
            report.recommendations.append("真实轨迹数据不足，至少需要 2 个点")
            return report

        # 运行仿真并获取轨迹
        sim_result = self.run_closed_loop(
            max_iterations=len(real_trajectory),
            gain=1.0,
            record_frames=False,
        )

        # 对齐轨迹长度
        n = min(len(real_trajectory), len(sim_result.trajectory_x))
        real_x = np.array([p[0] for p in real_trajectory[:n]])
        real_y = np.array([p[1] for p in real_trajectory[:n]])
        sim_x = np.array(sim_result.trajectory_x[:n])
        sim_y = np.array(sim_result.trajectory_y[:n])

        # 计算保真度
        error_x = real_x - sim_x
        error_y = real_y - sim_y
        rmse = np.sqrt(np.mean(error_x ** 2 + error_y ** 2))
        max_deviation = np.max(np.sqrt(error_x ** 2 + error_y ** 2))

        # 归一化保真度评分
        real_range = np.sqrt(
            (real_x.max() - real_x.min()) ** 2 +
            (real_y.max() - real_y.min()) ** 2
        )
        if real_range > 0:
            fidelity = max(0.0, 1.0 - rmse / real_range)
        else:
            fidelity = max(0.0, 1.0 - rmse)

        report.model_fidelity_score = float(fidelity)
        report.prediction_accuracy = {
            "rmse_px": float(rmse),
            "max_deviation_px": float(max_deviation),
            "correlation_x": float(np.corrcoef(real_x, sim_x)[0, 1]) if n > 1 else 0.0,
            "correlation_y": float(np.corrcoef(real_y, sim_y)[0, 1]) if n > 1 else 0.0,
        }

        # 参数估计
        report.parameter_estimates = self._estimate_parameters(
            real_trajectory, sim_result
        )

        # 生成建议
        if fidelity < 0.5:
            report.recommendations.append(
                "模型保真度较低，建议调整振动频率或幅度参数"
            )
        if fidelity < 0.8:
            report.recommendations.append(
                "考虑增加湍流模型或调整漂移参数以提高匹配度"
            )
        if rmse > 5.0:
            report.recommendations.append(
                f"RMSE 较大 ({rmse:.2f}px)，建议检查像素尺度映射"
            )
        if fidelity >= 0.9:
            report.recommendations.append(
                "模型匹配良好，可用于预测和控制算法验证"
            )

        # 如果有真实参数，进行对比
        if real_params is not None:
            for key, real_val in real_params.items():
                if key in report.parameter_estimates:
                    est_val = report.parameter_estimates[key]
                    diff_pct = abs(est_val - real_val) / (abs(real_val) + 1e-30) * 100
                    report.prediction_accuracy[f"param_diff_{key}_pct"] = diff_pct

        return report

    def _estimate_parameters(
        self,
        real_trajectory: List[Tuple[float, float]],
        sim_result: SimulationResult,
    ) -> Dict[str, float]:
        """从真实轨迹估计系统参数。

        Parameters
        ----------
        real_trajectory : List[Tuple[float, float]]
            真实系统轨迹。
        sim_result : SimulationResult
            仿真结果。

        Returns
        -------
        Dict[str, float]
            估计参数。
        """
        estimates: Dict[str, float] = {}

        if len(real_trajectory) < 3:
            return estimates

        real_x = np.array([p[0] for p in real_trajectory])
        real_y = np.array([p[1] for p in real_trajectory])

        # 估计振动频率 (通过 FFT)
        dx = np.diff(real_x)
        dy = np.diff(real_y)
        dr = np.sqrt(dx ** 2 + dy ** 2)

        if len(dr) > 4:
            # FFT 分析
            fft_vals = np.fft.rfft(dr - dr.mean())
            freqs = np.fft.rfftfreq(len(dr), d=self.config.simulation_dt_s)
            magnitudes = np.abs(fft_vals)

            # 找到主频率
            if len(magnitudes) > 1:
                # 跳过直流分量
                peak_idx = np.argmax(magnitudes[1:]) + 1
                if peak_idx < len(freqs):
                    estimates["dominant_frequency_hz"] = float(freqs[peak_idx])
                    estimates["dominant_amplitude_px"] = float(magnitudes[peak_idx])

        # 估计漂移速率
        total_displacement = np.sqrt(
            (real_x[-1] - real_x[0]) ** 2 + (real_y[-1] - real_y[0]) ** 2
        )
        total_time = len(real_trajectory) * self.config.simulation_dt_s
        if total_time > 0:
            estimates["drift_rate_px_per_s"] = float(total_displacement / total_time)

        # 估计稳态抖动
        if len(real_x) > 10:
            steady_x = real_x[-min(50, len(real_x)):]
            steady_y = real_y[-min(50, len(real_y)):]
            estimates["steady_state_jitter_x_px"] = float(np.std(steady_x))
            estimates["steady_state_jitter_y_px"] = float(np.std(steady_y))

        return estimates

    def get_beam_info(self) -> BeamParameters:
        """获取当前光束参数。

        Returns
        -------
        BeamParameters
            光束参数。
        """
        return BeamParameters(
            waist=self.beam.waist,
            rayleigh_range=self.beam.rayleigh_range(),
            divergence_half_angle=self.beam.divergence_half_angle(),
            wavelength=self.beam.wavelength,
            q_parameter=self.beam.q_parameter(0.0),
        )

    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统参数摘要。

        Returns
        -------
        Dict[str, Any]
            系统参数摘要。
        """
        cfg = self.config
        beam_info = self.get_beam_info()

        return {
            "beam": {
                "wavelength_nm": cfg.beam_wavelength_nm,
                "waist_um": cfg.beam_waist_um,
                "power_mW": cfg.beam_power_mw,
                "rayleigh_range_mm": beam_info.rayleigh_range * 1e3,
                "divergence_mrad": beam_info.divergence_half_angle * 1e3,
            },
            "optics": {
                "focal_length_mm": cfg.lens_focal_length_mm,
                "aperture_diameter_mm": cfg.aperture_diameter_mm,
                "n_elements": len(self._optical_elements),
            },
            "detector": {
                "pixel_size_um": cfg.detector_pixel_size_um,
                "resolution": cfg.detector_resolution,
                "read_noise_e": cfg.detector_read_noise_e,
                "quantization_bits": cfg.detector_quantization_bits,
                "quantum_efficiency": cfg.detector_quantum_efficiency,
            },
            "motor": {
                "step_size_um": cfg.motor_step_size_um,
                "backlash_steps": cfg.motor_backlash_steps,
                "max_velocity_steps_s": cfg.motor_max_velocity_steps_s,
            },
            "environment": {
                "vibration_amplitude_px": cfg.vibration_amplitude_px,
                "vibration_frequencies_hz": cfg.vibration_frequencies_hz,
                "drift_rate_px_per_s": cfg.drift_rate_px_per_s,
                "turbulence_enabled": cfg.enable_turbulence,
            },
            "simulation": {
                "dt_s": cfg.simulation_dt_s,
                "seed": cfg.seed,
            },
        }

    def export_config(self) -> Dict[str, Any]:
        """导出当前配置为字典。

        Returns
        -------
        Dict[str, Any]
            配置字典。
        """
        from dataclasses import asdict
        return asdict(self.config)

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "DigitalTwinSimulator":
        """从字典创建仿真器。

        Parameters
        ----------
        config_dict : Dict[str, Any]
            配置字典。

        Returns
        -------
        DigitalTwinSimulator
            仿真器实例。
        """
        config = SimulationConfig(**{
            k: v for k, v in config_dict.items()
            if k in SimulationConfig.__dataclass_fields__
        })
        return cls(config)
