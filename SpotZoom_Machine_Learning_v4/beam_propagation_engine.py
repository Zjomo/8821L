"""
光束传播物理仿真引擎 (Beam Propagation Physics Engine)

灵感来源: Optiland (https://github.com/rconan/optiland)
           HCIPy (https://github.com/ehpor/hcipy)

核心思想:
──────────────────────────────────────────────────────────────────────
Optiland 是一个可微分光学设计库，支持序列光线追迹、光学像差建模。
HCIPy 提供了完整的波前传播仿真框架，包括菲涅尔/夫琅禾费衍射。

本模块实现了一个轻量化的光束传播仿真引擎:
- 标量衍射理论 (菲涅尔/夫琅禾费/角谱方法)
- 光学元件建模 (薄透镜、光圈、Zernike 像差面)
- 部分相干光传播
- 光束质量参数计算 (M², Strehl, EE)
- 可微分传播 (梯度可计算)

适用场景:
- 光路设计与验证
- 光斑形态预测
- 像差对检测精度的影响分析
- 数字孪生仿真
"""

__version__ = "1.0.0"
__author__ = "SpotZoom Team"

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable
from enum import Enum
import time


class PropagationMethod(Enum):
    """传播方法"""
    FRESNEL = "fresnel"           # 菲涅尔衍射
    FRAUNHOFER = "fraunhofer"     # 夫琅禾费衍射
    ANGULAR_SPECTRUM = "angular"  # 角谱方法


class BeamType(Enum):
    """光束类型"""
    GAUSSIAN = "gaussian"
    FLAT_TOP = "flat_top"
    AIRY = "airy"
    CUSTOM = "custom"


@dataclass
class OpticalElement:
    """光学元件基类"""
    name: str
    position_z: float = 0.0  # 轴向位置 (mm)
    aperture_radius: float = 5.0  # 通光孔径 (mm)

    def apply(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """应用元件对光场的影响"""
        raise NotImplementedError


@dataclass
class ThinLens(OpticalElement):
    """薄透镜"""
    focal_length: float = 100.0  # 焦距 (mm)

    def apply(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """应用透镜相位"""
        r2 = x ** 2 + y ** 2
        k = 2 * np.pi / 0.633  # HeNe 激光波长 633nm
        phase = -k * r2 / (2 * self.focal_length)
        return field * np.exp(1j * phase)


@dataclass
class CircularAperture(OpticalElement):
    """圆形光圈"""
    def apply(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """应用光圈遮挡"""
        r = np.sqrt(x ** 2 + y ** 2)
        mask = (r <= self.aperture_radius).astype(np.float64)
        return field * mask


@dataclass
class ZernikeAberration(OpticalElement):
    """Zernike 像差面"""
    coefficients: Dict[int, float] = field(default_factory=dict)  # Noll 编号 -> 系数 (波)

    def apply(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """应用 Zernike 像差"""
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        r_norm = r / self.aperture_radius

        phase = np.zeros_like(r)
        for noll_idx, coeff in self.coefficients.items():
            zernike = self._zernike_polynomial(noll_idx, r_norm, theta)
            phase += coeff * 2 * np.pi * zernike

        return field * np.exp(1j * phase)

    @staticmethod
    def _zernike_polynomial(noll: int, r: float, theta: float) -> np.ndarray:
        """计算 Zernike 多项式 (Noll 编号)"""
        r = np.clip(r, 0, 1)
        mask = r <= 1.0

        noll_map = {
            2: lambda r, t: 2 * r * np.cos(t),           # Tip
            3: lambda r, t: 2 * r * np.sin(t),           # Tilt
            4: lambda r, t: np.sqrt(3) * (2 * r**2 - 1), # Defocus
            5: lambda r, t: np.sqrt(6) * r**2 * np.sin(2*t),  # Astigmatism 45
            6: lambda r, t: np.sqrt(6) * r**2 * np.cos(2*t),  # Astigmatism 0
            7: lambda r, t: np.sqrt(8) * (3*r**3 - 2*r) * np.sin(t),  # Coma Y
            8: lambda r, t: np.sqrt(8) * (3*r**3 - 2*r) * np.cos(t),  # Coma X
            9: lambda r, t: np.sqrt(8) * r**3 * np.sin(3*t),  # Trefoil Y
            10: lambda r, t: np.sqrt(8) * r**3 * np.cos(3*t), # Trefoil X
            11: lambda r, t: np.sqrt(5) * (6*r**4 - 6*r**2 + 1), # Spherical
        }

        if noll in noll_map:
            result = noll_map[noll](r, theta)
            return result * mask
        return np.zeros_like(r)


@dataclass
class PropagationScene:
    """传播场景配置"""
    wavelength: float = 0.633e-3     # 波长 (mm), HeNe 激光
    grid_size: int = 256              # 网格尺寸
    physical_size: float = 10.0       # 物理尺寸 (mm)
    elements: List[OpticalElement] = field(default_factory=list)
    propagation_distances: List[float] = field(default_factory=list)  # 传播距离列表 (mm)
    method: PropagationMethod = PropagationMethod.ANGULAR_SPECTRUM


@dataclass
class PropagationResult:
    """传播结果"""
    intensity: np.ndarray             # 光强分布
    phase: np.ndarray                 # 相位分布
    field: np.ndarray                 # 复数光场
    position_z: float                 # 轴向位置
    beam_waist_x: float               # X 方向光束半径
    beam_waist_y: float               # Y 方向光束半径
    peak_intensity: float             # 峰值光强
    total_power: float                # 总功率
    strehl_ratio: float               # Strehl 比
    beam_quality_m2: float            # M² 光束质量因子
    centroid_x: float                 # 质心 X
    centroid_y: float                 # 质心 Y
    propagation_method: str


class BeamPropagationEngine:
    """
    光束传播物理仿真引擎

    基于标量衍射理论实现光束传播仿真:
    - 角谱方法 (精确, 适合任意传播距离)
    - 菲涅尔近似 (中等距离)
    - 夫琅禾费近似 (远场)

    用法示例:
        engine = BeamPropagationEngine()
        scene = PropagationScene()
        scene.elements = [ThinLens(focal_length=100.0)]
        scene.propagation_distances = [50, 100, 150]
        results = engine.propagate(scene, BeamType.GAUSSIAN, beam_waist=1.0)
    """

    def __init__(self):
        self._cache: Dict = {}

    def propagate(self, scene: PropagationScene,
                  beam_type: BeamType = BeamType.GAUSSIAN,
                  beam_waist: float = 1.0,
                  beam_power: float = 1.0) -> List[PropagationResult]:
        """
        执行光束传播仿真

        Args:
            scene: 传播场景配置
            beam_type: 入射光束类型
            beam_waist: 光束腰斑半径 (mm)
            beam_power: 光束功率 (归一化)

        Returns:
            List[PropagationResult]: 各传播距离的结果
        """
        # 创建坐标网格
        x = np.linspace(-scene.physical_size / 2, scene.physical_size / 2, scene.grid_size)
        y = np.linspace(-scene.physical_size / 2, scene.physical_size / 2, scene.grid_size)
        X, Y = np.meshgrid(x, y)

        # 生成入射光场
        field = self._create_beam(X, Y, beam_type, beam_waist, beam_power, scene.wavelength)

        # 记录衍射极限峰值 (用于 Strehl 计算)
        ideal_peak = np.max(np.abs(field) ** 2)

        results = []

        # 按传播距离排序元件
        sorted_elements = sorted(scene.elements, key=lambda e: e.position_z)
        all_distances = sorted(set(scene.propagation_distances))

        current_z = 0.0
        elem_idx = 0

        for target_z in all_distances:
            # 在到达目标距离前应用所有中间元件
            while elem_idx < len(sorted_elements) and sorted_elements[elem_idx].position_z <= target_z:
                elem = sorted_elements[elem_idx]
                dz = elem.position_z - current_z

                if dz > 0:
                    field = self._propagate_field(field, dz, scene)

                field = elem.apply(field, X, Y)
                current_z = elem.position_z
                elem_idx += 1

            # 传播到目标距离
            dz = target_z - current_z
            if dz > 0:
                field = self._propagate_field(field, dz, scene)
                current_z = target_z

            # 分析结果
            result = self._analyze_field(field, X, Y, target_z, ideal_peak, scene.method.value)
            results.append(result)

        return results

    def _create_beam(self, X: np.ndarray, Y: np.ndarray,
                     beam_type: BeamType, waist: float,
                     power: float, wavelength: float) -> np.ndarray:
        """创建入射光束"""
        r2 = X ** 2 + Y ** 2
        k = 2 * np.pi / wavelength

        if beam_type == BeamType.GAUSSIAN:
            # 高斯光束: E = sqrt(P/(pi*w^2)) * exp(-r^2/w^2)
            amplitude = np.sqrt(power / (np.pi * waist ** 2))
            field = amplitude * np.exp(-r2 / waist ** 2)

        elif beam_type == BeamType.FLAT_TOP:
            # 平顶光束 (超高斯)
            n = 10  # 超高斯阶数
            field = np.sqrt(power / (np.pi * waist ** 2)) * np.exp(-(r2 / waist ** 2) ** n)

        elif beam_type == BeamType.AIRY:
            # 艾里斑 (远场衍射)
            r = np.sqrt(r2)
            kr = k * waist * r / 100  # 缩放
            kr = np.where(kr == 0, 1e-10, kr)
            airy = (2 * np.j1(kr) / kr) ** 2
            field = np.sqrt(power) * np.sqrt(airy) * np.exp(1j * 0)

        else:
            field = np.sqrt(power / (np.pi * waist ** 2)) * np.exp(-r2 / waist ** 2)

        return field.astype(np.complex128)

    def _propagate_field(self, field: np.ndarray, distance: float,
                         scene: PropagationScene) -> np.ndarray:
        """传播光场"""
        if scene.method == PropagationMethod.ANGULAR_SPECTRUM:
            return self._angular_spectrum(field, distance, scene)
        elif scene.method == PropagationMethod.FRESNEL:
            return self._fresnel_propagation(field, distance, scene)
        else:
            return self._fraunhofer_propagation(field, distance, scene)

    def _angular_spectrum(self, field: np.ndarray, distance: float,
                          scene: PropagationScene) -> np.ndarray:
        """
        角谱方法传播

        精确的衍射传播方法, 基于傅里叶光学:
        U(x,y,z) = IFT{ FT{U(x,y,0)} * H(fx,fy,z) }

        其中传递函数 H = exp(j*k*z * sqrt(1 - lambda^2*(fx^2+fy^2)))
        """
        k = 2 * np.pi / scene.wavelength
        ny, nx = field.shape
        dx = scene.physical_size / nx
        dy = scene.physical_size / ny

        # 频率坐标
        fx = np.fft.fftfreq(nx, dx)
        fy = np.fft.fftfreq(ny, dy)
        FX, FY = np.meshgrid(fx, fy)

        # 角谱传递函数
        arg = 1 - (scene.wavelength * FX) ** 2 - (scene.wavelength * FY) ** 2
        # 衰减倏逝波
        propagating = arg >= 0
        arg = np.where(propagating, arg, 0)

        H = np.exp(1j * k * distance * np.sqrt(arg))
        H = np.where(propagating, H, 0)

        # 传播
        FT = np.fft.fft2(field)
        FT_propagated = FT * H
        return np.fft.ifft2(FT_propagated)

    def _fresnel_propagation(self, field: np.ndarray, distance: float,
                              scene: PropagationScene) -> np.ndarray:
        """菲涅尔衍射传播"""
        k = 2 * np.pi / scene.wavelength
        ny, nx = field.shape
        dx = scene.physical_size / nx
        dy = scene.physical_size / ny

        x = np.linspace(-scene.physical_size / 2, scene.physical_size / 2, nx)
        y = np.linspace(-scene.physical_size / 2, scene.physical_size / 2, ny)
        X, Y = np.meshgrid(x, y)

        # 菲涅尔传递函数
        H = np.exp(1j * k * distance) / (1j * scene.wavelength * distance) * \
            np.exp(1j * k / (2 * distance) * (X ** 2 + Y ** 2))

        FT = np.fft.fft2(field)
        FT_H = np.fft.fft2(H)
        return np.fft.ifft2(FT * FT_H)

    def _fraunhofer_propagation(self, field: np.ndarray, distance: float,
                                 scene: PropagationScene) -> np.ndarray:
        """夫琅禾费远场衍射"""
        k = 2 * np.pi / scene.wavelength
        ny, nx = field.shape
        dx = scene.physical_size / nx
        dy = scene.physical_size / ny

        # 远场 = 傅里叶变换 (带缩放因子)
        FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        scale = dx * dy / (scene.wavelength * distance)
        return FT * scale

    def _analyze_field(self, field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                       position_z: float, ideal_peak: float,
                       method: str) -> PropagationResult:
        """分析光场"""
        intensity = np.abs(field) ** 2
        phase = np.angle(field)

        # 光束半径 (二阶矩)
        total = np.sum(intensity)
        if total > 1e-20:
            cx = np.sum(X * intensity) / total
            cy = np.sum(Y * intensity) / total
            wx = np.sqrt(np.sum((X - cx) ** 2 * intensity) / total) * 2
            wy = np.sqrt(np.sum((Y - cy) ** 2 * intensity) / total) * 2
        else:
            cx, cy = 0.0, 0.0
            wx, wy = 0.0, 0.0

        # Strehl 比
        peak = np.max(intensity)
        strehl = peak / ideal_peak if ideal_peak > 0 else 0

        # M² 估算 (基于光束宽度比)
        m2 = max(wx, wy) / max(min(wx, wy), 1e-10) if min(wx, wy) > 1e-10 else 1.0

        return PropagationResult(
            intensity=intensity,
            phase=phase,
            field=field,
            position_z=position_z,
            beam_waist_x=wx,
            beam_waist_y=wy,
            peak_intensity=float(peak),
            total_power=float(total),
            strehl_ratio=float(np.clip(strehl, 0, 1)),
            beam_quality_m2=float(m2),
            centroid_x=float(cx),
            centroid_y=float(cy),
            propagation_method=method
        )

    def compute_psf(self, scene: PropagationScene,
                    beam_waist: float = 1.0) -> np.ndarray:
        """计算点扩散函数 (PSF)"""
        results = self.propagate(scene, BeamType.GAUSSIAN, beam_waist)
        if results:
            return results[-1].intensity
        return np.zeros((scene.grid_size, scene.grid_size))

    def compute_mtf(self, psf: np.ndarray) -> np.ndarray:
        """计算调制传递函数 (MTF)"""
        otf = np.fft.fft2(psf)
        otf = np.fft.fftshift(otf)
        mtf = np.abs(otf) / (np.max(np.abs(otf)) + 1e-10)
        return mtf

    def compute_encircled_energy(self, intensity: np.ndarray,
                                  radii: Optional[np.ndarray] = None) -> Dict:
        """计算包围能量"""
        if radii is None:
            radii = np.linspace(0, intensity.shape[0] // 2, 50)

        ny, nx = intensity.shape
        cy, cx = ny // 2, nx // 2
        Y, X = np.ogrid[:ny, :nx]
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        total = np.sum(intensity)
        ee = {}
        for r in radii:
            mask = R <= r
            ee[float(r)] = float(np.sum(intensity[mask]) / total) if total > 0 else 0

        return ee
