"""
可组合光学管线引擎 (ComposableOpticalPipeline)

灵感来源:
- POPPY (NASA/STScI) — OpticalElement 级联架构，端到端 PSF 仿真
- HCIPy (https://github.com/ehpor/hcipy) — 光学传播与波前仿真框架
- PROPER (https://github.com/mperrin/poppy) — 光学传播仿真

算法原理:
- Angular Spectrum Propagation — 角谱传播法，精确模拟 Fresnel 衍射
- Fourier Optics — 傅里叶光学: 透镜 = 二次相位因子，远场 = FFT
- Zernike Polynomials — 正交多项式基函数，描述光学像差
- Optical Transfer Function (OTF) — 光学传递函数 = PSF 的傅里叶变换
- Strehl Ratio — Strehl 比: 实测 PSF 峰值 / 衍射极限 PSF 峰值
- Thin Lens Model — 薄透镜模型: 相位延迟 = -π r² / (λ f)
- Pipeline Composition — 管线组合: 元件级联，波前依次通过每个元件

功能:
- 将光学元件 (透镜、光瞳、像差源) 组合为管线
- 计算端到端 PSF 和 OTF
- 对比不同管线的 PSF 差异
- 计算 Strehl 比
- 支持元件的动态添加、移除、插入

依赖: numpy, opencv-python (仅用于图像处理辅助)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

LOGGER = logging.getLogger("SpotZoom.ComposableOpticalPipeline")

# 模块默认禁用标志
composable_pipeline_enabled: bool = False


# ======================== 数据类 ========================


@dataclass
class PipelineConfig:
    """管线配置参数。"""
    name: str = "default"  # 管线名称
    grid_size: int = 256  # 网格尺寸 (像素)
    wavelength: float = 632.8e-9  # 波长 (米)
    pixel_scale: float = 1e-6  # 像素尺度 (米/像素)
    aperture_diameter: float = 0.01  # 孔径直径 (米)
    focal_length: float = 0.1  # 焦距 (米)
    oversampling: int = 1  # 过采样因子


@dataclass
class PSFResult:
    """PSF 计算结果。"""
    psf: np.ndarray  # PSF 图像 (归一化, 2D)
    peak_value: float  # 峰值
    fwhm_x: float  # X 方向 FWHM (像素)
    fwhm_y: float  # Y 方向 FWHM (像素)
    strehl_ratio: float  # Strehl 比
    encircled_energy_80: float  # 80% 环围能量半径 (像素)
    total_energy: float  # 总能量
    grid_size: int  # 网格尺寸
    wavelength: float  # 波长 (米)
    pixel_scale: float  # 像素尺度 (米/像素)
    element_names: List[str] = field(default_factory=list)  # 元件名称列表


@dataclass
class OTFResult:
    """OTF 计算结果。"""
    otf: np.ndarray  # OTF (复数, 2D)
    mtf: np.ndarray  # MTF = |OTF| (2D)
    mtf_radial_average: np.ndarray  # MTF 径向平均 (1D)
    cutoff_frequency: float  # 截止频率 (cycles/meter)
    mtf_at_nyquist: float  # Nyquist 频率处的 MTF 值
    grid_size: int  # 网格尺寸
    wavelength: float  # 波长 (米)


@dataclass
class PipelineComparison:
    """管线对比结果。"""
    name_a: str  # 管线 A 名称
    name_b: str  # 管线 B 名称
    psf_difference_rms: float  # PSF 差异 RMS
    psf_correlation: float  # PSF 相关系数
    strehl_difference: float  # Strehl 比差异
    fwhm_difference_x: float  # FWHM X 差异 (像素)
    fwhm_difference_y: float  # FWHM Y 差异 (像素)
    mtf_difference_rms: float  # MTF 差异 RMS


# ======================== 光学元件接口 ========================


@runtime_checkable
class OpticalElement(Protocol):
    """光学元件接口 (Protocol)。

    所有光学元件必须实现 apply 方法，接收波前复振幅场并返回变换后的场。
    """

    name: str

    def apply(self, wavefront: np.ndarray) -> np.ndarray:
        """将光学元件的效果应用到波前。

        Parameters
        ----------
        wavefront : np.ndarray
            输入波前复振幅场 (2D, complex128)。

        Returns
        -------
        np.ndarray
            变换后的波前复振幅场 (2D, complex128)。
        """
        ...


# ======================== 具体光学元件 ========================


class ThinLens:
    """薄透镜元件。

    薄透镜对波前施加二次相位延迟:
        φ(r) = -π r² / (λ f)

    Parameters
    ----------
    focal_length : float
        焦距 (米)。
    pixel_scale : float
        像素尺度 (米/像素)。
    wavelength : float
        设计波长 (米)。
    decentration_x : float
        X 方向偏心 (像素)。
    decentration_y : float
        Y 方向偏心 (像素)。
    tilt_x : float
        X 方向倾斜 (弧度)。
    tilt_y : float
        Y 方向倾斜 (弧度)。
    name : str or None
        元件名称。
    """

    def __init__(
        self,
        focal_length: float = 0.1,
        pixel_scale: float = 1e-6,
        wavelength: float = 632.8e-9,
        decentration_x: float = 0.0,
        decentration_y: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        name: Optional[str] = None,
    ):
        self.focal_length = float(focal_length)
        self.pixel_scale = float(pixel_scale)
        self.wavelength = float(wavelength)
        self.decentration_x = float(decentration_x)
        self.decentration_y = float(decentration_y)
        self.tilt_x = float(tilt_x)
        self.tilt_y = float(tilt_y)
        self.name: str = name or f"ThinLens(f={focal_length:.3f}m)"

        if self.focal_length == 0:
            raise ValueError("焦距不能为零")

        LOGGER.debug(
            "ThinLens '%s': f=%.3fm, 偏心=(%.2f, %.2f)px, 倾斜=(%.4f, %.4f)rad",
            self.name, self.focal_length,
            self.decentration_x, self.decentration_y,
            self.tilt_x, self.tilt_y,
        )

    def apply(self, wavefront: np.ndarray) -> np.ndarray:
        """应用薄透镜效果到波前。

        Parameters
        ----------
        wavefront : np.ndarray
            输入波前复振幅场。

        Returns
        -------
        np.ndarray
            变换后的波前。
        """
        size = wavefront.shape[0]

        # 坐标网格 (以像素为单位)
        yy, xx = np.mgrid[:size, :size]
        cx = size / 2.0 + self.decentration_x
        cy = size / 2.0 + self.decentration_y

        # 转换为物理坐标 (米)
        dx = (xx - cx) * self.pixel_scale
        dy = (yy - cy) * self.pixel_scale
        r_sq = dx ** 2 + dy ** 2

        # 薄透镜相位: φ = -π r² / (λ f)
        phase = -np.pi * r_sq / (self.wavelength * self.focal_length)

        # 倾斜: 线性相位
        if self.tilt_x != 0 or self.tilt_y != 0:
            k = 2.0 * np.pi / self.wavelength
            phase += k * (self.tilt_x * dx + self.tilt_y * dy)

        return wavefront * np.exp(1j * phase)


class CircularAperture:
    """圆形光瞳元件。

    Parameters
    ----------
    diameter_pixels : float
        孔径直径 (像素)。
    soft_edge_width : float
        软边缘宽度 (像素)。0 = 硬边缘。
    center_x : float
        中心 X 坐标 (像素)。
    center_y : float
        中心 Y 坐标 (像素)。
    name : str or None
        元件名称。
    """

    def __init__(
        self,
        diameter_pixels: float = 128.0,
        soft_edge_width: float = 2.0,
        center_x: float = 0.0,
        center_y: float = 0.0,
        name: Optional[str] = None,
    ):
        self.diameter_pixels = float(diameter_pixels)
        self.soft_edge_width = float(soft_edge_width)
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.name: str = name or f"CircularAperture(D={diameter_pixels:.0f}px)"

        if self.diameter_pixels <= 0:
            raise ValueError("孔径直径必须 > 0")

        LOGGER.debug(
            "CircularAperture '%s': 直径=%.1fpx, 软边缘=%.1fpx",
            self.name, self.diameter_pixels, self.soft_edge_width,
        )

    def apply(self, wavefront: np.ndarray) -> np.ndarray:
        """应用圆形孔径到波前。

        Parameters
        ----------
        wavefront : np.ndarray
            输入波前复振幅场。

        Returns
        -------
        np.ndarray
            裁剪后的波前。
        """
        size = wavefront.shape[0]
        radius = self.diameter_pixels / 2.0

        yy, xx = np.mgrid[:size, :size]
        cx = size / 2.0 + self.center_x
        cy = size / 2.0 + self.center_y

        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        if self.soft_edge_width > 0:
            # 软边缘 (超高斯函数)
            n_edge = 10.0  # 超高斯阶数
            transmission = np.exp(
                -0.5 * ((r - radius) / self.soft_edge_width) ** n_edge
            )
            transmission = np.clip(transmission, 0.0, 1.0)
        else:
            # 硬边缘
            transmission = (r <= radius).astype(np.float64)

        return wavefront * transmission

    def get_transmission(self, grid_size: int) -> np.ndarray:
        """获取孔径透过率图。

        Parameters
        ----------
        grid_size : int
            网格尺寸。

        Returns
        -------
        np.ndarray
            透过率图 (0~1)。
        """
        size = int(grid_size)
        radius = self.diameter_pixels / 2.0

        yy, xx = np.mgrid[:size, :size]
        cx = size / 2.0 + self.center_x
        cy = size / 2.0 + self.center_y

        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        if self.soft_edge_width > 0:
            n_edge = 10.0
            transmission = np.exp(
                -0.5 * ((r - radius) / self.soft_edge_width) ** n_edge
            )
            transmission = np.clip(transmission, 0.0, 1.0)
        else:
            transmission = (r <= radius).astype(np.float64)

        return transmission


class ZernikeAberration:
    """Zernike 像差元件。

    使用 Zernike 多项式叠加产生波前像差。
    系数以波长为单位。

    Parameters
    ----------
    coefficients : Dict[int, float]
        Noll 索引到系数的映射 (单位: 波长)。
        例如 {4: 0.5, 5: 0.2} 表示离焦 0.5λ + 像散 0.2λ。
    aperture_diameter_pixels : float
        孔径直径 (像素)，用于归一化。
    wavelength : float
        波长 (米)。
    name : str or None
        元件名称。
    """

    # Noll 索引到 (n, m) 的映射
    _NOLL_TO_NM: Dict[int, Tuple[int, int]] = {
        1: (0, 0), 2: (1, 1), 3: (1, -1),
        4: (2, 0), 5: (2, 2), 6: (2, -2),
        7: (3, 1), 8: (3, -1), 9: (3, 3), 10: (3, -3),
        11: (4, 0), 12: (4, 2), 13: (4, -2),
        14: (4, 4), 15: (4, -4),
    }

    def __init__(
        self,
        coefficients: Optional[Dict[int, float]] = None,
        aperture_diameter_pixels: float = 128.0,
        wavelength: float = 632.8e-9,
        name: Optional[str] = None,
    ):
        self.coefficients: Dict[int, float] = dict(coefficients) if coefficients else {}
        self.aperture_diameter_pixels = float(aperture_diameter_pixels)
        self.wavelength = float(wavelength)
        self.name: str = name or f"ZernikeAberration({len(self.coefficients)} terms)"

        LOGGER.debug(
            "ZernikeAberration '%s': %d 项, 孔径=%.1fpx",
            self.name, len(self.coefficients), self.aperture_diameter_pixels,
        )

    def apply(self, wavefront: np.ndarray) -> np.ndarray:
        """应用 Zernike 像差到波前。

        Parameters
        ----------
        wavefront : np.ndarray
            输入波前复振幅场。

        Returns
        -------
        np.ndarray
            变换后的波前。
        """
        if not self.coefficients:
            return wavefront

        size = wavefront.shape[0]
        radius = self.aperture_diameter_pixels / 2.0

        # 坐标网格
        yy, xx = np.mgrid[:size, :size]
        cx, cy = size / 2.0, size / 2.0
        dx = (xx - cx) / radius
        dy = (yy - cy) / radius
        rho = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)

        # 叠加 Zernike 项
        phase = np.zeros((size, size), dtype=np.float64)

        for j, coeff in self.coefficients.items():
            if j not in self._NOLL_TO_NM:
                LOGGER.warning("ZernikeAberration: 不支持 Noll 索引 %d", j)
                continue

            n, m = self._NOLL_TO_NM[j]
            Z = self._compute_zernike(n, m, rho, theta)
            # 系数单位为波长，转换为弧度
            phase += coeff * 2.0 * np.pi * Z

        return wavefront * np.exp(1j * phase)

    def _compute_zernike(
        self,
        n: int,
        m: int,
        rho: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """计算 Zernike 多项式值。

        Parameters
        ----------
        n : int
            径向阶数。
        m : int
            角频率 (含符号)。
        rho : np.ndarray
            归一化径向坐标。
        theta : np.ndarray
            角坐标。

        Returns
        -------
        np.ndarray
            Zernike 多项式值。
        """
        R = self._zernike_radial(n, abs(m), rho)

        if m > 0:
            return R * np.cos(m * theta)
        elif m < 0:
            return R * np.sin(abs(m) * theta)
        else:
            return R

    @staticmethod
    def _zernike_radial(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
        """计算 Zernike 径向多项式 R_n^|m|(rho)。

        Parameters
        ----------
        n : int
            径向阶数。
        m_abs : int
            角频率绝对值。
        rho : np.ndarray
            归一化径向坐标。

        Returns
        -------
        np.ndarray
            径向多项式值。
        """
        R = np.zeros_like(rho, dtype=np.float64)

        if n == 0 and m_abs == 0:
            R[:] = 1.0
        elif n == 1 and m_abs == 1:
            R[:] = rho
        elif n == 2 and m_abs == 0:
            R[:] = 2.0 * rho ** 2 - 1.0
        elif n == 2 and m_abs == 2:
            R[:] = rho ** 2
        elif n == 3 and m_abs == 1:
            R[:] = 3.0 * rho ** 3 - 2.0 * rho
        elif n == 3 and m_abs == 3:
            R[:] = rho ** 3
        elif n == 4 and m_abs == 0:
            R[:] = 6.0 * rho ** 4 - 6.0 * rho ** 2 + 1.0
        elif n == 4 and m_abs == 2:
            R[:] = 4.0 * rho ** 4 - 3.0 * rho ** 2
        elif n == 4 and m_abs == 4:
            R[:] = rho ** 4
        else:
            LOGGER.warning(
                "ZernikeAberration: 径向多项式 n=%d, |m|=%d 未实现", n, m_abs
            )

        return R


class DefocusElement:
    """离焦元件。

    模拟离焦像差，等效于在焦点前后移动探测器。

    Parameters
    ----------
    defocus_waves : float
        离焦量 (波长单位)。正值 = 焦后, 负值 = 焦前。
    aperture_diameter_pixels : float
        孔径直径 (像素)。
    wavelength : float
        波长 (米)。
    name : str or None
        元件名称。
    """

    def __init__(
        self,
        defocus_waves: float = 1.0,
        aperture_diameter_pixels: float = 128.0,
        wavelength: float = 632.8e-9,
        name: Optional[str] = None,
    ):
        self.defocus_waves = float(defocus_waves)
        self.aperture_diameter_pixels = float(aperture_diameter_pixels)
        self.wavelength = float(wavelength)
        self.name: str = name or f"Defocus({defocus_waves:.2f}λ)"

        LOGGER.debug(
            "DefocusElement '%s': 离焦=%.2fλ, 孔径=%.1fpx",
            self.name, self.defocus_waves, self.aperture_diameter_pixels,
        )

    def apply(self, wavefront: np.ndarray) -> np.ndarray:
        """应用离焦效果到波前。

        离焦对应 Zernike Z4 (Noll j=4):
            Z_4(ρ) = 2ρ² - 1

        Parameters
        ----------
        wavefront : np.ndarray
            输入波前复振幅场。

        Returns
        -------
        np.ndarray
            变换后的波前。
        """
        size = wavefront.shape[0]
        radius = self.aperture_diameter_pixels / 2.0

        yy, xx = np.mgrid[:size, :size]
        cx, cy = size / 2.0, size / 2.0
        rho = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius

        # Zernike Z4 = 2ρ² - 1
        Z4 = 2.0 * rho ** 2 - 1.0

        # 转换为弧度
        phase = self.defocus_waves * 2.0 * np.pi * Z4

        return wavefront * np.exp(1j * phase)


# ======================== 管线引擎 ========================


class ComposableOpticalPipeline:
    """可组合光学管线引擎。

    将光学元件 (透镜、光瞳、像差源) 组合为管线，
    计算端到端 PSF。灵感来自 POPPY (NASA/STScI) 的级联架构。

    Parameters
    ----------
    config : PipelineConfig or None
        管线配置。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._elements: List[Any] = []

        # 缓存
        self._last_psf_result: Optional[PSFResult] = None
        self._last_otf_result: Optional[OTFResult] = None

        LOGGER.info(
            "ComposableOpticalPipeline '%s': 初始化完成 (grid=%d, λ=%.1fnm)",
            self.config.name, self.config.grid_size,
            self.config.wavelength * 1e9,
        )

    @property
    def num_elements(self) -> int:
        return len(self._elements)

    @property
    def element_names(self) -> List[str]:
        return [
            getattr(e, 'name', str(type(e).__name__))
            for e in self._elements
        ]

    def add_element(self, element: Any) -> None:
        """添加光学元件到管线末尾。

        Parameters
        ----------
        element : OpticalElement
            光学元件 (需实现 apply 方法)。
        """
        if not hasattr(element, 'apply'):
            raise TypeError(
                f"元件 {type(element).__name__} 未实现 apply() 方法"
            )

        self._elements.append(element)
        LOGGER.info(
            "ComposableOpticalPipeline: 添加元件 '%s' (位置 %d)",
            getattr(element, 'name', type(element).__name__),
            len(self._elements) - 1,
        )

    def remove_element(self, name: str) -> None:
        """移除指定名称的光学元件。

        Parameters
        ----------
        name : str
            元件名称。
        """
        original_len = len(self._elements)
        self._elements = [
            e for e in self._elements
            if getattr(e, 'name', '') != name
        ]
        if len(self._elements) < original_len:
            LOGGER.info(
                "ComposableOpticalPipeline: 移除元件 '%s'", name
            )
        else:
            LOGGER.warning(
                "ComposableOpticalPipeline: 未找到元件 '%s'", name
            )

    def insert_element(self, element: Any, index: int) -> None:
        """在指定位置插入光学元件。

        Parameters
        ----------
        element : OpticalElement
            光学元件。
        index : int
            插入位置 (0 = 最前面)。
        """
        if not hasattr(element, 'apply'):
            raise TypeError(
                f"元件 {type(element).__name__} 未实现 apply() 方法"
            )

        index = max(0, min(index, len(self._elements)))
        self._elements.insert(index, element)
        LOGGER.info(
            "ComposableOpticalPipeline: 在位置 %d 插入元件 '%s'",
            index, getattr(element, 'name', type(element).__name__),
        )

    def get_element(self, index: int) -> Optional[Any]:
        """获取指定位置的元件。"""
        if 0 <= index < len(self._elements):
            return self._elements[index]
        return None

    def clear(self) -> None:
        """清空所有元件。"""
        self._elements.clear()
        self._last_psf_result = None
        self._last_otf_result = None
        LOGGER.info("ComposableOpticalPipeline: 已清空所有元件")

    def compute_psf(
        self,
        grid_size: Optional[int] = None,
        wavelength: Optional[float] = None,
        pixel_scale: Optional[float] = None,
    ) -> PSFResult:
        """计算端到端 PSF。

        流程:
        1. 生成平面波前
        2. 依次通过每个光学元件
        3. 对出瞳做 FFT 得到焦面 PSF

        Parameters
        ----------
        grid_size : int or None
            网格尺寸。为 None 时使用配置值。
        wavelength : float or None
            波长 (米)。为 None 时使用配置值。
        pixel_scale : float or None
            像素尺度 (米/像素)。为 None 时使用配置值。

        Returns
        -------
        PSFResult
            PSF 计算结果。
        """
        size = int(grid_size) if grid_size is not None else self.config.grid_size
        wl = float(wavelength) if wavelength is not None else self.config.wavelength
        ps = float(pixel_scale) if pixel_scale is not None else self.config.pixel_scale

        if not self._elements:
            LOGGER.warning("ComposableOpticalPipeline: 管线为空，返回空 PSF")
            return self._empty_psf_result(size, wl, ps)

        # 1. 初始化平面波前
        wavefront = np.ones((size, size), dtype=np.complex128)

        # 2. 依次通过每个元件
        for element in self._elements:
            try:
                wavefront = element.apply(wavefront)
            except Exception as e:
                LOGGER.error(
                    "ComposableOpticalPipeline: 元件 '%s' 应用失败: %s",
                    getattr(element, 'name', 'unknown'), e,
                )
                raise

        # 3. FFT 得到焦面 PSF
        # PSF = |FT{wavefront}|²
        psf_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wavefront)))
        psf = np.abs(psf_field) ** 2

        # 归一化
        total = psf.sum()
        if total > 1e-12:
            psf = psf / total

        # 计算指标
        peak_value = float(psf.max())
        fwhm_x, fwhm_y = self._compute_fwhm(psf)

        # Strehl 比 (与理想 Airy PSF 比较)
        strehl = self._compute_strehl(psf, size)

        # 80% 环围能量半径
        ee80 = self._compute_encircled_energy_radius(psf, 0.80)

        result = PSFResult(
            psf=psf,
            peak_value=round(peak_value, 8),
            fwhm_x=round(fwhm_x, 4),
            fwhm_y=round(fwhm_y, 4),
            strehl_ratio=round(strehl, 6),
            encircled_energy_80=round(ee80, 4),
            total_energy=round(total, 8),
            grid_size=size,
            wavelength=wl,
            pixel_scale=ps,
            element_names=self.element_names,
        )

        self._last_psf_result = result

        LOGGER.info(
            "ComposableOpticalPipeline: PSF 计算完成 "
            "(peak=%.6f, FWHM=(%.2f, %.2f), Strehl=%.4f, %d 元件)",
            result.peak_value, result.fwhm_x, result.fwhm_y,
            result.strehl_ratio, len(self._elements),
        )

        return result

    def compute_otp(
        self,
        grid_size: Optional[int] = None,
        wavelength: Optional[float] = None,
        pixel_scale: Optional[float] = None,
    ) -> OTFResult:
        """计算光学传递函数 (OTF)。

        OTF = FT{PSF}，归一化使得 OTF(0,0) = 1。
        MTF = |OTF|。

        Parameters
        ----------
        grid_size : int or None
            网格尺寸。
        wavelength : float or None
            波长 (米)。
        pixel_scale : float or None
            像素尺度 (米/像素)。

        Returns
        -------
        OTFResult
            OTF 计算结果。
        """
        # 先计算 PSF
        psf_result = self.compute_psf(grid_size, wavelength, pixel_scale)
        psf = psf_result.psf
        size = psf_result.grid_size
        wl = psf_result.wavelength
        ps = psf_result.pixel_scale

        # OTF = FT{PSF}
        otf = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(psf))
        )

        # 归一化
        dc = otf[0, 0]
        if abs(dc) > 1e-12:
            otf = otf / dc

        # MTF
        mtf = np.abs(otf)

        # MTF 径向平均
        mtf_radial = self._radial_average(mtf, size)

        # 截止频率 (衍射极限)
        # f_c = D / (λ * size * pixel_scale) * size = D / λ (cycles/meter)
        cutoff_freq = self.config.aperture_diameter / wl

        # Nyquist 频率处的 MTF
        nyquist_idx = size // 2
        if nyquist_idx < mtf.shape[0]:
            mtf_at_nyquist = float(mtf[nyquist_idx, size // 2])
        else:
            mtf_at_nyquist = 0.0

        result = OTFResult(
            otf=otf,
            mtf=mtf,
            mtf_radial_average=mtf_radial,
            cutoff_frequency=round(cutoff_freq, 2),
            mtf_at_nyquist=round(mtf_at_nyquist, 6),
            grid_size=size,
            wavelength=wl,
        )

        self._last_otf_result = result

        LOGGER.info(
            "ComposableOpticalPipeline: OTF 计算完成 "
            "(截止频率=%.0f cycles/m, Nyquist MTF=%.4f)",
            result.cutoff_frequency, result.mtf_at_nyquist,
        )

        return result

    def compare_pipelines(
        self,
        other_pipeline: 'ComposableOpticalPipeline',
        grid_size: Optional[int] = None,
        wavelength: Optional[float] = None,
        pixel_scale: Optional[float] = None,
    ) -> PipelineComparison:
        """对比两个管线的 PSF 差异。

        Parameters
        ----------
        other_pipeline : ComposableOpticalPipeline
            另一个管线。
        grid_size : int or None
            网格尺寸。
        wavelength : float or None
            波长 (米)。
        pixel_scale : float or None
            像素尺度 (米/像素)。

        Returns
        -------
        PipelineComparison
            管线对比结果。
        """
        # 计算两个管线的 PSF
        psf_a = self.compute_psf(grid_size, wavelength, pixel_scale)
        psf_b = other_pipeline.compute_psf(grid_size, wavelength, pixel_scale)

        # 确保 PSF 尺寸相同
        size = min(psf_a.psf.shape[0], psf_b.psf.shape[0])
        a = psf_a.psf[:size, :size]
        b = psf_b.psf[:size, :size]

        # PSF 差异 RMS
        diff = a - b
        psf_diff_rms = float(np.sqrt(np.mean(diff ** 2)))

        # PSF 相关系数
        a_flat = a.ravel()
        b_flat = b.ravel()
        a_centered = a_flat - a_flat.mean()
        b_centered = b_flat - b_flat.mean()
        denom = (
            np.sqrt(np.sum(a_centered ** 2))
            * np.sqrt(np.sum(b_centered ** 2))
        )
        if denom > 1e-12:
            correlation = float(np.sum(a_centered * b_centered) / denom)
        else:
            correlation = 0.0

        # Strehl 差异
        strehl_diff = abs(psf_a.strehl_ratio - psf_b.strehl_ratio)

        # FWHM 差异
        fwhm_diff_x = abs(psf_a.fwhm_x - psf_b.fwhm_x)
        fwhm_diff_y = abs(psf_a.fwhm_y - psf_b.fwhm_y)

        # MTF 差异 RMS
        otf_a = self.compute_otp(grid_size, wavelength, pixel_scale)
        otf_b = other_pipeline.compute_otp(grid_size, wavelength, pixel_scale)

        mtf_size = min(otf_a.mtf.shape[0], otf_b.mtf.shape[0])
        mtf_a = otf_a.mtf[:mtf_size, :mtf_size]
        mtf_b = otf_b.mtf[:mtf_size, :mtf_size]
        mtf_diff = float(np.sqrt(np.mean((mtf_a - mtf_b) ** 2)))

        comparison = PipelineComparison(
            name_a=self.config.name,
            name_b=other_pipeline.config.name,
            psf_difference_rms=round(psf_diff_rms, 8),
            psf_correlation=round(correlation, 6),
            strehl_difference=round(strehl_diff, 6),
            fwhm_difference_x=round(fwhm_diff_x, 4),
            fwhm_difference_y=round(fwhm_diff_y, 4),
            mtf_difference_rms=round(mtf_diff, 6),
        )

        LOGGER.info(
            "ComposableOpticalPipeline: 管线对比 '%s' vs '%s' "
            "(PSF RMS=%.6f, 相关=%.4f, Strehl差=%.4f)",
            comparison.name_a, comparison.name_b,
            comparison.psf_difference_rms, comparison.psf_correlation,
            comparison.strehl_difference,
        )

        return comparison

    def get_strehl_ratio(
        self,
        grid_size: Optional[int] = None,
        wavelength: Optional[float] = None,
        pixel_scale: Optional[float] = None,
    ) -> float:
        """计算 Strehl 比。

        Parameters
        ----------
        grid_size : int or None
            网格尺寸。
        wavelength : float or None
            波长 (米)。
        pixel_scale : float or None
            像素尺度 (米/像素)。

        Returns
        -------
        float
            Strehl 比 [0, 1]。
        """
        psf_result = self.compute_psf(grid_size, wavelength, pixel_scale)
        return psf_result.strehl_ratio

    def reset(self) -> None:
        """重置管线缓存。"""
        self._last_psf_result = None
        self._last_otf_result = None
        LOGGER.info("ComposableOpticalPipeline: 缓存已重置")

    # ======================== 内部方法 ========================

    def _compute_fwhm(self, psf: np.ndarray) -> Tuple[float, float]:
        """计算 PSF 的 FWHM。

        Parameters
        ----------
        psf : np.ndarray
            归一化 PSF 图像。

        Returns
        -------
        Tuple[float, float]
            (fwhm_x, fwhm_y) 像素。
        """
        size = psf.shape[0]
        half = size // 2

        # X 方向剖面
        profile_x = psf[half, :].astype(np.float64)
        fwhm_x = self._fwhm_from_profile(profile_x)

        # Y 方向剖面
        profile_y = psf[:, half].astype(np.float64)
        fwhm_y = self._fwhm_from_profile(profile_y)

        return (max(fwhm_x, 0.1), max(fwhm_y, 0.1))

    def _fwhm_from_profile(self, profile: np.ndarray) -> float:
        """从 1D 剖面计算 FWHM。

        Parameters
        ----------
        profile : np.ndarray
            1D 强度剖面。

        Returns
        -------
        float
            FWHM (像素)。
        """
        peak = profile.max()
        if peak < 1e-12:
            return 0.0

        half_max = peak / 2.0

        # 找到半高点
        above = profile >= half_max
        indices = np.where(above)[0]

        if len(indices) < 2:
            return 0.0

        left = indices[0]
        right = indices[-1]

        # 线性插值精化
        if left > 0 and profile[left - 1] < half_max:
            frac = (half_max - profile[left - 1]) / max(
                profile[left] - profile[left - 1], 1e-12
            )
            left = left - 1 + frac

        if right < len(profile) - 1 and profile[right + 1] < half_max:
            frac = (half_max - profile[right + 1]) / max(
                profile[right] - profile[right + 1], 1e-12
            )
            right = right + 1 - frac

        return float(right - left)

    def _compute_strehl(self, psf: np.ndarray, grid_size: int) -> float:
        """计算 Strehl 比。

        Strehl 比 = 实测 PSF 峰值 / 衍射极限 PSF 峰值。

        Parameters
        ----------
        psf : np.ndarray
            归一化 PSF。
        grid_size : int
            网格尺寸。

        Returns
        -------
        float
            Strehl 比。
        """
        # 实测峰值
        measured_peak = psf.max()

        # 衍射极限 PSF (均匀圆孔的 Airy 斑)
        # 使用理想圆孔的 PSF 作为参考
        ideal_psf = self._compute_ideal_psf(grid_size)
        ideal_peak = ideal_psf.max()

        if ideal_peak < 1e-12:
            return 0.0

        strehl = measured_peak / ideal_peak
        return min(max(strehl, 0.0), 1.0)

    def _compute_ideal_psf(self, grid_size: int) -> np.ndarray:
        """计算理想衍射极限 PSF。

        Parameters
        ----------
        grid_size : int
            网格尺寸。

        Returns
        -------
        np.ndarray
            归一化理想 PSF。
        """
        # 查找管线中的孔径元件
        aperture = None
        for element in self._elements:
            if isinstance(element, CircularAperture):
                aperture = element
                break

        if aperture is not None:
            transmission = aperture.get_transmission(grid_size)
        else:
            # 默认圆孔径
            diameter = self.config.aperture_diameter / self.config.pixel_scale
            radius = diameter / 2.0
            yy, xx = np.mgrid[:grid_size, :grid_size]
            cx, cy = grid_size / 2.0, grid_size / 2.0
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            transmission = (r <= radius).astype(np.float64)

        # 理想 PSF = |FT{circular aperture}|²
        field = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(transmission))
        )
        psf = np.abs(field) ** 2

        # 归一化
        total = psf.sum()
        if total > 1e-12:
            psf = psf / total

        return psf

    def _compute_encircled_energy_radius(
        self,
        psf: np.ndarray,
        target_fraction: float = 0.80,
    ) -> float:
        """计算包含指定比例能量的半径。

        Parameters
        ----------
        psf : np.ndarray
            归一化 PSF。
        target_fraction : float
            目标能量比例 (0~1)。

        Returns
        -------
        float
            半径 (像素)。
        """
        size = psf.shape[0]
        cx, cy = size / 2.0, size / 2.0

        yy, xx = np.mgrid[:size, :size]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float64)

        # 按半径排序
        r_flat = r.ravel()
        psf_flat = psf.ravel()

        sort_idx = np.argsort(r_flat)
        r_sorted = r_flat[sort_idx]
        psf_sorted = psf_flat[sort_idx]

        # 累积能量
        cum_energy = np.cumsum(psf_sorted)
        total_energy = cum_energy[-1]

        if total_energy < 1e-12:
            return 0.0

        cum_fraction = cum_energy / total_energy

        # 找到目标比例对应的半径
        target_idx = np.searchsorted(cum_fraction, target_fraction)
        if target_idx >= len(r_sorted):
            return float(r_sorted[-1])

        return float(r_sorted[target_idx])

    def _radial_average(
        self,
        image: np.ndarray,
        grid_size: int,
    ) -> np.ndarray:
        """计算图像的径向平均。

        Parameters
        ----------
        image : np.ndarray
            2D 图像。
        grid_size : int
            网格尺寸。

        Returns
        -------
        np.ndarray
            径向平均 (1D)。
        """
        cx, cy = grid_size / 2.0, grid_size / 2.0
        yy, xx = np.mgrid[:grid_size, :grid_size]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.int32)

        max_r = int(r.max())
        radial_avg = np.zeros(max_r + 1, dtype=np.float64)
        counts = np.zeros(max_r + 1, dtype=np.int32)

        r_flat = r.ravel()
        img_flat = image.ravel().astype(np.float64)

        for i in range(len(r_flat)):
            ri = r_flat[i]
            radial_avg[ri] += img_flat[i]
            counts[ri] += 1

        # 避免除零
        valid = counts > 0
        radial_avg[valid] /= counts[valid]

        return radial_avg

    def _empty_psf_result(
        self,
        grid_size: int,
        wavelength: float,
        pixel_scale: float,
    ) -> PSFResult:
        """返回空 PSF 结果。"""
        return PSFResult(
            psf=np.zeros((grid_size, grid_size)),
            peak_value=0.0,
            fwhm_x=0.0,
            fwhm_y=0.0,
            strehl_ratio=0.0,
            encircled_energy_80=0.0,
            total_energy=0.0,
            grid_size=grid_size,
            wavelength=wavelength,
            pixel_scale=pixel_scale,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=== 可组合光学管线引擎测试 ===\n")

    # 1. 创建基础管线 (圆孔径 + 薄透镜)
    config = PipelineConfig(
        name="ideal_system",
        grid_size=256,
        wavelength=632.8e-9,
        pixel_scale=1e-6,
        aperture_diameter=0.01,
        focal_length=0.1,
    )

    pipeline = ComposableOpticalPipeline(config)
    pipeline.add_element(CircularAperture(diameter_pixels=128.0))
    pipeline.add_element(ThinLens(
        focal_length=0.1,
        pixel_scale=1e-6,
        wavelength=632.8e-9,
    ))

    print(f"管线 '{config.name}': {pipeline.num_elements} 个元件")
    print(f"元件列表: {pipeline.element_names}")

    # 2. 计算理想 PSF
    print(f"\n--- 理想系统 PSF ---")
    psf_result = pipeline.compute_psf()
    print(f"  峰值: {psf_result.peak_value:.8f}")
    print(f"  FWHM: ({psf_result.fwhm_x:.2f}, {psf_result.fwhm_y:.2f}) px")
    print(f"  Strehl: {psf_result.strehl_ratio:.4f}")
    print(f"  80% EE 半径: {psf_result.encircled_energy_80:.2f} px")

    # 3. 计算理想 OTF
    print(f"\n--- 理想系统 OTF ---")
    otf_result = pipeline.compute_otp()
    print(f"  截止频率: {otf_result.cutoff_frequency:.0f} cycles/m")
    print(f"  Nyquist MTF: {otf_result.mtf_at_nyquist:.4f}")

    # 4. 添加像差
    print(f"\n--- 含像差系统 ---")
    aberrated_pipeline = ComposableOpticalPipeline(PipelineConfig(
        name="aberrated_system",
        grid_size=256,
        wavelength=632.8e-9,
        pixel_scale=1e-6,
    ))
    aberrated_pipeline.add_element(CircularAperture(diameter_pixels=128.0))
    aberrated_pipeline.add_element(ZernikeAberration(
        coefficients={4: 0.5, 5: 0.3, 11: 0.2},  # 离焦 + 像散 + 球差
        aperture_diameter_pixels=128.0,
        wavelength=632.8e-9,
    ))
    aberrated_pipeline.add_element(ThinLens(
        focal_length=0.1,
        pixel_scale=1e-6,
        wavelength=632.8e-9,
    ))

    psf_aberrated = aberrated_pipeline.compute_psf()
    print(f"  峰值: {psf_aberrated.peak_value:.8f}")
    print(f"  FWHM: ({psf_aberrated.fwhm_x:.2f}, {psf_aberrated.fwhm_y:.2f}) px")
    print(f"  Strehl: {psf_aberrated.strehl_ratio:.4f}")

    # 5. 离焦测试
    print(f"\n--- 离焦测试 ---")
    defocus_pipeline = ComposableOpticalPipeline(PipelineConfig(
        name="defocused_system",
        grid_size=256,
        wavelength=632.8e-9,
        pixel_scale=1e-6,
    ))
    defocus_pipeline.add_element(CircularAperture(diameter_pixels=128.0))
    defocus_pipeline.add_element(DefocusElement(
        defocus_waves=1.0,
        aperture_diameter_pixels=128.0,
        wavelength=632.8e-9,
    ))
    defocus_pipeline.add_element(ThinLens(
        focal_length=0.1,
        pixel_scale=1e-6,
        wavelength=632.8e-9,
    ))

    for defocus in [0.0, 0.5, 1.0, 2.0]:
        # 更新离焦量
        for elem in defocus_pipeline._elements:
            if isinstance(elem, DefocusElement):
                elem.defocus_waves = defocus
                elem.name = f"Defocus({defocus:.1f}λ)"
        result = defocus_pipeline.compute_psf()
        print(f"  离焦={defocus:.1f}λ: Strehl={result.strehl_ratio:.4f}, "
              f"FWHM=({result.fwhm_x:.2f}, {result.fwhm_y:.2f})")

    # 6. 管线对比
    print(f"\n--- 管线对比 ---")
    comparison = pipeline.compare_pipelines(aberrated_pipeline)
    print(f"  '{comparison.name_a}' vs '{comparison.name_b}':")
    print(f"  PSF 差异 RMS: {comparison.psf_difference_rms:.8f}")
    print(f"  PSF 相关系数: {comparison.psf_correlation:.4f}")
    print(f"  Strehl 差异: {comparison.strehl_difference:.4f}")
    print(f"  FWHM 差异: ({comparison.fwhm_difference_x:.2f}, "
          f"{comparison.fwhm_difference_y:.2f})")
    print(f"  MTF 差异 RMS: {comparison.mtf_difference_rms:.6f}")

    # 7. 元件操作测试
    print(f"\n--- 元件操作 ---")
    test_pipeline = ComposableOpticalPipeline(PipelineConfig(name="test"))
    test_pipeline.add_element(CircularAperture(diameter_pixels=128.0, name="aperture"))
    test_pipeline.add_element(ThinLens(focal_length=0.1, pixel_scale=1e-6,
                                       wavelength=632.8e-9, name="lens"))
    print(f"  初始: {test_pipeline.element_names}")

    test_pipeline.insert_element(
        ZernikeAberration(coefficients={4: 0.3}, name="aberration"),
        index=1,
    )
    print(f"  插入后: {test_pipeline.element_names}")

    test_pipeline.remove_element("aberration")
    print(f"  移除后: {test_pipeline.element_names}")

    # 8. Strehl 比快捷方法
    print(f"\n--- Strehl 比快捷方法 ---")
    strehl = pipeline.get_strehl_ratio()
    print(f"  理想系统 Strehl: {strehl:.4f}")
    strehl_ab = aberrated_pipeline.get_strehl_ratio()
    print(f"  含像差系统 Strehl: {strehl_ab:.4f}")

    print("\n测试完成")
