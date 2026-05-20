"""
可微分光线追踪器 (DifferentiableRayTracer)

灵感来源:
- Optiland (https://github.com/optiland/optiland) — 纯 Python 光线追踪引擎，
  支持序列/非序列光学系统、像差分析、PSF 计算
- LightPipes — 光束传播与衍射仿真工具箱
- ABCD 矩阵光学 — 近轴光线传播的矩阵形式化方法

算法原理:
- 2D Paraxial Ray Tracing — 二维近轴光线追踪，
  光线用 [y, theta] (高度, 角度) 表示
- Thin Lens Model — 薄透镜模型: 近轴折射 + 可选 Seidel 像差修正
- Curved Mirror Reflection — 曲面镜反射: 焦距 f = R/2
- ABCD Matrix Formalism — ABCD 矩阵形式化系统级分析，
  级联矩阵描述完整光学系统
- PSF via Ray Histogram — 光线交点直方图法生成 PSF
- Numerical Gradient (Central Differences) — 中心差分法数值梯度，
  支持端到端参数优化
- Gradient Descent with Line Search — 梯度下降 + 回溯线搜索优化

功能:
- 构建多元素光学系统 (透镜、曲面镜、光阑)
- 2D 光线追踪 (近轴 + Seidel 像差)
- ABCD 矩阵系统分析
- PSF 计算与光斑尺寸评估
- 数值梯度计算 (中心差分)
- 端到端光学参数优化 (匹配目标 PSF)
- 系统健康状态报告

依赖: numpy
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.DifferentiableRayTracer")

# 模块默认禁用标志
differentiable_ray_tracer_enabled: bool = False


# ======================== 配置数据类 ========================


@dataclass
class RayTracerConfig:
    """光线追踪器配置。

    Parameters
    ----------
    max_elements : int
        光学系统最大元素数。
    wavelength : float
        默认波长 (米)。
    grid_size : int
        PSF 计算网格大小。
    gradient_step : float
        数值梯度步长 (相对步长)。
    image_plane_position : float or None
        像面位置 (米)。为 None 时自动取最后一个元素后方。
    propagation_step : float
        自由空间传播步长 (米)。
    psf_sigma : float
        PSF 高斯平滑核宽度 (像素)。
    optimization_learning_rate : float
        优化学习率。
    optimization_max_iter : int
        优化最大迭代次数。
    optimization_tolerance : float
        优化收敛容差。
    line_search_c1 : float
        Armijo 线搜索条件参数。
    line_search_rho : float
        回溯线搜索缩放因子。
    line_search_max_iter : int
        线搜索最大回溯次数。
    """
    max_elements: int = 20
    wavelength: float = 550e-9
    grid_size: int = 64
    gradient_step: float = 1e-6
    image_plane_position: Optional[float] = None
    propagation_step: float = 0.01
    psf_sigma: float = 1.0
    optimization_learning_rate: float = 0.001
    optimization_max_iter: int = 50
    optimization_tolerance: float = 1e-8
    line_search_c1: float = 1e-4
    line_search_rho: float = 0.5
    line_search_max_iter: int = 30


# ======================== 光学元素数据类 ========================


@dataclass
class OpticalElement:
    """光学元素。

    Parameters
    ----------
    element_type : str
        元素类型: 'lens', 'mirror', 'aperture_stop'。
    position : float
        沿光轴位置 (米)。
    parameters : Dict[str, Any]
        元素参数。
        - lens: {'focal_length': float, 'aperture': float,
                 'aberration_coeffs': Optional[Dict[str, float]]}
        - mirror: {'angle': float, 'curvature_radius': float}
        - aperture_stop: {'radius': float}
    """
    element_type: str = "lens"
    position: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """验证元素类型。"""
        valid_types = {"lens", "mirror", "aperture_stop"}
        if self.element_type not in valid_types:
            raise ValueError(
                f"无效的光学元素类型: '{self.element_type}'，"
                f"有效类型: {valid_types}"
            )


# ======================== 光线追踪结果数据类 ========================


@dataclass
class RayTraceResult:
    """光线追踪结果。

    Parameters
    ----------
    output_rays : np.ndarray
        输出光线，形状为 (N, 2)，每行 [y, theta]。
    psf : np.ndarray
        点扩散函数，形状为 (grid_size, grid_size)。
    spot_size : float
        光斑尺寸 (RMS 半径，米)。
    ray_positions_per_element : List[np.ndarray]
        每个元素处的光线高度列表。
    num_traced_rays : int
        成功追踪的光线数。
    num_blocked_rays : int
        被遮挡的光线数。
    """
    output_rays: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    psf: np.ndarray = field(default_factory=lambda: np.zeros((64, 64)))
    spot_size: float = 0.0
    ray_positions_per_element: List[np.ndarray] = field(default_factory=list)
    num_traced_rays: int = 0
    num_blocked_rays: int = 0


# ======================== 可微分光线追踪器 ========================


class DifferentiableRayTracer:
    """可微分光线追踪器。

    基于近轴近似的二维光线追踪引擎，支持透镜、曲面镜和光阑等
    光学元素。通过数值梯度计算实现端到端的光学系统参数优化。

    核心思想:
        将光学系统建模为可微分的计算图:
            rays -> [element_1] -> [element_2] -> ... -> image_plane -> PSF
        每个光学元素的操作均可微分，梯度通过中心差分法数值计算，
        支持对焦距、位置等参数的端到端优化。

    光线表示:
        2D 近轴光线用列向量 [y, theta]^T 表示:
            y: 光线相对于光轴的高度 (米)
            theta: 光线与光轴的夹角 (弧度)

    Parameters
    ----------
    config : RayTracerConfig or None
        追踪器配置。
    """

    def __init__(self, config: Optional[RayTracerConfig] = None) -> None:
        """初始化可微分光线追踪器。

        Parameters
        ----------
        config : RayTracerConfig or None
            追踪器配置。为 None 时使用默认配置。
        """
        self.config = config or RayTracerConfig()

        # 光学元素列表 (按位置排序)
        self._elements: List[OpticalElement] = []

        # 追踪统计
        self._total_trace_calls: int = 0
        self._total_optimization_calls: int = 0
        self._last_trace_time_s: float = 0.0

        LOGGER.info(
            "DifferentiableRayTracer: 初始化完成 "
            "(max_elements=%d, wavelength=%.1fnm, grid=%d, "
            "gradient_step=%.1e)",
            self.config.max_elements, self.config.wavelength * 1e9,
            self.config.grid_size, self.config.gradient_step,
        )

    # ======================== 光学元素管理 ========================

    def add_lens(
        self,
        position: float,
        focal_length: float,
        aperture: float,
        aberration_coeffs: Optional[Dict[str, float]] = None,
    ) -> int:
        """添加薄透镜元素。

        薄透镜模型:
            - 近轴折射: theta_out = theta_in - y / f
            - 可选 Seidel 像差修正 (球差、彗差、像散、场曲、畸变)

        Parameters
        ----------
        position : float
            透镜沿光轴位置 (米)。
        focal_length : float
            焦距 (米)。正值=会聚透镜，负值=发散透镜。
        aperture : float
            通光孔径半径 (米)。
        aberration_coeffs : Dict[str, float] or None
            Seidel 像差系数。支持:
                - 'S1': 球差 (spherical aberration)
                - 'S2': 彗差 (coma)
                - 'S3': 像散 (astigmatism)
                - 'S4': 场曲 (field curvature)
                - 'S5': 畸变 (distortion)

        Returns
        -------
        int
            元素索引。

        Raises
        ------
        ValueError
            参数不合法或超出最大元素数时抛出。
        """
        if len(self._elements) >= self.config.max_elements:
            raise ValueError(
                f"已达到最大元素数限制 ({self.config.max_elements})"
            )

        if abs(focal_length) < 1e-10:
            raise ValueError(f"焦距不能为零: focal_length={focal_length}")

        if aperture <= 0:
            raise ValueError(f"孔径必须为正: aperture={aperture}")

        # 默认像差系数为零
        coeffs = aberration_coeffs or {}
        valid_keys = {"S1", "S2", "S3", "S4", "S5"}
        for key in coeffs:
            if key not in valid_keys:
                LOGGER.warning(
                    "未知的像差系数: '%s'，有效系数: %s", key, valid_keys
                )

        element = OpticalElement(
            element_type="lens",
            position=float(position),
            parameters={
                "focal_length": float(focal_length),
                "aperture": float(aperture),
                "aberration_coeffs": dict(coeffs),
            },
        )

        self._elements.append(element)
        self._sort_elements()

        # 找到排序后的实际索引
        actual_index = self._elements.index(element)

        LOGGER.info(
            "DifferentiableRayTracer: 添加透镜 #%d "
            "(pos=%.4fm, f=%.4fm, aperture=%.4fm, aberrations=%s)",
            actual_index, position, focal_length, aperture,
            list(coeffs.keys()),
        )

        return actual_index

    def add_mirror(
        self,
        position: float,
        angle: float,
        curvature_radius: float,
    ) -> int:
        """添加曲面镜元素。

        曲面镜模型:
            - 焦距 f = R / 2 (R 为曲率半径)
            - 反射: theta_out = -theta_in - y / f_mirror
            - angle: 镜面法线与光轴的夹角 (弧度)

        Parameters
        ----------
        position : float
            镜面沿光轴位置 (米)。
        angle : float
            镜面法线与光轴的夹角 (弧度)。
        curvature_radius : float
            曲率半径 (米)。正值=凹面镜，负值=凸面镜。
            平面镜使用非常大的值 (如 1e10)。

        Returns
        -------
        int
            元素索引。

        Raises
        ------
        ValueError
            参数不合法或超出最大元素数时抛出。
        """
        if len(self._elements) >= self.config.max_elements:
            raise ValueError(
                f"已达到最大元素数限制 ({self.config.max_elements})"
            )

        if abs(curvature_radius) < 1e-10:
            raise ValueError(
                f"曲率半径不能为零: curvature_radius={curvature_radius}"
            )

        element = OpticalElement(
            element_type="mirror",
            position=float(position),
            parameters={
                "angle": float(angle),
                "curvature_radius": float(curvature_radius),
            },
        )

        self._elements.append(element)
        self._sort_elements()

        actual_index = self._elements.index(element)

        LOGGER.info(
            "DifferentiableRayTracer: 添加曲面镜 #%d "
            "(pos=%.4fm, angle=%.4frad, R=%.4fm, f=%.4fm)",
            actual_index, position, angle, curvature_radius,
            curvature_radius / 2.0,
        )

        return actual_index

    def add_aperture_stop(self, position: float, radius: float) -> int:
        """添加光阑 (孔径光阑)。

        光阑模型:
            - 阻挡 |y| > radius 的光线
            - 不改变通过光线的方向

        Parameters
        ----------
        position : float
            光阑沿光轴位置 (米)。
        radius : float
            光阑半径 (米)。

        Returns
        -------
        int
            元素索引。

        Raises
        ------
        ValueError
            参数不合法或超出最大元素数时抛出。
        """
        if len(self._elements) >= self.config.max_elements:
            raise ValueError(
                f"已达到最大元素数限制 ({self.config.max_elements})"
            )

        if radius <= 0:
            raise ValueError(f"光阑半径必须为正: radius={radius}")

        element = OpticalElement(
            element_type="aperture_stop",
            position=float(position),
            parameters={"radius": float(radius)},
        )

        self._elements.append(element)
        self._sort_elements()

        actual_index = self._elements.index(element)

        LOGGER.info(
            "DifferentiableRayTracer: 添加光阑 #%d "
            "(pos=%.4fm, radius=%.4fm)",
            actual_index, position, radius,
        )

        return actual_index

    def remove_element(self, index: int) -> None:
        """移除指定索引的光学元素。

        Parameters
        ----------
        index : int
            元素索引。

        Raises
        ------
        IndexError
            索引越界时抛出。
        """
        if index < 0 or index >= len(self._elements):
            raise IndexError(
                f"元素索引越界: {index} (共 {len(self._elements)} 个元素)"
            )
        removed = self._elements.pop(index)
        LOGGER.info(
            "DifferentiableRayTracer: 移除元素 #%d (type=%s, pos=%.4fm)",
            index, removed.element_type, removed.position,
        )

    def clear_elements(self) -> None:
        """清除所有光学元素。"""
        self._elements.clear()
        LOGGER.info("DifferentiableRayTracer: 已清除所有光学元素")

    def get_element(self, index: int) -> OpticalElement:
        """获取指定索引的光学元素。

        Parameters
        ----------
        index : int
            元素索引。

        Returns
        -------
        OpticalElement
            光学元素。

        Raises
        ------
        IndexError
            索引越界时抛出。
        """
        if index < 0 or index >= len(self._elements):
            raise IndexError(
                f"元素索引越界: {index} (共 {len(self._elements)} 个元素)"
            )
        return self._elements[index]

    def num_elements(self) -> int:
        """获取当前光学元素数量。

        Returns
        -------
        int
            元素数量。
        """
        return len(self._elements)

    def _sort_elements(self) -> None:
        """按位置排序光学元素。"""
        self._elements.sort(key=lambda e: e.position)

    # ======================== 光线追踪核心 ========================

    def trace_rays(
        self,
        rays: np.ndarray,
        num_elements: Optional[int] = None,
    ) -> RayTraceResult:
        """追踪光线通过光学系统。

        对每条光线 [y, theta]，依次通过所有光学元素，
        在每个元素处应用相应的变换 (折射/反射/遮挡)，
        最终在像面处收集光线位置。

        Parameters
        ----------
        rays : np.ndarray
            输入光线数组，形状为 (N, 2)，每行 [y, theta]。
            y: 光线高度 (米)，theta: 光线角度 (弧度)。
        num_elements : int or None
            追踪的光学元素数量。为 None 时追踪所有元素。

        Returns
        -------
        RayTraceResult
            光线追踪结果。

        Raises
        ------
        ValueError
            输入光线格式不合法时抛出。
        """
        start_time = time.perf_counter()

        if rays.ndim != 2 or rays.shape[1] != 2:
            raise ValueError(
                f"输入光线必须为形状 (N, 2) 的数组，当前形状: {rays.shape}"
            )

        if len(rays) == 0:
            LOGGER.warning("DifferentiableRayTracer: 输入光线为空")
            return RayTraceResult()

        if len(self._elements) == 0:
            LOGGER.warning("DifferentiableRayTracer: 无光学元素")
            return RayTraceResult(
                output_rays=rays.copy(),
                num_traced_rays=len(rays),
            )

        # 确定追踪的元素数量
        n_elem = num_elements if num_elements is not None else len(self._elements)
        n_elem = min(n_elem, len(self._elements))

        # 复制光线 (避免修改输入)
        current_rays = rays.astype(np.float64).copy()
        active_mask = np.ones(len(current_rays), dtype=bool)

        # 记录每个元素处的光线高度
        positions_per_element: List[np.ndarray] = []

        # 逐元素追踪
        for elem_idx in range(n_elem):
            elem = self._elements[elem_idx]
            positions_per_element.append(current_rays[:, 0].copy())

            # 自由空间传播到元素位置
            if elem_idx > 0:
                prev_pos = self._elements[elem_idx - 1].position
                distance = elem.position - prev_pos
                if distance > 0:
                    current_rays[active_mask, 0] += (
                        distance * current_rays[active_mask, 1]
                    )
            elif elem.position > 0:
                # 第一个元素不在原点，需要传播
                current_rays[active_mask, 0] += (
                    elem.position * current_rays[active_mask, 1]
                )

            # 应用元素变换
            active_mask = self._apply_element(
                current_rays, active_mask, elem
            )

        # 传播到像面
        image_pos = self._get_image_plane_position()
        last_elem_pos = self._elements[n_elem - 1].position
        if image_pos > last_elem_pos:
            distance_to_image = image_pos - last_elem_pos
            current_rays[active_mask, 0] += (
                distance_to_image * current_rays[active_mask, 1]
            )

        # 记录像面处的位置
        positions_per_element.append(current_rays[:, 0].copy())

        # 统计
        num_traced = int(np.sum(active_mask))
        num_blocked = len(current_rays) - num_traced

        # 计算 PSF
        psf = self._compute_psf_from_rays(current_rays[active_mask, 0])

        # 计算光斑尺寸 (RMS)
        if num_traced > 0:
            spot_y = current_rays[active_mask, 0]
            spot_size = float(np.sqrt(np.mean(spot_y ** 2)))
        else:
            spot_size = 0.0

        elapsed = time.perf_counter() - start_time
        self._total_trace_calls += 1
        self._last_trace_time_s = elapsed

        LOGGER.debug(
            "DifferentiableRayTracer: 追踪完成 "
            "(rays=%d, traced=%d, blocked=%d, spot_size=%.6fm, "
            "time=%.4fs)",
            len(rays), num_traced, num_blocked, spot_size, elapsed,
        )

        return RayTraceResult(
            output_rays=current_rays,
            psf=psf,
            spot_size=spot_size,
            ray_positions_per_element=positions_per_element,
            num_traced_rays=num_traced,
            num_blocked_rays=num_blocked,
        )

    def _apply_element(
        self,
        rays: np.ndarray,
        active_mask: np.ndarray,
        element: OpticalElement,
    ) -> np.ndarray:
        """对活跃光线应用光学元素变换。

        Parameters
        ----------
        rays : np.ndarray
            光线数组 (N, 2)，会被原地修改。
        active_mask : np.ndarray
            活跃光线掩码 (N,)。
        element : OpticalElement
            光学元素。

        Returns
        -------
        np.ndarray
            更新后的活跃光线掩码。
        """
        if element.element_type == "lens":
            return self._apply_lens(rays, active_mask, element)
        elif element.element_type == "mirror":
            return self._apply_mirror(rays, active_mask, element)
        elif element.element_type == "aperture_stop":
            return self._apply_aperture_stop(rays, active_mask, element)
        else:
            LOGGER.warning(
                "未知元素类型: '%s'，跳过", element.element_type
            )
            return active_mask

    def _apply_lens(
        self,
        rays: np.ndarray,
        active_mask: np.ndarray,
        element: OpticalElement,
    ) -> np.ndarray:
        """应用薄透镜变换。

        近轴折射:
            theta_out = theta_in - y / f

        Seidel 像差修正 (可选):
            - 球差 (S1): delta_theta = -S1 * y^3
            - 彗差 (S2): delta_theta = -S2 * y^2 * theta
            - 像散 (S3): delta_theta = -S3 * y * theta^2
            - 场曲 (S4): delta_theta = -S4 * theta^3
            - 畸变 (S5): delta_theta = -S5 * theta

        Parameters
        ----------
        rays : np.ndarray
            光线数组。
        active_mask : np.ndarray
            活跃光线掩码。
        element : OpticalElement
            透镜元素。

        Returns
        -------
        np.ndarray
            更新后的活跃掩码。
        """
        params = element.parameters
        f = params["focal_length"]
        aperture = params["aperture"]
        aberr = params.get("aberration_coeffs", {})

        # 孔径遮挡
        y_vals = rays[:, 0]
        blocked = active_mask & (np.abs(y_vals) > aperture)
        active_mask = active_mask & (~blocked)

        if not np.any(active_mask):
            return active_mask

        # 近轴折射
        idx = np.where(active_mask)[0]
        y = rays[idx, 0]
        theta = rays[idx, 1]

        # 基础折射
        theta_out = theta - y / f

        # Seidel 像差修正
        if aberr:
            S1 = aberr.get("S1", 0.0)
            S2 = aberr.get("S2", 0.0)
            S3 = aberr.get("S3", 0.0)
            S4 = aberr.get("S4", 0.0)
            S5 = aberr.get("S5", 0.0)

            delta_theta = (
                -S1 * y ** 3
                - S2 * y ** 2 * theta
                - S3 * y * theta ** 2
                - S4 * theta ** 3
                - S5 * theta
            )
            theta_out += delta_theta

        rays[idx, 1] = theta_out

        return active_mask

    def _apply_mirror(
        self,
        rays: np.ndarray,
        active_mask: np.ndarray,
        element: OpticalElement,
    ) -> np.ndarray:
        """应用曲面镜变换。

        反射模型:
            f_mirror = R / 2
            theta_out = -(theta_in + y / f_mirror)

        Parameters
        ----------
        rays : np.ndarray
            光线数组。
        active_mask : np.ndarray
            活跃光线掩码。
        element : OpticalElement
            镜面元素。

        Returns
        -------
        np.ndarray
            更新后的活跃掩码。
        """
        params = element.parameters
        R = params["curvature_radius"]
        angle = params["angle"]

        f_mirror = R / 2.0

        if not np.any(active_mask):
            return active_mask

        idx = np.where(active_mask)[0]
        y = rays[idx, 0]
        theta = rays[idx, 1]

        # 反射 + 折射 (曲面镜等效透镜)
        theta_out = -(theta + y / f_mirror)

        # 镜面角度修正 (简化模型)
        theta_out += 2.0 * angle

        rays[idx, 1] = theta_out

        return active_mask

    def _apply_aperture_stop(
        self,
        rays: np.ndarray,
        active_mask: np.ndarray,
        element: OpticalElement,
    ) -> np.ndarray:
        """应用光阑遮挡。

        Parameters
        ----------
        rays : np.ndarray
            光线数组。
        active_mask : np.ndarray
            活跃光线掩码。
        element : OpticalElement
            光阑元素。

        Returns
        -------
        np.ndarray
            更新后的活跃掩码。
        """
        radius = element.parameters["radius"]
        y_vals = rays[:, 0]
        blocked = active_mask & (np.abs(y_vals) > radius)
        active_mask = active_mask & (~blocked)

        return active_mask

    # ======================== PSF 计算 ========================

    def compute_psf(
        self,
        wavelength: float = 550e-9,
        grid_size: Optional[int] = None,
    ) -> np.ndarray:
        """计算点扩散函数 (PSF)。

        生成一组从轴上点发出的平行光线，追踪通过光学系统，
        在像面处收集光线位置并生成直方图 PSF。

        Parameters
        ----------
        wavelength : float
            波长 (米)，用于确定光线数量和衍射极限。
        grid_size : int or None
            PSF 网格大小。为 None 时使用配置值。

        Returns
        -------
        np.ndarray
            PSF 图像，形状为 (grid_size, grid_size)。
        """
        gs = grid_size or self.config.grid_size

        # 生成平行光线 (从轴上点发出)
        num_rays = max(256, gs * gs)
        max_height = self._estimate_system_aperture() * 0.9
        if max_height < 1e-10:
            max_height = 0.01  # 默认 1cm

        y_values = np.linspace(-max_height, max_height, num_rays)
        rays = np.column_stack([
            y_values,
            np.zeros(num_rays, dtype=np.float64),
        ])

        # 追踪光线
        result = self.trace_rays(rays)

        # 从追踪结果生成 PSF
        psf = self._compute_psf_from_rays(
            result.output_rays[:, 0], grid_size=gs
        )

        return psf

    def _compute_psf_from_rays(
        self,
        y_positions: np.ndarray,
        grid_size: Optional[int] = None,
    ) -> np.ndarray:
        """从光线像面位置生成 PSF。

        使用 2D 直方图方法: 将光线 y 坐标映射到 2D 网格
        (假设圆对称)，然后高斯平滑。

        Parameters
        ----------
        y_positions : np.ndarray
            像面处光线高度数组 (米)。
        grid_size : int or None
            PSF 网格大小。

        Returns
        -------
        np.ndarray
            PSF 图像。
        """
        gs = grid_size or self.config.grid_size

        if len(y_positions) == 0:
            return np.zeros((gs, gs), dtype=np.float64)

        # 确定像面范围
        y_range = np.max(np.abs(y_positions))
        if y_range < 1e-12:
            y_range = 1e-6  # 避免零范围

        # 将 1D y 坐标映射到 2D (假设圆对称)
        # 为每条光线生成一个随机角度，创建 2D 分布
        rng = np.random.RandomState(42)
        angles = rng.uniform(0, 2 * np.pi, len(y_positions))
        x_2d = y_positions * np.cos(angles)
        y_2d = y_positions * np.sin(angles)

        # 2D 直方图
        bins = gs
        psf, _, _ = np.histogram2d(
            y_2d, x_2d,
            bins=bins,
            range=[[-y_range, y_range], [-y_range, y_range]],
        )

        # 归一化
        total = psf.sum()
        if total > 1e-12:
            psf = psf / total

        # 高斯平滑
        sigma = self.config.psf_sigma
        if sigma > 0:
            psf = self._gaussian_smooth_2d(psf, sigma)

        return psf

    @staticmethod
    def _gaussian_smooth_2d(
        image: np.ndarray,
        sigma: float,
        kernel_size: Optional[int] = None,
    ) -> np.ndarray:
        """2D 高斯平滑。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        sigma : float
            高斯核标准差 (像素)。
        kernel_size : int or None
            核大小。为 None 时自动选择 (6*sigma+1)。

        Returns
        -------
        np.ndarray
            平滑后的图像。
        """
        if sigma <= 0:
            return image.copy()

        ks = kernel_size or max(3, int(6 * sigma + 1))
        if ks % 2 == 0:
            ks += 1

        # 生成 1D 高斯核
        half = ks // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        kernel_1d = np.exp(-x ** 2 / (2.0 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # 可分离卷积: 先水平再垂直
        smoothed = np.apply_along_axis(
            lambda row: np.convolve(row, kernel_1d, mode='same'),
            axis=1, arr=image,
        )
        smoothed = np.apply_along_axis(
            lambda col: np.convolve(col, kernel_1d, mode='same'),
            axis=0, arr=smoothed,
        )

        return smoothed

    def _get_image_plane_position(self) -> float:
        """获取像面位置。

        Returns
        -------
        float
            像面位置 (米)。
        """
        if self.config.image_plane_position is not None:
            return self.config.image_plane_position

        if len(self._elements) == 0:
            return 0.1  # 默认 10cm

        # 像面在最后一个元素后方 10cm
        return self._elements[-1].position + 0.1

    def _estimate_system_aperture(self) -> float:
        """估计系统最大通光孔径。

        Returns
        -------
        float
            最大孔径半径 (米)。
        """
        max_aperture = 0.0
        for elem in self._elements:
            if elem.element_type == "lens":
                max_aperture = max(max_aperture, elem.parameters["aperture"])
            elif elem.element_type == "aperture_stop":
                max_aperture = max(max_aperture, elem.parameters["radius"])
        return max_aperture if max_aperture > 0 else 0.01

    # ======================== ABCD 矩阵分析 ========================

    def get_system_matrix(self) -> np.ndarray:
        """获取光学系统的 ABCD 传递矩阵。

        将所有光学元素的 ABCD 矩阵级联，得到完整的系统矩阵。

        ABCD 矩阵定义:
            [y_out]     [A  B] [y_in]
            [theta_out] = [C  D] [theta_in]

        元素矩阵:
            - 自由空间传播距离 d:
                [[1, d], [0, 1]]
            - 薄透镜焦距 f:
                [[1, 0], [-1/f, 1]]
            - 曲面镜焦距 f_m = R/2:
                [[1, 0], [-1/f_m, 1]] (反射后方向反转)

        Returns
        -------
        np.ndarray
            2x2 ABCD 系统矩阵。

        Raises
        ------
        ValueError
            无光学元素时抛出。
        """
        if len(self._elements) == 0:
            raise ValueError("无光学元素，无法计算系统矩阵")

        # 单位矩阵
        system_matrix = np.eye(2, dtype=np.float64)

        prev_pos = 0.0

        for elem in self._elements:
            # 自由空间传播
            distance = elem.position - prev_pos
            if distance > 0:
                propagation = np.array([
                    [1.0, distance],
                    [0.0, 1.0],
                ], dtype=np.float64)
                system_matrix = propagation @ system_matrix

            # 元素矩阵
            if elem.element_type == "lens":
                f = elem.parameters["focal_length"]
                elem_matrix = np.array([
                    [1.0, 0.0],
                    [-1.0 / f, 1.0],
                ], dtype=np.float64)
                system_matrix = elem_matrix @ system_matrix

            elif elem.element_type == "mirror":
                R = elem.parameters["curvature_radius"]
                f_m = R / 2.0
                elem_matrix = np.array([
                    [1.0, 0.0],
                    [-1.0 / f_m, 1.0],
                ], dtype=np.float64)
                system_matrix = elem_matrix @ system_matrix

            # 光阑不影响 ABCD 矩阵 (仅遮挡光线)

            prev_pos = elem.position

        LOGGER.debug(
            "DifferentiableRayTracer: 系统矩阵 = [[%.6f, %.6f], [%.6f, %.6f]]",
            system_matrix[0, 0], system_matrix[0, 1],
            system_matrix[1, 0], system_matrix[1, 1],
        )

        return system_matrix

    def get_system_properties(self) -> Dict[str, float]:
        """从 ABCD 矩阵计算系统光学特性。

        Returns
        -------
        Dict[str, float]
            系统特性字典:
                - 'effective_focal_length': 有效焦距
                - 'back_focal_length': 后焦距
                - 'front_focal_length': 前焦距
                - 'magnification': 横向放大率
                - 'principal_plane_distance': 主面距离
        """
        if len(self._elements) == 0:
            return {}

        M = self.get_system_matrix()
        A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

        props: Dict[str, float] = {}

        # 有效焦距
        if abs(C) > 1e-15:
            props["effective_focal_length"] = -1.0 / C
        else:
            props["effective_focal_length"] = float("inf")

        # 后焦距 (BFD)
        if abs(C) > 1e-15:
            props["back_focal_length"] = A / C
        else:
            props["back_focal_length"] = float("inf")

        # 前焦距 (FFD)
        if abs(C) > 1e-15:
            props["front_focal_length"] = D / C
        else:
            props["front_focal_length"] = float("inf")

        # 横向放大率
        if abs(A) > 1e-15:
            props["magnification"] = 1.0 / A
        else:
            props["magnification"] = float("inf")

        # 行列式 (理想系统应为 1)
        props["determinant"] = float(A * D - B * C)

        return props

    # ======================== 数值梯度计算 ========================

    def compute_gradient(
        self,
        parameter_name: str,
        rays: np.ndarray,
        loss_fn: Callable[[RayTraceResult], float],
    ) -> float:
        """计算损失函数对系统参数的数值梯度。

        使用中心差分法:
            dL/dp = (L(p + eps) - L(p - eps)) / (2 * eps)

        Parameters
        ----------
        parameter_name : str
            参数名称，格式为 'element_{index}.{param_key}'。
            例如: 'element_0.focal_length', 'element_1.curvature_radius'。
        rays : np.ndarray
            输入光线数组 (N, 2)。
        loss_fn : Callable[[RayTraceResult], float]
            损失函数，接收 RayTraceResult，返回标量损失值。

        Returns
        -------
        float
            梯度值 dL/dp。

        Raises
        ------
        ValueError
            参数名称格式不合法或元素不存在时抛出。
        """
        # 解析参数名称
        elem_idx, param_key = self._parse_parameter_name(parameter_name)

        # 获取当前参数值
        original_value = self._get_parameter(elem_idx, param_key)
        if original_value is None:
            raise ValueError(
                f"无法获取参数: '{parameter_name}' "
                f"(element={elem_idx}, key={param_key})"
            )

        # 计算步长 (相对步长)
        eps = self.config.gradient_step * max(abs(original_value), 1e-10)

        # 正方向扰动
        self._set_parameter(elem_idx, param_key, original_value + eps)
        result_plus = self.trace_rays(rays)
        loss_plus = loss_fn(result_plus)

        # 负方向扰动
        self._set_parameter(elem_idx, param_key, original_value - eps)
        result_minus = self.trace_rays(rays)
        loss_minus = loss_fn(result_minus)

        # 恢复原始值
        self._set_parameter(elem_idx, param_key, original_value)

        # 中心差分
        gradient = (loss_plus - loss_minus) / (2.0 * eps)

        LOGGER.debug(
            "DifferentiableRayTracer: 梯度 dL/d(%s) = %.8e "
            "(eps=%.2e, L+=%.6e, L-=%.6e)",
            parameter_name, gradient, eps, loss_plus, loss_minus,
        )

        return float(gradient)

    def compute_gradient_vector(
        self,
        parameter_names: List[str],
        rays: np.ndarray,
        loss_fn: Callable[[RayTraceResult], float],
    ) -> np.ndarray:
        """计算多个参数的梯度向量。

        Parameters
        ----------
        parameter_names : List[str]
            参数名称列表。
        rays : np.ndarray
            输入光线数组。
        loss_fn : Callable[[RayTraceResult], float]
            损失函数。

        Returns
        -------
        np.ndarray
            梯度向量，形状为 (len(parameter_names),)。
        """
        gradients = np.zeros(len(parameter_names), dtype=np.float64)
        for i, name in enumerate(parameter_names):
            gradients[i] = self.compute_gradient(name, rays, loss_fn)
        return gradients

    def _parse_parameter_name(
        self, parameter_name: str
    ) -> Tuple[int, str]:
        """解析参数名称为 (元素索引, 参数键)。

        Parameters
        ----------
        parameter_name : str
            参数名称，格式: 'element_{index}.{param_key}'。

        Returns
        -------
        Tuple[int, str]
            (元素索引, 参数键)。

        Raises
        ------
        ValueError
            格式不合法时抛出。
        """
        if not parameter_name.startswith("element_"):
            raise ValueError(
                f"参数名称必须以 'element_' 开头: '{parameter_name}'"
            )

        parts = parameter_name.split(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"参数名称格式错误，期望 'element_{{idx}}.{{key}}': "
                f"'{parameter_name}'"
            )

        try:
            elem_idx = int(parts[0].replace("element_", ""))
        except ValueError:
            raise ValueError(
                f"无法解析元素索引: '{parts[0]}'"
            )

        if elem_idx < 0 or elem_idx >= len(self._elements):
            raise ValueError(
                f"元素索引越界: {elem_idx} (共 {len(self._elements)} 个元素)"
            )

        return elem_idx, parts[1]

    def _get_parameter(self, elem_idx: int, param_key: str) -> Optional[float]:
        """获取光学元素参数值。

        Parameters
        ----------
        elem_idx : int
            元素索引。
        param_key : str
            参数键。

        Returns
        -------
        float or None
            参数值。不存在时返回 None。
        """
        if elem_idx < 0 or elem_idx >= len(self._elements):
            return None

        elem = self._elements[elem_idx]

        # 特殊键: position
        if param_key == "position":
            return elem.position

        return elem.parameters.get(param_key)

    def _set_parameter(
        self, elem_idx: int, param_key: str, value: float
    ) -> None:
        """设置光学元素参数值。

        Parameters
        ----------
        elem_idx : int
            元素索引。
        param_key : str
            参数键。
        value : float
            新值。
        """
        if elem_idx < 0 or elem_idx >= len(self._elements):
            raise IndexError(f"元素索引越界: {elem_idx}")

        elem = self._elements[elem_idx]

        if param_key == "position":
            elem.position = value
            self._sort_elements()
        elif param_key == "focal_length":
            elem.parameters["focal_length"] = value
        elif param_key == "curvature_radius":
            elem.parameters["curvature_radius"] = value
        elif param_key == "aperture":
            elem.parameters["aperture"] = value
        elif param_key == "radius":
            elem.parameters["radius"] = value
        elif param_key == "angle":
            elem.parameters["angle"] = value
        elif param_key.startswith("aberration_"):
            # aberration_coeffs.S1 等
            aberr_key = param_key.replace("aberration_", "")
            if "aberration_coeffs" not in elem.parameters:
                elem.parameters["aberration_coeffs"] = {}
            elem.parameters["aberration_coeffs"][aberr_key] = value
        else:
            elem.parameters[param_key] = value

    # ======================== 参数优化 ========================

    def optimize_parameters(
        self,
        rays: np.ndarray,
        target_psf: np.ndarray,
        max_iter: Optional[int] = None,
        parameter_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """优化系统参数以匹配目标 PSF。

        使用梯度下降 + 回溯线搜索，最小化当前 PSF 与目标 PSF 之间的
        MSE 损失。

        Parameters
        ----------
        rays : np.ndarray
            输入光线数组 (N, 2)。
        target_psf : np.ndarray
            目标 PSF 图像。
        max_iter : int or None
            最大迭代次数。为 None 时使用配置值。
        parameter_names : List[str] or None
            需要优化的参数名称列表。为 None 时自动选择所有透镜的焦距。

        Returns
        -------
        Dict[str, Any]
            优化结果:
                - 'converged': bool 是否收敛
                - 'iterations': int 实际迭代次数
                - 'final_loss': float 最终损失值
                - 'loss_history': List[float] 损失历史
                - 'gradient_history': List[np.ndarray] 梯度历史
                - 'optimized_parameters': Dict[str, float] 优化后参数
                - 'elapsed_time_s': float 耗时
        """
        start_time = time.perf_counter()

        n_iter = max_iter or self.config.optimization_max_iter
        lr = self.config.optimization_learning_rate
        tol = self.config.optimization_tolerance

        # 自动选择优化参数
        if parameter_names is None:
            parameter_names = []
            for i, elem in enumerate(self._elements):
                if elem.element_type == "lens":
                    parameter_names.append(f"element_{i}.focal_length")
                elif elem.element_type == "mirror":
                    parameter_names.append(
                        f"element_{i}.curvature_radius"
                    )

        if not parameter_names:
            LOGGER.warning(
                "DifferentiableRayTracer: 无可优化参数"
            )
            return {
                "converged": False,
                "iterations": 0,
                "final_loss": 0.0,
                "loss_history": [],
                "gradient_history": [],
                "optimized_parameters": {},
                "elapsed_time_s": 0.0,
            }

        # 定义损失函数
        def _loss_fn(result: RayTraceResult) -> float:
            """MSE 损失: 当前 PSF 与目标 PSF 的均方误差。"""
            if result.psf.size != target_psf.size:
                # 重新计算 PSF 以匹配目标尺寸
                current_psf = self._compute_psf_from_rays(
                    result.output_rays[:, 0],
                    grid_size=target_psf.shape[0],
                )
            else:
                current_psf = result.psf

            # 归一化
            target_norm = target_psf / max(target_psf.sum(), 1e-12)
            current_norm = current_psf / max(current_psf.sum(), 1e-12)

            diff = current_norm - target_norm
            return float(np.mean(diff ** 2))

        # 记录初始参数值
        initial_params: Dict[str, float] = {}
        for name in parameter_names:
            idx, key = self._parse_parameter_name(name)
            val = self._get_parameter(idx, key)
            initial_params[name] = val if val is not None else 0.0

        loss_history: List[float] = []
        gradient_history: List[np.ndarray] = []
        converged = False

        LOGGER.info(
            "DifferentiableRayTracer: 开始优化 "
            "(params=%s, max_iter=%d, lr=%.6f, tol=%.2e)",
            parameter_names, n_iter, lr, tol,
        )

        # 初始损失
        initial_result = self.trace_rays(rays)
        current_loss = _loss_fn(initial_result)
        loss_history.append(current_loss)

        for it in range(1, n_iter + 1):
            # 计算梯度
            grad = self.compute_gradient_vector(
                parameter_names, rays, _loss_fn
            )
            gradient_history.append(grad.copy())

            # 梯度下降方向
            direction = -grad

            # 回溯线搜索
            step_size = self._line_search(
                rays, _loss_fn, parameter_names, direction, current_loss
            )

            # 参数更新
            for i, name in enumerate(parameter_names):
                idx, key = self._parse_parameter_name(name)
                current_val = self._get_parameter(idx, key)
                if current_val is not None:
                    new_val = current_val + step_size * direction[i]
                    self._set_parameter(idx, key, new_val)

            # 计算新损失
            new_result = self.trace_rays(rays)
            new_loss = _loss_fn(new_result)
            loss_history.append(new_loss)

            # 收敛检查
            loss_change = abs(current_loss - new_loss)
            if loss_change < tol and it > 3:
                converged = True
                LOGGER.info(
                    "DifferentiableRayTracer: 收敛于迭代 %d "
                    "(loss_change=%.2e < tol=%.2e)",
                    it, loss_change, tol,
                )
                break

            current_loss = new_loss

            if it % 10 == 0:
                LOGGER.debug(
                    "DifferentiableRayTracer: iter=%d/%d, "
                    "loss=%.6e, step=%.6e, grad_norm=%.4e",
                    it, n_iter, current_loss, step_size,
                    np.linalg.norm(grad),
                )

        # 收集优化后参数
        optimized_params: Dict[str, float] = {}
        for name in parameter_names:
            idx, key = self._parse_parameter_name(name)
            val = self._get_parameter(idx, key)
            optimized_params[name] = val if val is not None else 0.0

        elapsed = time.perf_counter() - start_time
        self._total_optimization_calls += 1

        result = {
            "converged": converged,
            "iterations": it if 'it' in dir() else n_iter,
            "final_loss": float(loss_history[-1]),
            "loss_history": loss_history,
            "gradient_history": gradient_history,
            "optimized_parameters": optimized_params,
            "elapsed_time_s": round(elapsed, 4),
        }

        LOGGER.info(
            "DifferentiableRayTracer: 优化完成. "
            "final_loss=%.6e, iterations=%d, converged=%s, "
            "elapsed=%.3fs",
            result["final_loss"], result["iterations"],
            result["converged"], result["elapsed_time_s"],
        )

        return result

    def _line_search(
        self,
        rays: np.ndarray,
        loss_fn: Callable[[RayTraceResult], float],
        parameter_names: List[str],
        direction: np.ndarray,
        current_loss: float,
    ) -> float:
        """回溯线搜索 (Armijo 条件)。

        Parameters
        ----------
        rays : np.ndarray
            输入光线。
        loss_fn : Callable
            损失函数。
        parameter_names : List[str]
            参数名称列表。
        direction : np.ndarray
            搜索方向。
        current_loss : float
            当前损失值。

        Returns
        -------
        float
            最优步长。
        """
        c1 = self.config.line_search_c1
        rho = self.config.line_search_rho
        max_ls_iter = self.config.line_search_max_iter

        alpha = self.config.optimization_learning_rate

        # 计算方向导数
        grad = -direction  # direction = -grad
        directional_deriv = float(grad @ direction)

        for _ in range(max_ls_iter):
            # 临时更新参数
            for i, name in enumerate(parameter_names):
                idx, key = self._parse_parameter_name(name)
                current_val = self._get_parameter(idx, key)
                if current_val is not None:
                    self._set_parameter(
                        idx, key, current_val + alpha * direction[i]
                    )

            # 评估损失
            result = self.trace_rays(rays)
            new_loss = loss_fn(result)

            # Armijo 充分下降条件
            if new_loss <= current_loss + c1 * alpha * directional_deriv:
                return alpha

            # 回溯: 恢复参数并缩小步长
            for i, name in enumerate(parameter_names):
                idx, key = self._parse_parameter_name(name)
                current_val = self._get_parameter(idx, key)
                if current_val is not None:
                    self._set_parameter(
                        idx, key, current_val - alpha * direction[i]
                    )

            alpha *= rho

        # 线搜索失败，返回最小步长
        LOGGER.debug(
            "DifferentiableRayTracer: 线搜索未找到合适步长，"
            "使用 alpha=%.2e", alpha,
        )
        return alpha

    # ======================== 系统健康报告 ========================

    def get_health_report(self) -> Dict[str, Any]:
        """返回追踪器状态报告。

        Returns
        -------
        Dict[str, Any]
            健康报告:
                - 'status': str 状态 ('healthy', 'warning', 'error')
                - 'num_elements': int 元素数量
                - 'element_types': Dict[str, int] 各类型元素数量
                - 'total_trace_calls': int 总追踪调用次数
                - 'total_optimization_calls': int 总优化调用次数
                - 'last_trace_time_s': float 上次追踪耗时
                - 'system_matrix': Optional[np.ndarray] 系统矩阵
                - 'system_properties': Dict[str, float] 系统特性
                - 'warnings': List[str] 警告信息
                - 'config_summary': Dict[str, Any] 配置摘要
        """
        warnings: List[str] = []
        status = "healthy"

        # 检查元素数量
        if len(self._elements) == 0:
            warnings.append("无光学元素")
            status = "warning"
        elif len(self._elements) >= self.config.max_elements:
            warnings.append(
                f"元素数量已达上限 ({self.config.max_elements})"
            )
            status = "warning"

        # 检查元素参数合理性
        for i, elem in enumerate(self._elements):
            if elem.element_type == "lens":
                f = elem.parameters.get("focal_length", 0)
                if abs(f) < 1e-6:
                    warnings.append(
                        f"透镜 #{i} 焦距过小: f={f:.2e}m"
                    )
                    status = "warning"
            elif elem.element_type == "mirror":
                R = elem.parameters.get("curvature_radius", 0)
                if abs(R) < 1e-6:
                    warnings.append(
                        f"曲面镜 #{i} 曲率半径过小: R={R:.2e}m"
                    )
                    status = "warning"

        # 检查元素位置重叠
        for i in range(1, len(self._elements)):
            pos_diff = abs(
                self._elements[i].position - self._elements[i - 1].position
            )
            if pos_diff < 1e-8:
                warnings.append(
                    f"元素 #{i} 与 #{i-1} 位置几乎重叠 "
                    f"(delta={pos_diff:.2e}m)"
                )
                status = "warning"

        # 统计元素类型
        type_counts: Dict[str, int] = {}
        for elem in self._elements:
            type_counts[elem.element_type] = (
                type_counts.get(elem.element_type, 0) + 1
            )

        # 系统矩阵和特性
        system_matrix = None
        system_properties = {}
        try:
            if len(self._elements) > 0:
                system_matrix = self.get_system_matrix()
                system_properties = self.get_system_properties()
        except Exception as e:
            warnings.append(f"系统矩阵计算失败: {str(e)}")
            status = "error"

        report = {
            "status": status,
            "num_elements": len(self._elements),
            "element_types": type_counts,
            "total_trace_calls": self._total_trace_calls,
            "total_optimization_calls": self._total_optimization_calls,
            "last_trace_time_s": round(self._last_trace_time_s, 6),
            "system_matrix": system_matrix,
            "system_properties": system_properties,
            "warnings": warnings,
            "config_summary": {
                "max_elements": self.config.max_elements,
                "wavelength_nm": self.config.wavelength * 1e9,
                "grid_size": self.config.grid_size,
                "gradient_step": self.config.gradient_step,
                "optimization_learning_rate": (
                    self.config.optimization_learning_rate
                ),
                "optimization_max_iter": self.config.optimization_max_iter,
            },
        }

        LOGGER.debug(
            "DifferentiableRayTracer: 健康报告 - status=%s, "
            "elements=%d, warnings=%d",
            status, len(self._elements), len(warnings),
        )

        return report

    def __repr__(self) -> str:
        """返回追踪器的字符串表示。

        Returns
        -------
        str
            字符串表示。
        """
        elem_str = ", ".join(
            f"{e.element_type}(pos={e.position:.4f})"
            for e in self._elements
        )
        return (
            f"DifferentiableRayTracer("
            f"elements=[{elem_str}], "
            f"config=RayTracerConfig("
            f"max_elements={self.config.max_elements}, "
            f"wavelength={self.config.wavelength * 1e9:.0f}nm"
            f"))"
        )


# ======================== 模块测试 ========================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("可微分光线追踪器 (DifferentiableRayTracer) 测试")
    print("=" * 60)

    # 创建追踪器
    tracer = DifferentiableRayTracer(
        config=RayTracerConfig(
            wavelength=550e-9,
            grid_size=64,
            gradient_step=1e-6,
        )
    )

    # 构建光学系统: 双透镜系统
    print("\n--- 构建光学系统 ---")
    tracer.add_lens(position=0.0, focal_length=0.05, aperture=0.01)
    tracer.add_lens(position=0.08, focal_length=0.03, aperture=0.008)
    tracer.add_aperture_stop(position=0.04, radius=0.006)

    print(f"系统元素数: {tracer.num_elements()}")
    print(f"系统表示: {tracer}")

    # 生成测试光线
    print("\n--- 光线追踪 ---")
    num_rays = 100
    y_vals = np.linspace(-0.005, 0.005, num_rays)
    rays = np.column_stack([
        y_vals,
        np.zeros(num_rays, dtype=np.float64),
    ])

    result = tracer.trace_rays(rays)
    print(f"追踪光线数: {result.num_traced_rays}")
    print(f"遮挡光线数: {result.num_blocked_rays}")
    print(f"光斑尺寸 (RMS): {result.spot_size * 1e6:.2f} um")
    print(f"输出光线形状: {result.output_rays.shape}")
    print(f"PSF 形状: {result.psf.shape}")
    print(f"PSF 峰值: {result.psf.max():.6f}")
    print(f"各元素处光线位置数: "
          f"{len(result.ray_positions_per_element)}")

    # ABCD 矩阵分析
    print("\n--- ABCD 矩阵分析 ---")
    M = tracer.get_system_matrix()
    print(f"系统矩阵:\n{M}")
    props = tracer.get_system_properties()
    for key, val in props.items():
        if isinstance(val, float) and abs(val) > 1e6:
            print(f"  {key}: inf")
        else:
            print(f"  {key}: {val:.6f}")

    # PSF 计算
    print("\n--- PSF 计算 ---")
    psf = tracer.compute_psf(wavelength=550e-9, grid_size=64)
    print(f"PSF 形状: {psf.shape}")
    print(f"PSF 总能量: {psf.sum():.6f}")
    print(f"PSF 峰值: {psf.max():.6f}")

    # 数值梯度
    print("\n--- 数值梯度计算 ---")

    def _spot_size_loss(r: RayTraceResult) -> float:
        """以光斑尺寸为损失。"""
        return r.spot_size

    grad = tracer.compute_gradient(
        "element_0.focal_length", rays, _spot_size_loss
    )
    print(f"d(spot_size)/d(f_0) = {grad:.8e}")

    grad2 = tracer.compute_gradient(
        "element_2.focal_length", rays, _spot_size_loss
    )
    print(f"d(spot_size)/d(f_1) = {grad2:.8e}")

    # 参数优化
    print("\n--- 参数优化 ---")
    target_psf = tracer.compute_psf(wavelength=550e-9, grid_size=32)

    # 扰动参数
    tracer._set_parameter(0, "focal_length", 0.055)

    opt_result = tracer.optimize_parameters(
        rays, target_psf, max_iter=20,
        parameter_names=["element_0.focal_length"],
    )
    print(f"优化收敛: {opt_result['converged']}")
    print(f"迭代次数: {opt_result['iterations']}")
    print(f"最终损失: {opt_result['final_loss']:.6e}")
    print(f"优化后参数: {opt_result['optimized_parameters']}")

    # 健康报告
    print("\n--- 系统健康报告 ---")
    report = tracer.get_health_report()
    print(f"状态: {report['status']}")
    print(f"元素数: {report['num_elements']}")
    print(f"元素类型: {report['element_types']}")
    print(f"追踪调用次数: {report['total_trace_calls']}")
    print(f"警告: {report['warnings']}")

    # 曲面镜测试
    print("\n--- 曲面镜测试 ---")
    tracer2 = DifferentiableRayTracer()
    tracer2.add_lens(position=0.0, focal_length=0.1, aperture=0.02)
    tracer2.add_mirror(position=0.15, angle=0.0, curvature_radius=0.2)

    result2 = tracer2.trace_rays(rays)
    print(f"含曲面镜系统 - 光斑尺寸: {result2.spot_size * 1e6:.2f} um")

    M2 = tracer2.get_system_matrix()
    print(f"含曲面镜系统矩阵:\n{M2}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
