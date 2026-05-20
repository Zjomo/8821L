"""
无波前传感器校正器 (SensorlessWavefrontCorrector)

灵感来源:
- HCIPy (https://github.com/ehpor/hcipy) — 高对比度成像中的无传感器波前校正
- WeisongZhao/Adaptive-Optics-simulation — 轻量级自适应光学仿真
- Prysm (https://github.com/brandondube/prysm) — Zernike 多项式与 PSF 分析

算法原理:
- Sensorless AO — 无需波前传感器，通过图像质量指标反馈迭代校正
- Zernike Modal Correction — 逐模态校正 Zernike 像差
- Noll Indexing — 标准 Zernike 项编号 (Noll 1976)
- Image Metric Optimization — 基于图像锐度/Strehl 比的优化

功能:
- 无波前传感器的自适应光学校正
- 支持 Strehl 比、图像锐度、熵等多种图像质量指标
- Noll 序 Zernike 模态逐项校正
- 支持 Noll 序 Zernike 系数输入/输出
- 校正结果诊断与报告

依赖: numpy, scipy (无外部深度学习框架依赖)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.SensorlessWavefrontCorrector")


# ---------------------------------------------------------------------------
# Noll 索引到 (n, m) 的映射
# ---------------------------------------------------------------------------
_NOLL_TO_NM: Dict[int, Tuple[int, int]] = {
    1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0), 5: (2, 2),
    6: (2, -2), 7: (3, 1), 8: (3, -1), 9: (3, 3), 10: (3, -3),
    11: (4, 0), 12: (4, 2), 13: (4, -2), 14: (4, 4), 15: (4, -4),
}

_NOLL_NAMES: Dict[int, str] = {
    1: "Piston (平移)", 2: "Tilt X (X方向倾斜)", 3: "Tilt Y (Y方向倾斜)",
    4: "Defocus (离焦)", 5: "Astigmatism 45° (45°像散)",
    6: "Astigmatism 0° (0°像散)", 7: "Coma X (X方向彗差)",
    8: "Coma Y (Y方向彗差)", 9: "Trefoil Oblique (斜三叶草)",
    10: "Trefoil Vertical (竖三叶草)", 11: "Spherical (球差)",
    12: "2nd Astigmatism 45°", 13: "2nd Astigmatism 0°",
    14: "Quadrafoil Oblique", 15: "Quadrafoil Vertical",
}


def _noll_to_zern(noll_index: int) -> Tuple[int, int]:
    """将 Noll 索引转换为 (n, m)。"""
    if noll_index in _NOLL_TO_NM:
        return _NOLL_TO_NM[noll_index]
    # 通用 Noll 索引计算
    n = 0
    found = False
    while not found:
        n1 = n * (n + 1) // 2 + 1
        n2 = (n + 1) * (n + 2) // 2
        if n1 <= noll_index <= n2:
            found = True
        else:
            n += 1
    # 在第 n 阶内找到 m
    remaining = noll_index - n * (n + 1) // 2 - 1
    m_values = []
    for mi in range(n, -1, -1):
        if (n - mi) % 2 == 0:
            m_values.append(mi)
            if mi != 0:
                m_values.append(-mi)
    if remaining < len(m_values):
        m = m_values[remaining]
    else:
        m = 0
    return (n, m)


def _zernike_polynomial(
    n: int, m: int, rho: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """计算单个 Zernike 多项式值。

    Parameters
    ----------
    n : int
        径向阶数。
    m : int
        角向频率。
    rho : np.ndarray
        归一化径向坐标 [0, 1]。
    theta : np.ndarray
        角度坐标 [0, 2*pi]。

    Returns
    -------
    np.ndarray
        Zernike 多项式值。
    """
    # 径向多项式 R_n^|m|(rho)
    abs_m = abs(m)
    R = np.zeros_like(rho, dtype=np.float64)
    for s in range((n - abs_m) // 2 + 1):
        coeff = ((-1) ** s * np.math.factorial(n - s)
                 / (np.math.factorial(s)
                    * np.math.factorial((n + abs_m) // 2 - s)
                    * np.math.factorial((n - abs_m) // 2 - s)))
        R += coeff * rho ** (n - 2 * s)

    # 角向部分
    if m >= 0:
        Z = R * np.cos(abs_m * theta)
    else:
        Z = R * np.sin(abs_m * theta)

    return Z


def _generate_zernike_basis(
    n_modes: int,
    grid_size: int,
    pupil_radius: float = 1.0,
) -> np.ndarray:
    """生成 Zernike 基函数矩阵。

    Parameters
    ----------
    n_modes : int
        Zernike 模式数量 (Noll 索引 1 到 n_modes)。
    grid_size : int
        网格大小。
    pupil_radius : float
        出瞳半径 (像素)。

    Returns
    -------
    np.ndarray, shape (n_modes, grid_size, grid_size)
        Zernike 基函数矩阵。
    """
    cy, cx = grid_size // 2, grid_size // 2
    y, x = np.ogrid[:grid_size, :grid_size]
    rho = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / pupil_radius
    theta = np.arctan2(y - cy, x - cx)

    # 出瞳掩模
    mask = rho <= 1.0

    basis = np.zeros((n_modes, grid_size, grid_size), dtype=np.float64)
    for i in range(n_modes):
        noll_idx = i + 1
        n, m = _noll_to_zern(noll_idx)
        Z = _zernike_polynomial(n, m, rho, theta)
        Z[~mask] = 0.0
        basis[i] = Z

    return basis


@dataclass
class CorrectorConfig:
    """校正器配置参数。"""
    # --- Zernike 参数 ---
    n_modes: int = 15  # Zernike 模式数量
    grid_size: int = 128  # 网格大小
    pupil_radius: float = 1.0  # 出瞳半径 (归一化)

    # --- 优化参数 ---
    metric: str = "strehl"  # 图像质量指标: "strehl", "sharpness", "entropy"
    n_iterations: int = 50  # 每个模式的迭代次数
    step_size: float = 0.1  # 初始步长 (波长单位)
    step_decay: float = 0.95  # 步长衰减因子
    convergence_threshold: float = 1e-4  # 收敛阈值

    # --- 校正策略 ---
    start_mode: int = 2  # 起始模式 (跳过 piston)
    mode_order: str = "noll"  # 模式顺序: "noll", "radial"
    sequential_correction: bool = True  # 是否逐模式校正

    # --- 图像参数 ---
    wavelength: float = 0.55  # 波长 (微米)
    na: float = 0.95  # 数值孔径
    pixel_size: float = 0.1  # 像素大小 (微米)


@dataclass
class CorrectionResult:
    """校正结果。"""
    zernike_coefficients: Dict[int, float] = field(default_factory=dict)
    corrected_coefficients: Dict[int, float] = field(default_factory=dict)
    initial_metric: float = 0.0
    final_metric: float = 0.0
    improvement_ratio: float = 0.0
    n_iterations_per_mode: Dict[int, int] = field(default_factory=dict)
    converged: bool = False
    wavefront_rms_before: float = 0.0
    wavefront_rms_after: float = 0.0


class SensorlessWavefrontCorrector:
    """无波前传感器自适应光学校正器。

    通过迭代优化图像质量指标来估计和校正 Zernike 波前像差，
    无需直接波前传感器测量。

    Parameters
    ----------
    config : CorrectorConfig, optional
        校正器配置参数。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[CorrectorConfig] = None):
        self._cfg = config if config is not None else CorrectorConfig()
        self._basis: Optional[np.ndarray] = None
        self._current_coefficients: Dict[int, float] = {}
        self._initialized = False

    def _initialize_basis(self):
        """初始化 Zernike 基函数。"""
        self._basis = _generate_zernike_basis(
            self._cfg.n_modes,
            self._cfg.grid_size,
            self._cfg.pupil_radius,
        )
        self._initialized = True
        LOGGER.info(
            "已生成 %d 阶 Zernike 基函数 (网格 %dx%d)",
            self._cfg.n_modes, self._cfg.grid_size, self._cfg.grid_size,
        )

    def _compute_metric(self, image: np.ndarray) -> float:
        """计算图像质量指标。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        float
            图像质量指标值 (越大越好)。
        """
        image = np.asarray(image, dtype=np.float64)
        if image.max() > 0:
            image = image / image.max()

        metric_type = self._cfg.metric

        if metric_type == "strehl":
            # Strehl 比: PSF 峰值与理想衍射极限之比
            # 简化估计: 归一化峰值强度
            return float(np.max(image))

        elif metric_type == "sharpness":
            # 图像锐度: 像素值的平方和
            return float(np.sum(image ** 2))

        elif metric_type == "entropy":
            # 负熵 (越低熵越好，取负使越大越好)
            prob = image.flatten()
            prob = prob / (prob.sum() + 1e-12)
            prob = prob[prob > 1e-12]
            entropy = -np.sum(prob * np.log(prob + 1e-12))
            return float(-entropy)

        else:
            raise ValueError(f"未知图像质量指标: {metric_type}")

    def _estimate_mode_coefficient(
        self,
        mode_index: int,
        image_func,  # Callable[[np.ndarray], np.ndarray]
    ) -> Tuple[float, int]:
        """使用爬山法估计单个 Zernike 模式的系数。

        Parameters
        ----------
        mode_index : int
            Zernike 模式索引 (0-based)。
        image_func : callable
            接受 Zernike 系数字典，返回模拟图像的函数。

        Returns
        -------
        Tuple[float, int]
            (最优系数, 实际迭代次数)。
        """
        step = self._cfg.step_size
        coeff = self._current_coefficients.get(mode_index + 1, 0.0)

        prev_metric = self._compute_metric(image_func(self._current_coefficients))

        for iteration in range(self._cfg.n_iterations):
            # 尝试正方向
            test_coeffs = dict(self._current_coefficients)
            test_coeffs[mode_index + 1] = coeff + step
            metric_pos = self._compute_metric(image_func(test_coeffs))

            # 尝试负方向
            test_coeffs[mode_index + 1] = coeff - step
            metric_neg = self._compute_metric(image_func(test_coeffs))

            # 选择更好的方向
            if metric_pos >= metric_neg and metric_pos > prev_metric:
                coeff += step
                prev_metric = metric_pos
            elif metric_neg > metric_pos and metric_neg > prev_metric:
                coeff -= step
                prev_metric = metric_neg
            else:
                # 收敛，减小步长
                step *= self._cfg.step_decay

            self._current_coefficients[mode_index + 1] = coeff

            if step < self._cfg.convergence_threshold:
                LOGGER.debug(
                    "模式 %d (Noll %d) 在 %d 次迭代后收敛",
                    mode_index, mode_index + 1, iteration + 1,
                )
                return coeff, iteration + 1

        return coeff, self._cfg.n_iterations

    def _compute_wavefront_rms(self, coefficients: Dict[int, float]) -> float:
        """计算波前误差 RMS (不含 piston)。"""
        if not self._initialized:
            self._initialize_basis()

        total = np.zeros(
            (self._cfg.grid_size, self._cfg.grid_size), dtype=np.float64
        )
        for noll_idx, coeff in coefficients.items():
            if noll_idx <= 1:
                continue  # 跳过 piston
            mode_idx = noll_idx - 1
            if mode_idx < len(self._basis):
                total += coeff * self._basis[mode_idx]

        # 仅在出瞳内计算
        cy, cx = self._cfg.grid_size // 2, self._cfg.grid_size // 2
        y, x = np.ogrid[:self._cfg.grid_size, :self._cfg.grid_size]
        rho = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / self._cfg.pupil_radius
        mask = rho <= 1.0

        if np.sum(mask) == 0:
            return 0.0

        return float(np.sqrt(np.mean(total[mask] ** 2)))

    def correct(
        self,
        image_func,
        initial_coefficients: Optional[Dict[int, float]] = None,
    ) -> CorrectionResult:
        """执行无传感器波前校正。

        Parameters
        ----------
        image_func : callable
            接受 Zernike 系数字典 (Dict[int, float])，
            返回模拟/实际图像 (np.ndarray) 的函数。
        initial_coefficients : Dict[int, float], optional
            初始 Zernike 系数估计。为 None 时从零开始。

        Returns
        -------
        CorrectionResult
            校正结果。
        """
        if not self._initialized:
            self._initialize_basis()

        # 初始化系数
        if initial_coefficients is not None:
            self._current_coefficients = dict(initial_coefficients)
        else:
            self._current_coefficients = {}

        # 计算初始指标
        initial_image = image_func(self._current_coefficients)
        initial_metric = self._compute_metric(initial_image)
        initial_rms = self._compute_wavefront_rms(self._current_coefficients)

        LOGGER.info(
            "开始无传感器校正: 初始 %s=%.4f, 波前RMS=%.4f",
            self._cfg.metric, initial_metric, initial_rms,
        )

        # 记录校正前的系数
        result_coefficients = dict(self._current_coefficients)
        iterations_per_mode: Dict[int, int] = {}

        # 确定校正模式顺序
        if self._cfg.mode_order == "noll":
            mode_indices = list(range(
                self._cfg.start_mode - 1, self._cfg.n_modes
            ))
        elif self._cfg.mode_order == "radial":
            # 按径向阶数排序
            modes_with_order = []
            for i in range(self._cfg.start_mode - 1, self._cfg.n_modes):
                n, m = _noll_to_zern(i + 1)
                modes_with_order.append((n, i))
            modes_with_order.sort(key=lambda x: x[0])
            mode_indices = [m[1] for m in modes_with_order]
        else:
            raise ValueError(f"未知模式顺序: {self._cfg.mode_order}")

        # 逐模式校正
        for mode_idx in mode_indices:
            noll_idx = mode_idx + 1
            n, m = _noll_to_zern(noll_idx)
            name = _NOLL_NAMES.get(noll_idx, f"Z{noll_idx}")

            LOGGER.debug("校正模式 %d (Noll %d, n=%d, m=%d): %s",
                         mode_idx, noll_idx, n, m, name)

            opt_coeff, n_iter = self._estimate_mode_coefficient(
                mode_idx, image_func
            )
            iterations_per_mode[noll_idx] = n_iter

            LOGGER.debug(
                "模式 %d: 最优系数=%.6f, 迭代次数=%d",
                noll_idx, opt_coeff, n_iter,
            )

        # 计算最终指标
        final_image = image_func(self._current_coefficients)
        final_metric = self._compute_metric(final_image)
        final_rms = self._compute_wavefront_rms(self._current_coefficients)

        # 计算改善比
        if initial_metric > 0:
            improvement = final_metric / initial_metric
        else:
            improvement = 1.0

        converged = improvement > 1.0 + self._cfg.convergence_threshold

        result = CorrectionResult(
            zernike_coefficients=result_coefficients,
            corrected_coefficients=dict(self._current_coefficients),
            initial_metric=initial_metric,
            final_metric=final_metric,
            improvement_ratio=improvement,
            n_iterations_per_mode=iterations_per_mode,
            converged=converged,
            wavefront_rms_before=initial_rms,
            wavefront_rms_after=final_rms,
        )

        LOGGER.info(
            "校正完成: %s %.4f -> %.4f (改善 %.2fx), "
            "波前RMS %.4f -> %.4f, 收敛=%s",
            self._cfg.metric,
            initial_metric, final_metric, improvement,
            initial_rms, final_rms, converged,
        )

        return result

    def get_zernike_basis(self) -> Optional[np.ndarray]:
        """获取 Zernike 基函数矩阵。

        Returns
        -------
        np.ndarray or None
            形状为 (n_modes, grid_size, grid_size) 的基函数矩阵。
        """
        if not self._initialized:
            self._initialize_basis()
        return self._basis

    def get_coefficients(self) -> Dict[int, float]:
        """获取当前 Zernike 系数。

        Returns
        -------
        Dict[int, float]
            Noll 索引到系数的映射。
        """
        return dict(self._current_coefficients)

    def set_coefficients(self, coefficients: Dict[int, float]):
        """设置 Zernike 系数。

        Parameters
        ----------
        coefficients : Dict[int, float]
            Noll 索引到系数的映射。
        """
        self._current_coefficients = dict(coefficients)
        LOGGER.info("已设置 %d 个 Zernike 系数", len(coefficients))

    def reset(self):
        """重置校正器状态。"""
        self._current_coefficients.clear()
        self._initialized = False
        self._basis = None
        LOGGER.info("无传感器校正器已重置")
