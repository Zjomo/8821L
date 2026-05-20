"""
Zernike 像差分析器 (ZernikeAberrationAnalyzer)

灵感来源:
- AOtools (https://github.com/AOtools/aotools) — Zernike 多项式波前像差分解
- HCIPy (https://github.com/ehpor/hcipy) — 高对比度成像仿真中的波前分析
- ISO 24157:2023 — 眼科光学器件 Zernike 系数表示标准

算法原理:
- Zernike Polynomials — 正交多项式基函数，描述光学像差
- Least Squares Fitting — 最小二乘法拟合 Zernike 系数
- Noll Indexing — 标准 Zernike 项编号方案 (Noll 1976)

功能:
- 从光斑图像拟合 Zernike 像差系数
- 诊断对准偏差来源 (离焦、像散、彗差、球差等)
- 提供像差分解报告用于光学系统调试
- 计算波前误差 RMS 和 Strehl 比估计

依赖: numpy, opencv-python (仅用于图像预处理)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# Noll 索引到 (n, m) 的映射 (Noll 1976)
# j=1: piston (0,0), j=2: tilt_x (1,1), j=3: tilt_y (1,-1),
# j=4: defocus (2,0), j=5: astigmatism_45 (2,2), j=6: astigmatism_0 (2,-2), ...
_NOLL_TO_NM: Dict[int, Tuple[int, int]] = {
    1: (0, 0),   # Piston
    2: (1, 1),   # Tilt X
    3: (1, -1),  # Tilt Y
    4: (2, 0),   # Defocus
    5: (2, 2),   # Oblique Astigmatism
    6: (2, -2),  # Vertical Astigmatism
    7: (3, 1),   # Vertical Coma
    8: (3, -1),  # Horizontal Coma
    9: (3, 3),   # Oblique Trefoil
    10: (3, -3), # Vertical Trefoil
    11: (4, 0),  # Primary Spherical
    12: (4, 2),  # Secondary Astigmatism
    13: (4, -2), # Secondary Astigmatism
    14: (4, 4),  # Oblique Quadrafoil
    15: (4, -4), # Vertical Quadrafoil
}

# Zernike 项的中文名称
_NOLL_NAMES: Dict[int, str] = {
    1: "Piston (平移)",
    2: "Tilt X (X方向倾斜)",
    3: "Tilt Y (Y方向倾斜)",
    4: "Defocus (离焦)",
    5: "Astigmatism 45° (45°像散)",
    6: "Astigmatism 0° (0°像散)",
    7: "Coma X (X方向彗差)",
    8: "Coma Y (Y方向彗差)",
    9: "Trefoil Oblique (斜三叶草)",
    10: "Trefoil Vertical (竖三叶草)",
    11: "Spherical (球差)",
    12: "2nd Astigmatism 45°",
    13: "2nd Astigmatism 0°",
    14: "Quadrafoil Oblique",
    15: "Quadrafoil Vertical",
}


@dataclass
class ZernikeReport:
    """Zernike 像差分析报告。"""
    coefficients: Dict[int, float]  # Noll 索引 -> 系数
    names: Dict[int, str]  # Noll 索引 -> 名称
    wavefront_rms: float  # 波前误差 RMS (不含 piston)
    strehl_estimate: float  # Strehl 比估计
    dominant_aberration: str  # 主导像差名称
    tilt_x: float  # X 方向倾斜 (像素)
    tilt_y: float  # Y 方向倾斜 (像素)
    defocus: float  # 离焦量
    astigmatism: float  # 总像散量
    is_well_aligned: bool  # 是否对准良好 (低像差)


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """计算 Zernike 径向多项式 R_n^|m|(rho)。

    Parameters
    ----------
    n : int
        径向阶数。
    m : int
        角向频率 (可正可负)。
    rho : np.ndarray
        归一化径向坐标 [0, 1]。

    Returns
    -------
    np.ndarray
        径向多项式值。
    """
    abs_m = abs(m)
    R = np.zeros_like(rho, dtype=np.float64)

    if n == 0 and abs_m == 0:
        R[:] = 1.0
    elif n == 1 and abs_m == 1:
        R[:] = rho
    elif n == 2 and abs_m == 0:
        R[:] = 2.0 * rho ** 2 - 1.0
    elif n == 2 and abs_m == 2:
        R[:] = rho ** 2
    elif n == 3 and abs_m == 1:
        R[:] = 3.0 * rho ** 3 - 2.0 * rho
    elif n == 3 and abs_m == 3:
        R[:] = rho ** 3
    elif n == 4 and abs_m == 0:
        R[:] = 6.0 * rho ** 4 - 6.0 * rho ** 2 + 1.0
    elif n == 4 and abs_m == 2:
        R[:] = 4.0 * rho ** 4 - 3.0 * rho ** 2
    elif n == 4 and abs_m == 4:
        R[:] = rho ** 4
    else:
        # 通用递推公式 (仅支持到 n=4)
        raise ValueError(f"Zernike radial polynomial not implemented for n={n}, m={m}")

    return R


def _zernike_polynomial(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """计算单个 Zernike 多项式值。

    Parameters
    ----------
    n, m : int
        Zernike 阶数和角向频率。
    rho, theta : np.ndarray
        极坐标 (归一化半径, 角度)。

    Returns
    -------
    np.ndarray
        Zernike 多项式值。
    """
    R = _zernike_radial(n, m, rho)
    if m > 0:
        return R * np.cos(m * theta)
    elif m < 0:
        return R * np.sin(abs(m) * theta)
    else:
        return R


class ZernikeAberrationAnalyzer:
    """Zernike 像差分析器。

    从光斑图像拟合 Zernike 多项式系数，诊断光学像差来源。

    Parameters
    ----------
    max_order : int
        最大 Zernike 阶数 (1-4)。阶数越高分析越精细但计算量越大。
    aperture_radius : float or None
        分析孔径半径 (像素)。为 None 时自动从光斑尺寸估计。
    alignment_threshold : float
        判定对准良好的波前 RMS 阈值 (像素)。
    """

    def __init__(
        self,
        max_order: int = 4,
        aperture_radius: Optional[float] = None,
        alignment_threshold: float = 0.5,
    ):
        self.max_order = min(max(1, int(max_order)), 4)
        self.aperture_radius = float(aperture_radius) if aperture_radius else None
        self.alignment_threshold = float(alignment_threshold)

        # 构建 Zernike 基矩阵 (预计算)
        self._grid_size = 64  # 内部网格分辨率
        self._build_basis()

    def _build_basis(self) -> None:
        """预计算 Zernike 基矩阵。"""
        g = self._grid_size
        yy, xx = np.mgrid[:g, :g]
        cx, cy = g / 2.0, g / 2.0
        dx = xx - cx
        dy = yy - cy
        rho = np.sqrt(dx ** 2 + dy ** 2) / (g / 2.0)
        theta = np.arctan2(dy, dx)

        # 圆孔径掩模
        self._mask = rho <= 1.0
        self._rho = rho
        self._theta = theta

        # 选择 Noll 索引 <= max_order
        self._noll_indices = [
            j for j, (n, _) in _NOLL_TO_NM.items() if n <= self.max_order
        ]

        # 构建基矩阵 (仅孔径内像素)
        mask_flat = self._mask.ravel()
        n_pixels = int(mask_flat.sum())
        self._basis_matrix = np.zeros((n_pixels, len(self._noll_indices)), dtype=np.float64)

        for col_idx, j in enumerate(self._noll_indices):
            n, m = _NOLL_TO_NM[j]
            Z = _zernike_polynomial(n, m, rho, theta)
            self._basis_matrix[:, col_idx] = Z.ravel()[mask_flat]

    def analyze(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> ZernikeReport:
        """分析光斑图像的 Zernike 像差。

        Parameters
        ----------
        frame : np.ndarray
            BGR 格式的图像帧。
        bbox : Tuple[int, int, int, int]
            (x1, y1, x2, y2) 光斑边界框。

        Returns
        -------
        ZernikeReport
            Zernike 像差分析报告。
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(int(x2), w), min(int(y2), h)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return self._empty_report()

        crop = frame[y1:y2, x1:x2]
        import cv2
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # 将光斑图像归一化并调整到内部网格尺寸
        gray_resized = cv2.resize(gray, (self._grid_size, self._grid_size),
                                  interpolation=cv2.INTER_AREA)

        # 背景扣除
        bg = np.percentile(gray_resized, 10)
        signal = np.maximum(gray_resized - bg, 0.0)
        total = signal.sum()
        if total < 1e-6:
            return self._empty_report()

        # 归一化
        signal = signal / total

        # 仅取孔径内像素
        mask_flat = self._mask.ravel()
        signal_flat = signal.ravel()[mask_flat]

        # 最小二乘拟合 Zernike 系数
        try:
            coeffs, _, _, _ = np.linalg.lstsq(self._basis_matrix, signal_flat, rcond=None)
        except np.linalg.LinAlgError:
            return self._empty_report()

        # 构建 Noll 索引到系数的映射
        coefficients: Dict[int, float] = {}
        for idx, j in enumerate(self._noll_indices):
            coefficients[j] = float(coeffs[idx])

        # 计算波前 RMS (不含 piston j=1)
        rms_values = [coefficients[j] for j in self._noll_indices if j != 1]
        wavefront_rms = float(np.sqrt(np.mean(np.array(rms_values) ** 2))) if rms_values else 0.0

        # Strehl 比估计 (Maréchal 近似: S ≈ exp(-(2π σ)²))
        strehl = float(np.exp(-(2.0 * np.pi * wavefront_rms) ** 2))

        # 提取关键像差
        tilt_x = coefficients.get(2, 0.0)
        tilt_y = coefficients.get(3, 0.0)
        defocus_val = coefficients.get(4, 0.0)
        astig_45 = coefficients.get(5, 0.0)
        astig_0 = coefficients.get(6, 0.0)
        astigmatism = float(np.sqrt(astig_45 ** 2 + astig_0 ** 2))

        # 主导像差
        dominant = "None"
        max_coeff = 0.0
        for j in self._noll_indices:
            if j == 1:  # 跳过 piston
                continue
            if abs(coefficients.get(j, 0.0)) > max_coeff:
                max_coeff = abs(coefficients.get(j, 0.0))
                dominant = _NOLL_NAMES.get(j, f"Z{j}")

        is_well_aligned = wavefront_rms < self.alignment_threshold

        return ZernikeReport(
            coefficients=coefficients,
            names={j: _NOLL_NAMES.get(j, f"Z{j}") for j in self._noll_indices},
            wavefront_rms=round(wavefront_rms, 6),
            strehl_estimate=round(strehl, 6),
            dominant_aberration=dominant,
            tilt_x=round(tilt_x, 6),
            tilt_y=round(tilt_y, 6),
            defocus=round(defocus_val, 6),
            astigmatism=round(astigmatism, 6),
            is_well_aligned=is_well_aligned,
        )

    def _empty_report(self) -> ZernikeReport:
        """返回空报告。"""
        return ZernikeReport(
            coefficients={}, names={}, wavefront_rms=0.0,
            strehl_estimate=0.0, dominant_aberration="N/A",
            tilt_x=0.0, tilt_y=0.0, defocus=0.0, astigmatism=0.0,
            is_well_aligned=False,
        )

    def get_aberration_summary(self, report: ZernikeReport) -> str:
        """生成人类可读的像差摘要。"""
        lines = [
            f"Zernike 像差分析报告",
            f"{'=' * 40}",
            f"波前误差 RMS: {report.wavefront_rms:.4f} px",
            f"Strehl 比估计: {report.strehl_estimate:.4f}",
            f"主导像差: {report.dominant_aberration}",
            f"对准状态: {'良好' if report.is_well_aligned else '需调整'}",
            f"",
            f"关键像差系数:",
            f"  X 倾斜: {report.tilt_x:+.4f}",
            f"  Y 倾斜: {report.tilt_y:+.4f}",
            f"  离焦:   {report.defocus:+.4f}",
            f"  像散:   {report.astigmatism:.4f}",
        ]
        return "\n".join(lines)
