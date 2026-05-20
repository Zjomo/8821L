"""
Zernike 公共工具模块 (v7.0)

将分散在 modal_controller.py, phase_retrieval_analyzer.py, zernike_analyzer.py
中的重复 Zernike 多项式实现统一抽取到此模块，消除代码重复。

灵感来源:
- HCIPy (https://docs.hcipy.org/) — 高对比度成像仿真框架
- AOtools (https://github.com/aotools/aotools) — 自适应光学工具集
- Noll 1976, "Zernike polynomials and atmospheric turbulence"

外部依赖: numpy
"""

import numpy as np
from typing import Tuple, Optional

# Noll 索引到 (n, m) 的映射表 (Noll 1976)
# n = 径向阶数, m = 角频率 (含符号表示 cos/sin)
_NOLL_TO_NM: dict = {
    1: (0, 0),    # Piston
    2: (1, 1),    # Tip (X-tilt)
    3: (1, -1),   # Tilt (Y-tilt)
    4: (2, 0),    # Defocus
    5: (2, -2),   # Oblique Astigmatism
    6: (2, 2),    # Vertical Astigmatism
    7: (3, -1),   # Vertical Coma
    8: (3, 1),    # Horizontal Coma
    9: (3, -3),   # Oblique Trefoil
    10: (3, 3),   # Vertical Trefoil
    11: (4, 0),   # Primary Spherical
    12: (4, 2),   # Vertical Secondary Astigmatism
    13: (4, -2),  # Oblique Secondary Astigmatism
    14: (4, 4),   # Vertical Quadrafoil
    15: (4, -4),  # Oblique Quadrafoil
}

# Zernike 模式名称 (中文)
_NOLL_NAMES: dict = {
    1: "活塞 (Piston)",
    2: "倾斜X (Tip)",
    3: "倾斜Y (Tilt)",
    4: "离焦 (Defocus)",
    5: "像散45° (Oblique Astigmatism)",
    6: "像散0° (Vertical Astigmatism)",
    7: "彗差Y (Vertical Coma)",
    8: "彗差X (Horizontal Coma)",
    9: "三叶草45° (Oblique Trefoil)",
    10: "三叶草0° (Vertical Trefoil)",
    11: "球差 (Primary Spherical)",
    12: "二级像散0°",
    13: "二级像散45°",
    14: "四叶草0°",
    15: "四叶草45°",
}

# Noll 阶数索引 (按径向阶数分组)
_NOLL_INDICES_BY_ORDER: dict = {
    0: [1],
    1: [2, 3],
    2: [4, 5, 6],
    3: [7, 8, 9, 10],
    4: [11, 12, 13, 14, 15],
}


def noll_to_nm(j: int) -> Tuple[int, int]:
    """将 Noll 索引转换为 (n, m) 对。

    Args:
        j: Noll 索引 (从 1 开始)

    Returns:
        (n, m): 径向阶数和角频率

    Raises:
        ValueError: 如果 j 超出支持范围
    """
    if j in _NOLL_TO_NM:
        return _NOLL_TO_NM[j]
    raise ValueError(f"Noll index {j} not supported (max: {len(_NOLL_TO_NM)})")


def noll_name(j: int) -> str:
    """获取 Noll 索引对应的模式名称。

    Args:
        j: Noll 索引 (从 1 开始)

    Returns:
        模式名称字符串
    """
    return _NOLL_NAMES.get(j, f"Z{j}")


def zernike_radial(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
    """计算 Zernike 径向多项式 R_n^|m|(rho)。

    使用递推关系式计算，支持 n <= 6。

    Args:
        n: 径向阶数
        m_abs: 角频率的绝对值 |m|
        rho: 归一化径向坐标数组 (0 <= rho <= 1)

    Returns:
        径向多项式值数组，与 rho 同形状
    """
    if n > 6:
        raise ValueError(f"Radial order n={n} exceeds maximum supported order (6)")

    rho = np.asarray(rho, dtype=np.float64)
    result = np.zeros_like(rho)

    if n == 0 and m_abs == 0:
        result[:] = 1.0
    elif n == 1 and m_abs == 1:
        result[:] = rho
    elif n == 2 and m_abs == 0:
        result[:] = 2.0 * rho**2 - 1.0
    elif n == 2 and m_abs == 2:
        result[:] = rho**2
    elif n == 3 and m_abs == 1:
        result[:] = 3.0 * rho**3 - 2.0 * rho
    elif n == 3 and m_abs == 3:
        result[:] = rho**3
    elif n == 4 and m_abs == 0:
        result[:] = 6.0 * rho**4 - 6.0 * rho**2 + 1.0
    elif n == 4 and m_abs == 2:
        result[:] = 4.0 * rho**4 - 3.0 * rho**2
    elif n == 4 and m_abs == 4:
        result[:] = rho**4
    elif n == 5 and m_abs == 1:
        result[:] = 10.0 * rho**5 - 12.0 * rho**3 + 3.0 * rho
    elif n == 5 and m_abs == 3:
        result[:] = 5.0 * rho**5 - 4.0 * rho**3
    elif n == 5 and m_abs == 5:
        result[:] = rho**5
    elif n == 6 and m_abs == 0:
        result[:] = 20.0 * rho**6 - 30.0 * rho**4 + 12.0 * rho**2 - 1.0
    elif n == 6 and m_abs == 2:
        result[:] = 15.0 * rho**6 - 20.0 * rho**4 + 6.0 * rho**2
    elif n == 6 and m_abs == 4:
        result[:] = 6.0 * rho**6 - 5.0 * rho**4
    elif n == 6 and m_abs == 6:
        result[:] = rho**6

    return result


def zernike_polynomial(
    j: int,
    rho: np.ndarray,
    theta: np.ndarray
) -> np.ndarray:
    """计算第 j 个 Noll 索引对应的 Zernike 多项式值。

    Z_j(rho, theta) = R_n^|m|(rho) * cos(m*theta)  当 m > 0
    Z_j(rho, theta) = R_n^|m|(rho) * sin(|m|*theta) 当 m < 0
    Z_j(rho, theta) = R_n^0(rho)                     当 m = 0

    Args:
        j: Noll 索引 (从 1 开始)
        rho: 归一化径向坐标 (0 <= rho <= 1)
        theta: 角坐标 (弧度)

    Returns:
        Zernike 多项式值数组
    """
    n, m = noll_to_nm(j)
    m_abs = abs(m)
    R = zernike_radial(n, m_abs, rho)

    if m > 0:
        angular = np.cos(m_abs * theta)
    elif m < 0:
        angular = np.sin(m_abs * theta)
    else:
        angular = np.ones_like(theta)

    return R * angular


def zernike_basis(
    max_j: int,
    npix: int = 128,
    outside: float = 0.0
) -> Tuple[np.ndarray, list]:
    """生成 Zernike 基函数矩阵。

    Args:
        max_j: 最大 Noll 索引
        npix: 图像尺寸 (npix x npix)
        outside: 圆孔外的填充值

    Returns:
        (basis_matrix, noll_indices):
            basis_matrix: 形状 (max_j, npix, npix) 的基函数数组
            noll_indices: Noll 索引列表 [1, 2, ..., max_j]
    """
    coords = np.linspace(-1, 1, npix)
    xx, yy = np.meshgrid(coords, coords)
    rho = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    # 圆孔掩模
    mask = rho <= 1.0

    noll_indices = list(range(1, max_j + 1))
    basis = np.zeros((max_j, npix, npix), dtype=np.float64)

    for i, j in enumerate(noll_indices):
        Z = zernike_polynomial(j, rho, theta)
        Z[~mask] = outside
        basis[i] = Z

    return basis, noll_indices


def fit_zernike(
    phase_map: np.ndarray,
    max_j: int = 15,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """使用最小二乘法将相位图拟合为 Zernike 系数。

    Args:
        phase_map: 二维相位图 (弧度)
        max_j: 最大 Noll 索引 (默认 15，即到 n=4)
        mask: 可选掩模 (True = 有效像素)

    Returns:
        (coefficients, residual_map):
            coefficients: Zernike 系数数组，索引 0 对应 Noll j=1
            residual_map: 拟合残差图
    """
    ny, nx = phase_map.shape
    npix = min(ny, nx)

    # 居中裁剪为正方形
    cy, cx = ny // 2, nx // 2
    half = npix // 2
    cropped = phase_map[cy - half:cy + half, cx - half:cx + half]

    if mask is not None:
        cropped_mask = mask[cy - half:cy + half, cx - half:cx + half]
    else:
        coords = np.linspace(-1, 1, npix)
        xx, yy = np.meshgrid(coords, coords)
        cropped_mask = (xx**2 + yy**2) <= 1.0

    # 生成基函数
    basis, noll_indices = zernike_basis(max_j, npix)

    # 最小二乘拟合 (仅使用掩模内的像素)
    valid_pixels = cropped_mask.flatten()
    A = basis[:, valid_pixels].T  # (N_valid, max_j)
    b = cropped.flatten()[valid_pixels]

    # 正规方程: coeffs = (A^T A)^{-1} A^T b
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = np.zeros(max_j)

    # 计算残差
    reconstructed = (A @ coeffs).reshape(npix, npix)
    residual = cropped.copy()
    residual[cropped_mask] -= reconstructed[cropped_mask]

    return coeffs, residual


def compute_strehl_from_zernike(
    coefficients: np.ndarray,
    wavelength: float = 0.6328e-6,
    aperture_diameter: float = 0.01
) -> float:
    """从 Zernike 系数计算 Maréchal 近似 Strehl 比。

    Strehl ≈ exp(-(2π σ)^2)，其中 σ 为波前 RMS (单位: 波长)

    Args:
        coefficients: Zernike 系数数组 (索引 0 对应 Noll j=1)
        wavelength: 波长 (米)
        aperture_diameter: 孔径直径 (米)

    Returns:
        Strehl 比 (0 到 1 之间)
    """
    # 去除活塞项 (j=1)
    if len(coefficients) > 1:
        rms_waves = np.sqrt(np.mean(coefficients[1:]**2))
    else:
        rms_waves = 0.0

    strehl = float(np.exp(-(2.0 * np.pi * rms_waves)**2))
    return min(max(strehl, 0.0), 1.0)


def get_supported_max_order() -> int:
    """获取当前支持的最大 Zernike 径向阶数。"""
    return 6


def get_noll_indices_for_order(n: int) -> list:
    """获取指定径向阶数对应的所有 Noll 索引。

    Args:
        n: 径向阶数

    Returns:
        Noll 索引列表
    """
    return _NOLL_INDICES_BY_ORDER.get(n, [])
