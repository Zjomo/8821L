"""峰值检测与拟合。"""

from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.optimize import curve_fit


def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float) -> np.ndarray:
    """高斯函数。"""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset


def gaussian_fit(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """对数据做高斯拟合，返回峰值和拟合参数。"""
    if x.size < 5 or y.size < 5:
        return None, None

    try:
        amplitude = float(np.max(y) - np.min(y))
        center = float(x[np.argmax(y)])
        sigma = float((x[-1] - x[0]) / 6.0) if x.size > 1 else 1.0
        offset = float(np.min(y))

        popt, _ = curve_fit(
            gaussian,
            x,
            y,
            p0=[amplitude, center, sigma, offset],
            maxfev=5000,
        )
        peak = float(gaussian(np.array([popt[1]]), *popt)[0])
        return peak, {
            "amplitude": float(popt[0]),
            "center": float(popt[1]),
            "sigma": float(popt[2]),
            "offset": float(popt[3]),
            "model": "gaussian",
        }
    except Exception:
        return None, None


def find_peak(y: np.ndarray, x: Optional[np.ndarray] = None) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """检测光谱峰值。

    返回：
        (fit_peak, fit_params)
    """
    if x is None:
        x = np.arange(len(y), dtype=np.float64)

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if y.size == 0:
        return None, None

    return gaussian_fit(x, y)
