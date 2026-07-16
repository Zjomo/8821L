"""光谱统计模块。

提供峰值、FWHM、质心、积分、SNR 等实时统计量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SpectrumStats:
    """单峰或整谱统计结果。"""

    peak_index: Optional[int] = None
    peak_x: Optional[float] = None
    peak_y: Optional[float] = None
    fwhm_x: Optional[float] = None
    fwhm_y: Optional[float] = None
    fwhm_left_x: Optional[float] = None
    fwhm_right_x: Optional[float] = None
    centroid_x: Optional[float] = None
    integral: Optional[float] = None
    baseline: Optional[float] = None
    snr: Optional[float] = None

    # 多峰场景
    num_peaks: int = 0
    peaks: List["SpectrumStats"] = field(default_factory=list)


def estimate_baseline(y: np.ndarray, percentile: float = 5.0) -> float:
    """用低分位数估计基线。"""
    if y.size == 0:
        return 0.0
    return float(np.percentile(y, percentile))


def estimate_snr(y: np.ndarray, peak_y: Optional[float] = None) -> float:
    """简单 SNR 估计：峰高 / 基线标准差。"""
    if y.size == 0:
        return 0.0
    baseline = estimate_baseline(y)
    noise_region = y[y <= baseline + (np.max(y) - baseline) * 0.1]
    if noise_region.size < 2:
        noise_region = y
    noise = float(np.std(noise_region)) + 1e-12
    signal = float(np.max(y)) if peak_y is None else float(peak_y)
    return signal / noise


def compute_centroid(x: np.ndarray, y: np.ndarray) -> float:
    """计算质心。"""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.size == 0 or np.sum(y) == 0:
        return float(np.mean(x)) if x.size else 0.0
    total = np.sum(y)
    return float(np.sum(x * y) / total)


def compute_integral(x: np.ndarray, y: np.ndarray) -> float:
    """用梯形法计算积分面积。"""
    if x.size == 0 or y.size == 0:
        return 0.0
    return float(np.trapz(y, x))


def find_fwhm(
    x: np.ndarray,
    y: np.ndarray,
    peak_index: Optional[int] = None,
    baseline: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """计算 FWHM 及左右边界。

    返回
    -------
    (fwhm_width, left_x, right_x)
    """
    if x.size < 3 or y.size < 3:
        return None, None, None

    if peak_index is None:
        peak_index = int(np.argmax(y))
    peak_index = max(0, min(peak_index, len(y) - 1))

    if baseline is None:
        baseline = estimate_baseline(y)

    peak_y = float(y[peak_index])
    half = baseline + (peak_y - baseline) / 2.0

    # 向左找半高点
    left_idx = peak_index
    for i in range(peak_index, -1, -1):
        left_idx = i
        if y[i] < half:
            break

    # 向右找半高点
    right_idx = peak_index
    for i in range(peak_index, len(y)):
        right_idx = i
        if y[i] < half:
            break

    # 线性插值获得更精确的 x
    def _interp_x(idx_a: int, idx_b: int) -> Optional[float]:
        if idx_a < 0 or idx_b >= len(y) or idx_a == idx_b:
            return float(x[idx_a]) if 0 <= idx_a < len(x) else None
        ya, yb = float(y[idx_a]), float(y[idx_b])
        if yb == ya:
            return float((x[idx_a] + x[idx_b]) / 2.0)
        t = (half - ya) / (yb - ya)
        return float(x[idx_a] + t * (x[idx_b] - x[idx_a]))

    left_x = _interp_x(left_idx, left_idx + 1)
    right_x = _interp_x(right_idx - 1, right_idx)

    if left_x is None or right_x is None:
        return None, left_x, right_x

    return right_x - left_x, left_x, right_x


def compute_spectrum_stats(
    x: Optional[np.ndarray],
    y: np.ndarray,
    find_peaks: bool = False,
    peak_prominence: Optional[float] = None,
    peak_distance: int = 5,
) -> SpectrumStats:
    """计算光谱统计量。

    参数
    ----------
    x : np.ndarray or None
        横坐标，None 时使用索引。
    y : np.ndarray
        光谱强度。
    find_peaks : bool
        是否进行多峰检测。
    peak_prominence : float
        峰谷相对 prominence 阈值，默认基于基线+半峰高估算。
    peak_distance : int
        峰之间最小距离（点数）。
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return SpectrumStats()

    x = np.asarray(x, dtype=np.float64) if x is not None else np.arange(len(y), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x 与 y 长度必须相同")

    baseline = estimate_baseline(y)
    peak_index = int(np.argmax(y))
    peak_x = float(x[peak_index])
    peak_y = float(y[peak_index])

    fwhm_width, fwhm_left_x, fwhm_right_x = find_fwhm(x, y, peak_index, baseline)
    centroid_x = compute_centroid(x, y)
    integral = compute_integral(x, y)
    snr = estimate_snr(y, peak_y)

    stats = SpectrumStats(
        peak_index=peak_index,
        peak_x=peak_x,
        peak_y=peak_y,
        fwhm_x=fwhm_width,
        fwhm_y=(baseline + (peak_y - baseline) / 2.0) if peak_y is not None else None,
        fwhm_left_x=fwhm_left_x,
        fwhm_right_x=fwhm_right_x,
        centroid_x=centroid_x,
        integral=integral,
        baseline=baseline,
        snr=snr,
    )

    if find_peaks:
        stats.peaks = _find_multiple_peaks(x, y, baseline, peak_prominence, peak_distance)
        stats.num_peaks = len(stats.peaks)

    return stats


def _find_multiple_peaks(
    x: np.ndarray,
    y: np.ndarray,
    baseline: float,
    prominence: Optional[float],
    distance: int,
) -> List[SpectrumStats]:
    """简单多峰检测。"""
    if prominence is None:
        # 默认 prominence 为基线到峰高的 20%
        prominence = (np.max(y) - baseline) * 0.2

    # 先平滑寻找局部极大值
    peaks: List[SpectrumStats] = []
    n = len(y)
    for i in range(distance, n - distance):
        window = y[i - distance : i + distance + 1]
        if y[i] == np.max(window) and y[i] > baseline + prominence:
            # 避免重复峰
            if peaks and i - peaks[-1].peak_index < distance:
                if y[i] > (peaks[-1].peak_y or 0):
                    peaks[-1] = _stats_for_peak(x, y, baseline, i)
                continue
            peaks.append(_stats_for_peak(x, y, baseline, i))

    return peaks


def _stats_for_peak(
    x: np.ndarray, y: np.ndarray, baseline: float, peak_index: int
) -> SpectrumStats:
    fwhm_width, left_x, right_x = find_fwhm(x, y, peak_index, baseline)
    return SpectrumStats(
        peak_index=peak_index,
        peak_x=float(x[peak_index]),
        peak_y=float(y[peak_index]),
        fwhm_x=fwhm_width,
        fwhm_y=baseline + (float(y[peak_index]) - baseline) / 2.0,
        fwhm_left_x=left_x,
        fwhm_right_x=right_x,
    )
