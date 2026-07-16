"""光谱后处理滤波。

兼容现有 0_measurement_workflow_real_virtual_same_detection_6.25.py 中的处理逻辑。
"""

from typing import List, Sequence

import numpy as np
from scipy.ndimage import median_filter


DEFAULT_REMOVE_ABOVE = 3000.0
DEFAULT_MEDIAN_WINDOW = 5


def remove_above_threshold(values: Sequence[float], threshold: float) -> List[float]:
    """删除高于阈值的点（替换为 0）。"""
    if threshold is None or threshold <= 0:
        return list(values)
    return [float(v) if float(v) <= threshold else 0.0 for v in values]


def median_filter_1d(values: Sequence[float], window: int) -> List[float]:
    """一维中值滤波。

    与 scipy.ndimage.median_filter 行为一致，边缘做镜像填充。
    """
    if window is None or window <= 1:
        return list(values)

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []

    filtered = median_filter(arr, size=window, mode="mirror")
    return filtered.tolist()
