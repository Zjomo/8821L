"""核心数据类型。"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np


@dataclass
class ROI:
    """光谱仪/相机 ROI。"""

    x: int = 0
    width: int = 0
    x_bin: int = 1
    y: int = 0
    height: int = 0
    y_bin: int = 1

    def __post_init__(self):
        self.x = int(self.x)
        self.width = int(self.width)
        self.x_bin = max(1, int(self.x_bin))
        self.y = int(self.y)
        self.height = int(self.height)
        self.y_bin = max(1, int(self.y_bin))


@dataclass
class SpectrometerResult:
    """光谱采集结果，与现有工作流兼容。"""

    ok: bool = True
    index: int = 0
    num_points: int = 0
    raw_y: np.ndarray = field(default_factory=lambda: np.array([]))
    fit_y: Optional[np.ndarray] = None
    wavelength: Optional[np.ndarray] = None
    csv_path: Optional[str] = None
    raw_original_peak: Optional[float] = None
    raw_filtered_peak: Optional[float] = None
    raw_median_peak: Optional[float] = None
    raw_peak: Optional[float] = None
    fit_peak: Optional[float] = None
    fit_params: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_workflow_dict(self) -> Dict[str, Any]:
        """转换为现有 MeasurementWorkflow 可消费的字典格式。"""
        return {
            "ok": self.ok,
            "index": self.index,
            "num_points": self.num_points,
            "raw_y": self.raw_y.tolist() if isinstance(self.raw_y, np.ndarray) else list(self.raw_y or []),
            "fit_y": self.fit_y.tolist() if isinstance(self.fit_y, np.ndarray) else (list(self.fit_y) if self.fit_y is not None else []),
            "wavelength": self.wavelength.tolist() if isinstance(self.wavelength, np.ndarray) else (list(self.wavelength) if self.wavelength is not None else []),
            "csv_path": self.csv_path,
            "raw_original_peak": self.raw_original_peak,
            "raw_filtered_peak": self.raw_filtered_peak,
            "raw_median_peak": self.raw_median_peak,
            "raw_peak": self.raw_peak,
            "fit_peak": self.fit_peak,
            "fit_params": self.fit_params,
            "metadata": self.metadata,
        }
