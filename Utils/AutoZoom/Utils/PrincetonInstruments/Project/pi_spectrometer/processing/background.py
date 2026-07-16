"""背景/暗场校正模块。

提供暗背景帧库管理与光谱数据校正，支持：
- 暗背景扣除（dark subtraction）
- 背景扣除（background subtraction）
- reference 归一化（平场校正）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from pi_spectrometer.core.types import SpectrometerResult


@dataclass
class BackgroundFrame:
    """单帧背景/暗场/参考数据。"""

    name: str
    y: np.ndarray
    kind: str = "dark"  # dark / background / reference
    metadata: Dict[str, Union[str, float, int]] = field(default_factory=dict)

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.y.ndim != 1:
            raise ValueError("BackgroundFrame.y 必须为一维数组")


class BackgroundFrameLibrary:
    """背景帧库：维护多组 dark/background/reference 帧。"""

    def __init__(self):
        self._frames: Dict[str, BackgroundFrame] = {}

    def add(self, frame: BackgroundFrame) -> None:
        """添加一帧到库中。"""
        self._frames[frame.name] = frame

    def remove(self, name: str) -> bool:
        """按名称删除帧。"""
        if name in self._frames:
            del self._frames[name]
            return True
        return False

    def get(self, name: str) -> Optional[BackgroundFrame]:
        """按名称获取帧。"""
        return self._frames.get(name)

    def list(self, kind: Optional[str] = None) -> List[str]:
        """列出所有帧名称，可按要求类型过滤。"""
        if kind is None:
            return list(self._frames.keys())
        return [name for name, frame in self._frames.items() if frame.kind == kind]

    def clear(self) -> None:
        """清空库。"""
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


def apply_background_correction(
    y: np.ndarray,
    dark: Optional[np.ndarray] = None,
    background: Optional[np.ndarray] = None,
    reference: Optional[np.ndarray] = None,
    clip_negative: bool = True,
) -> np.ndarray:
    """对一维光谱进行背景/暗场/参考校正。

    参数
    ----------
    y : np.ndarray
        原始光谱强度。
    dark : np.ndarray, optional
        暗背景帧，优先扣除。
    background : np.ndarray, optional
        背景帧，在 dark 之后扣除。
    reference : np.ndarray, optional
        参考/平场帧，用于归一化。
    clip_negative : bool
        是否将负值裁剪为 0。

    返回
    -------
    np.ndarray
        校正后的光谱。
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("输入光谱必须为一维数组")

    corrected = y.copy()

    if dark is not None:
        dark = np.asarray(dark, dtype=np.float64)
        _assert_same_length(corrected, dark, "dark")
        corrected = corrected - dark

    if background is not None:
        background = np.asarray(background, dtype=np.float64)
        _assert_same_length(corrected, background, "background")
        corrected = corrected - background

    if reference is not None:
        reference = np.asarray(reference, dtype=np.float64)
        _assert_same_length(corrected, reference, "reference")
        ref = reference.copy()
        if dark is not None:
            ref = ref - dark
        # 避免除以 0
        ref_safe = np.where(ref > 0, ref, 1.0)
        corrected = corrected / ref_safe

    if clip_negative:
        corrected = np.clip(corrected, 0.0, None)

    return corrected


def correct_spectrometer_result(
    result: SpectrometerResult,
    library: Optional[BackgroundFrameLibrary] = None,
    dark_name: Optional[str] = None,
    background_name: Optional[str] = None,
    reference_name: Optional[str] = None,
) -> SpectrometerResult:
    """基于 BackgroundFrameLibrary 校正 SpectrometerResult。

    返回一个新的 SpectrometerResult，raw_y 被校正，同时保留原始数据到
    metadata["raw_y_uncorrected"]。
    """
    dark = _lookup_frame(library, dark_name)
    background = _lookup_frame(library, background_name)
    reference = _lookup_frame(library, reference_name)

    corrected_y = apply_background_correction(
        result.raw_y, dark=dark, background=background, reference=reference
    )

    corrected = SpectrometerResult(
        ok=result.ok,
        index=result.index,
        num_points=result.num_points,
        raw_y=corrected_y,
        fit_y=result.fit_y,
        wavelength=result.wavelength,
        csv_path=result.csv_path,
        raw_original_peak=float(np.max(corrected_y)) if corrected_y.size else None,
        raw_filtered_peak=result.raw_filtered_peak,
        raw_median_peak=result.raw_median_peak,
        raw_peak=float(np.max(corrected_y)) if corrected_y.size else None,
        fit_peak=result.fit_peak,
        fit_params=result.fit_params,
        metadata=dict(result.metadata),
    )
    corrected.metadata["raw_y_uncorrected"] = result.raw_y.tolist()
    corrected.metadata["correction"] = {
        "dark": dark_name,
        "background": background_name,
        "reference": reference_name,
    }
    return corrected


def _lookup_frame(
    library: Optional[BackgroundFrameLibrary], name: Optional[str]
) -> Optional[np.ndarray]:
    if library is None or name is None:
        return None
    frame = library.get(name)
    return frame.y if frame is not None else None


def _assert_same_length(a: np.ndarray, b: np.ndarray, name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{name} 帧长度与光谱不匹配: {b.shape} vs {a.shape}"
        )
