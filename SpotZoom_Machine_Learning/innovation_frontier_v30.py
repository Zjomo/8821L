"""
SpotZoom innovation frontier v30.

This module provides a lightweight phase-lock refinement stage inspired by:
- OpenCV phase correlation (subpixel translation estimation)
- Trackpy iterative center refinement patterns
- CoTracker/Norfair temporal continuity constraints

The implementation is intentionally lightweight and depends on numpy + opencv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional import guard
    cv2 = None


@dataclass
class PhaseLockConfig:
    patch_radius: int = 24
    response_threshold: float = 0.12
    max_shift_px: float = 20.0
    blend: float = 0.35
    min_patch_std: float = 2.0
    min_refine_shift_px: float = 0.25
    min_confidence: float = 0.05
    use_hanning_window: bool = True


@dataclass
class PhaseLockResult:
    center: Tuple[float, float]
    confidence: float
    refined: bool
    rejected: bool
    reason: Optional[str]
    shift: Tuple[float, float]
    response: float
    predicted_center: Tuple[float, float]
    patch_std: float


class PhaseLockRefiner:
    def __init__(self, config: Optional[PhaseLockConfig] = None):
        self.config = config or PhaseLockConfig()
        self._prev_patch: Optional[np.ndarray] = None
        self._prev_center: Optional[Tuple[float, float]] = None
        self._window_cache_shape: Optional[Tuple[int, int]] = None
        self._window_cache: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_patch = None
        self._prev_center = None
        self._window_cache_shape = None
        self._window_cache = None

    @staticmethod
    def _safe_float(value: float, default: float = 0.0) -> float:
        try:
            x = float(value)
        except Exception:
            return default
        if not np.isfinite(x):
            return default
        return x

    @staticmethod
    def _as_gray_float32(frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim == 3:
            if cv2 is None:
                return None
            try:
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            except Exception:
                return None
        elif arr.ndim == 2:
            gray = arr
        else:
            return None
        gray32 = np.asarray(gray, dtype=np.float32)
        if gray32.size == 0:
            return None
        return gray32

    def _extract_patch(self, gray: np.ndarray, center: Tuple[float, float]) -> Optional[np.ndarray]:
        h, w = gray.shape[:2]
        if h <= 0 or w <= 0:
            return None
        radius = max(2, int(self.config.patch_radius))
        size = radius * 2 + 1

        cx = self._safe_float(center[0], default=np.nan)
        cy = self._safe_float(center[1], default=np.nan)
        if not np.isfinite(cx) or not np.isfinite(cy):
            return None

        xi = int(round(cx))
        yi = int(round(cy))

        x0 = xi - radius
        x1 = xi + radius + 1
        y0 = yi - radius
        y1 = yi + radius + 1

        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - w)
        pad_bottom = max(0, y1 - h)

        if pad_left or pad_top or pad_right or pad_bottom:
            src = np.pad(
                gray,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="edge",
            )
            x0 += pad_left
            x1 += pad_left
            y0 += pad_top
            y1 += pad_top
        else:
            src = gray

        patch = src[y0:y1, x0:x1]
        if patch.shape != (size, size):
            return None
        return np.asarray(patch, dtype=np.float32)

    def _get_window(self, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if not bool(self.config.use_hanning_window):
            return None
        if self._window_cache is not None and self._window_cache_shape == shape:
            return self._window_cache

        h, w = shape
        if h <= 1 or w <= 1:
            return None

        window = None
        if cv2 is not None:
            try:
                window = cv2.createHanningWindow((w, h), cv2.CV_32F)
            except Exception:
                window = None
        if window is None:
            wy = np.hanning(h).astype(np.float32)
            wx = np.hanning(w).astype(np.float32)
            window = np.outer(wy, wx).astype(np.float32)

        self._window_cache_shape = shape
        self._window_cache = window
        return window

    def update(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
    ) -> PhaseLockResult:
        conf = float(np.clip(self._safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = self._safe_float(center[0], default=np.nan)
        cy = self._safe_float(center[1], default=np.nan)
        if not np.isfinite(cx) or not np.isfinite(cy):
            return PhaseLockResult(
                center=(0.0, 0.0),
                confidence=conf,
                refined=False,
                rejected=True,
                reason="invalid_center",
                shift=(0.0, 0.0),
                response=0.0,
                predicted_center=(0.0, 0.0),
                patch_std=float("inf"),
            )

        if cv2 is None:
            return PhaseLockResult(
                center=(cx, cy),
                confidence=conf,
                refined=False,
                rejected=True,
                reason="opencv_unavailable",
                shift=(0.0, 0.0),
                response=0.0,
                predicted_center=(cx, cy),
                patch_std=0.0,
            )

        gray = self._as_gray_float32(frame)
        if gray is None:
            return PhaseLockResult(
                center=(cx, cy),
                confidence=conf,
                refined=False,
                rejected=True,
                reason="invalid_frame",
                shift=(0.0, 0.0),
                response=0.0,
                predicted_center=(cx, cy),
                patch_std=0.0,
            )

        curr_patch = self._extract_patch(gray, (cx, cy))
        if curr_patch is None:
            return PhaseLockResult(
                center=(cx, cy),
                confidence=conf,
                refined=False,
                rejected=True,
                reason="invalid_patch",
                shift=(0.0, 0.0),
                response=0.0,
                predicted_center=(cx, cy),
                patch_std=0.0,
            )

        patch_std = float(np.std(curr_patch))
        if self._prev_patch is None or self._prev_center is None:
            self._prev_patch = curr_patch.copy()
            self._prev_center = (cx, cy)
            return PhaseLockResult(
                center=(cx, cy),
                confidence=conf,
                refined=False,
                rejected=False,
                reason=None,
                shift=(0.0, 0.0),
                response=1.0,
                predicted_center=(cx, cy),
                patch_std=patch_std,
            )

        if patch_std < max(0.0, float(self.config.min_patch_std)):
            self._prev_patch = curr_patch.copy()
            self._prev_center = (cx, cy)
            return PhaseLockResult(
                center=(cx, cy),
                confidence=conf,
                refined=False,
                rejected=True,
                reason="patch_low_texture",
                shift=(0.0, 0.0),
                response=0.0,
                predicted_center=(cx, cy),
                patch_std=patch_std,
            )

        window = self._get_window(curr_patch.shape)
        shift_raw, response_raw = cv2.phaseCorrelate(self._prev_patch, curr_patch, window)
        dx = self._safe_float(shift_raw[0], default=0.0)
        dy = self._safe_float(shift_raw[1], default=0.0)
        response = self._safe_float(response_raw, default=0.0)

        prev_x, prev_y = self._prev_center
        pred_x = self._safe_float(prev_x, default=cx) + dx
        pred_y = self._safe_float(prev_y, default=cy) + dy
        predicted_center = (pred_x, pred_y)

        if response < float(self.config.response_threshold):
            self._prev_patch = curr_patch.copy()
            self._prev_center = (cx, cy)
            return PhaseLockResult(
                center=(cx, cy),
                confidence=conf,
                refined=False,
                rejected=True,
                reason="low_response",
                shift=(dx, dy),
                response=response,
                predicted_center=predicted_center,
                patch_std=patch_std,
            )

        error = float(np.hypot(pred_x - cx, pred_y - cy))
        if error > max(0.0, float(self.config.max_shift_px)):
            self._prev_patch = curr_patch.copy()
            self._prev_center = (cx, cy)
            return PhaseLockResult(
                center=(cx, cy),
                confidence=conf,
                refined=False,
                rejected=True,
                reason="prediction_mismatch",
                shift=(dx, dy),
                response=response,
                predicted_center=predicted_center,
                patch_std=patch_std,
            )

        if conf < float(self.config.min_confidence):
            self._prev_patch = curr_patch.copy()
            self._prev_center = (cx, cy)
            return PhaseLockResult(
                center=(cx, cy),
                confidence=conf,
                refined=False,
                rejected=True,
                reason="low_confidence",
                shift=(dx, dy),
                response=response,
                predicted_center=predicted_center,
                patch_std=patch_std,
            )

        min_refine = max(0.0, float(self.config.min_refine_shift_px))
        if error <= min_refine:
            self._prev_patch = curr_patch.copy()
            self._prev_center = (cx, cy)
            return PhaseLockResult(
                center=(cx, cy),
                confidence=conf,
                refined=False,
                rejected=False,
                reason=None,
                shift=(dx, dy),
                response=response,
                predicted_center=predicted_center,
                patch_std=patch_std,
            )

        base_blend = float(np.clip(self.config.blend, 0.0, 1.0))
        conf_term = (1.0 - conf) * 0.4
        resp_term = (1.0 - np.clip(response, 0.0, 1.0)) * 0.25
        blend = float(np.clip(base_blend + conf_term + resp_term, 0.05, 0.85))

        rx = (1.0 - blend) * cx + blend * pred_x
        ry = (1.0 - blend) * cy + blend * pred_y
        refined_center = (float(rx), float(ry))

        self._prev_center = refined_center
        refined_patch = self._extract_patch(gray, refined_center)
        self._prev_patch = refined_patch.copy() if refined_patch is not None else curr_patch.copy()

        return PhaseLockResult(
            center=refined_center,
            confidence=conf,
            refined=True,
            rejected=False,
            reason=None,
            shift=(dx, dy),
            response=response,
            predicted_center=predicted_center,
            patch_std=patch_std,
        )
