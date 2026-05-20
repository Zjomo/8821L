"""
SpotZoom innovation frontier v31.

This module introduces a lightweight beam-profile quality gate inspired by:
- laserbeamsize (ISO 11146 second-moment beam metrics)
- Open Beam Profiler (D4Sigma + beam-shape diagnostics)

The implementation is intentionally dependency-light (numpy + optional OpenCV).
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
class BeamProfileGateConfig:
    patch_radius: int = 26
    d4sigma_min_px: float = 2.0
    d4sigma_max_px: float = 96.0
    max_axis_ratio: float = 3.8
    min_signal_energy: float = 1200.0
    min_contrast_std: float = 1.5
    max_refine_shift_px: float = 18.0
    min_confidence: float = 0.04
    center_blend: float = 0.22
    confidence_gain: float = 1.02
    background_percentile: float = 20.0


@dataclass
class BeamProfileGateResult:
    center: Tuple[float, float]
    confidence: float
    accepted: bool
    refined: bool
    reason: Optional[str]
    d4sigma_x: float
    d4sigma_y: float
    axis_ratio: float
    signal_energy: float
    contrast_std: float


class BeamProfileGate:
    """Second-moment quality gate for robust center tracking."""

    def __init__(self, config: Optional[BeamProfileGateConfig] = None):
        self.config = config or BeamProfileGateConfig()
        self._last_good_center: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._last_good_center = None

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
        out = np.asarray(gray, dtype=np.float32)
        if out.size == 0:
            return None
        return out

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
            src = np.pad(gray, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
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

    def _profile_metrics(
        self,
        patch: np.ndarray,
    ) -> Tuple[bool, float, float, float, float, float, float, float]:
        bg_pct = float(np.clip(self.config.background_percentile, 0.0, 80.0))
        baseline = float(np.percentile(patch, bg_pct))
        signal = patch - baseline
        signal = np.clip(signal, 0.0, None)

        signal_energy = float(np.sum(signal))
        contrast_std = float(np.std(signal))
        if signal_energy <= 1e-6:
            return False, 0.0, 0.0, 1.0, signal_energy, contrast_std, 0.0, 0.0

        h, w = signal.shape[:2]
        ys, xs = np.indices((h, w), dtype=np.float64)
        weight = signal.astype(np.float64)
        mass = float(np.sum(weight))
        if mass <= 1e-9:
            return False, 0.0, 0.0, 1.0, signal_energy, contrast_std, 0.0, 0.0

        cx = float(np.sum(xs * weight) / mass)
        cy = float(np.sum(ys * weight) / mass)
        dx = xs - cx
        dy = ys - cy
        var_x = float(np.sum((dx * dx) * weight) / mass)
        var_y = float(np.sum((dy * dy) * weight) / mass)
        cov_xy = float(np.sum((dx * dy) * weight) / mass)

        cov = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float64)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 0.0, None)

        sigma_minor = float(np.sqrt(max(eigvals[0], 0.0)))
        sigma_major = float(np.sqrt(max(eigvals[1], 0.0)))
        d4_minor = 4.0 * sigma_minor
        d4_major = 4.0 * sigma_major
        axis_ratio = float(d4_major / max(d4_minor, 1e-6))
        return True, d4_major, d4_minor, axis_ratio, signal_energy, contrast_std, cx, cy

    def update(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
    ) -> BeamProfileGateResult:
        conf = float(np.clip(self._safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = self._safe_float(center[0], default=np.nan)
        cy = self._safe_float(center[1], default=np.nan)
        if not np.isfinite(cx) or not np.isfinite(cy):
            return BeamProfileGateResult(
                center=(0.0, 0.0),
                confidence=conf,
                accepted=False,
                refined=False,
                reason="invalid_center",
                d4sigma_x=0.0,
                d4sigma_y=0.0,
                axis_ratio=1.0,
                signal_energy=0.0,
                contrast_std=0.0,
            )

        gray = self._as_gray_float32(frame)
        if gray is None:
            return BeamProfileGateResult(
                center=(cx, cy),
                confidence=conf,
                accepted=False,
                refined=False,
                reason="invalid_frame",
                d4sigma_x=0.0,
                d4sigma_y=0.0,
                axis_ratio=1.0,
                signal_energy=0.0,
                contrast_std=0.0,
            )

        patch = self._extract_patch(gray, (cx, cy))
        if patch is None:
            return BeamProfileGateResult(
                center=(cx, cy),
                confidence=conf,
                accepted=False,
                refined=False,
                reason="invalid_patch",
                d4sigma_x=0.0,
                d4sigma_y=0.0,
                axis_ratio=1.0,
                signal_energy=0.0,
                contrast_std=0.0,
            )

        ok, d4_major, d4_minor, axis_ratio, signal_energy, contrast_std, pcx, pcy = self._profile_metrics(patch)
        if not ok:
            return BeamProfileGateResult(
                center=(cx, cy),
                confidence=conf,
                accepted=False,
                refined=False,
                reason="insufficient_signal",
                d4sigma_x=d4_major,
                d4sigma_y=d4_minor,
                axis_ratio=axis_ratio,
                signal_energy=signal_energy,
                contrast_std=contrast_std,
            )

        if conf < float(self.config.min_confidence):
            return BeamProfileGateResult(
                center=(cx, cy),
                confidence=conf,
                accepted=False,
                refined=False,
                reason="low_confidence",
                d4sigma_x=d4_major,
                d4sigma_y=d4_minor,
                axis_ratio=axis_ratio,
                signal_energy=signal_energy,
                contrast_std=contrast_std,
            )

        if signal_energy < float(self.config.min_signal_energy):
            return BeamProfileGateResult(
                center=(cx, cy),
                confidence=conf,
                accepted=False,
                refined=False,
                reason="low_signal_energy",
                d4sigma_x=d4_major,
                d4sigma_y=d4_minor,
                axis_ratio=axis_ratio,
                signal_energy=signal_energy,
                contrast_std=contrast_std,
            )

        if contrast_std < float(self.config.min_contrast_std):
            return BeamProfileGateResult(
                center=(cx, cy),
                confidence=conf,
                accepted=False,
                refined=False,
                reason="low_contrast",
                d4sigma_x=d4_major,
                d4sigma_y=d4_minor,
                axis_ratio=axis_ratio,
                signal_energy=signal_energy,
                contrast_std=contrast_std,
            )

        if d4_major < float(self.config.d4sigma_min_px) or d4_major > float(self.config.d4sigma_max_px):
            return BeamProfileGateResult(
                center=(cx, cy),
                confidence=conf,
                accepted=False,
                refined=False,
                reason="d4sigma_out_of_range",
                d4sigma_x=d4_major,
                d4sigma_y=d4_minor,
                axis_ratio=axis_ratio,
                signal_energy=signal_energy,
                contrast_std=contrast_std,
            )

        if axis_ratio > float(self.config.max_axis_ratio):
            return BeamProfileGateResult(
                center=(cx, cy),
                confidence=conf,
                accepted=False,
                refined=False,
                reason="axis_ratio_out_of_range",
                d4sigma_x=d4_major,
                d4sigma_y=d4_minor,
                axis_ratio=axis_ratio,
                signal_energy=signal_energy,
                contrast_std=contrast_std,
            )

        radius = max(2, int(self.config.patch_radius))
        meas_x = float(cx) + (float(pcx) - float(radius))
        meas_y = float(cy) + (float(pcy) - float(radius))
        shift = float(np.hypot(meas_x - cx, meas_y - cy))
        if shift > float(self.config.max_refine_shift_px):
            return BeamProfileGateResult(
                center=(cx, cy),
                confidence=conf,
                accepted=False,
                refined=False,
                reason="refine_shift_too_large",
                d4sigma_x=d4_major,
                d4sigma_y=d4_minor,
                axis_ratio=axis_ratio,
                signal_energy=signal_energy,
                contrast_std=contrast_std,
            )

        blend = float(np.clip(self.config.center_blend, 0.0, 1.0))
        refined_x = (1.0 - blend) * cx + blend * meas_x
        refined_y = (1.0 - blend) * cy + blend * meas_y
        refined = bool(shift > 1e-3)

        out_conf = float(np.clip(conf * float(self.config.confidence_gain), 0.01, 0.99))
        out_center = (float(refined_x), float(refined_y))
        self._last_good_center = out_center

        return BeamProfileGateResult(
            center=out_center,
            confidence=out_conf,
            accepted=True,
            refined=refined,
            reason=None,
            d4sigma_x=d4_major,
            d4sigma_y=d4_minor,
            axis_ratio=axis_ratio,
            signal_energy=signal_energy,
            contrast_std=contrast_std,
        )
