"""
SpotZoom innovation frontier v32.

This module adds a lightweight trajectory-coherence gate inspired by:
- CoTracker3 long-range point tracking consistency
- OpenCV Lucas-Kanade sparse optical flow with forward-backward checks
- Scientific imaging pipelines that favor conservative confidence-aware blending

The implementation intentionally depends only on numpy + optional OpenCV.
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
class TrajectoryCoherenceConfig:
    lk_window_size: int = 21
    lk_max_level: int = 2
    lk_max_iters: int = 16
    lk_eps: float = 0.03
    max_fb_error_px: float = 1.8
    max_track_error: float = 20.0
    max_innovation_px: float = 14.0
    max_refine_shift_px: float = 16.0
    min_refine_shift_px: float = 0.25
    min_confidence: float = 0.06
    blend_base: float = 0.38
    blend_min: float = 0.12
    blend_max: float = 0.62
    confidence_gain: float = 1.02


@dataclass
class TrajectoryCoherenceResult:
    center: Tuple[float, float]
    confidence: float
    accepted: bool
    refined: bool
    rejected: bool
    reason: Optional[str]
    predicted_center: Tuple[float, float]
    fb_error_px: float
    track_error: float
    innovation_px: float
    blend_weight: float


class TrajectoryCoherenceGate:
    """Track-consistency gate using forward-backward Lucas-Kanade flow."""

    def __init__(self, config: Optional[TrajectoryCoherenceConfig] = None):
        self.config = config or TrajectoryCoherenceConfig()
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_center: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_center = None

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
    def _as_gray_uint8(frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.size == 0:
            return None
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

        out = np.asarray(gray)
        if out.dtype != np.uint8:
            mn = float(np.min(out))
            mx = float(np.max(out))
            if mx <= mn:
                out = np.zeros_like(out, dtype=np.uint8)
            else:
                out = np.clip((out - mn) * (255.0 / (mx - mn)), 0.0, 255.0).astype(np.uint8)
        return out

    def _reject(
        self,
        center: Tuple[float, float],
        confidence: float,
        reason: str,
        predicted_center: Optional[Tuple[float, float]] = None,
        fb_error_px: float = 0.0,
        track_error: float = 0.0,
        innovation_px: float = 0.0,
        blend_weight: float = 0.0,
    ) -> TrajectoryCoherenceResult:
        pred = predicted_center if predicted_center is not None else center
        return TrajectoryCoherenceResult(
            center=(float(center[0]), float(center[1])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            accepted=False,
            refined=False,
            rejected=True,
            reason=reason,
            predicted_center=(float(pred[0]), float(pred[1])),
            fb_error_px=float(fb_error_px),
            track_error=float(track_error),
            innovation_px=float(innovation_px),
            blend_weight=float(blend_weight),
        )

    def update(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
    ) -> TrajectoryCoherenceResult:
        conf = float(np.clip(self._safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = self._safe_float(center[0], default=np.nan)
        cy = self._safe_float(center[1], default=np.nan)
        if not np.isfinite(cx) or not np.isfinite(cy):
            return self._reject((0.0, 0.0), conf, reason="invalid_center")

        gray = self._as_gray_uint8(frame)
        if gray is None:
            return self._reject((cx, cy), conf, reason="invalid_frame")
        if cv2 is None:
            return self._reject((cx, cy), conf, reason="opencv_unavailable")

        measured = np.asarray([cx, cy], dtype=np.float32)
        if self._prev_gray is None or self._prev_center is None:
            self._prev_gray = gray.copy()
            self._prev_center = (cx, cy)
            return TrajectoryCoherenceResult(
                center=(cx, cy),
                confidence=conf,
                accepted=True,
                refined=False,
                rejected=False,
                reason=None,
                predicted_center=(cx, cy),
                fb_error_px=0.0,
                track_error=0.0,
                innovation_px=0.0,
                blend_weight=0.0,
            )

        prev_pt = np.asarray([[self._prev_center]], dtype=np.float32)
        win_size = max(5, int(self.config.lk_window_size))
        if win_size % 2 == 0:
            win_size += 1
        lk_kwargs = {
            "winSize": (win_size, win_size),
            "maxLevel": max(0, int(self.config.lk_max_level)),
            "criteria": (
                int(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT),
                max(1, int(self.config.lk_max_iters)),
                max(1e-6, float(self.config.lk_eps)),
            ),
            "flags": 0,
            "minEigThreshold": 1e-4,
        }

        next_pt, st_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev_pt, None, **lk_kwargs)
        if next_pt is None or st_fwd is None or int(st_fwd.ravel()[0]) == 0:
            self._prev_gray = gray.copy()
            self._prev_center = (cx, cy)
            return self._reject((cx, cy), conf, reason="flow_forward_failed")

        pred = np.asarray(next_pt.reshape(-1, 2)[0], dtype=np.float32)
        track_error = float(err_fwd.ravel()[0]) if err_fwd is not None and err_fwd.size else 0.0

        back_pt, st_bwd, _err_bwd = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, next_pt, None, **lk_kwargs)
        if back_pt is None or st_bwd is None or int(st_bwd.ravel()[0]) == 0:
            self._prev_gray = gray.copy()
            self._prev_center = (cx, cy)
            return self._reject(
                (cx, cy),
                conf,
                reason="flow_backward_failed",
                predicted_center=(float(pred[0]), float(pred[1])),
                track_error=track_error,
            )

        prev_back = np.asarray(back_pt.reshape(-1, 2)[0], dtype=np.float32)
        prev_origin = prev_pt.reshape(-1, 2)[0]
        fb_error = float(np.linalg.norm(prev_origin - prev_back))
        innovation = float(np.linalg.norm(measured - pred))

        if track_error > float(self.config.max_track_error):
            self._prev_gray = gray.copy()
            self._prev_center = (cx, cy)
            return self._reject(
                (cx, cy),
                conf,
                reason="track_error_too_large",
                predicted_center=(float(pred[0]), float(pred[1])),
                fb_error_px=fb_error,
                track_error=track_error,
                innovation_px=innovation,
            )

        if fb_error > float(self.config.max_fb_error_px):
            self._prev_gray = gray.copy()
            self._prev_center = (cx, cy)
            return self._reject(
                (cx, cy),
                conf,
                reason="fb_error_too_large",
                predicted_center=(float(pred[0]), float(pred[1])),
                fb_error_px=fb_error,
                track_error=track_error,
                innovation_px=innovation,
            )

        if conf < float(self.config.min_confidence) and innovation > float(self.config.max_innovation_px):
            self._prev_gray = gray.copy()
            self._prev_center = (cx, cy)
            return self._reject(
                (cx, cy),
                conf,
                reason="low_confidence_high_innovation",
                predicted_center=(float(pred[0]), float(pred[1])),
                fb_error_px=fb_error,
                track_error=track_error,
                innovation_px=innovation,
            )

        coherence = float(np.clip(1.0 - fb_error / max(float(self.config.max_fb_error_px), 1e-6), 0.0, 1.0))
        blend = float(self.config.blend_base * coherence * (1.0 - 0.55 * conf))
        blend = float(np.clip(blend, float(self.config.blend_min), float(self.config.blend_max)))
        refined = (1.0 - blend) * measured + blend * pred

        shift_px = float(np.linalg.norm(refined - measured))
        if shift_px > float(self.config.max_refine_shift_px):
            self._prev_gray = gray.copy()
            self._prev_center = (cx, cy)
            return self._reject(
                (cx, cy),
                conf,
                reason="refine_shift_too_large",
                predicted_center=(float(pred[0]), float(pred[1])),
                fb_error_px=fb_error,
                track_error=track_error,
                innovation_px=innovation,
                blend_weight=blend,
            )

        if shift_px < float(self.config.min_refine_shift_px):
            refined = measured
            refined_flag = False
            blend = 0.0
        else:
            refined_flag = True

        out_center = (float(refined[0]), float(refined[1]))
        out_conf = float(np.clip(conf * float(self.config.confidence_gain), 0.01, 0.99))

        self._prev_gray = gray.copy()
        self._prev_center = out_center

        return TrajectoryCoherenceResult(
            center=out_center,
            confidence=out_conf,
            accepted=True,
            refined=refined_flag,
            rejected=False,
            reason=None,
            predicted_center=(float(pred[0]), float(pred[1])),
            fb_error_px=fb_error,
            track_error=track_error,
            innovation_px=innovation,
            blend_weight=blend,
        )
