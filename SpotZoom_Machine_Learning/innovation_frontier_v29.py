"""
SpotZoom innovation frontier v29.

This module provides a lightweight temporal reliability filter inspired by:
- CoTracker3 (long-term temporal consistency)
- ByteTrack (confidence-aware association)
- OpenWFS (stability-first control guardrails)
- DeepTrack2 (robustness against noisy scientific imaging)

The implementation intentionally depends only on numpy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class TemporalReliabilityConfig:
    history_size: int = 12
    min_confidence: float = 0.08
    velocity_smoothing: float = 0.62
    max_prediction_error_px: float = 18.0
    max_velocity_px_per_s: float = 520.0
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.35
    min_refine_shift_px: float = 0.35


@dataclass
class TemporalReliabilityResult:
    center: Tuple[float, float]
    confidence: float
    refined: bool
    rejected: bool
    reason: Optional[str]
    predicted_center: Tuple[float, float]
    prediction_error_px: float
    estimated_speed_px_s: float
    history_count: int


class TemporalReliabilityFilter:
    def __init__(self, config: Optional[TemporalReliabilityConfig] = None):
        self.config = config or TemporalReliabilityConfig()
        size = max(3, int(self.config.history_size))
        self._history: Deque[Tuple[float, float, float, float]] = deque(maxlen=size)
        self._velocity = np.zeros(2, dtype=np.float64)
        self._clock = 0.0

    def reset(self) -> None:
        self._history.clear()
        self._velocity[:] = 0.0
        self._clock = 0.0

    @staticmethod
    def _safe_float(value: float, default: float = 0.0) -> float:
        try:
            x = float(value)
        except Exception:
            return default
        if not np.isfinite(x):
            return default
        return x

    def _resolve_timestamp(self, timestamp_s: Optional[float]) -> float:
        if timestamp_s is None:
            self._clock += 1.0 / 30.0
            return float(self._clock)
        ts = self._safe_float(timestamp_s, default=self._clock + 1.0 / 30.0)
        if ts <= self._clock:
            ts = self._clock + 1.0 / 30.0
        self._clock = float(ts)
        return float(ts)

    def _estimate_velocity(self) -> np.ndarray:
        if len(self._history) < 2:
            return self._velocity.copy()
        arr = np.asarray(self._history, dtype=np.float64)
        pts = arr[:, :2]
        ts = arr[:, 2]
        dt = np.diff(ts)
        disp = np.diff(pts, axis=0)
        valid = dt > max(1e-6, float(self.config.min_dt_s) * 0.5)
        if not np.any(valid):
            return self._velocity.copy()
        inst_vel = disp[valid] / dt[valid, None]
        robust = np.median(inst_vel, axis=0)
        alpha = float(np.clip(self.config.velocity_smoothing, 0.0, 1.0))
        blended = alpha * self._velocity + (1.0 - alpha) * robust
        vmax = max(1.0, float(self.config.max_velocity_px_per_s))
        speed = float(np.linalg.norm(blended))
        if speed > vmax:
            blended = blended * (vmax / max(speed, 1e-6))
        return blended

    def update(
        self,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> TemporalReliabilityResult:
        cx = self._safe_float(center[0], default=np.nan)
        cy = self._safe_float(center[1], default=np.nan)
        conf = float(np.clip(self._safe_float(confidence, default=0.0), 0.0, 1.0))
        ts = self._resolve_timestamp(timestamp_s)

        if not np.isfinite(cx) or not np.isfinite(cy):
            return TemporalReliabilityResult(
                center=(0.0, 0.0),
                confidence=conf,
                refined=False,
                rejected=True,
                reason="invalid_center",
                predicted_center=(0.0, 0.0),
                prediction_error_px=float("inf"),
                estimated_speed_px_s=float(np.linalg.norm(self._velocity)),
                history_count=len(self._history),
            )

        curr = np.asarray([cx, cy], dtype=np.float64)
        if not self._history:
            self._history.append((cx, cy, ts, max(conf, 0.05)))
            return TemporalReliabilityResult(
                center=(cx, cy),
                confidence=conf,
                refined=False,
                rejected=False,
                reason=None,
                predicted_center=(cx, cy),
                prediction_error_px=0.0,
                estimated_speed_px_s=0.0,
                history_count=len(self._history),
            )

        prev_x, prev_y, prev_t, _prev_conf = self._history[-1]
        dt = float(np.clip(ts - prev_t, self.config.min_dt_s, self.config.max_dt_s))
        self._velocity = self._estimate_velocity()
        predicted = np.asarray([prev_x, prev_y], dtype=np.float64) + self._velocity * dt
        error = float(np.linalg.norm(curr - predicted))

        conf_floor = float(np.clip(self.config.min_confidence, 0.0, 1.0))
        adaptive_gate = max(
            1.0,
            float(self.config.max_prediction_error_px) * (0.9 + 0.7 * max(conf, conf_floor)),
        )
        speed = float(np.linalg.norm(self._velocity))
        if error > adaptive_gate and conf <= max(conf_floor, 0.25):
            self._history.append((cx, cy, ts, max(conf * 0.5, 0.05)))
            return TemporalReliabilityResult(
                center=(cx, cy),
                confidence=conf,
                refined=False,
                rejected=True,
                reason="prediction_gate_reject",
                predicted_center=(float(predicted[0]), float(predicted[1])),
                prediction_error_px=error,
                estimated_speed_px_s=speed,
                history_count=len(self._history),
            )

        blend_from_pred = float(np.clip(0.58 - 0.42 * conf, 0.1, 0.58))
        if error <= float(self.config.min_refine_shift_px):
            refined = curr
            refined_flag = False
        else:
            refined = (1.0 - blend_from_pred) * curr + blend_from_pred * predicted
            refined_flag = bool(np.linalg.norm(refined - curr) > 1e-9)

        rx = float(refined[0])
        ry = float(refined[1])
        self._history.append((rx, ry, ts, max(conf, 0.05)))
        return TemporalReliabilityResult(
            center=(rx, ry),
            confidence=conf,
            refined=refined_flag,
            rejected=False,
            reason=None,
            predicted_center=(float(predicted[0]), float(predicted[1])),
            prediction_error_px=error,
            estimated_speed_px_s=speed,
            history_count=len(self._history),
        )
