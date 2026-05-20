"""
SpotZoom innovation frontier v36.

This module adds a drift-compensated hysteresis association gate inspired by:
- ByteTrack low-score recovery via staged association.
- BoT-SORT camera-motion compensation and adaptive gating.
- Trackpy drift prediction based on average motion.
- Deep SORT hit/miss streak management for robust state transitions.

The implementation intentionally depends only on numpy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class DriftCompensatedAssociationConfig:
    high_confidence_threshold: float = 0.6
    low_confidence_threshold: float = 0.15
    history_size: int = 24
    residual_history_size: int = 18
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.4
    velocity_smoothing: float = 0.68
    drift_smoothing: float = 0.84
    max_velocity_px_per_s: float = 620.0
    max_drift_px_per_s: float = 280.0
    base_gate_px: float = 3.6
    residual_gate_scale: float = 1.9
    drift_gate_scale: float = 1.2
    max_association_distance_px: float = 32.0
    min_refine_shift_px: float = 0.2
    max_refine_shift_px: float = 20.0
    low_confidence_blend: float = 0.72
    high_confidence_blend: float = 0.2
    bridge_gate_multiplier: float = 1.25
    confidence_gain: float = 1.02
    confidence_penalty: float = 0.95
    hit_counter_max: int = 24
    miss_decay: int = 1
    accept_hysteresis_hits: int = 2
    reject_hysteresis_misses: int = 2


@dataclass
class DriftCompensatedAssociationResult:
    center: Tuple[float, float]
    confidence: float
    accepted: bool
    refined: bool
    rejected: bool
    reason: Optional[str]
    predicted_center: Tuple[float, float]
    association_distance_px: float
    adaptive_gate_px: float
    drift_offset_px: float
    innovation_mad_px: float
    hit_counter: int
    used_bridge: bool
    reject_streak: int


class DriftCompensatedAssociationGate:
    """Single-target association gate with drift compensation and hysteresis."""

    def __init__(self, config: Optional[DriftCompensatedAssociationConfig] = None):
        self.config = config or DriftCompensatedAssociationConfig()
        self._history: Deque[Tuple[float, float, float, float]] = deque(
            maxlen=max(3, int(self.config.history_size))
        )
        self._residuals: Deque[float] = deque(
            maxlen=max(4, int(self.config.residual_history_size))
        )
        self._velocity = np.zeros(2, dtype=np.float64)
        self._drift_velocity = np.zeros(2, dtype=np.float64)
        self._residual_ema = 0.0
        self._clock = 0.0
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak = 0
        self._reject_streak = 0

    def reset(self) -> None:
        self._history.clear()
        self._residuals.clear()
        self._velocity[:] = 0.0
        self._drift_velocity[:] = 0.0
        self._residual_ema = 0.0
        self._clock = 0.0
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak = 0
        self._reject_streak = 0

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

    @staticmethod
    def _clip_vector_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
        max_norm = max(0.0, float(max_norm))
        if max_norm <= 0.0:
            return np.zeros_like(vec, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm <= max_norm:
            return vec
        return vec * (max_norm / max(norm, 1e-6))

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
        return self._clip_vector_norm(blended, float(self.config.max_velocity_px_per_s))

    def _update_residual(self, value: float) -> None:
        x = max(0.0, float(value))
        decay = 0.78
        self._residual_ema = decay * float(self._residual_ema) + (1.0 - decay) * x
        self._residuals.append(x)

    def _innovation_mad(self) -> float:
        if len(self._residuals) < 4:
            return 0.0
        vals = np.asarray(self._residuals, dtype=np.float64)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        return float(max(0.0, mad))

    def _update_drift_velocity(self, innovation: np.ndarray, dt: float, confidence: float) -> None:
        if dt <= 1e-6:
            return
        inst = innovation / dt
        weight = 0.25 + 0.75 * float(np.clip(confidence, 0.0, 1.0))
        inst *= weight
        alpha = float(np.clip(self.config.drift_smoothing, 0.0, 1.0))
        blended = alpha * self._drift_velocity + (1.0 - alpha) * inst
        self._drift_velocity = self._clip_vector_norm(
            blended,
            float(self.config.max_drift_px_per_s),
        )

    def _adaptive_gate(self, confidence: float, drift_offset_px: float) -> float:
        conf = float(np.clip(confidence, 0.0, 1.0))
        gate = (
            float(self.config.base_gate_px)
            + float(self.config.residual_gate_scale) * float(self._residual_ema)
            + float(self.config.drift_gate_scale) * float(max(0.0, drift_offset_px))
        )
        if conf < float(self.config.low_confidence_threshold):
            gate *= 1.12
        elif conf >= float(self.config.high_confidence_threshold):
            gate *= 0.95
        gate = float(
            np.clip(
                gate,
                float(self.config.base_gate_px),
                float(self.config.max_association_distance_px),
            )
        )
        return gate

    def _append_history(
        self,
        center: Tuple[float, float],
        timestamp_s: float,
        confidence: float,
    ) -> None:
        cx = float(center[0])
        cy = float(center[1])
        conf = float(np.clip(confidence, 0.0, 1.0))
        self._history.append((cx, cy, float(timestamp_s), max(conf, 0.02)))

    def _result(
        self,
        *,
        center: Tuple[float, float],
        confidence: float,
        accepted: bool,
        refined: bool,
        rejected: bool,
        reason: Optional[str],
        predicted_center: Tuple[float, float],
        association_distance_px: float,
        adaptive_gate_px: float,
        drift_offset_px: float,
        innovation_mad_px: float,
        used_bridge: bool,
    ) -> DriftCompensatedAssociationResult:
        return DriftCompensatedAssociationResult(
            center=(float(center[0]), float(center[1])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            accepted=bool(accepted),
            refined=bool(refined),
            rejected=bool(rejected),
            reason=reason,
            predicted_center=(float(predicted_center[0]), float(predicted_center[1])),
            association_distance_px=float(association_distance_px),
            adaptive_gate_px=float(adaptive_gate_px),
            drift_offset_px=float(max(0.0, drift_offset_px)),
            innovation_mad_px=float(max(0.0, innovation_mad_px)),
            hit_counter=int(self._hit_counter),
            used_bridge=bool(used_bridge),
            reject_streak=int(self._reject_streak),
        )

    def update(
        self,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> DriftCompensatedAssociationResult:
        conf = float(np.clip(self._safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = self._safe_float(center[0], default=np.nan)
        cy = self._safe_float(center[1], default=np.nan)
        ts = self._resolve_timestamp(timestamp_s)

        if not np.isfinite(cx) or not np.isfinite(cy):
            self._hit_counter = max(0, self._hit_counter - max(1, int(self.config.miss_decay)))
            self._reject_streak += 1
            self._accept_streak = 0
            return self._result(
                center=(0.0, 0.0),
                confidence=conf,
                accepted=False,
                refined=False,
                rejected=True,
                reason="invalid_center",
                predicted_center=(0.0, 0.0),
                association_distance_px=float("inf"),
                adaptive_gate_px=0.0,
                drift_offset_px=0.0,
                innovation_mad_px=0.0,
                used_bridge=False,
            )

        curr = np.asarray([cx, cy], dtype=np.float64)
        if not self._history:
            self._append_history((cx, cy), ts, conf)
            self._hit_counter = min(max(1, int(self.config.hit_counter_max)), self._hit_counter + 1)
            self._accept_streak = 1
            self._reject_streak = 0
            return self._result(
                center=(cx, cy),
                confidence=conf,
                accepted=True,
                refined=False,
                rejected=False,
                reason=None,
                predicted_center=(cx, cy),
                association_distance_px=0.0,
                adaptive_gate_px=float(self.config.max_association_distance_px),
                drift_offset_px=0.0,
                innovation_mad_px=0.0,
                used_bridge=False,
            )

        prev_x, prev_y, prev_t, _prev_conf = self._history[-1]
        dt = float(np.clip(ts - prev_t, float(self.config.min_dt_s), float(self.config.max_dt_s)))
        self._velocity = self._estimate_velocity()
        predicted = np.asarray([prev_x, prev_y], dtype=np.float64) + (
            self._velocity + self._drift_velocity
        ) * dt

        innovation = curr - predicted
        distance = float(np.linalg.norm(innovation))
        clipped_dist = min(distance, float(self.config.max_association_distance_px))
        self._update_residual(clipped_dist)
        self._update_drift_velocity(innovation, dt, conf)
        drift_offset_px = float(np.linalg.norm(self._drift_velocity * dt))
        innovation_mad = self._innovation_mad()
        gate = self._adaptive_gate(conf, drift_offset_px)

        high_thr = float(np.clip(self.config.high_confidence_threshold, 0.0, 1.0))
        low_thr = float(np.clip(self.config.low_confidence_threshold, 0.0, 1.0))
        if high_thr < low_thr:
            high_thr = low_thr

        if distance <= gate:
            if conf < low_thr:
                blend = min(0.96, float(self.config.low_confidence_blend) + 0.12)
            elif conf < high_thr:
                blend = float(self.config.low_confidence_blend)
            else:
                blend = float(self.config.high_confidence_blend)
            blend = float(np.clip(blend, 0.0, 0.96))

            refined_vec = (1.0 - blend) * curr + blend * predicted
            shift_px = float(np.linalg.norm(refined_vec - curr))
            max_shift = max(0.0, float(self.config.max_refine_shift_px))
            if shift_px > max_shift:
                self._hit_counter = max(0, self._hit_counter - max(1, int(self.config.miss_decay)))
                self._reject_streak += 1
                self._accept_streak = 0
                out_conf = conf * float(self.config.confidence_penalty)
                self._append_history((float(predicted[0]), float(predicted[1])), ts, out_conf)
                return self._result(
                    center=(cx, cy),
                    confidence=out_conf,
                    accepted=False,
                    refined=False,
                    rejected=True,
                    reason="refine_shift_too_large",
                    predicted_center=(float(predicted[0]), float(predicted[1])),
                    association_distance_px=distance,
                    adaptive_gate_px=gate,
                    drift_offset_px=drift_offset_px,
                    innovation_mad_px=innovation_mad,
                    used_bridge=False,
                )

            refined_flag = shift_px >= float(self.config.min_refine_shift_px)
            out = refined_vec if refined_flag else curr
            out_conf = conf * float(self.config.confidence_gain)
            self._accept_streak += 1
            self._reject_streak = 0
            if self._accept_streak < max(1, int(self.config.accept_hysteresis_hits)):
                out_conf *= 0.995
            self._hit_counter = min(max(1, int(self.config.hit_counter_max)), self._hit_counter + 1)
            self._append_history((float(out[0]), float(out[1])), ts, out_conf)
            return self._result(
                center=(float(out[0]), float(out[1])),
                confidence=out_conf,
                accepted=True,
                refined=refined_flag,
                rejected=False,
                reason=None,
                predicted_center=(float(predicted[0]), float(predicted[1])),
                association_distance_px=distance,
                adaptive_gate_px=gate,
                drift_offset_px=drift_offset_px,
                innovation_mad_px=innovation_mad,
                used_bridge=False,
            )

        self._reject_streak += 1
        self._accept_streak = 0
        bridge_gate = min(
            float(self.config.max_association_distance_px),
            gate * max(1.0, float(self.config.bridge_gate_multiplier)),
        )
        can_bridge = (
            self._reject_streak <= max(1, int(self.config.reject_hysteresis_misses))
            and self._hit_counter > 0
            and distance <= bridge_gate
        )
        if can_bridge:
            blend = float(np.clip(float(self.config.low_confidence_blend) + 0.2, 0.0, 0.96))
            bridged = (1.0 - blend) * curr + blend * predicted
            shift_px = float(np.linalg.norm(bridged - curr))
            if shift_px <= max(0.0, float(self.config.max_refine_shift_px)):
                out_conf = conf * float(self.config.confidence_penalty)
                self._append_history((float(bridged[0]), float(bridged[1])), ts, out_conf)
                self._hit_counter = min(max(1, int(self.config.hit_counter_max)), self._hit_counter + 1)
                return self._result(
                    center=(float(bridged[0]), float(bridged[1])),
                    confidence=out_conf,
                    accepted=True,
                    refined=shift_px >= float(self.config.min_refine_shift_px),
                    rejected=False,
                    reason="hysteresis_bridge",
                    predicted_center=(float(predicted[0]), float(predicted[1])),
                    association_distance_px=distance,
                    adaptive_gate_px=gate,
                    drift_offset_px=drift_offset_px,
                    innovation_mad_px=innovation_mad,
                    used_bridge=True,
                )

        self._hit_counter = max(0, self._hit_counter - max(1, int(self.config.miss_decay)))
        out_conf = conf * float(self.config.confidence_penalty)
        self._append_history((float(predicted[0]), float(predicted[1])), ts, out_conf)
        return self._result(
            center=(cx, cy),
            confidence=out_conf,
            accepted=False,
            refined=False,
            rejected=True,
            reason="association_distance_too_large",
            predicted_center=(float(predicted[0]), float(predicted[1])),
            association_distance_px=distance,
            adaptive_gate_px=gate,
            drift_offset_px=drift_offset_px,
            innovation_mad_px=innovation_mad,
            used_bridge=False,
        )
