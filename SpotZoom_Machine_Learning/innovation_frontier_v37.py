"""
SpotZoom innovation frontier v37.

This module adds a trajectory-consensus association gate inspired by:
- ByteTrack staged high/low-confidence association.
- BoT-SORT global-motion-aware adaptive gating.
- OC-SORT observation-centric trajectory consistency.
- CoTracker long-range point-track coherence.
- Norfair hit-counter-driven lifecycle stabilization.

The implementation intentionally depends only on numpy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class TrajectoryConsensusAssociationConfig:
    high_confidence_threshold: float = 0.62
    low_confidence_threshold: float = 0.15
    history_size: int = 28
    residual_history_size: int = 20
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45
    velocity_smoothing: float = 0.7
    acceleration_smoothing: float = 0.82
    drift_smoothing: float = 0.86
    max_velocity_px_per_s: float = 680.0
    max_acceleration_px_per_s2: float = 2400.0
    max_drift_px_per_s: float = 300.0
    base_gate_px: float = 3.5
    residual_gate_scale: float = 1.8
    drift_gate_scale: float = 1.3
    coherence_gate_scale: float = 1.1
    max_association_distance_px: float = 34.0
    min_refine_shift_px: float = 0.2
    max_refine_shift_px: float = 22.0
    low_confidence_blend: float = 0.74
    high_confidence_blend: float = 0.18
    bridge_gate_multiplier: float = 1.3
    confidence_gain: float = 1.03
    confidence_penalty: float = 0.93
    coherence_floor: float = -0.35
    coherence_reject_margin: float = -0.75
    temperature_decay: float = 0.82
    hit_counter_max: int = 28
    miss_decay: int = 1
    accept_hysteresis_hits: int = 2
    reject_hysteresis_misses: int = 3


@dataclass
class TrajectoryConsensusAssociationResult:
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
    coherence: float
    temperature: float
    hit_counter: int
    used_bridge: bool
    reject_streak: int


class TrajectoryConsensusAssociationGate:
    """Single-target association gate with velocity/acceleration consensus and coherence-aware gating."""

    def __init__(self, config: Optional[TrajectoryConsensusAssociationConfig] = None):
        self.config = config or TrajectoryConsensusAssociationConfig()
        self._history: Deque[Tuple[float, float, float, float]] = deque(
            maxlen=max(4, int(self.config.history_size))
        )
        self._residuals: Deque[float] = deque(
            maxlen=max(4, int(self.config.residual_history_size))
        )
        self._velocity = np.zeros(2, dtype=np.float64)
        self._acceleration = np.zeros(2, dtype=np.float64)
        self._drift_velocity = np.zeros(2, dtype=np.float64)
        self._residual_ema = 0.0
        self._temperature = 1.0
        self._clock = 0.0
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak = 0
        self._reject_streak = 0

    def reset(self) -> None:
        self._history.clear()
        self._residuals.clear()
        self._velocity[:] = 0.0
        self._acceleration[:] = 0.0
        self._drift_velocity[:] = 0.0
        self._residual_ema = 0.0
        self._temperature = 1.0
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

    @staticmethod
    def _clip_vector_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
        max_norm = max(0.0, float(max_norm))
        if max_norm <= 0.0:
            return np.zeros_like(vec, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm <= max_norm:
            return vec
        return vec * (max_norm / max(norm, 1e-6))

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
        return self._clip_vector_norm(blended, float(self.config.max_velocity_px_per_s))

    def _estimate_acceleration(self) -> np.ndarray:
        if len(self._history) < 3:
            return self._acceleration.copy()
        arr = np.asarray(self._history, dtype=np.float64)
        pts = arr[:, :2]
        ts = arr[:, 2]
        dt = np.diff(ts)
        disp = np.diff(pts, axis=0)
        valid_v = dt > max(1e-6, float(self.config.min_dt_s) * 0.5)
        if np.count_nonzero(valid_v) < 2:
            return self._acceleration.copy()
        vel = disp[valid_v] / dt[valid_v, None]
        dtv = dt[valid_v]
        if vel.shape[0] < 2:
            return self._acceleration.copy()
        dvel = np.diff(vel, axis=0)
        dt2 = dtv[1:]
        valid_a = dt2 > max(1e-6, float(self.config.min_dt_s) * 0.5)
        if not np.any(valid_a):
            return self._acceleration.copy()
        inst_acc = dvel[valid_a] / dt2[valid_a, None]
        robust = np.median(inst_acc, axis=0)
        alpha = float(np.clip(self.config.acceleration_smoothing, 0.0, 1.0))
        blended = alpha * self._acceleration + (1.0 - alpha) * robust
        return self._clip_vector_norm(blended, float(self.config.max_acceleration_px_per_s2))

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

    def _update_temperature(self, innovation_mad_px: float, distance_px: float, confidence: float) -> float:
        conf = float(np.clip(confidence, 0.0, 1.0))
        decay = float(np.clip(self.config.temperature_decay, 0.0, 1.0))
        motion_term = 0.05 * float(max(0.0, innovation_mad_px)) + 0.01 * float(max(0.0, distance_px))
        conf_term = 0.12 * (1.0 - conf)
        rej_term = 0.06 * float(max(0, self._reject_streak))
        raw = decay * float(self._temperature) + (1.0 - decay) * (1.0 + motion_term + conf_term + rej_term)
        self._temperature = float(np.clip(raw, 0.75, 2.8))
        return self._temperature

    @staticmethod
    def _direction_coherence(innovation: np.ndarray, velocity: np.ndarray) -> float:
        inorm = float(np.linalg.norm(innovation))
        vnorm = float(np.linalg.norm(velocity))
        if inorm <= 1e-6 or vnorm <= 1e-6:
            return 0.0
        value = float(np.dot(innovation, velocity) / max(1e-6, inorm * vnorm))
        return float(np.clip(value, -1.0, 1.0))

    def _update_drift_velocity(self, innovation: np.ndarray, dt: float, confidence: float) -> None:
        if dt <= 1e-6:
            return
        inst = innovation / dt
        weight = 0.22 + 0.78 * float(np.clip(confidence, 0.0, 1.0))
        inst *= weight
        alpha = float(np.clip(self.config.drift_smoothing, 0.0, 1.0))
        blended = alpha * self._drift_velocity + (1.0 - alpha) * inst
        self._drift_velocity = self._clip_vector_norm(blended, float(self.config.max_drift_px_per_s))

    def _adaptive_gate(self, confidence: float, drift_offset_px: float, coherence: float, temperature: float) -> float:
        conf = float(np.clip(confidence, 0.0, 1.0))
        gate = (
            float(self.config.base_gate_px)
            + float(self.config.residual_gate_scale) * float(self._residual_ema)
            + float(self.config.drift_gate_scale) * float(max(0.0, drift_offset_px))
        )
        if coherence < float(self.config.coherence_floor):
            gate *= 1.0 + float(self.config.coherence_gate_scale) * min(1.0, abs(coherence)) * 0.25
        if conf < float(self.config.low_confidence_threshold):
            gate *= 1.12
        elif conf >= float(self.config.high_confidence_threshold):
            gate *= 0.94
        gate *= float(np.clip(temperature, 0.8, 2.4))
        gate = float(
            np.clip(
                gate,
                float(self.config.base_gate_px),
                float(self.config.max_association_distance_px),
            )
        )
        return gate

    def _append_history(self, center: Tuple[float, float], timestamp_s: float, confidence: float) -> None:
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
        coherence: float,
        temperature: float,
        used_bridge: bool,
    ) -> TrajectoryConsensusAssociationResult:
        return TrajectoryConsensusAssociationResult(
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
            coherence=float(np.clip(coherence, -1.0, 1.0)),
            temperature=float(max(0.0, temperature)),
            hit_counter=int(self._hit_counter),
            used_bridge=bool(used_bridge),
            reject_streak=int(self._reject_streak),
        )

    def update(
        self,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> TrajectoryConsensusAssociationResult:
        conf = float(np.clip(self._safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = self._safe_float(center[0], default=np.nan)
        cy = self._safe_float(center[1], default=np.nan)
        ts = self._resolve_timestamp(timestamp_s)

        if not np.isfinite(cx) or not np.isfinite(cy):
            self._hit_counter = max(0, self._hit_counter - max(1, int(self.config.miss_decay)))
            self._reject_streak += 1
            self._accept_streak = 0
            self._temperature = min(2.8, self._temperature * 1.08)
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
                coherence=0.0,
                temperature=self._temperature,
                used_bridge=False,
            )

        curr = np.asarray([cx, cy], dtype=np.float64)
        if not self._history:
            self._append_history((cx, cy), ts, conf)
            self._hit_counter = min(max(1, int(self.config.hit_counter_max)), self._hit_counter + 1)
            self._accept_streak = 1
            self._reject_streak = 0
            self._temperature = 1.0
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
                coherence=0.0,
                temperature=self._temperature,
                used_bridge=False,
            )

        prev_x, prev_y, prev_t, _prev_conf = self._history[-1]
        dt = float(np.clip(ts - prev_t, float(self.config.min_dt_s), float(self.config.max_dt_s)))

        self._velocity = self._estimate_velocity()
        self._acceleration = self._estimate_acceleration()
        predicted = np.asarray([prev_x, prev_y], dtype=np.float64) + (
            self._velocity + self._drift_velocity
        ) * dt + 0.5 * self._acceleration * (dt * dt)

        innovation = curr - predicted
        distance = float(np.linalg.norm(innovation))
        clipped_dist = min(distance, float(self.config.max_association_distance_px))
        self._update_residual(clipped_dist)
        self._update_drift_velocity(innovation, dt, conf)
        drift_offset_px = float(np.linalg.norm(self._drift_velocity * dt))
        innovation_mad = self._innovation_mad()
        coherence = self._direction_coherence(innovation, self._velocity + self._drift_velocity)
        temperature = self._update_temperature(innovation_mad, distance, conf)
        gate = self._adaptive_gate(conf, drift_offset_px, coherence, temperature)

        high_thr = float(np.clip(self.config.high_confidence_threshold, 0.0, 1.0))
        low_thr = float(np.clip(self.config.low_confidence_threshold, 0.0, 1.0))
        if high_thr < low_thr:
            high_thr = low_thr

        if coherence <= float(self.config.coherence_reject_margin) and conf <= high_thr:
            self._reject_streak += 1
            self._accept_streak = 0
            self._hit_counter = max(0, self._hit_counter - max(1, int(self.config.miss_decay)))
            out_conf = conf * float(self.config.confidence_penalty)
            self._append_history((float(predicted[0]), float(predicted[1])), ts, out_conf)
            return self._result(
                center=(cx, cy),
                confidence=out_conf,
                accepted=False,
                refined=False,
                rejected=True,
                reason="coherence_too_low",
                predicted_center=(float(predicted[0]), float(predicted[1])),
                association_distance_px=distance,
                adaptive_gate_px=gate,
                drift_offset_px=drift_offset_px,
                innovation_mad_px=innovation_mad,
                coherence=coherence,
                temperature=temperature,
                used_bridge=False,
            )

        if distance <= gate:
            if conf < low_thr:
                blend = min(0.97, float(self.config.low_confidence_blend) + 0.12)
            elif conf < high_thr:
                blend = float(self.config.low_confidence_blend)
            else:
                blend = float(self.config.high_confidence_blend)

            if coherence < 0.0:
                blend = min(0.97, blend + 0.06 * abs(coherence))
            else:
                blend = max(0.0, blend - 0.05 * coherence)
            blend = float(np.clip(blend, 0.0, 0.97))

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
                    coherence=coherence,
                    temperature=temperature,
                    used_bridge=False,
                )

            refined_flag = shift_px >= float(self.config.min_refine_shift_px)
            out = refined_vec if refined_flag else curr
            out_conf = conf * float(self.config.confidence_gain)
            if coherence < 0.0:
                out_conf *= float(np.clip(1.0 + 0.08 * coherence, 0.82, 1.0))
            self._accept_streak += 1
            self._reject_streak = 0
            if self._accept_streak < max(1, int(self.config.accept_hysteresis_hits)):
                out_conf *= 0.995
            self._hit_counter = min(max(1, int(self.config.hit_counter_max)), self._hit_counter + 1)
            self._append_history((float(out[0]), float(out[1])), ts, out_conf)
            self._temperature = max(0.85, self._temperature * 0.96)
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
                coherence=coherence,
                temperature=self._temperature,
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
            and coherence >= float(self.config.coherence_floor)
        )
        if can_bridge:
            blend = float(np.clip(float(self.config.low_confidence_blend) + 0.2, 0.0, 0.97))
            bridged = (1.0 - blend) * curr + blend * predicted
            shift_px = float(np.linalg.norm(bridged - curr))
            if shift_px <= max(0.0, float(self.config.max_refine_shift_px)):
                out_conf = conf * float(self.config.confidence_penalty)
                self._append_history((float(bridged[0]), float(bridged[1])), ts, out_conf)
                self._hit_counter = min(max(1, int(self.config.hit_counter_max)), self._hit_counter + 1)
                self._temperature = min(2.0, self._temperature * 1.02)
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
                    coherence=coherence,
                    temperature=self._temperature,
                    used_bridge=True,
                )

        self._hit_counter = max(0, self._hit_counter - max(1, int(self.config.miss_decay)))
        out_conf = conf * float(self.config.confidence_penalty)
        self._append_history((float(predicted[0]), float(predicted[1])), ts, out_conf)
        self._temperature = min(2.8, self._temperature * 1.05)
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
            coherence=coherence,
            temperature=self._temperature,
            used_bridge=False,
        )
