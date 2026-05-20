"""
SpotZoom innovation frontier v38.

This module introduces a Mahalanobis-hysteresis association gate inspired by:
- DeepSORT: Kalman + Mahalanobis innovation gating.
- ByteTrack: high/low-confidence staged association with bridge recovery.
- Norfair/BoT-SORT: lifecycle hysteresis and adaptive gating under jitter.

Only numpy is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class MahalanobisAssociationConfig:
    high_confidence_threshold: float = 0.64
    low_confidence_threshold: float = 0.15
    base_nis_gate: float = 6.4
    min_nis_gate: float = 2.5
    max_nis_gate: float = 14.0
    jitter_gate_scale: float = 1.6
    max_association_distance_px: float = 36.0
    max_refine_shift_px: float = 20.0
    min_refine_shift_px: float = 0.2
    bridge_gate_multiplier: float = 1.35
    confidence_gain: float = 1.03
    confidence_penalty: float = 0.93
    process_noise_position: float = 1.8
    process_noise_velocity: float = 24.0
    measurement_noise: float = 2.2
    initial_covariance: float = 14.0
    max_velocity_px_per_s: float = 700.0
    velocity_smoothing: float = 0.72
    jitter_decay: float = 0.86
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45
    temperature_decay: float = 0.84
    hit_counter_max: int = 30
    miss_decay: int = 1
    accept_hysteresis_hits: int = 2
    reject_hysteresis_misses: int = 3


@dataclass
class MahalanobisAssociationResult:
    center: Tuple[float, float]
    confidence: float
    accepted: bool
    refined: bool
    rejected: bool
    reason: Optional[str]
    predicted_center: Tuple[float, float]
    association_distance_px: float
    adaptive_gate_px: float
    nis: float
    jitter_px: float
    innovation_mahalanobis: float
    temperature: float
    hit_counter: int
    used_bridge: bool
    reject_streak: int


class MahalanobisAssociationGate:
    """Single-target Kalman association gate with NIS + hysteresis bridge recovery."""

    def __init__(self, config: Optional[MahalanobisAssociationConfig] = None):
        self.config = config or MahalanobisAssociationConfig()
        self._state: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._clock = 0.0
        self._prev_timestamp = None
        self._temperature = 1.0
        self._jitter_ema = 0.0
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak = 0
        self._reject_streak = 0

    def reset(self) -> None:
        self._state = None
        self._cov = None
        self._clock = 0.0
        self._prev_timestamp = None
        self._temperature = 1.0
        self._jitter_ema = 0.0
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
    def _clip_velocity(vec: np.ndarray, max_norm: float) -> np.ndarray:
        max_norm = max(0.0, float(max_norm))
        if vec.size < 2 or max_norm <= 0:
            return vec
        speed = float(np.linalg.norm(vec[:2]))
        if speed <= max_norm:
            return vec
        scale = max_norm / max(speed, 1e-6)
        clipped = vec.copy()
        clipped[:2] *= scale
        return clipped

    def _resolve_timestamp(self, timestamp_s: Optional[float]) -> float:
        if timestamp_s is None:
            self._clock += 1.0 / 30.0
            return float(self._clock)
        ts = self._safe_float(timestamp_s, default=self._clock + 1.0 / 30.0)
        if ts <= self._clock:
            ts = self._clock + 1.0 / 30.0
        self._clock = float(ts)
        return float(ts)

    def _transition(self, dt: float) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _process_noise(self, dt: float) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q_pos = float(max(1e-6, self.config.process_noise_position)) ** 2
        q_vel = float(max(1e-6, self.config.process_noise_velocity)) ** 2
        return np.array(
            [
                [q_pos + 0.25 * dt4 * q_vel, 0.0, 0.5 * dt3 * q_vel, 0.0],
                [0.0, q_pos + 0.25 * dt4 * q_vel, 0.0, 0.5 * dt3 * q_vel],
                [0.5 * dt3 * q_vel, 0.0, dt2 * q_vel, 0.0],
                [0.0, 0.5 * dt3 * q_vel, 0.0, dt2 * q_vel],
            ],
            dtype=np.float64,
        )

    def _predict(self, dt: float) -> None:
        if self._state is None or self._cov is None:
            return
        f = self._transition(dt)
        q = self._process_noise(dt)
        self._state = f @ self._state
        self._state[2:] = self._clip_velocity(self._state[2:], float(self.config.max_velocity_px_per_s))
        self._cov = f @ self._cov @ f.T + q
        self._cov = 0.5 * (self._cov + self._cov.T)

    def _effective_nis_gate(self, confidence: float) -> float:
        conf = float(np.clip(confidence, 0.0, 1.0))
        high_thr = float(np.clip(self.config.high_confidence_threshold, 0.0, 1.0))
        low_thr = float(np.clip(self.config.low_confidence_threshold, 0.0, 1.0))
        if high_thr < low_thr:
            high_thr = low_thr
        gate = float(max(0.01, self.config.base_nis_gate))
        if conf < low_thr:
            gate *= 1.12
        elif conf >= high_thr:
            gate *= 0.94
        gate += float(max(0.0, self.config.jitter_gate_scale)) * float(max(0.0, self._jitter_ema))
        gate *= float(np.clip(self._temperature, 0.8, 2.4))
        return float(np.clip(gate, float(self.config.min_nis_gate), float(self.config.max_nis_gate)))

    def _effective_distance_gate(self, confidence: float) -> float:
        conf = float(np.clip(confidence, 0.0, 1.0))
        gate = float(max(0.0, self.config.max_association_distance_px))
        if conf < float(self.config.low_confidence_threshold):
            gate *= 1.08
        gate += float(max(0.0, self.config.jitter_gate_scale)) * float(max(0.0, self._jitter_ema))
        return float(max(0.0, gate))

    def _update_temperature(self, confidence: float, nis: float, distance_px: float) -> float:
        decay = float(np.clip(self.config.temperature_decay, 0.0, 1.0))
        conf = float(np.clip(confidence, 0.0, 1.0))
        motion_term = 0.03 * float(max(0.0, nis)) + 0.01 * float(max(0.0, distance_px))
        conf_term = 0.12 * (1.0 - conf)
        rej_term = 0.05 * float(max(0, self._reject_streak))
        raw = decay * float(self._temperature) + (1.0 - decay) * (1.0 + motion_term + conf_term + rej_term)
        self._temperature = float(np.clip(raw, 0.75, 2.8))
        return self._temperature

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
        nis: float,
        jitter_px: float,
        innovation_mahalanobis: float,
        used_bridge: bool,
    ) -> MahalanobisAssociationResult:
        return MahalanobisAssociationResult(
            center=(float(center[0]), float(center[1])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            accepted=bool(accepted),
            refined=bool(refined),
            rejected=bool(rejected),
            reason=reason,
            predicted_center=(float(predicted_center[0]), float(predicted_center[1])),
            association_distance_px=float(max(0.0, association_distance_px)),
            adaptive_gate_px=float(max(0.0, adaptive_gate_px)),
            nis=float(max(0.0, nis)),
            jitter_px=float(max(0.0, jitter_px)),
            innovation_mahalanobis=float(max(0.0, innovation_mahalanobis)),
            temperature=float(max(0.0, self._temperature)),
            hit_counter=int(self._hit_counter),
            used_bridge=bool(used_bridge),
            reject_streak=int(self._reject_streak),
        )

    def _measurement_update(self, measurement: np.ndarray, r: np.ndarray) -> None:
        if self._state is None or self._cov is None:
            return
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        innovation = measurement - (h @ self._state)
        s = h @ self._cov @ h.T + r
        inv_s = np.linalg.pinv(s)
        k = self._cov @ h.T @ inv_s
        self._state = self._state + k @ innovation
        i = np.eye(4, dtype=np.float64)
        self._cov = (i - k @ h) @ self._cov
        self._cov = 0.5 * (self._cov + self._cov.T)
        self._state[2:] = self._clip_velocity(self._state[2:], float(self.config.max_velocity_px_per_s))

    def update(
        self,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> MahalanobisAssociationResult:
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
                confidence=conf * float(self.config.confidence_penalty),
                accepted=False,
                refined=False,
                rejected=True,
                reason="invalid_center",
                predicted_center=(0.0, 0.0),
                association_distance_px=float("inf"),
                adaptive_gate_px=0.0,
                nis=float("inf"),
                jitter_px=self._jitter_ema,
                innovation_mahalanobis=float("inf"),
                used_bridge=False,
            )

        if self._state is None or self._cov is None:
            self._state = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
            init_var = float(max(1e-6, self.config.initial_covariance)) ** 2
            self._cov = np.diag([init_var, init_var, init_var * 0.5, init_var * 0.5]).astype(np.float64)
            self._prev_timestamp = ts
            self._temperature = 1.0
            self._jitter_ema = 0.0
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
                nis=0.0,
                jitter_px=self._jitter_ema,
                innovation_mahalanobis=0.0,
                used_bridge=False,
            )

        prev_ts = float(self._prev_timestamp if self._prev_timestamp is not None else ts - (1.0 / 30.0))
        dt = float(np.clip(ts - prev_ts, float(self.config.min_dt_s), float(self.config.max_dt_s)))
        self._prev_timestamp = ts

        self._predict(dt)
        predicted = self._state[:2].copy()

        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        r_var = float(max(1e-6, self.config.measurement_noise)) ** 2
        r = np.diag([r_var, r_var]).astype(np.float64)
        z = np.array([cx, cy], dtype=np.float64)
        innovation = z - (h @ self._state)
        s = h @ self._cov @ h.T + r
        inv_s = np.linalg.pinv(s)
        nis = float(np.dot(innovation, inv_s @ innovation))
        distance = float(np.linalg.norm(innovation))
        mahal = float(np.sqrt(max(0.0, nis)))

        jitter_decay = float(np.clip(self.config.jitter_decay, 0.0, 1.0))
        self._jitter_ema = jitter_decay * self._jitter_ema + (1.0 - jitter_decay) * max(0.0, distance)
        nis_gate = self._effective_nis_gate(conf)
        dist_gate = self._effective_distance_gate(conf)
        px_gate = float(np.sqrt(max(1e-6, nis_gate) * max(1e-6, float(np.mean(np.diag(s))))))
        self._update_temperature(confidence=conf, nis=nis, distance_px=distance)

        accepted = (nis <= nis_gate) and (distance <= dist_gate)
        if accepted:
            self._measurement_update(z, r)
            out_center = self._state[:2].copy()
            shift_px = float(np.linalg.norm(out_center - z))
            if shift_px > float(max(0.0, self.config.max_refine_shift_px)):
                accepted = False
            else:
                out_conf = conf * float(self.config.confidence_gain)
                refined = bool(shift_px >= float(self.config.min_refine_shift_px))
                self._accept_streak += 1
                self._reject_streak = 0
                if self._accept_streak < max(1, int(self.config.accept_hysteresis_hits)):
                    out_conf *= 0.995
                self._hit_counter = min(max(1, int(self.config.hit_counter_max)), self._hit_counter + 1)
                self._temperature = max(0.84, self._temperature * 0.96)
                return self._result(
                    center=(float(out_center[0]), float(out_center[1])),
                    confidence=out_conf,
                    accepted=True,
                    refined=refined,
                    rejected=False,
                    reason=None,
                    predicted_center=(float(predicted[0]), float(predicted[1])),
                    association_distance_px=distance,
                    adaptive_gate_px=px_gate,
                    nis=nis,
                    jitter_px=self._jitter_ema,
                    innovation_mahalanobis=mahal,
                    used_bridge=False,
                )

        self._reject_streak += 1
        self._accept_streak = 0
        bridge_nis_gate = nis_gate * max(1.0, float(self.config.bridge_gate_multiplier))
        bridge_dist_gate = dist_gate * max(1.0, float(self.config.bridge_gate_multiplier))
        can_bridge = (
            self._reject_streak <= max(1, int(self.config.reject_hysteresis_misses))
            and self._hit_counter > 0
            and nis <= bridge_nis_gate
            and distance <= bridge_dist_gate
        )
        if can_bridge:
            low_thr = float(np.clip(self.config.low_confidence_threshold, 0.0, 1.0))
            bridge_blend = 0.78 if conf < low_thr else 0.62
            bridged = (1.0 - bridge_blend) * z + bridge_blend * predicted
            shift_px = float(np.linalg.norm(bridged - z))
            if shift_px <= float(max(0.0, self.config.max_refine_shift_px)):
                self._measurement_update(bridged, r)
                out_conf = conf * float(self.config.confidence_penalty)
                self._hit_counter = min(max(1, int(self.config.hit_counter_max)), self._hit_counter + 1)
                self._temperature = min(2.2, self._temperature * 1.01)
                return self._result(
                    center=(float(bridged[0]), float(bridged[1])),
                    confidence=out_conf,
                    accepted=True,
                    refined=bool(shift_px >= float(self.config.min_refine_shift_px)),
                    rejected=False,
                    reason="hysteresis_bridge",
                    predicted_center=(float(predicted[0]), float(predicted[1])),
                    association_distance_px=distance,
                    adaptive_gate_px=px_gate,
                    nis=nis,
                    jitter_px=self._jitter_ema,
                    innovation_mahalanobis=mahal,
                    used_bridge=True,
                )

        self._hit_counter = max(0, self._hit_counter - max(1, int(self.config.miss_decay)))
        self._temperature = min(2.8, self._temperature * 1.05)
        self._cov *= 1.08
        out_conf = conf * float(self.config.confidence_penalty)
        return self._result(
            center=(cx, cy),
            confidence=out_conf,
            accepted=False,
            refined=False,
            rejected=True,
            reason="mahalanobis_gate_reject",
            predicted_center=(float(predicted[0]), float(predicted[1])),
            association_distance_px=distance,
            adaptive_gate_px=px_gate,
            nis=nis,
            jitter_px=self._jitter_ema,
            innovation_mahalanobis=mahal,
            used_bridge=False,
        )
