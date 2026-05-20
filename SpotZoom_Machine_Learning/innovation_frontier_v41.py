"""
SpotZoom innovation frontier v41.

Temporal continuity association gate inspired by:
- CoTracker: long-range point continuity and online streaming updates.
- ByteTrack: low-confidence bridge association for missed detections.
- Norfair: distance-threshold tracking with hit-counter hysteresis.

This module intentionally keeps dependencies minimal (numpy only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TemporalContinuityConfig:
    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.18
    base_gate_px: float = 12.0
    min_gate_px: float = 4.0
    jitter_gate_scale: float = 1.25
    velocity_gate_scale: float = 0.22
    max_association_distance_px: float = 36.0
    max_refine_shift_px: float = 18.0
    continuity_min_score: float = 0.28
    bridge_gate_multiplier: float = 1.25
    confidence_gain: float = 1.02
    confidence_penalty: float = 0.94
    velocity_smoothing: float = 0.72
    max_velocity_px_per_s: float = 760.0
    temperature_decay: float = 0.84
    jitter_decay: float = 0.86
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45
    hit_counter_max: int = 32
    miss_decay: int = 1
    accept_hysteresis_hits: int = 2
    reject_hysteresis_misses: int = 3


@dataclass
class TemporalContinuityResult:
    center: Tuple[float, float]
    confidence: float
    accepted: bool
    refined: bool
    rejected: bool
    reason: Optional[str]
    predicted_center: Tuple[float, float]
    association_distance_px: float
    adaptive_gate_px: float
    continuity_score: float
    motion_score: float
    jitter_px: float
    temperature: float
    hit_counter: int
    reject_streak: int
    used_bridge: bool


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if not np.isfinite(x):
        return default
    return x


class TemporalContinuityGate:
    """Single-target temporal continuity association gate."""

    def __init__(self, config: Optional[TemporalContinuityConfig] = None):
        self.config = config or TemporalContinuityConfig()
        self._state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self._prev_timestamp: Optional[float] = None
        self._clock: float = 0.0
        self._jitter_ema: float = 0.0
        self._temperature: float = 1.0
        self._hit_counter: int = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak: int = 0
        self._reject_streak: int = 0

    def reset(self) -> None:
        self._state = None
        self._prev_timestamp = None
        self._clock = 0.0
        self._jitter_ema = 0.0
        self._temperature = 1.0
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak = 0
        self._reject_streak = 0

    def _resolve_timestamp(self, timestamp_s: Optional[float]) -> float:
        if timestamp_s is None:
            self._clock += 1.0 / 30.0
            return float(self._clock)
        ts = _safe_float(timestamp_s, default=self._clock + 1.0 / 30.0)
        if ts <= self._clock:
            ts = self._clock + 1.0 / 30.0
        self._clock = float(ts)
        return float(ts)

    def _clip_velocity(self, vx: float, vy: float) -> Tuple[float, float]:
        vmax = max(0.0, float(self.config.max_velocity_px_per_s))
        if vmax <= 0.0:
            return 0.0, 0.0
        speed = float(np.hypot(vx, vy))
        if speed <= vmax:
            return float(vx), float(vy)
        scale = vmax / max(speed, 1e-6)
        return float(vx * scale), float(vy * scale)

    def _effective_gate(self, confidence: float, predicted_speed_px_s: float) -> float:
        conf = float(np.clip(confidence, 0.0, 1.0))
        gate = float(max(self.config.min_gate_px, self.config.base_gate_px))
        if conf < float(self.config.low_confidence_threshold):
            gate *= 1.1
        elif conf >= float(self.config.high_confidence_threshold):
            gate *= 0.94

        speed_term = float(max(0.0, predicted_speed_px_s)) * float(max(0.0, self.config.velocity_gate_scale))
        gate += float(max(0.0, self.config.jitter_gate_scale)) * float(max(0.0, self._jitter_ema))
        gate += speed_term
        gate *= float(np.clip(self._temperature, 0.75, 2.4))
        gate = float(np.clip(gate, float(self.config.min_gate_px), float(self.config.max_association_distance_px)))
        return gate

    def _update_temperature(self, confidence: float, motion_score: float, continuity_score: float) -> float:
        decay = float(np.clip(self.config.temperature_decay, 0.0, 1.0))
        conf_term = 0.12 * (1.0 - float(np.clip(confidence, 0.0, 1.0)))
        motion_term = 0.26 * float(max(0.0, motion_score))
        continuity_term = 0.18 * (1.0 - float(np.clip(continuity_score, 0.0, 1.0)))
        rej_term = 0.05 * float(max(0, self._reject_streak))
        target = 1.0 + conf_term + motion_term + continuity_term + rej_term
        self._temperature = float(np.clip(decay * self._temperature + (1.0 - decay) * target, 0.7, 2.8))
        return self._temperature

    def _make_result(
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
        continuity_score: float,
        motion_score: float,
        used_bridge: bool,
    ) -> TemporalContinuityResult:
        return TemporalContinuityResult(
            center=(float(center[0]), float(center[1])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            accepted=bool(accepted),
            refined=bool(refined),
            rejected=bool(rejected),
            reason=reason,
            predicted_center=(float(predicted_center[0]), float(predicted_center[1])),
            association_distance_px=float(max(0.0, association_distance_px)),
            adaptive_gate_px=float(max(0.0, adaptive_gate_px)),
            continuity_score=float(np.clip(continuity_score, 0.0, 1.0)),
            motion_score=float(np.clip(motion_score, 0.0, 1.0)),
            jitter_px=float(max(0.0, self._jitter_ema)),
            temperature=float(max(0.0, self._temperature)),
            hit_counter=int(self._hit_counter),
            reject_streak=int(self._reject_streak),
            used_bridge=bool(used_bridge),
        )

    def update(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> TemporalContinuityResult:
        _ = frame  # Kept for API compatibility and future feature growth.
        conf = float(np.clip(_safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = _safe_float(center[0], default=np.nan)
        cy = _safe_float(center[1], default=np.nan)
        ts = self._resolve_timestamp(timestamp_s)

        pred = (0.0, 0.0)
        if self._state is not None:
            pred = (float(self._state[0]), float(self._state[1]))

        if not np.isfinite(cx) or not np.isfinite(cy):
            self._reject_streak += 1
            self._accept_streak = 0
            self._hit_counter = max(0, int(self._hit_counter) - int(max(1, self.config.miss_decay)))
            self._temperature = float(min(2.8, self._temperature + 0.12))
            return self._make_result(
                center=pred,
                confidence=conf * float(self.config.confidence_penalty),
                accepted=False,
                refined=False,
                rejected=True,
                reason="invalid_center",
                predicted_center=pred,
                association_distance_px=0.0,
                adaptive_gate_px=float(max(self.config.min_gate_px, self.config.base_gate_px)),
                continuity_score=0.0,
                motion_score=1.0,
                used_bridge=False,
            )

        if self._state is None:
            self._state = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
            self._prev_timestamp = float(ts)
            self._jitter_ema = 0.0
            self._temperature = 1.0
            self._accept_streak = 1
            self._reject_streak = 0
            self._hit_counter = min(int(self.config.hit_counter_max), self._hit_counter + 1)
            return self._make_result(
                center=(cx, cy),
                confidence=conf,
                accepted=True,
                refined=False,
                rejected=False,
                reason=None,
                predicted_center=(cx, cy),
                association_distance_px=0.0,
                adaptive_gate_px=float(max(self.config.min_gate_px, self.config.base_gate_px)),
                continuity_score=1.0,
                motion_score=0.0,
                used_bridge=False,
            )

        prev_ts = self._prev_timestamp if self._prev_timestamp is not None else (ts - 1.0 / 30.0)
        dt = float(np.clip(ts - prev_ts, float(self.config.min_dt_s), float(self.config.max_dt_s)))
        self._prev_timestamp = float(ts)

        pred_x = float(self._state[0] + self._state[2] * dt)
        pred_y = float(self._state[1] + self._state[3] * dt)
        pred_v = float(np.hypot(self._state[2], self._state[3]))

        obs_dx = float(cx - pred_x)
        obs_dy = float(cy - pred_y)
        dist = float(np.hypot(obs_dx, obs_dy))

        gate = self._effective_gate(conf, pred_v)
        bridge_gate = gate * float(max(1.0, self.config.bridge_gate_multiplier))
        assoc_cap = float(max(1e-6, self.config.max_association_distance_px))
        motion_score = float(np.clip(dist / assoc_cap, 0.0, 1.0))
        continuity_score = float(np.exp(-dist / max(gate, 1e-6)))

        low_conf = float(self.config.low_confidence_threshold)
        base_accept = dist <= gate and conf >= low_conf
        bridge_accept = (
            conf < low_conf
            and conf >= low_conf * 0.5
            and dist <= bridge_gate
            and continuity_score >= float(self.config.continuity_min_score)
        )
        hysteresis_accept = (
            not base_accept
            and not bridge_accept
            and self._reject_streak < int(max(1, self.config.reject_hysteresis_misses))
            and self._hit_counter > 0
            and dist <= bridge_gate * 1.08
            and continuity_score >= float(self.config.continuity_min_score) * 0.9
        )

        accepted = bool(base_accept or bridge_accept or hysteresis_accept)
        used_bridge = bool(bridge_accept or hysteresis_accept)
        if not accepted:
            self._reject_streak += 1
            self._accept_streak = 0
            self._hit_counter = max(0, int(self._hit_counter) - int(max(1, self.config.miss_decay)))
            self._jitter_ema = float(
                np.clip(
                    float(self.config.jitter_decay) * self._jitter_ema
                    + (1.0 - float(self.config.jitter_decay)) * dist,
                    0.0,
                    assoc_cap,
                )
            )
            self._update_temperature(conf, motion_score, continuity_score)
            reason = "association_distance_exceeded" if dist > bridge_gate else "low_confidence"
            return self._make_result(
                center=(pred_x, pred_y),
                confidence=conf * float(self.config.confidence_penalty),
                accepted=False,
                refined=False,
                rejected=True,
                reason=reason,
                predicted_center=(pred_x, pred_y),
                association_distance_px=dist,
                adaptive_gate_px=gate,
                continuity_score=continuity_score,
                motion_score=motion_score,
                used_bridge=False,
            )

        conf_norm = float(np.clip((conf - low_conf) / max(1e-6, 1.0 - low_conf), 0.0, 1.0))
        blend = 0.35 + 0.5 * continuity_score + 0.15 * conf_norm
        if used_bridge:
            blend *= 0.82
        blend = float(np.clip(blend, 0.22, 1.0))

        ref_x = pred_x + blend * obs_dx
        ref_y = pred_y + blend * obs_dy
        refine_shift = float(np.hypot(ref_x - cx, ref_y - cy))
        max_refine = float(max(0.0, self.config.max_refine_shift_px))
        if max_refine > 0.0 and refine_shift > max_refine:
            ratio = max_refine / max(refine_shift, 1e-6)
            ref_x = cx + (ref_x - cx) * ratio
            ref_y = cy + (ref_y - cy) * ratio
            refine_shift = float(np.hypot(ref_x - cx, ref_y - cy))

        refined = bool(refine_shift > 0.2)

        meas_vx = (ref_x - float(self._state[0])) / max(dt, 1e-6)
        meas_vy = (ref_y - float(self._state[1])) / max(dt, 1e-6)
        alpha = float(np.clip(self.config.velocity_smoothing, 0.0, 1.0))
        new_vx = alpha * float(self._state[2]) + (1.0 - alpha) * meas_vx
        new_vy = alpha * float(self._state[3]) + (1.0 - alpha) * meas_vy
        new_vx, new_vy = self._clip_velocity(new_vx, new_vy)

        self._state = np.array([ref_x, ref_y, new_vx, new_vy], dtype=np.float64)
        self._jitter_ema = float(
            np.clip(
                float(self.config.jitter_decay) * self._jitter_ema
                + (1.0 - float(self.config.jitter_decay)) * dist,
                0.0,
                assoc_cap,
            )
        )
        self._accept_streak += 1
        self._reject_streak = 0
        self._hit_counter = min(int(self.config.hit_counter_max), self._hit_counter + 1)
        self._update_temperature(conf, motion_score, continuity_score)

        out_conf = conf * float(self.config.confidence_gain)
        if used_bridge:
            out_conf = min(out_conf, max(conf, low_conf * 1.15))

        return self._make_result(
            center=(ref_x, ref_y),
            confidence=out_conf,
            accepted=True,
            refined=refined,
            rejected=False,
            reason=None,
            predicted_center=(pred_x, pred_y),
            association_distance_px=dist,
            adaptive_gate_px=gate,
            continuity_score=continuity_score,
            motion_score=motion_score,
            used_bridge=used_bridge,
        )

