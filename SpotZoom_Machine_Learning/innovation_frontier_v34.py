"""
SpotZoom innovation frontier v34.

This module adds a lightweight dual-threshold association bridge inspired by:
- ByteTrack: high/low confidence staged association
- Norfair: distance-threshold matching with hit-counter inertia
- CoTracker: trajectory continuity under short occlusion and appearance drops

The implementation intentionally depends only on numpy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class AssociationBridgeConfig:
    high_confidence_threshold: float = 0.52
    low_confidence_threshold: float = 0.14
    history_size: int = 16
    velocity_smoothing: float = 0.68
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.35
    max_velocity_px_per_s: float = 520.0
    base_gate_px: float = 4.0
    residual_gate_scale: float = 2.4
    max_association_distance_px: float = 24.0
    min_refine_shift_px: float = 0.2
    max_refine_shift_px: float = 16.0
    low_confidence_blend: float = 0.62
    high_confidence_blend: float = 0.18
    confidence_gain: float = 1.02
    confidence_penalty: float = 0.92
    residual_decay: float = 0.78
    hit_counter_max: int = 18
    miss_decay: int = 1


@dataclass
class AssociationBridgeResult:
    center: Tuple[float, float]
    confidence: float
    accepted: bool
    refined: bool
    rejected: bool
    reason: Optional[str]
    predicted_center: Tuple[float, float]
    association_distance_px: float
    adaptive_gate_px: float
    hit_counter: int
    used_low_conf_association: bool


class AssociationBridgeGate:
    """Low-confidence association bridge with temporal prediction and inertia."""

    def __init__(self, config: Optional[AssociationBridgeConfig] = None):
        self.config = config or AssociationBridgeConfig()
        self._history: Deque[Tuple[float, float, float, float]] = deque(
            maxlen=max(3, int(self.config.history_size))
        )
        self._velocity = np.zeros(2, dtype=np.float64)
        self._clock = 0.0
        self._residual_ema = 0.0
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))

    def reset(self) -> None:
        self._history.clear()
        self._velocity[:] = 0.0
        self._clock = 0.0
        self._residual_ema = 0.0
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))

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

    def _adaptive_gate(self, confidence: float) -> float:
        conf = float(np.clip(confidence, 0.0, 1.0))
        gate = float(self.config.base_gate_px) + float(self.config.residual_gate_scale) * float(self._residual_ema)
        if conf < float(self.config.low_confidence_threshold):
            gate *= 1.15
        elif conf >= float(self.config.high_confidence_threshold):
            gate *= 0.92
        gate = float(np.clip(gate, float(self.config.base_gate_px), float(self.config.max_association_distance_px)))
        return gate

    def _update_residual_ema(self, value: float) -> None:
        x = max(0.0, float(value))
        beta = float(np.clip(self.config.residual_decay, 0.0, 1.0))
        self._residual_ema = beta * float(self._residual_ema) + (1.0 - beta) * x

    def _append_history(self, center: Tuple[float, float], timestamp_s: float, confidence: float) -> None:
        cx = float(center[0])
        cy = float(center[1])
        conf = float(np.clip(confidence, 0.0, 1.0))
        self._history.append((cx, cy, float(timestamp_s), max(conf, 0.05)))

    def _result(
        self,
        center: Tuple[float, float],
        confidence: float,
        accepted: bool,
        refined: bool,
        rejected: bool,
        reason: Optional[str],
        predicted_center: Tuple[float, float],
        association_distance_px: float,
        adaptive_gate_px: float,
        used_low_conf_association: bool,
    ) -> AssociationBridgeResult:
        return AssociationBridgeResult(
            center=(float(center[0]), float(center[1])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            accepted=bool(accepted),
            refined=bool(refined),
            rejected=bool(rejected),
            reason=reason,
            predicted_center=(float(predicted_center[0]), float(predicted_center[1])),
            association_distance_px=float(association_distance_px),
            adaptive_gate_px=float(adaptive_gate_px),
            hit_counter=int(self._hit_counter),
            used_low_conf_association=bool(used_low_conf_association),
        )

    def update(
        self,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> AssociationBridgeResult:
        conf = float(np.clip(self._safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = self._safe_float(center[0], default=np.nan)
        cy = self._safe_float(center[1], default=np.nan)
        ts = self._resolve_timestamp(timestamp_s)

        if not np.isfinite(cx) or not np.isfinite(cy):
            self._hit_counter = max(0, self._hit_counter - max(1, int(self.config.miss_decay)))
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
                used_low_conf_association=False,
            )

        curr = np.asarray([cx, cy], dtype=np.float64)
        if not self._history:
            self._append_history((cx, cy), ts, conf)
            self._hit_counter = min(max(1, int(self.config.hit_counter_max)), self._hit_counter + 1)
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
                used_low_conf_association=False,
            )

        prev_x, prev_y, prev_t, _prev_conf = self._history[-1]
        dt = float(np.clip(ts - prev_t, float(self.config.min_dt_s), float(self.config.max_dt_s)))
        self._velocity = self._estimate_velocity()
        predicted = np.asarray([prev_x, prev_y], dtype=np.float64) + self._velocity * dt
        distance = float(np.linalg.norm(curr - predicted))
        gate = self._adaptive_gate(conf)
        self._update_residual_ema(min(distance, float(self.config.max_association_distance_px)))

        low_thr = float(np.clip(self.config.low_confidence_threshold, 0.0, 1.0))
        high_thr = float(np.clip(self.config.high_confidence_threshold, 0.0, 1.0))
        if high_thr < low_thr:
            high_thr = low_thr

        used_low_conf_assoc = conf < high_thr
        if distance <= gate:
            if conf < low_thr:
                blend = min(0.9, float(self.config.low_confidence_blend) + 0.12)
            elif conf < high_thr:
                blend = float(self.config.low_confidence_blend)
            else:
                blend = float(self.config.high_confidence_blend)
            blend = float(np.clip(blend, 0.0, 0.95))
            refined_vec = (1.0 - blend) * curr + blend * predicted
            shift_px = float(np.linalg.norm(refined_vec - curr))
            max_shift = max(0.0, float(self.config.max_refine_shift_px))
            if shift_px > max_shift:
                self._hit_counter = max(0, self._hit_counter - max(1, int(self.config.miss_decay)))
                self._append_history((float(predicted[0]), float(predicted[1])), ts, conf * float(self.config.confidence_penalty))
                return self._result(
                    center=(cx, cy),
                    confidence=conf * float(self.config.confidence_penalty),
                    accepted=False,
                    refined=False,
                    rejected=True,
                    reason="refine_shift_too_large",
                    predicted_center=(float(predicted[0]), float(predicted[1])),
                    association_distance_px=distance,
                    adaptive_gate_px=gate,
                    used_low_conf_association=used_low_conf_assoc,
                )

            if shift_px < float(self.config.min_refine_shift_px):
                out = curr
                refined = False
            else:
                out = refined_vec
                refined = True
            out_conf = conf * float(self.config.confidence_gain)
            self._hit_counter = min(max(1, int(self.config.hit_counter_max)), self._hit_counter + 1)
            self._append_history((float(out[0]), float(out[1])), ts, out_conf)
            return self._result(
                center=(float(out[0]), float(out[1])),
                confidence=out_conf,
                accepted=True,
                refined=refined,
                rejected=False,
                reason=None,
                predicted_center=(float(predicted[0]), float(predicted[1])),
                association_distance_px=distance,
                adaptive_gate_px=gate,
                used_low_conf_association=used_low_conf_assoc,
            )

        self._hit_counter = max(0, self._hit_counter - max(1, int(self.config.miss_decay)))
        can_bridge = (
            conf < high_thr
            and self._hit_counter > 0
            and distance <= float(self.config.max_association_distance_px)
        )
        if can_bridge:
            blend = float(np.clip(float(self.config.low_confidence_blend) + 0.2, 0.0, 0.95))
            bridged = (1.0 - blend) * curr + blend * predicted
            shift_px = float(np.linalg.norm(bridged - curr))
            if shift_px <= max(0.0, float(self.config.max_refine_shift_px)):
                self._append_history(
                    (float(bridged[0]), float(bridged[1])),
                    ts,
                    conf * float(self.config.confidence_penalty),
                )
                return self._result(
                    center=(float(bridged[0]), float(bridged[1])),
                    confidence=conf * float(self.config.confidence_penalty),
                    accepted=True,
                    refined=shift_px >= float(self.config.min_refine_shift_px),
                    rejected=False,
                    reason="bridge_recovery",
                    predicted_center=(float(predicted[0]), float(predicted[1])),
                    association_distance_px=distance,
                    adaptive_gate_px=gate,
                    used_low_conf_association=True,
                )

        self._append_history((float(predicted[0]), float(predicted[1])), ts, conf * float(self.config.confidence_penalty))
        return self._result(
            center=(cx, cy),
            confidence=conf * float(self.config.confidence_penalty),
            accepted=False,
            refined=False,
            rejected=True,
            reason="association_distance_too_large",
            predicted_center=(float(predicted[0]), float(predicted[1])),
            association_distance_px=distance,
            adaptive_gate_px=gate,
            used_low_conf_association=used_low_conf_assoc,
        )
