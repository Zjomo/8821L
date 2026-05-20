"""
SpotZoom innovation frontier v42.

Temporal-memory association gate inspired by:
- CoTracker3: long-range temporal continuity.
- ByteTrack: low-confidence bridge association.
- Norfair / DeepSORT: motion + appearance hybrid matching.
- Trackpy: prediction-centered adaptive search window.

The implementation is intentionally lightweight (numpy only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TemporalMemoryAssociationConfig:
    high_confidence_threshold: float = 0.72
    low_confidence_threshold: float = 0.2
    base_gate_px: float = 11.0
    min_gate_px: float = 4.0
    max_gate_px: float = 40.0
    jitter_gate_scale: float = 1.2
    velocity_gate_scale: float = 0.22
    appearance_weight: float = 0.34
    continuity_weight: float = 0.5
    confidence_weight: float = 0.16
    min_similarity: float = 0.14
    min_fusion_score: float = 0.28
    max_association_distance_px: float = 36.0
    max_refine_shift_px: float = 18.0
    min_refine_shift_px: float = 0.2
    bridge_gate_multiplier: float = 1.28
    confidence_gain: float = 1.02
    confidence_penalty: float = 0.93
    velocity_smoothing: float = 0.74
    appearance_momentum: float = 0.84
    jitter_decay: float = 0.86
    temperature_decay: float = 0.84
    max_velocity_px_per_s: float = 760.0
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45
    patch_radius: int = 16
    histogram_bins: int = 16
    hit_counter_max: int = 34
    miss_decay: int = 1
    accept_hysteresis_hits: int = 2
    reject_hysteresis_misses: int = 3


@dataclass
class TemporalMemoryAssociationResult:
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
    appearance_similarity: float
    fusion_score: float
    motion_score: float
    jitter_px: float
    temperature: float
    hit_counter: int
    reject_streak: int
    used_bridge: bool


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _normalize(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vec is None:
        return None
    try:
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return arr / norm


def _cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None or a.size != b.size:
        return None
    score = float(np.dot(a, b))
    if not np.isfinite(score):
        return None
    return float(np.clip(score, -1.0, 1.0))


def _to_gray(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 2:
        gray = np.asarray(arr, dtype=np.float32)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        # BGR/RGB robust average to avoid hard dependency on cv2.
        gray = np.mean(arr[..., :3].astype(np.float32), axis=2)
    else:
        return None
    if gray.size == 0:
        return None
    return gray


def build_patch_embedding(
    frame: np.ndarray,
    center: Tuple[float, float],
    *,
    radius: int = 16,
    bins: int = 16,
) -> Optional[np.ndarray]:
    gray = _to_gray(frame)
    if gray is None:
        return None
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return None
    cx = _safe_float(center[0], default=np.nan)
    cy = _safe_float(center[1], default=np.nan)
    if not np.isfinite(cx) or not np.isfinite(cy):
        return None

    r = max(4, int(radius))
    x0 = int(round(cx)) - r
    x1 = int(round(cx)) + r + 1
    y0 = int(round(cy)) - r
    y1 = int(round(cy)) + r + 1

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        src = np.pad(gray, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t
    else:
        src = gray

    patch = src[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    patch = np.asarray(patch, dtype=np.float32)

    hist_bins = max(8, int(bins))
    hist = np.histogram(np.clip(patch, 0.0, 255.0), bins=hist_bins, range=(0.0, 255.0), density=True)[0]
    gx = np.diff(patch, axis=1, prepend=patch[:, :1])
    gy = np.diff(patch, axis=0, prepend=patch[:1, :])
    grad_mag = np.sqrt(gx * gx + gy * gy)

    descriptor = np.concatenate(
        [
            hist.astype(np.float64),
            np.array(
                [
                    float(np.mean(patch)),
                    float(np.std(patch)),
                    float(np.percentile(patch, 10)),
                    float(np.percentile(patch, 90)),
                    float(np.mean(grad_mag)),
                    float(np.std(grad_mag)),
                ],
                dtype=np.float64,
            ),
        ],
        axis=0,
    )
    return _normalize(descriptor)


class TemporalMemoryAssociationGate:
    """Single-target temporal-memory association gate."""

    def __init__(self, config: Optional[TemporalMemoryAssociationConfig] = None):
        self.config = config or TemporalMemoryAssociationConfig()
        self._state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self._appearance_ref: Optional[np.ndarray] = None
        self._clock: float = 0.0
        self._prev_timestamp: Optional[float] = None
        self._jitter_ema: float = 0.0
        self._temperature: float = 1.0
        self._hit_counter: int = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak: int = 0
        self._reject_streak: int = 0

    def reset(self) -> None:
        self._state = None
        self._appearance_ref = None
        self._clock = 0.0
        self._prev_timestamp = None
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
        gate += float(max(0.0, self.config.jitter_gate_scale)) * float(max(0.0, self._jitter_ema))
        gate += float(max(0.0, self.config.velocity_gate_scale)) * float(max(0.0, predicted_speed_px_s))
        gate *= float(np.clip(self._temperature, 0.75, 2.6))
        gate = float(np.clip(gate, float(self.config.min_gate_px), float(self.config.max_gate_px)))
        gate = min(gate, float(max(self.config.min_gate_px, self.config.max_association_distance_px)))
        return float(gate)

    def _update_temperature(self, confidence: float, motion_score: float, similarity: float) -> float:
        decay = float(np.clip(self.config.temperature_decay, 0.0, 1.0))
        conf_term = 0.12 * (1.0 - float(np.clip(confidence, 0.0, 1.0)))
        motion_term = 0.24 * float(max(0.0, motion_score))
        sim_term = 0.18 * (1.0 - float(np.clip((similarity + 1.0) * 0.5, 0.0, 1.0)))
        rej_term = 0.05 * float(max(0, self._reject_streak))
        target = 1.0 + conf_term + motion_term + sim_term + rej_term
        self._temperature = float(np.clip(decay * self._temperature + (1.0 - decay) * target, 0.72, 2.9))
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
        appearance_similarity: float,
        fusion_score: float,
        motion_score: float,
        used_bridge: bool,
    ) -> TemporalMemoryAssociationResult:
        return TemporalMemoryAssociationResult(
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
            appearance_similarity=float(np.clip(appearance_similarity, -1.0, 1.0)),
            fusion_score=float(np.clip(fusion_score, 0.0, 1.0)),
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
    ) -> TemporalMemoryAssociationResult:
        conf = float(np.clip(_safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = _safe_float(center[0], default=np.nan)
        cy = _safe_float(center[1], default=np.nan)
        ts = self._resolve_timestamp(timestamp_s)

        predicted_center = (0.0, 0.0)
        if self._state is not None:
            predicted_center = (float(self._state[0]), float(self._state[1]))

        if not np.isfinite(cx) or not np.isfinite(cy):
            self._reject_streak += 1
            self._accept_streak = 0
            self._hit_counter = max(0, int(self._hit_counter) - int(max(1, self.config.miss_decay)))
            self._temperature = float(min(2.9, self._temperature + 0.12))
            return self._make_result(
                center=predicted_center,
                confidence=conf * float(self.config.confidence_penalty),
                accepted=False,
                refined=False,
                rejected=True,
                reason="invalid_center",
                predicted_center=predicted_center,
                association_distance_px=0.0,
                adaptive_gate_px=float(max(self.config.min_gate_px, self.config.base_gate_px)),
                continuity_score=0.0,
                appearance_similarity=0.0,
                fusion_score=0.0,
                motion_score=1.0,
                used_bridge=False,
            )

        embedding = build_patch_embedding(
            frame,
            (cx, cy),
            radius=max(4, int(self.config.patch_radius)),
            bins=max(8, int(self.config.histogram_bins)),
        )
        if self._state is None:
            self._state = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
            self._appearance_ref = embedding
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
                appearance_similarity=1.0,
                fusion_score=1.0,
                motion_score=0.0,
                used_bridge=False,
            )

        prev_ts = self._prev_timestamp if self._prev_timestamp is not None else (ts - 1.0 / 30.0)
        dt = float(np.clip(ts - prev_ts, float(self.config.min_dt_s), float(self.config.max_dt_s)))
        self._prev_timestamp = float(ts)

        pred_x = float(self._state[0] + self._state[2] * dt)
        pred_y = float(self._state[1] + self._state[3] * dt)
        pred_speed = float(np.hypot(self._state[2], self._state[3]))
        predicted_center = (pred_x, pred_y)

        obs_dx = float(cx - pred_x)
        obs_dy = float(cy - pred_y)
        dist = float(np.hypot(obs_dx, obs_dy))
        gate = self._effective_gate(conf, pred_speed)
        bridge_gate = gate * float(max(1.0, self.config.bridge_gate_multiplier))
        assoc_cap = float(max(1e-6, self.config.max_association_distance_px))
        motion_score = float(np.clip(dist / assoc_cap, 0.0, 1.0))
        continuity_score = float(np.exp(-dist / max(gate, 1e-6)))

        similarity = 0.0
        if embedding is not None and self._appearance_ref is not None:
            sim = _cosine_similarity(embedding, self._appearance_ref)
            if sim is not None:
                similarity = float(sim)

        conf_norm = float(np.clip((conf - self.config.low_confidence_threshold) / max(1e-6, 1.0 - self.config.low_confidence_threshold), 0.0, 1.0))
        fusion_score = (
            float(self.config.continuity_weight) * continuity_score
            + float(self.config.appearance_weight) * float(np.clip((similarity + 1.0) * 0.5, 0.0, 1.0))
            + float(self.config.confidence_weight) * conf_norm
        )
        fusion_score = float(np.clip(fusion_score, 0.0, 1.0))

        low_conf = float(self.config.low_confidence_threshold)
        base_accept = dist <= gate and conf >= low_conf and fusion_score >= float(self.config.min_fusion_score)
        bridge_accept = (
            conf < low_conf
            and conf >= low_conf * 0.5
            and dist <= bridge_gate
            and similarity >= float(self.config.min_similarity)
            and fusion_score >= float(self.config.min_fusion_score) * 0.9
        )
        hysteresis_accept = (
            not base_accept
            and not bridge_accept
            and self._reject_streak < int(max(1, self.config.reject_hysteresis_misses))
            and self._hit_counter > 0
            and dist <= bridge_gate * 1.08
            and fusion_score >= float(self.config.min_fusion_score) * 0.85
        )

        accepted = bool(base_accept or bridge_accept or hysteresis_accept)
        used_bridge = bool(bridge_accept or hysteresis_accept)

        if not accepted:
            self._reject_streak += 1
            self._accept_streak = 0
            self._hit_counter = max(0, int(self._hit_counter) - int(max(1, self.config.miss_decay)))
            self._jitter_ema = float(
                np.clip(
                    float(self.config.jitter_decay) * self._jitter_ema + (1.0 - float(self.config.jitter_decay)) * dist,
                    0.0,
                    assoc_cap,
                )
            )
            self._update_temperature(conf, motion_score, similarity)
            reason = "association_distance_exceeded" if dist > bridge_gate else "fusion_score_too_low"
            return self._make_result(
                center=predicted_center,
                confidence=conf * float(self.config.confidence_penalty),
                accepted=False,
                refined=False,
                rejected=True,
                reason=reason,
                predicted_center=predicted_center,
                association_distance_px=dist,
                adaptive_gate_px=gate,
                continuity_score=continuity_score,
                appearance_similarity=similarity,
                fusion_score=fusion_score,
                motion_score=motion_score,
                used_bridge=False,
            )

        blend = 0.32 + 0.5 * continuity_score + 0.18 * conf_norm
        if similarity > 0.0:
            blend += 0.1 * float(np.clip((similarity + 1.0) * 0.5, 0.0, 1.0))
        if used_bridge:
            blend *= 0.84
        blend = float(np.clip(blend, 0.2, 1.0))

        ref_x = pred_x + blend * obs_dx
        ref_y = pred_y + blend * obs_dy
        refine_shift = float(np.hypot(ref_x - cx, ref_y - cy))
        max_refine = float(max(0.0, self.config.max_refine_shift_px))
        if max_refine > 0.0 and refine_shift > max_refine:
            ratio = max_refine / max(refine_shift, 1e-6)
            ref_x = cx + (ref_x - cx) * ratio
            ref_y = cy + (ref_y - cy) * ratio
            refine_shift = float(np.hypot(ref_x - cx, ref_y - cy))

        refined = bool(refine_shift >= float(max(0.0, self.config.min_refine_shift_px)))

        vx_obs = (ref_x - float(self._state[0])) / max(dt, 1e-6)
        vy_obs = (ref_y - float(self._state[1])) / max(dt, 1e-6)
        alpha_v = float(np.clip(self.config.velocity_smoothing, 0.0, 1.0))
        vx_new = alpha_v * float(self._state[2]) + (1.0 - alpha_v) * vx_obs
        vy_new = alpha_v * float(self._state[3]) + (1.0 - alpha_v) * vy_obs
        vx_new, vy_new = self._clip_velocity(vx_new, vy_new)

        self._state = np.array([ref_x, ref_y, vx_new, vy_new], dtype=np.float64)
        self._jitter_ema = float(
            np.clip(
                float(self.config.jitter_decay) * self._jitter_ema + (1.0 - float(self.config.jitter_decay)) * dist,
                0.0,
                assoc_cap,
            )
        )
        self._reject_streak = 0
        self._accept_streak += 1
        self._hit_counter = min(int(self.config.hit_counter_max), self._hit_counter + 1)
        self._update_temperature(conf, motion_score, similarity)

        if embedding is not None:
            if self._appearance_ref is None:
                self._appearance_ref = embedding
            else:
                momentum = float(np.clip(self.config.appearance_momentum, 0.0, 1.0))
                merged = momentum * self._appearance_ref + (1.0 - momentum) * embedding
                self._appearance_ref = _normalize(merged)

        out_conf = float(np.clip(conf * float(self.config.confidence_gain), 0.0, 1.0))
        return self._make_result(
            center=(ref_x, ref_y),
            confidence=out_conf,
            accepted=True,
            refined=refined,
            rejected=False,
            reason=None,
            predicted_center=predicted_center,
            association_distance_px=dist,
            adaptive_gate_px=gate,
            continuity_score=continuity_score,
            appearance_similarity=similarity,
            fusion_score=fusion_score,
            motion_score=motion_score,
            used_bridge=used_bridge,
        )

