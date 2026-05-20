"""
SpotZoom innovation frontier v39.

Appearance-motion fusion association gate inspired by:
- Deep SORT: appearance descriptor + motion model association.
- ByteTrack: high/low-confidence staged association.
- BoT-SORT: robust association under motion perturbations.

Only numpy is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AppearanceMotionFusionConfig:
    high_confidence_threshold: float = 0.66
    low_confidence_threshold: float = 0.15
    base_gate_px: float = 34.0
    min_gate_px: float = 5.0
    max_gate_px: float = 72.0
    jitter_gate_scale: float = 1.35
    max_association_distance_px: float = 38.0
    max_refine_shift_px: float = 22.0
    min_refine_shift_px: float = 0.2
    appearance_weight: float = 0.35
    min_similarity: float = 0.2
    bridge_similarity: float = 0.12
    bridge_gate_multiplier: float = 1.35
    confidence_gain: float = 1.03
    confidence_penalty: float = 0.93
    velocity_smoothing: float = 0.72
    max_velocity_px_per_s: float = 720.0
    appearance_momentum: float = 0.82
    jitter_decay: float = 0.86
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45
    temperature_decay: float = 0.84
    hit_counter_max: int = 30
    miss_decay: int = 1
    accept_hysteresis_hits: int = 2
    reject_hysteresis_misses: int = 3


@dataclass
class AppearanceMotionFusionResult:
    center: Tuple[float, float]
    confidence: float
    accepted: bool
    refined: bool
    rejected: bool
    reason: Optional[str]
    predicted_center: Tuple[float, float]
    association_distance_px: float
    adaptive_gate_px: float
    motion_score: float
    appearance_similarity: float
    fusion_score: float
    jitter_px: float
    temperature: float
    hit_counter: int
    used_bridge: bool
    reject_streak: int


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if not np.isfinite(x):
        return default
    return x


def _normalize_embedding(embedding: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if embedding is None:
        return None
    try:
        vec = np.asarray(embedding, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if vec.size < 4:
        return None
    if not np.all(np.isfinite(vec)):
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return None
    return vec / norm


def _cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.size != b.size:
        return None
    sim = float(np.dot(a, b))
    if not np.isfinite(sim):
        return None
    return float(np.clip(sim, -1.0, 1.0))


def build_patch_embedding(frame: np.ndarray, bbox: Tuple[int, int, int, int], bins: int = 16) -> Optional[np.ndarray]:
    """Build a lightweight appearance descriptor from HSV histograms."""
    if frame is None:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in bbox)
    except Exception:
        return None
    h, w = frame.shape[:2]
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    if (x2 - x1) * (y2 - y1) < 36:
        return None
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    if patch.ndim == 2:
        gray = patch.astype(np.float32)
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        p10 = float(np.percentile(gray, 10))
        p90 = float(np.percentile(gray, 90))
        hist = np.histogram(np.clip(gray, 0, 255), bins=max(8, int(bins)), range=(0, 255), density=True)[0]
        descriptor = np.concatenate([hist.astype(np.float64), np.array([mean, std, p10, p90], dtype=np.float64)], axis=0)
        return _normalize_embedding(descriptor)

    patch_u8 = patch.astype(np.uint8)
    try:
        hsv = __import__("cv2").cvtColor(patch_u8, __import__("cv2").COLOR_BGR2HSV)
    except Exception:
        gray = np.mean(patch_u8.astype(np.float32), axis=2)
        hist = np.histogram(np.clip(gray, 0, 255), bins=max(8, int(bins)), range=(0, 255), density=True)[0]
        return _normalize_embedding(hist.astype(np.float64))

    bins = max(8, int(bins))
    h_hist = np.histogram(hsv[..., 0], bins=bins, range=(0, 180), density=True)[0]
    s_hist = np.histogram(hsv[..., 1], bins=bins, range=(0, 256), density=True)[0]
    v_hist = np.histogram(hsv[..., 2], bins=bins, range=(0, 256), density=True)[0]
    descriptor = np.concatenate([h_hist, s_hist, v_hist], axis=0).astype(np.float64)
    return _normalize_embedding(descriptor)


class AppearanceMotionFusionGate:
    """Single-target appearance-motion fusion association gate."""

    def __init__(self, config: Optional[AppearanceMotionFusionConfig] = None):
        self.config = config or AppearanceMotionFusionConfig()
        self._state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self._clock = 0.0
        self._prev_timestamp = None
        self._temperature = 1.0
        self._jitter_ema = 0.0
        self._appearance_ref: Optional[np.ndarray] = None
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak = 0
        self._reject_streak = 0

    def reset(self) -> None:
        self._state = None
        self._clock = 0.0
        self._prev_timestamp = None
        self._temperature = 1.0
        self._jitter_ema = 0.0
        self._appearance_ref = None
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
        if vmax <= 0:
            return 0.0, 0.0
        speed = float(np.hypot(vx, vy))
        if speed <= vmax:
            return vx, vy
        scale = vmax / max(speed, 1e-6)
        return float(vx * scale), float(vy * scale)

    def _effective_gate(self, confidence: float) -> float:
        conf = float(np.clip(confidence, 0.0, 1.0))
        gate = float(max(0.0, self.config.base_gate_px))
        if conf < float(self.config.low_confidence_threshold):
            gate *= 1.1
        elif conf >= float(self.config.high_confidence_threshold):
            gate *= 0.95
        gate += float(max(0.0, self.config.jitter_gate_scale)) * float(max(0.0, self._jitter_ema))
        gate *= float(np.clip(self._temperature, 0.8, 2.4))
        gate = float(np.clip(gate, float(self.config.min_gate_px), float(self.config.max_gate_px)))
        return gate

    def _update_temperature(self, confidence: float, motion_score: float, appearance_similarity: float) -> float:
        decay = float(np.clip(self.config.temperature_decay, 0.0, 1.0))
        conf_term = 0.14 * (1.0 - float(np.clip(confidence, 0.0, 1.0)))
        motion_term = 0.28 * float(max(0.0, motion_score))
        app_term = 0.0
        if np.isfinite(appearance_similarity):
            app_term = 0.2 * (1.0 - float(np.clip((appearance_similarity + 1.0) * 0.5, 0.0, 1.0)))
        rej_term = 0.05 * float(max(0, self._reject_streak))
        target = 1.0 + conf_term + motion_term + app_term + rej_term
        self._temperature = float(np.clip(decay * self._temperature + (1.0 - decay) * target, 0.75, 2.8))
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
        motion_score: float,
        appearance_similarity: float,
        fusion_score: float,
        used_bridge: bool,
    ) -> AppearanceMotionFusionResult:
        return AppearanceMotionFusionResult(
            center=(float(center[0]), float(center[1])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            accepted=bool(accepted),
            refined=bool(refined),
            rejected=bool(rejected),
            reason=reason,
            predicted_center=(float(predicted_center[0]), float(predicted_center[1])),
            association_distance_px=float(max(0.0, association_distance_px)),
            adaptive_gate_px=float(max(0.0, adaptive_gate_px)),
            motion_score=float(max(0.0, motion_score)),
            appearance_similarity=float(appearance_similarity),
            fusion_score=float(max(0.0, fusion_score)),
            jitter_px=float(max(0.0, self._jitter_ema)),
            temperature=float(max(0.0, self._temperature)),
            hit_counter=int(self._hit_counter),
            used_bridge=bool(used_bridge),
            reject_streak=int(self._reject_streak),
        )

    def update(
        self,
        center: Tuple[float, float],
        confidence: float,
        appearance_embedding: Optional[np.ndarray] = None,
        timestamp_s: Optional[float] = None,
    ) -> AppearanceMotionFusionResult:
        conf = float(np.clip(_safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = _safe_float(center[0], default=np.nan)
        cy = _safe_float(center[1], default=np.nan)
        ts = self._resolve_timestamp(timestamp_s)

        if not np.isfinite(cx) or not np.isfinite(cy):
            self._reject_streak += 1
            self._accept_streak = 0
            self._hit_counter = max(0, int(self._hit_counter) - int(max(1, self.config.miss_decay)))
            self._temperature = float(min(2.8, self._temperature + 0.12))
            if self._state is None:
                predicted = (0.0, 0.0)
            else:
                predicted = (float(self._state[0]), float(self._state[1]))
            return self._result(
                center=predicted,
                confidence=float(np.clip(conf * float(self.config.confidence_penalty), 0.0, 1.0)),
                accepted=False,
                refined=False,
                rejected=True,
                reason="invalid_measurement",
                predicted_center=predicted,
                association_distance_px=0.0,
                adaptive_gate_px=float(max(0.0, self.config.base_gate_px)),
                motion_score=1.0,
                appearance_similarity=np.nan,
                fusion_score=1.0,
                used_bridge=False,
            )

        measurement = np.array([float(cx), float(cy)], dtype=np.float64)
        embedding = _normalize_embedding(appearance_embedding)

        if self._state is None:
            self._state = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=np.float64)
            self._prev_timestamp = ts
            self._appearance_ref = embedding
            self._hit_counter = max(1, self._hit_counter)
            self._accept_streak = 1
            self._reject_streak = 0
            self._temperature = 1.0
            return self._result(
                center=(measurement[0], measurement[1]),
                confidence=conf,
                accepted=True,
                refined=False,
                rejected=False,
                reason=None,
                predicted_center=(measurement[0], measurement[1]),
                association_distance_px=0.0,
                adaptive_gate_px=float(max(0.0, self.config.base_gate_px)),
                motion_score=0.0,
                appearance_similarity=1.0 if embedding is not None else np.nan,
                fusion_score=0.0,
                used_bridge=False,
            )

        prev_ts = float(self._prev_timestamp) if self._prev_timestamp is not None else (ts - 1.0 / 30.0)
        dt = float(np.clip(ts - prev_ts, float(self.config.min_dt_s), float(self.config.max_dt_s)))
        self._prev_timestamp = ts

        px = float(self._state[0] + self._state[2] * dt)
        py = float(self._state[1] + self._state[3] * dt)
        predicted = np.array([px, py], dtype=np.float64)
        delta = measurement - predicted
        distance_px = float(np.linalg.norm(delta))
        self._jitter_ema = float(
            np.clip(
                float(self.config.jitter_decay) * float(self._jitter_ema)
                + (1.0 - float(self.config.jitter_decay)) * distance_px,
                0.0,
                128.0,
            )
        )

        gate_px = self._effective_gate(conf)
        max_dist = float(max(0.0, self.config.max_association_distance_px))
        gate_px = float(min(gate_px, max_dist if max_dist > 0 else gate_px))
        if gate_px <= 1e-6:
            gate_px = 1e-6
        motion_score = float(distance_px / gate_px)

        similarity = _cosine_similarity(embedding, self._appearance_ref)
        if similarity is None:
            similarity = np.nan
        if not np.isfinite(similarity):
            app_cost = 0.5
        else:
            app_cost = float(np.clip((1.0 - similarity) * 0.5, 0.0, 1.0))
        motion_cost = float(np.clip(motion_score, 0.0, 2.5))
        app_weight = float(np.clip(self.config.appearance_weight, 0.0, 1.0))
        fusion_score = (1.0 - app_weight) * motion_cost + app_weight * app_cost

        used_bridge = False
        reason = None
        accepted = True
        refined = False

        min_similarity = float(np.clip(self.config.min_similarity, -1.0, 1.0))
        low_conf = conf < float(np.clip(self.config.low_confidence_threshold, 0.0, 1.0))
        high_conf = conf >= float(np.clip(self.config.high_confidence_threshold, 0.0, 1.0))
        bridge_gate = float(gate_px * max(1.0, float(self.config.bridge_gate_multiplier)))
        bridge_similarity = float(np.clip(self.config.bridge_similarity, -1.0, 1.0))

        if distance_px > gate_px:
            if low_conf and distance_px <= bridge_gate:
                used_bridge = True
                if np.isfinite(similarity) and similarity < bridge_similarity:
                    accepted = False
                    reason = "bridge_similarity_too_low"
            else:
                accepted = False
                reason = "distance_gate_exceeded"

        if accepted and np.isfinite(similarity) and similarity < min_similarity:
            if not (low_conf and similarity >= bridge_similarity and distance_px <= bridge_gate):
                accepted = False
                reason = "appearance_similarity_too_low"

        if accepted and fusion_score > 1.05:
            if not (low_conf and distance_px <= bridge_gate):
                accepted = False
                reason = "fusion_score_too_high"

        if accepted and high_conf and motion_score > 1.15:
            accepted = False
            reason = "high_conf_motion_inconsistent"

        if accepted:
            blend = float(np.clip(0.3 + 0.4 * conf + 0.2 * (1.0 - min(1.0, motion_score)), 0.15, 0.95))
            fused = measurement * blend + predicted * (1.0 - blend)
            shift = float(np.linalg.norm(fused - measurement))
            if shift > float(max(0.0, self.config.max_refine_shift_px)):
                accepted = False
                reason = "refine_shift_too_large"
                fused = measurement
            if shift < float(max(0.0, self.config.min_refine_shift_px)):
                fused = measurement
            refined = bool(np.linalg.norm(fused - measurement) >= float(max(0.0, self.config.min_refine_shift_px)))

            meas_vx = (fused[0] - float(self._state[0])) / max(dt, 1e-6)
            meas_vy = (fused[1] - float(self._state[1])) / max(dt, 1e-6)
            vx = float(self.config.velocity_smoothing) * float(self._state[2]) + (1.0 - float(self.config.velocity_smoothing)) * meas_vx
            vy = float(self.config.velocity_smoothing) * float(self._state[3]) + (1.0 - float(self.config.velocity_smoothing)) * meas_vy
            vx, vy = self._clip_velocity(vx, vy)
            self._state = np.array([float(fused[0]), float(fused[1]), vx, vy], dtype=np.float64)

            if embedding is not None:
                if self._appearance_ref is None:
                    self._appearance_ref = embedding
                else:
                    m = float(np.clip(self.config.appearance_momentum, 0.0, 1.0))
                    mixed = m * self._appearance_ref + (1.0 - m) * embedding
                    self._appearance_ref = _normalize_embedding(mixed)

            self._accept_streak += 1
            self._reject_streak = 0
            self._hit_counter = min(int(max(1, self.config.hit_counter_max)), int(self._hit_counter) + 1)
            if self._accept_streak < int(max(1, self.config.accept_hysteresis_hits)):
                conf_out = conf
            else:
                conf_out = float(np.clip(conf * float(self.config.confidence_gain), 0.0, 1.0))
            self._update_temperature(conf_out, motion_score, similarity if np.isfinite(similarity) else np.nan)
            return self._result(
                center=(float(self._state[0]), float(self._state[1])),
                confidence=conf_out,
                accepted=True,
                refined=refined,
                rejected=False,
                reason=None,
                predicted_center=(float(predicted[0]), float(predicted[1])),
                association_distance_px=distance_px,
                adaptive_gate_px=gate_px,
                motion_score=motion_score,
                appearance_similarity=float(similarity) if np.isfinite(similarity) else np.nan,
                fusion_score=fusion_score,
                used_bridge=used_bridge,
            )

        self._accept_streak = 0
        self._reject_streak += 1
        self._hit_counter = max(0, int(self._hit_counter) - int(max(1, self.config.miss_decay)))
        penalized_conf = float(np.clip(conf * float(self.config.confidence_penalty), 0.0, 1.0))

        if self._state is not None:
            pred_vx, pred_vy = self._clip_velocity(float(self._state[2]), float(self._state[3]))
            self._state = np.array([float(predicted[0]), float(predicted[1]), pred_vx, pred_vy], dtype=np.float64)
        self._update_temperature(penalized_conf, motion_score, similarity if np.isfinite(similarity) else np.nan)

        rejected_hard = self._reject_streak >= int(max(1, self.config.reject_hysteresis_misses))
        chosen = predicted if rejected_hard else measurement
        return self._result(
            center=(float(chosen[0]), float(chosen[1])),
            confidence=penalized_conf,
            accepted=False,
            refined=False,
            rejected=True,
            reason=reason or "rejected",
            predicted_center=(float(predicted[0]), float(predicted[1])),
            association_distance_px=distance_px,
            adaptive_gate_px=gate_px,
            motion_score=motion_score,
            appearance_similarity=float(similarity) if np.isfinite(similarity) else np.nan,
            fusion_score=fusion_score,
            used_bridge=used_bridge,
        )
