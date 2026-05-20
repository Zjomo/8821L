"""
SpotZoom innovation frontier v44.

Occlusion-aware long-horizon association gate inspired by:
- TAPNext++ / TAPIR: long-range point tracking, occlusion handling, re-detection.
- SAM2 video predictor: persistent inference state / memory over video.
- ByteTrack: low-confidence bridge association.
- Norfair: lightweight real-time association with motion + appearance cues.

This module keeps a numpy-only runtime dependency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class OcclusionAwareAssociationConfig:
    high_confidence_threshold: float = 0.75
    low_confidence_threshold: float = 0.2
    min_bridge_confidence: float = 0.08
    base_gate_px: float = 10.5
    min_gate_px: float = 4.0
    max_gate_px: float = 52.0
    jitter_gate_scale: float = 1.15
    velocity_gate_scale: float = 0.22
    occlusion_gate_scale: float = 1.4
    memory_gate_scale: float = 1.25
    appearance_weight: float = 0.34
    continuity_weight: float = 0.48
    confidence_weight: float = 0.18
    min_similarity: float = 0.1
    min_fusion_score: float = 0.28
    reid_similarity_threshold: float = 0.42
    max_reacquire_age: int = 12
    max_association_distance_px: float = 42.0
    max_refine_shift_px: float = 22.0
    min_refine_shift_px: float = 0.2
    bridge_gate_multiplier: float = 1.35
    confidence_gain: float = 1.02
    confidence_penalty: float = 0.9
    velocity_smoothing: float = 0.72
    appearance_momentum: float = 0.86
    jitter_decay: float = 0.84
    occlusion_decay: float = 0.82
    texture_decay: float = 0.88
    temperature_decay: float = 0.86
    max_velocity_px_per_s: float = 780.0
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45
    history_size: int = 24
    memory_size: int = 16
    patch_radius: int = 16
    histogram_bins: int = 16
    hit_counter_max: int = 40
    miss_decay: int = 1
    max_bridge_reject_streak: int = 3


@dataclass
class OcclusionAwareAssociationResult:
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
    occlusion_prob: float
    temperature: float
    hit_counter: int
    reject_streak: int
    memory_age: int
    used_bridge: bool
    reacquire_used: bool


@dataclass
class _MemoryEntry:
    center: np.ndarray
    embedding: Optional[np.ndarray]
    confidence: float
    age: int


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
        gray = np.mean(arr[..., :3].astype(np.float32), axis=2)
    else:
        return None
    if gray.size == 0:
        return None
    return gray


def _extract_patch(
    gray: np.ndarray,
    center: Tuple[float, float],
    radius: int,
) -> Optional[np.ndarray]:
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
    return np.asarray(patch, dtype=np.float32)


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
    patch = _extract_patch(gray, center=center, radius=radius)
    if patch is None:
        return None

    hist_bins = max(8, int(bins))
    hist = np.histogram(
        np.clip(patch, 0.0, 255.0),
        bins=hist_bins,
        range=(0.0, 255.0),
        density=True,
    )[0]
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


class OcclusionAwareAssociationGate:
    """Single-target association gate with occlusion-aware re-detection memory."""

    def __init__(self, config: Optional[OcclusionAwareAssociationConfig] = None):
        self.config = config or OcclusionAwareAssociationConfig()
        self._state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self._appearance_ref: Optional[np.ndarray] = None
        self._clock: float = 0.0
        self._prev_timestamp: Optional[float] = None
        self._jitter_ema: float = 0.0
        self._occlusion_ema: float = 0.0
        self._texture_ema: float = 0.0
        self._temperature: float = 1.0
        self._hit_counter: int = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak: int = 0
        self._reject_streak: int = 0
        self._center_history: Deque[np.ndarray] = deque(maxlen=max(6, int(self.config.history_size)))
        self._memory_bank: Deque[_MemoryEntry] = deque(maxlen=max(4, int(self.config.memory_size)))

    def reset(self) -> None:
        self._state = None
        self._appearance_ref = None
        self._clock = 0.0
        self._prev_timestamp = None
        self._jitter_ema = 0.0
        self._occlusion_ema = 0.0
        self._texture_ema = 0.0
        self._temperature = 1.0
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak = 0
        self._reject_streak = 0
        self._center_history.clear()
        self._memory_bank.clear()

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

    def _update_temperature(self, confidence: float, dist: float, gate: float, occlusion: float) -> None:
        conf = float(np.clip(confidence, 0.0, 1.0))
        dist_ratio = float(np.clip(dist / max(gate, 1e-6), 0.0, 2.0))
        target = 1.0 + 0.28 * (1.0 - conf) + 0.24 * dist_ratio + 0.24 * float(np.clip(occlusion, 0.0, 1.0))
        blend = float(np.clip(1.0 - float(self.config.temperature_decay), 0.05, 0.45))
        self._temperature = float(np.clip((1.0 - blend) * self._temperature + blend * target, 0.65, 3.1))

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
        occlusion_prob: float,
        memory_age: int,
        used_bridge: bool,
        reacquire_used: bool,
    ) -> OcclusionAwareAssociationResult:
        return OcclusionAwareAssociationResult(
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
            occlusion_prob=float(np.clip(occlusion_prob, 0.0, 1.0)),
            temperature=float(max(0.0, self._temperature)),
            hit_counter=int(self._hit_counter),
            reject_streak=int(self._reject_streak),
            memory_age=int(max(0, memory_age)),
            used_bridge=bool(used_bridge),
            reacquire_used=bool(reacquire_used),
        )

    def _bump_memory_age(self) -> None:
        for entry in self._memory_bank:
            entry.age += 1

    def _append_memory(self, center: Tuple[float, float], embedding: Optional[np.ndarray], confidence: float) -> None:
        self._memory_bank.appendleft(
            _MemoryEntry(
                center=np.array([float(center[0]), float(center[1])], dtype=np.float64),
                embedding=embedding,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                age=0,
            )
        )

    def _best_memory_match(self, embedding: Optional[np.ndarray]) -> Tuple[float, int]:
        if embedding is None or not self._memory_bank:
            return 0.0, int(1e9)
        best_sim = -1.0
        best_age = int(1e9)
        max_age = max(1, int(self.config.max_reacquire_age))
        for entry in self._memory_bank:
            if entry.age > max_age:
                continue
            sim = _cosine_similarity(embedding, entry.embedding)
            if sim is None:
                continue
            if sim > best_sim:
                best_sim = float(sim)
                best_age = int(entry.age)
        if best_sim < -0.5:
            return 0.0, int(1e9)
        return float(best_sim), int(best_age)

    def update(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> OcclusionAwareAssociationResult:
        conf = float(np.clip(_safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = _safe_float(center[0], default=np.nan)
        cy = _safe_float(center[1], default=np.nan)
        ts = self._resolve_timestamp(timestamp_s)
        self._bump_memory_age()

        predicted_center = (0.0, 0.0)
        if self._state is not None:
            predicted_center = (float(self._state[0]), float(self._state[1]))

        if not np.isfinite(cx) or not np.isfinite(cy):
            self._reject_streak += 1
            self._accept_streak = 0
            self._hit_counter = max(0, int(self._hit_counter) - int(max(1, self.config.miss_decay)))
            self._temperature = float(min(2.95, self._temperature + 0.14))
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
                occlusion_prob=self._occlusion_ema,
                memory_age=int(1e9),
                used_bridge=False,
                reacquire_used=False,
            )

        gray = _to_gray(frame)
        patch = _extract_patch(gray, (cx, cy), radius=max(4, int(self.config.patch_radius))) if gray is not None else None
        texture_std = float(np.std(patch)) if patch is not None and patch.size > 0 else 0.0
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
            self._occlusion_ema = 0.0
            self._texture_ema = texture_std
            self._temperature = 1.0
            self._accept_streak = 1
            self._reject_streak = 0
            self._hit_counter = min(int(self.config.hit_counter_max), self._hit_counter + 1)
            self._center_history.append(np.array([cx, cy], dtype=np.float64))
            self._append_memory((cx, cy), embedding, conf)
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
                occlusion_prob=0.0,
                memory_age=0,
                used_bridge=False,
                reacquire_used=False,
            )

        prev_ts = self._prev_timestamp if self._prev_timestamp is not None else (ts - 1.0 / 30.0)
        dt = float(np.clip(ts - prev_ts, float(self.config.min_dt_s), float(self.config.max_dt_s)))
        self._prev_timestamp = float(ts)

        pred_x = float(self._state[0] + self._state[2] * dt)
        pred_y = float(self._state[1] + self._state[3] * dt)
        predicted_center = (pred_x, pred_y)
        speed = float(np.hypot(self._state[2], self._state[3]))

        obs_dx = float(cx - pred_x)
        obs_dy = float(cy - pred_y)
        dist = float(np.hypot(obs_dx, obs_dy))
        assoc_cap = float(max(1e-6, self.config.max_association_distance_px))

        texture_base = float(max(self._texture_ema, 1e-6))
        texture_drop = float(np.clip((texture_base - texture_std) / texture_base, 0.0, 1.0))
        self._texture_ema = float(
            np.clip(
                float(self.config.texture_decay) * self._texture_ema
                + (1.0 - float(self.config.texture_decay)) * texture_std,
                0.0,
                255.0,
            )
        )

        conf_term = float(
            np.clip(
                (float(self.config.low_confidence_threshold) - conf)
                / max(1e-6, float(self.config.low_confidence_threshold)),
                0.0,
                1.0,
            )
        )
        dist_term = float(np.clip((dist - float(self.config.base_gate_px)) / assoc_cap, 0.0, 1.0))
        raw_occ = float(np.clip(0.5 * conf_term + 0.35 * dist_term + 0.15 * texture_drop, 0.0, 1.0))
        self._occlusion_ema = float(
            np.clip(
                float(self.config.occlusion_decay) * self._occlusion_ema
                + (1.0 - float(self.config.occlusion_decay)) * raw_occ,
                0.0,
                1.0,
            )
        )

        gate = (
            float(self.config.base_gate_px)
            + float(self.config.jitter_gate_scale) * self._jitter_ema
            + float(self.config.velocity_gate_scale) * speed * dt
            + float(self.config.occlusion_gate_scale) * self._occlusion_ema * float(self.config.base_gate_px)
        )
        gate = float(np.clip(gate, float(self.config.min_gate_px), float(self.config.max_gate_px)))
        gate = float(min(gate, assoc_cap))
        bridge_gate = gate * float(max(1.0, self.config.bridge_gate_multiplier))

        continuity = float(np.clip(np.exp(-dist / max(gate, 1e-6)), 0.0, 1.0))
        appearance_sim = 0.0
        if embedding is not None and self._appearance_ref is not None:
            sim = _cosine_similarity(embedding, self._appearance_ref)
            if sim is not None:
                appearance_sim = float(sim)
        sim_norm = float(np.clip((appearance_sim + 1.0) * 0.5, 0.0, 1.0))

        memory_sim, memory_age = self._best_memory_match(embedding)
        memory_norm = float(np.clip((memory_sim + 1.0) * 0.5, 0.0, 1.0))
        best_sim = max(sim_norm, memory_norm)

        conf_norm = float(
            np.clip(
                (conf - float(self.config.low_confidence_threshold))
                / max(1e-6, 1.0 - float(self.config.low_confidence_threshold)),
                0.0,
                1.0,
            )
        )
        fusion_score = (
            float(self.config.continuity_weight) * continuity
            + float(self.config.appearance_weight) * best_sim
            + float(self.config.confidence_weight) * conf_norm
        )
        fusion_score = float(np.clip(fusion_score, 0.0, 1.0))

        low_conf = float(self.config.low_confidence_threshold)
        min_bridge_conf = float(self.config.min_bridge_confidence)
        base_accept = dist <= gate and conf >= low_conf and fusion_score >= float(self.config.min_fusion_score)
        bridge_accept = (
            conf < low_conf
            and conf >= min_bridge_conf
            and dist <= bridge_gate
            and appearance_sim >= float(self.config.min_similarity)
            and fusion_score >= float(self.config.min_fusion_score) * 0.9
            and self._reject_streak <= int(max(0, self.config.max_bridge_reject_streak))
        )
        reacquire_gate = min(
            assoc_cap,
            gate * float(max(1.0, self.config.memory_gate_scale)) + self._occlusion_ema * 8.0,
        )
        reacquire_accept = (
            not base_accept
            and memory_sim >= float(self.config.reid_similarity_threshold)
            and memory_age <= int(max(1, self.config.max_reacquire_age))
            and dist <= reacquire_gate
            and conf >= min_bridge_conf
            and fusion_score >= float(self.config.min_fusion_score) * 0.78
        )

        accepted = bool(base_accept or bridge_accept or reacquire_accept)
        used_bridge = bool(bridge_accept)
        reacquire_used = bool(reacquire_accept)
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
            self._update_temperature(conf, dist, gate, self._occlusion_ema)
            reason = "association_distance_exceeded" if dist > max(bridge_gate, reacquire_gate) else "fusion_score_too_low"
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
                continuity_score=continuity,
                appearance_similarity=appearance_sim,
                fusion_score=fusion_score,
                occlusion_prob=self._occlusion_ema,
                memory_age=memory_age,
                used_bridge=False,
                reacquire_used=False,
            )

        blend = 0.28 + 0.5 * continuity + 0.14 * conf_norm + 0.08 * best_sim
        if used_bridge:
            blend *= 0.8
        if reacquire_used:
            blend *= 0.75
        blend = float(np.clip(blend, 0.16, 1.0))

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
                float(self.config.jitter_decay) * self._jitter_ema
                + (1.0 - float(self.config.jitter_decay)) * dist,
                0.0,
                assoc_cap,
            )
        )
        self._center_history.append(np.array([ref_x, ref_y], dtype=np.float64))
        self._reject_streak = 0
        self._accept_streak += 1
        self._hit_counter = min(int(self.config.hit_counter_max), self._hit_counter + 1)
        self._update_temperature(conf, dist, gate, self._occlusion_ema)

        if embedding is not None:
            if self._appearance_ref is None:
                self._appearance_ref = embedding
            else:
                momentum = float(np.clip(self.config.appearance_momentum, 0.0, 1.0))
                merged = momentum * self._appearance_ref + (1.0 - momentum) * embedding
                self._appearance_ref = _normalize(merged)
        self._append_memory((ref_x, ref_y), embedding, conf)

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
            continuity_score=continuity,
            appearance_similarity=appearance_sim,
            fusion_score=fusion_score,
            occlusion_prob=self._occlusion_ema,
            memory_age=memory_age,
            used_bridge=used_bridge,
            reacquire_used=reacquire_used,
        )
