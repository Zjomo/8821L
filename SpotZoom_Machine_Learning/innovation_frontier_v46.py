"""
SpotZoom innovation frontier v46.

Memory-uncertainty co-association gate inspired by:
- CoTracker3: online long-horizon point tracking with chunked state updates.
- SAM2: streaming memory state in video predictor loops.
- ByteTrack: low-confidence bridge association under noisy observations.
- Norfair: lightweight distance-based tracking loops for real-time usage.

This module keeps a numpy-only runtime dependency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class MemoryUncertaintyAssociationConfig:
    high_confidence_threshold: float = 0.78
    low_confidence_threshold: float = 0.18
    base_gate_px: float = 10.2
    min_gate_px: float = 4.0
    max_gate_px: float = 48.0
    uncertainty_gate_scale: float = 1.5
    velocity_gate_scale: float = 0.2
    memory_gate_scale: float = 1.15
    appearance_weight: float = 0.22
    continuity_weight: float = 0.3
    memory_weight: float = 0.32
    confidence_weight: float = 0.16
    uncertainty_penalty_weight: float = 0.12
    min_similarity: float = 0.08
    memory_min_similarity: float = 0.1
    min_fusion_score: float = 0.28
    max_association_distance_px: float = 46.0
    max_refine_shift_px: float = 22.0
    min_refine_shift_px: float = 0.2
    bridge_gate_multiplier: float = 1.34
    confidence_gain: float = 1.02
    confidence_penalty: float = 0.9
    velocity_smoothing: float = 0.72
    appearance_momentum: float = 0.86
    uncertainty_decay: float = 0.82
    temperature_decay: float = 0.86
    max_velocity_px_per_s: float = 780.0
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45
    history_size: int = 24
    residual_window: int = 14
    memory_size: int = 18
    max_memory_age: int = 28
    patch_radius: int = 16
    histogram_bins: int = 16
    hit_counter_max: int = 40
    miss_decay: int = 1
    max_bridge_reject_streak: int = 3


@dataclass
class MemoryUncertaintyAssociationResult:
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
    memory_similarity: float
    fusion_score: float
    uncertainty_score: float
    entropy_score: float
    temperature: float
    hit_counter: int
    reject_streak: int
    memory_age: int
    used_bridge: bool
    memory_reacquire_used: bool


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


def _patch_entropy(patch: Optional[np.ndarray], bins: int) -> float:
    if patch is None or patch.size == 0:
        return 0.0
    n_bins = max(8, int(bins))
    hist = np.histogram(
        np.clip(patch, 0.0, 255.0),
        bins=n_bins,
        range=(0.0, 255.0),
        density=True,
    )[0]
    hist = np.clip(hist.astype(np.float64), 1e-12, None)
    entropy = float(-np.sum(hist * np.log(hist)))
    if not np.isfinite(entropy):
        return 0.0
    return float(max(0.0, entropy))


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


class MemoryUncertaintyAssociationGate:
    """Single-target association gate with streaming memory and uncertainty adaptation."""

    def __init__(self, config: Optional[MemoryUncertaintyAssociationConfig] = None):
        self.config = config or MemoryUncertaintyAssociationConfig()
        self._state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self._appearance_ref: Optional[np.ndarray] = None
        self._clock: float = 0.0
        self._prev_timestamp: Optional[float] = None
        self._temperature: float = 1.0
        self._uncertainty_ema: float = 0.0
        self._hit_counter: int = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak: int = 0
        self._reject_streak: int = 0
        self._center_history: Deque[np.ndarray] = deque(maxlen=max(6, int(self.config.history_size)))
        self._residual_history: Deque[float] = deque(maxlen=max(6, int(self.config.residual_window)))
        self._memory_bank: Deque[_MemoryEntry] = deque(maxlen=max(4, int(self.config.memory_size)))

    def reset(self) -> None:
        self._state = None
        self._appearance_ref = None
        self._clock = 0.0
        self._prev_timestamp = None
        self._temperature = 1.0
        self._uncertainty_ema = 0.0
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak = 0
        self._reject_streak = 0
        self._center_history.clear()
        self._residual_history.clear()
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

    def _estimate_uncertainty(self, residual_px: float, entropy_score: float, confidence: float) -> float:
        self._residual_history.append(float(max(0.0, residual_px)))
        residuals = np.asarray(list(self._residual_history), dtype=np.float64)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            robust_sigma = 0.0
        else:
            med = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - med)))
            robust_sigma = float(max(0.0, 1.4826 * mad))
        base = max(1e-6, float(self.config.base_gate_px))
        sigma_term = float(np.clip(robust_sigma / (base * 2.2), 0.0, 1.0))
        entropy_term = float(np.clip(entropy_score / 4.0, 0.0, 1.0))
        conf_term = float(np.clip(1.0 - confidence, 0.0, 1.0))
        raw = 0.45 * sigma_term + 0.33 * entropy_term + 0.22 * conf_term
        decay = float(np.clip(self.config.uncertainty_decay, 0.0, 1.0))
        self._uncertainty_ema = float(decay * self._uncertainty_ema + (1.0 - decay) * raw)
        return float(np.clip(self._uncertainty_ema, 0.0, 1.0))

    def _memory_match(
        self,
        center: np.ndarray,
        embedding: Optional[np.ndarray],
    ) -> Tuple[float, int, Optional[np.ndarray]]:
        if len(self._memory_bank) == 0:
            return 0.0, 0, None
        best_score = -1.0
        best_age = 0
        best_center = None
        best_distance = float("inf")
        for entry in self._memory_bank:
            app_sim = _cosine_similarity(embedding, entry.embedding)
            if app_sim is None:
                app_sim = 0.0
            spatial = float(np.hypot(*(center - entry.center)))
            spatial_term = float(np.clip(1.0 - spatial / max(1.0, float(self.config.max_association_distance_px)), 0.0, 1.0))
            age_ratio = float(np.clip(entry.age / max(1.0, float(self.config.max_memory_age)), 0.0, 1.0))
            age_term = float(np.clip(1.0 - age_ratio, 0.0, 1.0))
            score = 0.58 * float(np.clip((app_sim + 1.0) * 0.5, 0.0, 1.0)) + 0.27 * spatial_term + 0.15 * age_term
            if score > best_score or (abs(score - best_score) <= 1e-6 and spatial < best_distance):
                best_score = score
                best_age = int(entry.age)
                best_center = entry.center
                best_distance = spatial
        if best_score < 0.0:
            return 0.0, 0, None
        return float(np.clip(best_score, 0.0, 1.0)), best_age, best_center

    def _effective_gate(
        self,
        confidence: float,
        speed_px_s: float,
        uncertainty_score: float,
        memory_similarity: float,
    ) -> float:
        conf = float(np.clip(confidence, 0.0, 1.0))
        gate = float(max(self.config.min_gate_px, self.config.base_gate_px))
        if conf < float(self.config.low_confidence_threshold):
            gate *= 1.08
        elif conf >= float(self.config.high_confidence_threshold):
            gate *= 0.95
        gate += float(max(0.0, self.config.velocity_gate_scale)) * float(max(0.0, speed_px_s))
        gate += float(max(0.0, self.config.uncertainty_gate_scale)) * float(np.clip(uncertainty_score, 0.0, 1.0)) * float(
            max(1e-6, self.config.base_gate_px)
        )
        gate += float(max(0.0, self.config.memory_gate_scale)) * float(
            np.clip(1.0 - memory_similarity, 0.0, 1.0)
        ) * float(max(1e-6, self.config.base_gate_px)) * 0.4
        gate *= float(np.clip(self._temperature, 0.75, 2.95))
        gate = float(np.clip(gate, float(self.config.min_gate_px), float(self.config.max_gate_px)))
        gate = min(gate, float(max(self.config.min_gate_px, self.config.max_association_distance_px)))
        return float(gate)

    def _update_temperature(self, confidence: float, dist: float, gate: float, uncertainty_score: float) -> None:
        conf = float(np.clip(confidence, 0.0, 1.0))
        dist_ratio = float(np.clip(dist / max(gate, 1e-6), 0.0, 2.0))
        target = 1.0 + 0.2 * (1.0 - conf) + 0.26 * dist_ratio + 0.3 * float(np.clip(uncertainty_score, 0.0, 1.0))
        blend = float(np.clip(1.0 - float(self.config.temperature_decay), 0.05, 0.45))
        self._temperature = float(np.clip((1.0 - blend) * self._temperature + blend * target, 0.65, 3.15))

    def _age_memory_bank(self) -> None:
        if len(self._memory_bank) == 0:
            return
        max_age = max(1, int(self.config.max_memory_age))
        new_bank: Deque[_MemoryEntry] = deque(maxlen=self._memory_bank.maxlen)
        for entry in self._memory_bank:
            next_age = int(entry.age) + 1
            if next_age <= max_age:
                new_bank.append(
                    _MemoryEntry(
                        center=entry.center.copy(),
                        embedding=entry.embedding,
                        confidence=float(entry.confidence),
                        age=next_age,
                    )
                )
        self._memory_bank = new_bank

    def _add_memory(self, center: np.ndarray, embedding: Optional[np.ndarray], confidence: float) -> None:
        self._memory_bank.appendleft(
            _MemoryEntry(
                center=center.astype(np.float64),
                embedding=embedding,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                age=0,
            )
        )

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
        memory_similarity: float,
        fusion_score: float,
        uncertainty_score: float,
        entropy_score: float,
        memory_age: int,
        used_bridge: bool,
        memory_reacquire_used: bool,
    ) -> MemoryUncertaintyAssociationResult:
        return MemoryUncertaintyAssociationResult(
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
            memory_similarity=float(np.clip(memory_similarity, 0.0, 1.0)),
            fusion_score=float(np.clip(fusion_score, 0.0, 1.0)),
            uncertainty_score=float(np.clip(uncertainty_score, 0.0, 1.0)),
            entropy_score=float(max(0.0, entropy_score)),
            temperature=float(max(0.0, self._temperature)),
            hit_counter=int(self._hit_counter),
            reject_streak=int(self._reject_streak),
            memory_age=int(max(0, memory_age)),
            used_bridge=bool(used_bridge),
            memory_reacquire_used=bool(memory_reacquire_used),
        )

    def update(
        self,
        *,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> MemoryUncertaintyAssociationResult:
        cx = _safe_float(center[0], default=np.nan)
        cy = _safe_float(center[1], default=np.nan)
        conf = float(np.clip(_safe_float(confidence, default=0.0), 0.0, 1.0))
        if not np.isfinite(cx) or not np.isfinite(cy):
            return self._make_result(
                center=(0.0, 0.0),
                confidence=0.0,
                accepted=False,
                refined=False,
                rejected=True,
                reason="invalid_center",
                predicted_center=(0.0, 0.0),
                association_distance_px=0.0,
                adaptive_gate_px=float(max(0.0, self.config.base_gate_px)),
                continuity_score=0.0,
                appearance_similarity=0.0,
                memory_similarity=0.0,
                fusion_score=0.0,
                uncertainty_score=self._uncertainty_ema,
                entropy_score=0.0,
                memory_age=0,
                used_bridge=False,
                memory_reacquire_used=False,
            )

        ts = self._resolve_timestamp(timestamp_s)
        gray = _to_gray(frame)
        patch = None
        if gray is not None:
            patch = _extract_patch(gray, center=(cx, cy), radius=int(self.config.patch_radius))
        entropy_score = _patch_entropy(patch, int(self.config.histogram_bins))
        embedding = build_patch_embedding(
            frame,
            (cx, cy),
            radius=int(self.config.patch_radius),
            bins=int(self.config.histogram_bins),
        )

        self._age_memory_bank()

        if self._state is None:
            self._state = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
            self._appearance_ref = embedding
            self._prev_timestamp = ts
            seed_center = np.array([cx, cy], dtype=np.float64)
            self._center_history.append(seed_center.copy())
            self._add_memory(seed_center, embedding, conf)
            return self._make_result(
                center=(cx, cy),
                confidence=conf,
                accepted=True,
                refined=False,
                rejected=False,
                reason=None,
                predicted_center=(cx, cy),
                association_distance_px=0.0,
                adaptive_gate_px=float(max(0.0, self.config.base_gate_px)),
                continuity_score=1.0,
                appearance_similarity=1.0 if embedding is not None else 0.0,
                memory_similarity=1.0,
                fusion_score=1.0,
                uncertainty_score=0.0,
                entropy_score=entropy_score,
                memory_age=0,
                used_bridge=False,
                memory_reacquire_used=False,
            )

        prev_ts = self._prev_timestamp if self._prev_timestamp is not None else (ts - 1.0 / 30.0)
        dt = float(np.clip(ts - prev_ts, float(self.config.min_dt_s), float(self.config.max_dt_s)))
        self._prev_timestamp = ts

        px = float(self._state[0] + self._state[2] * dt)
        py = float(self._state[1] + self._state[3] * dt)
        dx = float(cx - px)
        dy = float(cy - py)
        dist = float(np.hypot(dx, dy))
        speed = float(np.hypot(float(self._state[2]), float(self._state[3])))

        appearance_similarity = _cosine_similarity(embedding, self._appearance_ref)
        if appearance_similarity is None:
            appearance_similarity = 0.0

        memory_similarity, memory_age, memory_center = self._memory_match(
            center=np.array([cx, cy], dtype=np.float64),
            embedding=embedding,
        )
        uncertainty_score = self._estimate_uncertainty(dist, entropy_score, conf)
        gate = self._effective_gate(conf, speed, uncertainty_score, memory_similarity)
        continuity = float(np.clip(1.0 - dist / max(gate, 1e-6), 0.0, 1.0))

        app_term = float(np.clip((appearance_similarity + 1.0) * 0.5, 0.0, 1.0))
        uncertainty_penalty = float(np.clip(uncertainty_score, 0.0, 1.0))
        fusion = (
            float(max(0.0, self.config.continuity_weight)) * continuity
            + float(max(0.0, self.config.appearance_weight)) * app_term
            + float(max(0.0, self.config.memory_weight)) * memory_similarity
            + float(max(0.0, self.config.confidence_weight)) * conf
            - float(max(0.0, self.config.uncertainty_penalty_weight)) * uncertainty_penalty
        )
        weight_sum = (
            float(max(0.0, self.config.continuity_weight))
            + float(max(0.0, self.config.appearance_weight))
            + float(max(0.0, self.config.memory_weight))
            + float(max(0.0, self.config.confidence_weight))
            + float(max(0.0, self.config.uncertainty_penalty_weight))
        )
        if weight_sum > 1e-6:
            fusion /= weight_sum
        fusion = float(np.clip(fusion, 0.0, 1.0))

        bridge_gate = gate
        used_bridge = False
        if conf <= float(self.config.low_confidence_threshold) or self._reject_streak > 0:
            bridge_gate = gate * float(max(1.0, self.config.bridge_gate_multiplier))
            used_bridge = dist > gate and dist <= bridge_gate

        need_memory_reacquire = (
            conf <= float(self.config.low_confidence_threshold)
            or self._reject_streak > 0
            or dist > gate * 0.65
        )
        memory_reacquire = (
            memory_center is not None
            and need_memory_reacquire
            and memory_similarity >= float(max(0.0, self.config.memory_min_similarity))
            and memory_age <= int(max(0, self.config.max_memory_age))
            and dist <= bridge_gate * 1.15
        )
        accept_by_measurement = (
            dist <= bridge_gate
            and fusion >= float(self.config.min_fusion_score)
            and appearance_similarity >= float(self.config.min_similarity)
        )
        accept = accept_by_measurement or memory_reacquire
        if conf >= float(self.config.high_confidence_threshold):
            accept = accept or (
                dist <= gate
                and fusion >= max(0.0, float(self.config.min_fusion_score) - 0.05)
            )

        if not accept:
            self._reject_streak += 1
            self._accept_streak = 0
            self._hit_counter = max(0, self._hit_counter - int(max(1, self.config.miss_decay)))
            self._update_temperature(conf, dist, gate, uncertainty_score)
            out_conf = float(np.clip(conf * float(self.config.confidence_penalty), 0.0, 1.0))
            reason = "association_distance_exceeded" if dist > bridge_gate else "fusion_or_similarity_too_low"
            return self._make_result(
                center=(cx, cy),
                confidence=out_conf,
                accepted=False,
                refined=False,
                rejected=True,
                reason=reason,
                predicted_center=(px, py),
                association_distance_px=dist,
                adaptive_gate_px=gate,
                continuity_score=continuity,
                appearance_similarity=appearance_similarity,
                memory_similarity=memory_similarity,
                fusion_score=fusion,
                uncertainty_score=uncertainty_score,
                entropy_score=entropy_score,
                memory_age=memory_age,
                used_bridge=used_bridge,
                memory_reacquire_used=False,
            )

        candidate_x = cx
        candidate_y = cy
        memory_reacquire_used = False
        if memory_reacquire and memory_center is not None:
            candidate_x = float(0.65 * cx + 0.35 * memory_center[0])
            candidate_y = float(0.65 * cy + 0.35 * memory_center[1])
            memory_reacquire_used = True

        alpha = float(np.clip(0.4 + 0.28 * continuity + 0.18 * memory_similarity + 0.08 * conf, 0.34, 0.93))
        rx = float(px + alpha * (candidate_x - px))
        ry = float(py + alpha * (candidate_y - py))
        shift = float(np.hypot(rx - cx, ry - cy))
        if shift < float(max(0.0, self.config.min_refine_shift_px)):
            rx, ry = cx, cy
        max_ref = float(max(0.0, self.config.max_refine_shift_px))
        step = float(np.hypot(rx - px, ry - py))
        if step > max_ref > 0.0:
            scale = max_ref / max(step, 1e-6)
            rx = float(px + (rx - px) * scale)
            ry = float(py + (ry - py) * scale)

        new_vx_raw = (rx - float(self._state[0])) / max(dt, 1e-6)
        new_vy_raw = (ry - float(self._state[1])) / max(dt, 1e-6)
        prev_vx = float(self._state[2])
        prev_vy = float(self._state[3])
        vel_smooth = float(np.clip(self.config.velocity_smoothing, 0.0, 1.0))
        vx = vel_smooth * prev_vx + (1.0 - vel_smooth) * new_vx_raw
        vy = vel_smooth * prev_vy + (1.0 - vel_smooth) * new_vy_raw
        vx, vy = self._clip_velocity(vx, vy)

        self._state = np.array([rx, ry, vx, vy], dtype=np.float64)
        refined_center = np.array([rx, ry], dtype=np.float64)
        self._center_history.append(refined_center.copy())

        if embedding is not None:
            if self._appearance_ref is None:
                self._appearance_ref = embedding
            else:
                momentum = float(np.clip(self.config.appearance_momentum, 0.0, 1.0))
                merged = momentum * self._appearance_ref + (1.0 - momentum) * embedding
                self._appearance_ref = _normalize(merged)

        self._add_memory(refined_center, embedding, conf)
        self._reject_streak = 0
        self._accept_streak += 1
        self._hit_counter = min(int(self.config.hit_counter_max), self._hit_counter + 1)
        self._update_temperature(conf, dist, gate, uncertainty_score)

        confidence_factor = float(np.clip(0.92 + 0.06 * memory_similarity - 0.05 * uncertainty_penalty, 0.78, 1.06))
        out_conf = float(np.clip(conf * float(self.config.confidence_gain) * confidence_factor, 0.0, 1.0))
        refined_flag = float(np.hypot(rx - cx, ry - cy)) > float(max(0.0, self.config.min_refine_shift_px))
        return self._make_result(
            center=(rx, ry),
            confidence=out_conf,
            accepted=True,
            refined=refined_flag,
            rejected=False,
            reason=None,
            predicted_center=(px, py),
            association_distance_px=dist,
            adaptive_gate_px=gate,
            continuity_score=continuity,
            appearance_similarity=appearance_similarity,
            memory_similarity=memory_similarity,
            fusion_score=fusion,
            uncertainty_score=uncertainty_score,
            entropy_score=entropy_score,
            memory_age=memory_age,
            used_bridge=used_bridge,
            memory_reacquire_used=memory_reacquire_used,
        )
