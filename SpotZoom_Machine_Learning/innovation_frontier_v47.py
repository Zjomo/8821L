"""
SpotZoom innovation frontier v47.

Adaptive-memory-budget association gate inspired by:
- CoTracker3: online/offline split with memory-efficient long-sequence processing.
- SAM2: streaming memory for real-time video loops and per-object predictor updates.
- SAM2RL: reinforcement-learning-inspired memory update policy for improved robustness.
- OpenWFS: asynchronous, hardware-agnostic pipeline style with explicit state synchronization.

This module intentionally keeps a numpy-only runtime dependency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class AdaptiveMemoryBudgetConfig:
    high_confidence_threshold: float = 0.78
    low_confidence_threshold: float = 0.2
    base_gate_px: float = 10.6
    min_gate_px: float = 4.0
    max_gate_px: float = 52.0
    uncertainty_gate_scale: float = 1.32
    velocity_gate_scale: float = 0.24
    budget_pressure_gate_scale: float = 0.42
    reject_ratio_gate_scale: float = 0.66
    continuity_weight: float = 0.3
    appearance_weight: float = 0.25
    memory_weight: float = 0.29
    confidence_weight: float = 0.16
    uncertainty_penalty_weight: float = 0.12
    min_similarity: float = 0.06
    memory_min_similarity: float = 0.12
    min_fusion_score: float = 0.3
    min_fusion_relax: float = 0.1
    max_association_distance_px: float = 50.0
    max_refine_shift_px: float = 22.0
    min_refine_shift_px: float = 0.2
    bridge_gate_multiplier: float = 1.3
    confidence_gain: float = 1.03
    confidence_penalty: float = 0.9
    velocity_smoothing: float = 0.74
    appearance_momentum: float = 0.86
    memory_budget: int = 14
    memory_size: int = 24
    max_memory_age: int = 30
    reject_window: int = 18
    patch_radius: int = 16
    histogram_bins: int = 16
    max_velocity_px_per_s: float = 820.0
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45


@dataclass
class AdaptiveMemoryBudgetResult:
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
    reject_ratio: float
    budget_pressure: float
    memory_size: int
    used_bridge: bool
    used_memory_path: bool


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


def _extract_patch(gray: np.ndarray, center: Tuple[float, float], radius: int) -> Optional[np.ndarray]:
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


class AdaptiveMemoryBudgetAssociationGate:
    """Single-target gate with memory budget control and reject-ratio adaptation."""

    def __init__(self, config: Optional[AdaptiveMemoryBudgetConfig] = None):
        self.config = config or AdaptiveMemoryBudgetConfig()
        self._state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self._appearance_ref: Optional[np.ndarray] = None
        self._prev_timestamp: Optional[float] = None
        self._uncertainty_ema: float = 0.0
        self._memory_bank: Deque[_MemoryEntry] = deque(maxlen=max(8, int(self.config.memory_size)))
        self._recent_rejects: Deque[int] = deque(maxlen=max(6, int(self.config.reject_window)))

    def reset(self) -> None:
        self._state = None
        self._appearance_ref = None
        self._prev_timestamp = None
        self._uncertainty_ema = 0.0
        self._memory_bank.clear()
        self._recent_rejects.clear()

    def _resolve_timestamp(self, timestamp_s: Optional[float]) -> float:
        ts = _safe_float(timestamp_s if timestamp_s is not None else np.nan, default=np.nan)
        if not np.isfinite(ts):
            import time
            ts = float(time.perf_counter())
        return ts

    def _clip_velocity(self, vx: float, vy: float) -> Tuple[float, float]:
        vmax = float(max(0.0, self.config.max_velocity_px_per_s))
        speed = float(np.hypot(vx, vy))
        if vmax <= 0.0 or speed <= vmax or speed <= 1e-9:
            return vx, vy
        scale = vmax / speed
        return float(vx * scale), float(vy * scale)

    def _memory_similarity(
        self,
        center: np.ndarray,
        embedding: Optional[np.ndarray],
    ) -> Tuple[float, int, Optional[np.ndarray]]:
        if not self._memory_bank:
            return 0.0, 9999, None

        best_score = -1.0
        best_age = 9999
        best_center = None
        for entry in self._memory_bank:
            spatial = float(np.exp(-np.linalg.norm(center - entry.center) / max(1.0, self.config.base_gate_px * 2.2)))
            app = _cosine_similarity(embedding, entry.embedding)
            if app is None:
                app = 0.0
            app_term = float(np.clip((app + 1.0) * 0.5, 0.0, 1.0))
            recency = float(np.exp(-0.09 * entry.age))
            score = 0.44 * spatial + 0.36 * app_term + 0.2 * recency
            if score > best_score:
                best_score = score
                best_age = int(entry.age)
                best_center = entry.center.copy()

        if not np.isfinite(best_score):
            best_score = 0.0
        return float(np.clip(best_score, 0.0, 1.0)), best_age, best_center

    def _age_memory(self) -> None:
        for entry in self._memory_bank:
            entry.age += 1

    def _prune_memory(self) -> None:
        budget = max(4, int(self.config.memory_budget))
        target = min(max(8, int(self.config.memory_size)), budget + max(2, budget // 3))
        if len(self._memory_bank) <= target:
            return

        # Keep high-confidence + recent entries first; this imitates a lightweight
        # memory policy without introducing heavy RL dependencies.
        ranked = sorted(
            self._memory_bank,
            key=lambda e: (0.62 * float(np.clip(e.confidence, 0.0, 1.0)) + 0.38 * float(np.exp(-0.09 * e.age))),
            reverse=True,
        )
        kept = ranked[:target]
        self._memory_bank = deque(kept, maxlen=max(8, int(self.config.memory_size)))

    def _add_memory(self, center: np.ndarray, embedding: Optional[np.ndarray], confidence: float) -> None:
        self._memory_bank.append(
            _MemoryEntry(
                center=np.asarray(center, dtype=np.float64).reshape(2),
                embedding=embedding,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                age=0,
            )
        )
        self._prune_memory()

    def _reject_ratio(self) -> float:
        if not self._recent_rejects:
            return 0.0
        return float(np.clip(np.mean(np.asarray(self._recent_rejects, dtype=np.float64)), 0.0, 1.0))

    def _budget_pressure(self) -> float:
        budget = max(1, int(self.config.memory_budget))
        pressure = float((len(self._memory_bank) - budget) / float(budget))
        return float(np.clip(pressure, 0.0, 1.0))

    def _estimate_uncertainty(self, distance: float, entropy_score: float, confidence: float) -> float:
        conf_term = float(np.clip(1.0 - confidence, 0.0, 1.0))
        dist_term = float(np.clip(distance / max(1.0, self.config.base_gate_px * 2.0), 0.0, 2.0))
        entropy_term = float(np.clip(entropy_score / 5.0, 0.0, 1.0))
        uncertainty = 0.42 * conf_term + 0.38 * dist_term + 0.2 * entropy_term
        self._uncertainty_ema = 0.78 * self._uncertainty_ema + 0.22 * uncertainty
        return float(np.clip(self._uncertainty_ema, 0.0, 1.2))

    def _effective_gate(self, confidence: float, speed: float, uncertainty: float) -> Tuple[float, float, float]:
        reject_ratio = self._reject_ratio()
        pressure = self._budget_pressure()
        gate = float(self.config.base_gate_px)
        gate += float(max(0.0, self.config.uncertainty_gate_scale)) * float(max(0.0, uncertainty))
        gate += float(max(0.0, self.config.velocity_gate_scale)) * float(max(0.0, speed)) * 0.01
        gate *= 1.0 + float(max(0.0, self.config.budget_pressure_gate_scale)) * pressure
        gate *= 1.0 + float(max(0.0, self.config.reject_ratio_gate_scale)) * reject_ratio

        if confidence <= float(self.config.low_confidence_threshold):
            gate *= float(max(1.0, self.config.bridge_gate_multiplier))

        gate = float(np.clip(gate, float(self.config.min_gate_px), float(self.config.max_gate_px)))
        gate = min(gate, float(max(0.0, self.config.max_association_distance_px)))
        return gate, reject_ratio, pressure

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
        reject_ratio: float,
        budget_pressure: float,
        used_bridge: bool,
        used_memory_path: bool,
    ) -> AdaptiveMemoryBudgetResult:
        return AdaptiveMemoryBudgetResult(
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
            uncertainty_score=float(max(0.0, uncertainty_score)),
            reject_ratio=float(np.clip(reject_ratio, 0.0, 1.0)),
            budget_pressure=float(np.clip(budget_pressure, 0.0, 1.0)),
            memory_size=int(len(self._memory_bank)),
            used_bridge=bool(used_bridge),
            used_memory_path=bool(used_memory_path),
        )

    def update(
        self,
        *,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> AdaptiveMemoryBudgetResult:
        cx = _safe_float(center[0], default=np.nan)
        cy = _safe_float(center[1], default=np.nan)
        conf = float(np.clip(_safe_float(confidence, default=0.0), 0.0, 1.0))

        if not np.isfinite(cx) or not np.isfinite(cy):
            self._recent_rejects.append(1)
            return self._make_result(
                center=(0.0, 0.0),
                confidence=conf * float(self.config.confidence_penalty),
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
                uncertainty_score=1.0,
                reject_ratio=self._reject_ratio(),
                budget_pressure=self._budget_pressure(),
                used_bridge=False,
                used_memory_path=False,
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

        self._age_memory()

        if self._state is None:
            self._state = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
            self._appearance_ref = embedding
            self._prev_timestamp = ts
            self._add_memory(np.array([cx, cy], dtype=np.float64), embedding, conf)
            self._recent_rejects.append(0)
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
                reject_ratio=self._reject_ratio(),
                budget_pressure=self._budget_pressure(),
                used_bridge=False,
                used_memory_path=False,
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

        memory_similarity, memory_age, memory_center = self._memory_similarity(
            center=np.array([cx, cy], dtype=np.float64),
            embedding=embedding,
        )
        uncertainty_score = self._estimate_uncertainty(dist, entropy_score, conf)
        gate, reject_ratio, pressure = self._effective_gate(conf, speed, uncertainty_score)

        continuity = float(np.clip(1.0 - dist / max(gate, 1e-6), 0.0, 1.0))
        app_term = float(np.clip((appearance_similarity + 1.0) * 0.5, 0.0, 1.0))
        fusion = (
            float(max(0.0, self.config.continuity_weight)) * continuity
            + float(max(0.0, self.config.appearance_weight)) * app_term
            + float(max(0.0, self.config.memory_weight)) * memory_similarity
            + float(max(0.0, self.config.confidence_weight)) * conf
            - float(max(0.0, self.config.uncertainty_penalty_weight)) * float(np.clip(uncertainty_score, 0.0, 1.0))
        )
        denom = (
            float(max(0.0, self.config.continuity_weight))
            + float(max(0.0, self.config.appearance_weight))
            + float(max(0.0, self.config.memory_weight))
            + float(max(0.0, self.config.confidence_weight))
            + float(max(0.0, self.config.uncertainty_penalty_weight))
        )
        if denom > 1e-6:
            fusion /= denom
        fusion = float(np.clip(fusion, 0.0, 1.0))

        dynamic_fusion_floor = float(
            np.clip(
                float(self.config.min_fusion_score)
                - float(max(0.0, self.config.min_fusion_relax)) * reject_ratio
                + 0.03 * pressure,
                0.12,
                0.9,
            )
        )

        used_bridge = dist > gate and dist <= gate * float(max(1.0, self.config.bridge_gate_multiplier))
        accept_by_measurement = (
            dist <= gate * (float(max(1.0, self.config.bridge_gate_multiplier)) if conf <= self.config.low_confidence_threshold else 1.0)
            and fusion >= dynamic_fusion_floor
            and appearance_similarity >= float(self.config.min_similarity)
        )

        use_memory_path = (
            memory_center is not None
            and memory_similarity >= float(max(0.0, self.config.memory_min_similarity))
            and memory_age <= int(max(0, self.config.max_memory_age))
            and dist <= gate * 1.2
            and (not accept_by_measurement)
        )

        accepted = accept_by_measurement or use_memory_path
        if conf >= float(self.config.high_confidence_threshold):
            accepted = accepted or (
                dist <= gate
                and fusion >= max(0.0, dynamic_fusion_floor - 0.05)
            )

        if not accepted:
            self._recent_rejects.append(1)
            out_conf = float(np.clip(conf * float(self.config.confidence_penalty), 0.0, 1.0))
            reason = "association_distance_exceeded" if dist > gate * float(max(1.0, self.config.bridge_gate_multiplier)) else "fusion_or_similarity_too_low"
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
                reject_ratio=reject_ratio,
                budget_pressure=pressure,
                used_bridge=used_bridge,
                used_memory_path=False,
            )

        rx = cx
        ry = cy
        if use_memory_path and memory_center is not None:
            rx = float(0.62 * cx + 0.38 * float(memory_center[0]))
            ry = float(0.62 * cy + 0.38 * float(memory_center[1]))

        alpha = float(np.clip(0.42 + 0.24 * continuity + 0.16 * memory_similarity + 0.1 * conf, 0.35, 0.92))
        rx = float(px + alpha * (rx - px))
        ry = float(py + alpha * (ry - py))

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

        if embedding is not None:
            if self._appearance_ref is None:
                self._appearance_ref = embedding
            else:
                momentum = float(np.clip(self.config.appearance_momentum, 0.0, 1.0))
                merged = momentum * self._appearance_ref + (1.0 - momentum) * embedding
                self._appearance_ref = _normalize(merged)

        self._add_memory(np.array([rx, ry], dtype=np.float64), embedding, conf)
        self._recent_rejects.append(0)

        conf_factor = float(np.clip(0.93 + 0.05 * memory_similarity - 0.04 * float(np.clip(uncertainty_score, 0.0, 1.0)), 0.78, 1.08))
        out_conf = float(np.clip(conf * float(self.config.confidence_gain) * conf_factor, 0.0, 1.0))
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
            reject_ratio=reject_ratio,
            budget_pressure=pressure,
            used_bridge=used_bridge,
            used_memory_path=use_memory_path,
        )
