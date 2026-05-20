"""
SpotZoom innovation frontier v51.

Observation-centric memory-control association gate inspired by:
- OC-SORT: observation-centric re-update under occlusion/re-detection gaps.
- SAM2RL / MA-SAM2: selective memory update policy and memory-budget control.
- Efficient-SAM2 style sparse memory recall to keep runtime predictable.

This module intentionally keeps a numpy-only runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ObservationCentricMemoryConfig:
    high_confidence_threshold: float = 0.72
    low_confidence_threshold: float = 0.16
    base_gate_px: float = 11.2
    min_gate_px: float = 4.0
    max_gate_px: float = 60.0
    velocity_gate_scale: float = 0.2
    uncertainty_gate_scale: float = 0.95
    occlusion_gate_scale: float = 0.85
    max_association_distance_px: float = 56.0
    max_refine_shift_px: float = 24.0
    min_similarity: float = 0.05
    min_fusion_score: float = 0.24
    distance_weight: float = 0.52
    memory_weight: float = 0.28
    motion_weight: float = 0.2
    confidence_gain: float = 1.02
    confidence_penalty: float = 0.88
    memory_budget: int = 16
    memory_retrieval_topk: int = 4
    memory_min_confidence: float = 0.24
    memory_min_novelty: float = 0.08
    memory_age_penalty: float = 0.03
    memory_decay: float = 0.94
    memory_occlusion_bonus: float = 0.12
    oru_min_gap_frames: int = 2
    oru_blend: float = 0.38
    velocity_smoothing: float = 0.75
    max_velocity_px_per_s: float = 900.0
    patch_radius: int = 16
    histogram_bins: int = 16
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45


@dataclass
class ObservationCentricMemoryResult:
    center: Tuple[float, float]
    confidence: float
    accepted: bool
    refined: bool
    rejected: bool
    reason: Optional[str]
    predicted_center: Tuple[float, float]
    association_distance_px: float
    adaptive_gate_px: float
    fusion_score: float
    memory_similarity: float
    motion_consistency: float
    uncertainty_score: float
    occlusion_gap: int
    used_oru: bool
    used_memory_recall: bool
    memory_size: int
    memory_update_action: str


@dataclass
class _MemoryEntry:
    embedding: np.ndarray
    quality: float
    age: int


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _clip_unit(value: float) -> float:
    return float(np.clip(_safe_float(value, 0.0), 0.0, 1.0))


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


class ObservationCentricMemoryAssociationGate:
    """Single-target gate with observation-centric re-update and memory-control policy."""

    def __init__(self, config: Optional[ObservationCentricMemoryConfig] = None):
        self.config = config or ObservationCentricMemoryConfig()
        self._state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self._prev_timestamp: Optional[float] = None
        self._last_accepted_center: Optional[np.ndarray] = None
        self._last_motion_dir: Optional[np.ndarray] = None
        self._miss_streak: int = 0
        self._uncertainty_ema: float = 0.0
        self._memory: List[_MemoryEntry] = []

    def reset(self) -> None:
        self._state = None
        self._prev_timestamp = None
        self._last_accepted_center = None
        self._last_motion_dir = None
        self._miss_streak = 0
        self._uncertainty_ema = 0.0
        self._memory = []

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

    def _memory_similarity(self, embedding: Optional[np.ndarray]) -> Tuple[float, bool]:
        if embedding is None or not self._memory:
            return 0.0, False
        topk = max(1, int(self.config.memory_retrieval_topk))
        candidates = sorted(self._memory, key=lambda m: (m.quality - 0.02 * m.age), reverse=True)[:topk]
        sims = []
        for item in candidates:
            sim = _cosine_similarity(embedding, item.embedding)
            if sim is None:
                continue
            sims.append(float(sim))
        if not sims:
            return 0.0, False
        best = float(max(sims))
        return best, True

    def _motion_consistency(self, residual: np.ndarray) -> float:
        if self._last_motion_dir is None:
            return 0.0
        r = _normalize(residual)
        if r is None:
            return 0.0
        score = float(np.dot(r, self._last_motion_dir))
        return float(np.clip(score, -1.0, 1.0))

    def _prune_memory(self) -> None:
        budget = max(4, int(self.config.memory_budget))
        if len(self._memory) <= budget:
            return
        age_pen = max(0.0, float(self.config.memory_age_penalty))
        scored = []
        for idx, item in enumerate(self._memory):
            score = float(item.quality) - age_pen * float(item.age)
            scored.append((score, idx, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        kept = [entry for (_, _, entry) in scored[:budget]]
        self._memory = kept

    def _update_memory(
        self,
        embedding: Optional[np.ndarray],
        confidence: float,
        novelty: float,
        occlusion_gap: int,
    ) -> str:
        if embedding is None:
            return "skip"

        for item in self._memory:
            item.age += 1
            item.quality *= float(np.clip(self.config.memory_decay, 0.0, 1.0))

        min_conf = float(np.clip(self.config.memory_min_confidence, 0.0, 1.0))
        min_novelty = float(max(0.0, self.config.memory_min_novelty))
        occlusion_bonus = float(max(0.0, self.config.memory_occlusion_bonus))

        quality = float(np.clip(confidence, 0.0, 1.0))
        quality += 0.45 * float(np.clip(novelty, 0.0, 1.0))
        if occlusion_gap > 0:
            quality += occlusion_bonus * min(1.0, occlusion_gap / 8.0)
        quality = float(np.clip(quality, 0.0, 1.6))

        if confidence < min_conf and novelty < min_novelty and occlusion_gap <= 0:
            self._prune_memory()
            return "skip"

        if len(self._memory) < max(4, int(self.config.memory_budget)):
            self._memory.append(_MemoryEntry(embedding=embedding.copy(), quality=quality, age=0))
            self._prune_memory()
            return "append"

        # replace the weakest entry when novelty or quality is meaningful
        weakest_idx = 0
        weakest_score = float("inf")
        age_pen = max(0.0, float(self.config.memory_age_penalty))
        for idx, item in enumerate(self._memory):
            score = float(item.quality) - age_pen * float(item.age)
            if score < weakest_score:
                weakest_score = score
                weakest_idx = idx

        if quality > weakest_score + 0.03 or novelty >= min_novelty:
            self._memory[weakest_idx] = _MemoryEntry(embedding=embedding.copy(), quality=quality, age=0)
            self._prune_memory()
            return "replace"

        self._prune_memory()
        return "skip"

    def _make_result(
        self,
        center_xy: np.ndarray,
        confidence: float,
        accepted: bool,
        refined: bool,
        rejected: bool,
        reason: Optional[str],
        predicted: np.ndarray,
        distance: float,
        gate_px: float,
        fusion_score: float,
        memory_similarity: float,
        motion_consistency: float,
        uncertainty_score: float,
        occlusion_gap: int,
        used_oru: bool,
        used_memory_recall: bool,
        memory_update_action: str,
    ) -> ObservationCentricMemoryResult:
        return ObservationCentricMemoryResult(
            center=(float(center_xy[0]), float(center_xy[1])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            accepted=bool(accepted),
            refined=bool(refined),
            rejected=bool(rejected),
            reason=reason,
            predicted_center=(float(predicted[0]), float(predicted[1])),
            association_distance_px=float(max(0.0, distance)),
            adaptive_gate_px=float(max(0.0, gate_px)),
            fusion_score=float(np.clip(fusion_score, 0.0, 1.0)),
            memory_similarity=float(np.clip(memory_similarity, -1.0, 1.0)),
            motion_consistency=float(np.clip(motion_consistency, -1.0, 1.0)),
            uncertainty_score=float(max(0.0, uncertainty_score)),
            occlusion_gap=int(max(0, occlusion_gap)),
            used_oru=bool(used_oru),
            used_memory_recall=bool(used_memory_recall),
            memory_size=int(len(self._memory)),
            memory_update_action=str(memory_update_action),
        )

    def update(
        self,
        *,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> ObservationCentricMemoryResult:
        obs = np.asarray(
            [
                _safe_float(center[0], default=np.nan),
                _safe_float(center[1], default=np.nan),
            ],
            dtype=np.float64,
        )
        if obs.size != 2 or not np.all(np.isfinite(obs)):
            raise ValueError("Invalid center for ObservationCentricMemoryAssociationGate.update")

        conf = _clip_unit(confidence)
        ts = self._resolve_timestamp(timestamp_s)
        if self._prev_timestamp is None:
            dt = 1.0 / 30.0
        else:
            dt = float(np.clip(ts - self._prev_timestamp, self.config.min_dt_s, self.config.max_dt_s))
        self._prev_timestamp = ts

        embedding = build_patch_embedding(
            frame,
            (float(obs[0]), float(obs[1])),
            radius=max(4, int(self.config.patch_radius)),
            bins=max(8, int(self.config.histogram_bins)),
        )

        if self._state is None:
            self._state = np.array([obs[0], obs[1], 0.0, 0.0], dtype=np.float64)
            self._last_accepted_center = obs.copy()
            self._miss_streak = 0
            _ = self._update_memory(embedding, conf, novelty=1.0, occlusion_gap=0)
            return self._make_result(
                center_xy=obs,
                confidence=conf,
                accepted=True,
                refined=False,
                rejected=False,
                reason=None,
                predicted=obs,
                distance=0.0,
                gate_px=max(0.0, float(self.config.base_gate_px)),
                fusion_score=1.0,
                memory_similarity=1.0,
                motion_consistency=0.0,
                uncertainty_score=0.0,
                occlusion_gap=0,
                used_oru=False,
                used_memory_recall=False,
                memory_update_action="append",
            )

        state = self._state
        pred_center = state[:2] + state[2:] * dt
        pred_v = state[2:]
        speed = float(np.hypot(pred_v[0], pred_v[1]))
        residual = obs - pred_center
        distance = float(np.linalg.norm(residual))

        conf_unc = 1.0 - conf
        dist_unc = np.clip(distance / max(float(self.config.max_association_distance_px), 1.0), 0.0, 1.5)
        raw_unc = 0.58 * conf_unc + 0.42 * dist_unc
        self._uncertainty_ema = 0.8 * self._uncertainty_ema + 0.2 * float(raw_unc)
        uncertainty = float(np.clip(self._uncertainty_ema, 0.0, 1.5))

        gate = float(self.config.base_gate_px)
        gate += float(self.config.velocity_gate_scale) * speed
        gate += float(self.config.uncertainty_gate_scale) * uncertainty * float(self.config.base_gate_px)
        gate += float(self.config.occlusion_gate_scale) * float(min(self._miss_streak, 12))
        gate = float(np.clip(gate, self.config.min_gate_px, self.config.max_gate_px))

        mem_sim, used_mem = self._memory_similarity(embedding)
        motion_consistency = self._motion_consistency(residual)

        dist_term = float(np.exp(-distance / max(gate, 1.0)))
        mem_term = float(np.clip((mem_sim + 1.0) * 0.5, 0.0, 1.0))
        motion_term = float(np.clip((motion_consistency + 1.0) * 0.5, 0.0, 1.0))
        fusion = (
            float(self.config.distance_weight) * dist_term
            + float(self.config.memory_weight) * mem_term
            + float(self.config.motion_weight) * motion_term
        )
        fusion = float(np.clip(fusion, 0.0, 1.0))

        within_distance = distance <= max(0.0, float(self.config.max_association_distance_px))
        within_gate = distance <= gate
        similarity_ok = mem_sim >= float(self.config.min_similarity) or dist_term >= 0.7
        high_conf_ok = conf >= float(self.config.high_confidence_threshold)
        low_conf_ok = conf >= float(self.config.low_confidence_threshold) and fusion >= float(self.config.min_fusion_score)
        accepted = within_distance and within_gate and similarity_ok and (high_conf_ok or low_conf_ok)

        if not accepted:
            self._miss_streak += 1
            damp = float(np.clip(self.config.confidence_penalty, 0.0, 1.0))
            out_conf = conf * damp
            self._state = np.array(
                [pred_center[0], pred_center[1], pred_v[0] * 0.9, pred_v[1] * 0.9],
                dtype=np.float64,
            )
            return self._make_result(
                center_xy=obs,
                confidence=out_conf,
                accepted=False,
                refined=False,
                rejected=True,
                reason="association_rejected",
                predicted=pred_center,
                distance=distance,
                gate_px=gate,
                fusion_score=fusion,
                memory_similarity=mem_sim,
                motion_consistency=motion_consistency,
                uncertainty_score=uncertainty,
                occlusion_gap=self._miss_streak,
                used_oru=False,
                used_memory_recall=used_mem,
                memory_update_action="skip",
            )

        occlusion_gap = int(self._miss_streak)
        self._miss_streak = 0

        used_oru = bool(occlusion_gap >= max(1, int(self.config.oru_min_gap_frames)))
        if used_oru and self._last_accepted_center is not None:
            # Observation-centric re-update: trust the latest observation more after a gap,
            # while preserving motion continuity from last accepted center.
            base = self._last_accepted_center
            rel = obs - base
            blend = float(np.clip(self.config.oru_blend + 0.06 * min(occlusion_gap, 6), 0.2, 0.8))
            refined_center = (1.0 - blend) * pred_center + blend * (base + rel)
        else:
            obs_weight = 0.86 if conf >= float(self.config.high_confidence_threshold) else 0.66
            refined_center = obs_weight * obs + (1.0 - obs_weight) * pred_center

        refine_shift = float(np.linalg.norm(refined_center - obs))
        if refine_shift > max(0.0, float(self.config.max_refine_shift_px)):
            refined_center = obs.copy()
            refine_shift = 0.0
        refined = refine_shift >= 0.15

        measured_v = (refined_center - state[:2]) / max(dt, self.config.min_dt_s)
        smooth = float(np.clip(self.config.velocity_smoothing, 0.0, 1.0))
        new_v = smooth * state[2:] + (1.0 - smooth) * measured_v
        vx, vy = self._clip_velocity(float(new_v[0]), float(new_v[1]))
        self._state = np.array([refined_center[0], refined_center[1], vx, vy], dtype=np.float64)

        motion_vec = refined_center - state[:2]
        norm_motion = _normalize(motion_vec)
        if norm_motion is not None:
            self._last_motion_dir = norm_motion
        self._last_accepted_center = refined_center.copy()

        novelty = float(np.clip(1.0 - max(0.0, mem_term), 0.0, 1.0))
        memory_action = self._update_memory(embedding, conf, novelty=novelty, occlusion_gap=occlusion_gap)

        out_conf = conf * max(0.0, float(self.config.confidence_gain))
        if used_oru:
            out_conf = min(out_conf, max(conf, conf + 0.06))

        return self._make_result(
            center_xy=refined_center,
            confidence=out_conf,
            accepted=True,
            refined=refined,
            rejected=False,
            reason=None,
            predicted=pred_center,
            distance=distance,
            gate_px=gate,
            fusion_score=fusion,
            memory_similarity=mem_sim,
            motion_consistency=motion_consistency,
            uncertainty_score=uncertainty,
            occlusion_gap=occlusion_gap,
            used_oru=used_oru,
            used_memory_recall=used_mem,
            memory_update_action=memory_action,
        )
