"""
SpotZoom innovation frontier v52.

Hierarchical memory + camera-motion-compensated association gate inspired by:
- BoT-SORT: camera-motion compensation (CMC) under moving-camera scenarios.
- SAM2Long: fixed-budget multi-path memory idea for long-horizon robustness.
- CoTracker3: online, memory-efficient processing for long streams.

This module intentionally keeps a numpy-only runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class HierarchicalCmcAssociationConfig:
    high_confidence_threshold: float = 0.72
    low_confidence_threshold: float = 0.16
    base_gate_px: float = 11.6
    min_gate_px: float = 4.0
    max_gate_px: float = 64.0
    uncertainty_gate_scale: float = 0.92
    cmc_gate_scale: float = 0.78
    cmc_prediction_blend: float = 0.65
    cmc_ema_decay: float = 0.84
    max_cmc_px: float = 30.0
    max_association_distance_px: float = 58.0
    max_refine_shift_px: float = 24.0
    min_similarity: float = 0.05
    min_fusion_score: float = 0.24
    min_low_confidence_fusion: float = 0.33
    min_low_confidence_memory_similarity: float = 0.2
    distance_weight: float = 0.46
    memory_weight: float = 0.30
    motion_weight: float = 0.16
    confidence_weight: float = 0.08
    confidence_gain: float = 1.03
    confidence_penalty: float = 0.90
    velocity_smoothing: float = 0.76
    max_velocity_px_per_s: float = 940.0
    short_memory_budget: int = 12
    long_memory_budget: int = 24
    short_memory_decay: float = 0.92
    long_memory_decay: float = 0.985
    memory_retrieval_topk: int = 4
    memory_age_penalty: float = 0.025
    memory_min_confidence: float = 0.24
    memory_min_novelty: float = 0.08
    scene_change_threshold: float = 0.38
    scene_reset_cooldown_frames: int = 4
    patch_radius: int = 16
    histogram_bins: int = 16
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45


@dataclass
class HierarchicalCmcAssociationResult:
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
    cmc_dx_px: float
    cmc_dy_px: float
    cmc_norm_px: float
    used_cmc: bool
    used_long_memory: bool
    scene_reset: bool
    memory_size: int
    memory_update_action: str
    innovation_score: float


@dataclass
class _MemoryEntry:
    embedding: np.ndarray
    center: np.ndarray
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


def _frame_embedding(frame: np.ndarray, bins: int = 24) -> Optional[np.ndarray]:
    gray = _to_gray(frame)
    if gray is None:
        return None
    low = gray[::6, ::6]
    if low.size == 0:
        return None
    hist = np.histogram(np.clip(low, 0.0, 255.0), bins=max(8, int(bins)), range=(0.0, 255.0), density=True)[0]
    gx = np.diff(low, axis=1, prepend=low[:, :1])
    gy = np.diff(low, axis=0, prepend=low[:1, :])
    grad = np.sqrt(gx * gx + gy * gy)
    desc = np.concatenate(
        [
            hist.astype(np.float64),
            np.array(
                [
                    float(np.mean(low)),
                    float(np.std(low)),
                    float(np.mean(grad)),
                    float(np.std(grad)),
                ],
                dtype=np.float64,
            ),
        ],
        axis=0,
    )
    return _normalize(desc)


class HierarchicalCmcAssociationGate:
    """Single-target gate with hierarchical memory and CMC-assisted association."""

    def __init__(self, config: Optional[HierarchicalCmcAssociationConfig] = None):
        self.config = config or HierarchicalCmcAssociationConfig()
        self._state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self._prev_timestamp: Optional[float] = None
        self._last_motion_dir: Optional[np.ndarray] = None
        self._short_memory: List[_MemoryEntry] = []
        self._long_memory: List[_MemoryEntry] = []
        self._prev_gray_centroid: Optional[np.ndarray] = None
        self._cmc_ema = np.zeros(2, dtype=np.float64)
        self._scene_embedding: Optional[np.ndarray] = None
        self._scene_reset_cooldown: int = 0
        self._miss_streak: int = 0
        self._uncertainty_ema: float = 0.0

    def reset(self) -> None:
        self._state = None
        self._prev_timestamp = None
        self._last_motion_dir = None
        self._short_memory = []
        self._long_memory = []
        self._prev_gray_centroid = None
        self._cmc_ema = np.zeros(2, dtype=np.float64)
        self._scene_embedding = None
        self._scene_reset_cooldown = 0
        self._miss_streak = 0
        self._uncertainty_ema = 0.0

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

    def _estimate_cmc(self, frame: np.ndarray) -> np.ndarray:
        gray = _to_gray(frame)
        if gray is None:
            return self._cmc_ema.copy()
        sample = gray[::4, ::4]
        if sample.size == 0:
            return self._cmc_ema.copy()

        baseline = float(np.percentile(sample, 60))
        mass = np.clip(sample.astype(np.float64) - baseline, 0.0, None)
        denom = float(np.sum(mass))
        if denom <= 1e-9 or not np.isfinite(denom):
            return self._cmc_ema.copy()

        ys = np.arange(sample.shape[0], dtype=np.float64)
        xs = np.arange(sample.shape[1], dtype=np.float64)
        cx = float(np.sum(mass * xs[None, :]) / denom)
        cy = float(np.sum(mass * ys[:, None]) / denom)
        centroid = np.array([cx * 4.0, cy * 4.0], dtype=np.float64)

        drift = np.zeros(2, dtype=np.float64)
        if self._prev_gray_centroid is not None:
            drift = centroid - self._prev_gray_centroid
            if not np.all(np.isfinite(drift)):
                drift = np.zeros(2, dtype=np.float64)
        self._prev_gray_centroid = centroid

        max_cmc = max(0.0, float(self.config.max_cmc_px))
        norm = float(np.linalg.norm(drift))
        if max_cmc > 0.0 and norm > max_cmc and norm > 1e-9:
            drift = drift * (max_cmc / norm)

        decay = float(np.clip(self.config.cmc_ema_decay, 0.0, 1.0))
        self._cmc_ema = decay * self._cmc_ema + (1.0 - decay) * drift
        if not np.all(np.isfinite(self._cmc_ema)):
            self._cmc_ema = np.zeros(2, dtype=np.float64)
        return self._cmc_ema.copy()

    def _scene_reset_if_needed(self, frame: np.ndarray) -> bool:
        emb = _frame_embedding(frame)
        if emb is None:
            return False
        if self._scene_embedding is None:
            self._scene_embedding = emb
            return False
        sim = _cosine_similarity(emb, self._scene_embedding)
        if sim is None:
            self._scene_embedding = emb
            return False
        self._scene_embedding = 0.85 * self._scene_embedding + 0.15 * emb
        self._scene_embedding = _normalize(self._scene_embedding)
        if self._scene_embedding is None:
            self._scene_embedding = emb

        if self._scene_reset_cooldown > 0:
            self._scene_reset_cooldown -= 1
            return False
        if float(sim) >= float(self.config.scene_change_threshold):
            return False

        self._short_memory = []
        self._state = None if self._state is None else self._state.copy()
        if self._state is not None:
            self._state[2:] = 0.0
        self._scene_reset_cooldown = max(1, int(self.config.scene_reset_cooldown_frames))
        return True

    def _memory_similarity(self, embedding: Optional[np.ndarray]) -> Tuple[float, bool]:
        if embedding is None:
            return 0.0, False
        topk = max(1, int(self.config.memory_retrieval_topk))
        age_pen = max(0.0, float(self.config.memory_age_penalty))

        candidates: List[Tuple[_MemoryEntry, bool]] = []
        for item in self._short_memory:
            candidates.append((item, False))
        for item in self._long_memory:
            candidates.append((item, True))
        if not candidates:
            return 0.0, False

        ranked = sorted(
            candidates,
            key=lambda p: (float(p[0].quality) - age_pen * float(p[0].age)),
            reverse=True,
        )[:topk]

        best_score = -1.0
        best_is_long = False
        for entry, is_long in ranked:
            sim = _cosine_similarity(embedding, entry.embedding)
            if sim is None:
                continue
            score = float(sim)
            if score > best_score:
                best_score = score
                best_is_long = bool(is_long)
        if best_score < -0.5:
            return 0.0, False
        return float(max(0.0, best_score)), best_is_long

    def _motion_consistency(self, residual: np.ndarray) -> float:
        if self._last_motion_dir is None:
            return 0.0
        r = _normalize(residual)
        if r is None:
            return 0.0
        score = float(np.dot(r, self._last_motion_dir))
        return float(np.clip(score, -1.0, 1.0))

    def _age_and_prune_memories(self) -> None:
        s_decay = float(np.clip(self.config.short_memory_decay, 0.0, 1.0))
        l_decay = float(np.clip(self.config.long_memory_decay, 0.0, 1.0))
        for item in self._short_memory:
            item.age += 1
            item.quality *= s_decay
        for item in self._long_memory:
            item.age += 1
            item.quality *= l_decay

        age_pen = max(0.0, float(self.config.memory_age_penalty))
        short_budget = max(4, int(self.config.short_memory_budget))
        long_budget = max(short_budget, int(self.config.long_memory_budget))
        self._short_memory = sorted(
            self._short_memory,
            key=lambda m: (float(m.quality) - age_pen * float(m.age)),
            reverse=True,
        )[:short_budget]
        self._long_memory = sorted(
            self._long_memory,
            key=lambda m: (float(m.quality) - 0.5 * age_pen * float(m.age)),
            reverse=True,
        )[:long_budget]

    def _add_memory(self, bank: List[_MemoryEntry], embedding: np.ndarray, center: np.ndarray, quality: float) -> None:
        bank.append(
            _MemoryEntry(
                embedding=np.asarray(embedding, dtype=np.float64),
                center=np.asarray(center, dtype=np.float64),
                quality=float(np.clip(quality, 0.0, 1.0)),
                age=0,
            )
        )

    def _update_memories(self, embedding: Optional[np.ndarray], center: np.ndarray, confidence: float) -> str:
        if embedding is None:
            return "skip"
        conf = _clip_unit(confidence)
        if conf < float(self.config.memory_min_confidence):
            return "skip"

        sim_short = 0.0
        for item in self._short_memory:
            sim = _cosine_similarity(embedding, item.embedding)
            if sim is not None:
                sim_short = max(sim_short, float(sim))

        sim_long = 0.0
        for item in self._long_memory:
            sim = _cosine_similarity(embedding, item.embedding)
            if sim is not None:
                sim_long = max(sim_long, float(sim))

        novelty = 1.0 - max(sim_short, sim_long)
        novelty_min = max(0.0, float(self.config.memory_min_novelty))
        action = "skip"

        if novelty >= novelty_min:
            self._add_memory(self._short_memory, embedding, center, quality=0.45 + 0.55 * conf)
            action = "add_short"
        elif sim_short >= sim_long and self._short_memory:
            best = max(self._short_memory, key=lambda m: _cosine_similarity(embedding, m.embedding) or -1.0)
            best.embedding = 0.9 * best.embedding + 0.1 * embedding
            normed = _normalize(best.embedding)
            if normed is not None:
                best.embedding = normed
            best.center = 0.85 * best.center + 0.15 * center
            best.quality = float(np.clip(0.7 * best.quality + 0.3 * conf, 0.0, 1.0))
            best.age = 0
            action = "refresh_short"
        elif self._long_memory:
            best = max(self._long_memory, key=lambda m: _cosine_similarity(embedding, m.embedding) or -1.0)
            best.embedding = 0.95 * best.embedding + 0.05 * embedding
            normed = _normalize(best.embedding)
            if normed is not None:
                best.embedding = normed
            best.center = 0.92 * best.center + 0.08 * center
            best.quality = float(np.clip(0.8 * best.quality + 0.2 * conf, 0.0, 1.0))
            best.age = 0
            action = "refresh_long"

        if conf >= float(self.config.high_confidence_threshold):
            self._add_memory(self._long_memory, embedding, center, quality=0.5 + 0.5 * conf)
            if action == "skip":
                action = "add_long"
            elif action == "add_short":
                action = "add_short+long"

        self._age_and_prune_memories()
        return action

    def update(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
        *,
        timestamp_s: Optional[float] = None,
    ) -> HierarchicalCmcAssociationResult:
        obs = np.array([_safe_float(center[0], np.nan), _safe_float(center[1], np.nan)], dtype=np.float64)
        conf = _clip_unit(confidence)
        if not np.all(np.isfinite(obs)):
            obs = np.zeros(2, dtype=np.float64)
            conf = 0.0

        self._age_and_prune_memories()
        scene_reset = bool(self._scene_reset_if_needed(frame))
        cmc = self._estimate_cmc(frame)
        cmc_norm = float(np.linalg.norm(cmc))

        ts = self._resolve_timestamp(timestamp_s)
        if self._prev_timestamp is None:
            dt = float(self.config.min_dt_s)
        else:
            dt = ts - self._prev_timestamp
            dt = float(np.clip(dt, self.config.min_dt_s, self.config.max_dt_s))
        self._prev_timestamp = ts

        if self._state is None:
            self._state = np.array([obs[0], obs[1], 0.0, 0.0], dtype=np.float64)
            embedding = build_patch_embedding(
                frame=frame,
                center=(float(obs[0]), float(obs[1])),
                radius=int(self.config.patch_radius),
                bins=int(self.config.histogram_bins),
            )
            memory_action = self._update_memories(embedding, obs, conf)
            self._miss_streak = 0
            self._uncertainty_ema = 0.0
            return HierarchicalCmcAssociationResult(
                center=(float(obs[0]), float(obs[1])),
                confidence=conf,
                accepted=True,
                refined=False,
                rejected=False,
                reason=None if not scene_reset else "scene_reset",
                predicted_center=(float(obs[0]), float(obs[1])),
                association_distance_px=0.0,
                adaptive_gate_px=float(np.clip(self.config.base_gate_px, self.config.min_gate_px, self.config.max_gate_px)),
                fusion_score=max(0.0, conf),
                memory_similarity=0.0,
                motion_consistency=0.0,
                cmc_dx_px=float(cmc[0]),
                cmc_dy_px=float(cmc[1]),
                cmc_norm_px=cmc_norm,
                used_cmc=cmc_norm > 1e-6,
                used_long_memory=False,
                scene_reset=scene_reset,
                memory_size=len(self._short_memory) + len(self._long_memory),
                memory_update_action=("scene_reset+" + memory_action) if scene_reset else memory_action,
                innovation_score=max(0.0, conf),
            )

        pred = self._state[:2] + self._state[2:] * dt + float(self.config.cmc_prediction_blend) * cmc
        if not np.all(np.isfinite(pred)):
            pred = np.asarray(obs, dtype=np.float64)
        residual = obs - pred
        assoc_dist = float(np.linalg.norm(residual))
        unc = float(np.clip(self._uncertainty_ema + 0.15 * self._miss_streak, 0.0, 2.0))
        cmc_factor = min(1.5, cmc_norm / max(1.0, float(self.config.base_gate_px)))
        gate = float(self.config.base_gate_px) * (
            1.0 + float(self.config.uncertainty_gate_scale) * unc + float(self.config.cmc_gate_scale) * cmc_factor
        )
        gate = float(np.clip(gate, float(self.config.min_gate_px), float(self.config.max_gate_px)))

        embedding = build_patch_embedding(
            frame=frame,
            center=(float(obs[0]), float(obs[1])),
            radius=int(self.config.patch_radius),
            bins=int(self.config.histogram_bins),
        )
        memory_sim, used_long_memory = self._memory_similarity(embedding)
        motion_cons = self._motion_consistency(residual)

        max_assoc = max(1e-6, float(self.config.max_association_distance_px))
        dist_score = float(np.clip(1.0 - assoc_dist / max_assoc, 0.0, 1.0))
        motion_score = float(np.clip(0.5 * (motion_cons + 1.0), 0.0, 1.0))
        fusion = (
            float(self.config.distance_weight) * dist_score
            + float(self.config.memory_weight) * max(0.0, memory_sim)
            + float(self.config.motion_weight) * motion_score
            + float(self.config.confidence_weight) * conf
        )
        fusion = float(np.clip(fusion, 0.0, 1.0))

        rejected = False
        reason: Optional[str] = None
        if assoc_dist > max_assoc:
            rejected = True
            reason = "assoc_distance_too_large"
        elif assoc_dist > gate and fusion < float(self.config.min_fusion_score):
            rejected = True
            reason = "outside_gate"
        elif conf < float(self.config.low_confidence_threshold):
            if fusion < float(self.config.min_low_confidence_fusion):
                rejected = True
                reason = "low_confidence_low_fusion"
            elif memory_sim < float(self.config.min_low_confidence_memory_similarity):
                rejected = True
                reason = "low_confidence_weak_memory"
        elif conf < float(self.config.high_confidence_threshold):
            if memory_sim < float(self.config.min_similarity) and fusion < float(self.config.min_fusion_score):
                rejected = True
                reason = "mid_confidence_weak_memory"

        if rejected:
            self._miss_streak += 1
            self._uncertainty_ema = float(np.clip(0.82 * self._uncertainty_ema + 0.18, 0.0, 2.0))
            if self._state is not None:
                self._state[:2] = pred
            return HierarchicalCmcAssociationResult(
                center=(float(pred[0]), float(pred[1])),
                confidence=float(np.clip(conf * float(self.config.confidence_penalty), 0.0, 1.0)),
                accepted=False,
                refined=False,
                rejected=True,
                reason=reason,
                predicted_center=(float(pred[0]), float(pred[1])),
                association_distance_px=assoc_dist,
                adaptive_gate_px=gate,
                fusion_score=fusion,
                memory_similarity=float(max(0.0, memory_sim)),
                motion_consistency=motion_cons,
                cmc_dx_px=float(cmc[0]),
                cmc_dy_px=float(cmc[1]),
                cmc_norm_px=cmc_norm,
                used_cmc=cmc_norm > 1e-6,
                used_long_memory=used_long_memory,
                scene_reset=scene_reset,
                memory_size=len(self._short_memory) + len(self._long_memory),
                memory_update_action="scene_reset" if scene_reset else "skip",
                innovation_score=fusion,
            )

        blend = float(np.clip(0.35 + 0.35 * dist_score + 0.2 * max(0.0, memory_sim), 0.2, 0.92))
        refined = pred * (1.0 - blend) + obs * blend
        if not np.all(np.isfinite(refined)):
            refined = obs.copy()
        shift = float(np.linalg.norm(refined - obs))
        if shift > float(self.config.max_refine_shift_px):
            refined = obs.copy()

        vx_meas = (refined[0] - self._state[0]) / dt
        vy_meas = (refined[1] - self._state[1]) / dt
        vx_meas, vy_meas = self._clip_velocity(vx_meas, vy_meas)
        alpha = float(np.clip(self.config.velocity_smoothing, 0.0, 1.0))
        vx = alpha * float(self._state[2]) + (1.0 - alpha) * float(vx_meas)
        vy = alpha * float(self._state[3]) + (1.0 - alpha) * float(vy_meas)
        vx, vy = self._clip_velocity(vx, vy)
        self._state = np.array([float(refined[0]), float(refined[1]), vx, vy], dtype=np.float64)

        motion_dir = _normalize(np.array([vx, vy], dtype=np.float64))
        if motion_dir is not None:
            self._last_motion_dir = motion_dir

        self._miss_streak = 0
        self._uncertainty_ema = float(np.clip(0.8 * self._uncertainty_ema, 0.0, 2.0))

        new_conf = conf * float(self.config.confidence_gain)
        new_conf = float(np.clip(new_conf + 0.08 * max(0.0, memory_sim), 0.0, 1.0))
        memory_action = self._update_memories(embedding, refined, new_conf)
        if scene_reset:
            memory_action = f"scene_reset+{memory_action}"

        return HierarchicalCmcAssociationResult(
            center=(float(refined[0]), float(refined[1])),
            confidence=new_conf,
            accepted=True,
            refined=True,
            rejected=False,
            reason=None,
            predicted_center=(float(pred[0]), float(pred[1])),
            association_distance_px=assoc_dist,
            adaptive_gate_px=gate,
            fusion_score=fusion,
            memory_similarity=float(max(0.0, memory_sim)),
            motion_consistency=motion_cons,
            cmc_dx_px=float(cmc[0]),
            cmc_dy_px=float(cmc[1]),
            cmc_norm_px=cmc_norm,
            used_cmc=cmc_norm > 1e-6,
            used_long_memory=used_long_memory,
            scene_reset=scene_reset,
            memory_size=len(self._short_memory) + len(self._long_memory),
            memory_update_action=memory_action,
            innovation_score=float(np.clip(0.5 * fusion + 0.5 * dist_score, 0.0, 1.0)),
        )
