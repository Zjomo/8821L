"""
SpotZoom innovation frontier v50.

Dual-threshold drift-compensated association gate inspired by:
- ByteTrack: keep low-score observations in a second-stage association path.
- Norfair: distance-first lightweight online tracking and drift-aware refinement.
- CoTracker3/SAM2: streaming state updates under long-horizon video loops.

This module intentionally keeps a numpy-only runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DualThresholdDriftAssociationConfig:
    high_confidence_threshold: float = 0.72
    low_confidence_threshold: float = 0.18
    base_gate_px: float = 10.8
    min_gate_px: float = 4.0
    max_gate_px: float = 56.0
    low_confidence_gate_scale: float = 1.45
    velocity_gate_scale: float = 0.2
    drift_gate_scale: float = 0.9
    max_association_distance_px: float = 52.0
    max_refine_shift_px: float = 24.0
    min_refine_shift_px: float = 0.15
    min_similarity: float = 0.06
    min_low_confidence_score: float = 0.26
    distance_weight: float = 0.58
    appearance_weight: float = 0.28
    drift_weight: float = 0.14
    confidence_gain: float = 1.02
    confidence_penalty: float = 0.9
    velocity_smoothing: float = 0.74
    appearance_momentum: float = 0.86
    drift_ema_decay: float = 0.82
    max_drift_px: float = 28.0
    max_velocity_px_per_s: float = 850.0
    patch_radius: int = 16
    histogram_bins: int = 16
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45


@dataclass
class DualThresholdDriftAssociationResult:
    center: Tuple[float, float]
    confidence: float
    accepted: bool
    refined: bool
    rejected: bool
    reason: Optional[str]
    predicted_center: Tuple[float, float]
    association_distance_px: float
    adaptive_gate_px: float
    appearance_similarity: float
    fusion_score: float
    drift_dx_px: float
    drift_dy_px: float
    drift_norm_px: float
    used_low_confidence_path: bool
    used_drift_comp: bool
    innovation_score: float


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


class DualThresholdDriftAssociationGate:
    """Single-target association gate with low-confidence salvage and drift compensation."""

    def __init__(self, config: Optional[DualThresholdDriftAssociationConfig] = None):
        self.config = config or DualThresholdDriftAssociationConfig()
        self._state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self._appearance_ref: Optional[np.ndarray] = None
        self._prev_timestamp: Optional[float] = None
        self._prev_gray_centroid: Optional[np.ndarray] = None
        self._drift_ema: np.ndarray = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        self._state = None
        self._appearance_ref = None
        self._prev_timestamp = None
        self._prev_gray_centroid = None
        self._drift_ema = np.zeros(2, dtype=np.float64)

    def _resolve_timestamp(self, timestamp_s: Optional[float]) -> float:
        ts = _safe_float(timestamp_s if timestamp_s is not None else np.nan, default=np.nan)
        if not np.isfinite(ts):
            import time

            ts = float(time.perf_counter())
        return ts

    def _estimate_global_drift(self, frame: np.ndarray) -> np.ndarray:
        gray = _to_gray(frame)
        if gray is None:
            return np.zeros(2, dtype=np.float64)
        sample = gray[::4, ::4]
        if sample.size == 0:
            return np.zeros(2, dtype=np.float64)

        sample = np.asarray(sample, dtype=np.float64)
        baseline = float(np.percentile(sample, 60))
        mass = np.clip(sample - baseline, 0.0, None)
        denom = float(np.sum(mass))
        if denom <= 1e-9 or not np.isfinite(denom):
            return np.zeros(2, dtype=np.float64)

        ys = np.arange(sample.shape[0], dtype=np.float64)
        xs = np.arange(sample.shape[1], dtype=np.float64)
        cx = float(np.sum(mass * xs[None, :]) / denom)
        cy = float(np.sum(mass * ys[:, None]) / denom)
        centroid = np.array([cx * 4.0, cy * 4.0], dtype=np.float64)

        if self._prev_gray_centroid is None:
            self._prev_gray_centroid = centroid
            return np.zeros(2, dtype=np.float64)

        drift = centroid - self._prev_gray_centroid
        self._prev_gray_centroid = centroid
        if not np.all(np.isfinite(drift)):
            return np.zeros(2, dtype=np.float64)

        max_drift = max(0.0, float(self.config.max_drift_px))
        norm = float(np.linalg.norm(drift))
        if max_drift > 0.0 and norm > max_drift and norm > 1e-8:
            drift = drift * (max_drift / norm)
        return drift

    def _clip_velocity(self, vx: float, vy: float) -> Tuple[float, float]:
        vmax = float(max(0.0, self.config.max_velocity_px_per_s))
        speed = float(np.hypot(vx, vy))
        if vmax <= 0.0 or speed <= vmax or speed <= 1e-9:
            return vx, vy
        scale = vmax / speed
        return float(vx * scale), float(vy * scale)

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
        adaptive_gate_px: float,
        appearance_similarity: float,
        fusion_score: float,
        drift_vec: np.ndarray,
        used_low_confidence_path: bool,
        used_drift_comp: bool,
        innovation_score: float,
    ) -> DualThresholdDriftAssociationResult:
        return DualThresholdDriftAssociationResult(
            center=(float(center_xy[0]), float(center_xy[1])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            accepted=bool(accepted),
            refined=bool(refined),
            rejected=bool(rejected),
            reason=reason,
            predicted_center=(float(predicted[0]), float(predicted[1])),
            association_distance_px=float(max(0.0, distance)),
            adaptive_gate_px=float(max(0.0, adaptive_gate_px)),
            appearance_similarity=float(np.clip(appearance_similarity, -1.0, 1.0)),
            fusion_score=float(np.clip(fusion_score, 0.0, 1.0)),
            drift_dx_px=float(drift_vec[0]),
            drift_dy_px=float(drift_vec[1]),
            drift_norm_px=float(np.hypot(drift_vec[0], drift_vec[1])),
            used_low_confidence_path=bool(used_low_confidence_path),
            used_drift_comp=bool(used_drift_comp),
            innovation_score=float(max(0.0, innovation_score)),
        )

    def update(
        self,
        *,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> DualThresholdDriftAssociationResult:
        obs = np.asarray(
            [
                _safe_float(center[0], default=np.nan),
                _safe_float(center[1], default=np.nan),
            ],
            dtype=np.float64,
        )
        if obs.size != 2 or not np.all(np.isfinite(obs)):
            raise ValueError("Invalid center for DualThresholdDriftAssociationGate.update")

        conf = _clip_unit(confidence)
        ts = self._resolve_timestamp(timestamp_s)
        if self._prev_timestamp is None:
            dt = 1.0 / 30.0
        else:
            dt = float(np.clip(ts - self._prev_timestamp, self.config.min_dt_s, self.config.max_dt_s))
        self._prev_timestamp = ts

        drift_obs = self._estimate_global_drift(frame)
        decay = float(np.clip(self.config.drift_ema_decay, 0.0, 0.999))
        self._drift_ema = decay * self._drift_ema + (1.0 - decay) * drift_obs
        drift_norm = float(np.linalg.norm(self._drift_ema))
        used_drift = drift_norm > 0.15

        embedding = build_patch_embedding(
            frame,
            (float(obs[0]), float(obs[1])),
            radius=max(4, int(self.config.patch_radius)),
            bins=max(8, int(self.config.histogram_bins)),
        )

        if self._state is None:
            self._state = np.array([obs[0], obs[1], 0.0, 0.0], dtype=np.float64)
            self._appearance_ref = embedding
            return self._make_result(
                center_xy=obs,
                confidence=conf,
                accepted=True,
                refined=False,
                rejected=False,
                reason=None,
                predicted=obs,
                distance=0.0,
                adaptive_gate_px=max(0.0, float(self.config.base_gate_px)),
                appearance_similarity=1.0,
                fusion_score=1.0,
                drift_vec=self._drift_ema,
                used_low_confidence_path=False,
                used_drift_comp=used_drift,
                innovation_score=0.0,
            )

        state = self._state
        pred_center = state[:2] + state[2:] * dt + self._drift_ema
        pred_v = state[2:]
        speed = float(np.hypot(pred_v[0], pred_v[1]))
        distance = float(np.linalg.norm(obs - pred_center))

        gate = float(self.config.base_gate_px)
        gate += float(self.config.velocity_gate_scale) * speed
        gate += float(self.config.drift_gate_scale) * drift_norm
        used_low_conf = conf < float(self.config.high_confidence_threshold)
        if used_low_conf:
            gate *= max(1.0, float(self.config.low_confidence_gate_scale))
        gate = float(np.clip(gate, self.config.min_gate_px, self.config.max_gate_px))

        app_sim = _cosine_similarity(embedding, self._appearance_ref)
        if app_sim is None:
            app_sim = 0.0
        app_term = float(np.clip((app_sim + 1.0) * 0.5, 0.0, 1.0))
        distance_term = float(np.exp(-distance / max(1.0, gate)))
        drift_term = float(np.exp(-drift_norm / max(1.0, float(self.config.max_drift_px))))
        innovation = float(np.linalg.norm(obs - state[:2]))
        innovation_term = float(np.exp(-innovation / max(1.0, gate)))
        fusion_score = (
            float(self.config.distance_weight) * distance_term
            + float(self.config.appearance_weight) * app_term
            + float(self.config.drift_weight) * drift_term
        )
        fusion_score = float(np.clip(0.75 * fusion_score + 0.25 * innovation_term, 0.0, 1.0))

        within_distance = distance <= max(0.0, float(self.config.max_association_distance_px))
        within_gate = distance <= gate
        high_conf_ok = conf >= float(self.config.high_confidence_threshold)
        low_conf_ok = conf >= float(self.config.low_confidence_threshold) and fusion_score >= float(
            self.config.min_low_confidence_score
        )
        similarity_ok = app_sim >= float(self.config.min_similarity) or distance_term >= 0.7
        accepted = within_distance and within_gate and similarity_ok and (high_conf_ok or low_conf_ok)

        if not accepted:
            damp = max(0.0, min(1.0, float(self.config.confidence_penalty)))
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
                adaptive_gate_px=gate,
                appearance_similarity=app_sim,
                fusion_score=fusion_score,
                drift_vec=self._drift_ema,
                used_low_confidence_path=used_low_conf,
                used_drift_comp=used_drift,
                innovation_score=innovation,
            )

        obs_weight = 0.85 if not used_low_conf else 0.62
        if conf < float(self.config.low_confidence_threshold):
            obs_weight = 0.5
        refined_center = obs_weight * obs + (1.0 - obs_weight) * pred_center
        refine_shift = float(np.linalg.norm(refined_center - obs))
        if refine_shift > max(0.0, float(self.config.max_refine_shift_px)):
            refined_center = obs.copy()
            refine_shift = 0.0
        refined = refine_shift >= max(0.0, float(self.config.min_refine_shift_px))

        measured_v = (refined_center - state[:2]) / max(dt, self.config.min_dt_s)
        smooth = float(np.clip(self.config.velocity_smoothing, 0.0, 1.0))
        new_v = smooth * state[2:] + (1.0 - smooth) * measured_v
        vx, vy = self._clip_velocity(float(new_v[0]), float(new_v[1]))
        self._state = np.array([refined_center[0], refined_center[1], vx, vy], dtype=np.float64)

        if embedding is not None:
            if self._appearance_ref is None:
                self._appearance_ref = embedding
            else:
                alpha = float(np.clip(1.0 - self.config.appearance_momentum, 0.0, 1.0))
                self._appearance_ref = _normalize((1.0 - alpha) * self._appearance_ref + alpha * embedding)

        out_conf = conf * max(0.0, float(self.config.confidence_gain))
        if used_low_conf:
            out_conf = min(out_conf, max(conf, conf + 0.08))

        return self._make_result(
            center_xy=refined_center,
            confidence=out_conf,
            accepted=True,
            refined=refined,
            rejected=False,
            reason=None,
            predicted=pred_center,
            distance=distance,
            adaptive_gate_px=gate,
            appearance_similarity=app_sim,
            fusion_score=fusion_score,
            drift_vec=self._drift_ema,
            used_low_confidence_path=used_low_conf,
            used_drift_comp=used_drift,
            innovation_score=innovation,
        )
