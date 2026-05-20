"""
SpotZoom innovation frontier v45.

Physics-informed turbulence association gate inspired by:
- slmsuite: calibration-aware spot stability loops for SLM control.
- HCIPy / AOtools: wavefront-quality constraints in adaptive optics.
- OpenWFS: iterative wavefront optimization and drift robustness.
- CoTracker3 / ByteTrack: online association under low-confidence observations.

This module keeps a numpy-only runtime dependency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class PhysicsInformedAssociationConfig:
    high_confidence_threshold: float = 0.76
    low_confidence_threshold: float = 0.2
    base_gate_px: float = 10.8
    min_gate_px: float = 4.0
    max_gate_px: float = 46.0
    turbulence_gate_scale: float = 1.35
    velocity_gate_scale: float = 0.22
    profile_gate_scale: float = 1.08
    appearance_weight: float = 0.26
    continuity_weight: float = 0.34
    profile_weight: float = 0.24
    confidence_weight: float = 0.16
    min_similarity: float = 0.08
    min_profile_similarity: float = 0.06
    min_fusion_score: float = 0.26
    max_association_distance_px: float = 44.0
    max_refine_shift_px: float = 22.0
    min_refine_shift_px: float = 0.2
    bridge_gate_multiplier: float = 1.32
    confidence_gain: float = 1.02
    confidence_penalty: float = 0.91
    velocity_smoothing: float = 0.72
    appearance_momentum: float = 0.86
    profile_momentum: float = 0.84
    turbulence_decay: float = 0.82
    temperature_decay: float = 0.86
    max_velocity_px_per_s: float = 780.0
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45
    history_size: int = 20
    residual_window: int = 14
    patch_radius: int = 16
    histogram_bins: int = 16
    profile_bins: int = 12
    hit_counter_max: int = 40
    miss_decay: int = 1
    max_bridge_reject_streak: int = 3


@dataclass
class PhysicsInformedAssociationResult:
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
    profile_similarity: float
    fusion_score: float
    turbulence_score: float
    strehl_proxy: float
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


def _build_radial_profile(patch: np.ndarray, bins: int) -> Optional[np.ndarray]:
    if patch is None or patch.size == 0:
        return None
    h, w = patch.shape[:2]
    if h <= 1 or w <= 1:
        return None

    yy, xx = np.indices((h, w))
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = float(np.max(rr))
    if rmax <= 1e-6:
        return None

    nb = max(8, int(bins))
    edges = np.linspace(0.0, rmax, num=nb + 1, dtype=np.float64)
    profile = np.zeros(nb, dtype=np.float64)
    counts = np.zeros(nb, dtype=np.float64)
    flat_r = rr.reshape(-1)
    flat_v = patch.reshape(-1).astype(np.float64)
    idx = np.searchsorted(edges, flat_r, side="right") - 1
    idx = np.clip(idx, 0, nb - 1)
    for i in range(flat_v.size):
        j = int(idx[i])
        profile[j] += float(flat_v[i])
        counts[j] += 1.0
    valid = counts > 0
    if not np.any(valid):
        return None
    profile[valid] /= counts[valid]
    return _normalize(profile)


def _estimate_strehl_proxy(patch: Optional[np.ndarray]) -> float:
    if patch is None or patch.size == 0:
        return 0.0
    peak = float(np.max(patch))
    mean = float(np.mean(patch))
    std = float(np.std(patch))
    denom = max(1e-3, mean + 0.5 * std)
    return float(np.clip(peak / denom, 0.0, 5.0))


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


class PhysicsInformedAssociationGate:
    """Single-target association gate with turbulence-aware physics priors."""

    def __init__(self, config: Optional[PhysicsInformedAssociationConfig] = None):
        self.config = config or PhysicsInformedAssociationConfig()
        self._state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self._appearance_ref: Optional[np.ndarray] = None
        self._profile_ref: Optional[np.ndarray] = None
        self._clock: float = 0.0
        self._prev_timestamp: Optional[float] = None
        self._temperature: float = 1.0
        self._turbulence_ema: float = 0.0
        self._hit_counter: int = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak: int = 0
        self._reject_streak: int = 0
        self._center_history: Deque[np.ndarray] = deque(maxlen=max(6, int(self.config.history_size)))
        self._residual_history: Deque[float] = deque(maxlen=max(6, int(self.config.residual_window)))

    def reset(self) -> None:
        self._state = None
        self._appearance_ref = None
        self._profile_ref = None
        self._clock = 0.0
        self._prev_timestamp = None
        self._temperature = 1.0
        self._turbulence_ema = 0.0
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak = 0
        self._reject_streak = 0
        self._center_history.clear()
        self._residual_history.clear()

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

    def _estimate_turbulence(self, residual_px: float, profile_similarity: float) -> float:
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
        profile_term = float(np.clip(1.0 - max(-1.0, min(1.0, profile_similarity)), 0.0, 1.0))
        raw = 0.62 * sigma_term + 0.38 * profile_term
        decay = float(np.clip(self.config.turbulence_decay, 0.0, 1.0))
        self._turbulence_ema = float(decay * self._turbulence_ema + (1.0 - decay) * raw)
        return float(np.clip(self._turbulence_ema, 0.0, 1.0))

    def _effective_gate(
        self,
        confidence: float,
        speed_px_s: float,
        turbulence: float,
        profile_similarity: float,
    ) -> float:
        conf = float(np.clip(confidence, 0.0, 1.0))
        gate = float(max(self.config.min_gate_px, self.config.base_gate_px))
        if conf < float(self.config.low_confidence_threshold):
            gate *= 1.08
        elif conf >= float(self.config.high_confidence_threshold):
            gate *= 0.95
        gate += float(max(0.0, self.config.velocity_gate_scale)) * float(max(0.0, speed_px_s))
        gate += float(max(0.0, self.config.turbulence_gate_scale)) * float(max(0.0, turbulence)) * float(
            max(1e-6, self.config.base_gate_px)
        )
        gate += float(max(0.0, self.config.profile_gate_scale)) * float(
            np.clip(1.0 - max(-1.0, min(1.0, profile_similarity)), 0.0, 1.0)
        ) * float(max(1e-6, self.config.base_gate_px)) * 0.5
        gate *= float(np.clip(self._temperature, 0.75, 2.95))
        gate = float(np.clip(gate, float(self.config.min_gate_px), float(self.config.max_gate_px)))
        gate = min(gate, float(max(self.config.min_gate_px, self.config.max_association_distance_px)))
        return float(gate)

    def _update_temperature(self, confidence: float, dist: float, gate: float, turbulence: float) -> None:
        conf = float(np.clip(confidence, 0.0, 1.0))
        dist_ratio = float(np.clip(dist / max(gate, 1e-6), 0.0, 2.0))
        target = 1.0 + 0.24 * (1.0 - conf) + 0.22 * dist_ratio + 0.26 * float(np.clip(turbulence, 0.0, 1.0))
        blend = float(np.clip(1.0 - float(self.config.temperature_decay), 0.05, 0.45))
        self._temperature = float(np.clip((1.0 - blend) * self._temperature + blend * target, 0.65, 3.15))

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
        profile_similarity: float,
        fusion_score: float,
        turbulence_score: float,
        strehl_proxy: float,
        used_bridge: bool,
    ) -> PhysicsInformedAssociationResult:
        return PhysicsInformedAssociationResult(
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
            profile_similarity=float(np.clip(profile_similarity, -1.0, 1.0)),
            fusion_score=float(np.clip(fusion_score, 0.0, 1.0)),
            turbulence_score=float(np.clip(turbulence_score, 0.0, 1.0)),
            strehl_proxy=float(max(0.0, strehl_proxy)),
            temperature=float(max(0.0, self._temperature)),
            hit_counter=int(self._hit_counter),
            reject_streak=int(self._reject_streak),
            used_bridge=bool(used_bridge),
        )

    def update(
        self,
        *,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> PhysicsInformedAssociationResult:
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
                profile_similarity=0.0,
                fusion_score=0.0,
                turbulence_score=self._turbulence_ema,
                strehl_proxy=0.0,
                used_bridge=False,
            )

        ts = self._resolve_timestamp(timestamp_s)
        gray = _to_gray(frame)
        patch = None
        if gray is not None:
            patch = _extract_patch(gray, center=(cx, cy), radius=int(self.config.patch_radius))
        embedding = build_patch_embedding(
            frame,
            (cx, cy),
            radius=int(self.config.patch_radius),
            bins=int(self.config.histogram_bins),
        )
        profile = _build_radial_profile(patch, int(self.config.profile_bins)) if patch is not None else None
        strehl_proxy = _estimate_strehl_proxy(patch)

        if self._state is None:
            self._state = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
            self._appearance_ref = embedding
            self._profile_ref = profile
            self._prev_timestamp = ts
            self._center_history.append(np.array([cx, cy], dtype=np.float64))
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
                profile_similarity=1.0 if profile is not None else 0.0,
                fusion_score=1.0,
                turbulence_score=0.0,
                strehl_proxy=strehl_proxy,
                used_bridge=False,
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
        profile_similarity = _cosine_similarity(profile, self._profile_ref)
        if profile_similarity is None:
            profile_similarity = 0.0

        turbulence = self._estimate_turbulence(dist, profile_similarity)
        gate = self._effective_gate(conf, speed, turbulence, profile_similarity)
        continuity = float(np.clip(1.0 - dist / max(gate, 1e-6), 0.0, 1.0))

        app_term = float(np.clip((appearance_similarity + 1.0) * 0.5, 0.0, 1.0))
        profile_term = float(np.clip((profile_similarity + 1.0) * 0.5, 0.0, 1.0))
        fusion = (
            float(max(0.0, self.config.continuity_weight)) * continuity
            + float(max(0.0, self.config.appearance_weight)) * app_term
            + float(max(0.0, self.config.profile_weight)) * profile_term
            + float(max(0.0, self.config.confidence_weight)) * conf
        )
        weight_sum = (
            float(max(0.0, self.config.continuity_weight))
            + float(max(0.0, self.config.appearance_weight))
            + float(max(0.0, self.config.profile_weight))
            + float(max(0.0, self.config.confidence_weight))
        )
        if weight_sum > 1e-6:
            fusion /= weight_sum
        fusion = float(np.clip(fusion, 0.0, 1.0))

        bridge_gate = gate
        used_bridge = False
        if conf <= float(self.config.low_confidence_threshold) or self._reject_streak > 0:
            bridge_gate = gate * float(max(1.0, self.config.bridge_gate_multiplier))
            used_bridge = dist > gate and dist <= bridge_gate

        accept = (
            dist <= bridge_gate
            and fusion >= float(self.config.min_fusion_score)
            and appearance_similarity >= float(self.config.min_similarity)
            and profile_similarity >= float(self.config.min_profile_similarity)
        )
        if conf >= float(self.config.high_confidence_threshold):
            accept = accept or (
                dist <= gate
                and fusion >= max(0.0, float(self.config.min_fusion_score) - 0.04)
                and profile_similarity >= float(self.config.min_profile_similarity) - 0.04
            )

        if not accept:
            self._reject_streak += 1
            self._accept_streak = 0
            self._hit_counter = max(0, self._hit_counter - int(max(1, self.config.miss_decay)))
            self._update_temperature(conf, dist, gate, turbulence)
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
                profile_similarity=profile_similarity,
                fusion_score=fusion,
                turbulence_score=turbulence,
                strehl_proxy=strehl_proxy,
                used_bridge=used_bridge,
            )

        alpha = float(np.clip(0.42 + 0.26 * continuity + 0.14 * profile_term + 0.12 * conf, 0.35, 0.92))
        rx = float(px + alpha * dx)
        ry = float(py + alpha * dy)
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
        self._center_history.append(np.array([rx, ry], dtype=np.float64))

        if embedding is not None:
            if self._appearance_ref is None:
                self._appearance_ref = embedding
            else:
                momentum = float(np.clip(self.config.appearance_momentum, 0.0, 1.0))
                merged = momentum * self._appearance_ref + (1.0 - momentum) * embedding
                self._appearance_ref = _normalize(merged)
        if profile is not None:
            if self._profile_ref is None:
                self._profile_ref = profile
            else:
                momentum = float(np.clip(self.config.profile_momentum, 0.0, 1.0))
                merged = momentum * self._profile_ref + (1.0 - momentum) * profile
                self._profile_ref = _normalize(merged)

        self._reject_streak = 0
        self._accept_streak += 1
        self._hit_counter = min(int(self.config.hit_counter_max), self._hit_counter + 1)
        self._update_temperature(conf, dist, gate, turbulence)

        confidence_factor = float(np.clip(0.92 + 0.08 * profile_term, 0.82, 1.06))
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
            profile_similarity=profile_similarity,
            fusion_score=fusion,
            turbulence_score=turbulence,
            strehl_proxy=strehl_proxy,
            used_bridge=used_bridge,
        )

