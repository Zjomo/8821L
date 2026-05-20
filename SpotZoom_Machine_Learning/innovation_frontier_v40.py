"""
SpotZoom innovation frontier v40.

Iterative phase-bridge association gate inspired by:
- OpenCV motion APIs (`phaseCorrelate` / `phaseCorrelateIterative`)
- CoTracker3 temporal track continuity
- ByteTrack / Norfair low-confidence bridge and hysteresis ideas

This module is intentionally lightweight and depends on numpy + opencv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional import guard
    cv2 = None


@dataclass
class IterativePhaseBridgeConfig:
    high_confidence_threshold: float = 0.68
    low_confidence_threshold: float = 0.16
    patch_radius: int = 22
    min_patch_std: float = 2.0
    response_threshold: float = 0.12
    response_bridge_threshold: float = 0.08
    base_gate_px: float = 13.5
    jitter_gate_scale: float = 1.25
    max_association_distance_px: float = 34.0
    max_refine_shift_px: float = 20.0
    min_refine_shift_px: float = 0.2
    bridge_gate_multiplier: float = 1.28
    phase_weight: float = 0.58
    confidence_gain: float = 1.02
    confidence_penalty: float = 0.94
    velocity_smoothing: float = 0.74
    max_velocity_px_per_s: float = 720.0
    jitter_decay: float = 0.84
    temperature_decay: float = 0.83
    min_dt_s: float = 1e-3
    max_dt_s: float = 0.45
    hit_counter_max: int = 30
    miss_decay: int = 1
    accept_hysteresis_hits: int = 2
    reject_hysteresis_misses: int = 3
    iterative_l2size: int = 7
    iterative_max_iters: int = 10
    use_hanning_window: bool = True


@dataclass
class IterativePhaseBridgeResult:
    center: Tuple[float, float]
    confidence: float
    accepted: bool
    refined: bool
    rejected: bool
    reason: Optional[str]
    shift: Tuple[float, float]
    response: float
    adaptive_gate_px: float
    shift_magnitude_px: float
    motion_score: float
    predicted_center: Tuple[float, float]
    jitter_px: float
    temperature: float
    hit_counter: int
    reject_streak: int
    used_bridge: bool
    used_iterative: bool


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if not np.isfinite(x):
        return default
    return x


class IterativePhaseBridgeGate:
    """Single-target iterative phase-correlation bridge gate."""

    def __init__(self, config: Optional[IterativePhaseBridgeConfig] = None):
        self.config = config or IterativePhaseBridgeConfig()
        self._prev_patch: Optional[np.ndarray] = None
        self._window_cache_shape: Optional[Tuple[int, int]] = None
        self._window_cache: Optional[np.ndarray] = None
        self._state: Optional[np.ndarray] = None  # [x, y, vx, vy]
        self._prev_timestamp: Optional[float] = None
        self._clock: float = 0.0
        self._jitter_ema: float = 0.0
        self._temperature: float = 1.0
        self._hit_counter: int = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak: int = 0
        self._reject_streak: int = 0

    def reset(self) -> None:
        self._prev_patch = None
        self._window_cache_shape = None
        self._window_cache = None
        self._state = None
        self._prev_timestamp = None
        self._clock = 0.0
        self._jitter_ema = 0.0
        self._temperature = 1.0
        self._hit_counter = max(1, int(self.config.hit_counter_max // 2))
        self._accept_streak = 0
        self._reject_streak = 0

    @staticmethod
    def _as_gray_float32(frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim == 3:
            if cv2 is None:
                return None
            try:
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            except Exception:
                return None
        elif arr.ndim == 2:
            gray = arr
        else:
            return None
        gray32 = np.asarray(gray, dtype=np.float32)
        if gray32.size == 0:
            return None
        return gray32

    def _extract_patch(self, gray: np.ndarray, center: Tuple[float, float]) -> Optional[np.ndarray]:
        h, w = gray.shape[:2]
        if h <= 0 or w <= 0:
            return None
        radius = max(2, int(self.config.patch_radius))
        size = radius * 2 + 1
        cx = _safe_float(center[0], default=np.nan)
        cy = _safe_float(center[1], default=np.nan)
        if not np.isfinite(cx) or not np.isfinite(cy):
            return None

        xi = int(round(cx))
        yi = int(round(cy))
        x0 = xi - radius
        x1 = xi + radius + 1
        y0 = yi - radius
        y1 = yi + radius + 1

        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - w)
        pad_bottom = max(0, y1 - h)
        if pad_left or pad_top or pad_right or pad_bottom:
            src = np.pad(gray, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
            x0 += pad_left
            x1 += pad_left
            y0 += pad_top
            y1 += pad_top
        else:
            src = gray

        patch = src[y0:y1, x0:x1]
        if patch.shape != (size, size):
            return None
        return np.asarray(patch, dtype=np.float32)

    def _get_window(self, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if not bool(self.config.use_hanning_window):
            return None
        if self._window_cache is not None and self._window_cache_shape == shape:
            return self._window_cache

        h, w = shape
        if h <= 1 or w <= 1:
            return None

        window = None
        if cv2 is not None:
            try:
                window = cv2.createHanningWindow((w, h), cv2.CV_32F)
            except Exception:
                window = None
        if window is None:
            wy = np.hanning(h).astype(np.float32)
            wx = np.hanning(w).astype(np.float32)
            window = np.outer(wy, wx).astype(np.float32)

        self._window_cache_shape = shape
        self._window_cache = window
        return window

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

    def _effective_gate(self, confidence: float) -> float:
        conf = float(np.clip(confidence, 0.0, 1.0))
        gate = float(max(0.0, self.config.base_gate_px))
        if conf < float(self.config.low_confidence_threshold):
            gate *= 1.1
        elif conf >= float(self.config.high_confidence_threshold):
            gate *= 0.94
        gate += float(max(0.0, self.config.jitter_gate_scale)) * float(max(0.0, self._jitter_ema))
        gate *= float(np.clip(self._temperature, 0.75, 2.4))
        gate = float(np.clip(gate, 2.0, max(2.0, float(self.config.max_association_distance_px))))
        return gate

    def _update_temperature(self, confidence: float, motion_score: float, response: float) -> float:
        decay = float(np.clip(self.config.temperature_decay, 0.0, 1.0))
        conf_term = 0.12 * (1.0 - float(np.clip(confidence, 0.0, 1.0)))
        motion_term = 0.24 * float(max(0.0, motion_score))
        response_term = 0.2 * (1.0 - float(np.clip(response, 0.0, 1.0)))
        rej_term = 0.05 * float(max(0, self._reject_streak))
        target = 1.0 + conf_term + motion_term + response_term + rej_term
        self._temperature = float(np.clip(decay * self._temperature + (1.0 - decay) * target, 0.7, 2.8))
        return self._temperature

    def _estimate_shift(
        self,
        prev_patch: np.ndarray,
        curr_patch: np.ndarray,
        window: Optional[np.ndarray],
    ) -> Tuple[float, float, float, bool]:
        if cv2 is None:
            return 0.0, 0.0, 0.0, False

        used_iterative = False
        dx = 0.0
        dy = 0.0
        response = 0.0

        if hasattr(cv2, "phaseCorrelateIterative"):
            try:
                shift = cv2.phaseCorrelateIterative(
                    prev_patch,
                    curr_patch,
                    int(max(1, self.config.iterative_l2size)),
                    int(max(1, self.config.iterative_max_iters)),
                )
                dx = _safe_float(shift[0], default=0.0)
                dy = _safe_float(shift[1], default=0.0)
                used_iterative = True
            except Exception:
                used_iterative = False

        if not used_iterative:
            shift_raw, response_raw = cv2.phaseCorrelate(prev_patch, curr_patch, window)
            dx = _safe_float(shift_raw[0], default=0.0)
            dy = _safe_float(shift_raw[1], default=0.0)
            response = float(np.clip(_safe_float(response_raw, default=0.0), 0.0, 1.0))
            return dx, dy, response, False

        try:
            _, response_raw = cv2.phaseCorrelate(prev_patch, curr_patch, window)
            response = float(np.clip(_safe_float(response_raw, default=0.0), 0.0, 1.0))
        except Exception:
            response = 0.0
        return dx, dy, response, True

    def update(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
        timestamp_s: Optional[float] = None,
    ) -> IterativePhaseBridgeResult:
        conf = float(np.clip(_safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = _safe_float(center[0], default=np.nan)
        cy = _safe_float(center[1], default=np.nan)
        ts = self._resolve_timestamp(timestamp_s)

        if not np.isfinite(cx) or not np.isfinite(cy):
            self._reject_streak += 1
            self._accept_streak = 0
            self._hit_counter = max(0, int(self._hit_counter) - int(max(1, self.config.miss_decay)))
            self._temperature = float(min(2.8, self._temperature + 0.12))
            pred = (0.0, 0.0) if self._state is None else (float(self._state[0]), float(self._state[1]))
            return IterativePhaseBridgeResult(
                center=pred,
                confidence=float(np.clip(conf * float(self.config.confidence_penalty), 0.0, 1.0)),
                accepted=False,
                refined=False,
                rejected=True,
                reason="invalid_center",
                shift=(0.0, 0.0),
                response=0.0,
                adaptive_gate_px=float(max(0.0, self.config.base_gate_px)),
                shift_magnitude_px=0.0,
                motion_score=1.0,
                predicted_center=pred,
                jitter_px=float(max(0.0, self._jitter_ema)),
                temperature=float(max(0.0, self._temperature)),
                hit_counter=int(self._hit_counter),
                reject_streak=int(self._reject_streak),
                used_bridge=False,
                used_iterative=False,
            )

        gray = self._as_gray_float32(frame)
        if gray is None or cv2 is None:
            return IterativePhaseBridgeResult(
                center=(cx, cy),
                confidence=conf,
                accepted=False,
                refined=False,
                rejected=True,
                reason="opencv_or_frame_unavailable",
                shift=(0.0, 0.0),
                response=0.0,
                adaptive_gate_px=float(max(0.0, self.config.base_gate_px)),
                shift_magnitude_px=0.0,
                motion_score=0.0,
                predicted_center=(cx, cy),
                jitter_px=float(max(0.0, self._jitter_ema)),
                temperature=float(max(0.0, self._temperature)),
                hit_counter=int(self._hit_counter),
                reject_streak=int(self._reject_streak),
                used_bridge=False,
                used_iterative=False,
            )

        curr_patch = self._extract_patch(gray, (cx, cy))
        if curr_patch is None:
            return IterativePhaseBridgeResult(
                center=(cx, cy),
                confidence=conf,
                accepted=False,
                refined=False,
                rejected=True,
                reason="invalid_patch",
                shift=(0.0, 0.0),
                response=0.0,
                adaptive_gate_px=float(max(0.0, self.config.base_gate_px)),
                shift_magnitude_px=0.0,
                motion_score=0.0,
                predicted_center=(cx, cy),
                jitter_px=float(max(0.0, self._jitter_ema)),
                temperature=float(max(0.0, self._temperature)),
                hit_counter=int(self._hit_counter),
                reject_streak=int(self._reject_streak),
                used_bridge=False,
                used_iterative=False,
            )

        patch_std = float(np.std(curr_patch))
        if self._prev_patch is None or self._state is None:
            self._prev_patch = curr_patch.copy()
            self._state = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
            self._prev_timestamp = ts
            self._accept_streak = 1
            self._reject_streak = 0
            self._temperature = 1.0
            self._hit_counter = max(1, self._hit_counter)
            return IterativePhaseBridgeResult(
                center=(cx, cy),
                confidence=conf,
                accepted=True,
                refined=False,
                rejected=False,
                reason=None,
                shift=(0.0, 0.0),
                response=1.0,
                adaptive_gate_px=float(max(0.0, self.config.base_gate_px)),
                shift_magnitude_px=0.0,
                motion_score=0.0,
                predicted_center=(cx, cy),
                jitter_px=float(max(0.0, self._jitter_ema)),
                temperature=float(max(0.0, self._temperature)),
                hit_counter=int(self._hit_counter),
                reject_streak=0,
                used_bridge=False,
                used_iterative=False,
            )

        prev_ts = float(self._prev_timestamp) if self._prev_timestamp is not None else (ts - 1.0 / 30.0)
        dt = float(np.clip(ts - prev_ts, float(self.config.min_dt_s), float(self.config.max_dt_s)))
        self._prev_timestamp = ts

        pred_x = float(self._state[0] + self._state[2] * dt)
        pred_y = float(self._state[1] + self._state[3] * dt)
        predicted = np.array([pred_x, pred_y], dtype=np.float64)

        if patch_std < max(0.0, float(self.config.min_patch_std)):
            self._prev_patch = curr_patch.copy()
            self._reject_streak += 1
            self._accept_streak = 0
            self._hit_counter = max(0, int(self._hit_counter) - int(max(1, self.config.miss_decay)))
            penalized = float(np.clip(conf * float(self.config.confidence_penalty), 0.0, 1.0))
            self._update_temperature(penalized, 0.0, 0.0)
            chosen = predicted if self._reject_streak >= int(max(1, self.config.reject_hysteresis_misses)) else np.array([cx, cy], dtype=np.float64)
            return IterativePhaseBridgeResult(
                center=(float(chosen[0]), float(chosen[1])),
                confidence=penalized,
                accepted=False,
                refined=False,
                rejected=True,
                reason="patch_low_texture",
                shift=(0.0, 0.0),
                response=0.0,
                adaptive_gate_px=float(max(0.0, self.config.base_gate_px)),
                shift_magnitude_px=0.0,
                motion_score=0.0,
                predicted_center=(float(predicted[0]), float(predicted[1])),
                jitter_px=float(max(0.0, self._jitter_ema)),
                temperature=float(max(0.0, self._temperature)),
                hit_counter=int(self._hit_counter),
                reject_streak=int(self._reject_streak),
                used_bridge=False,
                used_iterative=False,
            )

        window = self._get_window(curr_patch.shape)
        dx, dy, response, used_iterative = self._estimate_shift(self._prev_patch, curr_patch, window)
        shift_mag = float(np.hypot(dx, dy))
        self._jitter_ema = float(
            np.clip(
                float(self.config.jitter_decay) * float(self._jitter_ema)
                + (1.0 - float(self.config.jitter_decay)) * shift_mag,
                0.0,
                128.0,
            )
        )

        gate_px = self._effective_gate(conf)
        max_dist = float(max(0.0, self.config.max_association_distance_px))
        if max_dist > 0.0:
            gate_px = float(min(gate_px, max_dist))
        if gate_px <= 1e-6:
            gate_px = 1e-6
        motion_score = float(shift_mag / gate_px)
        low_conf = conf < float(np.clip(self.config.low_confidence_threshold, 0.0, 1.0))
        high_conf = conf >= float(np.clip(self.config.high_confidence_threshold, 0.0, 1.0))
        bridge_gate = float(gate_px * max(1.0, float(self.config.bridge_gate_multiplier)))

        reason: Optional[str] = None
        accepted = True
        used_bridge = False

        if response < float(np.clip(self.config.response_threshold, 0.0, 1.0)):
            if low_conf and response >= float(np.clip(self.config.response_bridge_threshold, 0.0, 1.0)):
                used_bridge = True
            else:
                accepted = False
                reason = "phase_response_too_low"

        if accepted and shift_mag > gate_px:
            if low_conf and shift_mag <= bridge_gate:
                used_bridge = True
            else:
                accepted = False
                reason = "phase_shift_gate_exceeded"

        if accepted and high_conf and motion_score > 1.2:
            accepted = False
            reason = "high_conf_motion_inconsistent"

        measurement = np.array([cx, cy], dtype=np.float64)
        phase_center = np.array([float(self._state[0]) + dx, float(self._state[1]) + dy], dtype=np.float64)
        phase_w = float(np.clip(float(self.config.phase_weight) * (0.65 + 0.35 * response), 0.05, 0.95))
        fused = phase_w * phase_center + (1.0 - phase_w) * measurement
        refine_shift = float(np.linalg.norm(fused - measurement))
        min_refine = float(max(0.0, self.config.min_refine_shift_px))

        if accepted and refine_shift > float(max(0.0, self.config.max_refine_shift_px)):
            accepted = False
            reason = "refine_shift_too_large"

        if accepted:
            if refine_shift < min_refine:
                fused = measurement
            refined = bool(np.linalg.norm(fused - measurement) >= min_refine)

            meas_vx = (float(fused[0]) - float(self._state[0])) / max(dt, 1e-6)
            meas_vy = (float(fused[1]) - float(self._state[1])) / max(dt, 1e-6)
            vx = float(self.config.velocity_smoothing) * float(self._state[2]) + (1.0 - float(self.config.velocity_smoothing)) * meas_vx
            vy = float(self.config.velocity_smoothing) * float(self._state[3]) + (1.0 - float(self.config.velocity_smoothing)) * meas_vy
            vx, vy = self._clip_velocity(vx, vy)
            self._state = np.array([float(fused[0]), float(fused[1]), vx, vy], dtype=np.float64)
            self._prev_patch = curr_patch.copy()

            self._accept_streak += 1
            self._reject_streak = 0
            self._hit_counter = min(int(max(1, self.config.hit_counter_max)), int(self._hit_counter) + 1)
            if self._accept_streak < int(max(1, self.config.accept_hysteresis_hits)):
                conf_out = conf
            else:
                conf_out = float(np.clip(conf * float(self.config.confidence_gain), 0.0, 1.0))
            self._update_temperature(conf_out, motion_score, response)
            return IterativePhaseBridgeResult(
                center=(float(self._state[0]), float(self._state[1])),
                confidence=conf_out,
                accepted=True,
                refined=refined,
                rejected=False,
                reason=None,
                shift=(float(dx), float(dy)),
                response=float(response),
                adaptive_gate_px=float(gate_px),
                shift_magnitude_px=float(shift_mag),
                motion_score=float(max(0.0, motion_score)),
                predicted_center=(float(predicted[0]), float(predicted[1])),
                jitter_px=float(max(0.0, self._jitter_ema)),
                temperature=float(max(0.0, self._temperature)),
                hit_counter=int(self._hit_counter),
                reject_streak=0,
                used_bridge=bool(used_bridge),
                used_iterative=bool(used_iterative),
            )

        self._prev_patch = curr_patch.copy()
        self._accept_streak = 0
        self._reject_streak += 1
        self._hit_counter = max(0, int(self._hit_counter) - int(max(1, self.config.miss_decay)))
        penalized = float(np.clip(conf * float(self.config.confidence_penalty), 0.0, 1.0))
        vx, vy = self._clip_velocity(float(self._state[2]), float(self._state[3]))
        self._state = np.array([float(predicted[0]), float(predicted[1]), vx, vy], dtype=np.float64)
        self._update_temperature(penalized, motion_score, response)

        hard_reject = self._reject_streak >= int(max(1, self.config.reject_hysteresis_misses))
        chosen = predicted if hard_reject else measurement
        return IterativePhaseBridgeResult(
            center=(float(chosen[0]), float(chosen[1])),
            confidence=penalized,
            accepted=False,
            refined=False,
            rejected=True,
            reason=reason or "rejected",
            shift=(float(dx), float(dy)),
            response=float(response),
            adaptive_gate_px=float(gate_px),
            shift_magnitude_px=float(shift_mag),
            motion_score=float(max(0.0, motion_score)),
            predicted_center=(float(predicted[0]), float(predicted[1])),
            jitter_px=float(max(0.0, self._jitter_ema)),
            temperature=float(max(0.0, self._temperature)),
            hit_counter=int(self._hit_counter),
            reject_streak=int(self._reject_streak),
            used_bridge=bool(used_bridge),
            used_iterative=bool(used_iterative),
        )
