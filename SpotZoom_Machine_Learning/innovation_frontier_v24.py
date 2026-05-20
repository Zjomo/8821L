"""
Frontier v24 modules for SpotZoom.

This module adds lightweight, dependency-free components inspired by:
- SAM 2 streaming memory for real-time video processing.
- DeepTrack2 simulation-driven robustness practices.
- Differentiable optics toolchains (TorchOptics, TurPy) for physics-aware constraints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Deque, Optional, Tuple

import cv2
import numpy as np


DetectionFn = Callable[[np.ndarray], Optional[Tuple[Tuple[int, int], float]]]


@dataclass
class StreamingMemoryConfig:
    template_half_size: int = 24
    search_radius: int = 56
    min_template_variance: float = 4.0
    match_threshold: float = 0.62
    max_templates: int = 6


@dataclass
class StreamingRecoveryResult:
    center: Optional[Tuple[int, int]]
    score: float
    accepted: bool
    reason: str


class StreamingMemoryTracker:
    """Template-memory tracker for detector-miss recovery."""

    def __init__(self, config: Optional[StreamingMemoryConfig] = None):
        self.config = config or StreamingMemoryConfig()
        max_templates = max(1, int(self.config.max_templates))
        self._templates: Deque[np.ndarray] = deque(maxlen=max_templates)
        self._last_center: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        self._templates.clear()
        self._last_center = None

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] >= 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.squeeze(frame).astype(np.uint8)

    def _extract_patch(self, gray: np.ndarray, center: Tuple[int, int]) -> Optional[np.ndarray]:
        h, w = gray.shape[:2]
        half = max(4, int(self.config.template_half_size))
        x = int(center[0])
        y = int(center[1])
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w, x + half + 1)
        y2 = min(h, y + half + 1)
        if x2 - x1 < 9 or y2 - y1 < 9:
            return None
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        return patch

    def update(self, frame: np.ndarray, center: Tuple[int, int]) -> bool:
        gray = self._to_gray(frame)
        patch = self._extract_patch(gray, center)
        if patch is None:
            return False

        variance = float(np.var(patch))
        if variance < float(self.config.min_template_variance):
            self._last_center = (int(center[0]), int(center[1]))
            return False

        template = patch.astype(np.float32)
        template -= float(np.mean(template))
        denom = float(np.std(template)) + 1e-6
        template /= denom
        self._templates.append(template)
        self._last_center = (int(center[0]), int(center[1]))
        return True

    def recover(self, frame: np.ndarray) -> StreamingRecoveryResult:
        if not self._templates or self._last_center is None:
            return StreamingRecoveryResult(center=None, score=0.0, accepted=False, reason="no_memory")

        gray = self._to_gray(frame).astype(np.float32)
        h, w = gray.shape[:2]
        radius = max(8, int(self.config.search_radius))
        cx, cy = int(self._last_center[0]), int(self._last_center[1])
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(w, cx + radius + 1)
        y2 = min(h, cy + radius + 1)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return StreamingRecoveryResult(center=None, score=0.0, accepted=False, reason="empty_roi")

        best_score = -1.0
        best_center: Optional[Tuple[int, int]] = None
        for template in self._templates:
            th, tw = template.shape[:2]
            if roi.shape[0] < th or roi.shape[1] < tw:
                continue
            response = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(response)
            if float(max_val) > best_score:
                px = x1 + int(max_loc[0]) + tw // 2
                py = y1 + int(max_loc[1]) + th // 2
                best_center = (int(px), int(py))
                best_score = float(max_val)

        if best_center is None:
            return StreamingRecoveryResult(center=None, score=0.0, accepted=False, reason="no_valid_template")

        threshold = float(self.config.match_threshold)
        accepted = bool(best_score >= threshold)
        if accepted:
            self._last_center = best_center
            return StreamingRecoveryResult(center=best_center, score=best_score, accepted=True, reason="matched")
        return StreamingRecoveryResult(center=best_center, score=best_score, accepted=False, reason="low_score")


@dataclass
class DomainRobustnessConfig:
    num_probes: int = 4
    noise_std: float = 2.5
    brightness_jitter: float = 0.08
    blur_probability: float = 0.35
    blur_kernel: int = 3
    max_center_std_px: float = 2.5
    max_conf_drop: float = 0.18
    min_valid_ratio: float = 0.5


@dataclass
class DomainRobustnessResult:
    center_std_px: float
    mean_confidence: float
    valid_ratio: float
    unstable: bool
    reason: str


class DomainRobustnessProbe:
    """Detector stability probe under synthetic domain perturbations."""

    def __init__(self, config: Optional[DomainRobustnessConfig] = None, seed: int = 20260513):
        self.config = config or DomainRobustnessConfig()
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(20260513)

    def _augment(self, frame: np.ndarray) -> np.ndarray:
        probe = frame.astype(np.float32)
        if self.config.brightness_jitter > 0:
            alpha = 1.0 + float(self._rng.uniform(-self.config.brightness_jitter, self.config.brightness_jitter))
            beta = float(self._rng.uniform(-self.config.brightness_jitter, self.config.brightness_jitter)) * 255.0
            probe = probe * alpha + beta
        noise_std = max(0.0, float(self.config.noise_std))
        if noise_std > 0:
            probe += self._rng.normal(0.0, noise_std, probe.shape).astype(np.float32)
        if float(self._rng.random()) < float(self.config.blur_probability):
            k = max(3, int(self.config.blur_kernel))
            if k % 2 == 0:
                k += 1
            probe = cv2.GaussianBlur(probe, (k, k), 0)
        return np.clip(probe, 0.0, 255.0).astype(np.uint8)

    def evaluate(
        self,
        frame: np.ndarray,
        base_center: Tuple[int, int],
        base_confidence: float,
        detect_fn: DetectionFn,
    ) -> DomainRobustnessResult:
        num_probes = max(1, int(self.config.num_probes))
        centers = []
        confidences = []
        for _ in range(num_probes):
            probe = self._augment(frame)
            result = detect_fn(probe)
            if result is None:
                continue
            center, conf = result
            centers.append((float(center[0]), float(center[1])))
            confidences.append(float(conf))

        valid_ratio = len(centers) / float(num_probes)
        if not centers:
            return DomainRobustnessResult(
                center_std_px=float("inf"),
                mean_confidence=0.0,
                valid_ratio=0.0,
                unstable=True,
                reason="all_failed",
            )

        center_arr = np.array(centers, dtype=np.float64)
        mean_center = np.mean(center_arr, axis=0)
        center_std_px = float(np.linalg.norm(mean_center - np.array(base_center, dtype=np.float64)))
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        conf_drop = max(0.0, float(base_confidence) - mean_conf)

        unstable = (
            center_std_px > float(self.config.max_center_std_px)
            or conf_drop > float(self.config.max_conf_drop)
            or valid_ratio < float(self.config.min_valid_ratio)
        )
        reasons = []
        if center_std_px > float(self.config.max_center_std_px):
            reasons.append("center_shift")
        if conf_drop > float(self.config.max_conf_drop):
            reasons.append("conf_drop")
        if valid_ratio < float(self.config.min_valid_ratio):
            reasons.append("low_valid_ratio")
        reason = ",".join(reasons) if reasons else "stable"
        return DomainRobustnessResult(
            center_std_px=center_std_px,
            mean_confidence=mean_conf,
            valid_ratio=valid_ratio,
            unstable=unstable,
            reason=reason,
        )


@dataclass
class PhysicsStepGuardConfig:
    base_max_step: int = 2000
    min_limit: int = 250
    max_limit: int = 4000
    jitter_gain: float = 0.03
    focus_reference: float = 35.0
    focus_gain: float = 0.45


@dataclass
class StepGuardResult:
    requested_step: int
    clamped_step: int
    dynamic_limit: int
    clamped: bool


class PhysicsStepGuard:
    """Clamp control steps with a lightweight physics-aware rule."""

    def __init__(self, config: Optional[PhysicsStepGuardConfig] = None):
        self.config = config or PhysicsStepGuardConfig()

    def reset(self) -> None:
        return

    def clamp(self, requested_step: int, jitter_px: float, focus_score: float) -> StepGuardResult:
        req = int(requested_step)
        req_mag = abs(req)
        sign = 1 if req >= 0 else -1

        jitter_term = 1.0 / (1.0 + max(0.0, float(jitter_px)) * float(self.config.jitter_gain))
        focus_delta = (max(0.0, float(focus_score)) - float(self.config.focus_reference)) / max(
            float(self.config.focus_reference), 1.0
        )
        focus_term = 1.0 + float(self.config.focus_gain) * math.tanh(focus_delta)
        raw_limit = int(round(float(self.config.base_max_step) * jitter_term * focus_term))
        limit = max(int(self.config.min_limit), min(int(self.config.max_limit), raw_limit))

        clamped_mag = min(req_mag, limit)
        clamped_step = sign * int(clamped_mag)
        return StepGuardResult(
            requested_step=req,
            clamped_step=clamped_step,
            dynamic_limit=limit,
            clamped=bool(clamped_mag != req_mag),
        )

