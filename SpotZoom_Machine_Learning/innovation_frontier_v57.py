"""
SpotZoom innovation frontier v57.

Visibility-calibrated persistence association gate inspired by:
- TAPIR / TAPNext++: visibility-aware long-range point tracking and re-entry.
- Track-On-R: verifier-guided reliability with compact memory.
- CoTracker3: online consistency under long temporal horizon.
- SAM2: streaming memory state for online video decisions.
- Norfair: lightweight distance-first modular association.

This module intentionally keeps a numpy-only runtime dependency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np


def _load_v56_primitives():
    try:
        from SpotZoom_Machine_Learning.innovation_frontier_v56 import (  # type: ignore
            OcclusionTreeAssociationConfig as _Cfg,
            OcclusionTreeAssociationGate as _Gate,
        )
        return _Cfg, _Gate
    except Exception:
        pass

    module_path = Path(__file__).resolve().with_name("innovation_frontier_v56.py")
    spec = importlib.util.spec_from_file_location("_spotzoom_ml_v56_fallback", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load innovation_frontier_v56 primitives")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg_cls = getattr(module, "OcclusionTreeAssociationConfig", None)
    gate_cls = getattr(module, "OcclusionTreeAssociationGate", None)
    if cfg_cls is None or gate_cls is None:
        raise ImportError("innovation_frontier_v56 missing expected classes")
    return cfg_cls, gate_cls


_V56Config, _V56Gate = _load_v56_primitives()


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _safe_center(raw, fallback: np.ndarray) -> np.ndarray:
    if not isinstance(raw, (tuple, list)) or len(raw) < 2:
        return fallback.copy()
    x = _safe_float(raw[0], float(fallback[0]))
    y = _safe_float(raw[1], float(fallback[1]))
    arr = np.asarray([x, y], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return fallback.copy()
    return arr


@dataclass
class OcclusionTreeAssociationConfig:
    high_confidence_threshold: float = 0.72
    low_confidence_threshold: float = 0.16
    base_gate_px: float = 12.4
    uncertainty_gate_scale: float = 0.96
    cmc_gate_scale: float = 0.80
    occlusion_gate_scale: float = 0.76
    short_memory_budget: int = 14
    long_memory_budget: int = 28
    occlusion_reupdate_threshold: float = 0.40
    max_association_distance_px: float = 62.0
    max_refine_shift_px: float = 24.0
    confidence_gain: float = 1.04
    reliability_ema_decay: float = 0.88
    reliability_reward: float = 0.14
    reliability_penalty: float = 0.18
    reliability_gate_scale: float = 0.55
    reliability_confidence_blend: float = 0.12
    jitter_window: int = 6
    jitter_gate_scale: float = 0.35
    jitter_reject_threshold: float = 0.92
    rescue_memory_similarity: float = 0.36
    rescue_occlusion_threshold: float = 0.34
    rescue_fusion_threshold: float = 0.28
    rescue_blend: float = 0.62
    verifier_ema_decay: float = 0.86
    verifier_min_score: float = 0.24
    verifier_margin: float = 0.06
    disagreement_gate_scale: float = 0.52
    disagreement_reject_threshold: float = 1.05
    consensus_blend: float = 0.58
    rescue_verifier_bonus: float = 0.10
    confidence_temperature: float = 0.18
    verifier_history: int = 8
    teacher_temperature: float = 0.33
    teacher_entropy_reject_threshold: float = 0.88
    teacher_agreement_min: float = 0.28
    teacher_consensus_blend: float = 0.44
    teacher_rescue_gain: float = 0.08
    teacher_history: int = 12
    visibility_min_score: float = 0.28
    visibility_entropy_penalty: float = 0.58
    visibility_drift_penalty: float = 0.44
    drift_guard_ratio: float = 1.06
    reentry_visibility_margin: float = 0.10
    reentry_verifier_min: float = 0.30
    persistence_history: int = 14
    persistence_center_inertia: float = 0.64
    stability_floor: float = 0.24
    stability_relax_gain: float = 0.22


@dataclass
class OcclusionTreeAssociationResult:
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
    occlusion_score: float
    used_cmc: bool
    used_long_memory: bool
    used_tree_memory: bool
    tree_path: str
    tree_switched: bool
    observation_reupdated: bool
    scene_reset: bool
    memory_size: int
    memory_update_action: str
    innovation_score: float
    reliability_score: float
    jitter_score: float
    used_rescue_path: bool
    reliability_recovered: bool
    verifier_score: float
    disagreement_score: float
    consensus_center: Tuple[float, float]
    verifier_rescue_used: bool
    disagreement_rejected: bool
    teacher_entropy: float
    teacher_agreement: float
    teacher_consensus_center: Tuple[float, float]
    teacher_rescue_used: bool
    teacher_rejected: bool
    visibility_score: float
    drift_ratio: float
    memory_stability: float
    persistence_center: Tuple[float, float]
    persistent_reentry_used: bool
    stability_relaxed: bool


class OcclusionTreeAssociationGate:
    """v57 wrapper on top of v56 with visibility-calibrated persistence gating."""

    def __init__(self, config: Optional[OcclusionTreeAssociationConfig] = None):
        self.config = config or OcclusionTreeAssociationConfig()
        v56_cfg = _V56Config(
            high_confidence_threshold=float(self.config.high_confidence_threshold),
            low_confidence_threshold=float(self.config.low_confidence_threshold),
            base_gate_px=float(self.config.base_gate_px),
            uncertainty_gate_scale=float(self.config.uncertainty_gate_scale),
            cmc_gate_scale=float(self.config.cmc_gate_scale),
            occlusion_gate_scale=float(self.config.occlusion_gate_scale),
            short_memory_budget=max(4, int(self.config.short_memory_budget)),
            long_memory_budget=max(max(4, int(self.config.short_memory_budget)), int(self.config.long_memory_budget)),
            occlusion_reupdate_threshold=float(self.config.occlusion_reupdate_threshold),
            max_association_distance_px=float(self.config.max_association_distance_px),
            max_refine_shift_px=float(self.config.max_refine_shift_px),
            confidence_gain=float(self.config.confidence_gain),
            reliability_ema_decay=float(self.config.reliability_ema_decay),
            reliability_reward=float(self.config.reliability_reward),
            reliability_penalty=float(self.config.reliability_penalty),
            reliability_gate_scale=float(self.config.reliability_gate_scale),
            reliability_confidence_blend=float(self.config.reliability_confidence_blend),
            jitter_window=max(3, int(self.config.jitter_window)),
            jitter_gate_scale=float(self.config.jitter_gate_scale),
            jitter_reject_threshold=float(self.config.jitter_reject_threshold),
            rescue_memory_similarity=float(self.config.rescue_memory_similarity),
            rescue_occlusion_threshold=float(self.config.rescue_occlusion_threshold),
            rescue_fusion_threshold=float(self.config.rescue_fusion_threshold),
            rescue_blend=float(self.config.rescue_blend),
            verifier_ema_decay=float(self.config.verifier_ema_decay),
            verifier_min_score=float(self.config.verifier_min_score),
            verifier_margin=float(self.config.verifier_margin),
            disagreement_gate_scale=float(self.config.disagreement_gate_scale),
            disagreement_reject_threshold=float(self.config.disagreement_reject_threshold),
            consensus_blend=float(self.config.consensus_blend),
            rescue_verifier_bonus=float(self.config.rescue_verifier_bonus),
            confidence_temperature=float(self.config.confidence_temperature),
            verifier_history=max(3, int(self.config.verifier_history)),
            teacher_temperature=float(self.config.teacher_temperature),
            teacher_entropy_reject_threshold=float(self.config.teacher_entropy_reject_threshold),
            teacher_agreement_min=float(self.config.teacher_agreement_min),
            teacher_consensus_blend=float(self.config.teacher_consensus_blend),
            teacher_rescue_gain=float(self.config.teacher_rescue_gain),
            teacher_history=max(4, int(self.config.teacher_history)),
        )
        self._base_gate = _V56Gate(config=v56_cfg)
        self._center_history: Deque[np.ndarray] = deque(maxlen=max(4, int(self.config.persistence_history)))
        self._visibility_history: Deque[float] = deque(maxlen=max(4, int(self.config.persistence_history)))
        self._visibility_ema: float = 0.56

    def reset(self) -> None:
        self._base_gate.reset()
        self._center_history = deque(maxlen=max(4, int(self.config.persistence_history)))
        self._visibility_history = deque(maxlen=max(4, int(self.config.persistence_history)))
        self._visibility_ema = 0.56

    def _history_center(self, fallback: np.ndarray) -> np.ndarray:
        if not self._center_history:
            return fallback.copy()
        arr = np.asarray(tuple(self._center_history), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return fallback.copy()
        out = np.mean(arr, axis=0)
        if not np.all(np.isfinite(out)):
            return fallback.copy()
        return out

    def _memory_stability(self, gate_px: float) -> float:
        if len(self._center_history) < 2:
            return 1.0
        arr = np.asarray(tuple(self._center_history), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return 0.0
        center = np.mean(arr, axis=0)
        spread = float(np.mean(np.linalg.norm(arr - center, axis=1)))
        gate = max(1e-6, float(gate_px))
        score = 1.0 - spread / max(1.0, gate)
        return float(np.clip(score, 0.0, 1.0))

    def update(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
        *,
        timestamp_s: Optional[float] = None,
    ) -> OcclusionTreeAssociationResult:
        base = self._base_gate.update(
            frame=frame,
            center=center,
            confidence=confidence,
            timestamp_s=timestamp_s,
        )

        obs = np.asarray([_safe_float(center[0], 0.0), _safe_float(center[1], 0.0)], dtype=np.float64)
        pred = _safe_center(getattr(base, "predicted_center", center), obs)
        out_center = _safe_center(getattr(base, "center", center), obs)
        gate = max(1e-6, _safe_float(getattr(base, "adaptive_gate_px", self.config.base_gate_px), self.config.base_gate_px))
        assoc_dist = _safe_float(
            getattr(base, "association_distance_px", float(np.linalg.norm(out_center - pred))),
            float(np.linalg.norm(out_center - pred)),
        )
        drift_ratio = float(np.clip(assoc_dist / gate, 0.0, 4.0))

        teacher_entropy = float(np.clip(_safe_float(getattr(base, "teacher_entropy", 1.0), 1.0), 0.0, 1.0))
        teacher_agreement = float(np.clip(_safe_float(getattr(base, "teacher_agreement", 0.0), 0.0), 0.0, 1.0))
        verifier_score = float(np.clip(_safe_float(getattr(base, "verifier_score", 0.0), 0.0), 0.0, 1.0))
        innovation_score = float(np.clip(_safe_float(getattr(base, "innovation_score", 0.0), 0.0), 0.0, 1.0))

        visibility = (
            0.30 * (1.0 - teacher_entropy)
            + 0.26 * teacher_agreement
            + 0.22 * verifier_score
            + 0.22 * innovation_score
        )
        visibility -= float(self.config.visibility_entropy_penalty) * max(0.0, teacher_entropy - 0.82)
        visibility -= float(self.config.visibility_drift_penalty) * max(0.0, drift_ratio - 1.0)
        visibility = float(np.clip(visibility, 0.0, 1.0))
        self._visibility_ema = float(np.clip(0.82 * self._visibility_ema + 0.18 * visibility, 0.0, 1.0))
        visibility_score = float(np.clip(0.58 * visibility + 0.42 * self._visibility_ema, 0.0, 1.0))

        memory_stability = self._memory_stability(gate_px=gate)
        stability_relaxed = bool(memory_stability >= float(self.config.stability_floor))
        relaxed_visibility_min = float(np.clip(self.config.visibility_min_score, 0.0, 1.0))
        if stability_relaxed:
            relaxed_visibility_min = max(
                0.0,
                relaxed_visibility_min - float(np.clip(self.config.stability_relax_gain, 0.0, 0.6)) * memory_stability,
            )

        accepted = bool(not getattr(base, "rejected", False))
        rejected = bool(not accepted)
        reason = str(getattr(base, "reason", "rejected")) if rejected else None
        out_conf = float(np.clip(_safe_float(getattr(base, "confidence", confidence), confidence), 0.0, 1.0))

        hard_drift = drift_ratio > float(max(0.1, self.config.drift_guard_ratio))
        low_visibility = visibility_score < relaxed_visibility_min
        low_verifier = verifier_score < float(np.clip(self.config.reentry_verifier_min, 0.0, 1.0))
        persistent_reentry_used = False

        if accepted and hard_drift and low_visibility:
            accepted = False
            rejected = True
            reason = "visibility_drift_rejected"

        if rejected and (not low_verifier):
            margin = float(np.clip(self.config.reentry_visibility_margin, 0.0, 0.5))
            if visibility_score >= min(1.0, relaxed_visibility_min + margin) and drift_ratio <= float(self.config.drift_guard_ratio) * 1.3:
                history_center = self._history_center(fallback=pred)
                inertia = float(np.clip(self.config.persistence_center_inertia, 0.0, 1.0))
                out_center = inertia * history_center + (1.0 - inertia) * out_center
                if not np.all(np.isfinite(out_center)):
                    out_center = _safe_center(getattr(base, "center", center), obs)
                assoc_dist = float(np.linalg.norm(out_center - pred))
                drift_ratio = float(np.clip(assoc_dist / gate, 0.0, 4.0))
                accepted = True
                rejected = False
                reason = None
                persistent_reentry_used = True

        if accepted:
            conf_gain = 0.08 * (visibility_score - 0.5) + 0.06 * (memory_stability - 0.5)
            out_conf = float(np.clip(out_conf + conf_gain, 0.0, 1.0))
            self._center_history.append(np.asarray([float(out_center[0]), float(out_center[1])], dtype=np.float64))
            self._visibility_history.append(float(visibility_score))
        else:
            out_conf = float(np.clip(out_conf * (0.84 + 0.12 * visibility_score), 0.0, 1.0))

        persistence_center = self._history_center(fallback=out_center)
        consensus_raw = getattr(base, "consensus_center", (out_center[0], out_center[1]))
        consensus = _safe_center(consensus_raw, out_center)

        return OcclusionTreeAssociationResult(
            center=(float(out_center[0]), float(out_center[1])),
            confidence=out_conf,
            accepted=accepted,
            refined=bool(getattr(base, "refined", accepted)),
            rejected=not accepted,
            reason=reason,
            predicted_center=(float(pred[0]), float(pred[1])),
            association_distance_px=float(assoc_dist),
            adaptive_gate_px=float(gate),
            fusion_score=float(np.clip(_safe_float(getattr(base, "fusion_score", 0.0), 0.0), 0.0, 1.0)),
            memory_similarity=float(np.clip(_safe_float(getattr(base, "memory_similarity", 0.0), 0.0), 0.0, 1.0)),
            motion_consistency=float(_safe_float(getattr(base, "motion_consistency", 0.0), 0.0)),
            cmc_dx_px=float(_safe_float(getattr(base, "cmc_dx_px", 0.0), 0.0)),
            cmc_dy_px=float(_safe_float(getattr(base, "cmc_dy_px", 0.0), 0.0)),
            cmc_norm_px=float(_safe_float(getattr(base, "cmc_norm_px", 0.0), 0.0)),
            occlusion_score=float(np.clip(_safe_float(getattr(base, "occlusion_score", 0.0), 0.0), 0.0, 1.0)),
            used_cmc=bool(getattr(base, "used_cmc", False)),
            used_long_memory=bool(getattr(base, "used_long_memory", False)),
            used_tree_memory=bool(getattr(base, "used_tree_memory", False)),
            tree_path=str(getattr(base, "tree_path", "none")),
            tree_switched=bool(getattr(base, "tree_switched", False)),
            observation_reupdated=bool(getattr(base, "observation_reupdated", False)),
            scene_reset=bool(getattr(base, "scene_reset", False)),
            memory_size=int(getattr(base, "memory_size", 0) or 0),
            memory_update_action=str(getattr(base, "memory_update_action", "skip")),
            innovation_score=innovation_score,
            reliability_score=float(np.clip(_safe_float(getattr(base, "reliability_score", 0.0), 0.0), 0.0, 1.0)),
            jitter_score=float(np.clip(_safe_float(getattr(base, "jitter_score", 0.0), 0.0), 0.0, 1.5)),
            used_rescue_path=bool(getattr(base, "used_rescue_path", False)),
            reliability_recovered=bool(getattr(base, "reliability_recovered", False)),
            verifier_score=verifier_score,
            disagreement_score=float(np.clip(_safe_float(getattr(base, "disagreement_score", 0.0), 0.0), 0.0, 2.5)),
            consensus_center=(float(consensus[0]), float(consensus[1])),
            verifier_rescue_used=bool(getattr(base, "verifier_rescue_used", False)),
            disagreement_rejected=bool(getattr(base, "disagreement_rejected", False)),
            teacher_entropy=teacher_entropy,
            teacher_agreement=teacher_agreement,
            teacher_consensus_center=tuple(
                _safe_center(getattr(base, "teacher_consensus_center", out_center), out_center).tolist()
            ),
            teacher_rescue_used=bool(getattr(base, "teacher_rescue_used", False)),
            teacher_rejected=bool(getattr(base, "teacher_rejected", False)),
            visibility_score=visibility_score,
            drift_ratio=drift_ratio,
            memory_stability=memory_stability,
            persistence_center=(float(persistence_center[0]), float(persistence_center[1])),
            persistent_reentry_used=persistent_reentry_used,
            stability_relaxed=stability_relaxed,
        )
