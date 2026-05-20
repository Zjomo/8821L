"""
SpotZoom innovation frontier v55.

Verifier-consensus memory-tree association gate inspired by:
- CoTracker3: online long-horizon trajectory consistency.
- Track-On / Track-On-R: verifier-guided reliability for point tracking.
- SAM2Long: compact long-memory behavior under occlusion.
- Norfair: distance-first modular tracking decisions.

This module intentionally keeps a numpy-only runtime dependency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np


def _load_v54_primitives():
    try:
        from SpotZoom_Machine_Learning.innovation_frontier_v54 import (  # type: ignore
            OcclusionTreeAssociationConfig as _Cfg,
            OcclusionTreeAssociationGate as _Gate,
        )
        return _Cfg, _Gate
    except Exception:
        pass

    module_path = Path(__file__).resolve().with_name("innovation_frontier_v54.py")
    spec = importlib.util.spec_from_file_location("_spotzoom_ml_v54_fallback", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load innovation_frontier_v54 primitives")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg_cls = getattr(module, "OcclusionTreeAssociationConfig", None)
    gate_cls = getattr(module, "OcclusionTreeAssociationGate", None)
    if cfg_cls is None or gate_cls is None:
        raise ImportError("innovation_frontier_v54 missing expected classes")
    return cfg_cls, gate_cls


_V54Config, _V54Gate = _load_v54_primitives()


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


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


class OcclusionTreeAssociationGate:
    """v55 wrapper on top of v54 with verifier-consensus correction."""

    def __init__(self, config: Optional[OcclusionTreeAssociationConfig] = None):
        self.config = config or OcclusionTreeAssociationConfig()
        v54_cfg = _V54Config(
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
        )
        self._base_gate = _V54Gate(config=v54_cfg)
        self._accepted_history: Deque[np.ndarray] = deque(maxlen=max(3, int(self.config.verifier_history)))
        self._verifier_ema: float = 0.62

    def reset(self) -> None:
        self._base_gate.reset()
        self._accepted_history = deque(maxlen=max(3, int(self.config.verifier_history)))
        self._verifier_ema = 0.62

    def _history_center(self) -> Optional[np.ndarray]:
        if not self._accepted_history:
            return None
        arr = np.asarray(tuple(self._accepted_history), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        out = np.mean(arr, axis=0)
        if not np.all(np.isfinite(out)):
            return None
        return out

    def _calc_disagreement(
        self,
        obs: np.ndarray,
        pred: np.ndarray,
        refined: np.ndarray,
        history_center: Optional[np.ndarray],
        gate_px: float,
    ) -> float:
        points = [obs, pred, refined]
        if history_center is not None:
            points.append(history_center)
        arr = np.asarray(points, dtype=np.float64)
        centroid = np.mean(arr, axis=0)
        spread = float(np.mean(np.linalg.norm(arr - centroid, axis=1)))
        score = spread / max(1e-6, float(gate_px))
        if not np.isfinite(score):
            return 0.0
        return float(np.clip(score, 0.0, 2.5))

    def _calc_verifier(
        self,
        *,
        fusion_score: float,
        memory_similarity: float,
        reliability_score: float,
        jitter_score: float,
        disagreement_score: float,
    ) -> float:
        disagreement_term = float(np.clip(1.0 - 0.8 * float(disagreement_score), 0.0, 1.0))
        jitter_term = float(np.clip(1.0 - 0.7 * float(jitter_score), 0.0, 1.0))
        raw = (
            0.32 * float(np.clip(fusion_score, 0.0, 1.0))
            + 0.24 * float(np.clip(memory_similarity, 0.0, 1.0))
            + 0.24 * float(np.clip(reliability_score, 0.0, 1.0))
            + 0.12 * jitter_term
            + 0.08 * disagreement_term
        )
        raw = float(np.clip(raw, 0.0, 1.0))
        decay = float(np.clip(self.config.verifier_ema_decay, 0.0, 1.0))
        self._verifier_ema = float(np.clip(decay * self._verifier_ema + (1.0 - decay) * raw, 0.0, 1.0))
        return float(np.clip(0.5 * raw + 0.5 * self._verifier_ema, 0.0, 1.0))

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
        pred_raw = getattr(base, "predicted_center", center)
        if not isinstance(pred_raw, (tuple, list)) or len(pred_raw) < 2:
            pred_raw = center
        pred = np.asarray([_safe_float(pred_raw[0], obs[0]), _safe_float(pred_raw[1], obs[1])], dtype=np.float64)

        base_center_raw = getattr(base, "center", center)
        if not isinstance(base_center_raw, (tuple, list)) or len(base_center_raw) < 2:
            base_center_raw = center
        base_center = np.asarray(
            [_safe_float(base_center_raw[0], obs[0]), _safe_float(base_center_raw[1], obs[1])],
            dtype=np.float64,
        )

        assoc_dist = float(getattr(base, "association_distance_px", float(np.linalg.norm(obs - pred))) or 0.0)
        base_gate = float(getattr(base, "adaptive_gate_px", float(self.config.base_gate_px)) or self.config.base_gate_px)
        history_center = self._history_center()
        disagreement_score = self._calc_disagreement(
            obs=obs,
            pred=pred,
            refined=base_center,
            history_center=history_center,
            gate_px=max(1.0, base_gate),
        )
        gate_scale = 1.0 + float(self.config.disagreement_gate_scale) * max(0.0, 0.45 - disagreement_score)
        adaptive_gate = float(base_gate * max(0.55, gate_scale))

        fusion_score = float(np.clip(_safe_float(getattr(base, "fusion_score", 0.0), 0.0), 0.0, 1.0))
        memory_similarity = float(np.clip(_safe_float(getattr(base, "memory_similarity", 0.0), 0.0), 0.0, 1.0))
        reliability_score = float(np.clip(_safe_float(getattr(base, "reliability_score", 0.5), 0.5), 0.0, 1.0))
        jitter_score = float(np.clip(_safe_float(getattr(base, "jitter_score", 0.0), 0.0), 0.0, 1.5))

        verifier_score = self._calc_verifier(
            fusion_score=fusion_score,
            memory_similarity=memory_similarity,
            reliability_score=reliability_score,
            jitter_score=jitter_score,
            disagreement_score=disagreement_score,
        )

        consensus = base_center.copy()
        if history_center is not None:
            blend = float(np.clip(self.config.consensus_blend, 0.0, 1.0))
            consensus = blend * base_center + (1.0 - blend) * history_center
            consensus = 0.78 * consensus + 0.22 * pred

        accepted = bool(not getattr(base, "rejected", False) and assoc_dist <= adaptive_gate)
        rejected = bool(not accepted)
        reason = str(getattr(base, "reason", "rejected")) if rejected else None
        out_center = base_center.copy()
        out_conf = float(np.clip(_safe_float(getattr(base, "confidence", confidence), confidence), 0.0, 1.0))
        used_rescue_path = bool(getattr(base, "used_rescue_path", False))
        verifier_rescue_used = False
        disagreement_rejected = False

        if (
            rejected
            and verifier_score >= float(self.config.verifier_min_score)
            and memory_similarity >= float(self.config.rescue_memory_similarity)
            and disagreement_score <= float(self.config.disagreement_reject_threshold) * 0.75
        ):
            rescue = 0.70 * consensus + 0.30 * pred
            rescue_dist = float(np.linalg.norm(rescue - pred))
            max_assoc = max(1e-6, float(self.config.max_association_distance_px))
            if rescue_dist <= max_assoc and rescue_dist <= adaptive_gate * 1.15:
                out_center = rescue
                assoc_dist = rescue_dist
                fusion_score = float(np.clip(fusion_score + float(self.config.rescue_verifier_bonus), 0.0, 1.0))
                accepted = True
                rejected = False
                reason = None
                used_rescue_path = True
                verifier_rescue_used = True

        if accepted and disagreement_score > float(self.config.disagreement_reject_threshold):
            margin = float(self.config.verifier_margin)
            if verifier_score < float(self.config.verifier_min_score) + margin:
                accepted = False
                rejected = True
                reason = "disagreement_rejected"
                disagreement_rejected = True

        if accepted:
            out_conf = float(
                np.clip(
                    out_conf + float(self.config.confidence_temperature) * (verifier_score - 0.5),
                    0.0,
                    1.0,
                )
            )
            self._accepted_history.append(np.asarray([float(out_center[0]), float(out_center[1])], dtype=np.float64))
        else:
            out_conf = float(np.clip(out_conf * 0.88, 0.0, 1.0))

        innovation = float(
            np.clip(
                0.45 * fusion_score
                + 0.35 * verifier_score
                + 0.20 * float(np.clip(1.0 - disagreement_score, 0.0, 1.0)),
                0.0,
                1.0,
            )
        )

        return OcclusionTreeAssociationResult(
            center=(float(out_center[0]), float(out_center[1])),
            confidence=out_conf,
            accepted=accepted,
            refined=bool(getattr(base, "refined", accepted)),
            rejected=not accepted,
            reason=reason,
            predicted_center=(float(pred[0]), float(pred[1])),
            association_distance_px=float(assoc_dist),
            adaptive_gate_px=float(adaptive_gate),
            fusion_score=float(fusion_score),
            memory_similarity=memory_similarity,
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
            innovation_score=innovation,
            reliability_score=reliability_score,
            jitter_score=jitter_score,
            used_rescue_path=used_rescue_path,
            reliability_recovered=bool(getattr(base, "reliability_recovered", False)),
            verifier_score=float(verifier_score),
            disagreement_score=float(disagreement_score),
            consensus_center=(float(consensus[0]), float(consensus[1])),
            verifier_rescue_used=verifier_rescue_used,
            disagreement_rejected=disagreement_rejected,
        )
