"""
SpotZoom innovation frontier v56.

Teacher-consensus verifier association gate inspired by:
- Track-On / Track-On2 / Track-On-R: compact memory + verifier-guided reliability.
- CoTracker3: long-horizon online trajectory consistency.
- SAM2: streaming memory state for video-time decisions.
- Norfair: lightweight distance-function-first modular association.

This module intentionally keeps a numpy-only runtime dependency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np


def _load_v55_primitives():
    try:
        from SpotZoom_Machine_Learning.innovation_frontier_v55 import (  # type: ignore
            OcclusionTreeAssociationConfig as _Cfg,
            OcclusionTreeAssociationGate as _Gate,
        )
        return _Cfg, _Gate
    except Exception:
        pass

    module_path = Path(__file__).resolve().with_name("innovation_frontier_v55.py")
    spec = importlib.util.spec_from_file_location("_spotzoom_ml_v55_fallback", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load innovation_frontier_v55 primitives")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg_cls = getattr(module, "OcclusionTreeAssociationConfig", None)
    gate_cls = getattr(module, "OcclusionTreeAssociationGate", None)
    if cfg_cls is None or gate_cls is None:
        raise ImportError("innovation_frontier_v55 missing expected classes")
    return cfg_cls, gate_cls


_V55Config, _V55Gate = _load_v55_primitives()


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
    teacher_temperature: float = 0.33
    teacher_entropy_reject_threshold: float = 0.88
    teacher_agreement_min: float = 0.28
    teacher_consensus_blend: float = 0.44
    teacher_rescue_gain: float = 0.08
    teacher_history: int = 12


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


class OcclusionTreeAssociationGate:
    """v56 wrapper on top of v55 with teacher-consensus uncertainty gating."""

    def __init__(self, config: Optional[OcclusionTreeAssociationConfig] = None):
        self.config = config or OcclusionTreeAssociationConfig()
        v55_cfg = _V55Config(
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
        )
        self._base_gate = _V55Gate(config=v55_cfg)
        self._teacher_history: Deque[np.ndarray] = deque(maxlen=max(4, int(self.config.teacher_history)))
        self._teacher_entropy_ema: float = 0.45

    def reset(self) -> None:
        self._base_gate.reset()
        self._teacher_history = deque(maxlen=max(4, int(self.config.teacher_history)))
        self._teacher_entropy_ema = 0.45

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        if logits.size == 0:
            return logits
        shifted = logits - np.max(logits)
        exps = np.exp(np.clip(shifted, -60.0, 60.0))
        denom = float(np.sum(exps))
        if not np.isfinite(denom) or denom <= 1e-12:
            return np.full_like(logits, 1.0 / float(logits.size))
        return exps / denom

    def _history_center(self) -> Optional[np.ndarray]:
        if not self._teacher_history:
            return None
        arr = np.asarray(tuple(self._teacher_history), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        out = np.mean(arr, axis=0)
        if not np.all(np.isfinite(out)):
            return None
        return out

    def _teacher_consensus(
        self,
        obs: np.ndarray,
        pred: np.ndarray,
        refined: np.ndarray,
        history: Optional[np.ndarray],
        gate_px: float,
    ) -> Tuple[np.ndarray, float, float]:
        candidates = [obs, pred, refined]
        if history is not None:
            candidates.append(history)
        cand = np.asarray(candidates, dtype=np.float64)
        if cand.ndim != 2 or cand.shape[1] != 2:
            return refined.copy(), 1.0, 0.0

        gate = max(1e-6, float(gate_px))
        dists = np.linalg.norm(cand - pred, axis=1)
        temp = max(1e-3, float(self.config.teacher_temperature))
        logits = -(dists / gate) / temp
        weights = self._softmax(logits)
        consensus = np.sum(cand * weights[:, None], axis=0)
        if not np.all(np.isfinite(consensus)):
            consensus = refined.copy()

        entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0))))
        norm = float(np.log(float(weights.size))) if weights.size > 1 else 1.0
        entropy = float(np.clip(entropy / max(1e-9, norm), 0.0, 1.0))

        spread = float(np.mean(np.linalg.norm(cand - consensus, axis=1)))
        agreement = float(np.clip(1.0 - spread / max(1e-6, gate), 0.0, 1.0))
        return consensus, entropy, agreement

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

        gate = float(getattr(base, "adaptive_gate_px", float(self.config.base_gate_px)) or self.config.base_gate_px)
        assoc_dist = float(getattr(base, "association_distance_px", float(np.linalg.norm(obs - pred))) or 0.0)
        history_center = self._history_center()

        teacher_center, teacher_entropy, teacher_agreement = self._teacher_consensus(
            obs=obs,
            pred=pred,
            refined=base_center,
            history=history_center,
            gate_px=max(1.0, gate),
        )

        decay = float(np.clip(self.config.verifier_ema_decay, 0.0, 1.0))
        self._teacher_entropy_ema = float(
            np.clip(
                decay * self._teacher_entropy_ema + (1.0 - decay) * teacher_entropy,
                0.0,
                1.0,
            )
        )
        entropy_gate = 0.60 * teacher_entropy + 0.40 * self._teacher_entropy_ema

        verifier_score = float(np.clip(_safe_float(getattr(base, "verifier_score", 0.0), 0.0), 0.0, 1.0))
        disagreement_score = float(np.clip(_safe_float(getattr(base, "disagreement_score", 0.0), 0.0), 0.0, 2.5))
        fusion_score = float(np.clip(_safe_float(getattr(base, "fusion_score", 0.0), 0.0), 0.0, 1.0))

        accepted = bool(not getattr(base, "rejected", False))
        rejected = bool(not accepted)
        reason = str(getattr(base, "reason", "rejected")) if rejected else None
        out_center = base_center.copy()
        out_conf = float(np.clip(_safe_float(getattr(base, "confidence", confidence), confidence), 0.0, 1.0))
        teacher_rescue_used = False
        teacher_rejected = False

        entropy_reject = entropy_gate > float(self.config.teacher_entropy_reject_threshold)
        weak_agreement = teacher_agreement < float(self.config.teacher_agreement_min)
        low_verifier = verifier_score < float(self.config.verifier_min_score) + float(self.config.verifier_margin)

        if accepted and (entropy_reject or weak_agreement) and low_verifier:
            accepted = False
            rejected = True
            teacher_rejected = True
            if entropy_reject and weak_agreement:
                reason = "teacher_entropy_and_agreement_rejected"
            elif entropy_reject:
                reason = "teacher_entropy_rejected"
            else:
                reason = "teacher_agreement_rejected"

        if rejected and not low_verifier and teacher_agreement >= float(self.config.teacher_agreement_min) * 0.9:
            rescue_blend = float(np.clip(self.config.teacher_consensus_blend + self.config.teacher_rescue_gain, 0.0, 1.0))
            rescue = rescue_blend * teacher_center + (1.0 - rescue_blend) * pred
            rescue_dist = float(np.linalg.norm(rescue - pred))
            if rescue_dist <= max(1e-6, float(self.config.max_association_distance_px)) and rescue_dist <= max(1.0, gate) * 1.2:
                out_center = rescue
                assoc_dist = rescue_dist
                fusion_score = float(np.clip(fusion_score + float(self.config.teacher_rescue_gain), 0.0, 1.0))
                accepted = True
                rejected = False
                reason = None
                teacher_rescue_used = True

        if accepted:
            blend = float(
                np.clip(
                    self.config.teacher_consensus_blend
                    + 0.18 * (verifier_score - 0.5)
                    - 0.16 * entropy_gate
                    + 0.10 * teacher_agreement,
                    0.18,
                    0.86,
                )
            )
            out_center = blend * teacher_center + (1.0 - blend) * base_center
            if not np.all(np.isfinite(out_center)):
                out_center = base_center.copy()
            out_conf = float(
                np.clip(
                    out_conf
                    + float(self.config.confidence_temperature) * (verifier_score - 0.5)
                    + 0.08 * (teacher_agreement - entropy_gate),
                    0.0,
                    1.0,
                )
            )
            self._teacher_history.append(np.asarray([float(out_center[0]), float(out_center[1])], dtype=np.float64))
        else:
            out_conf = float(np.clip(out_conf * (0.84 + 0.12 * max(0.0, 1.0 - entropy_gate)), 0.0, 1.0))

        innovation = float(
            np.clip(
                0.34 * fusion_score
                + 0.26 * verifier_score
                + 0.20 * float(np.clip(1.0 - disagreement_score, 0.0, 1.0))
                + 0.20 * float(np.clip(teacher_agreement - 0.6 * entropy_gate, 0.0, 1.0)),
                0.0,
                1.0,
            )
        )

        consensus_raw = getattr(base, "consensus_center", (out_center[0], out_center[1]))
        if not isinstance(consensus_raw, (tuple, list)) or len(consensus_raw) < 2:
            consensus_raw = (out_center[0], out_center[1])
        consensus_x = float(_safe_float(consensus_raw[0], out_center[0]))
        consensus_y = float(_safe_float(consensus_raw[1], out_center[1]))

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
            fusion_score=float(fusion_score),
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
            innovation_score=innovation,
            reliability_score=float(np.clip(_safe_float(getattr(base, "reliability_score", 0.0), 0.0), 0.0, 1.0)),
            jitter_score=float(np.clip(_safe_float(getattr(base, "jitter_score", 0.0), 0.0), 0.0, 1.5)),
            used_rescue_path=bool(getattr(base, "used_rescue_path", False)),
            reliability_recovered=bool(getattr(base, "reliability_recovered", False)),
            verifier_score=verifier_score,
            disagreement_score=disagreement_score,
            consensus_center=(consensus_x, consensus_y),
            verifier_rescue_used=bool(getattr(base, "verifier_rescue_used", False)),
            disagreement_rejected=bool(getattr(base, "disagreement_rejected", False)),
            teacher_entropy=float(entropy_gate),
            teacher_agreement=float(teacher_agreement),
            teacher_consensus_center=(float(teacher_center[0]), float(teacher_center[1])),
            teacher_rescue_used=teacher_rescue_used,
            teacher_rejected=teacher_rejected,
        )
