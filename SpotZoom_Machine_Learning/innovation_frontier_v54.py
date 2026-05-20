"""
SpotZoom innovation frontier v54.

Reliability-aware memory-tree association gate inspired by:
- CoTracker3: long-stream online consistency for sparse tracks.
- SAM2Long: fixed-budget multi-path memory behavior under occlusion.
- What You Have is What You Track (ICCV 2025): adaptive robustness to
  incomplete observations.
- Norfair/slmsuite/openwfs: lightweight distance-first real-time control loops.

This module intentionally keeps a numpy-only runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Deque, Optional, Tuple
from collections import deque

import numpy as np


def _load_v53_primitives():
    try:
        from SpotZoom_Machine_Learning.innovation_frontier_v53 import (  # type: ignore
            OcclusionTreeAssociationConfig as _Cfg,
            OcclusionTreeAssociationGate as _Gate,
        )
        return _Cfg, _Gate
    except Exception:
        pass

    module_path = Path(__file__).resolve().with_name("innovation_frontier_v53.py")
    spec = importlib.util.spec_from_file_location("_spotzoom_ml_v53_fallback", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load innovation_frontier_v53 primitives")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg_cls = getattr(module, "OcclusionTreeAssociationConfig", None)
    gate_cls = getattr(module, "OcclusionTreeAssociationGate", None)
    if cfg_cls is None or gate_cls is None:
        raise ImportError("innovation_frontier_v53 missing expected classes")
    return cfg_cls, gate_cls


_V53Config, _V53Gate = _load_v53_primitives()


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


class OcclusionTreeAssociationGate:
    """v54 wrapper on top of v53 with reliability self-calibration + rescue path."""

    def __init__(self, config: Optional[OcclusionTreeAssociationConfig] = None):
        self.config = config or OcclusionTreeAssociationConfig()
        v53_cfg = _V53Config(
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
        )
        self._base_gate = _V53Gate(config=v53_cfg)
        self._reliability_ema: float = 0.72
        self._residual_history: Deque[float] = deque(maxlen=max(3, int(self.config.jitter_window)))

    def reset(self) -> None:
        self._base_gate.reset()
        self._reliability_ema = 0.72
        self._residual_history = deque(maxlen=max(3, int(self.config.jitter_window)))

    def _calc_jitter(self, residual_norm: float) -> float:
        self._residual_history.append(float(max(0.0, residual_norm)))
        if len(self._residual_history) < 3:
            return 0.0
        arr = np.asarray(tuple(self._residual_history), dtype=np.float64)
        mean = float(np.mean(arr))
        if not np.isfinite(mean) or mean <= 1e-6:
            return 0.0
        score = float(np.std(arr) / mean)
        if not np.isfinite(score):
            return 0.0
        return float(np.clip(score / 1.6, 0.0, 1.5))

    def _update_reliability(self, accepted: bool, rescued: bool, fusion: float, assoc_dist: float, gate: float) -> Tuple[float, bool]:
        decay = float(np.clip(self.config.reliability_ema_decay, 0.0, 1.0))
        prev = float(np.clip(self._reliability_ema, 0.0, 1.0))
        if accepted:
            reward = float(np.clip(self.config.reliability_reward, 0.0, 1.0))
            quality = float(np.clip(fusion, 0.0, 1.0))
            if gate > 1e-6:
                quality = float(np.clip(0.55 * quality + 0.45 * np.clip(1.0 - assoc_dist / gate, 0.0, 1.0), 0.0, 1.0))
            if rescued:
                quality = float(np.clip(0.7 * quality, 0.0, 1.0))
            target = quality
            blend = (1.0 - decay) * reward
        else:
            penalty = float(np.clip(self.config.reliability_penalty, 0.0, 1.0))
            target = float(np.clip(1.0 - penalty, 0.0, 1.0))
            blend = (1.0 - decay)

        cur = (1.0 - blend) * prev + blend * target
        cur = float(np.clip(cur, 0.05, 0.99))
        self._reliability_ema = cur
        recovered = bool(accepted and cur > prev + 0.015)
        return cur, recovered

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

        assoc_dist = float(getattr(base, "association_distance_px", float(np.linalg.norm(obs - pred))) or 0.0)
        base_gate = float(getattr(base, "adaptive_gate_px", float(self.config.base_gate_px)) or self.config.base_gate_px)
        jitter_score = self._calc_jitter(assoc_dist)

        reliability_gap = float(np.clip(1.0 - self._reliability_ema, 0.0, 1.0))
        adaptive_gate = float(
            base_gate
            * (
                1.0
                + float(self.config.reliability_gate_scale) * reliability_gap
                + float(self.config.jitter_gate_scale) * float(np.clip(jitter_score, 0.0, 1.0))
            )
        )

        fusion_score = float(np.clip(_safe_float(getattr(base, "fusion_score", 0.0), 0.0), 0.0, 1.0))
        memory_similarity = float(np.clip(_safe_float(getattr(base, "memory_similarity", 0.0), 0.0), 0.0, 1.0))
        occlusion_score = float(np.clip(_safe_float(getattr(base, "occlusion_score", 0.0), 0.0), 0.0, 1.0))

        rejected = bool(getattr(base, "rejected", False)) or assoc_dist > adaptive_gate
        accepted = bool(not rejected)
        reason = str(getattr(base, "reason", "outside_adaptive_gate" if rejected else "")) if rejected else None
        out_center = getattr(base, "center", center)
        out_conf = float(np.clip(_safe_float(getattr(base, "confidence", confidence), confidence), 0.0, 1.0))
        used_rescue_path = False

        if (
            rejected
            and out_conf <= float(self.config.low_confidence_threshold)
            and memory_similarity >= float(self.config.rescue_memory_similarity)
            and fusion_score >= float(self.config.rescue_fusion_threshold)
            and (
                occlusion_score >= float(self.config.rescue_occlusion_threshold)
                or jitter_score >= float(self.config.jitter_reject_threshold)
            )
        ):
            blend = float(np.clip(float(self.config.rescue_blend) + 0.18 * memory_similarity, 0.35, 0.92))
            rescue = blend * pred + (1.0 - blend) * obs
            rescue_dist = float(np.linalg.norm(rescue - pred))
            max_assoc = max(1e-6, float(self.config.max_association_distance_px))
            if rescue_dist <= max_assoc and rescue_dist <= adaptive_gate * 1.10:
                out_center = (float(rescue[0]), float(rescue[1]))
                assoc_dist = rescue_dist
                fusion_score = float(np.clip(fusion_score + 0.07 * memory_similarity, 0.0, 1.0))
                rejected = False
                accepted = True
                reason = None
                used_rescue_path = True

        reliability_score, reliability_recovered = self._update_reliability(
            accepted=accepted,
            rescued=used_rescue_path,
            fusion=fusion_score,
            assoc_dist=assoc_dist,
            gate=max(1e-6, adaptive_gate),
        )

        if accepted:
            out_conf = float(
                np.clip(
                    out_conf + float(self.config.reliability_confidence_blend) * (reliability_score - 0.5),
                    0.0,
                    1.0,
                )
            )
        else:
            out_conf = float(np.clip(out_conf * 0.9, 0.0, 1.0))

        innovation = float(np.clip(0.6 * fusion_score + 0.4 * (1.0 - np.clip(assoc_dist / max(1e-6, adaptive_gate), 0.0, 1.0)), 0.0, 1.0))
        return OcclusionTreeAssociationResult(
            center=(float(out_center[0]), float(out_center[1])) if isinstance(out_center, (tuple, list)) else (float(obs[0]), float(obs[1])),
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
            occlusion_score=occlusion_score,
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
            reliability_score=float(reliability_score),
            jitter_score=float(np.clip(jitter_score, 0.0, 1.5)),
            used_rescue_path=used_rescue_path,
            reliability_recovered=reliability_recovered,
        )
