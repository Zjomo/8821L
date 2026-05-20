"""
SpotZoom innovation frontier v28.

This module adds a lightweight temporal consensus refiner inspired by:
- CoTracker (robust temporal point tracking)
- Norfair (tracking-by-detection stability)
- TrackMate (trajectory consistency constraints)
- DeepTrack2 (scientific imaging temporal robustness)

The implementation intentionally depends only on numpy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class TemporalConsensusConfig:
    """Configuration for temporal consensus refinement."""

    history_size: int = 8
    huber_delta_px: float = 4.0
    outlier_zscore: float = 2.8
    max_refine_shift_px: float = 14.0
    min_confidence: float = 0.08
    min_refine_shift_px: float = 0.35


@dataclass
class TemporalConsensusResult:
    """Result of one consensus refinement call."""

    center: Tuple[float, float]
    confidence: float
    uncertainty_px: float
    refined: bool
    rejected: bool
    reason: Optional[str]
    history_count: int


class TemporalConsensusRefiner:
    """Refine one detection center using robust temporal consensus."""

    def __init__(self, config: Optional[TemporalConsensusConfig] = None):
        self.config = config or TemporalConsensusConfig()
        size = max(3, int(self.config.history_size))
        self._history: Deque[Tuple[float, float, float]] = deque(maxlen=size)

    def reset(self) -> None:
        self._history.clear()

    @staticmethod
    def _safe_float(x: float, default: float = 0.0) -> float:
        try:
            xf = float(x)
        except Exception:
            return default
        if not np.isfinite(xf):
            return default
        return xf

    def _robust_center(self, pts: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, float]:
        eps = 1e-8
        w = np.clip(conf, 0.05, 1.0)
        c = (pts * w[:, None]).sum(axis=0) / (w.sum() + eps)
        delta = max(1e-3, float(self.config.huber_delta_px))
        for _ in range(3):
            residual = np.linalg.norm(pts - c[None, :], axis=1)
            huber = np.where(residual <= delta, 1.0, delta / np.maximum(residual, eps))
            wr = w * huber
            c = (pts * wr[:, None]).sum(axis=0) / (wr.sum() + eps)
        residual = np.linalg.norm(pts - c[None, :], axis=1)
        uncertainty = float(np.sqrt(np.sum((residual**2) * w) / (w.sum() + eps)))
        return c, uncertainty

    def update(
        self,
        center: Tuple[float, float],
        confidence: float,
    ) -> TemporalConsensusResult:
        cx = self._safe_float(center[0], default=np.nan)
        cy = self._safe_float(center[1], default=np.nan)
        conf = float(np.clip(self._safe_float(confidence, default=0.0), 0.0, 1.0))

        if not np.isfinite(cx) or not np.isfinite(cy):
            return TemporalConsensusResult(
                center=(0.0, 0.0),
                confidence=conf,
                uncertainty_px=float("inf"),
                refined=False,
                rejected=True,
                reason="invalid_center",
                history_count=len(self._history),
            )

        if conf < float(self.config.min_confidence):
            self._history.append((cx, cy, max(0.05, conf)))
            return TemporalConsensusResult(
                center=(cx, cy),
                confidence=conf,
                uncertainty_px=float("inf"),
                refined=False,
                rejected=True,
                reason="low_confidence",
                history_count=len(self._history),
            )

        # If too little history, bootstrap without refinement.
        if len(self._history) < 2:
            self._history.append((cx, cy, conf))
            return TemporalConsensusResult(
                center=(cx, cy),
                confidence=conf,
                uncertainty_px=0.0,
                refined=False,
                rejected=False,
                reason=None,
                history_count=len(self._history),
            )

        hist = np.asarray(self._history, dtype=np.float64)
        hist_pts = hist[:, :2]
        hist_conf = hist[:, 2]
        curr_pt = np.asarray([[cx, cy]], dtype=np.float64)
        curr_conf = np.asarray([max(0.05, conf)], dtype=np.float64)

        pts = np.concatenate([hist_pts, curr_pt], axis=0)
        confs = np.concatenate([hist_conf, curr_conf], axis=0)
        consensus, uncertainty = self._robust_center(pts, confs)

        shift = float(np.linalg.norm(curr_pt[0] - consensus))
        max_shift = max(0.0, float(self.config.max_refine_shift_px))
        if shift > max_shift:
            self._history.append((cx, cy, conf * 0.5 + 0.05))
            return TemporalConsensusResult(
                center=(cx, cy),
                confidence=conf,
                uncertainty_px=uncertainty,
                refined=False,
                rejected=True,
                reason="shift_too_large",
                history_count=len(self._history),
            )

        # Outlier gate against history centroid.
        hist_center, hist_uncertainty = self._robust_center(hist_pts, hist_conf)
        hist_shift = float(np.linalg.norm(curr_pt[0] - hist_center))
        sigma = max(1e-6, hist_uncertainty)
        z = hist_shift / sigma
        if z > float(self.config.outlier_zscore):
            self._history.append((cx, cy, conf * 0.5 + 0.05))
            return TemporalConsensusResult(
                center=(cx, cy),
                confidence=conf,
                uncertainty_px=uncertainty,
                refined=False,
                rejected=True,
                reason="outlier_rejected",
                history_count=len(self._history),
            )

        min_refine = max(0.0, float(self.config.min_refine_shift_px))
        if shift <= min_refine:
            self._history.append((cx, cy, conf))
            return TemporalConsensusResult(
                center=(cx, cy),
                confidence=conf,
                uncertainty_px=uncertainty,
                refined=False,
                rejected=False,
                reason=None,
                history_count=len(self._history),
            )

        # Adaptive blending: low confidence -> stronger pull to consensus.
        blend = float(np.clip(0.2 + 0.5 * (1.0 - conf), 0.2, 0.7))
        refined = (1.0 - blend) * curr_pt[0] + blend * consensus
        rx, ry = float(refined[0]), float(refined[1])
        self._history.append((rx, ry, max(conf, 0.15)))
        return TemporalConsensusResult(
            center=(rx, ry),
            confidence=conf,
            uncertainty_px=uncertainty,
            refined=True,
            rejected=False,
            reason=None,
            history_count=len(self._history),
        )

