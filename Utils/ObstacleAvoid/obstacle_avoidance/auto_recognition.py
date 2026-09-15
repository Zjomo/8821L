"""Offline automatic workspace recognition for Alg2 microscope videos.

This module deliberately does not issue stage commands.  It turns sampled
video frames into ``WorkspaceSnapshot`` objects that can be reviewed before
the same perception output is connected to Alg2's closed loop controller.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .models import (CoordinateTransform, Obstacle, Particle, Point,
                     SubstrateRegion, WorkspaceSnapshot)
from .vision import ParticleTracker, VisionResult, YoloDetector


@dataclass
class AutoRecognitionConfig:
    """Tunable parameters for the v1 microscope-video baseline."""

    yolo_imgsz: int = 640
    yolo_confidence: float = 0.45
    substrate_clusters: int = 8
    substrate_analysis_width: int = 420
    substrate_min_area_ratio: float = 0.08
    substrate_safety_margin_px: float = 5.0
    registration_width: int = 480
    registration_min_confidence: float = 0.05
    substrate_refresh_interval: int = 12
    substrate_alignment_max_error_px: float = 18.0
    tracker_max_jump_px: float = 90.0
    tracker_max_lost_frames: int = 3
    min_candidate_area_px2: float = 24.0
    max_candidate_area_ratio: float = 0.08
    max_candidates: int = 24
    ball_exclusion_scale: float = 1.35
    px_per_mm: float = 100.0
    minimum_overall_confidence: float = 0.35
    # Kept enabled for backwards-compatible offline analysis; Alg2 UI passes
    # ``auto_substrate=False`` and supplies the manual ROI instead.
    auto_substrate: bool = True


@dataclass
class RegionCandidate:
    """A non-semantic forbidden-region candidate from traditional vision."""

    kind: str
    polygon: List[Point]
    confidence: float
    area_px2: float

    def to_obstacle(self, index: int) -> Obstacle:
        return Obstacle(kind="polygon", polygon=self.polygon,
                        obstacle_id=f"{self.kind}-{index:03d}")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "polygon": [[float(x), float(y)] for x, y in self.polygon],
            "confidence": float(self.confidence),
            "area_px2": float(self.area_px2),
        }


@dataclass
class AutoRecognitionResult:
    frame_id: int
    timestamp_s: float
    substrate_polygon: List[Point]
    substrate_mask: np.ndarray = field(repr=False)
    substrate_confidence: float = 0.0
    registration_shift_px: Point = (0.0, 0.0)
    registration_confidence: float = 0.0
    particles: List[Particle] = field(default_factory=list)
    particles_inside_substrate: List[bool] = field(default_factory=list)
    beam_spot_px: Optional[Point] = None
    beam_confidence: float = 0.0
    candidates: List[RegionCandidate] = field(default_factory=list)
    overall_confidence: float = 0.0
    uncertain: bool = False
    uncertain_reason: str = ""
    instant_shift_px: Point = (0.0, 0.0)
    motion_method: str = "phase"
    track_events: List[str] = field(default_factory=list)
    overlay_mask_ratio: float = 0.0

    def to_snapshot(self, px_per_mm: float = 100.0,
                    safety_margin_px: float = 5.0) -> WorkspaceSnapshot:
        obstacles = [c.to_obstacle(i + 1)
                     for i, c in enumerate(self.candidates)]
        substrate = SubstrateRegion(self.substrate_polygon, safety_margin_px)
        h, w = self.substrate_mask.shape[:2]
        return WorkspaceSnapshot(
            frame_id=self.frame_id,
            timestamp=self.timestamp_s,
            substrate=substrate,
            obstacles=obstacles,
            particles=self.particles,
            transform=CoordinateTransform(px_per_mm=px_per_mm),
            frame_size=(w, h),
            overall_confidence=self.overall_confidence,
            uncertain=self.uncertain,
            uncertain_reason=self.uncertain_reason,
        )

    def to_dict(self) -> dict:
        particles = []
        for particle, inside in zip(self.particles,
                                    self.particles_inside_substrate):
            item = particle.to_dict()
            item["inside_substrate"] = bool(inside)
            particles.append(item)
        return {
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "substrate": {
                "polygon": [[float(x), float(y)]
                            for x, y in self.substrate_polygon],
                "confidence": self.substrate_confidence,
                "area_px2": int(np.count_nonzero(self.substrate_mask)),
            },
            "registration": {
                "shift_px": list(self.registration_shift_px),
                "confidence": self.registration_confidence,
                "instant_shift_px": list(self.instant_shift_px),
                "method": self.motion_method,
            },
            "particles": particles,
            "beam_spot": {
                "center_px": (list(self.beam_spot_px)
                              if self.beam_spot_px is not None else None),
                "confidence": self.beam_confidence,
            },
            "forbidden_region_candidates": [c.to_dict()
                                             for c in self.candidates],
            "overall_confidence": self.overall_confidence,
            "uncertain": self.uncertain,
            "uncertain_reason": self.uncertain_reason,
            "track_events": list(self.track_events),
            "overlay_mask_ratio": self.overlay_mask_ratio,
        }


def suppress_ui_overlays(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mask saturated recording/UI annotation colors before inference."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    saturated = (s > 115) & (v > 90)
    # Recorded blue/cyan/red/yellow drawing layers; preserve green laser.
    ui_hues = ((h <= 18) | (h >= 170) | ((h >= 90) & (h <= 135)) |
               ((h >= 20) & (h <= 34)))
    mask = (saturated & ui_hues).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    if not np.any(mask):
        return frame, mask
    return cv2.inpaint(frame, mask, 3.0, cv2.INPAINT_TELEA), mask


class GlobalMotionEstimator:
    """Estimate cumulative camera/sample translation with phase correlation."""

    def __init__(self, analysis_width: int = 480) -> None:
        self.analysis_width = analysis_width
        self._previous: Optional[np.ndarray] = None
        self._cumulative = np.zeros(2, dtype=np.float64)
        self.last_instant_shift: Point = (0.0, 0.0)
        self.last_method = "phase"

    def reset(self) -> None:
        self._previous = None
        self._cumulative[:] = 0.0
        self.last_instant_shift = (0.0, 0.0)
        self.last_method = "phase"

    def update(self, frame: np.ndarray) -> Tuple[Point, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.analysis_width / max(1, gray.shape[1]))
        small = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        small = cv2.GaussianBlur(small, (0, 0), 1.2)
        if self._previous is None:
            self._previous = small
            self.last_instant_shift = (0.0, 0.0)
            return (0.0, 0.0), 1.0
        if small.shape != self._previous.shape:
            self._previous = small
            self._cumulative[:] = 0.0
            return (0.0, 0.0), 0.0
        window = cv2.createHanningWindow(
            (small.shape[1], small.shape[0]), cv2.CV_32F)
        previous = self._previous
        shift, response = cv2.phaseCorrelate(previous, small, window)
        self._previous = small
        dx, dy = float(shift[0] / scale), float(shift[1] / scale)
        self.last_instant_shift = (dx, dy)
        self.last_method = "phase"
        # Phase correlation is fast but can lock onto moving particles. Try
        # ECC translation when its response is weak or the displacement is
        # implausibly large, using the previous frame as the template.
        if (not math.isfinite(float(response)) or float(response) < 0.08 or
                abs(dx) > frame.shape[1] * 0.12 or
                abs(dy) > frame.shape[0] * 0.12):
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                ecc, warp = cv2.findTransformECC(
                    previous, small, warp, cv2.MOTION_TRANSLATION,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4),
                    None, 1)
                ex, ey = float(warp[0, 2] / scale), float(warp[1, 2] / scale)
                if math.isfinite(ex) and math.isfinite(ey):
                    dx, dy, response = ex, ey, float(np.clip(ecc, 0.0, 1.0))
                    self.last_instant_shift = (dx, dy)
                    self.last_method = "ecc"
            except cv2.error:
                pass
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return tuple(self._cumulative), 0.0
        # A bad registration must not send the persistent mask off screen.
        if abs(dx) <= frame.shape[1] * 0.2 and abs(dy) <= frame.shape[0] * 0.2:
            self._cumulative += (dx, dy)
        else:
            response = 0.0
        return (float(self._cumulative[0]), float(self._cumulative[1])), \
            float(np.clip(response, 0.0, 1.0))


def _largest_polygon(mask: np.ndarray,
                     epsilon_ratio: float = 0.006) -> List[Point]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(2.0, epsilon_ratio * cv2.arcLength(contour, True))
    points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in points]


class SubstrateSegmenter:
    """Initialize a low-saturation substrate mask, then move it by registration."""

    def __init__(self, config: AutoRecognitionConfig) -> None:
        self.config = config
        self.reference_mask: Optional[np.ndarray] = None
        self.reference_polygon: List[Point] = []
        self.initial_confidence = 0.0

    @staticmethod
    def _horizontal_band(frame: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Find v1's two long substrate edges despite its colour gradient."""

        lab = cv2.cvtColor(cv2.GaussianBlur(frame, (0, 0), 9.0),
                           cv2.COLOR_BGR2LAB).astype(np.float32)
        gradient = (np.abs(cv2.Sobel(lab[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)) +
                    1.5 * np.abs(cv2.Sobel(lab[:, :, 2], cv2.CV_32F, 0, 1, ksize=3)))
        row_score = np.percentile(gradient, 80, axis=1).astype(np.float32)
        row_score = cv2.GaussianBlur(row_score[:, None], (1, 31), 0).ravel()
        candidates: List[int] = []
        for y in np.argsort(row_score)[::-1]:
            y = int(y)
            if all(abs(y - previous) > 35 for previous in candidates):
                candidates.append(y)
            if len(candidates) >= 12:
                break
        h, w = frame.shape[:2]
        pairs = []
        for first in candidates:
            for second in candidates:
                top, bottom = sorted((first, second))
                separation = bottom - top
                if 0.30 * h <= separation <= 0.60 * h:
                    pairs.append((float(row_score[top] + row_score[bottom]),
                                  top, bottom))
        if not pairs:
            return None, 0.0
        score, top, bottom = max(pairs)
        baseline = float(np.median(row_score) + 1e-6)
        edge_strength = score / (2.0 * baseline)
        if edge_strength < 1.35:
            return None, 0.0
        mask = np.zeros((h, w), np.uint8)
        mask[max(0, top):min(h, bottom + 1)] = 255
        confidence = float(np.clip(0.45 + 0.18 * (edge_strength - 1.0),
                                   0.45, 0.92))
        return mask, confidence

    def _refine_band_mask(self, frame: np.ndarray, band_mask: np.ndarray,
                          exclusion_mask: Optional[np.ndarray]) \
            -> Optional[np.ndarray]:
        """Use the edge band as a GrabCut seed, not as the final boundary."""

        h, w = frame.shape[:2]
        scale = min(1.0, self.config.substrate_analysis_width / max(1, w))
        sw, sh = max(16, round(w * scale)), max(16, round(h * scale))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        small_band = cv2.resize(band_mask, (sw, sh),
                                interpolation=cv2.INTER_NEAREST)
        rows = np.flatnonzero(np.any(small_band > 0, axis=1))
        if rows.size < 2:
            return None
        top, bottom = int(rows[0]), int(rows[-1])
        band_height = bottom - top + 1

        # Pixels in the coarse band are only probable foreground.  A narrow
        # central core teaches GrabCut the per-frame substrate colour, while
        # the outer strips teach it the current microscope background colour.
        seeds = np.full((sh, sw), cv2.GC_BGD, np.uint8)
        margin = max(3, round(0.06 * band_height))
        probable_top = max(0, top - margin)
        probable_bottom = min(sh, bottom + margin + 1)
        seeds[probable_top:probable_bottom] = cv2.GC_PR_BGD
        seeds[top:bottom + 1] = cv2.GC_PR_FGD

        inset_y = max(2, round(0.18 * band_height))
        core_top = min(bottom, top + inset_y)
        core_bottom = max(core_top + 1, bottom - inset_y + 1)
        core_left = round(0.10 * sw)
        core_right = max(core_left + 1, round(0.60 * sw))
        seeds[core_top:core_bottom, core_left:core_right] = cv2.GC_FGD

        if exclusion_mask is not None:
            excluded = cv2.resize(exclusion_mask, (sw, sh),
                                  interpolation=cv2.INTER_NEAREST) > 0
            # Balls and the beam are not background evidence.  Keeping them
            # probable foreground prevents holes in the workspace boundary.
            seeds[excluded & (small_band > 0)] = cv2.GC_PR_FGD

        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(small, seeds, None, background_model,
                        foreground_model, 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return None
        raw = np.where((seeds == cv2.GC_FGD) |
                       (seeds == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        close_size = max(3, (round(sw * 0.010) | 1))
        open_size = max(3, (round(sw * 0.005) | 1))
        raw = cv2.morphologyEx(
            raw, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (close_size, close_size)))
        raw = cv2.morphologyEx(
            raw, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (open_size, open_size)))

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(raw)
        if count <= 1:
            return None
        core = np.zeros_like(raw)
        core[core_top:core_bottom, core_left:core_right] = 1
        choices = []
        for component_id in range(1, count):
            component = labels == component_id
            overlap = int(np.count_nonzero(component & (core > 0)))
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            choices.append((overlap, area, component_id))
        overlap, _area, component_id = max(choices)
        if overlap < max(8, int(0.08 * np.count_nonzero(core))):
            return None

        component = (labels == component_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        filled = np.zeros_like(component)
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)],
                         -1, 255, -1)
        refined = cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST)

        band_area = max(1, np.count_nonzero(band_mask))
        area_ratio = np.count_nonzero(refined) / band_area
        ys, xs = np.nonzero(refined)
        if (not 0.45 <= area_ratio <= 1.18 or xs.size == 0 or
                np.ptp(xs) < 0.45 * w or np.ptp(ys) < 0.45 * (bottom - top)):
            return None
        return refined

    def initialize(self, frame: np.ndarray,
                   exclusion_mask: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, List[Point], float]:
        h, w = frame.shape[:2]
        band_mask, band_confidence = self._horizontal_band(frame)
        if band_mask is not None:
            refined = self._refine_band_mask(frame, band_mask, exclusion_mask)
            substrate_mask = refined if refined is not None else band_mask
            polygon = _largest_polygon(substrate_mask, epsilon_ratio=0.0025)
            self.reference_mask = substrate_mask
            self.reference_polygon = polygon
            refinement_bonus = 0.06 if refined is not None else -0.08
            confidence = float(np.clip(band_confidence + refinement_bonus,
                                       0.0, 0.96))
            self.initial_confidence = confidence
            return substrate_mask.copy(), polygon, confidence
        scale = min(1.0, self.config.substrate_analysis_width / w)
        sw, sh = max(8, round(w * scale)), max(8, round(h * scale))
        smooth = cv2.GaussianBlur(frame, (0, 0), max(2.0, 7.0 * scale))
        small = cv2.resize(smooth, (sw, sh), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Chroma receives more weight than brightness because v1 has a strong
        # illumination gradient but a persistent pale-purple substrate tint.
        features = lab.reshape(-1, 3).copy()
        features[:, 0] *= 0.45
        features[:, 1:] *= 1.8
        cv2.setRNGSeed(8821)
        _compact, labels, centers = cv2.kmeans(
            features, self.config.substrate_clusters, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2),
            4, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(sh, sw)
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(5, int(sw * 0.045) | 1),) * 2)
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(3, int(sw * 0.012) | 1),) * 2)
        image_area = float(sw * sh)
        center_pt = (sw // 2, sh // 2)
        # A single colour is insufficient for the gradient.  Rank clusters by
        # substrate likelihood, then combine the best chromatically adjacent
        # clusters before selecting a connected component.
        ranked = []
        for cluster_id in range(len(centers)):
            cluster = labels == cluster_id
            area_ratio = np.count_nonzero(cluster) / image_area
            mean_bgr = cv2.mean(small, mask=cluster.astype(np.uint8))[:3]
            purple_score = (mean_bgr[0] - mean_bgr[2]) / 35.0
            center_bonus = float(cluster[center_pt[1], center_pt[0]])
            ranked.append((1.2 * purple_score + 0.8 * center_bonus +
                           min(area_ratio, 0.35), cluster_id))
        best_mask = None
        best_score = -1e9
        ordered = [cluster_id for _score, cluster_id in sorted(ranked, reverse=True)]
        for keep_count in range(1, min(4, len(ordered)) + 1):
            selected = set(ordered[:keep_count])
            raw = np.isin(labels, list(selected)).astype(np.uint8) * 255
            raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel_close)
            raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel_open)
            count, components, stats, centroids = cv2.connectedComponentsWithStats(raw)
            for component_id in range(1, count):
                area = float(stats[component_id, cv2.CC_STAT_AREA])
                area_ratio = area / image_area
                if area_ratio < self.config.substrate_min_area_ratio:
                    continue
                component = (components == component_id).astype(np.uint8) * 255
                _x, _y, cw, ch = stats[component_id, :4]
                cx, cy = centroids[component_id]
                center_distance = math.hypot(cx - center_pt[0], cy - center_pt[1])
                center_score = 1.0 - center_distance / math.hypot(sw, sh)
                contains_center = component[center_pt[1], center_pt[0]] > 0
                span_score = 0.55 * cw / sw + 0.45 * ch / sh
                border_count = (np.count_nonzero(component[0]) +
                                np.count_nonzero(component[-1]) +
                                np.count_nonzero(component[:, 0]) +
                                np.count_nonzero(component[:, -1]))
                border_penalty = border_count / max(1.0, 2.0 * (sw + sh))
                mean_bgr = cv2.mean(small, mask=component)[:3]
                purple_score = (mean_bgr[0] - mean_bgr[2]) / 35.0
                # v1's substrate is a broad horizontal body. Penalize a union
                # that floods nearly the whole frame or fragments into a thin
                # colour-gradient stripe.
                useful_area = 1.0 - abs(area_ratio - 0.42)
                flood_penalty = max(0.0, area_ratio - 0.72) * 8.0
                score = (1.3 * float(contains_center) + 1.2 * center_score +
                         1.8 * span_score + 1.0 * purple_score +
                         0.8 * useful_area - 1.0 * border_penalty - flood_penalty)
                if score > best_score:
                    best_score, best_mask = score, component
        if best_mask is None:
            return np.zeros((h, w), np.uint8), [], 0.0
        # The colour cluster describes the substrate interior.  Its external
        # contour is filled so small defects remain available to the separate
        # damage detector rather than punching arbitrary holes in the workspace.
        contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(best_mask)
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        full = cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST)
        # Smooth staircase edges from the low-resolution colour model.
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, smooth_kernel)
        full = cv2.morphologyEx(full, cv2.MORPH_OPEN, smooth_kernel)
        if exclusion_mask is not None:
            # Exclusions affect confidence, not the outer substrate boundary.
            visible = cv2.bitwise_and(full, cv2.bitwise_not(exclusion_mask))
            visible_ratio = np.count_nonzero(visible) / max(1, np.count_nonzero(full))
        else:
            visible_ratio = 1.0
        polygon = _largest_polygon(full, epsilon_ratio=0.0025)
        area_ratio = np.count_nonzero(full) / float(h * w)
        confidence = float(np.clip(
            0.35 + 0.45 * min(1.0, area_ratio / 0.25) +
            0.20 * visible_ratio, 0.0, 1.0))
        self.reference_mask = full
        self.reference_polygon = polygon
        self.initial_confidence = confidence
        return full.copy(), polygon, confidence

    def at_shift(self, shift: Point) -> Tuple[np.ndarray, List[Point], float]:
        if self.reference_mask is None:
            raise RuntimeError("substrate segmenter has not been initialized")
        h, w = self.reference_mask.shape
        matrix = np.float32([[1.0, 0.0, shift[0]],
                             [0.0, 1.0, shift[1]]])
        mask = cv2.warpAffine(self.reference_mask, matrix, (w, h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT)
        visible = np.count_nonzero(mask) / max(1, np.count_nonzero(self.reference_mask))
        confidence = self.initial_confidence * min(1.0, visible / 0.9)
        return mask, _largest_polygon(mask, epsilon_ratio=0.0025), float(confidence)


class BeamSpotDetector:
    """Detect the compact core of the saturated cyan/green laser region."""

    def detect(self, frame: np.ndarray,
               particles: Sequence[Particle] = ()) \
            -> Tuple[Optional[Point], float, np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Broad hue limits retain the green core while rejecting pale substrate.
        # The balls peak around hue 43 in v1; the cyan laser core/tails are
        # mostly above 55.  The hue floor plus ball masks avoids false binding.
        mask = cv2.inRange(hsv, (55, 65, 120), (105, 255, 255))
        for particle in particles:
            cv2.circle(mask,
                       tuple(round(v) for v in particle.position_px),
                       max(3, round(particle.radius_px * 1.25)), 0, -1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return None, 0.0, mask
        choices = []
        for idx in range(1, count):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area < 8:
                continue
            component = labels == idx
            saturation = float(hsv[:, :, 1][component].mean()) / 255.0
            choices.append((area * saturation, idx, saturation))
        if not choices:
            return None, 0.0, mask
        _score, idx, saturation = max(choices)
        area = float(stats[idx, cv2.CC_STAT_AREA])
        confidence = float(np.clip(0.4 * saturation + 0.6 * min(1.0, area / 150.0),
                                   0.0, 1.0))
        return tuple(map(float, centroids[idx])), confidence, mask


class ForbiddenRegionDetector:
    """Extract conservative damage/obstacle candidates inside the substrate."""

    def __init__(self, config: AutoRecognitionConfig) -> None:
        self.config = config

    def detect(self, frame: np.ndarray, substrate_mask: np.ndarray,
               particles: Sequence[Particle], beam_mask: np.ndarray
               ) -> List[RegionCandidate]:
        valid = substrate_mask.copy()
        exclusion = beam_mask.copy()
        for particle in particles:
            radius = max(3, round(particle.radius_px *
                                  self.config.ball_exclusion_scale))
            cv2.circle(exclusion,
                       tuple(round(v) for v in particle.position_px),
                       radius, 255, -1)
        exclusion = cv2.dilate(exclusion, np.ones((7, 7), np.uint8))
        valid[exclusion > 0] = 0
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        low_frequency = cv2.GaussianBlur(lab, (0, 0), 13.0)
        residual = np.linalg.norm(lab - low_frequency, axis=2)
        values = residual[valid > 0]
        if values.size < 100:
            return []
        threshold = max(10.0, float(np.percentile(values, 97.5)))
        candidate_mask = ((residual >= threshold) & (valid > 0)).astype(np.uint8) * 255
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE,
                                          np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        max_area = frame.shape[0] * frame.shape[1] * self.config.max_candidate_area_ratio
        result: List[RegionCandidate] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = float(cv2.contourArea(contour))
            if area < self.config.min_candidate_area_px2 or area > max_area:
                continue
            epsilon = max(1.5, 0.025 * cv2.arcLength(contour, True))
            points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(points) < 3:
                continue
            contour_mask = np.zeros_like(candidate_mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            strength = float(residual[contour_mask > 0].mean())
            confidence = float(np.clip((strength - threshold * 0.65) /
                                       max(1.0, threshold), 0.2, 0.85))
            # Traditional vision cannot reliably separate a crack from debris;
            # keep the semantic status explicit until segmentation labels exist.
            result.append(RegionCandidate(
                kind="damage_or_obstacle_candidate",
                polygon=[(float(x), float(y)) for x, y in points],
                confidence=confidence,
                area_px2=area))
            if len(result) >= self.config.max_candidates:
                break
        return result


class Alg2AutoRecognizer:
    """Compose YOLO balls, registered substrate and forbidden candidates."""

    def __init__(self, weights: Optional[str] = None,
                 config: Optional[AutoRecognitionConfig] = None,
                 particle_detector=None,
                 manual_substrate_polygon: Optional[Sequence[Point]] = None) -> None:
        self.config = config or AutoRecognitionConfig()
        if particle_detector is None:
            if not weights:
                raise ValueError("weights are required without particle_detector")
            particle_detector = YoloDetector(
                weights, imgsz=self.config.yolo_imgsz,
                conf=self.config.yolo_confidence)
        self.particle_detector = particle_detector
        self.tracker = ParticleTracker(
            max_jump_px=self.config.tracker_max_jump_px,
            max_lost_frames=self.config.tracker_max_lost_frames)
        self.motion = GlobalMotionEstimator(self.config.registration_width)
        self.substrate = SubstrateSegmenter(self.config)
        self.beam = BeamSpotDetector()
        self.forbidden = ForbiddenRegionDetector(self.config)
        self.manual_substrate_polygon = list(manual_substrate_polygon or [])
        self._reference_candidates: Optional[List[RegionCandidate]] = None
        self._last_substrate_refresh = -1
        self._previous_track_ids: set[int] = set()

    def reset(self) -> None:
        self.tracker.reset()
        self.motion.reset()
        self.substrate = SubstrateSegmenter(self.config)
        self._reference_candidates = None
        self._last_substrate_refresh = -1
        self._previous_track_ids = set()

    def process(self, frame: np.ndarray, frame_id: int,
                timestamp_s: float = 0.0) -> AutoRecognitionResult:
        clean_frame, overlay_mask = suppress_ui_overlays(frame)
        shift, registration_confidence = self.motion.update(clean_frame)
        detections, _ambiguous = self.particle_detector.detect_particles(clean_frame)
        particles = self.tracker.update(detections, frame_id)
        beam_center, beam_confidence, beam_mask = self.beam.detect(
            clean_frame, particles)
        exclusion = beam_mask.copy()
        for particle in particles:
            cv2.circle(exclusion,
                       tuple(round(v) for v in particle.position_px),
                       max(2, round(particle.radius_px * 1.2)), 255, -1)
        tracking_reason = ""
        if not self.config.auto_substrate:
            polygon_seed = self.manual_substrate_polygon or [
                (0.0, 0.0), (float(frame.shape[1] - 1), 0.0),
                (float(frame.shape[1] - 1), float(frame.shape[0] - 1)),
                (0.0, float(frame.shape[0] - 1))]
            if self.substrate.reference_mask is None:
                substrate_mask = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillPoly(substrate_mask, [np.asarray(
                    polygon_seed, np.int32)], 255)
                self.substrate.reference_mask = substrate_mask
                self.substrate.reference_polygon = list(polygon_seed)
                self.substrate.initial_confidence = 1.0
            substrate_mask, polygon, substrate_confidence = \
                self.substrate.at_shift(shift)
            self._last_substrate_refresh = frame_id
        elif self.substrate.reference_mask is None:
            substrate_mask, polygon, substrate_confidence = \
                self.substrate.initialize(clean_frame, exclusion)
            self._last_substrate_refresh = frame_id
        else:
            substrate_mask, polygon, substrate_confidence = \
                self.substrate.at_shift(shift)
            interval = max(1, int(self.config.substrate_refresh_interval))
            if frame_id - self._last_substrate_refresh >= interval:
                candidate_segmenter = SubstrateSegmenter(self.config)
                fresh_mask, fresh_polygon, fresh_confidence = \
                    candidate_segmenter.initialize(frame, exclusion)
                if fresh_polygon:
                    predicted_edge = cv2.morphologyEx(
                        substrate_mask, cv2.MORPH_GRADIENT,
                        np.ones((3, 3), np.uint8))
                    fresh_edge = cv2.morphologyEx(
                        fresh_mask, cv2.MORPH_GRADIENT,
                        np.ones((3, 3), np.uint8))
                    distance = cv2.distanceTransform(
                        cv2.bitwise_not(predicted_edge), cv2.DIST_L2, 5)
                    edge_values = distance[fresh_edge > 0]
                    alignment_error = (float(np.median(edge_values))
                                       if edge_values.size else float("inf"))
                    if alignment_error <= self.config.substrate_alignment_max_error_px:
                        self.substrate = candidate_segmenter
                        self.motion.reset()
                        shift = (0.0, 0.0)
                        substrate_mask, polygon = fresh_mask, fresh_polygon
                        substrate_confidence = fresh_confidence
                        self._reference_candidates = None
                        self._last_substrate_refresh = frame_id
                    else:
                        tracking_reason = "substrate_tracking_lost"
        distance = cv2.distanceTransform(substrate_mask, cv2.DIST_L2, 5)
        inside = []
        for particle in particles:
            x, y = (round(particle.position_px[0]), round(particle.position_px[1]))
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                inside.append(bool(distance[y, x] >= particle.radius_px * 0.65))
            else:
                inside.append(False)
        if self._reference_candidates is None:
            self._reference_candidates = self.forbidden.detect(
                clean_frame, substrate_mask, particles, beam_mask)
        candidates = [RegionCandidate(
            kind=candidate.kind,
            polygon=[(x + shift[0], y + shift[1])
                     for x, y in candidate.polygon],
            confidence=candidate.confidence * max(0.4, registration_confidence),
            area_px2=candidate.area_px2)
            for candidate in self._reference_candidates]
        fresh = [p for p in particles if p.frame_id == frame_id]
        current_ids = {p.track_id for p in fresh}
        track_events = ([f"track_acquired:{tid}" for tid in
                         sorted(current_ids - self._previous_track_ids)] +
                        [f"track_missing:{tid}" for tid in
                         sorted(self._previous_track_ids - current_ids)])
        self._previous_track_ids = current_ids
        particle_confidence = min((p.confidence for p in fresh), default=0.0)
        overall = min(max(0.0, registration_confidence), particle_confidence)
        reasons = []
        if not polygon:
            reasons.append("substrate_not_found")
        if not fresh:
            reasons.append("no_fresh_particle_detection")
        if registration_confidence < self.config.registration_min_confidence:
            reasons.append("registration_low_confidence")
        if tracking_reason:
            reasons.append(tracking_reason)
        if overall < self.config.minimum_overall_confidence:
            reasons.append("overall_low_confidence")
        return AutoRecognitionResult(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            substrate_polygon=polygon,
            substrate_mask=substrate_mask,
            substrate_confidence=substrate_confidence,
            registration_shift_px=shift,
            registration_confidence=registration_confidence,
            particles=particles,
            particles_inside_substrate=inside,
            beam_spot_px=beam_center,
            beam_confidence=beam_confidence,
            candidates=candidates,
            overall_confidence=float(overall),
            uncertain=bool(reasons),
            uncertain_reason=";".join(reasons),
            instant_shift_px=self.motion.last_instant_shift,
            motion_method=self.motion.last_method,
            track_events=track_events,
            overlay_mask_ratio=float(np.count_nonzero(overlay_mask) /
                                     max(1, overlay_mask.size)),
        )

    @staticmethod
    def draw_overlay(frame: np.ndarray,
                     result: AutoRecognitionResult) -> np.ndarray:
        canvas = frame.copy()
        if result.substrate_polygon:
            poly = np.asarray(result.substrate_polygon, np.int32)
            cv2.polylines(canvas, [poly], True, (255, 120, 220), 2)
        for candidate in result.candidates:
            poly = np.asarray(candidate.polygon, np.int32)
            cv2.polylines(canvas, [poly], True, (0, 80, 255), 2)
        for particle, inside in zip(result.particles,
                                    result.particles_inside_substrate):
            center = tuple(round(v) for v in particle.position_px)
            color = (0, 220, 0) if inside else (0, 180, 255)
            cv2.circle(canvas, center, round(particle.radius_px), color, 2)
            cv2.putText(canvas, f"ball {particle.track_id} {'IN' if inside else 'OUT'}",
                        (center[0] + 8, center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if result.beam_spot_px is not None:
            center = tuple(round(v) for v in result.beam_spot_px)
            cv2.drawMarker(canvas, center, (0, 255, 255),
                           cv2.MARKER_CROSS, 24, 2)
        status = (f"frame={result.frame_id} conf={result.overall_confidence:.2f} "
                  f"shift=({result.registration_shift_px[0]:.1f},"
                  f"{result.registration_shift_px[1]:.1f})")
        cv2.putText(canvas, status, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(canvas, status, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas


class AutoRecognitionPipeline:
    """Adapt ``Alg2AutoRecognizer`` to the controller's VisionPipeline API."""

    def __init__(self, weights: Optional[str] = None,
                 config: Optional[AutoRecognitionConfig] = None,
                 particle_detector=None) -> None:
        self.recognizer = Alg2AutoRecognizer(
            weights=weights, config=config, particle_detector=particle_detector)
        self.tracker = self.recognizer.tracker
        self.latest_result: Optional[AutoRecognitionResult] = None

    def process(self, frame: np.ndarray, frame_id: int,
                expect_particle: bool = True) -> VisionResult:
        result = self.recognizer.process(frame, frame_id, time.time())
        if not expect_particle and result.uncertain_reason:
            reasons = [reason for reason in result.uncertain_reason.split(";")
                       if reason != "no_fresh_particle_detection"]
            result.uncertain_reason = ";".join(reasons)
            result.uncertain = bool(reasons)
        self.latest_result = result
        substrate = (SubstrateRegion(result.substrate_polygon,
                                     self.recognizer.config.substrate_safety_margin_px)
                     if result.substrate_polygon else None)
        return VisionResult(
            frame_id=frame_id,
            substrate=substrate,
            obstacles=[candidate.to_obstacle(index + 1)
                       for index, candidate in enumerate(result.candidates)],
            particles=result.particles,
            uncertain=result.uncertain,
            uncertain_reason=result.uncertain_reason,
            confidence=result.overall_confidence,
            frame=frame,
        )


def analyze_video(video_path: str, weights: str, output_dir: str,
                  stride: int = 10, max_frames: Optional[int] = None,
                  config: Optional[AutoRecognitionConfig] = None) -> dict:
    """Analyze sampled video frames and write JSONL, summary and overlay MP4."""

    if stride < 1:
        raise ValueError("stride must be >= 1")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    recognizer = Alg2AutoRecognizer(weights, config=config)
    jsonl_path = output / "recognition.jsonl"
    overlay_path = output / "recognition_overlay.mp4"
    writer = cv2.VideoWriter(str(overlay_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             max(1.0, fps / stride), (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create overlay video: {overlay_path}")
    analyzed = uncertain_count = 0
    particle_ids = set()
    inside_observations = 0
    confidence_sum = 0.0
    started = time.time()
    try:
        with jsonl_path.open("w", encoding="utf-8") as report:
            frame_id = -1
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_id += 1
                if frame_id % stride:
                    continue
                result = recognizer.process(frame, frame_id, frame_id / fps)
                report.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                writer.write(recognizer.draw_overlay(frame, result))
                analyzed += 1
                uncertain_count += int(result.uncertain)
                confidence_sum += result.overall_confidence
                particle_ids.update(p.track_id for p in result.particles)
                inside_observations += sum(result.particles_inside_substrate)
                if max_frames is not None and analyzed >= max_frames:
                    break
    finally:
        capture.release()
        writer.release()
    summary = {
        "video": os.path.abspath(video_path),
        "weights": os.path.abspath(weights),
        "source": {"width": width, "height": height, "fps": fps,
                   "total_frames": total_frames},
        "stride": stride,
        "analyzed_frames": analyzed,
        "unique_track_ids": sorted(particle_ids),
        "inside_substrate_observations": inside_observations,
        "uncertain_frames": uncertain_count,
        "mean_overall_confidence": confidence_sum / max(1, analyzed),
        "elapsed_s": time.time() - started,
        "outputs": {"jsonl": str(jsonl_path), "overlay_video": str(overlay_path)},
        "limitations": [
            "damage and obstacle are conservative traditional-vision candidates",
            "semantic separation requires a substrate/damage/obstacle segmentation model",
            "results must be reviewed before enabling real Alg2 motor control",
        ],
    }
    summary_path = output / "summary.json"
    summary["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Offline Alg2 workspace recognition (never moves motors)")
    parser.add_argument("--video", default=str(project / "Dataset" / "v1.mp4"))
    parser.add_argument("--weights", default=str(project / "weight" / "Duan_best.pt"))
    parser.add_argument("--output", default=str(project / "Reports" / "v1_auto_recognition"))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = analyze_video(args.video, args.weights, args.output,
                            stride=args.stride, max_frames=args.max_frames)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
