"""
SpotZoom innovation frontier v33.

This module adds a lightweight response-reliability gate inspired by:
- MOSSE/DCF tracking reliability (correlation response + PSR confidence)
- OpenCV CSRT-style local correlation tracking
- Microscopy pipelines that prefer conservative, confidence-aware center blending

Only numpy + optional OpenCV are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency guard
    cv2 = None


@dataclass
class ResponseReliabilityConfig:
    template_radius: int = 22
    search_radius: int = 44
    min_template_std: float = 2.0
    min_response: float = 0.2
    min_psr: float = 3.2
    psr_exclusion_radius: int = 5
    low_confidence_floor: float = 0.07
    low_confidence_response_boost: float = 0.08
    max_refine_shift_px: float = 18.0
    min_refine_shift_px: float = 0.2
    blend_base: float = 0.34
    blend_max: float = 0.72
    confidence_gain: float = 1.03
    template_update_momentum: float = 0.24
    template_update_confidence: float = 0.22
    template_update_response: float = 0.2


@dataclass
class ResponseReliabilityResult:
    center: Tuple[float, float]
    confidence: float
    accepted: bool
    refined: bool
    rejected: bool
    reason: Optional[str]
    response: float
    psr: float
    blend_weight: float
    template_age: int
    template_updated: bool


class ResponseReliabilityGate:
    """Template-matching reliability gate using response-map and PSR checks."""

    def __init__(self, config: Optional[ResponseReliabilityConfig] = None):
        self.config = config or ResponseReliabilityConfig()
        self._template: Optional[np.ndarray] = None
        self._template_age: int = 0

    def reset(self) -> None:
        self._template = None
        self._template_age = 0

    @staticmethod
    def _safe_float(value: float, default: float = 0.0) -> float:
        try:
            x = float(value)
        except Exception:
            return default
        if not np.isfinite(x):
            return default
        return x

    @staticmethod
    def _as_gray_float32(frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.size == 0:
            return None
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
        return np.asarray(gray, dtype=np.float32)

    def _extract_patch(
        self,
        gray: np.ndarray,
        center: Tuple[float, float],
        radius: int,
    ) -> Optional[np.ndarray]:
        h, w = gray.shape[:2]
        if h <= 0 or w <= 0:
            return None
        r = max(2, int(radius))
        size = 2 * r + 1
        cx = self._safe_float(center[0], default=np.nan)
        cy = self._safe_float(center[1], default=np.nan)
        if not np.isfinite(cx) or not np.isfinite(cy):
            return None

        xi = int(round(cx))
        yi = int(round(cy))
        x0 = xi - r
        x1 = xi + r + 1
        y0 = yi - r
        y1 = yi + r + 1

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
        if patch.shape != (size, size):
            return None
        return np.asarray(patch, dtype=np.float32)

    def _build_search_region(
        self,
        gray: np.ndarray,
        center: Tuple[float, float],
        radius: int,
    ) -> Optional[Tuple[np.ndarray, int, int]]:
        h, w = gray.shape[:2]
        if h <= 0 or w <= 0:
            return None

        r = max(3, int(radius))
        cx = int(round(self._safe_float(center[0], default=0.0)))
        cy = int(round(self._safe_float(center[1], default=0.0)))
        x0 = max(0, cx - r)
        y0 = max(0, cy - r)
        x1 = min(w, cx + r + 1)
        y1 = min(h, cy + r + 1)
        region = gray[y0:y1, x0:x1]
        if region.size == 0:
            return None
        return np.asarray(region, dtype=np.float32), x0, y0

    def _compute_psr(self, response_map: np.ndarray, peak_loc: Tuple[int, int]) -> float:
        rm = np.asarray(response_map, dtype=np.float32)
        if rm.ndim != 2 or rm.size == 0:
            return 0.0
        peak_x = int(peak_loc[0])
        peak_y = int(peak_loc[1])
        h, w = rm.shape[:2]
        if peak_x < 0 or peak_x >= w or peak_y < 0 or peak_y >= h:
            return 0.0

        ex = max(1, int(self.config.psr_exclusion_radius))
        x0 = max(0, peak_x - ex)
        x1 = min(w, peak_x + ex + 1)
        y0 = max(0, peak_y - ex)
        y1 = min(h, peak_y + ex + 1)

        mask = np.ones((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = False
        sidelobe = rm[mask]
        if sidelobe.size < 8:
            return 0.0

        peak = float(rm[peak_y, peak_x])
        mu = float(np.mean(sidelobe))
        sigma = float(np.std(sidelobe))
        if sigma <= 1e-6:
            return 0.0
        return float((peak - mu) / sigma)

    def _ensure_template(
        self,
        gray: np.ndarray,
        center: Tuple[float, float],
    ) -> bool:
        patch = self._extract_patch(gray, center, int(self.config.template_radius))
        if patch is None:
            return False
        if float(np.std(patch)) < float(self.config.min_template_std):
            return False
        self._template = patch
        self._template_age = 0
        return True

    def _reject(
        self,
        center: Tuple[float, float],
        confidence: float,
        reason: str,
        response: float = 0.0,
        psr: float = 0.0,
        blend_weight: float = 0.0,
    ) -> ResponseReliabilityResult:
        self._template_age += 1
        return ResponseReliabilityResult(
            center=(float(center[0]), float(center[1])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            accepted=False,
            refined=False,
            rejected=True,
            reason=reason,
            response=float(response),
            psr=float(psr),
            blend_weight=float(blend_weight),
            template_age=int(self._template_age),
            template_updated=False,
        )

    def update(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        confidence: float,
    ) -> ResponseReliabilityResult:
        conf = float(np.clip(self._safe_float(confidence, default=0.0), 0.0, 1.0))
        cx = self._safe_float(center[0], default=np.nan)
        cy = self._safe_float(center[1], default=np.nan)
        if not np.isfinite(cx) or not np.isfinite(cy):
            return self._reject((0.0, 0.0), conf, reason="invalid_center")

        gray = self._as_gray_float32(frame)
        if gray is None:
            return self._reject((cx, cy), conf, reason="invalid_frame")
        if cv2 is None:
            return self._reject((cx, cy), conf, reason="opencv_unavailable")

        if self._template is None:
            if not self._ensure_template(gray, (cx, cy)):
                return self._reject((cx, cy), conf, reason="template_init_failed")
            return ResponseReliabilityResult(
                center=(cx, cy),
                confidence=conf,
                accepted=True,
                refined=False,
                rejected=False,
                reason=None,
                response=0.0,
                psr=0.0,
                blend_weight=0.0,
                template_age=int(self._template_age),
                template_updated=True,
            )

        template = self._template
        search = self._build_search_region(gray, (cx, cy), int(self.config.search_radius))
        if search is None:
            return self._reject((cx, cy), conf, reason="invalid_search_region")
        search_patch, offset_x, offset_y = search

        th, tw = template.shape[:2]
        sh, sw = search_patch.shape[:2]
        if sh < th or sw < tw:
            return self._reject((cx, cy), conf, reason="search_region_too_small")

        response_map = cv2.matchTemplate(search_patch, template, cv2.TM_CCOEFF_NORMED)
        if response_map is None or response_map.size == 0:
            return self._reject((cx, cy), conf, reason="match_template_failed")

        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(response_map)
        peak = float(max_val)
        psr = float(self._compute_psr(response_map, (int(max_loc[0]), int(max_loc[1]))))

        conf_floor = float(np.clip(self.config.low_confidence_floor, 0.0, 1.0))
        low_conf_boost = float(max(0.0, self.config.low_confidence_response_boost))
        min_response = float(self.config.min_response)
        if conf < conf_floor:
            min_response += low_conf_boost

        if peak < min_response:
            return self._reject((cx, cy), conf, reason="low_response", response=peak, psr=psr)
        if psr < float(self.config.min_psr):
            return self._reject((cx, cy), conf, reason="low_psr", response=peak, psr=psr)

        peak_center_x = offset_x + int(max_loc[0]) + (tw - 1) * 0.5
        peak_center_y = offset_y + int(max_loc[1]) + (th - 1) * 0.5
        measured = np.asarray([cx, cy], dtype=np.float32)
        matched = np.asarray([peak_center_x, peak_center_y], dtype=np.float32)

        response_margin = float(np.clip((peak - min_response) / max(1e-6, 1.0 - min_response), 0.0, 1.0))
        blend = float(self.config.blend_base) * (0.45 + 0.55 * response_margin) * (1.0 - 0.45 * conf)
        blend = float(np.clip(blend, 0.0, float(self.config.blend_max)))
        refined_vec = (1.0 - blend) * measured + blend * matched
        shift_px = float(np.linalg.norm(refined_vec - measured))

        if shift_px > float(self.config.max_refine_shift_px):
            return self._reject(
                (cx, cy),
                conf,
                reason="refine_shift_too_large",
                response=peak,
                psr=psr,
                blend_weight=blend,
            )

        if shift_px < float(self.config.min_refine_shift_px):
            out_center = (cx, cy)
            refined_flag = False
            blend = 0.0
        else:
            out_center = (float(refined_vec[0]), float(refined_vec[1]))
            refined_flag = True

        template_updated = False
        should_update = (
            conf >= float(self.config.template_update_confidence)
            and peak >= float(self.config.template_update_response)
        )
        if should_update:
            new_patch = self._extract_patch(gray, out_center, int(self.config.template_radius))
            if new_patch is not None and new_patch.shape == template.shape:
                alpha = float(np.clip(self.config.template_update_momentum, 0.0, 1.0))
                self._template = (1.0 - alpha) * template + alpha * new_patch
                self._template = np.asarray(self._template, dtype=np.float32)
                template_updated = True
                self._template_age = 0
            else:
                self._template_age += 1
        else:
            self._template_age += 1

        out_conf = float(np.clip(conf * float(self.config.confidence_gain), 0.01, 0.99))
        return ResponseReliabilityResult(
            center=out_center,
            confidence=out_conf,
            accepted=True,
            refined=refined_flag,
            rejected=False,
            reason=None,
            response=peak,
            psr=psr,
            blend_weight=blend,
            template_age=int(self._template_age),
            template_updated=template_updated,
        )
