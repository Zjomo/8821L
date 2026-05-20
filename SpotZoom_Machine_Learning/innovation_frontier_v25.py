"""
Frontier v25 modules for SpotZoom.

This module adds lightweight, dependency-free components inspired by:
- DECODE (TuragaLab): Deep context-dependent localization with uncertainty prediction.
- Picasso (jungmannlab): GPU-accelerated sub-pixel fitting with drift correction.
- SeReNet (Tsinghua): Physics-driven self-supervised PSF estimation.
- REALM (MSiemons): Zernike-mode iterative AO correction.
- BayesDL-SIM / Bayesian DPA-TISR: Bayesian uncertainty quantification.
- SSR-SIM: Artifact statistics + physics prior self-supervised assessment.
- LF-denoising: Multi-angle redundancy self-supervised denoising.

All components are pure numpy+cv2 with zero external ML dependencies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Deque, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 1. UncertaintyAwareLocalizer  (inspired by DECODE)
#    Deep-context localization with per-spot uncertainty estimate.
# ============================================================

@dataclass
class UncertaintyLocalizerConfig:
    """Configuration for uncertainty-aware spot localization."""
    # Context window half-size around the spot for feature extraction
    context_half: int = 16
    # Minimum spot SNR to attempt localization
    min_snr: float = 3.0
    # Maximum number of context samples to keep for adaptive threshold
    max_context_samples: int = 200
    # Percentile for adaptive intensity threshold
    adaptive_percentile: float = 95.0
    # Minimum centroid shift between successive frames to flag instability
    instability_threshold_px: float = 2.0


@dataclass
class UncertaintyLocalization:
    """Result of uncertainty-aware spot localization."""
    center: Optional[Tuple[float, float]]
    uncertainty_px: float  # 1-sigma uncertainty in pixels
    snr: float
    context_quality: float  # 0-1, how well the context matches expected PSF
    is_stable: bool
    confidence: float  # 0-1, overall confidence score


class UncertaintyAwareLocalizer:
    """Performs spot localization with uncertainty quantification.

    Inspired by DECODE's per-emitter uncertainty prediction, this module
    estimates localization precision based on local SNR, PSF shape
    consistency, and temporal stability of the centroid.
    """

    def __init__(self, config: Optional[UncertaintyLocalizerConfig] = None):
        self.config = config or UncertaintyLocalizerConfig()
        self._prev_centers: Deque[Tuple[float, float]] = deque(maxlen=10)
        self._context_intensities: Deque[float] = deque(
            maxlen=self.config.max_context_samples
        )

    def reset(self) -> None:
        self._prev_centers.clear()
        self._context_intensities.clear()

    @staticmethod
    def _compute_snr(patch: np.ndarray) -> float:
        """Compute SNR of a spot patch: (peak - median) / std(background)."""
        if patch.size < 9:
            return 0.0
        median_bg = float(np.median(patch))
        std_bg = float(np.std(patch))
        if std_bg < 1e-6:
            return 0.0
        peak = float(np.max(patch))
        return (peak - median_bg) / std_bg

    @staticmethod
    def _compute_psf_symmetry(patch: np.ndarray) -> float:
        """Measure PSF symmetry (0=asymmetric, 1=perfectly symmetric)."""
        if patch.size < 9:
            return 0.0
        h, w = patch.shape[:2]
        cy, cx = h // 2, w // 2
        # Compare quadrants
        q1 = patch[:cy, :cx]
        q2 = patch[:cy, cx:]
        q3 = patch[cy:, :cx]
        q4 = patch[cy:, cx:]
        # Normalize sizes
        min_h = min(q1.shape[0], q2.shape[0], q3.shape[0], q4.shape[0])
        min_w = min(q1.shape[1], q2.shape[1], q3.shape[1], q4.shape[1])
        if min_h < 2 or min_w < 2:
            return 0.0
        q1 = q1[:min_h, :min_w].astype(np.float64)
        q2 = q2[:min_h, :min_w].astype(np.float64)
        q3 = q3[:min_h, :min_w].astype(np.float64)
        q4 = q4[:min_h, :min_w].astype(np.float64)
        mean_q = (q1 + q2 + q3 + q4) / 4.0
        if np.max(mean_q) < 1e-6:
            return 0.0
        var = (
            np.mean((q1 - mean_q) ** 2)
            + np.mean((q2 - mean_q) ** 2)
            + np.mean((q3 - mean_q) ** 2)
            + np.mean((q4 - mean_q) ** 2)
        )
        max_var = np.mean(mean_q ** 2) * 4.0
        if max_var < 1e-6:
            return 1.0
        return max(0.0, 1.0 - var / max_var)

    def _estimate_uncertainty(self, snr: float, symmetry: float,
                              patch_size: int) -> float:
        """Estimate localization uncertainty (sigma) in pixels.

        Based on the Thompson formula approximation:
        sigma ≈ s / sqrt(N) where s is PSF width and N is photon count (SNR^2).
        """
        if snr < 1.0:
            return float(patch_size)
        psf_width = max(1.0, patch_size / 4.0)
        sigma = psf_width / snr
        # Asymmetry penalty
        sigma *= (1.0 + 2.0 * (1.0 - symmetry))
        return min(sigma, float(patch_size))

    def _check_temporal_stability(self, center: Tuple[float, float]) -> bool:
        """Check if centroid is stable across recent frames."""
        if len(self._prev_centers) < 3:
            return True
        shifts = [
            math.hypot(center[0] - px, center[1] - py)
            for px, py in self._prev_centers
        ]
        return float(np.mean(shifts)) < self.config.instability_threshold_px

    def localize(self, frame: np.ndarray,
                 roi_center: Tuple[int, int]) -> UncertaintyLocalization:
        """Localize a spot within a frame and estimate uncertainty.

        Args:
            frame: Grayscale image (2D numpy array).
            roi_center: Approximate center (x, y) of the spot.

        Returns:
            UncertaintyLocalization with center, uncertainty, and confidence.
        """
        h, w = frame.shape[:2]
        half = self.config.context_half
        cx, cy = int(roi_center[0]), int(roi_center[1])

        # Extract context patch
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half + 1)
        y2 = min(h, cy + half + 1)
        patch = frame[y1:y2, x1:x2].copy()

        if patch.size < 9:
            return UncertaintyLocalization(
                center=None, uncertainty_px=999.0, snr=0.0,
                context_quality=0.0, is_stable=False, confidence=0.0,
            )

        # Compute SNR
        snr = self._compute_snr(patch)
        self._context_intensities.append(snr)

        if snr < self.config.min_snr:
            return UncertaintyLocalization(
                center=None, uncertainty_px=999.0, snr=snr,
                context_quality=0.0, is_stable=False,
                confidence=max(0.0, snr / self.config.min_snr),
            )

        # Weighted centroid
        patch_f = patch.astype(np.float64)
        total = float(np.sum(patch_f))
        if total < 1e-6:
            return UncertaintyLocalization(
                center=None, uncertainty_px=999.0, snr=snr,
                context_quality=0.0, is_stable=False, confidence=0.0,
            )

        yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
        centroid_x = float(np.sum(xx * patch_f) / total) + x1
        centroid_y = float(np.sum(yy * patch_f) / total) + y1
        center = (centroid_x, centroid_y)

        # PSF symmetry
        symmetry = self._compute_psf_symmetry(patch)

        # Uncertainty
        uncertainty = self._estimate_uncertainty(snr, symmetry, patch.shape[0])

        # Temporal stability
        is_stable = self._check_temporal_stability(center)
        self._prev_centers.append(center)

        # Context quality: how well current SNR matches history
        if len(self._context_intensities) > 5:
            hist = np.array(self._context_intensities)
            ref = float(np.percentile(hist, self.config.adaptive_percentile))
            context_quality = min(1.0, snr / max(ref, 1.0))
        else:
            context_quality = 0.5

        # Overall confidence
        confidence = (
            0.3 * min(1.0, snr / 10.0)
            + 0.3 * symmetry
            + 0.2 * context_quality
            + 0.2 * (1.0 if is_stable else 0.3)
        )

        return UncertaintyLocalization(
            center=center,
            uncertainty_px=uncertainty,
            snr=snr,
            context_quality=context_quality,
            is_stable=is_stable,
            confidence=min(1.0, max(0.0, confidence)),
        )


# ============================================================
# 2. DriftCorrector  (inspired by Picasso AIM drift correction)
#    Long-term drift correction using cross-correlation.
# ============================================================

@dataclass
class DriftCorrectorConfig:
    """Configuration for drift correction."""
    # Reference frame update interval (frames)
    reference_update_interval: int = 100
    # Maximum allowed drift in pixels before resetting reference
    max_drift_px: float = 50.0
    # Cross-correlation search radius (half-size of search window)
    search_radius: int = 30
    # Reference template radius (half-size, must be smaller than search radius)
    template_radius: int = 18
    # Sub-pixel refinement method: "parabolic" or "gaussian"
    subpixel_method: str = "parabolic"
    # Minimum correlation coefficient to accept
    min_correlation: float = 0.3


@dataclass
class DriftCorrection:
    """Result of drift correction."""
    dx: float  # drift in x (pixels)
    dy: float  # drift in y (pixels)
    correlation: float
    reference_updated: bool
    cumulative_dx: float
    cumulative_dy: float


class DriftCorrector:
    """Corrects long-term drift using template cross-correlation.

    Inspired by Picasso's AIM drift correction algorithm, this module
    maintains a reference frame and computes sub-pixel drift via
    normalized cross-correlation.
    """

    def __init__(self, config: Optional[DriftCorrectorConfig] = None):
        self.config = config or DriftCorrectorConfig()
        self._reference: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._cum_dx: float = 0.0
        self._cum_dy: float = 0.0
        self._drift_history: Deque[Tuple[float, float]] = deque(maxlen=100)

    def reset(self) -> None:
        self._reference = None
        self._frame_count = 0
        self._cum_dx = 0.0
        self._cum_dy = 0.0
        self._drift_history.clear()

    def _extract_center_patch(self, frame: np.ndarray, radius: int) -> Optional[np.ndarray]:
        """Extract a centered square patch from frame."""
        h, w = frame.shape[:2]
        r = max(1, int(radius))
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r + 1), min(h, cy + r + 1)
        patch = frame[y1:y2, x1:x2].copy()
        if patch.size < 9:
            return None
        return patch

    @staticmethod
    def _subpixel_peak(corr: np.ndarray, peak_y: int, peak_x: int,
                       method: str = "parabolic") -> Tuple[float, float]:
        """Refine peak position to sub-pixel accuracy."""
        h, w = corr.shape
        dy, dx = 0.0, 0.0

        if method == "parabolic":
            # Y direction
            if 0 < peak_y < h - 1:
                y0 = float(corr[peak_y - 1, peak_x])
                y1 = float(corr[peak_y, peak_x])
                y2 = float(corr[peak_y + 1, peak_x])
                denom = 2.0 * (2.0 * y1 - y0 - y2)
                if abs(denom) > 1e-10:
                    dy = (y0 - y2) / denom
            # X direction
            if 0 < peak_x < w - 1:
                x0 = float(corr[peak_y, peak_x - 1])
                x1 = float(corr[peak_y, peak_x])
                x2 = float(corr[peak_y, peak_x + 1])
                denom = 2.0 * (2.0 * x1 - x0 - x2)
                if abs(denom) > 1e-10:
                    dx = (x0 - x2) / denom
        elif method == "gaussian":
            if 0 < peak_y < h - 1:
                y0 = math.log(max(float(corr[peak_y - 1, peak_x]), 1e-10))
                y1 = math.log(max(float(corr[peak_y, peak_x]), 1e-10))
                y2 = math.log(max(float(corr[peak_y + 1, peak_x]), 1e-10))
                denom = 2.0 * (2.0 * y1 - y0 - y2)
                if abs(denom) > 1e-10:
                    dy = (y0 - y2) / denom
            if 0 < peak_x < w - 1:
                x0 = math.log(max(float(corr[peak_y, peak_x - 1]), 1e-10))
                x1 = math.log(max(float(corr[peak_y, peak_x]), 1e-10))
                x2 = math.log(max(float(corr[peak_y, peak_x + 1]), 1e-10))
                denom = 2.0 * (2.0 * x1 - x0 - x2)
                if abs(denom) > 1e-10:
                    dx = (x0 - x2) / denom

        return (peak_x + dx, peak_y + dy)

    def update(self, frame: np.ndarray) -> DriftCorrection:
        """Process a frame and compute drift correction.

        Args:
            frame: Grayscale image (2D numpy array).

        Returns:
            DriftCorrection with per-frame and cumulative drift.
        """
        self._frame_count += 1
        search_radius = max(2, int(self.config.search_radius))
        template_radius = max(1, int(self.config.template_radius))
        if template_radius >= search_radius:
            template_radius = max(1, search_radius - 1)
        search = self._extract_center_patch(frame, search_radius)
        template_now = self._extract_center_patch(frame, template_radius)

        if search is None or template_now is None:
            return DriftCorrection(
                dx=0.0, dy=0.0, correlation=0.0,
                reference_updated=False,
                cumulative_dx=self._cum_dx,
                cumulative_dy=self._cum_dy,
            )

        # Initialize or update reference
        reference_updated = False
        if self._reference is None:
            self._reference = template_now.copy()
            reference_updated = True
        elif (self._frame_count % self.config.reference_update_interval == 0):
            self._reference = template_now.copy()
            reference_updated = True

        if self._reference is None:
            return DriftCorrection(
                dx=0.0, dy=0.0, correlation=0.0,
                reference_updated=reference_updated,
                cumulative_dx=self._cum_dx,
                cumulative_dy=self._cum_dy,
            )

        # Keep reference shape valid for current search window.
        if (
            self._reference.shape[0] > search.shape[0]
            or self._reference.shape[1] > search.shape[1]
        ):
            self._reference = template_now.copy()
            reference_updated = True

        # Compute normalized cross-correlation
        ref_f = self._reference.astype(np.float32)
        search_f = search.astype(np.float32)
        corr = cv2.matchTemplate(search_f, ref_f, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(corr)

        # Sub-pixel refinement
        peak_x, peak_y = max_loc
        sub_x, sub_y = self._subpixel_peak(
            corr, peak_y, peak_x, self.config.subpixel_method
        )

        # Convert to drift (correlation peak relative to expected centered location)
        expected_x = (search.shape[1] - self._reference.shape[1]) / 2.0
        expected_y = (search.shape[0] - self._reference.shape[0]) / 2.0
        dx = sub_x - expected_x
        dy = sub_y - expected_y

        # Reject if correlation too low
        if max_val < self.config.min_correlation:
            dx, dy = 0.0, 0.0

        # Reject if drift too large (likely reference mismatch)
        drift_mag = math.hypot(dx, dy)
        if drift_mag > self.config.max_drift_px:
            self._reference = template_now.copy()
            reference_updated = True
            dx, dy = 0.0, 0.0

        self._cum_dx += dx
        self._cum_dy += dy
        self._drift_history.append((dx, dy))

        return DriftCorrection(
            dx=dx, dy=dy, correlation=float(max_val),
            reference_updated=reference_updated,
            cumulative_dx=self._cum_dx,
            cumulative_dy=self._cum_dy,
        )

    def get_drift_velocity(self) -> Tuple[float, float]:
        """Get average drift velocity (px/frame) from recent history."""
        if len(self._drift_history) < 2:
            return (0.0, 0.0)
        arr = np.array(self._drift_history)
        return (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))


# ============================================================
# 3. SelfSupervisedPSFEstimator  (inspired by SeReNet)
#    Physics-driven PSF estimation without paired training data.
# ============================================================

@dataclass
class PSFEstimatorConfig:
    """Configuration for self-supervised PSF estimation."""
    # Initial PSF sigma guess (pixels)
    initial_sigma: float = 2.0
    # Maximum iterations for PSF refinement
    max_iterations: int = 20
    # Convergence threshold (relative sigma change)
    convergence_threshold: float = 1e-3
    # Number of radial bins for PSF profile
    radial_bins: int = 32
    # Minimum number of spots needed for estimation
    min_spots: int = 5


@dataclass
class PSFEstimation:
    """Result of PSF estimation."""
    sigma_x: float
    sigma_y: float
    sigma_avg: float
    ellipticity: float  # 0=circular, 1=highly elliptical
    peak_to_background: float
    num_spots_used: int
    converged: bool
    iterations: int


class SelfSupervisedPSFEstimator:
    """Estimates PSF parameters from observed spot images.

    Inspired by SeReNet's physics-driven self-supervised approach,
    this module estimates PSF width, ellipticity, and peak-to-background
    ratio directly from observed spot data without requiring
    ground-truth training pairs.
    """

    def __init__(self, config: Optional[PSFEstimatorConfig] = None):
        self.config = config or PSFEstimatorConfig()
        self._spot_profiles: List[np.ndarray] = []
        self._current_sigma: float = self.config.initial_sigma

    def reset(self) -> None:
        self._spot_profiles.clear()
        self._current_sigma = self.config.initial_sigma

    def add_spot(self, patch: np.ndarray) -> None:
        """Add a spot patch to the estimation pool."""
        if patch.ndim != 2 or patch.size < 9:
            return
        self._spot_profiles.append(patch.astype(np.float64))
        # Keep pool bounded
        max_spots = self.config.min_spots * 10
        if len(self._spot_profiles) > max_spots:
            self._spot_profiles = self._spot_profiles[-max_spots:]

    def _fit_single_spot(self, patch: np.ndarray) -> Tuple[float, float, float]:
        """Fit 2D Gaussian to a single spot patch.

        Returns (sigma_x, sigma_y, peak_to_background).
        """
        h, w = patch.shape
        total = float(np.sum(patch))
        if total < 1e-6:
            return (self._current_sigma, self._current_sigma, 0.0)

        yy, xx = np.mgrid[:h, :w]
        cx = float(np.sum(xx * patch) / total)
        cy = float(np.sum(yy * patch) / total)

        dx = xx - cx
        dy = yy - cy
        sigma_x = math.sqrt(max(0.1, float(np.sum(dx ** 2 * patch) / total)))
        sigma_y = math.sqrt(max(0.1, float(np.sum(dy ** 2 * patch) / total)))

        # Peak to background
        center_val = float(patch[int(round(cy)), int(round(cx))])
        bg = float(np.median(patch))
        p2b = max(0.0, center_val - bg) / max(1.0, bg)

        return (sigma_x, sigma_y, p2b)

    def estimate(self) -> Optional[PSFEstimation]:
        """Estimate PSF from accumulated spots.

        Returns PSFEstimation or None if insufficient data.
        """
        if len(self._spot_profiles) < self.config.min_spots:
            return None

        prev_sigma = self._current_sigma
        converged = False

        for iteration in range(self.config.max_iterations):
            sigmas_x: List[float] = []
            sigmas_y: List[float] = []
            p2bs: List[float] = []

            for patch in self._spot_profiles:
                sx, sy, p2b = self._fit_single_spot(patch)
                # Reject outliers (more than 3x from current estimate)
                if (abs(sx - self._current_sigma) < 3 * self._current_sigma
                        and abs(sy - self._current_sigma) < 3 * self._current_sigma):
                    sigmas_x.append(sx)
                    sigmas_y.append(sy)
                    p2bs.append(p2b)

            if len(sigmas_x) < 3:
                break

            new_sigma = float(np.median(sigmas_x + sigmas_y))

            # Check convergence
            if abs(new_sigma - prev_sigma) < self.config.convergence_threshold * prev_sigma:
                converged = True
                self._current_sigma = new_sigma
                break

            prev_sigma = new_sigma
            self._current_sigma = new_sigma

        if not sigmas_x:
            return None

        avg_sx = float(np.median(sigmas_x))
        avg_sy = float(np.median(sigmas_y))
        avg_sigma = (avg_sx + avg_sy) / 2.0
        ellipticity = abs(avg_sx - avg_sy) / max(avg_sigma, 1e-6)
        avg_p2b = float(np.median(p2bs)) if p2bs else 0.0

        return PSFEstimation(
            sigma_x=avg_sx,
            sigma_y=avg_sy,
            sigma_avg=avg_sigma,
            ellipticity=min(1.0, ellipticity),
            peak_to_background=avg_p2b,
            num_spots_used=len(sigmas_x),
            converged=converged,
            iterations=iteration + 1,
        )


# ============================================================
# 4. ZernikeModeCorrector  (inspired by REALM)
#    Iterative Zernike-mode aberration correction.
# ============================================================

@dataclass
class ZernikeCorrectionConfig:
    """Configuration for Zernike-mode correction."""
    # Number of Zernike modes to correct (excluding piston)
    num_modes: int = 10
    # Maximum iterations per mode
    max_iterations: int = 15
    # Gain for each correction step (0-1)
    correction_gain: float = 0.3
    # Convergence threshold (relative metric improvement)
    convergence_threshold: float = 0.01
    # Image quality metric: "strehl", "entropy", "sharpness"
    quality_metric: str = "sharpness"


@dataclass
class ZernikeCorrectionStep:
    """Single step of Zernike correction."""
    mode_index: int
    coefficient_delta: float
    quality_before: float
    quality_after: float
    improved: bool


@dataclass
class ZernikeCorrectionResult:
    """Result of Zernike-mode correction."""
    coefficients: List[float]  # final Zernike coefficients
    total_iterations: int
    converged: bool
    quality_improvement: float  # relative improvement
    steps: List[ZernikeCorrectionStep]


class ZernikeModeCorrector:
    """Iterative Zernike-mode aberration correction.

    Inspired by REALM's sensorless AO approach, this module
    sequentially optimizes Zernike mode coefficients to maximize
    an image quality metric, without requiring a wavefront sensor.
    """

    def __init__(self, config: Optional[ZernikeCorrectionConfig] = None):
        self.config = config or ZernikeCorrectionConfig()
        self._coefficients: List[float] = [0.0] * self.config.num_modes
        self._best_quality: float = 0.0

    def reset(self) -> None:
        self._coefficients = [0.0] * self.config.num_modes
        self._best_quality = 0.0

    def _compute_quality(self, frame: np.ndarray) -> float:
        """Compute image quality metric."""
        if frame.size < 9:
            return 0.0
        patch = frame.astype(np.float64)

        if self.config.quality_metric == "strehl":
            # Approximate Strehl ratio: peak / (mean * N)
            peak = float(np.max(patch))
            mean_val = float(np.mean(patch))
            if mean_val < 1e-6:
                return 0.0
            return peak / (mean_val * patch.size) * 100.0

        elif self.config.quality_metric == "entropy":
            # Negative entropy (higher = sharper)
            hist, _ = np.histogram(patch.ravel(), bins=64, density=True)
            hist = hist[hist > 0]
            return -float(np.sum(hist * np.log(hist + 1e-10)))

        else:  # sharpness
            # Variance of Laplacian
            lap = cv2.Laplacian(patch, cv2.CV_64F)
            return float(np.var(lap))

    def step(self, frame: np.ndarray,
             apply_correction: Callable[[int, float], np.ndarray]
             ) -> ZernikeCorrectionResult:
        """Perform one full correction cycle over all modes.

        Args:
            frame: Current image frame.
            apply_correction: Function(mode_index, coefficient_delta) -> corrected_frame.

        Returns:
            ZernikeCorrectionResult with all correction steps.
        """
        steps: List[ZernikeCorrectionStep] = []
        base_quality = self._compute_quality(frame)
        self._best_quality = base_quality

        for mode_idx in range(self.config.num_modes):
            for iteration in range(self.config.max_iterations):
                # Try positive perturbation
                trial_coeff = self.config.correction_gain
                corrected = apply_correction(mode_idx, trial_coeff)
                quality_pos = self._compute_quality(corrected)

                # Try negative perturbation
                corrected_neg = apply_correction(mode_idx, -trial_coeff)
                quality_neg = self._compute_quality(corrected_neg)

                # Choose best direction
                if quality_pos >= quality_neg and quality_pos > base_quality:
                    delta = trial_coeff
                    new_quality = quality_pos
                elif quality_neg > base_quality:
                    delta = -trial_coeff
                    new_quality = quality_neg
                else:
                    delta = 0.0
                    new_quality = base_quality

                step = ZernikeCorrectionStep(
                    mode_index=mode_idx,
                    coefficient_delta=delta,
                    quality_before=base_quality,
                    quality_after=new_quality,
                    improved=new_quality > base_quality + 1e-6,
                )
                steps.append(step)

                self._coefficients[mode_idx] += delta
                base_quality = new_quality
                self._best_quality = max(self._best_quality, new_quality)

                # Check convergence for this mode
                if abs(delta) < self.config.convergence_threshold * max(
                    abs(self._coefficients[mode_idx]), 1e-6
                ):
                    break

        initial_quality = steps[0].quality_before if steps else 0.0
        improvement = (
            (self._best_quality - initial_quality)
            / max(abs(initial_quality), 1e-6)
        )

        return ZernikeCorrectionResult(
            coefficients=list(self._coefficients),
            total_iterations=len(steps),
            converged=improvement < self.config.convergence_threshold * 5,
            quality_improvement=improvement,
            steps=steps,
        )


# ============================================================
# 5. BayesianQualityGate  (inspired by BayesDL-SIM, Bayesian DPA-TISR)
#    Bayesian uncertainty-based quality gating for alignment decisions.
# ============================================================

@dataclass
class BayesianGateConfig:
    """Configuration for Bayesian quality gate."""
    # Prior belief in alignment quality (0-1)
    prior_quality: float = 0.5
    # Number of recent observations to use
    observation_window: int = 20
    # Uncertainty threshold to trigger "uncertain" state
    uncertainty_threshold: float = 0.3
    # Quality threshold to accept alignment
    quality_accept_threshold: float = 0.7
    # Quality threshold to reject alignment
    quality_reject_threshold: float = 0.3
    # False positive cost (penalizes accepting bad alignment)
    false_positive_cost: float = 3.0
    # False negative cost (penalizes rejecting good alignment)
    false_negative_cost: float = 1.0


@dataclass
class BayesianGateResult:
    """Result of Bayesian quality gate."""
    decision: str  # "accept", "reject", "uncertain"
    posterior_quality: float  # 0-1
    uncertainty: float  # 0-1
    expected_loss: float
    num_observations: int


class BayesianQualityGate:
    """Bayesian uncertainty-based quality gate for alignment decisions.

    Inspired by BayesDL-SIM and Bayesian DPA-TISR's uncertainty
    quantification, this module maintains a Bayesian belief over
    alignment quality and makes accept/reject/uncertain decisions
    that minimize expected loss.
    """

    def __init__(self, config: Optional[BayesianGateConfig] = None):
        self.config = config or BayesianGateConfig()
        self._alpha: float = self.config.prior_quality * 10.0  # Beta prior
        self._beta: float = (1.0 - self.config.prior_quality) * 10.0
        self._observations: Deque[float] = deque(
            maxlen=self.config.observation_window
        )

    def reset(self) -> None:
        self._alpha = self.config.prior_quality * 10.0
        self._beta = (1.0 - self.config.prior_quality) * 10.0
        self._observations.clear()

    def update(self, quality_observation: float) -> BayesianGateResult:
        """Update belief with a new quality observation and make decision.

        Args:
            quality_observation: Observed quality metric (0-1).

        Returns:
            BayesianGateResult with decision and statistics.
        """
        q = max(0.0, min(1.0, quality_observation))
        self._observations.append(q)

        # Update Beta posterior
        self._alpha += q
        self._beta += (1.0 - q)

        # Posterior mean and variance
        total = self._alpha + self._beta
        posterior = self._alpha / total
        variance = (self._alpha * self._beta) / (total ** 2 * (total + 1.0))
        uncertainty = math.sqrt(max(0.0, variance))

        # Expected loss for each decision
        # P(bad) = 1 - posterior
        p_bad = 1.0 - posterior
        loss_accept = p_bad * self.config.false_positive_cost
        loss_reject = posterior * self.config.false_negative_cost

        # Decision
        if uncertainty > self.config.uncertainty_threshold:
            decision = "uncertain"
        elif posterior >= self.config.quality_accept_threshold:
            decision = "accept"
        elif posterior <= self.config.quality_reject_threshold:
            decision = "reject"
        else:
            # Choose decision with lower expected loss
            decision = "accept" if loss_accept < loss_reject else "reject"

        expected_loss = min(loss_accept, loss_reject)

        return BayesianGateResult(
            decision=decision,
            posterior_quality=posterior,
            uncertainty=uncertainty,
            expected_loss=expected_loss,
            num_observations=len(self._observations),
        )


# ============================================================
# 6. ArtifactAwareAssessor  (inspired by SSR-SIM)
#    Self-supervised alignment quality assessment via artifact analysis.
# ============================================================

@dataclass
class ArtifactAssessorConfig:
    """Configuration for artifact-aware quality assessment."""
    # Radial bins for symmetry analysis
    radial_bins: int = 16
    # Angular bins for symmetry analysis
    angular_bins: int = 36
    # History length for temporal consistency check
    temporal_window: int = 30
    # Threshold for artifact flagging (deviation from ideal)
    artifact_threshold: float = 0.4


@dataclass
class ArtifactAssessment:
    """Result of artifact-aware quality assessment."""
    overall_quality: float  # 0-1
    radial_symmetry: float  # 0-1
    angular_symmetry: float  # 0-1
    temporal_consistency: float  # 0-1
    artifact_score: float  # 0-1, higher = more artifacts
    flags: List[str]  # list of detected artifact types


class ArtifactAwareAssessor:
    """Assesses alignment quality by analyzing artifact patterns.

    Inspired by SSR-SIM's artifact statistics analysis, this module
    detects alignment artifacts (asymmetry, ringing, temporal
    inconsistency) without requiring ground-truth reference images.
    """

    def __init__(self, config: Optional[ArtifactAssessorConfig] = None):
        self.config = config or ArtifactAssessorConfig()
        self._quality_history: Deque[float] = deque(
            maxlen=self.config.temporal_window
        )

    def reset(self) -> None:
        self._quality_history.clear()

    def _compute_radial_profile(self, patch: np.ndarray) -> np.ndarray:
        """Compute radial intensity profile from patch center."""
        h, w = patch.shape
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.mgrid[:h, :w]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float64)
        max_r = min(cy, cx)
        bins = np.linspace(0, max_r, self.config.radial_bins + 1)
        profile = np.zeros(self.config.radial_bins)
        for i in range(self.config.radial_bins):
            mask = (r >= bins[i]) & (r < bins[i + 1])
            if np.any(mask):
                profile[i] = float(np.mean(patch[mask]))
        return profile

    def _compute_angular_profile(self, patch: np.ndarray) -> np.ndarray:
        """Compute angular intensity profile from patch center."""
        h, w = patch.shape
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.mgrid[:h, :w]
        angles = np.arctan2(yy - cy, xx - cx)
        bins = np.linspace(-np.pi, np.pi, self.config.angular_bins + 1)
        profile = np.zeros(self.config.angular_bins)
        for i in range(self.config.angular_bins):
            mask = (angles >= bins[i]) & (angles < bins[i + 1])
            if np.any(mask):
                profile[i] = float(np.mean(patch[mask]))
        return profile

    def assess(self, patch: np.ndarray) -> ArtifactAssessment:
        """Assess alignment quality of a spot patch.

        Args:
            patch: Grayscale spot image patch.

        Returns:
            ArtifactAssessment with quality metrics and artifact flags.
        """
        flags: List[str] = []

        if patch.ndim != 2 or patch.size < 9:
            return ArtifactAssessment(
                overall_quality=0.0, radial_symmetry=0.0,
                angular_symmetry=0.0, temporal_consistency=0.0,
                artifact_score=1.0, flags=["invalid_patch"],
            )

        patch_f = patch.astype(np.float64)

        # Radial symmetry: how smooth the radial profile is
        radial = self._compute_radial_profile(patch_f)
        if radial.max() > 0:
            radial_norm = radial / radial.max()
            radial_smoothness = 1.0 - float(np.std(np.diff(radial_norm)))
        else:
            radial_smoothness = 0.0
        radial_symmetry = max(0.0, min(1.0, radial_smoothness))

        # Angular symmetry: how uniform the angular profile is
        angular = self._compute_angular_profile(patch_f)
        if angular.max() > 0:
            angular_norm = angular / angular.max()
            angular_uniformity = 1.0 - float(np.std(angular_norm))
        else:
            angular_uniformity = 0.0
        angular_symmetry = max(0.0, min(1.0, angular_uniformity))

        # Artifact detection
        artifact_score = 0.0

        # Check for ringing (oscillations in radial profile)
        if len(radial) > 4:
            diffs = np.diff(radial_norm)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            if sign_changes > len(radial) * 0.5:
                artifact_score += 0.3
                flags.append("ringing")

        # Check for asymmetry
        if angular_symmetry < 0.7:
            artifact_score += 0.3
            flags.append("asymmetry")

        # Check for saturation
        if np.any(patch_f >= 250):
            artifact_score += 0.2
            flags.append("saturation")

        # Check for low SNR
        median_bg = float(np.median(patch_f))
        std_bg = float(np.std(patch_f))
        if std_bg > 0:
            snr = (float(np.max(patch_f)) - median_bg) / std_bg
            if snr < 3.0:
                artifact_score += 0.2
                flags.append("low_snr")

        artifact_score = min(1.0, artifact_score)

        # Temporal consistency
        current_quality = 1.0 - artifact_score
        self._quality_history.append(current_quality)
        if len(self._quality_history) > 3:
            arr = np.array(self._quality_history)
            temporal_consistency = 1.0 - float(np.std(arr))
        else:
            temporal_consistency = 0.5

        # Overall quality
        overall = (
            0.3 * radial_symmetry
            + 0.3 * angular_symmetry
            + 0.2 * temporal_consistency
            + 0.2 * (1.0 - artifact_score)
        )

        return ArtifactAssessment(
            overall_quality=max(0.0, min(1.0, overall)),
            radial_symmetry=radial_symmetry,
            angular_symmetry=angular_symmetry,
            temporal_consistency=max(0.0, min(1.0, temporal_consistency)),
            artifact_score=artifact_score,
            flags=flags,
        )


# ============================================================
# 7. MultiFrameDenoiser  (inspired by LF-denoising)
#    Self-supervised denoising using multi-frame redundancy.
# ============================================================

@dataclass
class MultiFrameDenoiserConfig:
    """Configuration for multi-frame denoiser."""
    # Number of frames to accumulate
    accumulation_frames: int = 5
    # Temporal weighting: "uniform", "exponential", "quality_weighted"
    weighting_scheme: str = "exponential"
    # Decay factor for exponential weighting (newest frame = 1.0)
    decay_factor: float = 0.7
    # Minimum pixel value (prevent negative)
    min_pixel: float = 0.0
    # Maximum pixel value
    max_pixel: float = 255.0
    # Enable temporal outlier rejection
    outlier_rejection: bool = True
    # Outlier threshold (in standard deviations)
    outlier_threshold: float = 3.0


@dataclass
class DenoiseResult:
    """Result of multi-frame denoising."""
    denoised: np.ndarray
    # Denoised frame
    snr_improvement: float  # Estimated SNR improvement (dB)
    num_frames_used: int
    outliers_rejected: int


class MultiFrameDenoiser:
    """Self-supervised multi-frame denoiser.

    Inspired by LF-denoising's multi-angle redundancy approach,
    this module exploits temporal redundancy across consecutive
    frames to suppress noise without requiring clean reference data.
    """

    def __init__(self, config: Optional[MultiFrameDenoiserConfig] = None):
        self.config = config or MultiFrameDenoiserConfig()
        self._frame_buffer: Deque[np.ndarray] = deque(
            maxlen=self.config.accumulation_frames
        )
        self._prev_denoised: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._frame_buffer.clear()
        self._prev_denoised = None

    def _compute_weights(self, n: int) -> np.ndarray:
        """Compute temporal weights for frame accumulation."""
        if self.config.weighting_scheme == "uniform":
            return np.ones(n) / n
        elif self.config.weighting_scheme == "exponential":
            weights = np.array([
                self.config.decay_factor ** (n - 1 - i) for i in range(n)
            ])
            return weights / weights.sum()
        else:  # quality_weighted
            return np.ones(n) / n

    def denoise(self, frame: np.ndarray) -> DenoiseResult:
        """Denoise a frame using temporal accumulation.

        Args:
            frame: Grayscale image (2D numpy array).

        Returns:
            DenoiseResult with denoised frame and statistics.
        """
        self._frame_buffer.append(frame.copy())
        n = len(self._frame_buffer)

        if n < 2:
            return DenoiseResult(
                denoised=frame.copy(),
                snr_improvement=0.0,
                num_frames_used=1,
                outliers_rejected=0,
            )

        # Stack frames
        frames = np.array(list(self._frame_buffer), dtype=np.float64)
        weights = self._compute_weights(n)

        # Weighted mean
        weighted_sum = np.zeros_like(frames[0], dtype=np.float64)
        for i, (f, w) in enumerate(zip(frames, weights)):
            weighted_sum += f * w

        outliers_rejected = 0

        # Temporal outlier rejection
        if self.config.outlier_rejection and n >= 3:
            mean_frame = weighted_sum.copy()
            std_frame = np.std(frames, axis=0)
            # Avoid division by zero
            std_frame[std_frame < 1e-6] = 1e-6

            for i in range(n):
                deviation = np.abs(frames[i] - mean_frame) / std_frame
                outlier_mask = deviation > self.config.outlier_threshold
                outliers_rejected += int(np.sum(outlier_mask))
                # Replace outliers with weighted mean
                frames[i][outlier_mask] = mean_frame[outlier_mask]

            # Recompute weighted mean after outlier rejection
            weighted_sum = np.zeros_like(frames[0], dtype=np.float64)
            for f, w in zip(frames, weights):
                weighted_sum += f * w

        denoised = np.clip(
            weighted_sum, self.config.min_pixel, self.config.max_pixel
        ).astype(frame.dtype)

        # Estimate SNR improvement: sqrt(N) for independent frames
        snr_improvement = 10.0 * math.log10(math.sqrt(n))

        self._prev_denoised = denoised

        return DenoiseResult(
            denoised=denoised,
            snr_improvement=snr_improvement,
            num_frames_used=n,
            outliers_rejected=outliers_rejected,
        )
