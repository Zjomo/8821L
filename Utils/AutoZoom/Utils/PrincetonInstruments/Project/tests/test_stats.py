"""光谱统计模块测试。"""

import numpy as np
import pytest

from pi_spectrometer.processing import stats


def test_estimate_baseline():
    y = np.array([1.0, 2.0, 3.0, 100.0, 101.0])
    baseline = stats.estimate_baseline(y, percentile=10)
    assert baseline <= 3.0


def test_compute_centroid():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    centroid = stats.compute_centroid(x, y)
    assert pytest.approx(centroid, abs=1e-6) == 1.0


def test_compute_integral():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 2.0, 0.0])
    integral = stats.compute_integral(x, y)
    assert pytest.approx(integral, abs=1e-6) == 2.0


def test_find_fwhm_gaussian():
    # 理论 FWHM = 2 * sqrt(2 * ln 2) * sigma ≈ 2.355 * sigma
    sigma = 5.0
    x = np.linspace(-30, 30, 1000)
    y = np.exp(-0.5 * (x / sigma) ** 2)
    width, left, right = stats.find_fwhm(x, y)
    expected = 2.35482 * sigma
    assert width is not None
    assert pytest.approx(width, rel=0.05) == expected
    assert left < 0 < right


def test_compute_spectrum_stats_basic():
    x = np.linspace(500, 530, 300)
    y = np.exp(-0.5 * ((x - 515.0) / 2.0) ** 2)
    s = stats.compute_spectrum_stats(x, y)

    assert s.peak_index is not None
    assert pytest.approx(s.peak_x, abs=0.1) == 515.0
    assert pytest.approx(s.peak_y, abs=0.01) == 1.0
    assert s.fwhm_x is not None
    assert s.centroid_x is not None
    assert s.integral is not None
    assert s.snr is not None


def test_compute_spectrum_stats_multipeak():
    x = np.linspace(400, 600, 1000)
    y = (
        np.exp(-0.5 * ((x - 450.0) / 3.0) ** 2)
        + np.exp(-0.5 * ((x - 550.0) / 3.0) ** 2)
    )
    s = stats.compute_spectrum_stats(x, y, find_peaks=True, peak_distance=20)
    assert s.num_peaks == 2
    assert len(s.peaks) == 2


def test_compute_spectrum_stats_empty():
    s = stats.compute_spectrum_stats(np.array([]), np.array([]))
    assert s.peak_index is None


def test_compute_spectrum_stats_length_mismatch():
    with pytest.raises(ValueError):
        stats.compute_spectrum_stats(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
