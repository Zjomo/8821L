"""数据处理测试。"""

import numpy as np
import pytest

from pi_spectrometer.processing import filters, peaks


def test_remove_above_threshold():
    values = [100.0, 200.0, 3500.0, 300.0]
    filtered = filters.remove_above_threshold(values, 3000.0)
    assert filtered == [100.0, 200.0, 0.0, 300.0]


def test_remove_above_threshold_disabled():
    values = [100.0, 200.0, 3500.0]
    filtered = filters.remove_above_threshold(values, 0)
    assert filtered == values


def test_median_filter_1d():
    values = [1.0, 2.0, 100.0, 3.0, 4.0]
    filtered = filters.median_filter_1d(values, 3)
    assert len(filtered) == len(values)
    assert filtered[2] == 3.0


def test_median_filter_window_one():
    values = [1.0, 2.0, 3.0]
    assert filters.median_filter_1d(values, 1) == values


def test_find_peak(sample_spectrum):
    x, y = sample_spectrum
    fit_peak, params = peaks.find_peak(y, x)
    assert fit_peak is not None
    assert params is not None
    assert "center" in params
    assert "sigma" in params
    assert 500 < params["center"] < 530


def test_find_peak_empty():
    fit_peak, params = peaks.find_peak(np.array([]))
    assert fit_peak is None
    assert params is None
