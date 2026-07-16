"""核心抽象层测试。"""

import numpy as np
import pytest

from pi_spectrometer.core.base import SpectrometerBackend
from pi_spectrometer.core.exceptions import NotConnectedError
from pi_spectrometer.core.types import ROI, SpectrometerResult
from pi_spectrometer.picam.demo import MockSpectrometerBackend


def test_roi_creation():
    roi = ROI(x=10, width=100, x_bin=2, y=5, height=50, y_bin=1)
    assert roi.x == 10
    assert roi.width == 100
    assert roi.x_bin == 2
    assert roi.y_bin == 1


def test_roi_binning_clamped():
    roi = ROI(x=0, width=10, x_bin=0, y=0, height=10, y_bin=0)
    assert roi.x_bin == 1
    assert roi.y_bin == 1


def test_spectrometer_result_to_dict():
    result = SpectrometerResult(
        ok=True,
        index=1,
        num_points=1024,
        raw_y=np.array([1.0, 2.0, 3.0]),
        wavelength=np.array([500.0, 510.0, 520.0]),
    )
    d = result.to_workflow_dict()
    assert d["ok"] is True
    assert d["num_points"] == 1024
    assert d["raw_y"] == [1.0, 2.0, 3.0]
    assert d["wavelength"] == [500.0, 510.0, 520.0]


def test_mock_backend_lifecycle():
    backend = MockSpectrometerBackend()
    assert not backend.is_connected()
    assert backend.connect()
    assert backend.is_connected()
    backend.disconnect()
    assert not backend.is_connected()


def test_mock_backend_set_parameters():
    backend = MockSpectrometerBackend()
    backend.connect()
    backend.set_exposure(0.5)
    assert backend.get_exposure() == pytest.approx(0.5)
    backend.set_sensor_temperature(-30.0)
    assert backend.get_sensor_temperature() == pytest.approx(-30.0)
    roi = ROI(x=10, width=512, y=0, height=128)
    backend.set_roi(roi)
    assert backend.get_roi().width == 512


def test_mock_backend_acquire():
    backend = MockSpectrometerBackend()
    backend.connect()
    result = backend.acquire()
    assert isinstance(result, SpectrometerResult)
    assert result.ok
    assert result.num_points == backend.num_points
    assert result.raw_y is not None
    assert result.wavelength is not None
    assert result.fit_peak is not None


def test_backend_is_abstract():
    with pytest.raises(TypeError):
        SpectrometerBackend()
