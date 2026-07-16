"""背景/暗场校正模块测试。"""

import numpy as np
import pytest

from pi_spectrometer.core.types import SpectrometerResult
from pi_spectrometer.processing.background import (
    BackgroundFrame,
    BackgroundFrameLibrary,
    apply_background_correction,
    correct_spectrometer_result,
)


def test_background_frame_creation():
    y = np.array([1.0, 2.0, 3.0])
    frame = BackgroundFrame(name="dark1", y=y, kind="dark")
    assert frame.name == "dark1"
    assert frame.kind == "dark"
    assert np.allclose(frame.y, y)


def test_background_frame_invalid_dimension():
    with pytest.raises(ValueError):
        BackgroundFrame(name="bad", y=np.array([[1.0, 2.0]]))


def test_library_add_remove_list():
    lib = BackgroundFrameLibrary()
    lib.add(BackgroundFrame(name="a", y=np.array([1.0, 2.0])))
    lib.add(BackgroundFrame(name="b", y=np.array([3.0, 4.0]), kind="reference"))
    assert len(lib) == 2
    assert "a" in lib.list()
    assert lib.list(kind="reference") == ["b"]

    assert lib.remove("a")
    assert not lib.remove("not_exist")
    assert len(lib) == 1


def test_dark_subtraction():
    y = np.array([10.0, 20.0, 30.0])
    dark = np.array([1.0, 2.0, 3.0])
    corrected = apply_background_correction(y, dark=dark)
    assert np.allclose(corrected, [9.0, 18.0, 27.0])


def test_reference_normalization():
    y = np.array([10.0, 20.0, 30.0])
    dark = np.array([1.0, 1.0, 1.0])
    reference = np.array([5.0, 10.0, 15.0])
    corrected = apply_background_correction(y, dark=dark, reference=reference)
    expected = (y - dark) / (reference - dark)
    assert np.allclose(corrected, expected)


def test_negative_clipping():
    y = np.array([1.0, 2.0, 3.0])
    dark = np.array([5.0, 5.0, 5.0])
    corrected = apply_background_correction(y, dark=dark, clip_negative=True)
    assert np.all(corrected == 0.0)

    corrected_no_clip = apply_background_correction(
        y, dark=dark, clip_negative=False
    )
    assert np.allclose(corrected_no_clip, [-4.0, -3.0, -2.0])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        apply_background_correction(
            np.array([1.0, 2.0]), dark=np.array([1.0, 2.0, 3.0])
        )


def test_correct_spectrometer_result():
    y = np.array([10.0, 20.0, 30.0])
    dark = np.array([1.0, 2.0, 3.0])
    lib = BackgroundFrameLibrary()
    lib.add(BackgroundFrame(name="dark1", y=dark, kind="dark"))

    result = SpectrometerResult(raw_y=y, num_points=len(y))
    corrected = correct_spectrometer_result(result, library=lib, dark_name="dark1")

    assert np.allclose(corrected.raw_y, [9.0, 18.0, 27.0])
    assert corrected.metadata["correction"]["dark"] == "dark1"
    assert "raw_y_uncorrected" in corrected.metadata
