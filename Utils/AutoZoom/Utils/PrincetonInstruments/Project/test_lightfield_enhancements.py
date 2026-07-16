"""LightField UI 增强功能回归测试（无需 pytest）。

当 pytest 不可用时，可直接运行：
    python test_lightfield_enhancements.py
"""

from __future__ import annotations

import math
import sys
import traceback

import numpy as np

PROJECT_ROOT = __file__.rsplit("\\", 1)[0]
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pi_spectrometer.core.recipe import (
    AcquisitionRecipe,
    RecipeAction,
    RecipeRunner,
    RecipeStep,
)
from pi_spectrometer.core.types import SpectrometerResult
from pi_spectrometer.picam.demo import MockSpectrometerBackend
from pi_spectrometer.processing.background import (
    BackgroundFrame,
    BackgroundFrameLibrary,
    apply_background_correction,
    correct_spectrometer_result,
)
from pi_spectrometer.processing import stats


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures: list = []

    def add_pass(self):
        self.passed += 1

    def add_fail(self, name: str, exc: Exception):
        self.failed += 1
        self.failures.append((name, exc))


# 提前创建 QApplication，供 UI 测试使用
def _init_qt():
    try:
        from pi_spectrometer.ui.qt_compat import QtWidgets

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        return app
    except Exception as e:
        print(f"Qt 初始化失败: {e}")
        return None


_QT_APP = _init_qt()


def run_case(result: TestResult, name: str, func):
    try:
        func()
        print(f"PASS: {name}")
        result.add_pass()
    except Exception as e:
        print(f"FAIL: {name}")
        traceback.print_exc()
        result.add_fail(name, e)


def assert_close(a, b, rel=1e-6, abs_tol=1e-9):
    if not math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol):
        raise AssertionError(f"{a} !~ {b}")


# ---------------------------------------------------------------------------
# background.py tests
# ---------------------------------------------------------------------------
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


def test_background_library():
    lib = BackgroundFrameLibrary()
    lib.add(BackgroundFrame(name="dark1", y=np.array([1.0, 2.0]), kind="dark"))
    lib.add(BackgroundFrame(name="ref1", y=np.array([3.0, 4.0]), kind="reference"))
    assert len(lib) == 2
    assert lib.list(kind="dark") == ["dark1"]
    assert lib.remove("dark1")
    assert len(lib) == 1


def test_correct_spectrometer_result():
    y = np.array([10.0, 20.0, 30.0])
    dark = np.array([1.0, 2.0, 3.0])
    lib = BackgroundFrameLibrary()
    lib.add(BackgroundFrame(name="dark1", y=dark, kind="dark"))
    result = SpectrometerResult(raw_y=y, num_points=len(y))
    corrected = correct_spectrometer_result(result, library=lib, dark_name="dark1")
    assert np.allclose(corrected.raw_y, [9.0, 18.0, 27.0])
    assert corrected.metadata["correction"]["dark"] == "dark1"


# ---------------------------------------------------------------------------
# stats.py tests
# ---------------------------------------------------------------------------
def test_centroid():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    assert_close(stats.compute_centroid(x, y), 1.0)


def test_integral():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 2.0, 0.0])
    assert_close(stats.compute_integral(x, y), 2.0)


def test_fwhm_gaussian():
    sigma = 5.0
    x = np.linspace(-30, 30, 1000)
    y = np.exp(-0.5 * (x / sigma) ** 2)
    width, left, right = stats.find_fwhm(x, y)
    assert width is not None
    assert_close(width, 2.35482 * sigma, rel=0.05)
    assert left < 0 < right


def test_spectrum_stats():
    x = np.linspace(500, 530, 300)
    y = np.exp(-0.5 * ((x - 515.0) / 2.0) ** 2)
    s = stats.compute_spectrum_stats(x, y)
    assert_close(s.peak_x, 515.0, abs_tol=0.1)
    assert_close(s.peak_y, 1.0, abs_tol=0.01)
    assert s.fwhm_x is not None
    assert s.centroid_x is not None


def test_multipeak_detection():
    x = np.linspace(400, 600, 1000)
    y = (
        np.exp(-0.5 * ((x - 450.0) / 3.0) ** 2)
        + np.exp(-0.5 * ((x - 550.0) / 3.0) ** 2)
    )
    s = stats.compute_spectrum_stats(x, y, find_peaks=True, peak_distance=20)
    assert s.num_peaks == 2


# ---------------------------------------------------------------------------
# recipe.py tests
# ---------------------------------------------------------------------------
def test_recipe_serialization():
    recipe = AcquisitionRecipe(name="test")
    recipe.add_step(RecipeStep(action=RecipeAction.ACQUIRE, repeats=2))
    recipe.add_step(RecipeStep(action=RecipeAction.WAIT, params={"seconds": 0.1}))
    data = recipe.to_dict()
    restored = AcquisitionRecipe.from_dict(data)
    assert restored.name == "test"
    assert len(restored.steps) == 2
    assert restored.steps[0].repeats == 2


def test_recipe_runner_acquire():
    backend = MockSpectrometerBackend()
    backend.connect()
    step = RecipeStep(action=RecipeAction.ACQUIRE)
    runner = RecipeRunner(backend=backend, steps=[step])
    results = runner.run()
    assert len(results) == 1
    assert results[0].ok


def test_recipe_runner_repeat():
    backend = MockSpectrometerBackend()
    backend.connect()
    step = RecipeStep(action=RecipeAction.ACQUIRE, repeats=3)
    runner = RecipeRunner(backend=backend, steps=[step])
    results = runner.run()
    assert len(results) == 3


def test_recipe_runner_stop():
    backend = MockSpectrometerBackend()
    backend.connect()
    step = RecipeStep(action=RecipeAction.ACQUIRE, repeats=100)
    runner = RecipeRunner(backend=backend, steps=[step])

    def stop_after_first(*args, **kwargs):
        runner.stop()

    runner.on_step_done = stop_after_first
    results = runner.run()
    assert len(results) <= 2
    assert not runner.is_running()


# ---------------------------------------------------------------------------
# UI smoke tests
# ---------------------------------------------------------------------------
def test_plot_widget_enhanced():
    from pi_spectrometer.ui.plot_widget import PlotWidget

    plot = PlotWidget()
    x = np.linspace(500, 600, 100)
    y = np.exp(-0.5 * ((x - 550) / 10) ** 2)
    plot.update_plot(y, x, title="test")
    cursor = plot.add_cursor(x_pos=550)
    assert cursor is not None
    assert len(plot.get_cursor_positions()) == 1
    plot.clear_cursors()
    assert len(plot.get_cursor_positions()) == 0


def test_main_window_panels():
    from pi_spectrometer.ui.main_window import MainWindow

    window = MainWindow()
    assert hasattr(window, "tabs")
    assert window.tabs.count() >= 4
    assert hasattr(window, "recipe_list")
    window.close()


def test_main_window_recipe_add_remove():
    from pi_spectrometer.ui.main_window import MainWindow

    window = MainWindow()
    window.on_recipe_add(RecipeAction.ACQUIRE)
    window.on_recipe_add(RecipeAction.WAIT)
    assert window.recipe_list.count() == 2
    window.recipe_list.setCurrentRow(0)
    window.on_recipe_remove()
    assert window.recipe_list.count() == 1
    window.close()


def main():
    result = TestResult()
    tests = [
        ("dark subtraction", test_dark_subtraction),
        ("reference normalization", test_reference_normalization),
        ("background library", test_background_library),
        ("correct spectrometer result", test_correct_spectrometer_result),
        ("centroid", test_centroid),
        ("integral", test_integral),
        ("fwhm gaussian", test_fwhm_gaussian),
        ("spectrum stats", test_spectrum_stats),
        ("multipeak detection", test_multipeak_detection),
        ("recipe serialization", test_recipe_serialization),
        ("recipe runner acquire", test_recipe_runner_acquire),
        ("recipe runner repeat", test_recipe_runner_repeat),
        ("recipe runner stop", test_recipe_runner_stop),
        ("plot widget enhanced", test_plot_widget_enhanced),
        ("main window panels", test_main_window_panels),
        ("main window recipe add/remove", test_main_window_recipe_add_remove),
    ]

    for name, func in tests:
        run_case(result, name, func)

    print(f"\n结果: {result.passed} 通过, {result.failed} 失败")
    if result.failures:
        print("\n失败项:")
        for name, exc in result.failures:
            print(f"  - {name}: {exc}")
        sys.exit(1)
    print("所有测试通过!")


if __name__ == "__main__":
    main()
