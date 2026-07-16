"""UI 增强功能冒烟测试。"""

import numpy as np
import pytest

from pi_spectrometer.core.recipe import RecipeAction


def test_plot_widget_enhanced_methods():
    from pi_spectrometer.ui.plot_widget import PlotWidget

    plot = PlotWidget()
    x = np.linspace(500, 600, 100)
    y = np.exp(-0.5 * ((x - 550) / 10) ** 2)
    plot.update_plot(y, x, title="test")

    cursor = plot.add_cursor(x_pos=550)
    assert cursor is not None
    assert len(plot.get_cursor_positions()) == 1

    plot.annotate_peak(550, 1.0, "peak")
    plot.annotate_fwhm(540, 560, 0.5)
    plot.clear_peak_annotations()
    plot.clear_cursors()
    assert len(plot.get_cursor_positions()) == 0

    plot.add_history_trace(y, x)
    plot.clear_history()

    plot.set_log_mode(True)
    plot.set_linear_mode()

    x_out, y_out = plot.get_data()
    assert x_out is not None
    assert y_out is not None


def test_main_window_panels_exist():
    from pi_spectrometer.ui.main_window import MainWindow

    window = MainWindow()
    assert hasattr(window, "tabs")
    assert window.tabs.count() >= 4
    assert hasattr(window, "bg_capture_btn")
    assert hasattr(window, "stats_peak_label")
    assert hasattr(window, "recipe_list")
    assert hasattr(window, "autosave_check")
    window.close()


def test_main_window_bg_library_empty():
    from pi_spectrometer.ui.main_window import MainWindow

    window = MainWindow()
    assert len(window.bg_library) == 0
    assert window.current_dark_name is None
    window.close()


def test_main_window_recipe_add_remove():
    from pi_spectrometer.ui.main_window import MainWindow

    window = MainWindow()
    assert window.recipe_list.count() == 0

    window.on_recipe_add(RecipeAction.ACQUIRE)
    window.on_recipe_add(RecipeAction.WAIT)
    assert window.recipe_list.count() == 2

    window.recipe_list.setCurrentRow(0)
    window.on_recipe_remove()
    assert window.recipe_list.count() == 1

    window.close()


def test_main_window_autosave_template():
    from pi_spectrometer.ui.main_window import MainWindow

    window = MainWindow()
    window.on_autosave_template_changed("test_{index:03d}.csv")
    assert window.auto_save_template == "test_{index:03d}.csv"
    window.close()
