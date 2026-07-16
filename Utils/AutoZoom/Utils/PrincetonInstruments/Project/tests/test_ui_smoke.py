"""UI 冒烟测试。"""

import pytest


def test_import_ui():
    """测试 UI 模块可导入。"""
    from pi_spectrometer.ui.main_window import MainWindow
    assert MainWindow is not None


@pytest.mark.skipif(
    True,
    reason="UI 启动测试需要 QApplication，建议在本地手动运行 python -m pi_spectrometer.ui.main_window",
)
def test_main_window_show(qtbot):
    from pi_spectrometer.ui.main_window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.isVisible()
    window.close()
