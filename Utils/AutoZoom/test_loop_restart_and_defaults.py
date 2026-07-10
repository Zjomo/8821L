"""
测试 AutoZoom Focus 的循环结束后重启能力及默认参数。

覆盖：
  1. 默认循环次数为 30，默认补焦间隔为 1.5s
  2. 启动闭环自然结束后，UI 控件恢复为可再次启动/建立参考
  3. 可连续启动两次闭环，第二次能正常结束
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np

# 将 AutoZoom 根目录加入路径
AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from autofocus_qt_ui.qt_compat import QApplication
from autofocus_qt_ui.config import DEFAULT_CYCLES, DEFAULT_INTERVAL_S
from autofocus_qt_ui.app import AutofocusMainWindow
from Focus.config import AutofocusConfig
from Focus.controller import AutofocusController
from Focus.metrics import FocusMetricsCalculator
from Focus.scorer import FocusScorer
from Focus.z_axis import ZAxisController

app = QApplication(sys.argv)


def _make_cfg() -> AutofocusConfig:
    return AutofocusConfig(
        capture_mode="screen_region",
        capture_area=(0, 0, 100, 100),
        focus_roi=(0, 0, 100, 100),
        z_enabled=False,
        autofocus_enabled=False,
    )


def _make_controller(cfg: AutofocusConfig) -> AutofocusController:
    metrics = FocusMetricsCalculator(cfg)
    scorer = FocusScorer(cfg, metrics)
    # 构造一个假的参考基线，使 score_ratio 能返回稳定值
    scorer.focus_reference = {
        "roi_metric_ref": {
            "highfreq_ratio": 0.5,
            "tenengrad": 100.0,
            "brenner": 10.0,
            "red_blue_ratio": 0.5,
            "modified_laplacian": 50.0,
            "dct_energy": 0.5,
            "smd": 50.0,
            "entropy": 5.0,
        }
    }
    scorer.focus_reference_ready = True
    z_axis = ZAxisController(cfg)
    return AutofocusController(cfg, scorer, metrics, z_axis)


def _fake_capture_live(self_score: float = 1.0):
    """返回固定分数的假 capture_live。"""
    def _capture() -> Dict[str, Any]:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        return {
            "ok": True,
            "full_rgb": image,
            "roi_rgb": image,
            "full": {},
            "roi_metrics": {
                "highfreq_ratio": 0.5 * self_score,
                "tenengrad": 100.0 * self_score,
                "brenner": 10.0 * self_score,
                "red_blue_ratio": 0.5 * self_score,
                "modified_laplacian": 50.0 * self_score,
                "dct_energy": 0.5 * self_score,
                "smd": 50.0 * self_score,
                "entropy": 5.0 * self_score,
            },
        }
    return _capture


def test_default_cycles_and_interval() -> None:
    """确认 UI 默认循环次数为 30，间隔为 1.5s。"""
    assert DEFAULT_CYCLES == 30, f"期望 DEFAULT_CYCLES=30，实际 {DEFAULT_CYCLES}"
    assert DEFAULT_INTERVAL_S == 1.5, f"期望 DEFAULT_INTERVAL_S=1.5，实际 {DEFAULT_INTERVAL_S}"

    window = AutofocusMainWindow()
    assert window.cycles_spin.value() == 30, f"UI 循环轮数默认值错误：{window.cycles_spin.value()}"
    assert abs(window.interval_spin.value() - 1.5) < 1e-9, (
        f"UI 间隔默认值错误：{window.interval_spin.value()}"
    )
    print("PASS: default cycles=30 and interval=1.5s")


def _wait_loop_finished(window: AutofocusMainWindow, timeout: float = 10.0) -> None:
    """等待闭环自然结束。"""
    deadline = time.time() + timeout
    while window._thread is not None and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.05)


def test_loop_finished_allows_restart() -> None:
    """闭环自然结束后，UI 状态应允许再次启动或建立参考。"""
    window = AutofocusMainWindow()
    try:
        cfg = _make_cfg()
        controller = _make_controller(cfg)
        controller.metrics_calc.capture_live = _fake_capture_live(1.0)

        runtime_args = {"output": "focus_output", "cycles": 1, "interval": 0.0}
        window._thread = window._build_and_start_thread(controller, cfg, runtime_args)

        _wait_loop_finished(window)

        assert window._thread is None, "循环结束后线程引用未释放"
        assert window.start_btn.isEnabled(), "循环结束后启动按钮应可用"
        assert window.build_ref_btn.isEnabled(), "循环结束后建立参考按钮应可用"
        assert not window.stop_btn.isEnabled(), "循环结束后停止按钮应禁用"
        assert "循环结束" in window.log_edit.toPlainText(), "日志应包含循环结束提示"
        print("PASS: loop finished allows restart")
    finally:
        window.close()
        QApplication.processEvents()


def test_loop_can_run_twice() -> None:
    """连续启动两次闭环，第二次也能正常结束。"""
    window = AutofocusMainWindow()
    try:
        cfg = _make_cfg()
        controller = _make_controller(cfg)
        controller.metrics_calc.capture_live = _fake_capture_live(1.0)

        for run in range(2):
            runtime_args = {"output": "focus_output", "cycles": 1, "interval": 0.0}
            window._thread = window._build_and_start_thread(controller, cfg, runtime_args)

            _wait_loop_finished(window)

            assert window._thread is None, f"第 {run + 1} 次循环结束后线程引用未释放"
            assert window.start_btn.isEnabled(), f"第 {run + 1} 次循环结束后启动按钮应可用"

        print("PASS: loop can run twice")
    finally:
        window.close()
        QApplication.processEvents()


if __name__ == "__main__":
    test_default_cycles_and_interval()
    test_loop_finished_allows_restart()
    test_loop_can_run_twice()
    print("\nAll tests passed!")
    app.quit()
    QApplication.processEvents()
