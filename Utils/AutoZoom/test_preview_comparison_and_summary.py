"""
测试 AutoZoom Focus 的实时预览基准图对比与 2x2 闭环回顾弹窗。

覆盖：
  1. build_reference 保存基准 ROI 图像
  2. RoiPreviewLabel 支持基准图/实时图对比模式
  3. LoopSummaryDialog 从快照中正确挑选 4 张图像
  4. 主窗口在闭环期间收集快照、结束后弹出回顾弹窗
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from autofocus_qt_ui.qt_compat import QApplication
from autofocus_qt_ui.app import AutofocusMainWindow
from autofocus_qt_ui.widgets import RoiPreviewLabel, LoopSummaryDialog
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
    """返回固定分数和 ROI 图像的假 capture_live。"""
    def _capture() -> Dict[str, Any]:
        image = np.full((100, 100, 3), int(128 * self_score), dtype=np.uint8)
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


def test_reference_image_saved() -> None:
    """build_reference 应在 focus_reference 中保存 roi_rgb 和 full_rgb。"""
    cfg = _make_cfg()
    scorer = FocusScorer(cfg, FocusMetricsCalculator(cfg))

    fake_image = np.zeros((50, 50, 3), dtype=np.uint8)
    fake_live = {
        "ok": True,
        "full_rgb": fake_image,
        "roi_rgb": fake_image,
        "full": {},
        "roi_metrics": {
            "highfreq_ratio": 0.5,
            "tenengrad": 100.0,
            "brenner": 10.0,
            "red_blue_ratio": 0.5,
            "modified_laplacian": 50.0,
            "dct_energy": 0.5,
            "smd": 50.0,
            "entropy": 5.0,
        },
    }
    scorer.metrics_calc.capture_live = lambda: fake_live
    scorer.metrics_calc.capture_and_save = lambda *a, **k: fake_live

    ref = scorer.build_reference(capture_count=1)
    assert ref.get("roi_rgb") is not None, "基准 ROI 图像未保存"
    assert ref.get("full_rgb") is not None, "基准全图未保存"
    assert np.array_equal(ref["roi_rgb"], fake_image), "保存的 ROI 图像与采集不一致"
    print("PASS: reference image saved")


def test_preview_comparison_mode() -> None:
    """RoiPreviewLabel 在对比模式下应同时绘制基准图和实时图。"""
    label = RoiPreviewLabel()
    label.resize(400, 300)
    ref = np.full((80, 80, 3), [255, 0, 0], dtype=np.uint8)  # 红色基准
    live = np.full((80, 80, 3), [0, 255, 0], dtype=np.uint8)  # 绿色实时

    label.set_reference_image(ref)
    label.set_preview_image(live)
    label.set_comparison_mode(True)

    assert label._comparison_mode is True
    assert label._reference_image is not None
    assert label._original_image is not None
    assert label._reference_pixmap is not None
    assert label._pixmap is not None

    # 退出对比模式后只保留实时图
    label.set_comparison_mode(False)
    assert label._comparison_mode is False
    print("PASS: preview comparison mode")


def test_loop_summary_dialog_selects_four_images() -> None:
    """LoopSummaryDialog 应始终返回 4 张图像。"""
    reference = np.zeros((60, 60, 3), dtype=np.uint8)
    snapshots = [
        (1, np.full((60, 60, 3), 10, dtype=np.uint8)),
        (2, np.full((60, 60, 3), 20, dtype=np.uint8)),
        (3, np.full((60, 60, 3), 30, dtype=np.uint8)),
        (4, np.full((60, 60, 3), 40, dtype=np.uint8)),
        (5, np.full((60, 60, 3), 50, dtype=np.uint8)),
        (6, np.full((60, 60, 3), 60, dtype=np.uint8)),
    ]
    dialog = LoopSummaryDialog(snapshots=snapshots, reference_image=reference)
    images = dialog._select_images(snapshots, reference)
    assert len(images) == 4, f"应返回 4 张图像，实际 {len(images)}"
    assert images[0][2] == "基准图", "第一张应为基准图"
    assert images[-1][0] == 6, "最后一张应为最后一轮"

    # 无基准图时也应返回 4 张
    dialog2 = LoopSummaryDialog(snapshots=snapshots, reference_image=None)
    images2 = dialog2._select_images(snapshots, None)
    assert len(images2) == 4, f"无基准图时也应返回 4 张，实际 {len(images2)}"
    print("PASS: loop summary dialog selects four images")


def test_window_collects_snapshots_and_shows_summary() -> None:
    """闭环运行期间主窗口应收集快照，结束后应创建回顾弹窗。"""
    window = AutofocusMainWindow()
    try:
        cfg = _make_cfg()
        controller = _make_controller(cfg)
        controller.metrics_calc.capture_live = _fake_capture_live(1.0)

        # 模拟已建立参考
        window._reference_image = np.zeros((60, 60, 3), dtype=np.uint8)

        runtime_args = {"output": "focus_output", "cycles": 2, "interval": 0.0}
        window._thread = window._build_and_start_thread(controller, cfg, runtime_args)

        deadline = time.time() + 10.0
        while window._thread is not None and time.time() < deadline:
            QApplication.processEvents()
            time.sleep(0.05)

        assert window._thread is None, "闭环结束后线程引用未释放"
        assert len(window._loop_snapshots) >= 2, (
            f"应至少收集 2 轮快照，实际 {len(window._loop_snapshots)}"
        )

        # 手动触发弹窗（通常由 QTimer 异步触发）
        window._show_loop_summary_dialog()
        QApplication.processEvents()
        assert window._summary_dialog is not None, "回顾弹窗未创建"
        assert window._summary_dialog.isVisible(), "回顾弹窗未显示"
        window._summary_dialog.close()

        # 退出对比模式
        assert window.preview_label._comparison_mode is False
        print("PASS: window collects snapshots and shows summary")
    finally:
        if window._summary_dialog is not None:
            try:
                window._summary_dialog.close()
            except Exception:
                pass
        window.close()
        QApplication.processEvents()


if __name__ == "__main__":
    test_reference_image_saved()
    test_preview_comparison_mode()
    test_loop_summary_dialog_selects_four_images()
    test_window_collects_snapshots_and_shows_summary()
    print("\nAll tests passed!")
    app.quit()
    QApplication.processEvents()
