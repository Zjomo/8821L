"""
测试 7_16 完整循环测量的两项修复：
1. 补焦基准参考图为空时，自动弹出 ROI 选择窗口；
2. 每轮循环只向“角度-拟合峰值列表”追加一个点，避免循环次数翻倍。
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

_module_path = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_16.py"
_spec = importlib.util.spec_from_file_location(
    "measurement_workflow_7_16_focus_plot", str(_module_path)
)
_measurement_workflow = importlib.util.module_from_spec(_spec)
sys.modules["measurement_workflow_7_16_focus_plot"] = _measurement_workflow
_spec.loader.exec_module(_measurement_workflow)  # type: ignore[union-attr]

MeasurementConfig = _measurement_workflow.MeasurementConfig
MeasurementWorkflow = _measurement_workflow.MeasurementWorkflow


def _make_config(**kwargs: Any) -> MeasurementConfig:
    """构造 virtual 模式配置。"""
    defaults: Dict[str, Any] = {
        "hardware_mode": "virtual",
        "sub_loop_iterations_per_cycle": 1,
        "save_root": "measurement_output_test",
    }
    defaults.update(kwargs)
    return MeasurementConfig(**defaults)


def _make_workflow(cfg: MeasurementConfig) -> MeasurementWorkflow:
    return MeasurementWorkflow(cfg)


def _setup_focus_mocks(wf: MeasurementWorkflow) -> None:
    """为 capture_focus_reference 准备 Focus 组件 mock。"""
    wf._focus_metrics_calc = MagicMock()
    wf._focus_metrics_calc.cfg.focus_roi = (0, 0, 100, 100)
    wf._focus_scorer = MagicMock()
    wf._focus_scorer.build_reference_from_image.return_value = {
        "full_rgb": np.zeros((100, 100, 3), dtype=np.uint8),
        "roi_metric_ref": {},
    }


def test_capture_focus_reference_falls_back_to_roi_selection() -> None:
    """截图为空时应触发交互式 ROI 选择，选择成功后建立参考并弹窗。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(save_root=tmpdir)
        wf = _make_workflow(cfg)
        wf.output_root = Path(tmpdir)
        _setup_focus_mocks(wf)

        test_image = np.zeros((200, 200, 3), dtype=np.uint8)

        with patch.object(wf, "_ensure_focus_components") as mock_ensure, \
             patch.object(
                 wf, "_capture_current_focus_frame",
                 side_effect=[None, test_image],
             ) as mock_capture, \
             patch.object(
                 wf, "_select_focus_roi_interactively", return_value=True
             ) as mock_select, \
             patch("measurement_workflow_7_16_focus_plot.cv2.imshow") as mock_imshow, \
             patch("measurement_workflow_7_16_focus_plot.cv2.waitKey", return_value=-1):
            result = wf.capture_focus_reference(cycle_index=1)

        assert result is True, "选择 ROI 后应成功建立参考"
        mock_ensure.assert_called_once()
        mock_select.assert_called_once()
        assert mock_capture.call_count == 2, "应重试截图一次"
        mock_imshow.assert_called_once()
        args, _ = mock_imshow.call_args
        assert args[0] == "Focus Reference Baseline Image"
        print("PASS: capture_focus_reference_falls_back_to_roi_selection")


def test_capture_focus_reference_skips_when_roi_cancelled() -> None:
    """截图为空且用户取消 ROI 选择时，应跳过参考建立，不弹窗。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(save_root=tmpdir)
        wf = _make_workflow(cfg)
        wf.output_root = Path(tmpdir)
        _setup_focus_mocks(wf)

        with patch.object(wf, "_ensure_focus_components"), \
             patch.object(
                 wf, "_capture_current_focus_frame", return_value=None
             ), \
             patch.object(
                 wf, "_select_focus_roi_interactively", return_value=False
             ) as mock_select, \
             patch("measurement_workflow_7_16_focus_plot.cv2.imshow") as mock_imshow:
            result = wf.capture_focus_reference(cycle_index=1)

        assert result is False, "用户取消时应返回 False"
        mock_select.assert_called_once()
        mock_imshow.assert_not_called()
        print("PASS: capture_focus_reference_skips_when_roi_cancelled")


def _mock_run_one_cycle_dependencies(wf: MeasurementWorkflow, sub_loop_count: int) -> Dict[str, Any]:
    """对 run_one_cycle 做统一 mock，只关注 append_plot_point 参数。"""
    mocks: Dict[str, Any] = {}

    mocks["follower"] = MagicMock()
    wf._ensure_rule_ab_follower_with_startup_retry = MagicMock(return_value=mocks["follower"])
    wf.apply_runtime_rule_ab_params_to_follower = MagicMock()
    wf._ensure_rule_ab_feature_tracker_installed = MagicMock()

    mocks["angle_result"] = {
        "ok": True,
        "angle_deg": 30.0,
        "reason": "ok",
        "angle_source": "yolo_obb_bmask_longest_edge",
    }
    wf.detect_step7_yolo_obb_angle_once = MagicMock(return_value=mocks["angle_result"])

    def _capture_ref(cycle_index: int) -> bool:
        wf._focus_reference_ready = True
        return True

    mocks["capture_focus_reference"] = MagicMock(side_effect=_capture_ref)
    wf.capture_focus_reference = mocks["capture_focus_reference"]

    mocks["paths"] = {"spectrum_csv": "spectrum_test.csv", "cycle_dir": "cycle_test"}
    wf.build_save_path = MagicMock(return_value=mocks["paths"])

    captured_calls: List[Dict[str, Any]] = []

    def fake_acquire_and_save(*args: Any, **kwargs: Any) -> bool:
        captured_calls.append(dict(kwargs))
        return True

    wf._acquire_and_save_spectrum = MagicMock(side_effect=fake_acquire_and_save)

    wf.laser_on = MagicMock(side_effect=lambda: wf.context.update({"laser_on": True}))
    wf.laser_off = MagicMock(side_effect=lambda: wf.context.update({"laser_on": False}))

    wf.run_rule_ab_until_angle_delta = MagicMock(
        return_value={"ok": True, "final_delta": 3.5, "reason": "ok", "records": []}
    )
    wf.run_rule_ac_until_threshold = MagicMock(return_value={"ok": True, "reason": "ok"})
    wf.run_autofocus_if_needed = MagicMock(
        return_value={"score": 0.98, "triggered": False, "autofocus_ok": True}
    )
    wf._is_midrun_recalibration_requested = MagicMock(return_value=False)

    mocks["captured_calls"] = captured_calls
    return mocks


def test_run_one_cycle_appends_single_plot_point() -> None:
    """sub_loop_iterations_per_cycle>0 时，每轮循环只应有一个 append_plot_point=True。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(save_root=tmpdir, sub_loop_iterations_per_cycle=3)
        wf = _make_workflow(cfg)
        mocks = _mock_run_one_cycle_dependencies(wf, sub_loop_count=3)

        result = wf.run_one_cycle(cycle_index=1)
        assert result is True, f"run_one_cycle 应返回 True，实际 {result}"

        calls = mocks["captured_calls"]
        assert len(calls) == 1 + 3, f"应采集 4 次光谱，实际 {len(calls)}"

        true_count = sum(1 for c in calls if c.get("append_plot_point") is True)
        assert true_count == 1, f"每轮应只追加 1 个绘图点，实际 {true_count}"

        # 有子循环时，最后一个子循环才允许 append
        assert calls[-1]["append_plot_point"] is True, "最后一轮子循环应 append"
        assert all(
            c["append_plot_point"] is False for c in calls[:-1]
        ), "之前的采集不应 append"

        print("PASS: run_one_cycle_appends_single_plot_point")


def test_run_one_cycle_no_sub_loop_appends_initial_point() -> None:
    """sub_loop_iterations_per_cycle=0 时，初始光谱采集应追加绘图点。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(save_root=tmpdir, sub_loop_iterations_per_cycle=0)
        wf = _make_workflow(cfg)
        mocks = _mock_run_one_cycle_dependencies(wf, sub_loop_count=0)

        result = wf.run_one_cycle(cycle_index=1)
        assert result is True

        calls = mocks["captured_calls"]
        assert len(calls) == 1
        assert calls[0]["append_plot_point"] is True
        print("PASS: run_one_cycle_no_sub_loop_appends_initial_point")


def test_save_cycle_result_respects_append_flag() -> None:
    """save_cycle_result 的 append_plot_point=False 时不应向列表追加数据点。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(save_root=tmpdir)
        wf = _make_workflow(cfg)
        wf.output_root = Path(tmpdir)

        paths = {
            "spectrum_csv": str(Path(tmpdir) / "spectrum.csv"),
            "cycle_dir": str(Path(tmpdir) / "cycle"),
        }
        angle_result = {"ok": True, "angle_deg": 30.0}
        labview_result = {"num_points": 100, "fit_peak": 500.0}

        wf.context["x_axis_values"] = [1, 2, 3]
        wf.context["raw_values"] = [10, 20, 30]
        wf.context["raw_median_values"] = [12, 22, 32]
        wf.context["save_angle_deg"] = 30.0
        wf.context["fit_peak"] = 500.0

        wf.save_cycle_result(
            cycle_index=1,
            paths=paths,
            angle_before_result=angle_result,
            angle_after_result=angle_result,
            labview_result=labview_result,
            append_plot_point=False,
        )
        assert len(wf.plot_points) == 0, "append=False 时不应添加绘图点"

        wf.save_cycle_result(
            cycle_index=2,
            paths=paths,
            angle_before_result=angle_result,
            angle_after_result=angle_result,
            labview_result=labview_result,
            append_plot_point=True,
        )
        assert len(wf.plot_points) == 1, "append=True 时应添加一个绘图点"
        assert wf.plot_points[0]["cycle_index"] == 2

        print("PASS: save_cycle_result_respects_append_flag")


if __name__ == "__main__":
    tests = [
        test_capture_focus_reference_falls_back_to_roi_selection,
        test_capture_focus_reference_skips_when_roi_cancelled,
        test_run_one_cycle_appends_single_plot_point,
        test_run_one_cycle_no_sub_loop_appends_initial_point,
        test_save_cycle_result_respects_append_flag,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            failed += 1
            print(f"FAIL: {test.__name__}: {e}")
            traceback.print_exc()

    if failed == 0:
        print("\n所有测试通过")
    else:
        print(f"\n失败测试数：{failed}")
        sys.exit(1)
