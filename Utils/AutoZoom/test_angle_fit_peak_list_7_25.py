"""
测试 UI 右侧“角度-拟合峰值列表”的序号 0 初始化点显示逻辑。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))


def _import_7_25():
    """动态导入 7_25 主模块。"""
    import importlib.util

    module_name = "measurement_workflow_7_25"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_25.py"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_append_plot_point_zero_recorded() -> None:
    """初始光谱采集（cycle_index=0）的角度与拟合峰值应被记录到 plot_points。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        wf = MeasurementWorkflow(cfg)

        ctx = {
            "save_angle_deg": 25.5,
            "fit_peak": 1234.56,
            "raw_values": [1, 2, 3],
            "raw_median_values": [1.5, 2.0, 2.5],
            "x_axis_values": [400, 500, 600],
        }

        wf._append_plot_point(cycle_index=0, context_snapshot=ctx)

        assert len(wf.plot_points) == 1
        point = wf.plot_points[0]
        assert point["cycle_index"] == 0
        assert point["angle_deg"] == 25.5
        assert point["fit_peak"] == 1234.56

        print("PASS: append_plot_point_zero_recorded")


def test_update_angle_fit_peak_plot_starts_at_zero() -> None:
    """右侧列表中，cycle_index=0 的初始化点应显示为序号 0。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow
    MeasurementWorkflowGUI = _module.MeasurementWorkflowGUI

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        wf = MeasurementWorkflow(cfg)

        wf.plot_points = [
            {"cycle_index": 0, "angle_deg": 10.0, "fit_peak": 100.0},
            {"cycle_index": 1, "angle_deg": 12.0, "fit_peak": 110.0},
        ]

        # 不初始化完整 GUI，仅构造空对象并设置必需属性
        with patch.object(MeasurementWorkflowGUI, "__init__", lambda self, root: None):
            gui = MeasurementWorkflowGUI(None)
            gui.angle_fit_tree = MagicMock()
            gui.angle_fit_list_status_var = MagicMock()
            gui.fig = None
            gui.plot_status_var = MagicMock()

            gui.update_angle_fit_peak_plot(wf)

        calls = gui.angle_fit_tree.insert.call_args_list
        # 过滤掉 padding 行（values 为 ("", "", "")）
        data_calls = [c for c in calls if c.kwargs.get("values") != ("", "", "")]

        assert len(data_calls) == 2, f"应有 2 条数据记录，实际 {len(data_calls)}"
        # 第一条记录序号应为 0
        assert data_calls[0].kwargs["values"][0] == 0
        assert data_calls[0].kwargs["values"][1] == "10.000000"
        assert data_calls[0].kwargs["values"][2] == "100.000000"
        # 第二条记录序号应为 1
        assert data_calls[1].kwargs["values"][0] == 1
        assert data_calls[1].kwargs["values"][1] == "12.000000"
        assert data_calls[1].kwargs["values"][2] == "110.000000"

        print("PASS: update_angle_fit_peak_plot_starts_at_zero")


def test_initial_append_plot_is_always_true_in_run_one_cycle() -> None:
    """
    run_one_cycle 中，无论 sub_loop_count 是否大于 0，
    初始光谱采集都应向角度-拟合峰值列表追加绘图点（序号 0）。
    """
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.sub_loop_iterations_per_cycle = 3  # 有子循环
        wf = MeasurementWorkflow(cfg)

        # 模拟 run_one_cycle 走到 _acquire_and_save_spectrum 前的状态
        wf._focus_roi_selected = True
        wf._focus_reference_ready = True
        wf.angle_module = MagicMock()
        wf.light = MagicMock()
        wf.laser_stage = MagicMock()

        # 直接调用 _append_plot_point 模拟初始光谱采集追加序号 0 点
        ctx = {
            "save_angle_deg": 30.0,
            "fit_peak": 200.0,
            "raw_values": [],
            "raw_median_values": [],
            "x_axis_values": [],
        }
        wf._append_plot_point(cycle_index=0, context_snapshot=ctx)

        assert len(wf.plot_points) == 1
        assert wf.plot_points[0]["cycle_index"] == 0
        assert wf.plot_points[0]["angle_deg"] == 30.0
        assert wf.plot_points[0]["fit_peak"] == 200.0

        print("PASS: initial_append_plot_is_always_true_in_run_one_cycle")


if __name__ == "__main__":
    tests = [
        test_append_plot_point_zero_recorded,
        test_update_angle_fit_peak_plot_starts_at_zero,
        test_initial_append_plot_is_always_true_in_run_one_cycle,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            failed += 1
            print(f"FAIL: {test.__name__}: {e}")

    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        sys.exit(1)
