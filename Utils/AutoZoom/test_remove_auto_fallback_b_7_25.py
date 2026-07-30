"""
验证 _call_initialize_abc_with_loaded_calibration 已不再自动生成默认 B 点兜底。
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))


def _import_7_25():
    """动态导入 7_25 主模块。"""
    module_name = "measurement_workflow_7_25"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_25.py"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_no_auto_fallback_b_when_missing() -> None:
    """标定包缺少 B 正点时，调用初始化后仍应为空。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.hardware_mode = "virtual"
        wf = MeasurementWorkflow(cfg)

        # 构造一个只包含 A、C 点的标定状态
        state = _module.CalibrationState()
        state.rule_ab_a_positive_points = [[100.0, 100.0]]
        state.rule_ab_a_negative_points = []
        state.global_c_positive_points = [[200.0, 200.0]]
        state.global_c_negative_points = []
        # B 点为空
        state.rule_ab_b_positive_points = []
        state.rule_ab_b_negative_points = []

        follower = MagicMock()
        follower.cfg = MagicMock()
        follower.abc_segmenter = MagicMock()
        follower.segmenter = MagicMock()

        with patch.object(wf, "_capture_current_rule_ab_frame", MagicMock()) as mock_capture:
            wf._call_initialize_abc_with_loaded_calibration(follower, state)

        assert state.rule_ab_b_positive_points == [], "B 正点仍应为空"
        mock_capture.assert_not_called()
        print("PASS: no_auto_fallback_b_when_missing")


def test_existing_b_points_preserved() -> None:
    """标定包已包含 B 点时，应保留原值。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.hardware_mode = "virtual"
        wf = MeasurementWorkflow(cfg)

        state = _module.CalibrationState()
        state.rule_ab_a_positive_points = [[100.0, 100.0]]
        state.rule_ab_b_positive_points = [[150.0, 120.0]]
        state.global_c_positive_points = [[200.0, 200.0]]

        follower = MagicMock()
        follower.cfg = MagicMock()
        follower.abc_segmenter = MagicMock()
        follower.segmenter = MagicMock()

        with patch.object(wf, "_capture_current_rule_ab_frame", MagicMock()) as mock_capture:
            wf._call_initialize_abc_with_loaded_calibration(follower, state)

        assert state.rule_ab_b_positive_points == [[150.0, 120.0]], "B 正点应保持不变"
        mock_capture.assert_not_called()
        print("PASS: existing_b_points_preserved")


def test_injected_b_points_empty_when_missing() -> None:
    """缺少 B 点时，注入 follower 的 B 点也应为空。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.hardware_mode = "virtual"
        wf = MeasurementWorkflow(cfg)

        state = _module.CalibrationState()
        state.rule_ab_a_positive_points = [[100.0, 100.0]]
        state.global_c_positive_points = [[200.0, 200.0]]
        state.rule_ab_b_positive_points = []

        follower = MagicMock()
        follower.cfg = MagicMock()
        follower.abc_segmenter = MagicMock()
        follower.segmenter = MagicMock()

        injected = {}

        def fake_setattr(obj, name, value):
            injected[name] = value

        with patch.object(wf, "_safe_setattr", fake_setattr):
            wf._call_initialize_abc_with_loaded_calibration(follower, state)

        points_dict = injected.get("preloaded_points", {})
        assert points_dict.get("B", {}).get("positive") == [], (
            f"注入的 B 点应为空：{points_dict.get('B', {})}"
        )
        print("PASS: injected_b_points_empty_when_missing")


def test_no_log_for_fallback_b() -> None:
    """缺少 B 点时，不应再记录自动生成默认 B 点的日志。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.hardware_mode = "virtual"
        wf = MeasurementWorkflow(cfg)

        state = _module.CalibrationState()
        state.rule_ab_a_positive_points = [[100.0, 100.0]]
        state.global_c_positive_points = [[200.0, 200.0]]
        state.rule_ab_b_positive_points = []

        follower = MagicMock()
        follower.cfg = MagicMock()
        follower.abc_segmenter = MagicMock()
        follower.segmenter = MagicMock()

        logs = []
        with patch.object(wf, "log", logs.append):
            wf._call_initialize_abc_with_loaded_calibration(follower, state)

        assert not any("自动生成默认 B 点" in msg for msg in logs), "不应再记录兜底 B 点日志"
        print("PASS: no_log_for_fallback_b")


if __name__ == "__main__":
    tests = [
        test_no_auto_fallback_b_when_missing,
        test_existing_b_points_preserved,
        test_injected_b_points_empty_when_missing,
        test_no_log_for_fallback_b,
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
