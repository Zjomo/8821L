"""
验证手动停止测量时正确关闭激光的功能。

覆盖：
  1. request_stop 在激光开启时应调用 laser_off
  2. request_stop 在激光已关闭时不应重复调用 laser_off
  3. laser_off 失败时应有日志记录，不阻塞流程
  4. finish 方法正确关闭激光
  5. close_all 方法正确关闭激光
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


def test_request_stop_turns_off_laser_when_on() -> None:
    """激光开启状态下，request_stop 应调用 laser_off 并更新状态。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.hardware_mode = "virtual"
        wf = MeasurementWorkflow(cfg)

        wf.context["laser_on"] = True
        wf.is_measuring = True

        laser_off_called = []

        def fake_laser_off():
            laser_off_called.append(True)
            wf.context["laser_on"] = False

        with patch.object(wf, "laser_off", fake_laser_off):
            with patch.object(wf, "save_summary_xlsx", MagicMock()):
                wf.request_stop()

        assert wf.stop_requested == True
        assert wf.is_measuring == False
        assert laser_off_called == [True], "laser_off 应被调用一次"
        assert wf.context.get("laser_on") == False

        print("PASS: request_stop_turns_off_laser_when_on")


def test_request_stop_does_not_call_laser_off_when_already_off() -> None:
    """激光已关闭状态下，request_stop 不应重复调用 laser_off。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.hardware_mode = "virtual"
        wf = MeasurementWorkflow(cfg)

        wf.context["laser_on"] = False
        wf.is_measuring = True

        laser_off_called = []

        def fake_laser_off():
            laser_off_called.append(True)

        with patch.object(wf, "laser_off", fake_laser_off):
            with patch.object(wf, "save_summary_xlsx", MagicMock()):
                wf.request_stop()

        assert wf.stop_requested == True
        assert laser_off_called == [], "laser_off 不应被调用"
        assert wf.context.get("laser_on") == False

        print("PASS: request_stop_does_not_call_laser_off_when_already_off")


def test_request_stop_handles_laser_off_failure() -> None:
    """laser_off 失败时应有日志记录，不阻塞 request_stop 流程。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.hardware_mode = "virtual"
        wf = MeasurementWorkflow(cfg)

        wf.context["laser_on"] = True
        wf.is_measuring = True

        def fake_laser_off():
            raise RuntimeError("激光关闭失败")

        with patch.object(wf, "laser_off", fake_laser_off):
            with patch.object(wf, "save_summary_xlsx", MagicMock()):
                wf.request_stop()

        assert wf.stop_requested == True
        assert wf.is_measuring == False
        assert wf.context.get("laser_on") == True

        print("PASS: request_stop_handles_laser_off_failure")


def test_finish_turns_off_laser() -> None:
    """finish 方法应在测量结束时关闭激光。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.hardware_mode = "virtual"
        wf = MeasurementWorkflow(cfg)

        wf.context["laser_on"] = True
        wf.is_measuring = True

        laser_off_called = []

        def fake_laser_off():
            laser_off_called.append(True)
            wf.context["laser_on"] = False

        with patch.object(wf, "laser_off", fake_laser_off):
            with patch.object(wf, "save_summary_xlsx", MagicMock()):
                wf.finish()

        assert laser_off_called == [True], "finish 应调用 laser_off"
        assert wf.context.get("laser_on") == False
        assert wf.is_measuring == False

        print("PASS: finish_turns_off_laser")


def test_close_all_turns_off_laser() -> None:
    """close_all 方法应在关闭设备时关闭激光（需 laser_stage 已初始化）。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.hardware_mode = "virtual"
        wf = MeasurementWorkflow(cfg)

        wf.context["laser_on"] = True

        laser_off_called = []

        def fake_laser_off():
            laser_off_called.append(True)
            wf.context["laser_on"] = False

        # 必须设置 laser_stage 不为 None，否则 close_all 不会调用 laser_off
        wf.laser_stage = MagicMock()
        wf.laser_stage.close = MagicMock()

        with patch.object(wf, "laser_off", fake_laser_off):
            with patch.object(wf, "save_summary_xlsx", MagicMock()):
                wf.close_all()

        assert laser_off_called == [True], "close_all 应调用 laser_off"
        assert wf.context.get("laser_on") == False

        print("PASS: close_all_turns_off_laser")


def test_close_all_does_not_call_laser_off_when_already_off() -> None:
    """close_all 在激光已关闭时不应重复调用 laser_off。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.hardware_mode = "virtual"
        wf = MeasurementWorkflow(cfg)

        wf.context["laser_on"] = False

        laser_off_called = []

        def fake_laser_off():
            laser_off_called.append(True)

        wf.laser_stage = MagicMock()
        wf.laser_stage.close = MagicMock()

        with patch.object(wf, "laser_off", fake_laser_off):
            with patch.object(wf, "save_summary_xlsx", MagicMock()):
                wf.close_all()

        assert laser_off_called == [], "close_all 不应调用 laser_off"
        assert wf.context.get("laser_on") == False

        print("PASS: close_all_does_not_call_laser_off_when_already_off")


if __name__ == "__main__":
    tests = [
        test_request_stop_turns_off_laser_when_on,
        test_request_stop_does_not_call_laser_off_when_already_off,
        test_request_stop_handles_laser_off_failure,
        test_finish_turns_off_laser,
        test_close_all_turns_off_laser,
        test_close_all_does_not_call_laser_off_when_already_off,
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