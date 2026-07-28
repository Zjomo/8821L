"""
验证已从“6. 光谱仪通信 / 单次光谱采集”区域彻底移除“自动保存光谱数据”复选框及相关逻辑。
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

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


def _get_main_source() -> str:
    module_path = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_25.py"
    return module_path.read_text(encoding="utf-8")


def test_config_has_no_save_single_spectrum_enabled() -> None:
    """MeasurementConfig 中不应再包含 save_single_spectrum_enabled 字段。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig

    cfg = MeasurementConfig()
    assert not hasattr(cfg, "save_single_spectrum_enabled")
    assert "save_single_spectrum_enabled" not in dir(cfg)

    print("PASS: config_has_no_save_single_spectrum_enabled")


def test_gui_has_no_save_single_spectrum_var() -> None:
    """MeasurementWorkflowGUI 实例化后不应再包含 save_single_spectrum_var 属性。"""
    _module = _import_7_25()
    MeasurementWorkflowGUI = _module.MeasurementWorkflowGUI

    with patch.object(MeasurementWorkflowGUI, "__init__", lambda self, root: None):
        gui = MeasurementWorkflowGUI(None)
        assert not hasattr(gui, "save_single_spectrum_var")

    print("PASS: gui_has_no_save_single_spectrum_var")


def test_measure_spectrum_once_does_not_reference_checkbox() -> None:
    """measure_spectrum_once 方法源码中不应再引用已删除的变量/文本。"""
    _module = _import_7_25()
    source = inspect.getsource(_module.MeasurementWorkflowGUI.measure_spectrum_once)

    assert "save_single_spectrum_var" not in source
    assert "save_single_spectrum_enabled" not in source
    assert "自动保存光谱数据" not in source

    print("PASS: measure_spectrum_once_does_not_reference_checkbox")


def test_ui_does_not_contain_auto_save_checkbox() -> None:
    """主文件源码中不应再出现“自动保存光谱数据”复选框及对应变量。"""
    source = _get_main_source()

    assert "自动保存光谱数据" not in source
    assert "save_single_spectrum_var" not in source

    print("PASS: ui_does_not_contain_auto_save_checkbox")


if __name__ == "__main__":
    tests = [
        test_config_has_no_save_single_spectrum_enabled,
        test_gui_has_no_save_single_spectrum_var,
        test_measure_spectrum_once_does_not_reference_checkbox,
        test_ui_does_not_contain_auto_save_checkbox,
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
