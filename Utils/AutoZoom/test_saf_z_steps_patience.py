"""光谱补焦循环 Z轴移动步数 / 方向判断连续次数 UI 参数测试。"""

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_MODULE_NAME = "mwf_8_3_saf_test"

spec = importlib.util.spec_from_file_location(
    _MODULE_NAME,
    str(PROJECT_ROOT / "0_measurement_workflow_real_virtual_same_detection_8_3.py"),
)
_module = importlib.util.module_from_spec(spec)
sys.modules[_MODULE_NAME] = _module
spec.loader.exec_module(_module)

MeasurementConfig = _module.MeasurementConfig
MeasurementWorkflowGUI = _module.MeasurementWorkflowGUI


class SafZStepsPatienceConfigTests(unittest.TestCase):
    """测试 MeasurementConfig 中新增字段。"""

    def test_config_has_z_search_steps(self):
        cfg = MeasurementConfig()
        self.assertTrue(hasattr(cfg, "focus_z_search_steps"))
        self.assertEqual(cfg.focus_z_search_steps, 10)

    def test_config_has_z_patience(self):
        cfg = MeasurementConfig()
        self.assertTrue(hasattr(cfg, "focus_z_patience"))
        self.assertEqual(cfg.focus_z_patience, 3)

    def test_config_custom_values(self):
        cfg = MeasurementConfig(focus_z_search_steps=25, focus_z_patience=5)
        self.assertEqual(cfg.focus_z_search_steps, 25)
        self.assertEqual(cfg.focus_z_patience, 5)


class SafZStepsPatienceGUITests(unittest.TestCase):
    """测试 GUI 变量和 _make_saf_config 传递。"""

    def setUp(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        self.root.destroy()

    def test_gui_has_z_search_steps_var(self):
        self.assertTrue(hasattr(MeasurementWorkflowGUI, "_build_ui"))
        import inspect
        src = inspect.getsource(MeasurementWorkflowGUI._build_ui)
        self.assertIn("saf_z_search_steps_var", src)
        self.assertIn("saf_z_patience_var", src)

    def test_make_saf_config_passes_new_params(self):
        """验证 _make_saf_config 把新参数传入 AutofocusConfig。"""
        import inspect
        src = inspect.getsource(MeasurementWorkflowGUI._make_saf_config)
        self.assertIn("z_search_steps", src)
        self.assertIn("z_patience", src)

    def test_sync_config_passes_new_params(self):
        """验证 _build_measurement_config_from_ui 传递新参数。"""
        import inspect
        # 找到包含 focus_search_strategy 的配置构建方法
        gui_methods = [
            name for name, _ in inspect.getmembers(MeasurementWorkflowGUI, inspect.isfunction)
        ]
        found = False
        for name in gui_methods:
            src = inspect.getsource(getattr(MeasurementWorkflowGUI, name))
            if "focus_z_search_steps" in src and "focus_z_patience" in src:
                found = True
                break
        self.assertTrue(found, "应有方法同时包含 focus_z_search_steps 和 focus_z_patience")

    def test_ui_layout_has_new_controls(self):
        """验证 UI 布局代码包含新标签和输入框。"""
        import inspect
        src = inspect.getsource(MeasurementWorkflowGUI._build_ui)
        self.assertIn("Z轴移动步数", src)
        self.assertIn("方向判断连续次数", src)

    def test_make_focus_config_passes_new_params(self):
        """验证 MeasurementWorkflow._make_focus_config 传递新参数。"""
        import inspect
        src = inspect.getsource(_module.MeasurementWorkflow._make_focus_config)
        self.assertIn("z_search_steps", src)
        self.assertIn("z_patience", src)


if __name__ == "__main__":
    unittest.main()
