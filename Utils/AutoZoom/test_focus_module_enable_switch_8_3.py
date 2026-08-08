"""验证完整循环测量的“启用补焦模块”总开关。"""

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent
MODULE_NAME = "mwf_8_3_focus_module_switch_test"

spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    str(PROJECT_ROOT / "0_measurement_workflow_real_virtual_same_detection_8_3.py"),
)
module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = module
spec.loader.exec_module(module)

MeasurementConfig = module.MeasurementConfig
MeasurementWorkflow = module.MeasurementWorkflow
MeasurementWorkflowGUI = module.MeasurementWorkflowGUI


class FocusModuleEnableSwitchTests(unittest.TestCase):
    def test_config_default_enables_focus_module(self):
        cfg = MeasurementConfig()
        self.assertTrue(hasattr(cfg, "focus_module_enabled"))
        self.assertTrue(cfg.focus_module_enabled)

    def test_gui_builds_focus_module_checkbox_below_step7_preflight(self):
        src = inspect.getsource(MeasurementWorkflowGUI._build_ui)
        self.assertIn("Step7 A/C 标定预检", src)
        self.assertIn("启用补焦模块", src)
        self.assertIn("focus_module_enabled_var", src)

    def test_build_config_syncs_focus_module_checkbox(self):
        src = inspect.getsource(MeasurementWorkflowGUI.build_config_from_ui)
        self.assertIn("focus_module_enabled=bool(self.focus_module_enabled_var.get())", src)

    def test_ui_trace_tracks_focus_module_checkbox(self):
        src = inspect.getsource(MeasurementWorkflowGUI._bind_ui_parameter_traces)
        self.assertIn("self.focus_module_enabled_var", src)

    def test_disabled_capture_focus_reference_skips_without_components(self):
        wf = MeasurementWorkflow.__new__(MeasurementWorkflow)
        wf.cfg = SimpleNamespace(focus_module_enabled=False)
        wf.log = MagicMock()
        wf._ensure_focus_components = MagicMock(side_effect=AssertionError("should not init focus components"))

        ok = MeasurementWorkflow.capture_focus_reference(wf, cycle_index=1)

        self.assertTrue(ok)
        wf._ensure_focus_components.assert_not_called()
        self.assertTrue(any("补焦模块未启用" in str(c.args[0]) for c in wf.log.call_args_list))

    def test_disabled_run_autofocus_skips_without_scoring(self):
        wf = MeasurementWorkflow.__new__(MeasurementWorkflow)
        wf.cfg = SimpleNamespace(focus_module_enabled=False)
        wf.log = MagicMock()
        wf.compute_current_focus_score = MagicMock(side_effect=AssertionError("should not compute focus score"))

        result = MeasurementWorkflow.run_autofocus_if_needed(wf, cycle_index=1)

        self.assertTrue(result["skipped"])
        self.assertEqual(result["reason"], "focus_module_disabled")
        wf.compute_current_focus_score.assert_not_called()

    def test_run_one_cycle_step10_is_guarded_by_focus_module_switch(self):
        src = inspect.getsource(MeasurementWorkflow.run_one_cycle)
        self.assertIn("补焦模块未启用：跳过 Step 1.5", src)
        self.assertIn("Step 10：补焦模块未启用，跳过", src)
        self.assertIn("while self._is_focus_module_enabled():", src)
        self.assertIn('"reason": "focus_module_disabled"', src)


if __name__ == "__main__":
    unittest.main()
