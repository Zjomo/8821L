"""验证 8_3 流程停止/结束时会释放 Newport 8742/8743 实例。"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent
MODULE_NAME = "mwf_8_3_newport_release_test"

spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    str(PROJECT_ROOT / "0_measurement_workflow_real_virtual_same_detection_8_3.py"),
)
module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = module
spec.loader.exec_module(module)

MeasurementWorkflow = module.MeasurementWorkflow


class NewportReleaseOnStopTests(unittest.TestCase):
    def _make_workflow(self):
        wf = MeasurementWorkflow.__new__(MeasurementWorkflow)
        wf.log = MagicMock()
        wf.notify_update = MagicMock()
        wf.save_summary_xlsx = MagicMock()
        wf.laser_off = MagicMock()
        wf.excitation_light_off = MagicMock()
        wf.context = {
            "laser_on": False,
            "excitation_light_on": False,
        }
        wf.laser_stage = SimpleNamespace(close=MagicMock())
        wf._focus_controller = SimpleNamespace(close=MagicMock())
        wf.stop_requested = False
        wf.is_measuring = True
        wf.step9_stop_requested = False
        wf.light = None
        return wf

    def test_request_stop_releases_8742_and_8743(self):
        wf = self._make_workflow()

        MeasurementWorkflow.request_stop(wf)

        self.assertIsNone(wf.laser_stage)
        self.assertIsNone(wf._focus_controller)
        self.assertTrue(any("8743 instance released" in str(c.args[0]) for c in wf.log.call_args_list))
        self.assertTrue(any("8742 focus instance released" in str(c.args[0]) for c in wf.log.call_args_list))
        wf.laser_stage is None

    def test_finish_releases_8742_and_8743(self):
        wf = self._make_workflow()

        MeasurementWorkflow.finish(wf)

        self.assertIsNone(wf.laser_stage)
        self.assertIsNone(wf._focus_controller)
        self.assertTrue(any("8743 instance released" in str(c.args[0]) for c in wf.log.call_args_list))
        self.assertTrue(any("8742 focus instance released" in str(c.args[0]) for c in wf.log.call_args_list))

    def test_release_helper_closes_existing_objects(self):
        wf = self._make_workflow()
        stage = wf.laser_stage
        focus = wf._focus_controller

        MeasurementWorkflow._release_newport_motion_controllers(wf, reason="test")

        stage.close.assert_called_once()
        focus.close.assert_called_once()
        self.assertIsNone(wf.laser_stage)
        self.assertIsNone(wf._focus_controller)


if __name__ == "__main__":
    unittest.main()
