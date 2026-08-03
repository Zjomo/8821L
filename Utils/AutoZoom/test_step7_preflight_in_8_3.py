"""Step7 A/C 标定预检功能单元测试（8_3 文件）。"""

import importlib.util
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
_MODULE_NAME = "mwf_8_3_under_test"

spec = importlib.util.spec_from_file_location(
    _MODULE_NAME,
    str(PROJECT_ROOT / "0_measurement_workflow_real_virtual_same_detection_8_3.py"),
)
_module = importlib.util.module_from_spec(spec)
sys.modules[_MODULE_NAME] = _module
spec.loader.exec_module(_module)

MeasurementWorkflow = _module.MeasurementWorkflow
MeasurementConfig = _module.MeasurementConfig
CalibrationState = _module.CalibrationState
MeasurementWorkflowGUI = _module.MeasurementWorkflowGUI


class Step7PreflightHelperTests(unittest.TestCase):
    """测试新增/移植的辅助方法。"""

    def setUp(self):
        self.cfg = MeasurementConfig()
        self.cfg.save_root = tempfile.mkdtemp(prefix="step7_preflight_test_")
        self.wf = MeasurementWorkflow(self.cfg)

    def tearDown(self):
        shutil.rmtree(self.cfg.save_root, ignore_errors=True)

    def test_mask_to_2d_bool(self):
        # 2D bool
        m = np.array([[True, False], [False, True]], dtype=bool)
        r = MeasurementWorkflow._mask_to_2d_bool(m)
        self.assertEqual(r.shape, (2, 2))
        self.assertTrue(r.dtype == bool)

        # 2D uint8
        m = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        r = MeasurementWorkflow._mask_to_2d_bool(m)
        self.assertTrue(r[0, 1])
        self.assertFalse(r[0, 0])

        # 3D 单通道
        m = np.zeros((2, 2, 1), dtype=np.uint8)
        m[0, 0, 0] = 1
        r = MeasurementWorkflow._mask_to_2d_bool(m)
        self.assertEqual(r.shape, (2, 2))
        self.assertTrue(r[0, 0])

        # 3D 四通道 alpha
        m = np.zeros((2, 2, 4), dtype=np.uint8)
        m[0, 0, 3] = 255
        r = MeasurementWorkflow._mask_to_2d_bool(m)
        self.assertTrue(r[0, 0])

        # target_shape resize
        m = np.ones((4, 4), dtype=bool)
        r = MeasurementWorkflow._mask_to_2d_bool(m, target_shape=(2, 2))
        self.assertEqual(r.shape, (2, 2))

    def test_preflight_mask_metrics(self):
        # None
        metrics = self.wf._preflight_mask_metrics(None)
        self.assertFalse(metrics["ok"])
        self.assertEqual(metrics["area_px"], 0)

        # 有效 mask
        m = np.zeros((10, 10), dtype=bool)
        m[2:5, 3:7] = True
        metrics = self.wf._preflight_mask_metrics(m)
        self.assertTrue(metrics["ok"])
        self.assertEqual(metrics["area_px"], 12)
        self.assertAlmostEqual(metrics["center_x"], 4.5)
        self.assertAlmostEqual(metrics["center_y"], 3.0)
        self.assertEqual(metrics["bbox_xyxy"], [3, 2, 6, 4])

    def test_fit_circle_from_mask(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, 255, -1)
        result = MeasurementWorkflow._fit_circle_from_mask(img > 0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["center_xy"][0], 50, delta=1)
        self.assertAlmostEqual(result["center_xy"][1], 50, delta=1)
        self.assertAlmostEqual(result["radius_px"], 30, delta=1)
        self.assertGreater(result["circularity"], 0.8)

    def test_generate_octagon_route_points(self):
        pts = MeasurementWorkflow._generate_octagon_route_points((50, 50), 30, follow_direction=1)
        self.assertEqual(len(pts), 8)
        self.assertNotEqual(pts[0], pts[-1])
        for x, y in pts:
            d = math.hypot(x - 50, y - 50)
            self.assertAlmostEqual(d, 30, delta=1)

    def test_find_preflight_a_template_mask(self):
        tmp = Path(self.cfg.save_root) / "feature_profile"
        tmp.mkdir(parents=True, exist_ok=True)
        target = tmp / "a_sam2_initial_mask_cleaned.png"
        cv2.imwrite(str(target), np.ones((10, 10), dtype=np.uint8) * 255)
        state = CalibrationState()
        state.feature_profile_dir = str(tmp)
        found = self.wf._find_preflight_a_template_mask(state)
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "a_sam2_initial_mask_cleaned.png")

    def test_save_step7_preflight_overlay(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        a_mask = np.zeros((50, 50), dtype=bool)
        a_mask[20:30, 20:30] = True
        c_mask = np.zeros((50, 50), dtype=bool)
        c_mask[10:40, 10:40] = True
        route = [(10, 10), (40, 10), (40, 40), (10, 40)]
        out = Path(self.cfg.save_root) / "overlay.png"
        path = self.wf._save_step7_preflight_overlay(
            image_rgb=img,
            a_mask=a_mask,
            a_template_mask=None,
            c_mask=c_mask,
            route_points=route,
            a_center=(25, 25),
            nearest_xy=(25, 10),
            out_path=out,
            passed=True,
        )
        self.assertTrue(Path(path).exists())
        self.assertGreater(Path(path).stat().st_size, 0)

    def test_build_and_save_circle_edge_route(self):
        c_path = Path(self.cfg.save_root) / "calibration" / "c"
        c_path.mkdir(parents=True, exist_ok=True)
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, 255, -1)
        cv2.imwrite(str(c_path / "static_c_mask.png"), img)
        route = self.wf._build_and_save_circle_edge_route(c_path)
        self.assertEqual(len(route), 8)
        route_file = c_path / "static_c_edge_route.json"
        self.assertTrue(route_file.exists())
        with route_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["route_shape"], "octagon")


class Step7PreflightIntegrationTests(unittest.TestCase):
    """测试 validate_step7_ac_calibration_preflight 主流程。"""

    def setUp(self):
        self.cfg = MeasurementConfig()
        self.cfg.save_root = tempfile.mkdtemp(prefix="step7_preflight_integ_")
        self.cfg.rule_ab_c_edge_route_source = "circle"
        self.wf = MeasurementWorkflow(self.cfg)

    def tearDown(self):
        shutil.rmtree(self.cfg.save_root, ignore_errors=True)

    def _make_calibration_state(self):
        state = CalibrationState()
        state.rule_ab_a_positive_points = [[100, 100]]
        return state

    def _create_c_calibration(self):
        c_dir = Path(self.cfg.save_root) / "calibration" / "c"
        c_dir.mkdir(parents=True, exist_ok=True)
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, 255, -1)
        cv2.imwrite(str(c_dir / "static_c_mask.png"), img)
        self.wf._build_and_save_circle_edge_route(c_dir)
        return c_dir

    @patch.object(MeasurementWorkflow, "_capture_and_build_scene_with_ab_retry")
    @patch.object(MeasurementWorkflow, "ensure_rule_ab_follower")
    @patch.object(MeasurementWorkflow, "_reset_rule_ab_follower_after_transient_error")
    @patch.object(MeasurementWorkflow, "_postprocess_scene_ab_masks")
    @patch.object(MeasurementWorkflow, "prepare_runtime_from_full_calibration", side_effect=lambda s: s)
    @patch.object(MeasurementWorkflow, "preflight_check_calibration")
    @patch.object(MeasurementWorkflow, "_get_loaded_calibration_state", return_value=None)
    @patch.object(MeasurementWorkflow, "_force_cfg_to_strict_full_calibration_c")
    def test_validate_step7_ac_calibration_preflight_pass(
        self,
        mock_force,
        mock_get_loaded,
        mock_preflight,
        mock_prepare,
        mock_postprocess,
        mock_reset,
        mock_ensure,
        mock_capture,
    ):
        state = self._make_calibration_state()
        mock_preflight.return_value = state
        c_dir = self._create_c_calibration()
        mock_force.return_value = str(c_dir)

        scene = MagicMock()
        scene.a = MagicMock()
        scene.a.mask = np.zeros((100, 100), dtype=bool)
        scene.a.mask[40:60, 40:60] = True
        mock_capture.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), scene, {})

        result = self.wf.validate_step7_ac_calibration_preflight()
        self.assertTrue(result["ok"])
        self.assertTrue(result["pass_items"]["a_positive_points_ok"])
        self.assertTrue(result["pass_items"]["c_static_mask_ok"])
        self.assertTrue(result["pass_items"]["c_route_ok"])
        self.assertTrue(result["pass_items"]["current_a_mask_ok"])
        self.assertIn("overlay_path", result)
        self.assertIn("json_path", result)

    @patch.object(MeasurementWorkflow, "_capture_and_build_scene_with_ab_retry")
    @patch.object(MeasurementWorkflow, "ensure_rule_ab_follower")
    @patch.object(MeasurementWorkflow, "_reset_rule_ab_follower_after_transient_error")
    @patch.object(MeasurementWorkflow, "_postprocess_scene_ab_masks")
    @patch.object(MeasurementWorkflow, "prepare_runtime_from_full_calibration", side_effect=lambda s: s)
    @patch.object(MeasurementWorkflow, "preflight_check_calibration")
    @patch.object(MeasurementWorkflow, "_get_loaded_calibration_state", return_value=None)
    @patch.object(MeasurementWorkflow, "_force_cfg_to_strict_full_calibration_c")
    def test_validate_step7_ac_calibration_preflight_fail_a_mask(
        self,
        mock_force,
        mock_get_loaded,
        mock_preflight,
        mock_prepare,
        mock_postprocess,
        mock_reset,
        mock_ensure,
        mock_capture,
    ):
        state = self._make_calibration_state()
        mock_preflight.return_value = state
        c_dir = self._create_c_calibration()
        mock_force.return_value = str(c_dir)

        scene = MagicMock()
        scene.a = MagicMock()
        scene.a.mask = np.zeros((100, 100), dtype=bool)
        mock_capture.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), scene, {})

        with self.assertRaises(RuntimeError) as ctx:
            self.wf.validate_step7_ac_calibration_preflight()
        self.assertIn("A mask 为空", str(ctx.exception))


class Step7PreflightGUITests(unittest.TestCase):
    """测试 UI 入口存在且可调用。"""

    def test_gui_methods_exist(self):
        self.assertTrue(hasattr(MeasurementWorkflowGUI, "step7_ac_preflight_thread"))
        self.assertTrue(hasattr(MeasurementWorkflowGUI, "step7_ac_preflight"))

    def test_step7_ac_preflight_thread_runs_in_thread(self):
        """用 object.__new__ 构造一个最小 GUI 实例，验证 thread 入口会调用 run_in_thread。"""
        import tkinter as tk

        root = tk.Tk()
        try:
            gui = object.__new__(MeasurementWorkflowGUI)
            gui.root = root
            gui.is_busy = False
            gui.flow_status_var = tk.StringVar(root)
            gui.log = MagicMock()
            gui.set_var = MeasurementWorkflowGUI.set_var.__get__(gui, MeasurementWorkflowGUI)
            gui._prepare_for_manual_calibration = MagicMock()
            gui.ensure_workflow = MagicMock()
            gui.sync_config_from_ui_to_workflow = MagicMock()
            wf = MagicMock()
            wf.validate_step7_ac_calibration_preflight.return_value = {
                "ok": True,
                "overlay_path": "/tmp/overlay.png",
                "json_path": "/tmp/result.json",
            }
            gui.ensure_workflow.return_value = wf

            threads = []
            gui.run_in_thread = lambda target: threads.append(target)

            gui.step7_ac_preflight_thread()
            self.assertEqual(len(threads), 1)
            self.assertEqual(threads[0].__func__, gui.step7_ac_preflight.__func__)
            self.assertIs(threads[0].__self__, gui)

            # 直接调用 target 验证完整流程
            threads[0]()
            wf.validate_step7_ac_calibration_preflight.assert_called_once()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
