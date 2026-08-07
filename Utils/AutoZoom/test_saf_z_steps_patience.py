"""光谱补焦循环 Z轴移动步数 / 方向判断连续次数 UI 参数测试。"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
AutofocusConfig = _module.AutofocusConfig
HillClimbSearch = __import__("Focus.search", fromlist=["HillClimbSearch"]).HillClimbSearch


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

    def test_config_has_direction_probe_params(self):
        cfg = MeasurementConfig()
        self.assertTrue(hasattr(cfg, "focus_z_direction_probe_steps"))
        self.assertTrue(hasattr(cfg, "focus_z_direction_probe_stage_count"))
        self.assertTrue(hasattr(cfg, "focus_z_direction_probe_step_interval"))
        self.assertTrue(hasattr(cfg, "focus_z_direction_probe_samples"))
        self.assertTrue(hasattr(cfg, "focus_z_direction_probe_points_per_step"))
        self.assertEqual(cfg.focus_z_direction_probe_steps, (10, 20, 30))
        self.assertEqual(cfg.focus_z_direction_probe_stage_count, 3)
        self.assertEqual(cfg.focus_z_direction_probe_step_interval, 10)
        self.assertEqual(cfg.focus_z_direction_probe_samples, 3)
        self.assertEqual(cfg.focus_z_direction_probe_points_per_step, 5)

    def test_config_custom_values(self):
        cfg = MeasurementConfig(focus_z_search_steps=25, focus_z_patience=5)
        self.assertEqual(cfg.focus_z_search_steps, 25)
        self.assertEqual(cfg.focus_z_patience, 5)

    def test_config_has_local_refine_params(self):
        cfg = MeasurementConfig()
        self.assertTrue(hasattr(cfg, "focus_z_local_refine_enabled"))
        self.assertTrue(hasattr(cfg, "focus_z_local_refine_decay"))
        self.assertTrue(hasattr(cfg, "focus_z_local_refine_min_step"))
        self.assertTrue(hasattr(cfg, "focus_z_local_refine_max_rounds"))
        self.assertTrue(cfg.focus_z_local_refine_enabled)
        self.assertEqual(cfg.focus_z_local_refine_decay, 0.5)
        self.assertEqual(cfg.focus_z_local_refine_min_step, 1)
        self.assertEqual(cfg.focus_z_local_refine_max_rounds, 4)


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
        self.assertIn("z_direction_probe_steps", src)
        self.assertIn("z_direction_probe_stage_count", src)
        self.assertIn("z_direction_probe_step_interval", src)
        self.assertIn("z_direction_probe_samples", src)
        self.assertIn("z_direction_probe_points_per_step", src)

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
        self.assertIn("阶段数/步数间隔", src)
        self.assertIn("重复采样", src)

    def test_make_focus_config_passes_new_params(self):
        """验证 MeasurementWorkflow._make_focus_config 传递新参数。"""
        import inspect
        src = inspect.getsource(_module.MeasurementWorkflow._make_focus_config)
        self.assertIn("z_search_steps", src)
        self.assertIn("z_patience", src)
        self.assertIn("z_direction_probe_steps", src)
        self.assertIn("z_direction_probe_stage_count", src)
        self.assertIn("z_direction_probe_step_interval", src)
        self.assertIn("z_direction_probe_samples", src)
        self.assertIn("z_direction_probe_points_per_step", src)
        self.assertIn("z_local_refine_enabled", src)
        self.assertIn("z_local_refine_decay", src)
        self.assertIn("z_local_refine_min_step", src)
        self.assertIn("z_local_refine_max_rounds", src)

    def test_ui_layout_has_local_refine_controls(self):
        """验证 UI 布局代码包含局部细搜控件。"""
        import inspect
        src = inspect.getsource(MeasurementWorkflowGUI._build_ui)
        self.assertIn("启用局部细搜", src)
        self.assertIn("细搜衰减", src)
        self.assertIn("最小步数", src)
        self.assertIn("最大轮数", src)

    def test_make_saf_config_passes_local_refine_params(self):
        """验证 _make_saf_config 把局部细搜参数传入 AutofocusConfig。"""
        import inspect
        src = inspect.getsource(MeasurementWorkflowGUI._make_saf_config)
        self.assertIn("z_local_refine_enabled", src)
        self.assertIn("z_local_refine_decay", src)
        self.assertIn("z_local_refine_min_step", src)
        self.assertIn("z_local_refine_max_rounds", src)

    def test_ui_layout_has_direction_probe_controls(self):
        """验证 UI 布局代码包含动态双向采样控件。"""
        import inspect
        src = inspect.getsource(MeasurementWorkflowGUI._build_ui)
        self.assertIn("阶段数/步数间隔", src)
        self.assertIn("重复采样", src)
        self.assertIn("每档点数", src)

    def test_ui_layout_has_focus_score_preview_button(self):
        """验证 UI 提供 FocusScore 曲线预览按钮。"""
        import inspect
        src = inspect.getsource(MeasurementWorkflowGUI._build_ui)
        self.assertIn("预览FocusScore曲线", src)
        self.assertIn("open_focus_score_preview", src)

    def test_focus_score_preview_methods_exist(self):
        """验证 FocusScore 实时预览窗口的核心方法存在。"""
        import inspect
        src = inspect.getsource(MeasurementWorkflowGUI.open_focus_score_preview)
        self.assertIn("FigureCanvasTkAgg", src)
        self.assertIn("_schedule_focus_score_preview_sample", src)
        src_compute = inspect.getsource(MeasurementWorkflowGUI._compute_focus_score_preview_value)
        self.assertIn("spectrum_autofocus_loop", src_compute)
        self.assertIn("compute_current_focus_score", src_compute)

    def test_make_saf_config_runtime_passes_direction_probe_params(self):
        """运行时验证 _make_saf_config 的动态双向采样参数传递。"""
        class DummyVar:
            def __init__(self, value):
                self._value = value

            def get(self):
                return self._value

        dummy = SimpleNamespace(
            saf_capture_area_var=DummyVar("0,0,640,480"),
            saf_roi_var=DummyVar("0,0,300,300"),
            saf_trigger_ratio_var=DummyVar(0.95),
            saf_stop_ratio_var=DummyVar(0.95),
            saf_trigger_count_var=DummyVar(3),
            saf_trigger_absolute_var=DummyVar(True),
            saf_detection_only_var=DummyVar(False),
            saf_passive_mode_var=DummyVar(True),
            saf_passive_attempts_var=DummyVar(10),
            saf_passive_good_var=DummyVar(5),
            saf_disable_auto_stop_var=DummyVar(False),
            saf_z_enabled_var=DummyVar(True),
            saf_z_axis_var=DummyVar(1),
            saf_z_speed_var=DummyVar(100),
            saf_z_accel_var=DummyVar(100),
            saf_search_strategy_var=DummyVar("hill_climb"),
            saf_z_search_steps_var=DummyVar(10),
            saf_z_patience_var=DummyVar(3),
            saf_z_direction_probe_stage_count_var=DummyVar(3),
            saf_z_direction_probe_step_interval_var=DummyVar(10),
            saf_z_direction_probe_samples_var=DummyVar(3),
            saf_z_direction_probe_points_var=DummyVar(5),
            saf_z_local_refine_enabled_var=DummyVar(True),
            saf_z_local_refine_decay_var=DummyVar(0.5),
            saf_z_local_refine_min_step_var=DummyVar(1),
            saf_z_local_refine_max_rounds_var=DummyVar(4),
            _parse_focus_roi=lambda text: tuple(int(v) for v in str(text).split(",")),
            _parse_int_sequence=lambda text: tuple(int(v) for v in str(text).split(",")),
        )
        cfg = MeasurementWorkflowGUI._make_saf_config(dummy)
        self.assertEqual(cfg.z_direction_probe_steps, (10, 20, 30))
        self.assertEqual(cfg.z_direction_probe_stage_count, 3)
        self.assertEqual(cfg.z_direction_probe_step_interval, 10)
        self.assertEqual(cfg.z_direction_probe_samples, 3)
        self.assertEqual(cfg.z_direction_probe_points_per_step, 5)
        self.assertTrue(cfg.z_local_refine_enabled)

    def test_make_focus_config_runtime_passes_direction_probe_params(self):
        """运行时验证完整循环 _make_focus_config 的动态双向采样参数传递。"""
        dummy_cfg = SimpleNamespace(
            saf_capture_area=(0, 0, 640, 480),
            saf_focus_roi=(0, 0, 300, 300),
            focus_trigger_ratio=0.95,
            focus_stop_ratio=0.95,
            focus_trigger_count=3,
            focus_trigger_absolute=True,
            focus_detection_only=False,
            focus_z_enabled=True,
            focus_z_axis=1,
            focus_z_speed=100,
            focus_z_accel=100,
            focus_search_strategy="hill_climb",
            focus_z_search_steps=10,
            focus_z_patience=3,
            focus_z_direction_probe_steps=(10, 20, 30),
            focus_z_direction_probe_stage_count=3,
            focus_z_direction_probe_step_interval=10,
            focus_z_direction_probe_samples=3,
            focus_z_direction_probe_points_per_step=5,
            focus_z_local_refine_enabled=True,
            focus_z_local_refine_decay=0.5,
            focus_z_local_refine_min_step=1,
            focus_z_local_refine_max_rounds=4,
        )
        dummy = SimpleNamespace(
            cfg=dummy_cfg,
            _parse_focus_roi_text=lambda value: tuple(int(v) for v in value),
        )
        cfg = _module.MeasurementWorkflow._make_focus_config(dummy)
        self.assertEqual(cfg.z_direction_probe_steps, (10, 20, 30))
        self.assertEqual(cfg.z_direction_probe_stage_count, 3)
        self.assertEqual(cfg.z_direction_probe_step_interval, 10)
        self.assertEqual(cfg.z_direction_probe_samples, 3)
        self.assertEqual(cfg.z_direction_probe_points_per_step, 5)
        self.assertTrue(cfg.z_local_refine_enabled)


class HillClimbLocalRefineTests(unittest.TestCase):
    """测试 hill_climb 粗搜后的 best_pos 左右局部细搜。"""

    def _run_search(self, local_refine_enabled: bool):
        cfg = AutofocusConfig(
            autofocus_focus_trigger_ratio=1.1,
            autofocus_stop_ratio=1.1,
            z_probe_steps=10,
            z_search_steps=10,
            z_min_improve_ratio=0.0,
            z_patience=2,
            z_max_iter=8,
            z_max_total_steps=100,
            z_settle_time_s=0.0,
            z_local_refine_enabled=local_refine_enabled,
            z_local_refine_decay=0.5,
            z_local_refine_min_step=1,
            z_local_refine_max_rounds=3,
        )
        state = {"pos": 0}
        logs = []

        def score_at(pos: int) -> float:
            return 1.0 - ((pos - 15) ** 2) / 1000.0

        def move(delta: int) -> None:
            state["pos"] += int(delta)

        def measure(phase: str, iteration: int):
            return score_at(state["pos"]), None

        search = HillClimbSearch(cfg, move, measure, logs.append)
        result = search.search(
            initial_score=score_at(0),
            target=1.1,
            max_total_steps=100,
            max_iter=8,
            patience=2,
            min_improve=0.0,
            upper_target=None,
        )
        return result, logs

    def test_local_refine_improves_best_pos_around_coarse_peak(self):
        result, logs = self._run_search(local_refine_enabled=True)
        self.assertEqual(result["best_relative_z_steps"], 15)
        self.assertAlmostEqual(result["best_score"], 1.0)
        self.assertTrue(any("局部细搜更新最佳" in line for line in logs))

    def test_without_local_refine_keeps_coarse_best_pos(self):
        result, _ = self._run_search(local_refine_enabled=False)
        self.assertEqual(result["best_relative_z_steps"], 10)
        self.assertLess(result["best_score"], 1.0)


class DynamicDirectionSamplingTests(unittest.TestCase):
    """测试动态双向采样的方向判断与中位数采样。"""

    def _make_search(self, score_map, probe_steps=(10, 20, 30), sample_count=3, min_improve=0.02):
        cfg = AutofocusConfig(
            autofocus_focus_trigger_ratio=1.1,
            autofocus_stop_ratio=1.1,
            z_probe_steps=10,
            z_direction_probe_steps=tuple(probe_steps),
            z_direction_probe_samples=sample_count,
            z_search_steps=10,
            z_min_improve_ratio=min_improve,
            z_patience=2,
            z_max_iter=4,
            z_max_total_steps=100,
            z_settle_time_s=0.0,
            z_local_refine_enabled=False,
        )
        state = {"pos": 0}
        logs = []

        def move(delta: int) -> None:
            state["pos"] += int(delta)

        def measure(phase: str, iteration: int):
            scores = score_map.get(state["pos"])
            if not scores:
                return None, None
            idx = measure.calls.get(state["pos"], 0)
            score = scores[idx % len(scores)]
            measure.calls[state["pos"]] = idx + 1
            return score, None

        measure.calls = {}
        search = HillClimbSearch(cfg, move, measure, logs.append)
        return search, logs

    def test_direction_sampling_stops_after_first_decisive_stage(self):
        score_map = {
            0: [1.00, 1.00, 1.00],
            10: [0.70, 1.20, 0.90],  # median = 0.90
            20: [0.80, 1.10, 0.85],  # median = 0.85
            30: [0.90, 0.91, 0.92],  # median = 0.91
            40: [1.01, 1.02, 1.03],  # median = 1.02
            50: [1.07, 1.08, 1.09],  # median = 1.08
            100: [1.20, 1.21, 1.22],
            150: [1.30, 1.31, 1.32],
        }
        search, logs = self._make_search(score_map, probe_steps=(10, 20, 30), min_improve=0.02)
        direction, probe_step, score, pos = search._determine_direction_with_dynamic_sampling(
            probe_steps=[10, 20, 30],
            min_improve=0.02,
            sample_count=3,
            points_per_step=5,
        )
        self.assertEqual(direction, +1)
        self.assertEqual(probe_step, 10)
        self.assertEqual(pos, 50)
        self.assertAlmostEqual(score, 1.08, places=2)
        self.assertTrue(any("probe_step=10" in line for line in logs))
        self.assertFalse(any("probe_step=20" in line for line in logs))
        self.assertFalse(any("probe_step=30" in line for line in logs))
        self.assertTrue(any("samples=[10:" in line and "50:" in line for line in logs))
        self.assertTrue(any("动态多点采样确定方向" in line for line in logs))

    def test_direction_sampling_tries_next_stage_when_previous_not_decisive(self):
        score_map = {
            0: [1.00, 1.00, 1.00],
            10: [0.98, 0.99, 1.00],
            20: [0.99, 1.00, 1.01],
            30: [0.98, 0.99, 1.00],
            40: [0.99, 1.00, 1.01],
            50: [0.98, 0.99, 1.00],
            60: [1.04, 1.05, 1.06],
            80: [1.02, 1.03, 1.04],
            100: [1.01, 1.02, 1.03],
        }
        search, logs = self._make_search(score_map, probe_steps=(10, 20, 30), min_improve=0.02)
        direction, probe_step, score, pos = search._determine_direction_with_dynamic_sampling(
            probe_steps=[10, 20, 30],
            min_improve=0.02,
            sample_count=3,
            points_per_step=5,
        )
        self.assertEqual(direction, +1)
        self.assertEqual(probe_step, 20)
        self.assertEqual(pos, 60)
        self.assertAlmostEqual(score, 1.05, places=2)
        self.assertTrue(any("probe_step=10" in line for line in logs))
        self.assertTrue(any("probe_step=20" in line for line in logs))
        self.assertFalse(any("probe_step=30" in line for line in logs))

    def test_direction_sampling_no_improve_returns_none(self):
        score_map = {
            0: [1.00, 1.00, 1.00],
            10: [0.90, 0.91, 0.89],
            20: [0.90, 0.91, 0.89],
            30: [0.90, 0.91, 0.89],
            40: [0.90, 0.91, 0.89],
            50: [0.90, 0.91, 0.89],
            60: [0.90, 0.91, 0.89],
            80: [0.90, 0.91, 0.89],
            90: [0.90, 0.91, 0.89],
            100: [0.90, 0.91, 0.89],
            120: [0.90, 0.91, 0.89],
            150: [0.90, 0.91, 0.89],
        }
        search, _ = self._make_search(score_map, probe_steps=(10, 20, 30), min_improve=0.05)
        direction, probe_step, score, pos = search._determine_direction_with_dynamic_sampling(
            probe_steps=[10, 20, 30],
            min_improve=0.05,
            sample_count=3,
            points_per_step=5,
        )
        self.assertIsNone(direction)
        self.assertIsNone(probe_step)
        self.assertEqual(score, 1.0)
        self.assertIsNone(pos)


if __name__ == "__main__":
    unittest.main()
