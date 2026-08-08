"""反馈闭环搜索策略 (FeedbackClosedLoopSearch) 单元测试。"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from Focus.config import AutofocusConfig
from Focus.search import (
    BaseFocusSearch,
    FeedbackClosedLoopSearch,
    FocusSearchStopped,
    create_search,
)


class TestConfigFields(unittest.TestCase):
    """测试 AutofocusConfig 新增字段。"""

    def test_new_fields_exist(self):
        cfg = AutofocusConfig()
        self.assertEqual(cfg.focus_consecutive_good_checks, 3)
        self.assertEqual(cfg.focus_max_failed_autofocus_attempts, 5)
        self.assertEqual(cfg.focus_max_stale_checks, 5)
        self.assertEqual(cfg.focus_missing_score_max_checks, 3)
        self.assertEqual(cfg.focus_wait_timeout_s, 0.0)
        self.assertEqual(cfg.focus_trend_window, 10)

    def test_custom_values(self):
        cfg = AutofocusConfig(
            focus_consecutive_good_checks=7,
            focus_max_failed_autofocus_attempts=10,
            focus_max_stale_checks=8,
            focus_missing_score_max_checks=4,
            focus_wait_timeout_s=30.0,
            focus_trend_window=15,
        )
        self.assertEqual(cfg.focus_consecutive_good_checks, 7)
        self.assertEqual(cfg.focus_max_failed_autofocus_attempts, 10)
        self.assertEqual(cfg.focus_max_stale_checks, 8)
        self.assertEqual(cfg.focus_missing_score_max_checks, 4)
        self.assertEqual(cfg.focus_wait_timeout_s, 30.0)
        self.assertEqual(cfg.focus_trend_window, 15)


class TestCreateSearch(unittest.TestCase):
    """测试 create_search 注册新策略。"""

    def test_creates_feedback_closed_loop(self):
        cfg = AutofocusConfig()
        move_fn = lambda delta: None
        measure_fn = lambda phase, it: (0.9, None)
        log_fn = lambda msg: None
        search = create_search("feedback_closed_loop", cfg, move_fn, measure_fn, log_fn)
        self.assertIsInstance(search, FeedbackClosedLoopSearch)

    def test_creates_feedback_closed_loop_case_insensitive(self):
        cfg = AutofocusConfig()
        search = create_search("Feedback_Closed_Loop", cfg, lambda d: None, lambda p, i: (0.9, None), lambda m: None)
        self.assertIsInstance(search, FeedbackClosedLoopSearch)


class TestFeedbackClosedLoopSearch(unittest.TestCase):
    """测试 FeedbackClosedLoopSearch 退出条件。"""

    def setUp(self):
        self.cfg = AutofocusConfig()
        self.cfg.z_search_steps = 10
        self.cfg.z_probe_steps = 10
        self.cfg.z_max_iter = 100
        self.cfg.z_max_total_steps = 400
        self.cfg.z_min_improve_ratio = 0.005
        self.cfg.focus_consecutive_good_checks = 3
        self.cfg.focus_max_failed_autofocus_attempts = 5
        self.cfg.focus_max_stale_checks = 5
        self.cfg.focus_missing_score_max_checks = 3
        self.cfg.focus_wait_timeout_s = 0.0
        self.cfg.focus_trend_window = 10

    def _make_search(self, scores: List[Optional[float]], cfg: AutofocusConfig = None):
        """构造一个 FeedbackClosedLoopSearch，measure_fn 按顺序返回 scores。"""
        cfg = cfg or self.cfg
        score_iter = iter(scores)
        pos_log: List[int] = []

        def move_fn(delta: int):
            pos_log.append(delta)

        def measure_fn(phase: str, iteration: int) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
            try:
                s = next(score_iter)
            except StopIteration:
                s = None
            return (s, None)

        log_fn = lambda msg: None
        search = FeedbackClosedLoopSearch(cfg, move_fn, measure_fn, log_fn, should_stop=None)
        return search, pos_log

    def test_already_in_tolerance(self):
        """初始分数已在容差区间内，应立即返回。"""
        search, _ = self._make_search([0.97])
        result = search.search(
            initial_score=0.97,
            target=0.95,
            max_total_steps=400,
            max_iter=100,
            patience=3,
            min_improve=0.005,
            upper_target=1.05,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["reason"], "already_in_tolerance")

    def test_consecutive_good_exit(self):
        """3.1 连续达标 3 次后退出。"""
        # 初始 0.80（未达标），方向探测阶段需要大量分数
        # 之后每次返回 0.97（达标），连续 3 次退出
        scores = [0.80]  # start
        # 方向探测阶段：center + 2方向 * 5点 = ~11次测量
        scores.extend([0.80] * 20)  # 方向探测
        # 反馈递进阶段：连续返回达标分数
        scores.extend([0.97, 0.97, 0.97])
        search, _ = self._make_search(scores)
        result = search.search(
            initial_score=0.80,
            target=0.95,
            max_total_steps=400,
            max_iter=100,
            patience=3,
            min_improve=0.005,
            upper_target=1.05,
        )
        self.assertTrue(result["ok"])

    def test_missing_score_exit(self):
        """3.4 分数为空连续 3 次后退出。"""
        # 初始 0.80（未达标），方向探测阶段需要部分分数递增以确定方向
        scores = [0.80]  # start
        # 方向探测：center 采样 + 各方向探测
        # 让 +1 方向有递增趋势，使其被选为方向
        probe_scores = []
        for i in range(3):  # sample_count=3
            probe_scores.append(0.80)  # center median
        # +1 direction: 5 points, each sampled 3 times
        for p in range(5):
            for s in range(3):
                probe_scores.append(0.81 + p * 0.01)  # 递增
        # return to center
        for i in range(3):
            probe_scores.append(0.80)
        # -1 direction: 5 points, each sampled 3 times (lower scores)
        for p in range(5):
            for s in range(3):
                probe_scores.append(0.79)
        # return to center
        for i in range(3):
            probe_scores.append(0.80)
        scores.extend(probe_scores)
        # 反馈递进阶段：连续返回 None
        scores.extend([None, None, None])
        search, _ = self._make_search(scores)
        result = search.search(
            initial_score=0.80,
            target=0.95,
            max_total_steps=400,
            max_iter=100,
            patience=3,
            min_improve=0.005,
            upper_target=1.05,
        )
        self.assertFalse(result["ok"])

    def test_max_total_steps_exit(self):
        """3.6 达到最大步数后退出。"""
        cfg = AutofocusConfig()
        cfg.z_search_steps = 100
        cfg.z_probe_steps = 10
        cfg.z_max_iter = 100
        cfg.z_max_total_steps = 50
        cfg.z_min_improve_ratio = 0.005
        cfg.focus_consecutive_good_checks = 100
        cfg.focus_max_failed_autofocus_attempts = 100
        cfg.focus_max_stale_checks = 100
        cfg.focus_missing_score_max_checks = 100
        cfg.focus_wait_timeout_s = 0.0
        cfg.focus_trend_window = 10

        scores = [0.80]  # start
        # 方向探测：让 +1 方向有递增趋势
        for i in range(3):
            scores.append(0.80)
        for p in range(5):
            for s in range(3):
                scores.append(0.81 + p * 0.01)
        for i in range(3):
            scores.append(0.80)
        for p in range(5):
            for s in range(3):
                scores.append(0.79)
        for i in range(3):
            scores.append(0.80)
        # 反馈递进：不达标但持续递增（避免触发 stale/failed）
        scores.extend([0.82] * 50)
        search, _ = self._make_search(scores, cfg=cfg)
        result = search.search(
            initial_score=0.80,
            target=0.95,
            max_total_steps=50,
            max_iter=100,
            patience=3,
            min_improve=0.005,
            upper_target=1.05,
        )
        self.assertFalse(result["ok"])

    def test_no_initial_score(self):
        """初始分数为空应返回失败。"""
        search, _ = self._make_search([None])
        result = search.search(
            initial_score=None,
            target=0.95,
            max_total_steps=400,
            max_iter=100,
            patience=3,
            min_improve=0.005,
            upper_target=1.05,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "no_initial_score")


class TestGUISourceIntegration(unittest.TestCase):
    """测试 8_3 文件中 GUI 代码包含新参数。"""

    def test_gui_has_new_vars(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mwf_fcl_test",
            str(PROJECT_ROOT / "0_measurement_workflow_real_virtual_same_detection_8_3.py"),
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["mwf_fcl_test"] = module
        spec.loader.exec_module(module)

        import inspect
        src = inspect.getsource(module.MeasurementWorkflowGUI._build_ui)
        self.assertIn("saf_consecutive_good_checks_var", src)
        self.assertIn("saf_max_failed_autofocus_var", src)
        self.assertIn("saf_max_stale_checks_var", src)
        self.assertIn("saf_missing_score_max_checks_var", src)
        self.assertIn("saf_wait_timeout_var", src)
        self.assertIn("saf_trend_window_var", src)
        self.assertIn("feedback_closed_loop", src)
        self.assertIn("连续达标退出", src)
        self.assertIn("连续失败退出", src)
        self.assertIn("无改善退出", src)
        self.assertIn("空分数退出", src)
        self.assertIn("趋势窗口", src)

    def test_config_has_new_fields(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mwf_fcl_test2",
            str(PROJECT_ROOT / "0_measurement_workflow_real_virtual_same_detection_8_3.py"),
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["mwf_fcl_test2"] = module
        spec.loader.exec_module(module)

        cfg = module.MeasurementConfig()
        self.assertEqual(cfg.focus_consecutive_good_checks, 3)
        self.assertEqual(cfg.focus_max_failed_autofocus_attempts, 5)
        self.assertEqual(cfg.focus_max_stale_checks, 5)
        self.assertEqual(cfg.focus_missing_score_max_checks, 3)
        self.assertEqual(cfg.focus_wait_timeout_s, 0.0)
        self.assertEqual(cfg.focus_trend_window, 10)

    def test_make_saf_config_passes_new_params(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mwf_fcl_test3",
            str(PROJECT_ROOT / "0_measurement_workflow_real_virtual_same_detection_8_3.py"),
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["mwf_fcl_test3"] = module
        spec.loader.exec_module(module)

        import inspect
        src = inspect.getsource(module.MeasurementWorkflowGUI._make_saf_config)
        self.assertIn("focus_consecutive_good_checks", src)
        self.assertIn("focus_max_failed_autofocus_attempts", src)
        self.assertIn("focus_max_stale_checks", src)
        self.assertIn("focus_missing_score_max_checks", src)
        self.assertIn("focus_wait_timeout_s", src)
        self.assertIn("focus_trend_window", src)


if __name__ == "__main__":
    unittest.main()
