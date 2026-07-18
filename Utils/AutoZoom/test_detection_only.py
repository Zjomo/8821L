"""
测试 FocusScore 检测模式（detection_only）。

覆盖：
  1. 默认 detection_only 关闭
  2. 检测模式下 controller 触发后不调用 run_closed_loop
  3. 检测模式下 worker 保存触发图像和分数文件
  4. 非检测模式下触发后正常执行闭环补焦
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from autofocus_qt_ui.qt_compat import QApplication
from autofocus_qt_ui.app import AutofocusMainWindow
from autofocus_qt_ui.worker import AutofocusWorker
from Focus.config import AutofocusConfig
from Focus.controller import AutofocusController
from Focus.metrics import FocusMetricsCalculator
from Focus.scorer import FocusScorer
from Focus.z_axis import ZAxisController

app = QApplication.instance() or QApplication(sys.argv)


def _make_cfg(detection_only: bool = False) -> AutofocusConfig:
    return AutofocusConfig(
        capture_mode="screen_region",
        capture_area=(0, 0, 100, 100),
        focus_roi=(0, 0, 100, 100),
        z_enabled=False,
        autofocus_enabled=True,
        autofocus_focus_trigger_ratio=0.95,
        autofocus_trigger_absolute=True,
        autofocus_detection_only=detection_only,
        autofocus_focus_trigger_count=1,
    )


class _FakeMetricsCalc:
    """用于 controller 级别测试的伪采集器。"""

    def __init__(self, roi_metrics: Dict[str, float]):
        self.cfg = _make_cfg()
        self._roi_metrics = roi_metrics
        self._image = np.zeros((100, 100, 3), dtype=np.uint8)

    def capture_live(self) -> Dict[str, Any]:
        return {
            "full_rgb": self._image,
            "roi_rgb": self._image,
            "full": {},
            "roi_metrics": self._roi_metrics,
        }

    def capture_and_save(self, **kwargs) -> Dict[str, Any]:
        return self.capture_live()


class _FakeScorer:
    """用于 controller 级别测试的伪评分器。"""

    def __init__(self, score: float):
        self._score = score

    def score_ratio(self, roi_metrics):
        return self._score, {}

    def set_reference(self, ref):
        pass


class _FakeZAxis:
    def connect(self):
        raise RuntimeError("不应在 detection_only 模式下连接硬件")

    def move_relative(self, delta):
        raise RuntimeError("不应在 detection_only 模式下移动 Z 轴")

    def close(self):
        pass


def test_default_detection_only_is_false() -> None:
    """默认不开启 FocusScore 检测模式。"""
    cfg = AutofocusConfig()
    assert cfg.autofocus_detection_only is False, "默认 detection_only 应为 False"
    print("PASS: default detection_only is false")


def test_controller_detection_only_skips_closed_loop() -> None:
    """检测模式下触发补焦不应调用 run_closed_loop（避免硬件操作）。"""
    cfg = _make_cfg(detection_only=True)
    scorer = _FakeScorer(score=0.85)
    metrics = _FakeMetricsCalc(roi_metrics={"laplacian_variance": 10.0})
    z_axis = _FakeZAxis()
    ctrl = AutofocusController(cfg, scorer, metrics, z_axis)
    ctrl.on_log = lambda msg: None

    # 设置参考就绪，使 score_ratio 返回非 None
    ctrl.scorer.focus_reference_ready = True

    result = ctrl.check_and_autofocus(cycle_index=1, save_dir=None)

    assert result.get("triggered") is True, "分数 0.85 应触发补焦"
    assert result.get("detection_only") is True, "应标记为 detection_only"
    assert result.get("focus_score_ratio") == 0.85
    print("PASS: controller detection_only skips closed loop")


def test_controller_normal_mode_runs_closed_loop() -> None:
    """非检测模式下触发补焦会执行闭环搜索。"""
    cfg = _make_cfg(detection_only=False)
    scorer = _FakeScorer(score=0.85)
    metrics = _FakeMetricsCalc(roi_metrics={"laplacian_variance": 10.0})

    call_count = {"closed_loop": 0}

    class _ZAxisNoOp:
        def connect(self):
            pass

        def move_relative(self, delta):
            pass

        def close(self):
            pass

    z_axis = _ZAxisNoOp()
    ctrl = AutofocusController(cfg, scorer, metrics, z_axis)
    ctrl.on_log = lambda msg: None

    original_run_closed_loop = ctrl.run_closed_loop

    def _patched_run_closed_loop(initial_focus_score=None):
        call_count["closed_loop"] += 1
        return {
            "ok": True,
            "reason": "test",
            "initial_score": initial_focus_score,
            "best_score": 0.96,
            "best_relative_z_steps": 0,
            "final_relative_z_steps": 0,
            "history": [],
        }

    ctrl.run_closed_loop = _patched_run_closed_loop

    # 设置参考就绪
    ctrl.scorer.focus_reference_ready = True

    result = ctrl.check_and_autofocus(cycle_index=1, save_dir=None)

    assert call_count["closed_loop"] == 1, "非检测模式下应调用一次闭环补焦"
    assert result.get("detection_only") is None or result.get("detection_only") is False
    print("PASS: controller normal mode runs closed loop")


class _MockController:
    """用于 worker 级别测试的伪控制器。"""

    def __init__(self, results: List[Dict[str, Any]], cfg: AutofocusConfig):
        self._results = results
        self._idx = 0
        self.metrics_calc = MagicMock()
        self.metrics_calc.cfg = cfg
        self.z_axis = MagicMock()

    def check_and_autofocus(
        self, cycle_index: int = 0, save_dir: Any = None
    ) -> Dict[str, Any]:
        result = self._results[self._idx % len(self._results)]
        self._idx += 1
        return result


def _result_with_score(score: float, triggered: bool = True, detection_only: bool = False) -> Dict[str, Any]:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    return {
        "ok": True,
        "full_rgb": image,
        "roi_rgb": image,
        "full": {},
        "roi_metrics": {},
        "focus_score_ratio": score,
        "triggered": triggered,
        "detection_only": detection_only,
    }


def test_worker_saves_detection_image() -> None:
    """检测模式下 worker 触发后应保存图像和分数文件。"""
    cfg = _make_cfg(detection_only=True)
    results = [_result_with_score(0.85, triggered=True, detection_only=True)]
    controller = _MockController(results, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        worker = AutofocusWorker()
        worker.configure(
            cfg=cfg,
            args={"cycles": 1, "interval": 0.0, "output": tmpdir},
            controller=controller,
        )
        worker._state.running = True
        worker._run_loop()

        # worker 在 output/cycles/ 下创建 cycle_0001 子目录
        cycle_dir = Path(tmpdir) / "cycles"
        cycle_dirs = list(cycle_dir.glob("cycle_*"))
        assert len(cycle_dirs) == 1, f"应创建一个 cycle 目录，实际 {cycle_dirs}"
        saved_dir = cycle_dirs[0]
        img_path = saved_dir / "trigger_detection.jpg"
        score_path = saved_dir / "trigger_detection_score.txt"

        assert img_path.exists(), f"应保存触发图像：{img_path}"
        assert score_path.exists(), f"应保存触发分数文件：{score_path}"
        content = score_path.read_text(encoding="utf-8")
        assert "focus_score_ratio=0.850000" in content, f"分数文件内容错误：{content}"

    print("PASS: worker saves detection image")


def test_ui_has_detection_only_button() -> None:
    """主窗口应包含 FocusScore检测 按钮且默认未按下。"""
    window = AutofocusMainWindow()
    try:
        assert hasattr(window, "detection_only_btn"), "UI 应包含 detection_only_btn"
        assert window.detection_only_btn.isChecked() is False, "默认未开启检测模式"
        assert window.detection_only_btn.text() == "FocusScore检测"
        print("PASS: UI has detection only button")
    finally:
        window.close()
        QApplication.processEvents()


def main() -> None:
    test_default_detection_only_is_false()
    test_controller_detection_only_skips_closed_loop()
    test_controller_normal_mode_runs_closed_loop()
    test_worker_saves_detection_image()
    test_ui_has_detection_only_button()
    print("\n所有 FocusScore 检测模式测试通过!")


if __name__ == "__main__":
    main()
