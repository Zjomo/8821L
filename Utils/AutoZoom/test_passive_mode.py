"""
测试 AutoZoom Focus 的被动补焦模式 V2、默认阈值及 FocusScore 相关行为。

覆盖：
  1. 默认触发阈值 0.95、目标阈值 0.95、被动模式默认关闭/最大 10 次/连续达标 3 次
  2. UI 控件反映上述默认值，且勾选被动模式后循环轮数被禁用
  3. 被动模式开启时忽略循环轮数，按间隔运行
  4. 连续达标 n 次后自动停止
  5. 分数未达标时不自动停止（持续运行），手动停止有效
  6. 被动模式关闭时恢复主动模式，按循环轮数停止
  7. 自动补焦禁用时被动模式不生效
  8. 使用 FocusSimulator 验证被动模式最终能把分数拉回 0.95 以上
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from autofocus_qt_ui.qt_compat import QApplication
from autofocus_qt_ui.config import (
    DEFAULT_PASSIVE_CONSECUTIVE_GOOD,
    DEFAULT_PASSIVE_MAX_ATTEMPTS,
    DEFAULT_PASSIVE_MODE,
    DEFAULT_STOP_RATIO,
    DEFAULT_TRIGGER_RATIO,
)
from autofocus_qt_ui.app import AutofocusMainWindow
from autofocus_qt_ui.worker import AutofocusWorker
from Focus.config import AutofocusConfig
from Focus.controller import AutofocusController
from Focus.metrics import FocusMetricsCalculator
from Focus.scorer import FocusScorer
from Focus.z_axis import ZAxisController
from Focus.simulator import create_demo_environment

app = QApplication(sys.argv)


def _make_cfg(
    passive_mode: bool = DEFAULT_PASSIVE_MODE,
    passive_max_attempts: int = DEFAULT_PASSIVE_MAX_ATTEMPTS,
    passive_consecutive_good: int = DEFAULT_PASSIVE_CONSECUTIVE_GOOD,
    autofocus_enabled: bool = True,
    trigger_ratio: float = DEFAULT_TRIGGER_RATIO,
    stop_ratio: float = DEFAULT_STOP_RATIO,
) -> AutofocusConfig:
    return AutofocusConfig(
        capture_mode="screen_region",
        capture_area=(0, 0, 100, 100),
        focus_roi=(0, 0, 100, 100),
        z_enabled=False,
        autofocus_enabled=autofocus_enabled,
        autofocus_focus_trigger_ratio=trigger_ratio,
        autofocus_stop_ratio=stop_ratio,
        autofocus_passive_mode=passive_mode,
        autofocus_passive_max_attempts=passive_max_attempts,
        autofocus_passive_consecutive_good=passive_consecutive_good,
    )


class _MockController:
    """用于 worker 级别测试的伪控制器，按顺序返回预设结果。"""

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


def _result_with_score(score: float) -> Dict[str, Any]:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    return {
        "ok": True,
        "full_rgb": image,
        "roi_rgb": image,
        "full": {},
        "roi_metrics": {},
        "focus_score_ratio": score,
    }


def _run_worker(
    cfg: AutofocusConfig,
    controller: Any,
    cycles: int = 1,
    interval: float = 0.0,
    stop_after_score_cycles: Optional[int] = None,
) -> AutofocusWorker:
    """构造并直接运行 worker 的循环，返回 worker 实例。

    若提供 stop_after_score_cycles，则在 score_updated 信号达到指定轮数时调用
    stop_loop()，用于在被动模式（忽略 cycles）下手动停止测试。
    """
    worker = AutofocusWorker()
    worker.configure(
        cfg=cfg, args={"cycles": cycles, "interval": interval}, controller=controller
    )
    worker._state.running = True

    if stop_after_score_cycles is not None:
        def _stop_on_cycle(cycle: int, score: float) -> None:
            if cycle >= stop_after_score_cycles:
                worker.stop_loop()

        worker.score_updated.connect(_stop_on_cycle)

    worker._run_loop()
    return worker


def test_default_thresholds() -> None:
    """Focus 模块与 UI 配置的默认阈值应符合需求。"""
    cfg = AutofocusConfig()
    assert cfg.autofocus_focus_trigger_ratio == 0.95, (
        f"默认触发阈值应为 0.95，实际 {cfg.autofocus_focus_trigger_ratio}"
    )
    assert cfg.autofocus_stop_ratio == 0.95, (
        f"默认目标阈值应为 0.95，实际 {cfg.autofocus_stop_ratio}"
    )
    assert cfg.autofocus_passive_mode is False, "被动模式默认应关闭"
    assert cfg.autofocus_passive_max_attempts == 10, "默认最大连续补焦次数应为 10"
    assert cfg.autofocus_passive_consecutive_good == 3, "默认连续达标次数应为 3"

    assert DEFAULT_TRIGGER_RATIO == 0.95
    assert DEFAULT_STOP_RATIO == 0.95
    assert DEFAULT_PASSIVE_MODE is False
    assert DEFAULT_PASSIVE_MAX_ATTEMPTS == 10
    assert DEFAULT_PASSIVE_CONSECUTIVE_GOOD == 3
    print("PASS: default thresholds and passive defaults")


def test_default_passive_consecutive_good() -> None:
    """连续达标次数默认值为 3。"""
    cfg = AutofocusConfig()
    assert cfg.autofocus_passive_consecutive_good == 3
    print("PASS: default passive consecutive good count")


def test_ui_controls_reflect_defaults() -> None:
    """主窗口 UI 控件初始值应与默认配置一致。"""
    window = AutofocusMainWindow()
    try:
        assert abs(window.trigger_ratio_spin.value() - 0.95) < 1e-9, (
            f"UI 触发阈值默认值错误：{window.trigger_ratio_spin.value()}"
        )
        assert abs(window.stop_ratio_spin.value() - 0.95) < 1e-9, (
            f"UI 目标阈值默认值错误：{window.stop_ratio_spin.value()}"
        )
        assert window.passive_check.isChecked() is False, "UI 被动模式默认应未勾选"
        assert window.passive_attempts_spin.value() == 10, (
            f"UI 最大连续补焦次数默认值错误：{window.passive_attempts_spin.value()}"
        )
        assert window.passive_good_spin.value() == 3, (
            f"UI 连续达标次数默认值错误：{window.passive_good_spin.value()}"
        )
        assert window.cycles_spin.isEnabled() is True, "未勾选被动模式时循环轮数应可用"

        # 勾选被动模式后循环轮数应被禁用
        window.passive_check.setChecked(True)
        window._on_passive_changed(window.passive_check.checkState())
        assert window.cycles_spin.isEnabled() is False, "勾选被动模式后循环轮数应被禁用"

        print("PASS: UI controls reflect defaults")
    finally:
        window.close()
        QApplication.processEvents()


def test_passive_mode_repeats_until_above_trigger() -> None:
    """被动开启且连续达标次数为 1 时，分数恢复达标后自动停止。"""
    cfg = _make_cfg(passive_mode=True, passive_consecutive_good=1)
    results = [_result_with_score(s) for s in [0.85, 0.88, 0.96]]
    controller = _MockController(results, cfg)

    worker = _run_worker(cfg, controller, cycles=1, interval=0.0)

    assert controller._idx == 3, f"期望调用 3 次，实际 {controller._idx}"
    assert worker._state.focus_score == 0.96, (
        f"最终分数应为 0.96，实际 {worker._state.focus_score}"
    )
    print("PASS: passive mode repeats until above trigger")


def test_passive_mode_runs_indefinitely_when_not_recovered() -> None:
    """被动模式下分数始终未达标时，循环不会自动停止（直到手动停止）。"""
    cfg = _make_cfg(passive_mode=True, passive_consecutive_good=3)
    results = [_result_with_score(0.85) for _ in range(10)]
    controller = _MockController(results, cfg)

    # 运行 5 轮后手动停止
    worker = _run_worker(
        cfg, controller, cycles=1, interval=0.0, stop_after_score_cycles=5
    )

    assert controller._idx == 5, f"期望调用 5 次，实际 {controller._idx}"
    assert worker._state.focus_score == 0.85, (
        f"最终分数应为 0.85，实际 {worker._state.focus_score}"
    )
    print("PASS: passive mode runs indefinitely when not recovered")


def test_passive_mode_ignores_cycles() -> None:
    """被动模式下忽略循环轮数，cycles=1 也能运行多轮。"""
    cfg = _make_cfg(passive_mode=True, passive_consecutive_good=3)
    results = [_result_with_score(0.96) for _ in range(10)]
    controller = _MockController(results, cfg)

    worker = _run_worker(cfg, controller, cycles=1, interval=0.0)

    # 连续达标 3 次后停止，因此共调用 3 次
    assert controller._idx == 3, f"期望调用 3 次，实际 {controller._idx}"
    print("PASS: passive mode ignores cycles")


def test_passive_mode_consecutive_good_stop() -> None:
    """被动模式下分数连续 n 次达标后自动停止。"""
    cfg = _make_cfg(passive_mode=True, passive_consecutive_good=4)
    results = [_result_with_score(0.96) for _ in range(10)]
    controller = _MockController(results, cfg)

    worker = _run_worker(cfg, controller, cycles=1, interval=0.0)

    assert controller._idx == 4, f"期望调用 4 次，实际 {controller._idx}"
    assert worker._state.focus_score == 0.96
    print("PASS: passive mode consecutive good stop")


def test_passive_mode_manual_stop() -> None:
    """被动模式下可通过 stop_loop 手动停止。"""
    cfg = _make_cfg(passive_mode=True, passive_consecutive_good=100)
    results = [_result_with_score(0.96) for _ in range(100)]
    controller = _MockController(results, cfg)

    worker = _run_worker(
        cfg, controller, cycles=1, interval=0.0, stop_after_score_cycles=5
    )

    assert controller._idx == 5, f"期望调用 5 次后手动停止，实际 {controller._idx}"
    assert worker._state.running is False
    print("PASS: passive mode manual stop")


def test_passive_mode_off_keeps_single_autofocus() -> None:
    """被动关闭时恢复主动模式，按循环轮数停止，每轮仅调用一次 check_and_autofocus。"""
    cfg = _make_cfg(passive_mode=False)
    results = [_result_with_score(0.85) for _ in range(10)]
    controller = _MockController(results, cfg)

    worker = _run_worker(cfg, controller, cycles=1, interval=0.0)

    assert controller._idx == 1, f"期望调用 1 次，实际 {controller._idx}"
    print("PASS: passive mode off keeps single autofocus")


def test_passive_disabled_when_autofocus_off() -> None:
    """自动补焦禁用时，即使勾选被动模式也不连续补焦。"""
    cfg = _make_cfg(passive_mode=True, autofocus_enabled=False)
    results = [_result_with_score(0.85) for _ in range(10)]
    controller = _MockController(results, cfg)

    worker = _run_worker(cfg, controller, cycles=1, interval=0.0)

    assert controller._idx == 1, f"期望调用 1 次，实际 {controller._idx}"
    print("PASS: passive disabled when autofocus off")


def test_integration_simulator_passive_recovery() -> None:
    """使用 FocusSimulator 验证被动模式最终能把分数拉回 0.95 以上。"""
    cfg = _make_cfg(passive_mode=True, passive_consecutive_good=1)
    # 使用非 screen_region 模式，避免 worker 调用主线程截图
    cfg.capture_mode = "usb_camera"
    # 模拟模式：peak_z=50，初始 z=0，blur_scale=0.3，起始明显离焦
    simulator, metrics_calc, scorer, controller = create_demo_environment(
        cfg, peak_z=50, blur_scale=0.3, initial_z=0
    )
    # 预先建立参考（在聚焦峰值处）
    simulator.virtual_z_axis.move_absolute(50)
    ref = controller.build_reference()
    assert ref is not None, "参考建立失败"

    # 回到离焦位置
    simulator.virtual_z_axis.move_absolute(0)

    worker = AutofocusWorker()
    worker.configure(
        cfg=cfg,
        args={"cycles": 1, "interval": 0.0},
        controller=controller,
        simulator=simulator,
    )

    scores: List[float] = []
    worker.score_updated.connect(lambda c, s: scores.append(s))

    worker._state.running = True
    worker._run_loop()

    assert len(scores) >= 1, "应至少有一次分数更新"
    final_score = scores[-1]
    assert final_score >= 0.95, (
        f"被动模式最终分数应 ≥ 0.95，实际 {final_score}"
    )
    print(f"PASS: simulator passive recovery final_score={final_score:.4f}")


if __name__ == "__main__":
    test_default_thresholds()
    test_default_passive_consecutive_good()
    test_ui_controls_reflect_defaults()
    test_passive_mode_repeats_until_above_trigger()
    test_passive_mode_runs_indefinitely_when_not_recovered()
    test_passive_mode_ignores_cycles()
    test_passive_mode_consecutive_good_stop()
    test_passive_mode_manual_stop()
    test_passive_mode_off_keeps_single_autofocus()
    test_passive_disabled_when_autofocus_off()
    test_integration_simulator_passive_recovery()
    print("\nAll tests passed!")
    app.quit()
    QApplication.processEvents()
