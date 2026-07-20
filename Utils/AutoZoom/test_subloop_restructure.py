"""
测试 7_16 完整循环测量的子循环重构。

覆盖：
  1. MeasurementConfig 子循环次数字段默认值
  2. laser_on() 幂等保护（已 ON 时不再调用轴运动）
  3. run_one_cycle 子循环迭代次数控制
  4. 聚焦参考图仅在 cycle_index=1 建立一次
  5. sub_loop_iterations_per_cycle=0 时跳过子循环
  6. 所有保存角度均来自 Step1 的 YOLO-OBB baseline
  7. 子循环结束后安全兜底关闭激光

所有测试在 virtual 硬件模式下运行，不依赖真实设备。
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

_module_path = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_16.py"
_spec = importlib.util.spec_from_file_location(
    "measurement_workflow_7_16_subloop", str(_module_path)
)
_measurement_workflow = importlib.util.module_from_spec(_spec)
sys.modules["measurement_workflow_7_16_subloop"] = _measurement_workflow
_spec.loader.exec_module(_measurement_workflow)  # type: ignore[union-attr]

MeasurementConfig = _measurement_workflow.MeasurementConfig
MeasurementWorkflow = _measurement_workflow.MeasurementWorkflow


def _make_config(**kwargs: Any) -> MeasurementConfig:
    """构造 virtual 模式配置，避免连接真实硬件。"""
    defaults: Dict[str, Any] = {
        "hardware_mode": "virtual",
        "sub_loop_iterations_per_cycle": 1,
        "save_root": "measurement_output_test",
        "laser_axis": 2,
        "laser_on_steps": 400,
        "laser_off_steps": 400,
    }
    defaults.update(kwargs)
    return MeasurementConfig(**defaults)


def _make_workflow(cfg: MeasurementConfig) -> MeasurementWorkflow:
    """构造 workflow，并重定向输出根目录到临时目录，避免污染真实数据。"""
    wf = MeasurementWorkflow(cfg)
    return wf


def _mock_run_one_cycle_dependencies(wf: MeasurementWorkflow, sub_loop_count: int) -> Dict[str, Any]:
    """
    对 run_one_cycle 所依赖的完整流程方法做统一 mock，返回 mocks 字典便于断言。
    """
    mocks: Dict[str, Any] = {}

    mocks["follower"] = MagicMock()
    wf._ensure_rule_ab_follower_with_startup_retry = MagicMock(return_value=mocks["follower"])
    wf.apply_runtime_rule_ab_params_to_follower = MagicMock()
    wf._ensure_rule_ab_feature_tracker_installed = MagicMock()

    mocks["angle_result"] = {
        "ok": True,
        "angle_deg": 30.0,
        "reason": "ok",
        "angle_source": "yolo_obb_bmask_longest_edge",
    }
    wf.detect_step7_yolo_obb_angle_once = MagicMock(return_value=mocks["angle_result"])

    # capture_focus_reference 内部会设置 _focus_reference_ready=True
    original_capture_focus_reference = wf.capture_focus_reference

    def _capture_ref(cycle_index: int) -> bool:
        wf._focus_reference_ready = True
        return True

    mocks["capture_focus_reference"] = MagicMock(side_effect=_capture_ref)
    wf.capture_focus_reference = mocks["capture_focus_reference"]

    mocks["paths"] = {
        "spectrum_csv": "spectrum_test.csv",
        "cycle_dir": "cycle_test",
    }
    wf.build_save_path = MagicMock(return_value=mocks["paths"])

    mocks["acquire_return"] = True
    wf._acquire_and_save_spectrum = MagicMock(return_value=mocks["acquire_return"])

    def _laser_on_side_effect() -> None:
        wf.context["laser_on"] = True

    wf.laser_on = MagicMock(side_effect=_laser_on_side_effect)
    wf.laser_off = MagicMock(side_effect=lambda: wf.context.update({"laser_on": False}))

    mocks["rule_ab_result"] = {
        "ok": True,
        "final_angle": 33.5,
        "final_delta": 3.5,
        "records": [],
        "reason": "ok",
    }
    wf.run_rule_ab_until_angle_delta = MagicMock(return_value=mocks["rule_ab_result"])

    mocks["rule_ac_result"] = {
        "ok": True,
        "reason": "ok",
    }
    wf.run_rule_ac_until_threshold = MagicMock(return_value=mocks["rule_ac_result"])

    mocks["autofocus_result"] = {
        "score": 0.98,
        "triggered": False,
        "autofocus_ok": True,
    }
    wf.run_autofocus_if_needed = MagicMock(return_value=mocks["autofocus_result"])

    wf._is_midrun_recalibration_requested = MagicMock(return_value=False)

    return mocks


def test_config_default_sub_loop_iterations() -> None:
    """默认 sub_loop_iterations_per_cycle 应为 1。"""
    cfg = MeasurementConfig()
    assert cfg.sub_loop_iterations_per_cycle == 1, (
        f"默认值应为 1，实际为 {cfg.sub_loop_iterations_per_cycle}"
    )
    print("PASS: config_default_sub_loop_iterations")


def test_laser_on_idempotent() -> None:
    """laser_on() 在 context['laser_on']=True 时不应重复调用轴运动。"""
    cfg = _make_config()
    wf = _make_workflow(cfg)
    wf.laser_stage = MagicMock()

    with patch.object(wf, "laser_move_and_wait") as mock_move:
        wf.laser_on()
        assert wf.context["laser_on"] is True
        assert mock_move.call_count == 1, "第一次调用应触发轴运动"

        wf.laser_on()
        assert mock_move.call_count == 1, "第二次调用不应重复触发轴运动"
        assert wf.context["laser_on"] is True

    print("PASS: laser_on_idempotent")


def test_run_one_cycle_sub_loop_count() -> None:
    """设置 sub_loop_iterations_per_cycle=N 时，Step 7-14 子循环体应执行 N 次。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(
            save_root=tmpdir,
            sub_loop_iterations_per_cycle=3,
        )
        wf = _make_workflow(cfg)
        mocks = _mock_run_one_cycle_dependencies(wf, sub_loop_count=3)

        result = wf.run_one_cycle(cycle_index=1)

        assert result is True, f"run_one_cycle 应返回 True，实际 {result}"

        # Step 3-6 初始光谱采集应只执行 1 次
        assert wf._acquire_and_save_spectrum.call_count == 1 + 3, (
            f"初始 + 子循环光谱采集应为 4 次，实际 {wf._acquire_and_save_spectrum.call_count}"
        )

        # Step 7 激光打开应在每次子循环开头执行
        assert wf.laser_on.call_count == 3, (
            f"laser_on 应调用 3 次，实际 {wf.laser_on.call_count}"
        )

        # Step 8/9/10 应在每次子循环执行
        assert wf.run_rule_ab_until_angle_delta.call_count == 3
        assert wf.run_rule_ac_until_threshold.call_count == 3
        assert wf.run_autofocus_if_needed.call_count == 3

        # 子循环结束后应关闭激光
        assert wf.laser_off.call_count == 1, (
            f"子循环结束应关闭激光 1 次，实际 {wf.laser_off.call_count}"
        )

        # 聚焦参考图应在第一轮建立
        assert mocks["capture_focus_reference"].call_count == 1

        print("PASS: run_one_cycle_sub_loop_count")


def test_focus_reference_only_first_cycle() -> None:
    """run_one_cycle 在第 1 轮建立聚焦参考图，第 2 轮不再建立。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(
            save_root=tmpdir,
            sub_loop_iterations_per_cycle=1,
        )
        wf = _make_workflow(cfg)
        mocks = _mock_run_one_cycle_dependencies(wf, sub_loop_count=1)

        # 第 1 轮
        wf.run_one_cycle(cycle_index=1)
        assert mocks["capture_focus_reference"].call_count == 1

        # 第 2 轮：参考图已就绪，不应再次建立
        wf.run_one_cycle(cycle_index=2)
        assert mocks["capture_focus_reference"].call_count == 1, (
            "第 2 轮不应再调用 capture_focus_reference"
        )

        print("PASS: focus_reference_only_first_cycle")


def test_sub_loop_zero() -> None:
    """sub_loop_iterations_per_cycle=0 时，只做 Step 1-6，不进入 Step 7-14。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(
            save_root=tmpdir,
            sub_loop_iterations_per_cycle=0,
        )
        wf = _make_workflow(cfg)
        mocks = _mock_run_one_cycle_dependencies(wf, sub_loop_count=0)

        result = wf.run_one_cycle(cycle_index=1)

        assert result is True
        assert wf._acquire_and_save_spectrum.call_count == 1, (
            "只做初始光谱采集 1 次"
        )
        assert wf.laser_on.call_count == 0
        assert wf.run_rule_ab_until_angle_delta.call_count == 0
        assert wf.run_rule_ac_until_threshold.call_count == 0
        assert wf.run_autofocus_if_needed.call_count == 0
        # 安全兜底：即使未进入子循环，若 laser_on=False，laser_off 逻辑上会被跳过
        assert wf.laser_off.call_count == 0

        print("PASS: sub_loop_zero")


def test_save_angle_from_step1() -> None:
    """所有 start_save_cycle_result_async 的 angle_before_result 应来自 Step1 baseline。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(
            save_root=tmpdir,
            sub_loop_iterations_per_cycle=2,
        )
        wf = _make_workflow(cfg)
        mocks = _mock_run_one_cycle_dependencies(wf, sub_loop_count=2)

        captured_calls: List[Dict[str, Any]] = []

        def fake_acquire_and_save(
            cycle_index: int,
            paths: Dict[str, Any],
            angle_result: Dict[str, Any],
            **kwargs: Any,
        ) -> bool:
            captured_calls.append(angle_result)
            return True

        wf._acquire_and_save_spectrum = MagicMock(side_effect=fake_acquire_and_save)

        wf.run_one_cycle(cycle_index=1)

        assert len(captured_calls) == 1 + 2, f"应保存 3 次，实际 {len(captured_calls)}"
        for call in captured_calls:
            assert call.get("ok") is True
            assert call.get("angle_deg") == 30.0
            assert call.get("angle_source") == "yolo_obb_bmask_longest_edge"

        print("PASS: save_angle_from_step1")


if __name__ == "__main__":
    tests = [
        test_config_default_sub_loop_iterations,
        test_laser_on_idempotent,
        test_run_one_cycle_sub_loop_count,
        test_focus_reference_only_first_cycle,
        test_sub_loop_zero,
        test_save_angle_from_step1,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            failed += 1
            print(f"FAIL: {test.__name__}: {e}")
            traceback.print_exc()

    if failed == 0:
        print("\nAll tests passed!")
    else:
        print(f"\n{failed}/{len(tests)} tests failed.")
        sys.exit(1)
