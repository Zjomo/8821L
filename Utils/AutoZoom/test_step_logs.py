"""
测试运行完整循环测量过程中每一步(Step)是否都在日志中体现。

通过源码静态扫描验证 run_one_cycle() 方法中每个 Step 都包含：
  self.log("========== Step X：<描述> ==========")
"""

from __future__ import annotations

import sys
from pathlib import Path
import re

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

MAIN_FILE = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_23.py"


def _read_run_one_cycle_source() -> str:
    """读取主文件中 run_one_cycle 方法的源码。"""
    source = MAIN_FILE.read_text(encoding="utf-8")

    # 找到 run_one_cycle 的开始
    start_match = re.search(r"def run_one_cycle\(self, cycle_index: int\) -> bool:", source)
    if not start_match:
        raise RuntimeError("未找到 run_one_cycle 方法")

    # 找到下一个 def（_safe_context_for_json）
    rest = source[start_match.end():]
    next_def_match = re.search(r"^    def \w+", rest, re.MULTILINE)
    if not next_def_match:
        raise RuntimeError("未找到 run_one_cycle 结束位置")

    end_pos = start_match.end() + next_def_match.start()
    return source[start_match.start():end_pos]


def test_run_one_cycle_has_step_logs() -> None:
    """run_one_cycle 方法应包含 Step 1-9 的日志。"""
    run_one_cycle_source = _read_run_one_cycle_source()

    expected_steps = [
        ("Step 1", "Bmask最长边角度检测"),
        ("Step 2", "生成保存路径"),
        ("Step 3", "照明光 OFF"),
        ("Step 4", "LabVIEW 光谱采集"),
        ("Step 5", "照明光 ON"),
        ("Step 5.5", "保存数据"),
        ("Step 6", "打开激光"),
        ("Step 7", "A推动B"),
        ("Step 8", "关闭激光"),
        ("Step 9", "检测颜色区域中心"),
    ]

    for step_label, step_desc in expected_steps:
        pattern = f"========== {step_label}："
        assert pattern in run_one_cycle_source, (
            f"run_one_cycle 中缺少日志：{pattern}（描述应包含「{step_desc}」）"
        )

    print(f"PASS: run_one_cycle_has_step_logs（{len(expected_steps)} 步）")


def test_step_log_format_consistent() -> None:
    """所有 Step 日志应使用统一的 ========== Step X：<描述> ========== 格式。"""
    run_one_cycle_source = _read_run_one_cycle_source()

    # 匹配 ========== Step X：<描述> ==========
    step_pattern = re.compile(
        r'self\.log\("=+\s*Step\s+([\d.]+)[：:]\s*([^"=]+?)\s*=+"\)',
        re.UNICODE,
    )
    matches = step_pattern.findall(run_one_cycle_source)

    assert len(matches) >= 9, f"run_one_cycle 至少应有 9 个 Step 日志，实际 {len(matches)}"

    # 验证 step 编号连续
    expected_order = ["1", "2", "3", "4", "5", "5.5", "6", "7", "8", "9"]
    actual_order = [m[0] for m in matches]

    for i, expected_step in enumerate(expected_order):
        assert actual_order[i] == expected_step, (
            f"Step 编号顺序错误：期望 {expected_step}，实际 {actual_order[i]}"
        )

    print(f"PASS: step_log_format_consistent（{len(matches)} 个 Step 日志格式正确）")


def test_step_logs_cover_full_cycle() -> None:
    """Step 日志应覆盖完整循环测量的所有关键阶段。"""
    run_one_cycle_source = _read_run_one_cycle_source()

    # 检查每个 Step 都出现在 run_one_cycle 中
    required_keywords = {
        "Step 1": "Bmask最长边角度检测",
        "Step 2": "生成保存路径",
        "Step 3": "照明光 OFF",
        "Step 4": "LabVIEW 光谱采集",
        "Step 5": "照明光 ON",
        "Step 5.5": "保存数据",
        "Step 6": "打开激光",
        "Step 7": "A推动B",
        "Step 8": "关闭激光",
        "Step 9": "检测颜色区域中心",
    }

    for step, keyword in required_keywords.items():
        assert keyword in run_one_cycle_source, f"Step {step} 缺少关键词「{keyword}」"

    print(f"PASS: step_logs_cover_full_cycle（{len(required_keywords)} 个关键步骤）")


def test_step_log_via_runtime() -> None:
    """通过 runtime 方式：mock 完整流程，捕获每一步的 log() 调用。"""
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "measurement_workflow_7_23", str(MAIN_FILE)
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["measurement_workflow_7_23"] = _module
    _spec.loader.exec_module(_module)  # type: ignore[union-attr]

    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    cfg = MeasurementConfig()
    cfg.hardware_mode = "virtual"
    cfg.max_cycles = 1

    wf = MeasurementWorkflow(cfg)

    # 用 list 收集所有 log 调用
    log_calls: list = []
    wf.on_log = lambda msg: log_calls.append(msg)
    # LogManager 会先于 on_log 被写入文件，要确保 LogManager 不会写两次
    if wf._log_manager is not None:
        wf._log_manager.on_log = None

    # mock 关键方法
    wf._ensure_rule_ab_follower_with_startup_retry = lambda label: None
    wf.apply_runtime_rule_ab_params_to_follower = lambda follower, reason: None
    wf._ensure_rule_ab_feature_tracker_installed = lambda follower, state: None
    wf._get_loaded_calibration_state = lambda: None
    wf._is_midrun_recalibration_requested = lambda: False

    # 关键：mock 角度检测
    wf.detect_step7_yolo_obb_angle_once = lambda **kwargs: {
        "ok": True, "angle_deg": 10.0, "reason": "test", "angle_source": "test"
    }

    # mock 光谱采集
    wf.request_labview_spectrum = lambda idx: {"ok": True, "num_points": 1024}

    # mock 光照
    wf.light_off = lambda: None
    wf.light_on = lambda: None

    # mock 激光
    wf.laser_on = lambda: None
    wf.laser_off = lambda: None

    # mock RuleAB / RuleAC
    wf.run_rule_ab_until_angle_delta = lambda **kwargs: {
        "ok": True, "final_angle": 12.0, "final_delta": 2.0, "records": [], "reason": "ok"
    }
    wf.run_rule_ac_until_threshold = lambda: {
        "ok": True, "final_delta": 0.0, "reason": "ok"
    }

    # mock 保存
    wf.start_save_cycle_result_async = lambda **kwargs: None
    wf.build_save_path = lambda idx: {"spectrum_csv": f"test_{idx}.csv"}

    # mock 跳过判断
    wf._is_nonfatal_bmask_angle_failure_reason = lambda reason: False

    # 模拟 stop
    wf.stop_requested = False

    # 调用 run_one_cycle（捕获异常以防完整流程缺东西）
    try:
        wf.run_one_cycle(1)
    except Exception:
        # 即便异常，只要日志已记录就 OK
        pass

    # 收集到的日志中应包含 Step 1-9
    all_logs = "\n".join(log_calls)

    expected_step_logs = [
        "========== Step 1：",
        "========== Step 2：",
        "========== Step 3：",
        "========== Step 4：",
        "========== Step 5：",
        "========== Step 5.5：",
        "========== Step 6：",
        "========== Step 7：",
        "========== Step 8：",
        "========== Step 9：",
    ]

    missing = []
    for step_log in expected_step_logs:
        if step_log not in all_logs:
            missing.append(step_log)

    assert len(missing) == 0, f"runtime 流程中缺少日志：{missing}"

    print(f"PASS: step_log_via_runtime（{len(expected_step_logs)} 个 Step 日志全部输出）")


if __name__ == "__main__":
    tests = [
        test_run_one_cycle_has_step_logs,
        test_step_log_format_consistent,
        test_step_logs_cover_full_cycle,
        test_step_log_via_runtime,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            import traceback
            failed += 1
            print(f"FAIL: {test.__name__}: {e}")
            traceback.print_exc()

    if failed == 0:
        print(f"\nAll {len(tests)} tests passed!")
    else:
        print(f"\n{failed}/{len(tests)} tests failed.")
        sys.exit(1)