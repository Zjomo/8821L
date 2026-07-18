"""
测试 SpectrumAutofocusLoop 光谱补焦循环模块。

覆盖：
  1. ROI 字符串解析（合法 / 非法）
  2. ROI 超出图像边界时被正确裁剪
  3. 以单张截图建立 Focus 参考基线
  4. 当前 FocusScore 已达标时的被动补焦行为
  5. 当前 FocusScore 未达标时的被动补焦恢复
  6. 外部停止信号能中断被动补焦循环
  7. 光谱测量光照顺序：OFF → spectrum → ON → save
  8. 被动补焦结束后 CSV/PNG FocusScore 曲线文件被创建

所有测试不依赖真实硬件，使用 MagicMock 与 FocusSimulator。
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, call

import numpy as np

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from Focus.config import AutofocusConfig
from spectrum_autofocus_loop import SpectrumAutofocusLoop

try:
    import matplotlib

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


def _make_cfg() -> AutofocusConfig:
    """构造测试用 AutofocusConfig，避免 screen_region 触发跨线程截图问题。"""
    return AutofocusConfig(
        capture_mode="usb_camera",
        capture_area=(0, 0, 400, 400),
        focus_roi=(100, 100, 200, 200),
        z_enabled=False,
        autofocus_enabled=True,
    )


def _make_test_image(h: int = 400, w: int = 400) -> np.ndarray:
    """构造一张有明确边缘的测试图，便于计算聚焦指标。"""
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = [255, 255, 255]
    return image


def _make_loop(
    cfg: AutofocusConfig,
    output_dir: Path,
    workflow: Any = None,
    should_stop: Any = None,
) -> SpectrumAutofocusLoop:
    """构造 SpectrumAutofocusLoop，并把间隔设为 0 以加速测试。"""
    loop = SpectrumAutofocusLoop(
        workflow=workflow or MagicMock(),
        cfg=cfg,
        output_dir=str(output_dir),
        should_stop=should_stop,
    )
    loop.interval_s = 0.0
    return loop


def test_parse_focus_roi_valid() -> None:
    """合法 ROI 字符串应被正确解析。"""
    assert SpectrumAutofocusLoop._parse_roi_text("10,20,300,400") == (
        10,
        20,
        300,
        400,
    )
    assert SpectrumAutofocusLoop._parse_roi_text("  5 , 6 , 100 , 200 ") == (
        5,
        6,
        100,
        200,
    )
    print("PASS: parse_focus_roi_valid")


def test_parse_focus_roi_invalid() -> None:
    """非法 ROI 字符串应抛出 ValueError。"""
    invalid_inputs = ["1,2,3", "a,b,c,d", "1,2,3,", ""]
    for text in invalid_inputs:
        try:
            SpectrumAutofocusLoop._parse_roi_text(text)
            assert False, f"输入 '{text}' 应抛出 ValueError"
        except ValueError:
            pass
    print("PASS: parse_focus_roi_invalid")


def test_roi_clamping() -> None:
    """当 cfg.focus_roi 超出截图边界时，build_reference_from_single_capture 应自动裁剪。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = _make_cfg()
        cfg.focus_roi = (250, 250, 300, 300)  # 超出 400x400
        loop = _make_loop(cfg, tmp_path)
        loop._reference_image = _make_test_image(400, 400)

        ref = loop.build_reference_from_single_capture()
        x, y, w, h = ref["roi"]

        assert x + w <= 400, f"ROI 宽度越界：{x}+{w}"
        assert y + h <= 400, f"ROI 高度越界：{y}+{h}"
        assert loop._reference_ready
        print(f"PASS: roi_clamping -> {ref['roi']}")


def test_build_reference_from_single_capture() -> None:
    """以单张图像建立参考后，scorer 的参考基线应就绪。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = _make_cfg()
        loop = _make_loop(cfg, tmp_path)
        loop._reference_image = _make_test_image(400, 400)

        ref = loop.build_reference_from_single_capture()

        assert loop._reference_ready, "loop._reference_ready 应为 True"
        assert loop._scorer is not None
        assert loop._scorer.focus_reference_ready, "scorer 参考基线未就绪"
        assert "roi_metric_ref" in ref
        assert ref["capture_count"] == 1
        print("PASS: build_reference_from_single_capture")


def test_passive_autofocus_skips_when_score_above_trigger() -> None:
    """FocusScore 持续高于触发阈值时，只需连续达标 5 次即停止，不触发额外补焦。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = _make_cfg()
        loop = _make_loop(cfg, tmp_path)
        loop._reference_ready = True

        controller = MagicMock()
        controller.check_and_autofocus.return_value = {"focus_score_ratio": 0.96}
        loop._create_controller = lambda: (controller, None)

        result = loop.run_passive_autofocus_detection(cycle_index=1)

        assert controller.check_and_autofocus.call_count == 5
        assert result.get("focus_score_ratio") == 0.96
        print("PASS: passive_autofocus_skips_when_score_above_trigger")


def test_passive_autofocus_triggers_recovery() -> None:
    """FocusScore 先低于阈值再恢复到阈值以上，被动补焦应最终停止且分数达标。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = _make_cfg()
        loop = _make_loop(cfg, tmp_path)
        loop._reference_ready = True

        controller = MagicMock()
        controller.check_and_autofocus.side_effect = [
            {"focus_score_ratio": 0.85},
            {"focus_score_ratio": 0.88},
            {"focus_score_ratio": 0.96},
            {"focus_score_ratio": 0.97},
            {"focus_score_ratio": 0.98},
            {"focus_score_ratio": 0.99},
            {"focus_score_ratio": 0.99},
        ]
        loop._create_controller = lambda: (controller, None)

        result = loop.run_passive_autofocus_detection(cycle_index=1)

        assert result.get("focus_score_ratio") >= 0.95
        assert controller.check_and_autofocus.call_count == 7
        print("PASS: passive_autofocus_triggers_recovery")


def test_stop_requested_interrupts_loop() -> None:
    """外部 should_stop 返回 True 时，被动补焦循环应立即优雅退出。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = _make_cfg()

        stop_sequence = [False, False, True]
        call_log: List[bool] = []

        def should_stop() -> bool:
            idx = len(call_log)
            result = stop_sequence[idx] if idx < len(stop_sequence) else True
            call_log.append(result)
            return result

        loop = _make_loop(cfg, tmp_path, should_stop=should_stop)
        loop._reference_ready = True

        controller = MagicMock()
        controller.check_and_autofocus.return_value = {"focus_score_ratio": 0.85}
        loop._create_controller = lambda: (controller, None)

        loop.run_passive_autofocus_detection(cycle_index=1)

        # 第 3 轮循环开头检测到停止，因此只调用 2 次 check_and_autofocus
        assert controller.check_and_autofocus.call_count == 2
        print("PASS: stop_requested_interrupts_loop")


def test_measure_spectrum_once_light_sequence() -> None:
    """验证光谱测量调用顺序：关照明 → 请求光谱 → 开照明 → 异步保存。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = _make_cfg()

        workflow = MagicMock()
        workflow.request_labview_spectrum.return_value = {
            "num_points": 1024,
            "raw_peak": 500.0,
            "fit_peak": 501.0,
        }
        workflow.build_save_path.return_value = {
            "spectrum_csv": str(tmp_path / "spectrum_0001.csv")
        }

        loop = _make_loop(cfg, tmp_path, workflow=workflow)
        loop.measure_spectrum_once(cycle_index=1)

        names = [c[0] for c in workflow.mock_calls]
        assert "light_off" in names
        assert "request_labview_spectrum" in names
        assert "light_on" in names
        assert "start_save_cycle_result_async" in names

        off_idx = names.index("light_off")
        spectrum_idx = names.index("request_labview_spectrum")
        on_idx = names.index("light_on")
        save_idx = names.index("start_save_cycle_result_async")

        assert off_idx < spectrum_idx < on_idx < save_idx, (
            f"调用顺序错误：{names}"
        )
        workflow.light_on.assert_called_once()
        workflow.light_off.assert_called_once()
        workflow.request_labview_spectrum.assert_called_once_with(1)
        print("PASS: measure_spectrum_once_light_sequence")


def test_focus_score_curve_saved() -> None:
    """被动补焦结束后应生成 FocusScore 历史 CSV，并在 matplotlib 可用时生成 PNG 曲线。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = _make_cfg()
        loop = _make_loop(cfg, tmp_path)
        loop._reference_ready = True

        controller = MagicMock()
        controller.check_and_autofocus.side_effect = [
            {"focus_score_ratio": 0.96}
        ] * 5
        loop._create_controller = lambda: (controller, None)

        loop.run_passive_autofocus_detection(cycle_index=1)

        csv_path = tmp_path / "focus_score_history_cycle_0001.csv"
        png_path = tmp_path / "focus_score_curve_cycle_0001.png"

        assert csv_path.exists(), "FocusScore CSV 未生成"
        content = csv_path.read_text(encoding="utf-8-sig")
        assert "focus_score_ratio" in content
        assert "0.96" in content

        if MATPLOTLIB_AVAILABLE:
            assert png_path.exists(), "FocusScore PNG 曲线未生成"

        print("PASS: focus_score_curve_saved")


def test_roi_to_screen_capture_area() -> None:
    """验证相对 ROI 到屏幕截图区域的坐标转换。"""
    screen_area = (100, 200, 800, 600)
    relative_roi = (50, 60, 300, 400)
    new_capture, new_focus = SpectrumAutofocusLoop.roi_to_screen_capture_area(
        screen_area, relative_roi
    )
    assert new_capture == (150, 260, 300, 400), f"new_capture_area 错误：{new_capture}"
    assert new_focus == (0, 0, 300, 400), f"new_focus_roi 错误：{new_focus}"
    print("PASS: roi_to_screen_capture_area")


def test_select_saf_roi_source_uses_sync() -> None:
    """通过源码检查确认 select_saf_roi 已集成 ROI 同步逻辑。"""
    main_source = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_6.25.py"
    assert main_source.exists(), f"主程序源码不存在：{main_source}"
    source = main_source.read_text(encoding="utf-8")

    required_patterns = [
        "SpectrumAutofocusLoop.roi_to_screen_capture_area(",
        "self.capture_area_var.set(",
        "loop.cfg.capture_area = new_capture_area",
        "loop.cfg.focus_roi = new_focus_roi",
        "wf.cfg.capture_area = new_capture_area",
        "loop._reference_image = None",
        "loop._reference_ready = False",
    ]

    for needle in required_patterns:
        assert needle in source, f"主程序 select_saf_roi 中未找到：{needle}"

    print("PASS: select_saf_roi_source_uses_sync")


if __name__ == "__main__":
    tests = [
        test_parse_focus_roi_valid,
        test_parse_focus_roi_invalid,
        test_roi_clamping,
        test_build_reference_from_single_capture,
        test_passive_autofocus_skips_when_score_above_trigger,
        test_passive_autofocus_triggers_recovery,
        test_stop_requested_interrupts_loop,
        test_measure_spectrum_once_light_sequence,
        test_focus_score_curve_saved,
        test_roi_to_screen_capture_area,
        test_select_saf_roi_source_uses_sync,
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
