"""
测试 0_measurement_workflow_real_virtual_same_detection_7_25.py 中补焦参考图的全屏 ROI 逻辑。

覆盖：
  1. _capture_full_screen_frame 返回全屏尺寸图像
  2. _select_focus_roi_interactively 使用全屏底图，saf_capture_area 同步为全屏
  3. 补焦 ROI/截图区域与 capture_area / focus_roi 完全隔离
  4. _make_focus_config 使用 saf_capture_area / saf_focus_roi
  5. capture_focus_reference 在全屏 ROI 下能建立参考并保存到文件
  6. begin_new_run_session 每次运行都会重置聚焦状态，强制重新选择 ROI
  7. ROI 选择窗口置顶、移动、彻底销毁，避免界面残留
  8. 过小选区被拒绝，stop_requested 能中断选择
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

import pyautogui


def _import_7_25():
    """动态导入 7_25 主模块。"""
    _module_path = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_25.py"
    _spec = importlib.util.spec_from_file_location(
        "measurement_workflow_7_25", str(_module_path)
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["measurement_workflow_7_25"] = _module
    _spec.loader.exec_module(_module)  # type: ignore[union-attr]
    return _module


def test_capture_full_screen_frame_returns_full_screen_size() -> None:
    """_capture_full_screen_frame 应返回全屏尺寸图像。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        wf = MeasurementWorkflow(cfg)

        # mock pyautogui.screenshot 返回 1920x1080 全屏图像
        fake_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fake_pil = MagicMock()
        fake_pil.__array__ = lambda: fake_screen

        with patch.object(pyautogui, "screenshot", return_value=fake_pil):
            image_rgb = wf._capture_full_screen_frame()

        assert image_rgb is not None
        assert image_rgb.shape == (1080, 1920, 3)
        print("PASS: capture_full_screen_frame_returns_full_screen_size")


def test_select_focus_roi_interactively_uses_full_screen() -> None:
    """_select_focus_roi_interactively 应使用全屏底图并设置全屏 saf_capture_area。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    # 构造一个 1920x1080 的全屏底图
    fake_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        # 初始 capture_area 与 focus_roi，用于验证不被修改
        cfg.capture_area = (100, 100, 800, 600)
        cfg.focus_roi = (50, 50, 200, 200)
        wf = MeasurementWorkflow(cfg)

        # 用于捕获 setMouseCallback 注册的回调
        mouse_callback = [None]

        def fake_set_mouse_callback(name, callback):
            mouse_callback[0] = callback

        # 用户在 0.85 缩放窗口中框选 (200,100)->(425,340)
        # 反缩放后约为 (235,117)->(500,400)
        fake_events = [
            (1, 200, 100, 0, None),   # LBUTTONDOWN
            (0, 425, 340, 0, None),   # MOUSEMOVE
            (4, 425, 340, 0, None),   # LBUTTONUP
        ]

        def fake_imshow(name, img):
            # 在每次 imshow 时按顺序触发鼠标事件
            if mouse_callback[0] is not None:
                for event in fake_events:
                    mouse_callback[0](event[0], event[1], event[2], event[3], event[4])

        def fake_waitKey(_):
            return 13  # Enter

        with patch.object(wf, "_capture_full_screen_frame", return_value=fake_screen):
            with patch.object(_module.cv2, "namedWindow", MagicMock()) as mock_named:
                with patch.object(_module.cv2, "resizeWindow", MagicMock()):
                    with patch.object(_module.cv2, "setWindowProperty", MagicMock()) as mock_topmost:
                        with patch.object(_module.cv2, "moveWindow", MagicMock()) as mock_move:
                            with patch.object(_module.cv2, "setMouseCallback", fake_set_mouse_callback):
                                with patch.object(_module.cv2, "imshow", fake_imshow):
                                    with patch.object(_module.cv2, "waitKey", fake_waitKey):
                                        with patch.object(_module.cv2, "destroyWindow", MagicMock()) as mock_destroy:
                                            with patch.object(_module.cv2, "destroyAllWindows", MagicMock()) as mock_destroy_all:
                                                result = wf._select_focus_roi_interactively()

        assert result == True, "ROI 选择应成功"
        assert wf.cfg.saf_capture_area == (0, 0, 1920, 1080), (
            f"saf_capture_area 应为全屏：{wf.cfg.saf_capture_area}"
        )

        # 验证 ROI 在合理范围内
        x, y, w, h = wf.cfg.saf_focus_roi
        assert x >= 0 and y >= 0 and w > 0 and h > 0
        assert x + w <= 1920 and y + h <= 1080

        # 关键：capture_area 与 focus_roi 不能被修改
        assert wf.cfg.capture_area == (100, 100, 800, 600)
        assert wf.cfg.focus_roi == (50, 50, 200, 200)

        # 验证窗口置顶、移动、彻底销毁均被调用
        mock_topmost.assert_called_once()
        mock_move.assert_called_once()
        mock_destroy.assert_called_once()
        mock_destroy_all.assert_called_once()

        print("PASS: select_focus_roi_interactively_uses_full_screen")


def test_select_focus_roi_rejects_tiny_selection() -> None:
    """拖拽选区过小时应拒绝确认，要求重新选择。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    fake_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        wf = MeasurementWorkflow(cfg)

        mouse_callback = [None]

        def fake_set_mouse_callback(name, callback):
            mouse_callback[0] = callback

        # 用户单击几乎未拖动，选区只有 1x1
        fake_events = [
            (1, 200, 100, 0, None),   # LBUTTONDOWN
            (4, 200, 100, 0, None),   # LBUTTONUP（无拖动）
        ]

        call_count = [0]

        def fake_imshow(name, img):
            call_count[0] += 1
            if mouse_callback[0] is not None and call_count[0] <= 2:
                for event in fake_events:
                    mouse_callback[0](event[0], event[1], event[2], event[3], event[4])

        def fake_waitKey(_):
            # 第一次返回 Enter，由于选区过小而 continue；第二次返回 ESC 取消
            if call_count[0] <= 2:
                return 13
            return 27

        with patch.object(wf, "_capture_full_screen_frame", return_value=fake_screen):
            with patch.object(_module.cv2, "namedWindow", MagicMock()):
                with patch.object(_module.cv2, "resizeWindow", MagicMock()):
                    with patch.object(_module.cv2, "setWindowProperty", MagicMock()):
                        with patch.object(_module.cv2, "moveWindow", MagicMock()):
                            with patch.object(_module.cv2, "setMouseCallback", fake_set_mouse_callback):
                                with patch.object(_module.cv2, "imshow", fake_imshow):
                                    with patch.object(_module.cv2, "waitKey", fake_waitKey):
                                        with patch.object(_module.cv2, "destroyWindow", MagicMock()):
                                            with patch.object(_module.cv2, "destroyAllWindows", MagicMock()):
                                                result = wf._select_focus_roi_interactively()

        assert result == False, "过小选区应被取消"
        print("PASS: select_focus_roi_rejects_tiny_selection")


def test_select_focus_roi_respects_stop_request() -> None:
    """循环过程中收到 stop_requested 应立即退出并返回 False。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    fake_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        wf = MeasurementWorkflow(cfg)
        wf.stop_requested = True

        with patch.object(wf, "_capture_full_screen_frame", return_value=fake_screen):
            with patch.object(_module.cv2, "namedWindow", MagicMock()):
                with patch.object(_module.cv2, "resizeWindow", MagicMock()):
                    with patch.object(_module.cv2, "setWindowProperty", MagicMock()):
                        with patch.object(_module.cv2, "moveWindow", MagicMock()):
                            with patch.object(_module.cv2, "setMouseCallback", MagicMock()):
                                with patch.object(_module.cv2, "imshow", MagicMock()):
                                    with patch.object(_module.cv2, "waitKey", return_value=-1):
                                        with patch.object(_module.cv2, "destroyWindow", MagicMock()):
                                            with patch.object(_module.cv2, "destroyAllWindows", MagicMock()):
                                                result = wf._select_focus_roi_interactively()

        assert result == False, "stop_requested 时应返回 False"
        print("PASS: select_focus_roi_respects_stop_request")


def test_make_focus_config_uses_saf_areas() -> None:
    """_make_focus_config 应使用 saf_capture_area / saf_focus_roi，而非 capture_area。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.capture_area = (100, 100, 800, 600)
        cfg.focus_roi = (50, 50, 200, 200)
        cfg.saf_capture_area = (0, 0, 1920, 1080)
        cfg.saf_focus_roi = (300, 200, 400, 300)
        wf = MeasurementWorkflow(cfg)

        focus_cfg = wf._make_focus_config()
        actual_capture = tuple(int(v) for v in focus_cfg.capture_area)
        actual_focus = tuple(int(v) for v in focus_cfg.focus_roi)
        assert actual_capture == (0, 0, 1920, 1080), f"实际 capture_area={actual_capture}"
        assert actual_focus == (300, 200, 400, 300), f"实际 focus_roi={actual_focus}"

        print("PASS: make_focus_config_uses_saf_areas")


def test_focus_area_isolation_from_calibration() -> None:
    """修改 saf_capture_area / saf_focus_roi 不影响标定/角度检测区域。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.capture_area = (100, 100, 800, 600)
        cfg.focus_roi = (50, 50, 200, 200)
        wf = MeasurementWorkflow(cfg)

        original_capture_area = tuple(cfg.capture_area)
        original_focus_roi = tuple(cfg.focus_roi)

        # 模拟全屏 ROI 选择后的赋值
        wf.cfg.saf_capture_area = (0, 0, 1920, 1080)
        wf.cfg.saf_focus_roi = (300, 200, 400, 300)

        assert wf.cfg.capture_area == original_capture_area
        assert wf.cfg.focus_roi == original_focus_roi
        assert wf.cfg.saf_capture_area == (0, 0, 1920, 1080)
        assert wf.cfg.saf_focus_roi == (300, 200, 400, 300)

        print("PASS: focus_area_isolation_from_calibration")


def test_capture_focus_reference_with_full_screen_roi() -> None:
    """在全屏 ROI 配置下，capture_focus_reference 应能建立参考。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.saf_capture_area = (0, 0, 1920, 1080)
        cfg.saf_focus_roi = (300, 200, 400, 300)
        wf = MeasurementWorkflow(cfg)
        # 跳过强制 ROI 选择窗口，直接验证参考建立逻辑
        wf._focus_roi_selected = True

        # 构造全屏测试图
        fake_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fake_screen[200:500, 300:700] = 255  # ROI 区域为白色

        # mock _capture_current_focus_frame 返回全屏图
        with patch.object(wf, "_capture_current_focus_frame", return_value=fake_screen):
            with patch.object(_module.cv2, "imshow", MagicMock()):
                with patch.object(_module.cv2, "waitKey", return_value=1):
                    result = wf.capture_focus_reference(cycle_index=1)

        assert result == True, "应成功建立聚焦参考"
        assert wf._focus_reference_ready == True
        assert wf._focus_reference_image is not None

        print("PASS: capture_focus_reference_with_full_screen_roi")


def test_begin_new_run_session_resets_focus_state() -> None:
    """每次新运行会话开始时，必须重置补焦 ROI 与参考图状态，确保重新选择。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        wf = MeasurementWorkflow(cfg)

        # 模拟上一轮运行已经完成过 ROI 选择和参考建立
        wf._focus_roi_selected = True
        wf._focus_reference_ready = True
        wf._focus_reference_image = np.zeros((100, 100, 3), dtype=np.uint8)

        wf.begin_new_run_session()

        assert wf._focus_roi_selected == False, "begin_new_run_session 应重置 _focus_roi_selected"
        assert wf._focus_reference_ready == False, "begin_new_run_session 应重置 _focus_reference_ready"
        assert wf._focus_reference_image is None, "begin_new_run_session 应清空 _focus_reference_image"

        print("PASS: begin_new_run_session_resets_focus_state")


def test_capture_focus_reference_requires_roi_selection() -> None:
    """用户未完成 ROI 选择时，capture_focus_reference 应阻塞直到选择完成。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        wf = MeasurementWorkflow(cfg)

        # 前一次取消，第二次成功
        select_returns = [False, True]
        with patch.object(
            wf, "_select_focus_roi_interactively", side_effect=select_returns
        ) as mock_select:
            with patch.object(
                wf, "_show_roi_selection_required_dialog", MagicMock()
            ) as mock_dialog:
                fake_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
                with patch.object(
                    wf, "_capture_current_focus_frame", return_value=fake_screen
                ):
                    with patch.object(_module.cv2, "imshow", MagicMock()):
                        with patch.object(_module.cv2, "waitKey", return_value=1):
                            result = wf.capture_focus_reference(cycle_index=1)

        assert result == True, "最终应成功建立参考"
        assert wf._focus_roi_selected == True
        assert wf._focus_reference_ready == True
        assert mock_select.call_count == 2
        assert mock_dialog.call_count == 1

        print("PASS: capture_focus_reference_requires_roi_selection")


def test_capture_focus_reference_returns_false_when_stopped() -> None:
    """收到停止请求时，阻塞循环应立即退出并返回 False。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        wf = MeasurementWorkflow(cfg)
        wf.stop_requested = True

        with patch.object(
            wf, "_select_focus_roi_interactively", MagicMock()
        ) as mock_select:
            with patch.object(
                wf, "_show_roi_selection_required_dialog", MagicMock()
            ) as mock_dialog:
                result = wf.capture_focus_reference(cycle_index=1)

        assert result == False, "停止请求时应返回 False"
        assert wf._focus_reference_ready == False
        assert mock_select.called == False
        assert mock_dialog.called == False

        print("PASS: capture_focus_reference_returns_false_when_stopped")


def test_run_one_cycle_stops_when_focus_reference_fails() -> None:
    """capture_focus_reference 失败时，run_one_cycle 应停止测量。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_root = tmpdir
        wf = MeasurementWorkflow(cfg)

        # 让 Step1 角度检测直接通过
        with patch.object(
            wf, "_ensure_rule_ab_follower_with_startup_retry", return_value=MagicMock()
        ):
            with patch.object(wf, "_ensure_rule_ab_feature_tracker_installed"):
                with patch.object(
                    wf,
                    "detect_step7_yolo_obb_angle_once",
                    return_value={"ok": True, "angle_deg": 30.0, "angle_source": "test"},
                ):
                    with patch.object(
                        wf, "capture_focus_reference", return_value=False
                    ) as mock_capture:
                        result = wf.run_one_cycle(cycle_index=1)

        assert result == False, "参考建立失败时应返回 False"
        assert wf.stop_requested == True
        mock_capture.assert_called_once_with(1)

        print("PASS: run_one_cycle_stops_when_focus_reference_fails")


if __name__ == "__main__":
    tests = [
        test_capture_full_screen_frame_returns_full_screen_size,
        test_select_focus_roi_interactively_uses_full_screen,
        test_select_focus_roi_rejects_tiny_selection,
        test_select_focus_roi_respects_stop_request,
        test_make_focus_config_uses_saf_areas,
        test_focus_area_isolation_from_calibration,
        test_capture_focus_reference_with_full_screen_roi,
        test_begin_new_run_session_resets_focus_state,
        test_capture_focus_reference_requires_roi_selection,
        test_capture_focus_reference_returns_false_when_stopped,
        test_run_one_cycle_stops_when_focus_reference_fails,
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
