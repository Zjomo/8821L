"""
测试 0_measurement_workflow_real_virtual_same_detection_7_16.py 新增功能：
1. UI 面板参数实时同步到 workflow 配置；
2. 完整循环测量采集补焦参考基准图时自动弹出实时窗口。
"""

from __future__ import annotations

import sys
import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

# 文件名为数字开头，需要用 spec 方式加载并注册到 sys.modules
_MODULE_NAME = "measurement_workflow_real_virtual_same_detection_7_16"
_FILE_PATH = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_16.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _FILE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


MODULE = _load_module()
MeasurementWorkflowGUI = MODULE.MeasurementWorkflowGUI
MeasurementWorkflow = MODULE.MeasurementWorkflow
MeasurementConfig = MODULE.MeasurementConfig


def test_ui_parameter_real_time_update():
    """修改 UI 参数后，非运行状态下 workflow.cfg 应实时更新。"""
    root = MODULE.tk.Tk()
    try:
        gui = MeasurementWorkflowGUI(root)
        cfg = MeasurementConfig()
        gui.workflow = MeasurementWorkflow(cfg, on_log=lambda *_: None, on_update=lambda *_: None)

        original = int(gui.max_cycles_var.get())
        new_value = original + 5
        gui.max_cycles_var.set(new_value)
        root.update_idletasks()

        assert gui.workflow.cfg.max_cycles == new_value, (
            f"实时更新失败：期望 {new_value}，实际 {gui.workflow.cfg.max_cycles}"
        )
        print(f"PASS: ui_parameter_real_time_update ({original} -> {new_value})")
    finally:
        root.destroy()


def test_ui_parameter_update_not_during_measurement():
    """测量运行中修改 UI 参数，本轮不应立即同步（由下一轮开始前统一读取）。"""
    root = MODULE.tk.Tk()
    try:
        gui = MeasurementWorkflowGUI(root)
        cfg = MeasurementConfig()
        wf = MeasurementWorkflow(cfg, on_log=lambda *_: None, on_update=lambda *_: None)
        wf.is_measuring = True
        gui.workflow = wf

        original = int(gui.max_cycles_var.get())
        gui.max_cycles_var.set(original + 5)
        root.update_idletasks()

        # 运行中不立即同步，因此保持原值
        assert gui.workflow.cfg.max_cycles == original, (
            f"运行中不应实时同步：期望 {original}，实际 {gui.workflow.cfg.max_cycles}"
        )
        print(f"PASS: ui_parameter_update_not_during_measurement")
    finally:
        root.destroy()


def test_capture_focus_reference_shows_popup():
    """capture_focus_reference 成功时应调用 cv2.imshow 弹出实时窗口。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = MeasurementConfig()
        wf = MeasurementWorkflow(cfg, on_log=lambda *_: None, on_update=lambda *_: None)
        wf.output_root = Path(tmp)

        # 伪造 Focus 组件与截图
        wf._focus_metrics_calc = MagicMock()
        wf._focus_metrics_calc.cfg.focus_roi = (0, 0, 100, 100)
        wf._focus_scorer = MagicMock()
        wf._focus_scorer.build_reference_from_image.return_value = {
            "full_rgb": np.zeros((100, 100, 3), dtype=np.uint8),
            "roi_metric_ref": {},
        }

        test_image = np.zeros((200, 200, 3), dtype=np.uint8)

        with patch.object(wf, "_ensure_focus_components") as mock_ensure, \
             patch.object(wf, "_capture_current_focus_frame", return_value=test_image), \
             patch("measurement_workflow_real_virtual_same_detection_7_16.cv2.imshow") as mock_imshow, \
             patch("measurement_workflow_real_virtual_same_detection_7_16.cv2.waitKey", return_value=-1):
            result = wf.capture_focus_reference(cycle_index=1)

        assert result is True, "capture_focus_reference 应返回 True"
        mock_ensure.assert_called_once()
        mock_imshow.assert_called_once()
        args, _ = mock_imshow.call_args
        assert args[0] == "Focus Reference Baseline Image", f"窗口名称错误：{args[0]}"
        assert args[1].shape == test_image.shape
        print("PASS: capture_focus_reference_shows_popup")


if __name__ == "__main__":
    test_ui_parameter_real_time_update()
    test_ui_parameter_update_not_during_measurement()
    test_capture_focus_reference_shows_popup()
    print("\n所有测试通过")
