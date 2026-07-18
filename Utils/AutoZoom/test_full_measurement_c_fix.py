"""
测试 7_16 完整测量 C 运行时相关修复。

覆盖：
  1. _find_contours_compat 兼容 OpenCV 3.x/4.x 返回值
  2. _materialize_full_calibration_c_for_runtime 在 mask 为彩色/灰度时都能正确处理
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

import importlib.util

_module_path = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_16.py"
_spec = importlib.util.spec_from_file_location(
    "measurement_workflow_7_16", str(_module_path)
)
_measurement_workflow_7_16 = importlib.util.module_from_spec(_spec)
sys.modules["measurement_workflow_7_16"] = _measurement_workflow_7_16
_spec.loader.exec_module(_measurement_workflow_7_16)  # type: ignore[union-attr]

MeasurementConfig = _measurement_workflow_7_16.MeasurementConfig
CalibrationState = _measurement_workflow_7_16.CalibrationState
MeasurementWorkflow = _measurement_workflow_7_16.MeasurementWorkflow


def _make_minimal_state(static_c_map_dir: Path) -> CalibrationState:
    """构造一个最小可用的 CalibrationState。"""
    state = CalibrationState()
    state.static_c_map_dir = str(static_c_map_dir)
    state.static_c_mask_path = ""
    state.static_c_edge_route_path = ""
    state.angle_num = 0
    state.angle_cw = 0
    state.step9_target_x_px = 500.0
    state.step9_target_y_px = 380.0
    state.step9_color_h = 131
    state.step9_color_s = 164
    state.step9_color_v = 204
    state.step9_color_h_tol = 20
    state.step9_color_s_tol = 40
    state.step9_color_v_tol = 40
    state.step9_color_mode = "include"
    state.step9_color_min_area_px = 100
    state.step9_color_morph_kernel = 3
    state.step9_center_tolerance_px = 5.0
    return state


def test_find_contours_compat() -> None:
    """_find_contours_compat 应在当前 OpenCV 下返回 contours 列表。"""
    image = np.zeros((50, 50), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (40, 40), 255, -1)
    contours = MeasurementWorkflow._find_contours_compat(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    assert isinstance(contours, list), "应返回 list"
    assert len(contours) == 1, f"应检测到 1 个轮廓，实际 {len(contours)}"
    print("PASS: find_contours_compat")


def test_materialize_with_grayscale_mask() -> None:
    """灰度 static_c_mask.png 应能正常生成运行时 C 目录和 meta 文件。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        static_dir = Path(tmpdir) / "static_c"
        static_dir.mkdir()
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (30, 30), (70, 70), 255, -1)
        cv2.imwrite(str(static_dir / "static_c_mask.png"), mask)

        cfg = MeasurementConfig()
        workflow = MeasurementWorkflow(cfg)
        workflow.output_root = Path(tmpdir)
        workflow.run_session_dir = Path(tmpdir) / "session"
        workflow.context: Dict[str, Any] = {}

        state = _make_minimal_state(static_dir)
        new_state = workflow._materialize_full_calibration_c_for_runtime(state)

        runtime_dir = Path(tmpdir) / "session" / "full_calibration_static_c"
        assert runtime_dir.exists(), "应创建运行时 C 目录"
        assert (runtime_dir / "static_c_mask.png").exists()
        assert (runtime_dir / "static_c_meta.json").exists()
        assert (runtime_dir / "static_c_map_meta.json").exists()
        assert (runtime_dir / "static_c_reference.csv").exists()
        assert (runtime_dir / "static_c_pixels_yx.npy").exists()
        assert new_state.static_c_map_dir == str(runtime_dir.resolve())
        print("PASS: materialize_with_grayscale_mask")


def test_materialize_with_color_mask() -> None:
    """彩色 static_c_mask.png（3 通道）应能被自动转灰度并正常生成 meta。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        static_dir = Path(tmpdir) / "static_c"
        static_dir.mkdir()
        mask_gray = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask_gray, (30, 30), (70, 70), 255, -1)
        mask_color = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(static_dir / "static_c_mask.png"), mask_color)

        cfg = MeasurementConfig()
        workflow = MeasurementWorkflow(cfg)
        workflow.output_root = Path(tmpdir)
        workflow.run_session_dir = Path(tmpdir) / "session"
        workflow.context = {}

        state = _make_minimal_state(static_dir)
        new_state = workflow._materialize_full_calibration_c_for_runtime(state)

        runtime_dir = Path(tmpdir) / "session" / "full_calibration_static_c"
        assert runtime_dir.exists()
        assert (runtime_dir / "static_c_meta.json").exists()
        assert (runtime_dir / "static_c_reference.csv").exists()
        # 确认参考 csv 内包含 41*41=1681 个像素点（cv2.rectangle 画满闭合矩形）
        csv_text = (runtime_dir / "static_c_reference.csv").read_text(encoding="utf-8")
        lines = csv_text.strip().splitlines()
        assert len(lines) == 1682, f"应有 1681 个像素行，实际 {len(lines) - 1}"
        print("PASS: materialize_with_color_mask")


def test_materialize_meta_geometry() -> None:
    """生成的 static_c_meta.json 应包含有效的几何信息。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        static_dir = Path(tmpdir) / "static_c"
        static_dir.mkdir()
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.rectangle(mask, (40, 40), (80, 80), 255, -1)
        cv2.imwrite(str(static_dir / "static_c_mask.png"), mask)

        cfg = MeasurementConfig()
        workflow = MeasurementWorkflow(cfg)
        workflow.output_root = Path(tmpdir)
        workflow.run_session_dir = Path(tmpdir) / "session"
        workflow.context = {}

        state = _make_minimal_state(static_dir)
        workflow._materialize_full_calibration_c_for_runtime(state)

        import json
        meta_path = Path(tmpdir) / "session" / "full_calibration_static_c" / "static_c_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta.get("center") is not None
        assert meta.get("bbox") is not None
        assert meta.get("contour") is not None
        assert meta.get("area", 0) > 0
        print("PASS: materialize_meta_geometry")


def main() -> None:
    test_find_contours_compat()
    test_materialize_with_grayscale_mask()
    test_materialize_with_color_mask()
    test_materialize_meta_geometry()
    print("\n完整测量 C 运行时修复测试通过!")


if __name__ == "__main__":
    main()
