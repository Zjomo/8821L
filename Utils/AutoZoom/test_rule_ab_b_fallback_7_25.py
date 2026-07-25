"""
测试 0_measurement_workflow_real_virtual_same_detection_7_25.py 中
RuleAB 初始化时 B 正点缺失的兼容逻辑。

覆盖：
  1. B 点缺失且 A 点存在时，默认 B 点基于 A 点偏移生成
  2. B 点和 A 点都缺失时，默认 B 点使用图像中心
  3. 标定包已有 B 点时，保持原有值不变
  4. 默认 B 点坐标被限制在图像边界内
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))


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


def _make_state(_module, a_points=None, b_points=None):
    """构造 CalibrationState，仅设置 A/B 正点。"""
    state = _module.CalibrationState()
    if a_points is not None:
        state.rule_ab_a_positive_points = [[float(x), float(y)] for x, y in a_points]
    if b_points is not None:
        state.rule_ab_b_positive_points = [[float(x), float(y)] for x, y in b_points]
    return state


def test_default_b_generated_from_a_point() -> None:
    """B 点缺失且 A 点存在时，默认 B 点应基于 A 点偏移生成。"""
    _module = _import_7_25()
    cfg = _module.MeasurementConfig()
    wf = _module.MeasurementWorkflow(cfg)

    fake_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    with patch.object(wf, "_capture_current_rule_ab_frame", return_value=fake_frame):
        state = _make_state(_module, a_points=[(100, 200)], b_points=[])
        follower = MagicMock()
        follower.initialize_abc_with_calibration = MagicMock()

        wf._call_initialize_abc_with_loaded_calibration(follower, state)

    b_points = state.rule_ab_b_positive_points
    assert len(b_points) == 1, f"应生成 1 个默认 B 点，实际 {len(b_points)}"
    bx, by = b_points[0]
    assert bx == 200.0, f"默认 B 点 x 应为 200.0，实际 {bx}"
    assert by == 200.0, f"默认 B 点 y 应为 200.0，实际 {by}"
    follower.initialize_abc_with_calibration.assert_called_once()
    print("PASS: default_b_generated_from_a_point")


def test_default_b_uses_image_center_when_no_a() -> None:
    """B 点和 A 点都缺失时，默认 B 点应使用图像中心。"""
    _module = _import_7_25()
    cfg = _module.MeasurementConfig()
    wf = _module.MeasurementWorkflow(cfg)

    fake_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    with patch.object(wf, "_capture_current_rule_ab_frame", return_value=fake_frame):
        state = _make_state(_module, a_points=[], b_points=[])
        follower = MagicMock()
        follower.initialize_abc_with_calibration = MagicMock()

        wf._call_initialize_abc_with_loaded_calibration(follower, state)

    b_points = state.rule_ab_b_positive_points
    assert len(b_points) == 1, f"应生成 1 个默认 B 点，实际 {len(b_points)}"
    bx, by = b_points[0]
    assert bx == 400.0, f"默认 B 点 x 应为图像中心 400.0，实际 {bx}"
    assert by == 300.0, f"默认 B 点 y 应为图像中心 300.0，实际 {by}"
    print("PASS: default_b_uses_image_center_when_no_a")


def test_existing_b_points_unchanged() -> None:
    """标定包已有 B 点时，不应生成默认 B 点，保持原有值。"""
    _module = _import_7_25()
    cfg = _module.MeasurementConfig()
    wf = _module.MeasurementWorkflow(cfg)

    state = _make_state(_module, a_points=[(100, 200)], b_points=[(150, 250)])
    follower = MagicMock()
    follower.initialize_abc_with_calibration = MagicMock()

    wf._call_initialize_abc_with_loaded_calibration(follower, state)

    assert state.rule_ab_b_positive_points == [[150.0, 250.0]], (
        f"已有 B 点不应被修改：{state.rule_ab_b_positive_points}"
    )
    print("PASS: existing_b_points_unchanged")


def test_default_b_clamped_to_image_bounds() -> None:
    """A 点靠近右/下边界时，默认 B 点应被限制在图像范围内。"""
    _module = _import_7_25()
    cfg = _module.MeasurementConfig()
    wf = _module.MeasurementWorkflow(cfg)

    fake_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    with patch.object(wf, "_capture_current_rule_ab_frame", return_value=fake_frame):
        # A 点在右下角，偏移后会超出 800x600
        state = _make_state(_module, a_points=[(780, 590)], b_points=[])
        follower = MagicMock()
        follower.initialize_abc_with_calibration = MagicMock()

        wf._call_initialize_abc_with_loaded_calibration(follower, state)

    b_points = state.rule_ab_b_positive_points
    assert len(b_points) == 1, f"应生成 1 个默认 B 点，实际 {len(b_points)}"
    bx, by = b_points[0]
    assert 0 <= bx < 800, f"默认 B 点 x={bx} 超出图像宽度"
    assert 0 <= by < 600, f"默认 B 点 y={by} 超出图像高度"
    assert bx == 799.0, f"默认 B 点 x 应被限制为 799.0，实际 {bx}"
    assert by == 590.0, f"默认 B 点 y 无 x 方向偏移，应保持 590.0，实际 {by}"
    print("PASS: default_b_clamped_to_image_bounds")


if __name__ == "__main__":
    test_default_b_generated_from_a_point()
    test_default_b_uses_image_center_when_no_a()
    test_existing_b_points_unchanged()
    test_default_b_clamped_to_image_bounds()
    print("All tests passed.")
