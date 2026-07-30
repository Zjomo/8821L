"""
测试 0_measurement_workflow_real_virtual_same_detection_7_25.py 中
RuleAB 初始化时 B 正点缺失的行为。

当前版本已删除自动兜底生成 B 点的逻辑，因此：
  1. B 点缺失时不再自动生成默认 B 点
  2. 标定包已有 B 点时，保持原有值不变
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


def test_no_default_b_generated_when_missing() -> None:
    """B 点缺失时，不应再自动生成默认 B 点。"""
    _module = _import_7_25()
    cfg = _module.MeasurementConfig()
    wf = _module.MeasurementWorkflow(cfg)

    fake_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    with patch.object(wf, "_capture_current_rule_ab_frame", return_value=fake_frame) as mock_capture:
        state = _make_state(_module, a_points=[(100, 200)], b_points=[])
        follower = MagicMock()
        follower.initialize_abc_with_calibration = MagicMock()

        wf._call_initialize_abc_with_loaded_calibration(follower, state)

    b_points = state.rule_ab_b_positive_points
    assert len(b_points) == 0, f"B 点缺失时不应生成默认 B 点，实际 {len(b_points)}"
    mock_capture.assert_not_called()
    print("PASS: no_default_b_generated_when_missing")


def test_existing_b_points_unchanged() -> None:
    """标定包已有 B 点时，保持原有值不变。"""
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


if __name__ == "__main__":
    test_no_default_b_generated_when_missing()
    test_existing_b_points_unchanged()
    print("All tests passed.")
