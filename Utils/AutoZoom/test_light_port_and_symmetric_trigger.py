"""
测试：
1. 照明光串口变更后重新连接生效；
2. FocusScore_ratio 触发阈值改为以 1.0 为中心的对称区间。
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

AUTOZOOM_ROOT = os.path.dirname(os.path.abspath(__file__))
if AUTOZOOM_ROOT not in sys.path:
    sys.path.insert(0, AUTOZOOM_ROOT)

from Focus.config import AutofocusConfig
from Focus.controller import AutofocusController, focus_score_ratio_in_tolerance


class DummyZAxis:
    def connect(self):
        pass

    def move_relative(self, delta):
        pass


class DummyMetricsCalc:
    def capture_live(self):
        return {"roi_metrics": {}}


class DummyScorer:
    def __init__(self, scores):
        self.scores = scores
        self.idx = 0
        self.ref = 1.0

    def score_ratio(self, roi_metrics):
        if self.idx >= len(self.scores):
            return None, {}
        score = self.scores[self.idx]
        self.idx += 1
        return score, {}

    def build_reference(self, *args, **kwargs):
        pass

    @property
    def focus_reference_ready(self):
        return True


def _make_controller(scores, trigger_ratio=0.95, trigger_count=3):
    cfg = AutofocusConfig(
        autofocus_enabled=True,
        autofocus_focus_trigger_ratio=trigger_ratio,
        autofocus_focus_trigger_count=trigger_count,
        z_enabled=False,
    )
    scorer = DummyScorer(scores)
    metrics = DummyMetricsCalc()
    z_axis = DummyZAxis()
    ctrl = AutofocusController(cfg, scorer, metrics, z_axis)
    ctrl.on_log = lambda msg: None
    return ctrl


def test_focus_score_ratio_in_tolerance():
    assert focus_score_ratio_in_tolerance(0.95, 0.95) is True
    assert focus_score_ratio_in_tolerance(1.00, 0.95) is True
    assert focus_score_ratio_in_tolerance(1.05, 0.95) is True
    assert focus_score_ratio_in_tolerance(0.94, 0.95) is False
    assert focus_score_ratio_in_tolerance(1.06, 0.95) is False
    assert focus_score_ratio_in_tolerance(None, 0.95) is False
    print("PASS: focus_score_ratio_in_tolerance")


def test_trigger_below_lower():
    """FocusScore_ratio 连续低于下限应触发补焦。"""
    ctrl = _make_controller([0.90, 0.91, 0.92], trigger_count=3)
    for _ in range(3):
        need, reasons = ctrl.evaluate_trigger(0.90)
    assert need is True, f"应触发补焦，但 need={need}"
    assert "超出 [0.9500, 1.0500]" in reasons[0], f"理由不正确: {reasons}"
    print("PASS: trigger below lower")


def test_trigger_above_upper():
    """FocusScore_ratio 连续高于上限也应触发补焦。"""
    ctrl = _make_controller([1.10, 1.11, 1.12], trigger_count=3)
    for _ in range(3):
        need, reasons = ctrl.evaluate_trigger(1.10)
    assert need is True, f"应触发补焦，但 need={need}"
    assert "超出 [0.9500, 1.0500]" in reasons[0], f"理由不正确: {reasons}"
    print("PASS: trigger above upper")


def test_reset_when_back_in_tolerance():
    """回到允许区间后连续计数应清零。"""
    ctrl = _make_controller([0.90, 1.00, 1.00], trigger_count=3)
    ctrl.evaluate_trigger(0.90)
    assert ctrl.consecutive_focus_low_count == 1
    ctrl.evaluate_trigger(1.00)
    assert ctrl.consecutive_focus_low_count == 0
    ctrl.evaluate_trigger(1.00)
    assert ctrl.consecutive_focus_low_count == 0
    print("PASS: reset when back in tolerance")


def _check_connect_light_source(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    assert "def connect_light(self):" in source, f"{path}: 未找到 connect_light 方法"
    assert "current_port = getattr(self.light, \"port\", None)" in source, (
        f"{path}: connect_light 未检查当前端口"
    )
    assert "if current_port != self.cfg.light_port:" in source, (
        f"{path}: connect_light 未在端口变化时重新连接"
    )


def test_light_port_change_in_source_6_25():
    """检查 6.25 workflow 源码中 connect_light 已加入串口变更重新连接逻辑。"""
    path = os.path.join(
        AUTOZOOM_ROOT, "0_measurement_workflow_real_virtual_same_detection_6.25.py"
    )
    _check_connect_light_source(path)
    print("PASS: light port change source check (6.25)")


def test_light_port_change_in_source_7_16():
    """检查 7_16 workflow 源码中 connect_light 已加入串口变更重新连接逻辑。"""
    path = os.path.join(
        AUTOZOOM_ROOT, "0_measurement_workflow_real_virtual_same_detection_7_16.py"
    )
    _check_connect_light_source(path)
    print("PASS: light port change source check (7_16)")


def test_illumination_relay_port_change():
    """IlluminationRelay 对象支持重新创建以使用新端口。"""
    from control.illumination_relay import IlluminationRelay

    relay = IlluminationRelay(port="COM20")
    assert relay.port == "COM20"
    relay2 = IlluminationRelay(port="COM19")
    assert relay2.port == "COM19"
    print("PASS: illumination relay port attribute")


def main():
    test_focus_score_ratio_in_tolerance()
    test_trigger_below_lower()
    test_trigger_above_upper()
    test_reset_when_back_in_tolerance()
    test_light_port_change_in_source_6_25()
    test_light_port_change_in_source_7_16()
    test_illumination_relay_port_change()
    print("\n所有测试通过!")


if __name__ == "__main__":
    main()
