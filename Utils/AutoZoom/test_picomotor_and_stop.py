"""
测试 AutoZoom Focus 模块的 Picomotor 连接配置与停止机制。

覆盖：
  1. AutofocusConfig 新增 Picomotor 连接字段及默认值
  2. ZAxisController 使用配置中的 conn/backend/timeout 等参数
  3. ZAxisController.check_available 对控制器索引的校验
  4. 搜索策略在 should_stop 返回 True 时立即抛出 FocusSearchStopped
  5. AutofocusController.run_closed_loop 捕获 FocusSearchStopped 并返回停止结果
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np

# 将 AutoZoom 根目录加入路径，保证 Focus 模块可导入
AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from Focus.config import AutofocusConfig
from Focus.controller import AutofocusController
from Focus.metrics import FocusMetricsCalculator
from Focus.scorer import FocusScorer
from Focus.search import (
    create_search,
    CurveFitSearch,
    FocusSearchStopped,
    FullSweepSearch,
    GoldenSectionSearch,
    HillClimbSearch,
)
from Focus.z_axis import ZAxisController


def _make_cfg(**overrides: Any) -> AutofocusConfig:
    defaults: Dict[str, Any] = {
        "capture_mode": "screen_region",
        "capture_area": (0, 0, 100, 100),
        "focus_roi": (0, 0, 100, 100),
        "z_enabled": True,
        "z_axis": 1,
        "z_picomotor_conn": 2,
        "z_picomotor_backend": "pyusb",
        "z_picomotor_timeout": 3.0,
        "z_picomotor_multiaddr": True,
        "z_picomotor_scan": False,
        "z_picomotor_velocity": 150,
        "z_picomotor_acceleration": 1500,
    }
    defaults.update(overrides)
    return AutofocusConfig(**defaults)


def test_config_defaults() -> None:
    """确认新增字段存在且默认值合理。"""
    cfg = AutofocusConfig()
    assert cfg.z_picomotor_conn == 0
    assert cfg.z_picomotor_backend == "auto"
    assert cfg.z_picomotor_timeout == 5.0
    assert cfg.z_picomotor_multiaddr is False
    assert cfg.z_picomotor_scan is True
    assert cfg.z_picomotor_velocity is None
    assert cfg.z_picomotor_acceleration is None
    print("PASS: config defaults")


def test_config_overrides() -> None:
    """确认配置字段可被覆盖。"""
    cfg = _make_cfg()
    assert cfg.z_picomotor_conn == 2
    assert cfg.z_picomotor_backend == "pyusb"
    assert cfg.z_picomotor_timeout == 3.0
    assert cfg.z_picomotor_multiaddr is True
    assert cfg.z_picomotor_scan is False
    assert cfg.z_picomotor_velocity == 150
    assert cfg.z_picomotor_acceleration == 1500
    print("PASS: config overrides")


def test_z_axis_conn_kwargs() -> None:
    """ZAxisController 应从配置构造连接参数，不再硬编码 conn=0。"""
    cfg = _make_cfg()
    z = ZAxisController(cfg)
    kwargs = z._conn_kwargs()
    assert kwargs["conn"] == 2
    assert kwargs["backend"] == "pyusb"
    assert kwargs["timeout"] == 3.0
    assert kwargs["multiaddr"] is True
    assert kwargs["scan"] is False
    print("PASS: z_axis conn kwargs")


def test_z_axis_check_available_index_out_of_range() -> None:
    """conn 超出实际设备数量时应给出明确错误。"""
    cfg = _make_cfg(z_picomotor_conn=5)
    z = ZAxisController(cfg)

    fake_newport = MagicMock()
    fake_newport.get_usb_devices_number_picomotor = MagicMock(return_value=2)

    import Focus.z_axis as z_axis_module

    with patch.object(z_axis_module, "Newport", fake_newport):
        ok, msg = z.check_available()
        assert ok is False, f"期望不可用，但返回 ok={ok}"
        assert "5" in msg, f"错误信息未包含索引 5: {msg}"
        assert "0~1" in msg, f"错误信息未包含有效范围: {msg}"
    print("PASS: z_axis check_available index out of range")


def test_z_axis_check_available_ok() -> None:
    """conn 在有效范围内时应返回可用。"""
    cfg = _make_cfg(z_picomotor_conn=1)
    z = ZAxisController(cfg)
    fake_newport = MagicMock()
    fake_newport.get_usb_devices_number_picomotor = MagicMock(return_value=3)

    import Focus.z_axis as z_axis_module

    with patch.object(z_axis_module, "Newport", fake_newport):
        ok, msg = z.check_available()
        assert ok is True
        assert msg is None
    print("PASS: z_axis check_available ok")


def test_search_stops_immediately() -> None:
    """所有搜索策略在 should_stop 为 True 时立即抛出 FocusSearchStopped。"""
    cfg = _make_cfg()

    def move_fn(delta: int) -> None:
        pass

    def measure_fn(phase: str, iteration: int):
        # 返回逐渐提升的分数，避免策略因分数问题提前结束
        return 0.5 + iteration * 0.01, {}

    stopped = {"flag": False}

    def should_stop() -> bool:
        return stopped["flag"]

    for strategy in ["hill_climb", "full_sweep", "curve_fit", "golden_section"]:
        search = create_search(strategy, cfg, move_fn, measure_fn, print, should_stop)
        # 启动后立刻请求停止
        stopped["flag"] = True
        try:
            search.search(
                initial_score=0.5,
                target=0.95,
                max_total_steps=200,
                max_iter=40,
                patience=3,
                min_improve=0.0,
            )
            raise AssertionError(f"{strategy} 未在停止时抛出异常")
        except FocusSearchStopped:
            pass
        stopped["flag"] = False
    print("PASS: all search strategies stop immediately")


def test_search_stops_during_settle() -> None:
    """停止信号应能中断 _settle 中的 sleep。"""
    cfg = _make_cfg(z_settle_time_s=10.0)

    stop_flag = {"stopped": False}

    def should_stop() -> bool:
        return stop_flag["stopped"]

    search = HillClimbSearch(
        cfg,
        move_fn=lambda _d: None,
        measure_fn=lambda _p, _i: (0.5, {}),
        log_fn=lambda _m: None,
        should_stop=should_stop,
    )

    start = time.time()
    stop_flag["stopped"] = True
    try:
        search._settle()
    except FocusSearchStopped:
        pass
    elapsed = time.time() - start
    assert elapsed < 1.0, f"settle 未响应停止，耗时 {elapsed}s"
    print("PASS: search stops during settle")


def test_controller_run_closed_loop_stopped() -> None:
    """Controller 在搜索被停止时应返回 stopped_by_user 而非崩溃。"""
    cfg = _make_cfg(z_enabled=True)
    metrics = FocusMetricsCalculator(cfg)
    scorer = FocusScorer(cfg, metrics)
    z_axis = ZAxisController(cfg)
    # 避免连接真实硬件
    z_axis.connect = lambda: z_axis  # type: ignore
    z_axis.move_relative = lambda _steps: None  # type: ignore
    controller = AutofocusController(cfg, scorer, metrics, z_axis)

    stop_flag = {"stopped": False}
    controller.should_stop = lambda: stop_flag["stopped"]

    # 让 measure_fn 返回有效分数，使搜索进入循环；随后触发停止
    call_count = {"n": 0}

    def fake_measure():
        call_count["n"] += 1
        if call_count["n"] >= 2:
            stop_flag["stopped"] = True
        return {
            "roi_metrics": {
                "highfreq": 0.5,
                "tenengrad": 100.0,
                "brenner": 10.0,
                "red_blue": 0.5,
            }
        }

    controller.metrics_calc.capture_live = fake_measure  # type: ignore

    result = controller.run_closed_loop(initial_focus_score=0.5)
    assert result.get("reason") == "stopped_by_user"
    assert result.get("ok") is False
    print("PASS: controller run_closed_loop stopped")


def test_worker_stop_sets_running_false() -> None:
    """Worker.stop_loop 应将 running 状态置为 False。"""
    from autofocus_qt_ui.worker import AutofocusWorker

    worker = AutofocusWorker()
    worker._state.running = True
    worker.stop_loop()
    assert worker._state.running is False
    print("PASS: worker stop_loop")


if __name__ == "__main__":
    test_config_defaults()
    test_config_overrides()
    test_z_axis_conn_kwargs()
    test_z_axis_check_available_index_out_of_range()
    test_z_axis_check_available_ok()
    test_search_stops_immediately()
    test_search_stops_during_settle()
    test_controller_run_closed_loop_stopped()
    test_worker_stop_sets_running_false()
    print("\nAll tests passed!")
