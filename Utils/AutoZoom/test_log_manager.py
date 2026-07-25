"""
测试 LogManager 日志实时保存功能。

覆盖：
  1. LogManager 基本写入与文件创建
  2. 日志文件路径正确性
  3. 线程安全写入
  4. 日志管理器启动/关闭
  5. MeasurementConfig 中日志保存参数
  6. MeasurementWorkflow 集成测试
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# 路径设置
AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from log_manager import LogManager


def test_log_manager_creates_file() -> None:
    """LogManager 应正确创建日志文件。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_manager = LogManager(log_dir=tmpdir, enabled=True)
        log_manager.start()

        log_manager.log("[2026-07-20 10:00:00] 测试日志行1")
        log_manager.log("[2026-07-20 10:00:01] 测试日志行2")

        log_manager.stop()

        log_path = log_manager.get_log_path()
        assert log_path is not None, "日志路径应为非空"
        assert log_path.exists(), f"日志文件应存在：{log_path}"

        content = log_path.read_text(encoding="utf-8")
        assert "测试日志行1" in content
        assert "测试日志行2" in content

        print("PASS: log_manager_creates_file")


def test_log_manager_disabled() -> None:
    """禁用时 LogManager 不应创建文件，但仍能调用 log()。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_manager = LogManager(log_dir=tmpdir, enabled=False)
        log_manager.start()

        log_manager.log("[2026-07-20 10:00:00] 这条日志不应写入文件")

        log_manager.stop()

        # 禁用时日志路径应为 None
        log_path = log_manager.get_log_path()
        assert log_path is None, "禁用时日志路径应为 None"

        print("PASS: log_manager_disabled")


def test_log_manager_stats() -> None:
    """LogManager 统计信息应正确。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_manager = LogManager(log_dir=tmpdir, enabled=True)
        log_manager.start()

        log_manager.log("[2026-07-20 10:00:00] 行1")
        log_manager.log("[2026-07-20 10:00:01] 行2")
        log_manager.log("[2026-07-20 10:00:02] 行3")

        stats = log_manager.get_stats()
        assert stats["total_lines"] == 3
        assert stats["enabled"] == True
        assert stats["current_log_path"] is not None

        log_manager.stop()

        print("PASS: log_manager_stats")


def test_log_manager_thread_safety() -> None:
    """多线程并发写入应无数据丢失。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_manager = LogManager(log_dir=tmpdir, enabled=True)
        log_manager.start()

        num_threads = 5
        lines_per_thread = 100

        def write_logs(thread_id: int) -> None:
            for i in range(lines_per_thread):
                log_manager.log(f"[2026-07-20 10:00:00] Thread-{thread_id} Line-{i}")

        threads = []
        for t_id in range(num_threads):
            t = threading.Thread(target=write_logs, args=(t_id,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        log_manager.stop()

        # 验证文件内容
        log_path = log_manager.get_log_path()
        assert log_path is not None
        content = log_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        # 允许少量空行
        non_empty_lines = [l for l in lines if l.strip()]
        expected_lines = num_threads * lines_per_thread
        assert len(non_empty_lines) == expected_lines, (
            f"期望 {expected_lines} 行，实际 {len(non_empty_lines)} 行"
        )

        print("PASS: log_manager_thread_safety")


def test_log_manager_context_manager() -> None:
    """LogManager 作为上下文管理器应正确启动和关闭。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with LogManager(log_dir=tmpdir, enabled=True) as log_manager:
            log_manager.log("[2026-07-20 10:00:00] 上下文管理器测试")

        log_path = log_manager.get_log_path()
        assert log_path is not None
        assert log_path.exists()

        content = log_path.read_text(encoding="utf-8")
        assert "上下文管理器测试" in content

        print("PASS: log_manager_context_manager")


def test_log_manager_callback() -> None:
    """LogManager 回调函数应被正确调用。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        callback_log: list = []

        def on_log(msg: str) -> None:
            callback_log.append(msg)

        log_manager = LogManager(log_dir=tmpdir, enabled=True, on_log=on_log)
        log_manager.start()

        log_manager.log("[2026-07-20 10:00:00] 回调测试")
        log_manager.stop()

        assert len(callback_log) == 1
        assert "回调测试" in callback_log[0]

        print("PASS: log_manager_callback")


def test_measurement_config_log_params() -> None:
    """MeasurementConfig 应包含日志保存参数。"""
    # 动态导入主模块
    import importlib.util
    _module_path = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_23.py"
    _spec = importlib.util.spec_from_file_location(
        "measurement_workflow_7_23", str(_module_path)
    )
    _measurement_workflow_7_23 = importlib.util.module_from_spec(_spec)
    sys.modules["measurement_workflow_7_23"] = _measurement_workflow_7_23
    _spec.loader.exec_module(_measurement_workflow_7_23)  # type: ignore[union-attr]

    MeasurementConfig = _measurement_workflow_7_23.MeasurementConfig

    cfg = MeasurementConfig()
    assert hasattr(cfg, "save_log_to_file"), "MeasurementConfig 应有 save_log_to_file 属性"
    assert hasattr(cfg, "log_dir"), "MeasurementConfig 应有 log_dir 属性"
    assert cfg.save_log_to_file == True  # 默认值
    assert cfg.log_dir == "./Log"  # 默认值

    print("PASS: measurement_config_log_params")


def test_workflow_log_manager_integration() -> None:
    """MeasurementWorkflow 应正确集成 LogManager。"""
    import importlib.util
    _module_path = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_23.py"
    _spec = importlib.util.spec_from_file_location(
        "measurement_workflow_7_23", str(_module_path)
    )
    _measurement_workflow_7_23 = importlib.util.module_from_spec(_spec)
    sys.modules["measurement_workflow_7_23"] = _measurement_workflow_7_23
    _spec.loader.exec_module(_measurement_workflow_7_23)  # type: ignore[union-attr]

    MeasurementConfig = _measurement_workflow_7_23.MeasurementConfig
    MeasurementWorkflow = _measurement_workflow_7_23.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MeasurementConfig()
        cfg.save_log_to_file = True
        cfg.log_dir = tmpdir

        wf = MeasurementWorkflow(cfg)

        # 验证日志管理器已初始化
        assert wf._log_manager is not None, "LogManager 应已初始化"
        assert wf._log_manager.enabled == True

        # 写入日志
        wf.log("集成测试日志行")

        # 验证文件内容
        log_path = wf._log_manager.get_log_path()
        assert log_path is not None
        content = log_path.read_text(encoding="utf-8")
        assert "集成测试日志行" in content

        # 关闭
        wf._shutdown_log_manager()
        assert wf._log_manager is None

        print("PASS: workflow_log_manager_integration")


def test_log_manager_creates_directory() -> None:
    """LogManager 应自动创建不存在的日志目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建一个不存在的子目录路径
        log_dir = Path(tmpdir) / "new_log_dir" / "nested"
        assert not log_dir.exists()

        log_manager = LogManager(log_dir=str(log_dir), enabled=True)
        log_manager.start()

        log_manager.log("[2026-07-20 10:00:00] 目录创建测试")
        log_manager.stop()

        assert log_dir.exists(), "日志目录应被自动创建"

        log_path = log_manager.get_log_path()
        assert log_path is not None
        assert log_path.exists()

        print("PASS: log_manager_creates_directory")


if __name__ == "__main__":
    tests = [
        test_log_manager_creates_file,
        test_log_manager_disabled,
        test_log_manager_stats,
        test_log_manager_thread_safety,
        test_log_manager_context_manager,
        test_log_manager_callback,
        test_measurement_config_log_params,
        test_workflow_log_manager_integration,
        test_log_manager_creates_directory,
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