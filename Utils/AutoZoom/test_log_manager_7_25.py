"""
测试 0_measurement_workflow_real_virtual_same_detection_7_25.py 中的日志实时保存功能。

覆盖：
  1. LogManager 基本写入与文件创建
  2. 日志文件路径正确性
  3. 线程安全写入
  4. 日志管理器启动/关闭
  5. MeasurementConfig 中日志保存参数（7_25 版本）
  6. MeasurementWorkflow 集成测试（7_25 版本）
  7. GUI 配置变化时重建 LogManager
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


def _import_7_25():
    """动态导入 7_25 主模块，避免污染 sys.modules。"""
    import importlib.util
    _module_path = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_25.py"
    _spec = importlib.util.spec_from_file_location(
        "measurement_workflow_7_25", str(_module_path)
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["measurement_workflow_7_25"] = _module
    _spec.loader.exec_module(_module)  # type: ignore[union-attr]
    return _module


def test_log_manager_creates_file_7_25() -> None:
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

        print("PASS: log_manager_creates_file_7_25")


def test_log_manager_disabled_7_25() -> None:
    """禁用时 LogManager 不应创建文件，但仍能调用 log()。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_manager = LogManager(log_dir=tmpdir, enabled=False)
        log_manager.start()

        log_manager.log("[2026-07-20 10:00:00] 这条日志不应写入文件")

        log_manager.stop()

        log_path = log_manager.get_log_path()
        assert log_path is None, "禁用时日志路径应为 None"

        print("PASS: log_manager_disabled_7_25")


def test_log_manager_thread_safety_7_25() -> None:
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

        log_path = log_manager.get_log_path()
        assert log_path is not None
        content = log_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        non_empty_lines = [l for l in lines if l.strip()]
        expected_lines = num_threads * lines_per_thread
        assert len(non_empty_lines) == expected_lines, (
            f"期望 {expected_lines} 行，实际 {len(non_empty_lines)} 行"
        )

        print("PASS: log_manager_thread_safety_7_25")


def test_measurement_config_log_params_7_25() -> None:
    """7_25 的 MeasurementConfig 应包含日志保存参数。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig

    cfg = MeasurementConfig()
    assert hasattr(cfg, "save_log_to_file"), "MeasurementConfig 应有 save_log_to_file 属性"
    assert hasattr(cfg, "log_dir"), "MeasurementConfig 应有 log_dir 属性"
    assert cfg.save_log_to_file == True
    assert cfg.log_dir == "./Log"

    print("PASS: measurement_config_log_params_7_25")


def test_workflow_log_manager_integration_7_25() -> None:
    """7_25 的 MeasurementWorkflow 应正确集成 LogManager。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

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

        print("PASS: workflow_log_manager_integration_7_25")


def test_log_config_change_reinitializes_manager_7_25() -> None:
    """GUI 中修改 log_dir 后，LogManager 应重建到新目录。"""
    _module = _import_7_25()
    MeasurementConfig = _module.MeasurementConfig
    MeasurementWorkflow = _module.MeasurementWorkflow

    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = Path(tmpdir) / "dir1"
        dir2 = Path(tmpdir) / "dir2"

        cfg = MeasurementConfig()
        cfg.save_log_to_file = True
        cfg.log_dir = str(dir1)

        wf = MeasurementWorkflow(cfg)
        wf.log("第一条日志")

        path1 = wf._log_manager.get_log_path()
        assert path1 is not None
        assert str(path1).startswith(str(dir1))

        # 模拟 sync_config_from_ui_to_workflow 中配置变化后的重建
        cfg2 = MeasurementConfig()
        cfg2.save_log_to_file = True
        cfg2.log_dir = str(dir2)
        wf.cfg = cfg2
        wf._shutdown_log_manager()
        wf._init_log_manager()
        wf.log("第二条日志")

        path2 = wf._log_manager.get_log_path()
        assert path2 is not None
        assert str(path2).startswith(str(dir2))

        # 验证 dir1 只有第一条，dir2 只有第二条
        content1 = path1.read_text(encoding="utf-8")
        content2 = path2.read_text(encoding="utf-8")
        assert "第一条日志" in content1
        assert "第二条日志" in content2
        assert "第二条日志" not in content1

        wf._shutdown_log_manager()

        print("PASS: log_config_change_reinitializes_manager_7_25")


if __name__ == "__main__":
    tests = [
        test_log_manager_creates_file_7_25,
        test_log_manager_disabled_7_25,
        test_log_manager_thread_safety_7_25,
        test_measurement_config_log_params_7_25,
        test_workflow_log_manager_integration_7_25,
        test_log_config_change_reinitializes_manager_7_25,
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
