"""
日志管理模块：实时保存运行日志到文件。

功能：
  1. 线程安全的日志写入
  2. 自动按日期分割日志文件
  3. 支持多订阅者（同时写入文件和回调）
  4. 日志文件自动创建与轮转
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional


class LogManager:
    """
    日志管理器：实时保存日志到 ./Log 目录。

    使用方式：
        log_manager = LogManager(log_dir="./Log")
        log_manager.start()
        log_manager.log("测量开始")
        # ...
        log_manager.stop()
    """

    def __init__(
        self,
        log_dir: str = "./Log",
        enabled: bool = True,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        """
        初始化日志管理器。

        参数
        ----------
        log_dir : str
            日志保存目录，默认为 "./Log"。
        enabled : bool
            是否启用日志保存，默认 True。
        on_log : Callable[[str], None], optional
            额外的日志回调函数（例如同时输出到控制台）。
        """
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.on_log = on_log

        self._lock = threading.Lock()
        self._file_handle: Optional[object] = None
        self._current_log_path: Optional[Path] = None
        self._current_date: str = ""
        self._started = False

        # 统计
        self._total_lines = 0
        self._total_bytes = 0

    def start(self) -> None:
        """启动日志管理器，创建日志目录和当日日志文件。"""
        if not self.enabled:
            return

        with self._lock:
            if self._started:
                return

            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self._ensure_log_file()
                self._started = True
            except Exception as e:
                print(f"[LogManager] 启动失败：{e}")
                self.enabled = False

    def stop(self) -> None:
        """停止日志管理器，关闭文件句柄。"""
        with self._lock:
            if self._file_handle is not None:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
                self._file_handle = None
            self._started = False

    def log(self, message: str) -> None:
        """
        写入一条日志。

        参数
        ----------
        message : str
            日志内容（不含时间戳，由调用方添加）。
        """
        if not self.enabled:
            if self.on_log is not None:
                self.on_log(message)
            return

        with self._lock:
            # 检查是否需要切换到新日期的文件
            today = datetime.now().strftime("%Y-%m-%d")
            if today != self._current_date:
                self._rotate_log_file(today)

            # 写入文件
            if self._file_handle is not None:
                try:
                    line = message + "\n"
                    self._file_handle.write(line)
                    self._file_handle.flush()
                    self._total_lines += 1
                    self._total_bytes += len(line.encode("utf-8"))
                except Exception as e:
                    print(f"[LogManager] 写入失败：{e}")

        # 回调
        if self.on_log is not None:
            self.on_log(message)

    def _ensure_log_file(self) -> None:
        """确保日志文件已打开。"""
        today = datetime.now().strftime("%Y-%m-%d")
        self._rotate_log_file(today)

    def _rotate_log_file(self, date_str: str) -> None:
        """切换到指定日期的日志文件。"""
        # 关闭旧文件
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None

        # 创建新文件
        self._current_date = date_str
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"log_{date_str.replace('-', '')}_{timestamp}.txt"
        self._current_log_path = self.log_dir / filename

        try:
            self._file_handle = open(self._current_log_path, "a", encoding="utf-8")
        except Exception as e:
            print(f"[LogManager] 无法创建日志文件：{e}")
            self.enabled = False

    def get_log_path(self) -> Optional[Path]:
        """返回当前日志文件路径。"""
        return self._current_log_path

    def get_stats(self) -> dict:
        """返回日志统计信息。"""
        return {
            "total_lines": self._total_lines,
            "total_bytes": self._total_bytes,
            "current_log_path": str(self._current_log_path) if self._current_log_path else None,
            "enabled": self.enabled,
        }

    def __enter__(self) -> "LogManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# 全局单例（可选使用）
_global_log_manager: Optional[LogManager] = None
_global_lock = threading.Lock()


def get_global_log_manager() -> Optional[LogManager]:
    """获取全局日志管理器实例。"""
    return _global_log_manager


def init_global_log_manager(
    log_dir: str = "./Log",
    enabled: bool = True,
    on_log: Optional[Callable[[str], None]] = None,
) -> LogManager:
    """初始化全局日志管理器。"""
    global _global_log_manager
    with _global_lock:
        if _global_log_manager is None:
            _global_log_manager = LogManager(log_dir=log_dir, enabled=enabled, on_log=on_log)
            _global_log_manager.start()
        return _global_log_manager


def shutdown_global_log_manager() -> None:
    """关闭全局日志管理器。"""
    global _global_log_manager
    with _global_lock:
        if _global_log_manager is not None:
            _global_log_manager.stop()
            _global_log_manager = None