"""
安全保护与限位管理器 (SafetyManager)

灵感来源:
- 工业控制安全标准 (IEC 61508)
- artiq 实验控制框架的安全机制
- 实验室设备保护实践

功能:
- 软件限位: 限制单次移动和累计移动距离
- 超时保护: 限制单次对准和总运行时间
- 紧急停止: 提供全局紧急停止接口
- 状态监控: 实时监控设备状态，异常时自动停止
- 线程安全: 所有状态操作均通过锁保护 (v19 修复)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

LOGGER = logging.getLogger("SpotZoom.Safety")


@dataclass
class SafetyLimits:
    """安全限位参数。"""
    # 单次移动限位 (步数)
    max_single_move_x: int = 5000
    max_single_move_y: int = 5000

    # 累计移动限位 (步数)
    max_total_move_x: int = 100000
    max_total_move_y: int = 100000

    # Z 轴限位
    max_z_up_total: int = 100
    max_z_down_total: int = 100

    # 时间限位 (秒)
    max_single_align_time_s: float = 300.0  # 5 分钟
    max_total_run_time_s: float = 1800.0  # 30 分钟
    max_settle_timeout_s: float = 10.0

    # 对准限位
    max_align_rounds: int = 100
    max_iterations: int = 20


@dataclass
class SafetyState:
    """安全状态快照。"""
    is_emergency_stopped: bool = False
    total_x_steps: int = 0
    total_y_steps: int = 0
    total_z_up: int = 0
    total_z_down: int = 0
    align_start_time: Optional[float] = None
    run_start_time: Optional[float] = None
    violation_count: int = 0
    last_violation: Optional[str] = None


class SafetyManager:
    """安全保护与限位管理器 (线程安全)。

    所有状态读写操作均通过 threading.Lock 保护，确保在多线程
    环境下 (如主对准循环 + 回调线程) 的安全性。

    Parameters
    ----------
    limits : SafetyLimits or None
        安全限位参数。为 None 时使用默认值。
    on_violation : Callable or None
        违规回调函数。接收 (violation_type, message) 参数。
    """

    def __init__(
        self,
        limits: Optional[SafetyLimits] = None,
        on_violation: Optional[Callable[[str, str], None]] = None,
    ):
        self.limits = limits or SafetyLimits()
        self.state = SafetyState()
        self._on_violation = on_violation
        self._run_start_time: Optional[float] = None
        self._lock = threading.Lock()

    def start_run(self) -> None:
        """标记运行开始。"""
        with self._lock:
            self._run_start_time = time.time()
            self.state.run_start_time = self._run_start_time
        LOGGER.info("SafetyManager: run started")

    def start_align(self) -> None:
        """标记单次对准开始。"""
        with self._lock:
            self.state.align_start_time = time.time()

    def check_move_x(self, steps: int) -> bool:
        """检查 X 轴移动是否在安全范围内。

        Returns
        -------
        bool
            True 表示安全，允许移动。
        """
        with self._lock:
            if self.state.is_emergency_stopped:
                self._violation("EMERGENCY_STOP", "Emergency stop is active")
                return False

            abs_steps = abs(steps)
            if abs_steps > self.limits.max_single_move_x:
                self._violation(
                    "X_MOVE_LIMIT",
                    f"X single move {steps} exceeds limit {self.limits.max_single_move_x}"
                )
                return False

            if self.state.total_x_steps + abs_steps > self.limits.max_total_move_x:
                self._violation(
                    "X_TOTAL_LIMIT",
                    f"X total steps {self.state.total_x_steps + abs_steps} would exceed limit {self.limits.max_total_move_x}"
                )
                return False

            return True

    def check_move_y(self, steps: int) -> bool:
        """检查 Y 轴移动是否在安全范围内。"""
        with self._lock:
            if self.state.is_emergency_stopped:
                self._violation("EMERGENCY_STOP", "Emergency stop is active")
                return False

            abs_steps = abs(steps)
            if abs_steps > self.limits.max_single_move_y:
                self._violation(
                    "Y_MOVE_LIMIT",
                    f"Y single move {steps} exceeds limit {self.limits.max_single_move_y}"
                )
                return False

            if self.state.total_y_steps + abs_steps > self.limits.max_total_move_y:
                self._violation(
                    "Y_TOTAL_LIMIT",
                    f"Y total steps {self.state.total_y_steps + abs_steps} would exceed limit {self.limits.max_total_move_y}"
                )
                return False

            return True

    def check_z_move(self, direction: str) -> bool:
        """检查 Z 轴移动是否在安全范围内。"""
        with self._lock:
            if self.state.is_emergency_stopped:
                self._violation("EMERGENCY_STOP", "Emergency stop is active")
                return False

            if direction == "up":
                if self.state.total_z_up + 1 > self.limits.max_z_up_total:
                    self._violation("Z_UP_LIMIT", f"Z up total {self.state.total_z_up + 1} exceeds limit")
                    return False
            elif direction == "down":
                if self.state.total_z_down + 1 > self.limits.max_z_down_total:
                    self._violation("Z_DOWN_LIMIT", f"Z down total {self.state.total_z_down + 1} exceeds limit")
                    return False

            return True

    def record_move_x(self, steps: int) -> None:
        """记录 X 轴移动（移动成功后调用）。"""
        with self._lock:
            self.state.total_x_steps += abs(steps)

    def record_move_y(self, steps: int) -> None:
        """记录 Y 轴移动。"""
        with self._lock:
            self.state.total_y_steps += abs(steps)

    def record_z_move(self, direction: str) -> None:
        """记录 Z 轴移动。"""
        with self._lock:
            if direction == "up":
                self.state.total_z_up += 1
            elif direction == "down":
                self.state.total_z_down += 1

    def check_time(self) -> bool:
        """检查运行时间是否超限。"""
        with self._lock:
            if self.state.is_emergency_stopped:
                return False

            now = time.time()

            # 检查总运行时间
            if self._run_start_time is not None:
                elapsed = now - self._run_start_time
                if elapsed > self.limits.max_total_run_time_s:
                    self._violation("TIMEOUT_TOTAL", f"Total run time {elapsed:.1f}s exceeds limit {self.limits.max_total_run_time_s}s")
                    return False

            # 检查单次对准时间
            if self.state.align_start_time is not None:
                align_elapsed = now - self.state.align_start_time
                if align_elapsed > self.limits.max_single_align_time_s:
                    self._violation("TIMEOUT_ALIGN", f"Align time {align_elapsed:.1f}s exceeds limit {self.limits.max_single_align_time_s}s")
                    return False

            return True

    def emergency_stop(self, reason: str = "User requested") -> None:
        """触发紧急停止。"""
        with self._lock:
            self.state.is_emergency_stopped = True
            self._violation("EMERGENCY_STOP", reason)
        LOGGER.critical("SafetyManager: EMERGENCY STOP triggered: %s", reason)

    def clear_emergency_stop(self) -> None:
        """清除紧急停止状态（需人工确认安全后调用）。"""
        with self._lock:
            self.state.is_emergency_stopped = False
        LOGGER.info("SafetyManager: emergency stop cleared")

    def get_status(self) -> dict:
        """获取当前安全状态摘要。"""
        with self._lock:
            return {
                "emergency_stopped": self.state.is_emergency_stopped,
                "total_x_steps": self.state.total_x_steps,
                "total_y_steps": self.state.total_y_steps,
                "total_z_up": self.state.total_z_up,
                "total_z_down": self.state.total_z_down,
                "x_remaining": max(0, self.limits.max_total_move_x - self.state.total_x_steps),
                "y_remaining": max(0, self.limits.max_total_move_y - self.state.total_y_steps),
                "z_up_remaining": max(0, self.limits.max_z_up_total - self.state.total_z_up),
                "z_down_remaining": max(0, self.limits.max_z_down_total - self.state.total_z_down),
                "violation_count": self.state.violation_count,
                "last_violation": self.state.last_violation,
            }

    def _violation(self, vtype: str, message: str) -> None:
        """记录违规。注意: 调用者必须已持有 self._lock。"""
        self.state.violation_count += 1
        self.state.last_violation = f"[{vtype}] {message}"
        LOGGER.warning("Safety violation: [%s] %s", vtype, message)
        if self._on_violation is not None:
            try:
                self._on_violation(vtype, message)
            except Exception as e:
                LOGGER.error("Safety violation callback failed: %s", e, exc_info=True)

    def reset(self) -> None:
        """重置安全状态（新运行开始时调用）。"""
        with self._lock:
            self.state = SafetyState()
            self._run_start_time = None
        LOGGER.info("SafetyManager: state reset")
