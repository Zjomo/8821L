"""位移台后端：虚拟（DryRunStage，见 simulator.py）与真实串口电机。

真实电机模式（SerialXYStage）设计要点：
- 硬件确认门控：未显式 ``confirmed=True``（CLI --real-hardware-confirm /
  UI 勾选确认框）时拒绝任何命令 —— 防止误触真实设备。
- 指令模板可配置：默认 G 代码相对运动 ``G91 G1 X{dx} Y{dy}``，
  适配其他控制器时改 cmd_template 即可。
- 软限位：累计位置不得超过 workspace_mm；单步不得超过 max_step_mm。
- 应答校验：写入后等待 ack 关键字，超时抛 StageError（控制器转 FAULT）。
- 方向标定：axes_sign 兼容电机接线方向相反的机型。
"""
from __future__ import annotations

import math
import threading
import time
from typing import List, Optional, Tuple

from .models import StageCommand
from .simulator import StageError
from .simulator import XYStageProtocol  # noqa: F401  (再导出便于驱动接入)


class SerialXYStage(XYStageProtocol):
    """真实 XY 位移台（串口）。与 DryRunStage 同协议，可直接注入控制器。"""

    def __init__(self, port: str, baudrate: int = 115200,
                 cmd_template: str = "G91 G1 X{dx:.4f} Y{dy:.4f}\n",
                 ack_keyword: str = "ok",
                 axes_sign: Tuple[float, float] = (1.0, 1.0),
                 workspace_mm: Tuple[float, float] = (100.0, 100.0),
                 max_step_mm: float = 1.0,
                 timeout_s: float = 2.0,
                 settle_s: float = 0.05,
                 confirmed: bool = False) -> None:
        if port is None or port == "":
            raise StageError("serial port not specified (e.g. COM3)")
        if not confirmed:
            raise StageError(
                "real hardware NOT confirmed: pass --real-hardware-confirm / "
                "勾选硬件确认后再控制真实电机")
        self.port = port
        self.baudrate = baudrate
        self.cmd_template = cmd_template
        self.ack_keyword = ack_keyword.lower()
        self.axes_sign = axes_sign
        self.workspace_mm = workspace_mm
        self.max_step_mm = max_step_mm
        self.timeout_s = timeout_s
        self.settle_s = settle_s
        self.commands: List[StageCommand] = []
        self.position_mm = [0.0, 0.0]     # 累计位移估计（软限位用）
        self._last: Optional[StageCommand] = None
        self._seq = 0
        self._lock = threading.Lock()
        self._ser = self._open(port, baudrate, timeout_s)

    @staticmethod
    def _open(port: str, baudrate: int, timeout_s: float):
        try:
            import serial  # pyserial
        except ImportError as exc:  # pragma: no cover
            raise StageError("pyserial 未安装: pip install pyserial") from exc
        try:
            return serial.Serial(port, baudrate, timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 - 串口打开失败原因多样
            raise StageError(f"cannot open {port}@{baudrate}: {exc}") from exc

    def move_by(self, dx_mm: float, dy_mm: float,
                task_id: str = "", track_id: int = -1,
                waypoint_index: int = -1, frame_id: int = -1,
                plan_version: int = -1) -> bool:
        dx_mm *= self.axes_sign[0]
        dy_mm *= self.axes_sign[1]
        if (dx_mm ** 2 + dy_mm ** 2) ** 0.5 > self.max_step_mm + 1e-9:
            raise StageError(
                f"step {math.hypot(dx_mm, dy_mm):.3f}mm > "
                f"max_step_mm={self.max_step_mm}")
        nx = self.position_mm[0] + dx_mm
        ny = self.position_mm[1] + dy_mm
        if abs(nx) > self.workspace_mm[0] or abs(ny) > self.workspace_mm[1]:
            raise StageError(
                f"target ({nx:.2f},{ny:.2f})mm outside workspace "
                f"{self.workspace_mm}")
        cmd_str = self.cmd_template.format(dx=dx_mm, dy=dy_mm)
        with self._lock:
            self._seq += 1
            cmd = StageCommand(seq=self._seq, timestamp=time.time(),
                               dx_mm=dx_mm, dy_mm=dy_mm, task_id=task_id,
                               track_id=track_id, waypoint_index=waypoint_index,
                               frame_id=frame_id,
                               source_plan_version=plan_version)
            # ---- 写入 + 等待应答
            try:
                self._ser.reset_input_buffer()
                self._ser.write(cmd_str.encode("ascii"))
                self._ser.flush()
            except Exception as exc:  # noqa: BLE001
                raise StageError(f"serial write failed: {exc}") from exc
            deadline = time.time() + self.timeout_s
            buf = b""
            acked = False
            while time.time() < deadline:
                chunk = self._ser.read(64)
                if chunk:
                    buf += chunk
                    if self.ack_keyword in buf.decode("ascii",
                                                      errors="ignore").lower():
                        acked = True
                        break
            if not acked:
                raise StageError(
                    f"stage ack timeout ({self.timeout_s}s), "
                    f"reply={buf[:64]!r}")
            self.commands.append(cmd)
            self._last = cmd
            self.position_mm = [nx, ny]
        if self.settle_s > 0:
            time.sleep(self.settle_s)   # 等机械稳定再让视觉复检
        return True

    @property
    def last_command(self) -> Optional[StageCommand]:
        return self._last

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:  # noqa: BLE001
            pass


class PicoMotorStage(XYStageProtocol):
    """Newport Picomotor 8742/8743 位移台（pylablib 后端）。

    对应项目 API 封装（API/8742_8743_函数封装_UI模块.txt）：
    - 连接：Newport.Picomotor8742(conn=USB索引, backend="auto", timeout=5.0)
    - 相对移动：dev.move_by(axis, steps) + wait_move(axis)（阻塞到走完）
    - 急停：dev.stop(axis="all", immediate=True)
    单位契约：move_by(dx_mm, dy_mm) -> 换算 steps = mm * steps_per_mm
    （MTM 平台 ~30nm/步 ≈ 33333 steps/mm，须按实际位移台现场标定）。
    安全机制与 SerialXYStage 一致：confirmed 门控 / 单步限位 / 软限位 /
    方向标定 axes_sign / 失位检测由控制器层（spot_slip）负责。
    """

    def __init__(self, conn: int = 0,
                 x_axis: int = 1, y_axis: int = 2,
                 steps_per_mm: float = 1000.0,
                 axes_sign: Tuple[float, float] = (1.0, 1.0),
                 workspace_mm: Tuple[float, float] = (100.0, 100.0),
                 max_step_mm: float = 1.0,
                 speed_steps: Optional[float] = None,
                 settle_s: float = 0.05,
                 confirmed: bool = False,
                 timeout: float = 5.0) -> None:
        if not confirmed:
            raise StageError(
                "real hardware NOT confirmed: pass --real-hardware-confirm / "
                "勾选硬件确认后再控制真实电机")
        if steps_per_mm <= 0:
            raise StageError("steps_per_mm must be > 0")
        self.conn = conn
        self.x_axis = int(x_axis)
        self.y_axis = int(y_axis)
        self.steps_per_mm = float(steps_per_mm)
        self.axes_sign = axes_sign
        self.workspace_mm = workspace_mm
        self.max_step_mm = max_step_mm
        self.settle_s = settle_s
        self.commands: List[StageCommand] = []
        self.position_mm = [0.0, 0.0]     # 累计位移估计（软限位用）
        self._last: Optional[StageCommand] = None
        self._seq = 0
        self._lock = threading.Lock()
        self.dev = self._open(conn, timeout)
        if speed_steps is not None and speed_steps > 0:
            # 两轴同速：速度指令失败不致命（部分固件轴号未配置）
            for ax in (self.x_axis, self.y_axis):
                try:
                    self.dev.setup_velocity(axis=ax, speed=speed_steps)
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _open(conn, timeout):
        try:
            from pylablib.devices import Newport  # 懒加载（pylablib）
        except ImportError as exc:  # pragma: no cover
            raise StageError("pylablib 未安装: pip install pylablib") from exc
        try:
            return Newport.Picomotor8742(conn, backend="auto", timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            raise StageError(f"cannot open Picomotor conn={conn}: {exc}") from exc

    @staticmethod
    def usb_device_count() -> int:
        from pylablib.devices import Newport
        return Newport.get_usb_devices_number_picomotor()

    def move_by(self, dx_mm: float, dy_mm: float,
                task_id: str = "", track_id: int = -1,
                waypoint_index: int = -1, frame_id: int = -1,
                plan_version: int = -1) -> bool:
        dx_mm *= self.axes_sign[0]
        dy_mm *= self.axes_sign[1]
        if math.hypot(dx_mm, dy_mm) > self.max_step_mm + 1e-9:
            raise StageError(
                f"step {math.hypot(dx_mm, dy_mm):.3f}mm > "
                f"max_step_mm={self.max_step_mm}")
        nx = self.position_mm[0] + dx_mm
        ny = self.position_mm[1] + dy_mm
        if abs(nx) > self.workspace_mm[0] or abs(ny) > self.workspace_mm[1]:
            raise StageError(
                f"target ({nx:.2f},{ny:.2f})mm outside workspace "
                f"{self.workspace_mm}")
        dx_steps = int(round(dx_mm * self.steps_per_mm))
        dy_steps = int(round(dy_mm * self.steps_per_mm))
        with self._lock:
            self._seq += 1
            cmd = StageCommand(seq=self._seq, timestamp=time.time(),
                               dx_mm=dx_mm, dy_mm=dy_mm, task_id=task_id,
                               track_id=track_id, waypoint_index=waypoint_index,
                               frame_id=frame_id,
                               source_plan_version=plan_version)
            try:
                if dx_steps != 0:
                    self.dev.move_by(axis=self.x_axis, steps=dx_steps)
                    self.dev.wait_move(axis=self.x_axis)
                if dy_steps != 0:
                    self.dev.move_by(axis=self.y_axis, steps=dy_steps)
                    self.dev.wait_move(axis=self.y_axis)
            except Exception as exc:  # noqa: BLE001 - 驱动异常统一转 StageError
                raise StageError(f"picomotor move failed: {exc}") from exc
            self.commands.append(cmd)
            self._last = cmd
            self.position_mm = [nx, ny]
        if self.settle_s > 0:
            time.sleep(self.settle_s)   # 等机械稳定再让视觉复检
        return True

    def stop_all(self) -> None:
        """急停：immediate=True 需 axis='all'（API 封装约定）。"""
        with self._lock:   # 与 move_by 串行，避免 USB 并发访问
            try:
                self.dev.stop(axis="all", immediate=True)
            except Exception as exc:  # noqa: BLE001
                raise StageError(f"picomotor estop failed: {exc}") from exc

    @property
    def last_command(self) -> Optional[StageCommand]:
        return self._last

    def close(self) -> None:
        try:
            self.dev.close()
        except Exception:  # noqa: BLE001
            pass
