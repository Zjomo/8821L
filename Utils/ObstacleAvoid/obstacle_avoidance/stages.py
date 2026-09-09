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
from .motion import (AxisMotionConfig, MotionConfig, MotionTelemetry,
                      VirtualXYZStage, default_motion_config)
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
                 x_axis: int = 1, y_axis: int = 2, z_axis: int = 3,
                 steps_per_mm: float = 1000.0,
                 steps_per_mm_by_axis: Optional[dict] = None,
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
        self.z_axis = int(z_axis)
        self.axis_map = {"x": self.x_axis, "y": self.y_axis,
                         "z": self.z_axis}
        if len(set(self.axis_map.values())) != 3:
            raise StageError("Picomotor X/Y/Z axis numbers must be distinct")
        self.steps_per_mm = float(steps_per_mm)
        self.steps_per_mm_by_axis = {"x": self.steps_per_mm,
                                     "y": self.steps_per_mm,
                                     "z": self.steps_per_mm}
        for name, value in (steps_per_mm_by_axis or {}).items():
            key = str(name).lower()
            if key in self.steps_per_mm_by_axis:
                if float(value) <= 0:
                    raise StageError(f"{key} steps_per_mm must be > 0")
                self.steps_per_mm_by_axis[key] = float(value)
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
            # X/Y/Z 同速：速度指令失败不致命（部分固件轴号未配置）
            for ax in sorted(set(self.axis_map.values())):
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

    @staticmethod
    def probe_usb(conn: int = 0, timeout: float = 5.0) -> dict:
        """Connect briefly and return controller identity and available axes.

        Detection is deliberately read-only: no motor movement or velocity
        changes are issued.  The connection is closed in all cases so probing
        does not keep a USB controller locked while the user edits mappings.
        """
        dev = None
        try:
            from pylablib.devices import Newport
            dev = Newport.Picomotor8742(int(conn), backend="auto",
                                        timeout=timeout)
            get_id = getattr(dev, "get_id", None)
            identity = get_id() if callable(get_id) else f"Picomotor8742[{conn}]"
            get_axes = getattr(dev, "get_all_axes", None)
            axes = list(get_axes()) if callable(get_axes) else []
            return {"conn": int(conn), "id": str(identity),
                    "axes": [int(a) for a in axes], "ok": True}
        except Exception as exc:  # noqa: BLE001 - probe must not abort UI
            return {"conn": int(conn), "id": "", "axes": [],
                    "ok": False, "error": str(exc)}
        finally:
            if dev is not None:
                try:
                    dev.close()
                except Exception:  # noqa: BLE001 - best-effort probe cleanup
                    pass

    @classmethod
    def scan_usb_controllers(cls, timeout: float = 5.0) -> list[dict]:
        """Enumerate USB controllers and their available motor axes."""
        try:
            count = int(cls.usb_device_count())
        except Exception as exc:  # noqa: BLE001 - report scan failure to UI
            return [{"conn": 0, "id": "", "axes": [], "ok": False,
                     "error": str(exc)}]
        if count <= 0:
            return []
        return [cls.probe_usb(index, timeout=timeout)
                for index in range(count)]

    def axis_for(self, axis: str) -> int:
        try:
            return self.axis_map[str(axis).lower()]
        except KeyError as exc:
            raise StageError(f"unknown Picomotor axis {axis!r}") from exc

    def move_axis_steps(self, axis: str, steps: int) -> None:
        """Move one mapped axis by signed integer steps and wait for settle."""
        steps = int(steps)
        if steps == 0:
            return
        motor_axis = self.axis_for(axis)
        try:
            self.dev.move_by(axis=motor_axis, steps=steps)
            self.dev.wait_move(axis=motor_axis)
        except Exception as exc:  # noqa: BLE001 - normalize driver failures
            raise StageError(f"picomotor axis {axis} move failed: {exc}") from exc

    def move_axis_to_steps(self, axis: str, position: int) -> None:
        """Move one mapped axis to an absolute controller step position."""
        motor_axis = self.axis_for(axis)
        try:
            move_to = getattr(self.dev, "move_to", None)
            if not callable(move_to):
                current = getattr(self.dev, "get_position")(axis=motor_axis)
                self.move_axis_steps(axis, int(position) - int(current))
                return
            move_to(axis=motor_axis, position=int(position))
            self.dev.wait_move(axis=motor_axis)
        except Exception as exc:  # noqa: BLE001 - normalize driver failures
            raise StageError(f"picomotor axis {axis} absolute move failed: {exc}") from exc

    def prepare_focus(self, z_safe_um: float = 5.0,
                      z_focus_um: float = 0.0) -> None:
        """Perform the fixed-beam Alg2 safe-height/focus sequence on Z."""
        steps_per_um = self.steps_per_mm_by_axis["z"] / 1000.0
        get_position = getattr(self.dev, "get_position", None)
        if not callable(get_position):
            self.move_axis_steps("z", int(round(float(z_safe_um) * steps_per_um)))
            self.move_axis_steps(
                "z", int(round((float(z_focus_um) - float(z_safe_um)) * steps_per_um)))
            return
        try:
            current = int(get_position(axis=self.z_axis))
        except Exception:
            current = 0
        origin = current
        safe = origin + int(round(float(z_safe_um) * steps_per_um))
        focus = origin + int(round(float(z_focus_um) * steps_per_um))
        self.move_axis_to_steps("z", safe)
        self.move_axis_to_steps("z", focus)

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
        dx_steps = int(round(dx_mm * self.steps_per_mm_by_axis["x"]))
        dy_steps = int(round(dy_mm * self.steps_per_mm_by_axis["y"]))
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


class PicoMotorXYZStage:
    """XYZ-stage facade for the UI backed by one 8742/8743 controller.

    The controller API uses integer motor steps while the UI motion model uses
    micrometres.  ``VirtualXYZStage`` supplies the validated XYZ profiles,
    soft limits and telemetry; its callback converts each accepted move to
    signed Picomotor steps using the detected X/Y/Z mapping.
    """

    def __init__(self, conn: int = 0, x_axis: int = 1, y_axis: int = 2,
                 z_axis: int = 3, steps_per_mm: float = 1000.0,
                 profiles: Optional[dict] = None,
                 confirmed: bool = False, max_step_mm: float = 1.0,
                 speed_steps: Optional[float] = None) -> None:
        if steps_per_mm <= 0:
            raise StageError("steps_per_mm must be > 0")
        self.driver = PicoMotorStage(
            conn=conn, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
            steps_per_mm=steps_per_mm, speed_steps=speed_steps,
            confirmed=confirmed, max_step_mm=max_step_mm)
        config = default_motion_config()
        axes = dict(config.axes)
        for name, changes in (profiles or {}).items():
            key = str(name).lower()
            if key not in axes:
                continue
            data = axes[key].to_dict()
            data.update(changes or {})
            data["name"] = key
            axes[key] = AxisMotionConfig(**data).validate()
        # Panel units are micrometres; the Picomotor driver is configured in
        # millimetres.  Keep per-axis calibration in sync with the profile.
        self.driver.steps_per_mm_by_axis.update({
            axis: axes[axis].steps_per_unit * 1000.0
            for axis in ("x", "y", "z")})
        self._stage = VirtualXYZStage(
            MotionConfig(axes=axes), on_move=self._apply_target)

    @property
    def enabled(self) -> bool:
        return self._stage.enabled

    @property
    def position(self) -> dict:
        return self._stage.position

    @property
    def config(self) -> MotionConfig:
        return self._stage.config

    @property
    def last_telemetry(self) -> Optional[MotionTelemetry]:
        return self._stage.last_telemetry

    def _apply_target(self, target: dict) -> None:
        before = self._stage.position
        delta_um = {axis: float(target[axis] - before[axis])
                    for axis in ("x", "y", "z")
                    if abs(target[axis] - before[axis]) > 1e-12}
        if not delta_um:
            return
        # A single UI move may contain more than one axis.  Keep the command
        # atomic at the validation layer, then issue each mapped motor axis.
        for axis, value_um in delta_um.items():
            steps_per_unit = self._stage.config.axis(axis).steps_per_unit
            steps = int(round(value_um * steps_per_unit))
            if steps == 0:
                continue
            self.driver.move_axis_steps(axis, steps)

    def move_by(self, delta: dict, source: str = "ui") -> MotionTelemetry:
        return self._stage.move_by(delta, source=source)

    def move_to(self, position: dict, source: str = "ui") -> MotionTelemetry:
        return self._stage.move_to(position, source=source)

    def configure_axis(self, name: str, **changes) -> None:
        self._stage.configure_axis(name, **changes)
        axis = str(name).lower()
        self.driver.steps_per_mm_by_axis[axis] = (
            self._stage.config.axis(axis).steps_per_unit * 1000.0)

    def home(self, axes=None, source: str = "home") -> MotionTelemetry:
        return self._stage.home(axes=axes, source=source)

    def zero(self, axes=None) -> None:
        names = [str(a).lower() for a in (axes or ("x", "y", "z"))]
        for name in names:
            axis = self.driver.axis_for(name)
            try:
                setter = getattr(self.driver.dev, "set_position_reference", None)
                if callable(setter):
                    setter(axis=axis, position=0)
            except Exception as exc:  # noqa: BLE001 - normalize driver errors
                raise StageError(f"picomotor zero {name} failed: {exc}") from exc
        self._stage.zero(axes=names)

    def enable(self) -> None:
        self._stage.enable()

    def stop(self) -> None:
        self._stage.stop()
        self.driver.stop_all()

    estop = stop

    def stop_all(self) -> None:
        self.driver.stop_all()

    def close(self) -> None:
        self.driver.close()
