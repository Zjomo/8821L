# control/stage34.py
from __future__ import annotations

import time
from typing import Optional, Dict, Any, Tuple

from pylablib.devices import Thorlabs


class Stage34:
    """
    Thorlabs KinesisPiezoMotor 四通道封装。

    兼容 logic/rule_ab.py 的调用方式：
        Stage34(conn=..., x_channel=..., y_channel=..., ...)
        execute_rule_action(...)
        get_position()
        stop_all()
        close()

    默认控制器序列号：
        97101208

    默认 A推动B 使用：
        x_channel = 3
        y_channel = 4

    如果你希望 A推动B 用 1/2 通道，可以在 rule_ab.py 的 RuntimeConfig 中改：
        stage_x_channel = 1
        stage_y_channel = 2
    """

    def __init__(
        self,
        conn: Optional[Any] = None,
        serial_number: str = "97101208",

        x_channel: int = 3,
        y_channel: int = 4,

        default_velocity: int = 10,
        default_acceleration: int = 10,
        default_max_voltage: int = 100,

        max_voltage: Optional[float] = None,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,

        step_x: int = 100,
        step_y: int = 100,

        x_sign: int = 1,
        y_sign: int = 1,

        auto_enable: bool = True,
        auto_setup: Optional[bool] = None,

        settle_time_s: float = 0.05,
        **kwargs,
    ):
        if isinstance(conn, str) and conn.strip():
            self.serial_number = conn.strip()
        else:
            self.serial_number = str(serial_number)

        self.x_channel = int(x_channel)
        self.y_channel = int(y_channel)

        self.step_x = int(step_x)
        self.step_y = int(step_y)

        self.x_sign = 1 if int(x_sign) >= 0 else -1
        self.y_sign = 1 if int(y_sign) >= 0 else -1

        self.max_voltage = int(round(float(default_max_voltage if max_voltage is None else max_voltage)))
        self.velocity = int(round(float(default_velocity if velocity is None else velocity)))
        self.acceleration = int(round(float(default_acceleration if acceleration is None else acceleration)))

        self.settle_time_s = float(settle_time_s)

        if auto_setup is None:
            auto_setup = bool(auto_enable)

        self.stage: Optional[Thorlabs.KinesisPiezoMotor] = None
        self.dev: Optional[Thorlabs.KinesisPiezoMotor] = None
        self.is_open = False

        self._channel_configured: Dict[int, bool] = {
            1: False,
            2: False,
            3: False,
            4: False,
        }

        # KinesisPiezoMotor 通常是开环设备，这里记录本程序发出的累计步数，
        # 供 rule_ab.py 的 get_position() 日志使用。
        self._pos_x = 0.0
        self._pos_y = 0.0

        self.open()

        if auto_setup:
            self.setup_all_channels()

    # ------------------------------------------------------------
    # 连接 / 关闭
    # ------------------------------------------------------------

    def open(self):
        if self.stage is not None and self.is_open:
            return

        print(f"[Stage34] 正在连接 Thorlabs.KinesisPiezoMotor: {self.serial_number}")
        self.stage = Thorlabs.KinesisPiezoMotor(self.serial_number)
        self.dev = self.stage
        self.is_open = True
        print("[Stage34] 连接成功")

    def close(self):
        if self.stage is None:
            self.stage = None
            self.dev = None
            self.is_open = False
            return

        try:
            self.stage.close()
            print("[Stage34] 已关闭")
        finally:
            self.stage = None
            self.dev = None
            self.is_open = False
            for ch in self._channel_configured:
                self._channel_configured[ch] = False

    def __enter__(self) -> "Stage34":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _require_open(self):
        if self.stage is None or not self.is_open:
            raise RuntimeError("Stage34 未连接，请先调用 open()")

    @staticmethod
    def _check_channel(channel: int):
        if int(channel) not in (1, 2, 3, 4):
            raise ValueError(f"channel 必须是 1/2/3/4，当前 channel={channel}")

    # ------------------------------------------------------------
    # 参数设置
    # ------------------------------------------------------------

    def setup_channel(
        self,
        channel: int,
        max_voltage: Optional[float] = None,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
    ):
        self._require_open()
        self._check_channel(channel)

        max_voltage_i = int(round(float(self.max_voltage if max_voltage is None else max_voltage)))
        velocity_i = int(round(float(self.velocity if velocity is None else velocity)))
        acceleration_i = int(round(float(self.acceleration if acceleration is None else acceleration)))

        assert self.stage is not None

        print(
            f"[Stage34 DEBUG] setup_drive CH{channel}: "
            f"max_voltage={max_voltage_i}({type(max_voltage_i)}), "
            f"velocity={velocity_i}({type(velocity_i)}), "
            f"acceleration={acceleration_i}({type(acceleration_i)})"
        )

        self.stage.setup_drive(
            max_voltage=max_voltage_i,
            velocity=velocity_i,
            acceleration=acceleration_i,
            channel=int(channel),
        )

        self._channel_configured[int(channel)] = True

        print(
            f"[Stage34] CH{channel} 参数设置完成："
            f"max_voltage={max_voltage_i}, velocity={velocity_i}, acceleration={acceleration_i}"
        )

    def setup_all_channels(self):
        for ch in (1, 2, 3, 4):
            self.setup_channel(ch)

    def ensure_channel_ready(self, channel: int):
        self._check_channel(channel)
        if not self._channel_configured[int(channel)]:
            self.setup_channel(channel)

    # ------------------------------------------------------------
    # 基础运动
    # ------------------------------------------------------------

    def move_by(
        self,
        channel: int,
        distance: float,
        wait_s: Optional[float] = None,
    ):
        self._require_open()
        self._check_channel(channel)
        self.ensure_channel_ready(channel)

        assert self.stage is not None

        distance = float(distance)
        print(f"[Stage34] CH{channel} move_by distance={distance}")

        self.stage.move_by(distance=distance, channel=int(channel))

        if int(channel) == int(self.x_channel):
            self._pos_x += distance
        if int(channel) == int(self.y_channel):
            self._pos_y += distance

        if wait_s is None:
            wait_s = self.settle_time_s

        if wait_s and wait_s > 0:
            time.sleep(float(wait_s))

    def move_positive(self, channel: int, distance: float = 100):
        self.move_by(channel=channel, distance=abs(float(distance)))

    def move_negative(self, channel: int, distance: float = 100):
        self.move_by(channel=channel, distance=-abs(float(distance)))

    # ------------------------------------------------------------
    # 兼容 rule_ab.py 的接口
    # ------------------------------------------------------------

    def execute_rule_action(
        self,
        action_id: int,
        direction: str = "",
        step: Optional[float] = None,
        wait: bool = True,
        timeout: Optional[float] = None,
    ):
        """
        action_id 定义与 rule_ab.py 一致：
            0 = STAY
            1 = UP
            2 = DOWN
            3 = LEFT
            4 = RIGHT

        默认图像坐标到平台运动映射：
            LEFT  -> x_channel 负方向
            RIGHT -> x_channel 正方向
            UP    -> y_channel 负方向
            DOWN  -> y_channel 正方向

        如果实际运动方向反了，只改 rule_ab.py / RuntimeConfig 里的：
            stage_x_sign
            stage_y_sign
        """
        if step is None:
            step = self.step_x

        step = abs(float(step))

        direction = str(direction).lower().strip()

        if action_id == 0 or direction == "stay":
            print("[Stage34] execute_rule_action: STAY，不移动")
            return

        if action_id == 3 or direction == "left":
            dist = -step * self.x_sign
            self.move_by(channel=self.x_channel, distance=dist)

        elif action_id == 4 or direction == "right":
            dist = step * self.x_sign
            self.move_by(channel=self.x_channel, distance=dist)

        elif action_id == 1 or direction == "up":
            dist = -step * self.y_sign
            self.move_by(channel=self.y_channel, distance=dist)

        elif action_id == 2 or direction == "down":
            dist = step * self.y_sign
            self.move_by(channel=self.y_channel, distance=dist)

        else:
            raise ValueError(
                f"未知 action_id={action_id}, direction={direction}，"
                "有效动作：0/STAY, 1/UP, 2/DOWN, 3/LEFT, 4/RIGHT"
            )

        if wait:
            if timeout is not None and timeout > 0:
                time.sleep(min(float(timeout), self.settle_time_s))
            else:
                time.sleep(self.settle_time_s)

    def move_channel(self, channel: int, direction: int, step_size: float):
        distance = abs(float(step_size)) if int(direction) >= 0 else -abs(float(step_size))
        self.move_by(channel=channel, distance=distance)

    def move_axis(self, axis: str, direction: int, step_size: float):
        axis = str(axis).lower().strip()

        axis_to_channel = {
            "x": self.x_channel,
            "horizontal": self.x_channel,
            "h": self.x_channel,

            "y": self.y_channel,
            "vertical": self.y_channel,
            "v": self.y_channel,

            "ch1": 1,
            "channel1": 1,
            "ch2": 2,
            "channel2": 2,
            "z": 3,
            "ch3": 3,
            "channel3": 3,
            "ch4": 4,
            "channel4": 4,
        }

        if axis not in axis_to_channel:
            raise ValueError(f"未知 axis={axis}，可用：{list(axis_to_channel.keys())}")

        channel = axis_to_channel[axis]
        self.move_channel(channel=channel, direction=direction, step_size=step_size)

    def get_position(self) -> Tuple[float, float]:
        return self._pos_x, self._pos_y

    def stop_all(self):
        if self.stage is None:
            return

        for ch in (1, 2, 3, 4):
            try:
                if hasattr(self.stage, "stop"):
                    self.stage.stop(channel=ch)
                elif hasattr(self.stage, "stop_motion"):
                    self.stage.stop_motion(channel=ch)
            except Exception:
                pass

        print("[Stage34] stop_all 已执行")

    # ------------------------------------------------------------
    # 测试函数
    # ------------------------------------------------------------

    def test_channels_1_2(self, distance: float = 100):
        self.move_by(channel=1, distance=-abs(float(distance)))
        self.move_by(channel=1, distance=abs(float(distance)))
        self.move_by(channel=2, distance=-abs(float(distance)))
        self.move_by(channel=2, distance=abs(float(distance)))

    def test_channels_3_4(self, distance: float = 100):
        self.move_by(channel=3, distance=-abs(float(distance)))
        self.move_by(channel=3, distance=abs(float(distance)))
        self.move_by(channel=4, distance=-abs(float(distance)))
        self.move_by(channel=4, distance=abs(float(distance)))


if __name__ == "__main__":
    stage = Stage34(conn="97101208", x_channel=3, y_channel=4)
    try:
        print("========== 测试 CH1 / CH2 ==========")
        stage.test_channels_1_2(distance=100)

        print("========== 测试 CH3 / CH4 ==========")
        stage.test_channels_3_4(distance=100)
    finally:
        stage.close()
