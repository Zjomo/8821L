# control/stage34.py
from __future__ import annotations

import time
from typing import Optional, Dict

from pylablib.devices import Thorlabs


class Stage34:
    """
    Thorlabs KinesisPiezoMotor 四通道封装类。

    对应你的控制器：
        serial_number = "97101208"

    通道对应：
        channel 1
        channel 2
        channel 3
        channel 4

    注意：
        1. 不要在不同地方重复打开同一个控制器；
        2. 这个类只连接一次 KinesisPiezoMotor；
        3. 1/2/3/4 四个通道都通过同一个 self.stage 控制。
    """

    def __init__(
        self,
        serial_number: str = "97101208",
        max_voltage: float = 100,
        velocity: float = 500,
        acceleration: float = 500,
        auto_setup: bool = True,
        settle_time_s: float = 0.05,
    ):
        self.serial_number = str(serial_number)
        self.max_voltage = float(max_voltage)
        self.velocity = float(velocity)
        self.acceleration = float(acceleration)
        self.settle_time_s = float(settle_time_s)

        self.stage: Optional[Thorlabs.KinesisPiezoMotor] = None
        self.is_open = False

        # 记录每个通道是否已经 setup_drive
        self._channel_configured: Dict[int, bool] = {
            1: False,
            2: False,
            3: False,
            4: False,
        }

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
        self.is_open = True
        print("[Stage34] 连接成功")

    def close(self):
        if self.stage is None:
            self.is_open = False
            return

        try:
            self.stage.close()
            print("[Stage34] 已关闭")
        finally:
            self.stage = None
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

    # ------------------------------------------------------------
    # 参数设置
    # ------------------------------------------------------------

    @staticmethod
    def _check_channel(channel: int):
        if int(channel) not in (1, 2, 3, 4):
            raise ValueError(f"channel 必须是 1/2/3/4，当前 channel={channel}")

    def setup_channel(
        self,
        channel: int,
        max_voltage: Optional[float] = None,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
    ):
        """
        设置单个通道参数。
        """
        self._require_open()
        self._check_channel(channel)

        max_voltage = self.max_voltage if max_voltage is None else float(max_voltage)
        velocity = self.velocity if velocity is None else float(velocity)
        acceleration = self.acceleration if acceleration is None else float(acceleration)

        assert self.stage is not None

        self.stage.setup_drive(
            max_voltage=max_voltage,
            velocity=velocity,
            acceleration=acceleration,
            channel=int(channel),
        )

        self._channel_configured[int(channel)] = True

        print(
            f"[Stage34] CH{channel} 参数设置完成："
            f"max_voltage={max_voltage}, velocity={velocity}, acceleration={acceleration}"
        )

    def setup_all_channels(self):
        """
        初始化 1/2/3/4 四个通道。
        """
        for ch in (1, 2, 3, 4):
            self.setup_channel(ch)

    def ensure_channel_ready(self, channel: int):
        self._check_channel(channel)
        if not self._channel_configured[int(channel)]:
            self.setup_channel(channel)

    # ------------------------------------------------------------
    # 基础运动
    # ------------------------------------------------------------

    def move_by(self, channel: int, distance: float, wait_s: Optional[float] = None):
        """
        指定通道相对运动。

        参数：
            channel: 1/2/3/4
            distance: 正负步长，例如 -100 或 +100
        """
        self._require_open()
        self._check_channel(channel)
        self.ensure_channel_ready(channel)

        assert self.stage is not None

        print(f"[Stage34] CH{channel} move_by distance={distance}")
        self.stage.move_by(distance=float(distance), channel=int(channel))

        if wait_s is None:
            wait_s = self.settle_time_s

        if wait_s and wait_s > 0:
            time.sleep(float(wait_s))

    def move_positive(self, channel: int, distance: float = 100):
        """
        指定通道正向运动。
        """
        self.move_by(channel=channel, distance=abs(float(distance)))

    def move_negative(self, channel: int, distance: float = 100):
        """
        指定通道反向运动。
        """
        self.move_by(channel=channel, distance=-abs(float(distance)))

    # ------------------------------------------------------------
    # 兼容 RuleAB / RuleAC 可能使用的方向接口
    # ------------------------------------------------------------

    def move_channel(self, channel: int, direction: int, step_size: float):
        """
        通用方向运动接口。

        direction:
            +1 -> 正向
            -1 -> 反向
        """
        if int(direction) >= 0:
            distance = abs(float(step_size))
        else:
            distance = -abs(float(step_size))

        self.move_by(channel=channel, distance=distance)

    def move_axis(self, axis: str, direction: int, step_size: float):
        """
        轴名运动接口。

        默认映射：
            horizontal -> channel 1
            vertical   -> channel 2
            ch3        -> channel 3
            ch4        -> channel 4

        你后续如果希望 horizontal/vertical 对应别的通道，只需要改这里。
        """
        axis = str(axis).lower().strip()

        axis_to_channel = {
            "x": 1,
            "horizontal": 1,
            "h": 1,

            "y": 2,
            "vertical": 2,
            "v": 2,

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

    # ------------------------------------------------------------
    # 按你的原始测试程序封装成两个测试函数
    # ------------------------------------------------------------

    def test_channels_1_2(self, distance: float = 100):
        """
        对应你的第一个测试程序：
            CH1 -100/+100
            CH2 -100/+100
        """
        self.setup_channel(1)
        self.move_by(channel=1, distance=-abs(float(distance)))
        self.move_by(channel=1, distance=abs(float(distance)))

        self.setup_channel(2)
        self.move_by(channel=2, distance=-abs(float(distance)))
        self.move_by(channel=2, distance=abs(float(distance)))

    def test_channels_3_4(self, distance: float = 100):
        """
        对应你的第二个测试程序：
            CH3 -100/+100
            CH4 -100/+100
        """
        self.setup_channel(3)
        self.move_by(channel=3, distance=-abs(float(distance)))
        self.move_by(channel=3, distance=abs(float(distance)))

        self.setup_channel(4)
        self.move_by(channel=4, distance=-abs(float(distance)))
        self.move_by(channel=4, distance=abs(float(distance)))


if __name__ == "__main__":
    # 单独运行本文件时，用于测试 1/2/3/4 四个通道。
    stage = Stage34(serial_number="97101208")

    try:
        print("========== 测试 CH1 / CH2 ==========")
        stage.test_channels_1_2(distance=100)

        print("========== 测试 CH3 / CH4 ==========")
        stage.test_channels_3_4(distance=100)

    finally:
        stage.close()