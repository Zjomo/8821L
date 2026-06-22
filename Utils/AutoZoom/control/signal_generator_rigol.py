# control/signal_generator_rigol.py
from __future__ import annotations

import time
from typing import Optional, Literal

import pyvisa


WaveformType = Literal["SIN", "SQU", "RAMP", "PULS", "NOIS", "DC", "USER"]


class RigolDG4062Controller:
    """
    Rigol DG4062 信号发生器控制模块。

    支持功能：
        1. 连接 / 关闭 Rigol DG4062
        2. 打开 / 关闭指定通道输出
        3. 查询输出状态
        4. 设置波形类型
        5. 设置频率
        6. 设置高电平 / 低电平
        7. 设置脉冲占空比
        8. 设置脉冲延时
        9. 设置脉冲上升沿 / 下降沿时间
        10. 一次性设置完整 Pulse 参数

    LabVIEW 前面板对应关系：
        低电平 / V       -> set_voltage_levels(low_v, high_v)
        高电平 / V       -> set_voltage_levels(low_v, high_v)
        频率 / Hz        -> set_frequency(freq_hz)
        占空比 / %       -> set_pulse_duty(duty_percent)
        延时 / s         -> set_pulse_delay(delay_s)
        输出信号波形     -> set_waveform("PULS")
        上升下降沿 / s   -> set_pulse_edge_time(...)
        Output           -> on() / off()
    """

    def __init__(
        self,
        resource_name: str = "USB0::0x1AB1::0x0641::DG4E192200870::INSTR",
        channel: int = 1,
        timeout_ms: int = 3000,
        command_delay_s: float = 0.05,
        auto_connect: bool = False,
    ):
        if channel not in (1, 2):
            raise ValueError("Rigol DG4062 channel 必须是 1 或 2")

        self.resource_name = resource_name
        self.channel = channel
        self.timeout_ms = timeout_ms
        self.command_delay_s = command_delay_s

        self.rm: Optional[pyvisa.ResourceManager] = None
        self.instrument = None

        if auto_connect:
            self.open()

    # =========================
    # 基础连接
    # =========================

    def open(self) -> None:
        """
        打开 Rigol 连接。
        建议任务开始时打开一次，不要每次 on/off 都重新连接。
        """
        if self.instrument is not None:
            return

        self.rm = pyvisa.ResourceManager()

        print("[RigolDG4062] 可用设备：", self.rm.list_resources())

        self.instrument = self.rm.open_resource(self.resource_name)
        self.instrument.timeout = self.timeout_ms

        try:
            idn = self.instrument.query("*IDN?").strip()
        except Exception:
            idn = "UNKNOWN"

        print(f"[RigolDG4062] 连接成功: {idn}")

    def close(self) -> None:
        """
        关闭 Rigol 连接。
        """
        if self.instrument is not None:
            try:
                self.instrument.close()
            finally:
                self.instrument = None
                print("[RigolDG4062] 设备连接已关闭")

        if self.rm is not None:
            try:
                self.rm.close()
            except Exception:
                pass
            finally:
                self.rm = None

    def _ensure_open(self) -> None:
        if self.instrument is None:
            self.open()

    def write(self, command: str) -> None:
        """
        发送 SCPI 写命令。
        """
        self._ensure_open()

        print(f"[RigolDG4062] WRITE: {command}")
        self.instrument.write(command)

        if self.command_delay_s > 0:
            time.sleep(self.command_delay_s)

    def query(self, command: str) -> str:
        """
        发送 SCPI 查询命令。
        """
        self._ensure_open()

        print(f"[RigolDG4062] QUERY: {command}")
        result = self.instrument.query(command)

        if self.command_delay_s > 0:
            time.sleep(self.command_delay_s)

        return result

    # =========================
    # 通道选择
    # =========================

    def _ch(self, channel: Optional[int] = None) -> int:
        """
        获取当前要控制的通道。
        """
        ch = self.channel if channel is None else channel

        if ch not in (1, 2):
            raise ValueError("channel 必须是 1 或 2")

        return ch

    # =========================
    # 输出开关
    # =========================

    def on(self, channel: Optional[int] = None) -> None:
        """
        开启指定通道输出。
        """
        ch = self._ch(channel)
        self.write(f":OUTP{ch}:STAT ON")
        print(f"[RigolDG4062] 通道 {ch} 输出已开启")

    def off(self, channel: Optional[int] = None) -> None:
        """
        关闭指定通道输出。
        """
        ch = self._ch(channel)
        self.write(f":OUTP{ch}:STAT OFF")
        print(f"[RigolDG4062] 通道 {ch} 输出已关闭")

    def on_all(self) -> None:
        """
        同时开启 CH1 和 CH2。
        """
        self.on(channel=1)
        self.on(channel=2)

    def off_all(self) -> None:
        """
        同时关闭 CH1 和 CH2。
        """
        self.off(channel=1)
        self.off(channel=2)

    def get_status(self, channel: Optional[int] = None) -> int:
        """
        获取指定通道输出状态。

        返回：
            1 = ON
            0 = OFF
        """
        ch = self._ch(channel)
        status = self.query(f":OUTP{ch}:STAT?").strip().upper()

        if status in ("ON", "1"):
            return 1
        return 0

    # =========================
    # 波形参数设置
    # =========================

    def set_waveform(
        self,
        waveform: WaveformType = "PULS",
        channel: Optional[int] = None,
    ) -> None:
        """
        设置波形类型。

        常用：
            "SIN"  = 正弦波
            "SQU"  = 方波
            "RAMP" = 斜波
            "PULS" = 脉冲波
            "DC"   = 直流
        """
        ch = self._ch(channel)
        self.write(f":SOUR{ch}:FUNC {waveform}")
        print(f"[RigolDG4062] 通道 {ch} 波形已设置为 {waveform}")

    def set_frequency(
        self,
        freq_hz: float,
        channel: Optional[int] = None,
    ) -> None:
        """
        设置频率，单位 Hz。

        例如：
            6000 Hz -> 6.000k
        """
        if freq_hz <= 0:
            raise ValueError("freq_hz 必须大于 0")

        ch = self._ch(channel)
        self.write(f":SOUR{ch}:FREQ {freq_hz}")
        print(f"[RigolDG4062] 通道 {ch} 频率已设置为 {freq_hz} Hz")

    def set_voltage_levels(
        self,
        low_v: float,
        high_v: float,
        channel: Optional[int] = None,
    ) -> None:
        """
        设置低电平和高电平，单位 V。

        对应 LabVIEW：
            低电平 / V
            高电平 / V

        注意：
            high_v 必须大于 low_v。
        """
        if high_v <= low_v:
            raise ValueError("high_v 必须大于 low_v")

        ch = self._ch(channel)

        self.write(f":SOUR{ch}:VOLT:LOW {low_v}")
        self.write(f":SOUR{ch}:VOLT:HIGH {high_v}")

        print(
            f"[RigolDG4062] 通道 {ch} 电平已设置: "
            f"LOW={low_v} V, HIGH={high_v} V"
        )

    def set_pulse_duty(
        self,
        duty_percent: float,
        channel: Optional[int] = None,
    ) -> None:
        """
        设置 Pulse 占空比，单位 %。

        对应 LabVIEW：
            占空比 / %

        例如：
            3.0 表示 3%
        """
        if not (0 < duty_percent < 100):
            raise ValueError("duty_percent 必须在 0 到 100 之间")

        ch = self._ch(channel)
        self.write(f":SOUR{ch}:PULS:DCYC {duty_percent}")
        print(f"[RigolDG4062] 通道 {ch} 脉冲占空比已设置为 {duty_percent}%")

    def set_pulse_delay(
        self,
        delay_s: float,
        channel: Optional[int] = None,
    ) -> None:
        """
        设置 Pulse 延时，单位 s。

        对应 LabVIEW：
            延时 / s
        """
        if delay_s < 0:
            raise ValueError("delay_s 不能小于 0")

        ch = self._ch(channel)
        self.write(f":SOUR{ch}:PULS:DEL {delay_s}")
        print(f"[RigolDG4062] 通道 {ch} 脉冲延时已设置为 {delay_s} s")

    def set_pulse_edge_time(
        self,
        edge_time_s: float,
        edge: Literal["BOTH", "RISE", "FALL"] = "BOTH",
        channel: Optional[int] = None,
    ) -> None:
        """
        设置 Pulse 上升沿 / 下降沿时间，单位 s。

        对应 LabVIEW：
            上升下降沿 / s

        edge:
            "BOTH" = 同时设置上升沿和下降沿
            "RISE" = 只设置上升沿
            "FALL" = 只设置下降沿

        注意：
            不同 Rigol 固件的 SCPI 命令可能略有差异。
            常见命令为：
                :SOUR1:PULS:TRAN 1e-6
            或：
                :SOUR1:PULS:TRAN:LEAD 1e-6
                :SOUR1:PULS:TRAN:TRA 1e-6
        """
        if edge_time_s < 0:
            raise ValueError("edge_time_s 不能小于 0")

        ch = self._ch(channel)

        if edge == "BOTH":
            self.write(f":SOUR{ch}:PULS:TRAN {edge_time_s}")
        elif edge == "RISE":
            self.write(f":SOUR{ch}:PULS:TRAN:LEAD {edge_time_s}")
        elif edge == "FALL":
            self.write(f":SOUR{ch}:PULS:TRAN:TRA {edge_time_s}")
        else:
            raise ValueError("edge 必须是 BOTH、RISE 或 FALL")

        print(
            f"[RigolDG4062] 通道 {ch} 脉冲边沿时间已设置: "
            f"edge={edge}, edge_time={edge_time_s} s"
        )

    # =========================
    # 一次性设置完整 Pulse
    # =========================

    def configure_pulse(
        self,
        low_v: float,
        high_v: float,
        freq_hz: float,
        duty_percent: float,
        delay_s: float = 0.0,
        edge_time_s: Optional[float] = None,
        edge: Literal["BOTH", "RISE", "FALL"] = "BOTH",
        output_on: bool = False,
        channel: Optional[int] = None,
    ) -> None:
        """
        一次性配置 Pulse 波形参数。

        对应 LabVIEW 前面板中的一整组参数：
            输出信号波形 = Pulse
            频率 / Hz
            低电平 / V
            高电平 / V
            占空比 / %
            延时 / s
            上升下降沿 / s
            Output Enable

        参数：
            low_v:
                低电平，单位 V

            high_v:
                高电平，单位 V

            freq_hz:
                频率，单位 Hz

            duty_percent:
                占空比，单位 %

            delay_s:
                延时，单位 s

            edge_time_s:
                上升/下降沿时间，单位 s。
                如果为 None，则不主动设置边沿时间。

            edge:
                BOTH / RISE / FALL

            output_on:
                True  = 配置完成后打开输出
                False = 配置完成后不打开输出
        """
        ch = self._ch(channel)

        print(f"[RigolDG4062] 开始配置通道 {ch} Pulse 参数")

        # 建议先关输出，再改参数，避免改参数过程中输出异常
        self.off(channel=ch)

        self.set_waveform("PULS", channel=ch)
        self.set_frequency(freq_hz, channel=ch)
        self.set_voltage_levels(low_v=low_v, high_v=high_v, channel=ch)
        self.set_pulse_duty(duty_percent=duty_percent, channel=ch)
        self.set_pulse_delay(delay_s=delay_s, channel=ch)

        if edge_time_s is not None:
            self.set_pulse_edge_time(
                edge_time_s=edge_time_s,
                edge=edge,
                channel=ch,
            )

        if output_on:
            self.on(channel=ch)

        print(f"[RigolDG4062] 通道 {ch} Pulse 参数配置完成")

    # =========================
    # 查询常用参数
    # =========================

    def get_frequency(self, channel: Optional[int] = None) -> float:
        ch = self._ch(channel)
        return float(self.query(f":SOUR{ch}:FREQ?").strip())

    def get_waveform(self, channel: Optional[int] = None) -> str:
        ch = self._ch(channel)
        return self.query(f":SOUR{ch}:FUNC?").strip()

    def get_voltage_low(self, channel: Optional[int] = None) -> float:
        ch = self._ch(channel)
        return float(self.query(f":SOUR{ch}:VOLT:LOW?").strip())

    def get_voltage_high(self, channel: Optional[int] = None) -> float:
        ch = self._ch(channel)
        return float(self.query(f":SOUR{ch}:VOLT:HIGH?").strip())

    def get_pulse_duty(self, channel: Optional[int] = None) -> float:
        ch = self._ch(channel)
        return float(self.query(f":SOUR{ch}:PULS:DCYC?").strip())

    def get_pulse_delay(self, channel: Optional[int] = None) -> float:
        ch = self._ch(channel)
        return float(self.query(f":SOUR{ch}:PULS:DEL?").strip())

    # =========================
    # 上下文管理器
    # =========================

    def __enter__(self) -> "RigolDG4062Controller":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()