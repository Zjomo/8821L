import time
import serial


class IlluminationRelay:
    """
    串口继电器照明控制模块。

    对应你的测试代码：
        开灯: A0 01 00 A1
        关灯: A0 01 01 A2

    用法：
        light = IlluminationRelay(port="COM13")
        light.open()
        light.on()
        light.off()
        light.close()
    """

    CMD_ON = bytes.fromhex("A0 01 00 A1")
    CMD_OFF = bytes.fromhex("A0 01 01 A2")

    def __init__(
        self,
        port: str = "COM13",
        baudrate: int = 9600,
        timeout: float = 1.0,
        command_delay_s: float = 0.05,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.command_delay_s = command_delay_s
        self.ser = None

    def open(self) -> None:
        """
        打开串口。
        建议在程序启动时打开一次，不要每次开关灯都反复打开/关闭串口。
        """
        if self.ser is not None and self.ser.is_open:
            return

        self.ser = serial.Serial(
            self.port,
            self.baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=self.timeout,
        )

        print(f"[IlluminationRelay] 串口已打开: {self.port}, is_open={self.ser.is_open}")

    def close(self) -> None:
        """
        关闭串口。
        建议在整个任务结束时关闭。
        """
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            print(f"[IlluminationRelay] 串口已关闭: {self.port}")

    def _write_cmd(self, cmd: bytes, name: str) -> None:
        """
        发送串口命令。
        """
        if self.ser is None or not self.ser.is_open:
            self.open()

        print(f"[IlluminationRelay] 发送{name}: {cmd.hex(' ').upper()}")

        self.ser.write(cmd)
        self.ser.flush()

        if self.command_delay_s > 0:
            time.sleep(self.command_delay_s)

    def on(self) -> None:
        """
        开照明光。
        """
        self._write_cmd(self.CMD_ON, "开灯")

    def off(self) -> None:
        """
        关照明光。
        """
        self._write_cmd(self.CMD_OFF, "关灯")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()