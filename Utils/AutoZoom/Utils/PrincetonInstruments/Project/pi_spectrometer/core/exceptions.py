"""核心异常类。"""


class SpectrometerError(Exception):
    """光谱仪通用错误。"""

    def __init__(self, message: str, error_code: int = 0):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class NotConnectedError(SpectrometerError):
    """设备未连接时抛出的错误。"""

    def __init__(self, message: str = "光谱仪尚未连接"):
        super().__init__(message)


class PICamError(SpectrometerError):
    """PICam SDK 返回的错误。"""

    def __init__(self, message: str, error_code: int = 0):
        super().__init__(message, error_code)
