"""光谱仪后端抽象基类。"""

from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np

from pi_spectrometer.core.types import ROI, SpectrometerResult


class SpectrometerBackend(ABC):
    """所有光谱仪后端的抽象基类。

    该接口屏蔽硬件细节，使上层工作流可以在不修改业务逻辑的情况下切换光谱仪。
    """

    name: str = "base"

    def __init__(self):
        self._connected = False
        self._exposure_seconds: float = 0.1
        self._sensor_temperature: Optional[float] = None
        self._roi: ROI = ROI()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    @abstractmethod
    def connect(self) -> bool:
        """连接设备。返回是否成功。"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """断开设备连接。"""
        ...

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # 参数配置
    # ------------------------------------------------------------------
    @abstractmethod
    def set_exposure(self, seconds: float) -> None:
        """设置曝光时间（秒）。"""
        ...

    def get_exposure(self) -> float:
        return self._exposure_seconds

    @abstractmethod
    def set_sensor_temperature(self, celsius: float) -> None:
        """设置传感器目标温度（摄氏度）。"""
        ...

    def get_sensor_temperature(self) -> Optional[float]:
        return self._sensor_temperature

    @abstractmethod
    def set_roi(self, roi: ROI) -> None:
        """设置 ROI。"""
        ...

    def get_roi(self) -> ROI:
        return self._roi

    # ------------------------------------------------------------------
    # 采集
    # ------------------------------------------------------------------
    @abstractmethod
    def acquire(self, num_frames: int = 1) -> SpectrometerResult:
        """执行一次采集。"""
        ...

    @abstractmethod
    def get_wavelength_axis(self) -> Optional[np.ndarray]:
        """获取波长横坐标（nm），如果不支持则返回 None。"""
        ...

    # ------------------------------------------------------------------
    # 设备能力查询
    # ------------------------------------------------------------------
    def supports_temperature(self) -> bool:
        return True

    def supports_roi(self) -> bool:
        return True

    def supports_wavelength_axis(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # 上下文管理
    # ------------------------------------------------------------------
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
