"""
Princeton Instruments 光谱仪 Python 控制包。

提供统一的 SpectrometerBackend 抽象接口，以及基于 PICam SDK 的直接控制实现。
"""

__version__ = "0.1.0"
__all__ = [
    "SpectrometerBackend",
    "SpectrometerResult",
    "PICamError",
    "PICamCamera",
    "DemoCamera",
]

from pi_spectrometer.core.base import SpectrometerBackend
from pi_spectrometer.core.types import SpectrometerResult
from pi_spectrometer.core.exceptions import PICamError
from pi_spectrometer.picam.camera import PICamCamera
from pi_spectrometer.picam.demo import DemoCamera
