"""核心抽象层。"""

from pi_spectrometer.core.base import SpectrometerBackend
from pi_spectrometer.core.types import SpectrometerResult, ROI
from pi_spectrometer.core.exceptions import PICamError, SpectrometerError, NotConnectedError

__all__ = [
    "SpectrometerBackend",
    "SpectrometerResult",
    "ROI",
    "PICamError",
    "SpectrometerError",
    "NotConnectedError",
]
