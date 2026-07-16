"""软件模拟相机后端。

无需安装 PICam SDK 和真实硬件即可测试上层逻辑。
支持与 IsoPlane 单色仪集成进行波长轴计算。
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from pi_spectrometer.core.base import SpectrometerBackend
from pi_spectrometer.core.types import ROI, SpectrometerResult
from pi_spectrometer.picam.camera import PICamCamera

if TYPE_CHECKING:
    from pi_spectrometer.picam.isoplane import IsoPlaneBackend


class DemoCamera(PICamCamera):
    """使用 PICam 软件模拟相机的便捷后端。

    继承 PICamCamera 的所有功能，但使用软件模拟相机。
    可选集成 IsoPlane 单色仪以获得波长轴。
    """

    name = "picam_demo"

    def __init__(
        self,
        dll_path: Optional[str] = None,
        demo_model: int = 1200,
        demo_serial: str = "Demo-1",
        isoplane: Optional["IsoPlaneBackend"] = None,
        center_wavelength_nm: float = 550.0,
        grating_index: int = 1,
    ):
        """
        参数：
            dll_path: Picam.dll 路径（仍需 PICam SDK）
            demo_model: 模拟相机型号
            demo_serial: 模拟相机序列号
            isoplane: 可选的 IsoPlane 单色仪后端
            center_wavelength_nm: 模拟中心波长（无 IsoPlane 时使用）
            grating_index: 模拟光栅索引（无 IsoPlane 时使用）
        """
        super().__init__(
            dll_path=dll_path,
            camera_index=0,
            demo=True,
            demo_model=demo_model,
            demo_serial=demo_serial,
            isoplane=isoplane,
        )
        self._center_wavelength_nm = center_wavelength_nm
        self._grating_index = grating_index

    def get_wavelength_axis(self) -> Optional[np.ndarray]:
        """获取波长轴。

        优先级：
            1. 使用 IsoPlane 后端（如果有）
            2. 否则使用模拟的波长轴
        """
        # 先尝试父类方法（使用 IsoPlane）
        result = super().get_wavelength_axis()
        if result is not None:
            return result

        # 模拟波长轴
        self._ensure_connected()
        try:
            width, height = self._binding._get_frame_shape(self._handle)
        except Exception:
            width = 1024

        # 使用 IsoPlane 计算方法（即使没有真实单色仪）
        from pi_spectrometer.picam.isoplane import IsoPlaneBackend

        focal_length_mm = IsoPlaneBackend.FOCAL_LENGTH_MM
        grating_info = IsoPlaneBackend.GRATINGS.get(self._grating_index, {"density": 300})
        groove_density = grating_info["density"]

        dispersion_nm_per_mm = 1e6 / (groove_density * focal_length_mm)
        pixel_size_mm = 13.0 / 1000.0  # 典型像素宽度

        half_width_nm = (width / 2) * pixel_size_mm * dispersion_nm_per_mm

        return np.linspace(
            self._center_wavelength_nm - half_width_nm,
            self._center_wavelength_nm + half_width_nm,
            width
        )


class MockSpectrometerBackend(SpectrometerBackend):
    """完全脱离 PICam DLL 的模拟光谱仪后端。

    用于在没有任何 PI SDK 的环境中运行单元测试。
    """

    name = "mock"

    def __init__(
        self,
        num_points: int = 1024,
        center_nm: float = 516.0,
        sigma_nm: float = 18.0,
        noise: float = 15.0,
    ):
        super().__init__()
        self.num_points = num_points
        self.center_nm = center_nm
        self.sigma_nm = sigma_nm
        self.noise = noise

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def set_exposure(self, seconds: float) -> None:
        self._exposure_seconds = float(seconds)

    def get_exposure(self) -> float:
        return self._exposure_seconds

    def set_sensor_temperature(self, celsius: float) -> None:
        self._sensor_temperature = float(celsius)

    def get_sensor_temperature(self) -> Optional[float]:
        return self._sensor_temperature

    def set_roi(self, roi) -> None:
        self._roi = roi

    def get_roi(self):
        return self._roi

    def get_wavelength_axis(self) -> Optional[np.ndarray]:
        return np.linspace(self.center_nm - 50, self.center_nm + 50, self.num_points)

    def acquire(self, num_frames: int = 1):
        from pi_spectrometer.core.types import SpectrometerResult
        from pi_spectrometer.processing import filters, peaks

        x = self.get_wavelength_axis()
        y = 1000.0 * np.exp(-0.5 * ((x - self.center_nm) / self.sigma_nm) ** 2)
        y += np.random.normal(0, self.noise, size=x.shape)

        threshold_values = filters.remove_above_threshold(
            y.tolist(), filters.DEFAULT_REMOVE_ABOVE
        )
        median_values = filters.median_filter_1d(
            threshold_values, filters.DEFAULT_MEDIAN_WINDOW
        )
        median_array = np.array(median_values, dtype=np.float64)

        fit_peak, fit_params = peaks.find_peak(y)

        return SpectrometerResult(
            ok=True,
            num_points=len(y),
            raw_y=y,
            fit_y=median_array,
            wavelength=x,
            raw_original_peak=float(np.max(y)),
            raw_filtered_peak=float(max(threshold_values)) if threshold_values else None,
            raw_median_peak=float(np.max(median_array)) if median_array.size else None,
            raw_peak=float(np.max(median_array)) if median_array.size else None,
            fit_peak=fit_peak,
            fit_params=fit_params,
            metadata={
                "exposure_s": self._exposure_seconds,
                "temperature_c": self._sensor_temperature,
                "simulated": True,
            },
        )
