"""PICam 相机/光谱仪后端实现。

支持与 IsoPlane 单色仪集成：
    - 可选连接 IsoPlane 后端获取波长轴
    - 采集时同步波长信息到结果
"""

from __future__ import annotations

import time
from typing import Optional, List, Any, TYPE_CHECKING

import numpy as np

from pi_spectrometer.core.base import SpectrometerBackend
from pi_spectrometer.core.exceptions import PICamError, NotConnectedError
from pi_spectrometer.core.types import ROI, SpectrometerResult
from pi_spectrometer.picam.binding import PICamBinding, PicamCameraID, PicamHandle
from pi_spectrometer.picam import constants as pic
from pi_spectrometer.processing import filters, peaks

if TYPE_CHECKING:
    from pi_spectrometer.picam.isoplane import IsoPlaneBackend


class PICamCamera(SpectrometerBackend):
    """基于 PICam SDK 的 PI 相机/光谱仪后端。

    可选集成 IsoPlane 单色仪：
        - 提供波长轴校准
        - 同步波长设置
    """

    name = "picam"

    def __init__(
        self,
        dll_path: Optional[str] = None,
        camera_index: int = 0,
        demo: bool = False,
        demo_model: int = 1200,
        demo_serial: str = "Demo-1",
        isoplane: Optional["IsoPlaneBackend"] = None,
    ):
        """
        参数：
            dll_path: Picam.dll 路径
            camera_index: 相机索引
            demo: 是否使用软件模拟相机
            demo_model: 模拟相机型号
            demo_serial: 模拟相机序列号
            isoplane: 可选的 IsoPlane 单色仪后端，用于波长轴计算
        """
        super().__init__()
        self.dll_path = dll_path
        self.camera_index = camera_index
        self.demo = demo
        self.demo_model = demo_model
        self.demo_serial = demo_serial
        self._isoplane = isoplane

        self._binding: Optional[PICamBinding] = None
        self._handle: Optional[PicamHandle] = None
        self._camera_id: Optional[PicamCameraID] = None
        self._demo_connected: bool = False

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        if self._connected:
            return True

        self._binding = PICamBinding(self.dll_path)

        if self.demo:
            self._camera_id = self._binding.connect_demo_camera(
                self.demo_model, self.demo_serial
            )
            self._demo_connected = True
        else:
            cameras = self._binding.get_available_cameras()
            if not cameras:
                raise NotConnectedError("未找到可用的 PI 相机")
            if self.camera_index >= len(cameras):
                raise NotConnectedError(
                    f"相机索引 {self.camera_index} 超出范围（共 {len(cameras)} 台）"
                )
            self._camera_id = cameras[self.camera_index]

        self._handle = self._binding.open_camera(self._camera_id)
        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._binding is None:
            return

        if self._handle is not None:
            try:
                self._binding.close_camera(self._handle)
            except Exception:
                pass
            self._handle = None

        if self._demo_connected and self._camera_id is not None:
            try:
                self._binding.disconnect_demo_camera(self._camera_id)
            except Exception:
                pass
            self._demo_connected = False

        self._camera_id = None
        self._connected = False

    def _ensure_connected(self) -> None:
        if not self._connected or self._handle is None:
            raise NotConnectedError("PI 相机未连接")

    # ------------------------------------------------------------------
    # 参数配置
    # ------------------------------------------------------------------
    def set_exposure(self, seconds: float) -> None:
        self._ensure_connected()
        ms = float(seconds) * 1000.0
        self._binding.set_parameter_float(
            self._handle, pic.PicamParameter.ExposureTime, ms
        )
        self._exposure_seconds = float(seconds)
        self._commit()

    def get_exposure(self) -> float:
        self._ensure_connected()
        try:
            ms = self._binding.get_parameter_float(
                self._handle, pic.PicamParameter.ExposureTime
            )
            self._exposure_seconds = ms / 1000.0
        except PICamError:
            pass
        return self._exposure_seconds

    def set_sensor_temperature(self, celsius: float) -> None:
        self._ensure_connected()
        self._binding.set_parameter_float(
            self._handle, pic.PicamParameter.SensorTemperatureSetPoint, float(celsius)
        )
        self._commit()

    def get_sensor_temperature(self) -> Optional[float]:
        self._ensure_connected()
        try:
            temp = self._binding.get_parameter_float(
                self._handle, pic.PicamParameter.SensorTemperatureReading
            )
            self._sensor_temperature = float(temp)
            return self._sensor_temperature
        except PICamError:
            return None

    def set_roi(self, roi: ROI) -> None:
        self._ensure_connected()
        self._binding.set_roi(
            self._handle,
            [(roi.x, roi.width, roi.x_bin, roi.y, roi.height, roi.y_bin)],
        )
        self._roi = roi
        self._commit()

    def _commit(self) -> None:
        self._ensure_connected()
        self._binding.commit_parameters(self._handle)

    # ------------------------------------------------------------------
    # 采集
    # ------------------------------------------------------------------
    def acquire(self, num_frames: int = 1) -> SpectrometerResult:
        self._ensure_connected()

        # 设置帧数
        # PICam 的 readout_count 即帧数，通过同步采集获取
        data = self._binding.acquire(
            self._handle, readout_count=num_frames, timeout_ms=-1
        )

        # 对于光谱应用，通常沿 x 方向对 ROI 行求和
        if data.ndim == 3:
            spectrum = np.mean(data[0], axis=0)
        elif data.ndim == 2:
            spectrum = np.mean(data, axis=0)
        else:
            spectrum = data

        wavelength = self.get_wavelength_axis()

        # 数据处理
        threshold_values = filters.remove_above_threshold(
            spectrum.tolist(), filters.DEFAULT_REMOVE_ABOVE
        )
        median_values = filters.median_filter_1d(
            threshold_values, filters.DEFAULT_MEDIAN_WINDOW
        )
        median_array = np.array(median_values, dtype=np.float64)

        raw_peak = float(np.max(spectrum)) if spectrum.size else None
        raw_filtered_peak = float(max(threshold_values)) if threshold_values else None
        raw_median_peak = float(np.max(median_array)) if median_array.size else None
        raw_peak_for_workflow = raw_median_peak

        fit_peak, fit_params = peaks.find_peak(spectrum)

        result = SpectrometerResult(
            ok=True,
            num_points=int(spectrum.size),
            raw_y=spectrum,
            fit_y=median_array,
            wavelength=wavelength,
            raw_original_peak=raw_peak,
            raw_filtered_peak=raw_filtered_peak,
            raw_median_peak=raw_median_peak,
            raw_peak=raw_peak_for_workflow,
            fit_peak=fit_peak,
            fit_params=fit_params,
            metadata={
                "exposure_s": self._exposure_seconds,
                "temperature_c": self._sensor_temperature,
            },
        )
        return result

    def get_wavelength_axis(self) -> Optional[np.ndarray]:
        """获取波长轴。

        优先级：
            1. 如果有 IsoPlane 后端，使用其波长轴计算
            2. 否则返回 None，由上层通过外部 xlsx 补充
        """
        self._ensure_connected()

        if self._isoplane is not None:
            try:
                # 获取当前帧宽度
                width, height = self._binding._get_frame_shape(self._handle)

                # 使用 IsoPlane 计算波长轴
                return self._isoplane.calculate_wavelength_axis(
                    num_pixels=width,
                    center_wavelength_nm=self._isoplane.get_wavelength(),
                    grating_index=self._isoplane.get_grating(),
                )
            except Exception:
                pass

        # PICam 没有直接的波长参数
        return None

    # ------------------------------------------------------------------
    # IsoPlane 集成
    # ------------------------------------------------------------------
    def set_isoplane(self, isoplane: Optional["IsoPlaneBackend"]) -> None:
        """设置 IsoPlane 单色仪后端。"""
        self._isoplane = isoplane

    def get_isoplane(self) -> Optional["IsoPlaneBackend"]:
        """获取 IsoPlane 单色仪后端。"""
        return self._isoplane

    # ------------------------------------------------------------------
    # 门控支持（PI-MAX4）
    # ------------------------------------------------------------------
    def set_gating_mode(self, mode: int) -> None:
        """设置门控模式。"""
        self._ensure_connected()
        self._binding.set_parameter_int(
            self._handle, pic.PicamParameter.GatingMode, int(mode)
        )
        self._commit()

    def set_gate_width(self, nanoseconds: float) -> None:
        """设置门宽（ns）。"""
        self._ensure_connected()
        self._binding.set_parameter_float(
            self._handle, pic.PicamParameter.GateWidth, float(nanoseconds)
        )
        self._commit()

    def set_gate_delay(self, nanoseconds: float) -> None:
        """设置门延迟（ns）。"""
        self._ensure_connected()
        self._binding.set_parameter_float(
            self._handle, pic.PicamParameter.GateDelay, float(nanoseconds)
        )
        self._commit()

    def set_intensifier_gain(self, gain: int) -> None:
        """设置 intensifier 增益。"""
        self._ensure_connected()
        self._binding.set_parameter_int(
            self._handle, pic.PicamParameter.IntensifierGain, int(gain)
        )
        self._commit()

    # ------------------------------------------------------------------
    # 设备信息
    # ------------------------------------------------------------------
    def get_camera_info(self) -> dict:
        """获取相机信息字典。"""
        if self._camera_id is None:
            return {}
        return {
            "model": self._camera_id.model,
            "interface": self._camera_id.computer_interface,
            "serial_number": self._camera_id.serial_number.decode("utf-8", errors="ignore"),
            "sensor_name": self._camera_id.sensor_name.decode("utf-8", errors="ignore"),
            "demo": self.demo,
        }
