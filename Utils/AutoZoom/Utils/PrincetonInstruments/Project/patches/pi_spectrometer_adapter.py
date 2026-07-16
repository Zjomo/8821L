"""PI 光谱仪直接控制适配器。

该适配器提供与现有 `LabVIEWTCPServer` 兼容的接口，
但底层通过 PICam SDK 直接控制 PI 光谱仪，无需启动 LabVIEW TCP Server。

用法（替换现有工作流中的 LabVIEWTCPServer）：

    from pi_spectrometer.patches.pi_spectrometer_adapter import PISpectrometerAdapter

    server = PISpectrometerAdapter(
        backend_type="picam",      # 或 "demo"
        output_dir="labview_csv_output",
    )
    server.start_server_async()
    ready = server.wait_for_ready()   # 直接返回 ready
    result = server.request_measure(index=1)
    # result 结构与 LabVIEWTCPServer 返回一致
    server.close()
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from pi_spectrometer.core.base import SpectrometerBackend
from pi_spectrometer.core.exceptions import SpectrometerError
from pi_spectrometer.core.types import ROI
from pi_spectrometer.picam.camera import PICamCamera
from pi_spectrometer.picam.demo import DemoCamera
from pi_spectrometer.picam.isoplane import IsoPlaneBackend


LogCallback = Any


class PISpectrometerAdapter:
    """兼容 LabVIEWTCPServer 接口的 PI 直接控制适配器。"""

    def __init__(
        self,
        backend_type: str = "demo",
        output_dir: str | Path = "labview_csv_output",
        on_log: Optional[LogCallback] = None,
        dll_path: Optional[str] = None,
        camera_index: int = 0,
        exposure: float = 0.1,
        sensor_temperature: float = -25.0,
        roi: Optional[ROI] = None,
        # IsoPlane 单色仪参数
        isoplane_dll_path: Optional[str] = None,
        isoplane_device_index: int = 0,
        center_wavelength_nm: Optional[float] = None,
        grating_index: Optional[int] = None,
        entrance_slit_um: Optional[int] = None,
        exit_slit_um: Optional[int] = None,
    ):
        """
        参数：
            backend_type:
                "picam"  - 真实 PICam 相机
                "demo"   - PICam 软件模拟相机（需要 PICam SDK）
            output_dir:
                CSV 保存目录，与现有工作流兼容
            on_log:
                日志回调，签名为 on_log(msg: str) -> None
            dll_path:
                可选，Picam.dll 路径
            camera_index:
                真实相机索引
            exposure:
                默认曝光时间（秒）
            sensor_temperature:
                默认目标温度（摄氏度）
            roi:
                默认 ROI
            isoplane_dll_path:
                ARC_SpectraPro.dll 路径，用于控制 IsoPlane 单色仪
            isoplane_device_index:
                单色仪设备索引
            center_wavelength_nm:
                初始中心波长（nm）
            grating_index:
                初始光栅索引（0-3）
            entrance_slit_um:
                入口狭缝宽度（微米）
            exit_slit_um:
                出口狭缝宽度（微米）
        """
        self.backend_type = backend_type
        self.output_dir = Path(output_dir)
        self.on_log = on_log
        self.dll_path = dll_path
        self.camera_index = camera_index
        self.exposure = exposure
        self.sensor_temperature = sensor_temperature
        self.roi = roi or ROI(x=0, width=1024, y=0, height=256)

        # IsoPlane 参数
        self.isoplane_dll_path = isoplane_dll_path
        self.isoplane_device_index = isoplane_device_index
        self.center_wavelength_nm = center_wavelength_nm
        self.grating_index = grating_index
        self.entrance_slit_um = entrance_slit_um
        self.exit_slit_um = exit_slit_um

        self.backend: Optional[SpectrometerBackend] = None
        self._isoplane: Optional[IsoPlaneBackend] = None

        # 与 LabVIEWTCPServer 保持兼容的状态字段
        self.is_server_running = False
        self.is_connected = False
        self.is_ready = False
        self.measure_count = 0
        self.host = "picam_direct"
        self.port = 0

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def log(self, msg: str) -> None:
        if self.on_log is not None:
            self.on_log(msg)
        else:
            print(msg)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start_server_async(self) -> None:
        """非阻塞启动（与 LabVIEWTCPServer 接口一致）。"""
        with self._lock:
            if self.is_server_running:
                self.log("[PIAdapter] 已经运行")
                return
            thread = threading.Thread(target=self.start_server_blocking, daemon=True)
            thread.start()

    def start_server_blocking(self) -> None:
        """阻塞方式启动并连接相机（可选连接 IsoPlane 单色仪）。"""
        try:
            with self._lock:
                self.is_server_running = True
                self.is_connected = False
                self.is_ready = False

            self.log(f"[PIAdapter] 启动后端: {self.backend_type}")

            # 1. 尝试连接 IsoPlane 单色仪（如果配置了）
            if self.isoplane_dll_path or self.center_wavelength_nm:
                try:
                    self._isoplane = IsoPlaneBackend(
                        dll_path=self.isoplane_dll_path,
                        device_index=self.isoplane_device_index,
                    )
                    self._isoplane.connect()
                    self.log("[PIAdapter] IsoPlane 单色仪已连接")

                    # 配置波长和光栅
                    if self.center_wavelength_nm is not None:
                        self._isoplane.set_wavelength(self.center_wavelength_nm)
                    if self.grating_index is not None:
                        self._isoplane.set_grating(self.grating_index)
                    if self.entrance_slit_um is not None:
                        self._isoplane.set_entrance_slit(self.entrance_slit_um)
                    if self.exit_slit_um is not None:
                        self._isoplane.set_exit_slit(self.exit_slit_um)

                except Exception as e:
                    self.log(f"[PIAdapter] IsoPlane 连接失败（继续使用相机）: {e}")
                    self._isoplane = None

            # 2. 创建相机后端
            if self.backend_type == "picam":
                self.backend = PICamCamera(
                    dll_path=self.dll_path,
                    camera_index=self.camera_index,
                    isoplane=self._isoplane,
                )
            elif self.backend_type in ("demo", "picam_demo"):
                self.backend = DemoCamera(
                    dll_path=self.dll_path,
                    isoplane=self._isoplane,
                    center_wavelength_nm=self.center_wavelength_nm or 550.0,
                    grating_index=self.grating_index or 1,
                )
            else:
                raise ValueError(f"不支持的 backend_type: {self.backend_type}，请使用 'picam' 或 'demo'")

            # 3. 连接相机
            self.backend.connect()
            self.backend.set_exposure(self.exposure)
            if self.backend.supports_temperature():
                self.backend.set_sensor_temperature(self.sensor_temperature)
            if self.backend.supports_roi():
                self.backend.set_roi(self.roi)

            self.is_connected = True
            self.log("[PIAdapter] 相机已连接并准备就绪")

        except Exception as e:
            self.log(f"[PIAdapter] 启动失败: {e}")
            self.close()

    def wait_for_ready(self) -> Dict[str, Any]:
        """等待准备就绪（直接返回 ready，无需等待 LabVIEW）。"""
        try:
            if not self.is_connected or self.backend is None:
                raise RuntimeError("相机未连接")

            self.is_ready = True
            self.log("[PIAdapter] 已就绪，可以开始采集")
            return {"ok": True, "reason": "ready"}
        except Exception as e:
            self.is_ready = False
            self.log(f"[PIAdapter] 等待 READY 失败: {e}")
            return {"ok": False, "reason": str(e)}

    def request_measure(
        self,
        command: str = "MEASURE",
        index: Optional[int] = None,
        save_csv: bool = True,
    ) -> Dict[str, Any]:
        """执行一次采集并返回与 LabVIEWTCPServer 兼容的字典。"""
        try:
            if self.backend is None:
                raise RuntimeError("相机未连接")

            if index is None:
                self.measure_count += 1
                index = self.measure_count
            else:
                self.measure_count = max(self.measure_count, int(index))

            self.log(f"[PIAdapter] 第 {index} 次采集开始...")
            result = self.backend.acquire(num_frames=1)
            result.index = index

            values = result.raw_y.tolist() if isinstance(result.raw_y, np.ndarray) else list(result.raw_y)
            fit_values = result.fit_y.tolist() if isinstance(result.fit_y, np.ndarray) else []
            wavelength = result.wavelength.tolist() if isinstance(result.wavelength, np.ndarray) else []

            csv_path = None
            if save_csv:
                csv_path = self._save_csv(
                    index=index,
                    values=values,
                    fit_values=fit_values,
                    wavelength=wavelength,
                )

            self.log(f"[PIAdapter] 第 {index} 次采集完成，点数: {len(values)}")

            return {
                "ok": True,
                "reason": "ok",
                "index": index,
                "csv_path": str(csv_path) if csv_path else None,
                "values": values,
                "fit_values": fit_values,
                "wavelength": wavelength,
                "num_points": len(values),
                "raw_data_text": "",
                # 现有工作流 _parse_labview_result 可能需要的字段
                "raw_y": values,
                "fit_y": fit_values,
                "raw_original_peak": result.raw_original_peak,
                "raw_filtered_peak": result.raw_filtered_peak,
                "raw_median_peak": result.raw_median_peak,
                "raw_peak": result.raw_peak,
                "fit_peak": result.fit_peak,
            }

        except Exception as e:
            self.log(f"[PIAdapter] request_measure 失败: {e}")
            return {
                "ok": False,
                "reason": str(e),
                "index": index if index is not None else 0,
                "csv_path": None,
                "values": [],
                "fit_values": [],
                "wavelength": [],
                "num_points": 0,
                "raw_data_text": "",
                "raw_y": [],
                "fit_y": [],
            }

    def _save_csv(
        self,
        index: int,
        values: list,
        fit_values: list,
        wavelength: list,
    ) -> Optional[Path]:
        """保存 CSV，与现有输出目录兼容。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"pi_spectrum_{index:04d}.csv"
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "wavelength_nm", "intensity", "fit_intensity"])
                for i, intensity in enumerate(values):
                    writer.writerow(
                        [
                            i,
                            wavelength[i] if i < len(wavelength) else i,
                            intensity,
                            fit_values[i] if i < len(fit_values) else "",
                        ]
                    )
            return path
        except Exception as e:
            self.log(f"[PIAdapter] CSV 保存失败: {e}")
            return None

    def close(self) -> None:
        """关闭相机和单色仪。"""
        with self._lock:
            # 关闭相机
            if self.backend is not None:
                try:
                    self.backend.disconnect()
                except Exception as e:
                    self.log(f"[PIAdapter] 断开相机时出错: {e}")
                self.backend = None

            # 关闭 IsoPlane 单色仪
            if self._isoplane is not None:
                try:
                    self._isoplane.disconnect()
                    self.log("[PIAdapter] IsoPlane 单色仪已断开")
                except Exception as e:
                    self.log(f"[PIAdapter] 断开单色仪时出错: {e}")
                self._isoplane = None

            self.is_connected = False
            self.is_ready = False
            self.is_server_running = False

        self.log("[PIAdapter] 已关闭")

    def _ensure_connected(self) -> None:
        if not self.is_connected or self.backend is None:
            raise RuntimeError("PI 相机尚未连接")
