"""IsoPlane 单色仪后端。

控制 Princeton Instruments IsoPlane SCT-320 光谱仪，通过 ARC_SpectraPro.dll。
提供波长控制、光栅切换、狭缝调节等功能。

与 PICam 相机配合使用，可构成完整的光谱测量系统：
    - PICam: 控制 CCD/CMOS 探测器，采集光谱图像
    - IsoPlane: 控制单色仪，设置中心波长、光栅、狭缝

波长轴计算：
    基于 IsoPlane 的焦距和光栅参数，计算像素到波长的映射。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from pi_spectrometer.picam.arc_binding import ARCSpectraBinding, ARCSpectraError


class IsoPlaneBackend:
    """IsoPlane 单色仪后端。

    用于：
        1. 设置/读取中心波长
        2. 切换光栅
        3. 调节狭缝宽度
        4. 计算波长轴

    注意：IsoPlane 只控制光谱仪，不直接采集数据。
    实际光谱采集由 PICam 相机完成。
    """

    # IsoPlane SCT-320 焦距（mm）
    FOCAL_LENGTH_MM = 320.0

    # 常见光栅配置（lines/mm）
    GRATINGS = {
        0: {"density": 150, "blaze_nm": 500, "range_nm": (200, 2400)},
        1: {"density": 300, "blaze_nm": 500, "range_nm": (200, 1200)},
        2: {"density": 600, "blaze_nm": 500, "range_nm": (200, 600)},
        3: {"density": 1200, "blaze_nm": 500, "range_nm": (200, 300)},
    }

    def __init__(
        self,
        dll_path: Optional[str] = None,
        device_index: int = 0,
    ):
        """
        参数：
            dll_path: ARC_SpectraPro.dll 路径，None 则自动搜索
            device_index: 单色仪索引（多台时使用）
        """
        self.dll_path = dll_path
        self.device_index = device_index

        self._binding: Optional[ARCSpectraBinding] = None
        self._handle: Optional[int] = None
        self._connected = False

        # 当前状态
        self._center_wavelength_nm: float = 550.0
        self._current_grating: int = 1
        self._entrance_slit_um: int = 50
        self._exit_slit_um: int = 50

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """连接单色仪。"""
        if self._connected:
            return True

        try:
            self._binding = ARCSpectraBinding(self.dll_path)
            count = self._binding.search_for_mono()

            if count == 0:
                raise ARCSpectraError("未找到连接的单色仪")

            self._handle = self._binding.open_mono(self.device_index)
            self._connected = True

            # 读取当前状态
            self._sync_state()

            return True

        except Exception as e:
            self._connected = False
            raise ARCSpectraError(f"连接单色仪失败: {e}")

    def disconnect(self) -> None:
        """断开单色仪连接。"""
        if self._binding and self._handle is not None:
            try:
                self._binding.close_mono(self._handle)
            except Exception:
                pass
        self._handle = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _sync_state(self) -> None:
        """从设备同步当前状态。"""
        if self._handle is None:
            return
        try:
            self._center_wavelength_nm = self._binding.get_wavelength_nm(self._handle)
            self._current_grating = self._binding.get_grating(self._handle)
            self._entrance_slit_um = self._binding.get_slit_width(self._handle, 0)
            self._exit_slit_um = self._binding.get_slit_width(self._handle, 1)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 波长控制
    # ------------------------------------------------------------------
    def set_wavelength(self, wavelength_nm: float) -> None:
        """设置中心波长（nm）。"""
        if self._handle is None:
            raise ARCSpectraError("单色仪未连接")
        self._binding.set_wavelength_nm(self._handle, wavelength_nm)
        self._center_wavelength_nm = wavelength_nm

    def get_wavelength(self) -> float:
        """获取当前中心波长（nm）。"""
        if self._handle is None:
            return self._center_wavelength_nm
        self._center_wavelength_nm = self._binding.get_wavelength_nm(self._handle)
        return self._center_wavelength_nm

    def get_wavelength_range(self) -> Tuple[float, float]:
        """获取可用波长范围（min_nm, max_nm）。"""
        if self._handle is None:
            return 0.0, 1400.0
        return self._binding.get_wavelength_range(self._handle)

    # ------------------------------------------------------------------
    # 光栅控制
    # ------------------------------------------------------------------
    def set_grating(self, grating_index: int) -> None:
        """设置光栅索引（0-3）。"""
        if self._handle is None:
            raise ARCSpectraError("单色仪未连接")
        self._binding.set_grating(self._handle, grating_index)
        self._current_grating = grating_index

    def get_grating(self) -> int:
        """获取当前光栅索引。"""
        if self._handle is None:
            return self._current_grating
        self._current_grating = self._binding.get_grating(self._handle)
        return self._current_grating

    def get_grating_info(self, grating_index: int = None) -> dict:
        """获取光栅信息。"""
        if grating_index is None:
            grating_index = self._current_grating
        return self.GRATINGS.get(grating_index, {"density": 0, "blaze_nm": 0, "range_nm": (0, 0)})

    # ------------------------------------------------------------------
    # 狭缝控制
    # ------------------------------------------------------------------
    def set_entrance_slit(self, width_um: int) -> None:
        """设置入口狭缝宽度（微米）。"""
        if self._handle is None:
            raise ARCSpectraError("单色仪未连接")
        self._binding.set_slit_width(self._handle, width_um, 0)
        self._entrance_slit_um = width_um

    def set_exit_slit(self, width_um: int) -> None:
        """设置出口狭缝宽度（微米）。"""
        if self._handle is None:
            raise ARCSpectraError("单色仪未连接")
        self._binding.set_slit_width(self._handle, width_um, 1)
        self._exit_slit_um = width_um

    def get_entrance_slit(self) -> int:
        """获取入口狭缝宽度（微米）。"""
        if self._handle is None:
            return self._entrance_slit_um
        self._entrance_slit_um = self._binding.get_slit_width(self._handle, 0)
        return self._entrance_slit_um

    def get_exit_slit(self) -> int:
        """获取出口狭缝宽度（微米）。"""
        if self._handle is None:
            return self._exit_slit_um
        self._exit_slit_um = self._binding.get_slit_width(self._handle, 1)
        return self._exit_slit_um

    # ------------------------------------------------------------------
    # 波长轴计算
    # ------------------------------------------------------------------
    def calculate_wavelength_axis(
        self,
        num_pixels: int,
        center_wavelength_nm: float = None,
        grating_index: int = None,
        pixel_width_um: float = 13.0,
    ) -> np.ndarray:
        """计算波长轴。

        基于 IsoPlane 的光学参数，计算每个像素对应的波长。

        参数：
            num_pixels: 探测器像素数（沿光谱方向）
            center_wavelength_nm: 中心波长（nm），None 则使用当前值
            grating_index: 光栅索引，None 则使用当前值
            pixel_width_um: 像素宽度（微米）

        返回：
            numpy 数组，每个像素对应的波长（nm）
        """
        if center_wavelength_nm is None:
            center_wavelength_nm = self._center_wavelength_nm
        if grating_index is None:
            grating_index = self._current_grating

        grating_info = self.GRATINGS.get(grating_index, {"density": 300})
        groove_density = grating_info["density"]  # lines/mm

        # IsoPlane SCT-320 光学参数
        focal_length_mm = self.FOCAL_LENGTH_MM

        # 计算色散（nm/mm）
        # dλ/dx = (1 / (groove_density * focal_length)) * 1e6  nm/mm
        # 对于固定光栅角度（接近 Littrow 条件）
        # 简化公式：色散 ≈ wavelength / (groove_density * focal_length * cos(theta))
        # 这里使用近似：色散 ≈ 1e6 / (groove_density * focal_length) nm/mm

        dispersion_nm_per_mm = 1e6 / (groove_density * focal_length_mm)

        # 像素到 mm
        pixel_size_mm = pixel_width_um / 1000.0

        # 波长范围
        half_width_nm = (num_pixels / 2) * pixel_size_mm * dispersion_nm_per_mm

        # 生成波长轴
        wavelength_axis = np.linspace(
            center_wavelength_nm - half_width_nm,
            center_wavelength_nm + half_width_nm,
            num_pixels
        )

        return wavelength_axis

    # ------------------------------------------------------------------
    # 设备信息
    # ------------------------------------------------------------------
    def get_info(self) -> dict:
        """获取设备信息。"""
        if self._handle is None:
            return {
                "model": "IsoPlane SCT-320",
                "serial": "Unknown",
                "focal_length_mm": self.FOCAL_LENGTH_MM,
                "connected": False,
            }

        try:
            return {
                "model": self._binding.get_model(self._handle),
                "serial": self._binding.get_serial(self._handle),
                "focal_length_mm": self._binding.get_focal_length(self._handle),
                "wavelength_nm": self._center_wavelength_nm,
                "grating": self._current_grating,
                "entrance_slit_um": self._entrance_slit_um,
                "exit_slit_um": self._exit_slit_um,
                "connected": True,
            }
        except Exception:
            return {
                "model": "IsoPlane SCT-320",
                "serial": "Unknown",
                "focal_length_mm": self.FOCAL_LENGTH_MM,
                "connected": self._connected,
            }