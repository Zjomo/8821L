"""PICam SDK ctypes 绑定层。

负责加载 Picam.dll、封装 C 函数调用、错误码转换。
支持从环境变量 PicamRoot 自动定位 DLL。
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Any

import numpy as np

from pi_spectrometer.core.exceptions import PICamError
from pi_spectrometer.picam import constants as pic


# ---------------------------------------------------------------------------
# 结构体定义
# ---------------------------------------------------------------------------
class PicamCameraID(ctypes.Structure):
    """PICam 相机 ID 结构体。"""

    _fields_ = [
        ("model", ctypes.c_int),
        ("computer_interface", ctypes.c_int),
        ("serial_number", ctypes.c_char * 64),
        ("sensor_name", ctypes.c_char * 64),
    ]


class PicamRoi(ctypes.Structure):
    """单个 ROI。"""

    _fields_ = [
        ("x", ctypes.c_int),
        ("width", ctypes.c_int),
        ("x_binning", ctypes.c_int),
        ("y", ctypes.c_int),
        ("height", ctypes.c_int),
        ("y_binning", ctypes.c_int),
    ]


class PicamRois(ctypes.Structure):
    """ROI 数组包装。"""

    _fields_ = [
        ("roi_array", ctypes.POINTER(PicamRoi)),
        ("roi_count", ctypes.c_int),
    ]


PicamHandle = ctypes.c_void_p


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------
def check_error(error_code: int, context: str = "") -> None:
    """检查 PICam 错误码并在失败时抛出 PICamError。"""
    if error_code == pic.PicamError.None_:
        return
    msg = pic.ERROR_MESSAGES.get(error_code, f"Unknown PICam error {error_code}")
    if context:
        msg = f"{context}: {msg}"
    raise PICamError(msg, error_code)


# ---------------------------------------------------------------------------
# DLL 加载
# ---------------------------------------------------------------------------
def _find_picam_dll() -> Optional[Path]:
    """自动查找 Picam.dll 路径。"""
    # 1. 环境变量 PicamRoot
    picam_root = os.environ.get("PicamRoot")
    if picam_root:
        candidate = Path(picam_root) / "Runtime" / "Picam.dll"
        if candidate.exists():
            return candidate
        candidate = Path(picam_root) / "Picam.dll"
        if candidate.exists():
            return candidate

    # 2. 常见安装路径
    common_paths = [
        Path(r"C:\Program Files\Princeton Instruments\Picam\Runtime\Picam.dll"),
        Path(r"C:\Program Files (x86)\Princeton Instruments\Picam\Runtime\Picam.dll"),
        Path(r"C:\Program Files\Common Files\Princeton Instruments\Picam\Runtime\Picam.dll"),
    ]
    for candidate in common_paths:
        if candidate.exists():
            return candidate

    return None


class PICamBinding:
    """PICam SDK 绑定对象（单例风格，每个进程一个库实例）。"""

    _instance: Optional["PICamBinding"] = None

    def __new__(cls, dll_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, dll_path: Optional[str] = None):
        if self._initialized:
            return
        self.dll_path = dll_path
        self._lib: Optional[ctypes.CDLL] = None
        self._initialize()
        self._initialized = True

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        if self.dll_path:
            dll_file = Path(self.dll_path)
        else:
            dll_file = _find_picam_dll()

        if dll_file is None or not dll_file.exists():
            raise PICamError(
                "无法找到 Picam.dll。请安装 PICam SDK 并设置 PicamRoot 环境变量，"
                "或在构造 PICamBinding 时传入 dll_path。\n"
                "提示：如果没有真实硬件，请在 UI 中选择 '无 SDK 模拟' 后端。"
            )

        self.dll_path = str(dll_file)
        try:
            self._lib = ctypes.CDLL(str(dll_file))
        except OSError as e:
            raise PICamError(
                f"加载 Picam.dll 失败: {e}\n"
                "提示：如果没有真实硬件，请在 UI 中选择 '无 SDK 模拟' 后端。"
            ) from e

        self._setup_function_signatures()
        err = self._lib.Picam_InitializeLibrary()
        check_error(err, "Picam_InitializeLibrary")

    def _setup_function_signatures(self) -> None:
        """设置 C 函数签名。"""
        lib = self._lib

        # Initialize / Uninitialize
        lib.Picam_InitializeLibrary.restype = ctypes.c_int
        lib.Picam_UninitializeLibrary.restype = ctypes.c_int

        # Camera discovery
        lib.Picam_GetAvailableCameraIDs.argtypes = [
            ctypes.POINTER(ctypes.POINTER(PicamCameraID)),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.Picam_GetAvailableCameraIDs.restype = ctypes.c_int

        # Picam_FreeCameraIDs 在某些版本中不存在，需要容错
        if hasattr(lib, 'Picam_FreeCameraIDs'):
            lib.Picam_FreeCameraIDs.argtypes = [ctypes.POINTER(PicamCameraID)]
            lib.Picam_FreeCameraIDs.restype = ctypes.c_int
        else:
            # 旧版本 SDK 可能没有此函数，设置为 None 表示跳过
            lib.Picam_FreeCameraIDs = None

        # Demo camera
        lib.Picam_ConnectDemoCamera.argtypes = [
            ctypes.c_int,  # model
            ctypes.c_char_p,  # serial_number
            ctypes.POINTER(PicamCameraID),
        ]
        lib.Picam_ConnectDemoCamera.restype = ctypes.c_int

        lib.Picam_DisconnectDemoCamera.argtypes = [ctypes.POINTER(PicamCameraID)]
        lib.Picam_DisconnectDemoCamera.restype = ctypes.c_int

        # Open / Close
        lib.Picam_OpenCamera.argtypes = [
            ctypes.POINTER(PicamCameraID),
            ctypes.POINTER(PicamHandle),
        ]
        lib.Picam_OpenCamera.restype = ctypes.c_int

        lib.Picam_CloseCamera.argtypes = [PicamHandle]
        lib.Picam_CloseCamera.restype = ctypes.c_int

        # Get / Set parameters
        lib.Picam_GetParameterIntegerValue.argtypes = [
            PicamHandle, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
        ]
        lib.Picam_GetParameterIntegerValue.restype = ctypes.c_int

        lib.Picam_GetParameterLargeIntegerValue.argtypes = [
            PicamHandle, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)
        ]
        lib.Picam_GetParameterLargeIntegerValue.restype = ctypes.c_int

        lib.Picam_GetParameterFloatingPointValue.argtypes = [
            PicamHandle, ctypes.c_int, ctypes.POINTER(ctypes.c_double)
        ]
        lib.Picam_GetParameterFloatingPointValue.restype = ctypes.c_int

        lib.Picam_SetParameterIntegerValue.argtypes = [
            PicamHandle, ctypes.c_int, ctypes.c_int
        ]
        lib.Picam_SetParameterIntegerValue.restype = ctypes.c_int

        lib.Picam_SetParameterLargeIntegerValue.argtypes = [
            PicamHandle, ctypes.c_int, ctypes.c_longlong
        ]
        lib.Picam_SetParameterLargeIntegerValue.restype = ctypes.c_int

        lib.Picam_SetParameterFloatingPointValue.argtypes = [
            PicamHandle, ctypes.c_int, ctypes.c_double
        ]
        lib.Picam_SetParameterFloatingPointValue.restype = ctypes.c_int

        # ROI
        lib.Picam_GetParameterRoisValue.argtypes = [
            PicamHandle, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(PicamRois))
        ]
        lib.Picam_GetParameterRoisValue.restype = ctypes.c_int

        lib.Picam_SetParameterRoisValue.argtypes = [
            PicamHandle, ctypes.c_int, ctypes.POINTER(PicamRois)
        ]
        lib.Picam_SetParameterRoisValue.restype = ctypes.c_int

        lib.Picam_DestroyRois.argtypes = [ctypes.POINTER(PicamRois)]
        lib.Picam_DestroyRois.restype = ctypes.c_int

        # Commit
        lib.Picam_CommitParameters.argtypes = [
            PicamHandle, ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), ctypes.POINTER(ctypes.c_int)
        ]
        lib.Picam_CommitParameters.restype = ctypes.c_int

        lib.Picam_DestroyCommitParameters.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.Picam_DestroyCommitParameters.restype = ctypes.c_int

        # Acquire
        lib.Picam_Acquire.argtypes = [
            PicamHandle,
            ctypes.c_int,  # readout_count
            ctypes.c_int,  # readout_time_out
            ctypes.POINTER(ctypes.c_void_p),  # readout_array
            ctypes.POINTER(ctypes.c_int64),  # readout_count_returned
        ]
        lib.Picam_Acquire.restype = ctypes.c_int

        lib.Picam_DestroyReadouts.argtypes = [ctypes.c_void_p]
        lib.Picam_DestroyReadouts.restype = ctypes.c_int

    # ------------------------------------------------------------------
    # 库生命周期
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """关闭 PICam 库。"""
        if self._lib is not None:
            try:
                self._lib.Picam_UninitializeLibrary()
            except Exception:
                pass
            self._lib = None
            PICamBinding._instance = None

    def __del__(self):
        # 不自动 shutdown，避免句柄顺序问题；由显式调用处理
        pass

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @property
    def lib(self) -> ctypes.CDLL:
        if self._lib is None:
            raise PICamError("PICam 库未初始化")
        return self._lib

    # ------------------------------------------------------------------
    # 相机发现
    # ------------------------------------------------------------------
    def get_available_cameras(self) -> List[PicamCameraID]:
        """返回可用的真实相机列表。"""
        ids_ptr = ctypes.POINTER(PicamCameraID)()
        count = ctypes.c_int()
        err = self.lib.Picam_GetAvailableCameraIDs(ctypes.byref(ids_ptr), ctypes.byref(count))
        check_error(err, "Picam_GetAvailableCameraIDs")

        cameras = []
        for i in range(count.value):
            cameras.append(ids_ptr[i])

        # 某些旧版 SDK 没有 Picam_FreeCameraIDs，跳过释放
        if hasattr(self.lib, 'Picam_FreeCameraIDs') and self.lib.Picam_FreeCameraIDs is not None:
            try:
                self.lib.Picam_FreeCameraIDs(ids_ptr)
            except Exception:
                pass  # 忽略释放失败
        return cameras

    def connect_demo_camera(self, model: int, serial_number: str) -> PicamCameraID:
        """连接一个软件模拟相机。"""
        cam_id = PicamCameraID()
        err = self.lib.Picam_ConnectDemoCamera(
            ctypes.c_int(model),
            serial_number.encode("utf-8"),
            ctypes.byref(cam_id),
        )
        check_error(err, "Picam_ConnectDemoCamera")
        return cam_id

    def disconnect_demo_camera(self, cam_id: PicamCameraID) -> None:
        err = self.lib.Picam_DisconnectDemoCamera(ctypes.byref(cam_id))
        check_error(err, "Picam_DisconnectDemoCamera")

    # ------------------------------------------------------------------
    # 相机打开/关闭
    # ------------------------------------------------------------------
    def open_camera(self, cam_id: PicamCameraID) -> PicamHandle:
        handle = PicamHandle()
        err = self.lib.Picam_OpenCamera(ctypes.byref(cam_id), ctypes.byref(handle))
        check_error(err, "Picam_OpenCamera")
        return handle

    def close_camera(self, handle: PicamHandle) -> None:
        if handle:
            err = self.lib.Picam_CloseCamera(handle)
            check_error(err, "Picam_CloseCamera")

    # ------------------------------------------------------------------
    # 参数读写
    # ------------------------------------------------------------------
    def get_parameter_int(self, handle: PicamHandle, parameter: int) -> int:
        value = ctypes.c_int()
        err = self.lib.Picam_GetParameterIntegerValue(handle, ctypes.c_int(parameter), ctypes.byref(value))
        check_error(err, f"GetParameterIntegerValue(0x{parameter:08X})")
        return value.value

    def get_parameter_float(self, handle: PicamHandle, parameter: int) -> float:
        value = ctypes.c_double()
        err = self.lib.Picam_GetParameterFloatingPointValue(handle, ctypes.c_int(parameter), ctypes.byref(value))
        check_error(err, f"GetParameterFloatingPointValue(0x{parameter:08X})")
        return value.value

    def set_parameter_int(self, handle: PicamHandle, parameter: int, value: int) -> None:
        err = self.lib.Picam_SetParameterIntegerValue(handle, ctypes.c_int(parameter), ctypes.c_int(value))
        check_error(err, f"SetParameterIntegerValue(0x{parameter:08X}, {value})")

    def set_parameter_float(self, handle: PicamHandle, parameter: int, value: float) -> None:
        err = self.lib.Picam_SetParameterFloatingPointValue(handle, ctypes.c_int(parameter), ctypes.c_double(value))
        check_error(err, f"SetParameterFloatingPointValue(0x{parameter:08X}, {value})")

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------
    def set_roi(self, handle: PicamHandle, roi_list: List[Tuple[int, int, int, int, int, int]]) -> None:
        """设置 ROI 列表。每个 ROI 为 (x, width, xbin, y, height, ybin)。"""
        count = len(roi_list)
        rois = PicamRois()
        rois.roi_count = count
        rois.roi_array = (PicamRoi * count)()
        for i, (x, w, xb, y, h, yb) in enumerate(roi_list):
            rois.roi_array[i] = PicamRoi(x, w, xb, y, h, yb)

        err = self.lib.Picam_SetParameterRoisValue(handle, pic.PicamParameter.ActiveWidth, ctypes.byref(rois))
        check_error(err, "SetParameterRoisValue")

    # ------------------------------------------------------------------
    # 参数提交
    # ------------------------------------------------------------------
    def commit_parameters(self, handle: PicamHandle) -> None:
        failed_params = ctypes.POINTER(ctypes.c_int)()
        failed_count = ctypes.c_int()
        err = self.lib.Picam_CommitParameters(handle, ctypes.byref(failed_params), ctypes.byref(failed_count))

        if failed_count.value > 0:
            self.lib.Picam_DestroyCommitParameters(failed_params)
        check_error(err, "Picam_CommitParameters")

    # ------------------------------------------------------------------
    # 同步采集
    # ------------------------------------------------------------------
    def acquire(self, handle: PicamHandle, readout_count: int = 1, timeout_ms: int = -1) -> np.ndarray:
        """同步采集 readout_count 帧，返回 numpy 数组。"""
        readout_array = ctypes.c_void_p()
        readout_count_returned = ctypes.c_int64()

        err = self.lib.Picam_Acquire(
            handle,
            ctypes.c_int(readout_count),
            ctypes.c_int(timeout_ms),
            ctypes.byref(readout_array),
            ctypes.byref(readout_count_returned),
        )
        check_error(err, "Picam_Acquire")

        # 通过 ADC 参数推断数据类型：简化处理为 float64
        # 实际应根据 BitDepth 决定 uin16/int16/uint32 等
        width, height = self._get_frame_shape(handle)
        total_pixels = width * height * readout_count_returned.value

        # 默认按 uint16 读取，再转 float64
        buffer = (ctypes.c_uint16 * total_pixels).from_address(readout_array.value)
        data = np.ctypeslib.as_array(buffer).copy().astype(np.float64)
        data = data.reshape((readout_count_returned.value, height, width))

        self.lib.Picam_DestroyReadouts(readout_array)
        return data

    def _get_frame_shape(self, handle: PicamHandle) -> Tuple[int, int]:
        """获取当前帧的宽高。"""
        try:
            width = self.get_parameter_int(handle, pic.PicamParameter.ActiveWidth)
            height = self.get_parameter_int(handle, pic.PicamParameter.ActiveHeight)
        except PICamError:
            width = height = 1
        return width, height


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------
def get_binding(dll_path: Optional[str] = None) -> PICamBinding:
    """获取 PICamBinding 实例。"""
    return PICamBinding(dll_path)
