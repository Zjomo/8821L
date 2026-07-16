"""ARC SpectraPro SDK ctypes 绑定层。

控制 Princeton Instruments / Acton Research 单色仪（IsoPlane, SpectraPro 系列）。
通过 ARC_SpectraPro.dll 实现 FTDI USB 通信。

关键功能：
    - 波长设置/读取（nm）
    - 光栅切换
    - 狭缝宽度控制
    - 扫描速率设置
"""

from __future__ import annotations

import ctypes
import os
import struct
from pathlib import Path
from typing import Optional, List, Tuple, Any


class ARCSpectraError(Exception):
    """ARC SpectraPro SDK 错误。"""
    pass


def _get_python_bits() -> int:
    """返回当前 Python 解释器的位数（32 或 64）。"""
    return struct.calcsize("P") * 8


def _get_dll_bits(dll_path: str) -> int:
    """读取 PE 头，返回 DLL 的位数（32 或 64）。"""
    with open(dll_path, "rb") as f:
        dos_header = f.read(64)
    if len(dos_header) < 64 or dos_header[:2] != b"MZ":
        return 32  # 默认假设为 32 位
    pe_offset = struct.unpack_from("<I", dos_header, 0x3C)[0]
    with open(dll_path, "rb") as f:
        f.seek(pe_offset)
        pe_header = f.read(6)
    if len(pe_header) < 6 or pe_header[:4] != b"PE\x00\x00":
        return 32
    machine = struct.unpack_from("<H", pe_header, 4)[0]
    return 64 if machine == 0x8664 else 32


def _find_arc_dll() -> Optional[Path]:
    """查找 ARC_SpectraPro.dll 路径。"""
    # 1. 环境变量
    arc_root = os.environ.get("ARC_SpectraPro_Root")
    if arc_root:
        candidate = Path(arc_root) / "ARC_SpectraPro.dll"
        if candidate.exists():
            return candidate

    # 2. 项目内置目录
    project_dir = Path(__file__).resolve().parent
    while project_dir.parent != project_dir:
        candidate = project_dir / "ISOPLANEControl" / "ARC_SpectraPro.dll"
        if candidate.exists():
            return candidate
        candidate = project_dir / "PrincetonInstruments" / "ISOPLANEControl" / "ARC_SpectraPro.dll"
        if candidate.exists():
            return candidate
        project_dir = project_dir.parent

    # 3. 常见安装路径
    common_paths = [
        Path(r"C:\Program Files\Princeton Instruments\ISOPLANEControl\ARC_SpectraPro.dll"),
        Path(r"C:\Program Files (x86)\Princeton Instruments\ISOPLANEControl\ARC_SpectraPro.dll"),
        Path(r"C:\Acton\ARC_SpectraPro.dll"),
    ]
    for candidate in common_paths:
        if candidate.exists():
            return candidate

    return None


class ARCSpectraBinding:
    """ARC SpectraPro SDK 绑定对象。"""

    _instance: Optional["ARCSpectraBinding"] = None

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
        self._mono_handles: List[int] = []
        self._initialize()
        self._initialized = True

    def _initialize(self) -> None:
        if self.dll_path:
            dll_file = Path(self.dll_path)
        else:
            dll_file = _find_arc_dll()

        if dll_file is None or not dll_file.exists():
            raise ARCSpectraError(
                "无法找到 ARC_SpectraPro.dll。请安装 ISOPLANEControl 并设置 ARC_SpectraPro_Root 环境变量。"
            )

        self.dll_path = str(dll_file)

        # 检查 Python 与 DLL 的位数匹配
        dll_bits = _get_dll_bits(self.dll_path)
        py_bits = _get_python_bits()
        if dll_bits != py_bits:
            raise ARCSpectraError(
                f"Python/DLL 位数不匹配：当前 Python 是 {py_bits} 位，"
                f"ARC_SpectraPro.dll 是 {dll_bits} 位。\n"
                f"请使用 {dll_bits} 位 Python 解释器运行本项目。\n"
                f"建议：安装 Python 3.9 {dll_bits}-bit，并重新创建虚拟环境。\n"
                f"下载地址：https://www.python.org/downloads/release/python-3913/"
            )

        try:
            # 将 DLL 目录添加到 PATH，确保依赖 DLL 可找到
            dll_dir = str(Path(dll_file).parent)
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            self._lib = ctypes.CDLL(str(dll_file))
        except OSError as e:
            err_msg = str(e)
            if "不是有效的 Win32 应用程序" in err_msg or "%1 is not a valid Win32" in err_msg:
                err_msg += (
                    "\n提示：位数不匹配。当前 Python 是 {} 位，DLL 是 {} 位。"
                    "请使用 {} 位 Python。".format(py_bits, dll_bits, dll_bits)
                )
            raise ARCSpectraError(f"加载 ARC_SpectraPro.dll 失败: {err_msg}") from e

        self._setup_function_signatures()

    def _setup_function_signatures(self) -> None:
        """设置 C 函数签名。"""
        lib = self._lib

        # 版本信息
        lib.ARC_Ver.argtypes = [ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long)]
        lib.ARC_Ver.restype = ctypes.c_uint16

        # 设备搜索
        lib.ARC_Search_For_Mono.argtypes = [ctypes.POINTER(ctypes.c_long)]
        lib.ARC_Search_For_Mono.restype = ctypes.c_uint16

        # 设备打开/关闭
        lib.ARC_Open_Mono.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_long)]
        lib.ARC_Open_Mono.restype = ctypes.c_uint16

        lib.ARC_Open_Mono_Port.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_long)]
        lib.ARC_Open_Mono_Port.restype = ctypes.c_uint16

        lib.ARC_Close_Mono.argtypes = [ctypes.c_long]
        lib.ARC_Close_Mono.restype = ctypes.c_uint16

        # 设备验证
        lib.ARC_Valid_Mono_Enum.argtypes = [ctypes.c_long]
        lib.ARC_Valid_Mono_Enum.restype = ctypes.c_uint16

        # 波长（nm）
        lib.ARC_get_Mono_Wavelength_nm.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_double)]
        lib.ARC_get_Mono_Wavelength_nm.restype = ctypes.c_uint16

        lib.ARC_set_Mono_Wavelength_nm.argtypes = [ctypes.c_long, ctypes.c_double]
        lib.ARC_set_Mono_Wavelength_nm.restype = ctypes.c_uint16

        # 波长范围
        lib.ARC_get_Mono_Wavelength_Min_nm.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_double)]
        lib.ARC_get_Mono_Wavelength_Min_nm.restype = ctypes.c_uint16

        lib.ARC_get_Mono_Wavelength_Cutoff_nm.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_double)]
        lib.ARC_get_Mono_Wavelength_Cutoff_nm.restype = ctypes.c_uint16

        # 光栅
        lib.ARC_get_Mono_Grating.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_long)]
        lib.ARC_get_Mono_Grating.restype = ctypes.c_uint16

        lib.ARC_set_Mono_Grating.argtypes = [ctypes.c_long, ctypes.c_long]
        lib.ARC_set_Mono_Grating.restype = ctypes.c_uint16

        lib.ARC_get_Mono_Turret_Gratings.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_long)]
        lib.ARC_get_Mono_Turret_Gratings.restype = ctypes.c_uint16

        lib.ARC_get_Mono_Grating_Density.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.POINTER(ctypes.c_long)]
        lib.ARC_get_Mono_Grating_Density.restype = ctypes.c_uint16

        # 狭缝
        lib.ARC_get_Mono_Slit_Width.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.POINTER(ctypes.c_long)]
        lib.ARC_get_Mono_Slit_Width.restype = ctypes.c_uint16

        lib.ARC_set_Mono_Slit_Width.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.ARC_set_Mono_Slit_Width.restype = ctypes.c_uint16

        # 扫描
        lib.ARC_get_Mono_Scan_Rate_nm_min.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_double)]
        lib.ARC_get_Mono_Scan_Rate_nm_min.restype = ctypes.c_uint16

        lib.ARC_set_Mono_Scan_Rate_nm_min.argtypes = [ctypes.c_long, ctypes.c_double]
        lib.ARC_set_Mono_Scan_Rate_nm_min.restype = ctypes.c_uint16

        lib.ARC_Mono_Start_Scan_To_nm.argtypes = [ctypes.c_long, ctypes.c_double]
        lib.ARC_Mono_Start_Scan_To_nm.restype = ctypes.c_uint16

        lib.ARC_Mono_Scan_Done.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_long), ctypes.c_double]
        lib.ARC_Mono_Scan_Done.restype = ctypes.c_uint16

        # 设备信息
        lib.ARC_get_Mono_Model_CString.argtypes = [ctypes.c_long, ctypes.c_char_p]
        lib.ARC_get_Mono_Model_CString.restype = ctypes.c_uint16

        lib.ARC_get_Mono_Serial_CString.argtypes = [ctypes.c_long, ctypes.c_char_p]
        lib.ARC_get_Mono_Serial_CString.restype = ctypes.c_uint16

        lib.ARC_get_Mono_Focallength.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_double)]
        lib.ARC_get_Mono_Focallength.restype = ctypes.c_uint16

        lib.ARC_get_Mono_Precision.argtypes = [ctypes.c_long]
        lib.ARC_get_Mono_Precision.restype = ctypes.c_uint16

    # ------------------------------------------------------------------
    # 设备发现
    # ------------------------------------------------------------------
    def search_for_mono(self) -> int:
        """搜索连接的单色仪，返回找到的数量。"""
        count = ctypes.c_long()
        ret = self._lib.ARC_Search_For_Mono(ctypes.byref(count))
        if ret != 0:
            raise ARCSpectraError(f"搜索单色仪失败: 错误码 {ret}")
        return count.value

    def open_mono(self, device_index: int = 0) -> int:
        """打开指定索引的单色仪，返回句柄。"""
        handle = ctypes.c_long()
        ret = self._lib.ARC_Open_Mono(ctypes.c_long(device_index), ctypes.byref(handle))
        if ret != 0:
            raise ARCSpectraError(f"打开单色仪 {device_index} 失败: 错误码 {ret}")
        self._mono_handles.append(handle.value)
        return handle.value

    def close_mono(self, handle: int) -> None:
        """关闭单色仪。"""
        ret = self._lib.ARC_Close_Mono(ctypes.c_long(handle))
        if handle in self._mono_handles:
            self._mono_handles.remove(handle)
        if ret != 0:
            raise ARCSpectraError(f"关闭单色仪失败: 错误码 {ret}")

    # ------------------------------------------------------------------
    # 波长控制
    # ------------------------------------------------------------------
    def get_wavelength_nm(self, handle: int) -> float:
        """获取当前中心波长（nm）。"""
        wavelength = ctypes.c_double()
        ret = self._lib.ARC_get_Mono_Wavelength_nm(ctypes.c_long(handle), ctypes.byref(wavelength))
        if ret != 0:
            raise ARCSpectraError(f"读取波长失败: 错误码 {ret}")
        return wavelength.value

    def set_wavelength_nm(self, handle: int, wavelength: float) -> None:
        """设置中心波长（nm）。"""
        ret = self._lib.ARC_set_Mono_Wavelength_nm(ctypes.c_long(handle), ctypes.c_double(wavelength))
        if ret != 0:
            raise ARCSpectraError(f"设置波长 {wavelength} nm 失败: 错误码 {ret}")

    def get_wavelength_range(self, handle: int) -> Tuple[float, float]:
        """获取波长范围（min_nm, max_nm）。"""
        min_nm = ctypes.c_double()
        max_nm = ctypes.c_double()
        self._lib.ARC_get_Mono_Wavelength_Min_nm(ctypes.c_long(handle), ctypes.byref(min_nm))
        self._lib.ARC_get_Mono_Wavelength_Cutoff_nm(ctypes.c_long(handle), ctypes.byref(max_nm))
        return min_nm.value, max_nm.value

    # ------------------------------------------------------------------
    # 光栅控制
    # ------------------------------------------------------------------
    def get_grating(self, handle: int) -> int:
        """获取当前光栅索引。"""
        grating = ctypes.c_long()
        ret = self._lib.ARC_get_Mono_Grating(ctypes.c_long(handle), ctypes.byref(grating))
        if ret != 0:
            raise ARCSpectraError(f"读取光栅失败: 错误码 {ret}")
        return grating.value

    def set_grating(self, handle: int, grating_index: int) -> None:
        """设置光栅索引。"""
        ret = self._lib.ARC_set_Mono_Grating(ctypes.c_long(handle), ctypes.c_long(grating_index))
        if ret != 0:
            raise ARCSpectraError(f"设置光栅 {grating_index} 失败: 错误码 {ret}")

    def get_grating_density(self, handle: int, grating_index: int) -> int:
        """获取光栅密度（lines/mm）。"""
        density = ctypes.c_long()
        ret = self._lib.ARC_get_Mono_Grating_Density(
            ctypes.c_long(handle), ctypes.c_long(grating_index), ctypes.byref(density)
        )
        if ret != 0:
            return 0
        return density.value

    def get_turret_gratings(self, handle: int) -> int:
        """获取当前转塔上的光栅数量。"""
        count = ctypes.c_long()
        ret = self._lib.ARC_get_Mono_Turret_Gratings(ctypes.c_long(handle), ctypes.byref(count))
        if ret != 0:
            return 0
        return count.value

    # ------------------------------------------------------------------
    # 狭缝控制
    # ------------------------------------------------------------------
    def get_slit_width(self, handle: int, slit_index: int = 0) -> int:
        """获取狭缝宽度（微米）。slit_index: 0=入口, 1=出口, 2=中间。"""
        width = ctypes.c_long()
        ret = self._lib.ARC_get_Mono_Slit_Width(
            ctypes.c_long(handle), ctypes.c_long(slit_index), ctypes.byref(width)
        )
        if ret != 0:
            return 0
        return width.value

    def set_slit_width(self, handle: int, width_um: int, slit_index: int = 0) -> None:
        """设置狭缝宽度（微米）。"""
        ret = self._lib.ARC_set_Mono_Slit_Width(
            ctypes.c_long(handle), ctypes.c_long(slit_index), ctypes.c_long(width_um)
        )
        if ret != 0:
            raise ARCSpectraError(f"设置狭缝宽度 {width_um} um 失败: 错误码 {ret}")

    # ------------------------------------------------------------------
    # 扫描控制
    # ------------------------------------------------------------------
    def get_scan_rate(self, handle: int) -> float:
        """获取扫描速率（nm/min）。"""
        rate = ctypes.c_double()
        ret = self._lib.ARC_get_Mono_Scan_Rate_nm_min(ctypes.c_long(handle), ctypes.byref(rate))
        if ret != 0:
            return 0.0
        return rate.value

    def set_scan_rate(self, handle: int, rate_nm_per_min: float) -> None:
        """设置扫描速率（nm/min）。"""
        ret = self._lib.ARC_set_Mono_Scan_Rate_nm_min(
            ctypes.c_long(handle), ctypes.c_double(rate_nm_per_min)
        )
        if ret != 0:
            raise ARCSpectraError(f"设置扫描速率失败: 错误码 {ret}")

    def start_scan_to(self, handle: int, target_nm: float) -> None:
        """开始扫描到目标波长。"""
        ret = self._lib.ARC_Mono_Start_Scan_To_nm(
            ctypes.c_long(handle), ctypes.c_double(target_nm)
        )
        if ret != 0:
            raise ARCSpectraError(f"开始扫描失败: 错误码 {ret}")

    def scan_done(self, handle: int, target_nm: float) -> bool:
        """检查扫描是否完成。"""
        done = ctypes.c_long()
        ret = self._lib.ARC_Mono_Scan_Done(
            ctypes.c_long(handle), ctypes.byref(done), ctypes.c_double(target_nm)
        )
        return done.value != 0

    # ------------------------------------------------------------------
    # 设备信息
    # ------------------------------------------------------------------
    def get_model(self, handle: int) -> str:
        """获取设备型号。"""
        buf = ctypes.create_string_buffer(256)
        ret = self._lib.ARC_get_Mono_Model_CString(ctypes.c_long(handle), buf)
        if ret != 0:
            return "Unknown"
        return buf.value.decode("utf-8", errors="ignore")

    def get_serial(self, handle: int) -> str:
        """获取序列号。"""
        buf = ctypes.create_string_buffer(256)
        ret = self._lib.ARC_get_Mono_Serial_CString(ctypes.c_long(handle), buf)
        if ret != 0:
            return "Unknown"
        return buf.value.decode("utf-8", errors="ignore")

    def get_focal_length(self, handle: int) -> float:
        """获取焦距（mm）。"""
        focal = ctypes.c_double()
        ret = self._lib.ARC_get_Mono_Focallength(ctypes.c_long(handle), ctypes.byref(focal))
        if ret != 0:
            return 0.0
        return focal.value

    def get_precision(self, handle: int) -> int:
        """获取波长精度（小数位数）。"""
        return self._lib.ARC_get_Mono_Precision(ctypes.c_long(handle))

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def close_all(self) -> None:
        """关闭所有打开的单色仪。"""
        for handle in self._mono_handles[:]:
            try:
                self.close_mono(handle)
            except Exception:
                pass

    def __del__(self):
        self.close_all()


def get_arc_binding(dll_path: Optional[str] = None) -> ARCSpectraBinding:
    """获取 ARCSpectraBinding 实例。"""
    return ARCSpectraBinding(dll_path)