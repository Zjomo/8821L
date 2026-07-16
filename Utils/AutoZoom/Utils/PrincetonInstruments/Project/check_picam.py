"""检查 Picam.dll 版本和位数。"""
import struct
import ctypes
from ctypes import wintypes

path = r"C:\Program Files\Common Files\Princeton Instruments\Picam\Runtime\Picam.dll"

with open(path, "rb") as f:
    data = f.read()

pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
machine = struct.unpack_from("<H", data, pe_offset + 4)[0]
print(f"Picam.dll 路径: {path}")
print(f"Picam.dll 架构: {'x64' if machine == 0x8664 else 'x86'}")

# 检查导出函数
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ISOPLANEControl.analyze_dlls import pe_export_names

try:
    arch, count, names = pe_export_names(path)
    print(f"Picam.dll 导出函数总数: {count}")
    if "Picam_FreeCameraIDs" in names:
        print("Picam_FreeCameraIDs: 存在")
    else:
        print("Picam_FreeCameraIDs: 不存在（版本过旧）")
except Exception as e:
    print(f"解析 Picam.dll 失败: {e}")

# 获取文件版本
wapi = ctypes.windll.version
filename = path.encode("utf-16-le")
sz = wapi.GetFileVersionInfoSizeW(filename, None)
if sz:
    buf = ctypes.create_string_buffer(sz)
    wapi.GetFileVersionInfoW(filename, 0, sz, buf)
    val = ctypes.c_void_p()
    length = wintypes.UINT()
    lang = b"\\VarFileInfo\\Translation"
    if wapi.VerQueryValueA(buf, lang, ctypes.byref(val), ctypes.byref(length)):
        lang_code = ctypes.string_at(val, 4)
        lang_id, codepage = struct.unpack("<HH", lang_code)
        for key in ["FileVersion", "ProductVersion", "ProductName"]:
            query = f"\\StringFileInfo\\{lang_id:04x}{codepage:04x}\\{key}".encode()
            ret = wapi.VerQueryValueA(buf, query, ctypes.byref(val), ctypes.byref(length))
            if ret:
                value = ctypes.string_at(val, length.value).decode("utf-8", errors="ignore").strip()
                print(f"{key}: {value}")
