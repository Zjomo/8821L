"""测试 PICam 绑定初始化过程。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pi_spectrometer.picam.binding import _find_picam_dll

path = _find_picam_dll()
print(f"Picam.dll 路径: {path}")

import ctypes
lib = ctypes.CDLL(str(path))
print(f"Picam.dll 加载成功")

# 逐行模拟 _setup_function_signatures，看哪个函数失败
functions_to_check = [
    ("Picam_InitializeLibrary", []),
    ("Picam_UninitializeLibrary", []),
    ("Picam_GetAvailableCameraIDs", [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_int)]),  # 简化
    ("Picam_FreeCameraIDs", [ctypes.POINTER(None)]),
]

for name, _ in functions_to_check:
    try:
        attr = getattr(lib, name)
        print(f"  {name}: 可访问")
    except AttributeError as e:
        print(f"  {name}: 不可访问 - {e}")
