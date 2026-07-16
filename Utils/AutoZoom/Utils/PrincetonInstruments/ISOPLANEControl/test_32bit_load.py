"""测试 32 位 Python 加载 ARC DLL。"""
import ctypes
import os
import struct
import sys

DLLS = [
    "ARC_Instrument.dll",
    "ARC_Spectra.dll",
    "ARC_SpectraPro.dll",
]

print(f"Python: {sys.executable}")
print(f"Python bits: {struct.calcsize('P') * 8}")
print(f"Working dir: {os.getcwd()}")

for dll in DLLS:
    path = os.path.abspath(dll)
    print(f"\n=== {dll} ===")
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        continue
    try:
        lib = ctypes.CDLL(path)
        print(f"加载成功")
        # 尝试访问一个函数
        try:
            print(f"ARC_Ver 地址: {lib.ARC_Ver}")
        except Exception as e:
            print(f"访问 ARC_Ver 失败: {e}")
    except Exception as e:
        print(f"加载失败: {e}")

print("\n测试完成")
