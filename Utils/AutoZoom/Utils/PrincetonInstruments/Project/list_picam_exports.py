"""列出 Picam.dll 的导出函数名。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ISOPLANEControl.analyze_dlls import pe_export_names

path = r"C:\Program Files\Common Files\Princeton Instruments\Picam\Runtime\Picam.dll"
arch, count, names = pe_export_names(path)
print(f"架构: {arch}, 导出函数总数: {count}, 具名导出: {len(names)}\n")

names.sort()
for name in names:
    if "Free" in name or "Destroy" in name or "CameraID" in name or "Initialize" in name or "Uninitialize" in name or "GetAvailable" in name:
        print(f"  {name}")

print("\n--- 所有导出函数 ---")
for name in names:
    print(name)
