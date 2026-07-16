"""诊断 Princeton Instruments 硬件连接环境。"""
import ctypes
import os
import struct
import sys

# 把项目根目录加入路径
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)


def check_python_bits():
    bits = struct.calcsize("P") * 8
    print(f"[Python] 解释器: {sys.executable}")
    print(f"[Python] 位数: {bits}-bit")
    return bits


def check_picam_sdk():
    print("\n[PICam SDK]")
    from pi_spectrometer.picam.binding import _find_picam_dll

    dll_path = _find_picam_dll()
    if dll_path is None:
        print("  状态: 未找到 Picam.dll")
        print("  建议: 安装 PICam SDK（64-bit），通常位于 C:\\Program Files\\Common Files\\Princeton Instruments\\Picam\\Runtime")
        return False

    print(f"  Picam.dll: {dll_path}")
    try:
        lib = ctypes.CDLL(str(dll_path))
        print("  加载: 成功")
    except Exception as e:
        print(f"  加载: 失败 - {e}")
        return False

    required = [
        "Picam_InitializeLibrary",
        "Picam_GetAvailableCameraIDs",
        "Picam_FreeCameraIDs",
        "Picam_UninitializeLibrary",
    ]
    missing = []
    for name in required:
        try:
            getattr(lib, name)
            print(f"  {name}: 存在")
        except AttributeError:
            print(f"  {name}: 不存在")
            missing.append(name)

    if missing:
        print(f"  警告: 缺少 {len(missing)} 个函数，PICam SDK 版本可能过旧")
        print("  建议: 升级到最新版 PICam SDK（64-bit）")
        return False
    return True


def check_arc_sdk():
    print("\n[ARC SDK (IsoPlane/SpectraPro)]")
    from pi_spectrometer.picam.arc_binding import _find_arc_dll, _get_dll_bits

    dll_path = _find_arc_dll()
    if dll_path is None:
        print("  状态: 未找到 ARC_SpectraPro.dll")
        print("  建议: 安装 ISOPLANEControl，或设置 ARC_SpectraPro_Root 环境变量")
        return False

    print(f"  ARC_SpectraPro.dll: {dll_path}")
    dll_bits = _get_dll_bits(str(dll_path))
    print(f"  DLL 位数: {dll_bits}-bit")

    py_bits = struct.calcsize("P") * 8
    if dll_bits != py_bits:
        print(f"  错误: Python ({py_bits}-bit) 与 DLL ({dll_bits}-bit) 位数不匹配")
        print(f"  建议: 使用 {dll_bits}-bit Python 解释器运行本项目")
        print(f"  下载: https://www.python.org/downloads/release/python-3913/")
        return False

    try:
        lib = ctypes.CDLL(str(dll_path))
        print("  加载: 成功")
    except Exception as e:
        print(f"  加载: 失败 - {e}")
        return False

    try:
        print(f"  ARC_Ver 地址: {lib.ARC_Ver}")
    except Exception as e:
        print(f"  访问 ARC_Ver 失败: {e}")
        return False

    return True


def check_com_ports():
    print("\n[串口设备]")
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("  未检测到串口设备")
        else:
            for p in ports:
                print(f"  {p.device}: {p.description} ({p.hwid})")
    except ImportError:
        print("  未安装 pyserial，无法扫描串口")


def main():
    print("=" * 60)
    print("Princeton Instruments 硬件环境诊断")
    print("=" * 60)

    check_python_bits()
    picam_ok = check_picam_sdk()
    arc_ok = check_arc_sdk()
    check_com_ports()

    print("\n" + "=" * 60)
    print("诊断摘要")
    print("=" * 60)
    print(f"PICam 相机/探测器: {'可用' if picam_ok else '不可用'}")
    print(f"ARC 单色仪/光谱仪: {'可用' if arc_ok else '不可用'}")

    if not picam_ok or not arc_ok:
        print("\n建议操作:")
        if not picam_ok:
            print("  1. 安装/升级 PICam SDK（64-bit）")
            print("     https://www.princetoninstruments.com/products/software/picam")
        if not arc_ok:
            print("  2. 使用与 ARC_SpectraPro.dll 位数匹配的 Python 解释器")
            print("     若 DLL 为 32-bit，请安装 32-bit Python 3.9")


if __name__ == "__main__":
    main()
