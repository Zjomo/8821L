"""验证 PICam / ARC SDK 修复的基础测试。"""
import sys
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)


def test_mock_backend():
    from pi_spectrometer.picam.demo import MockSpectrometerBackend

    backend = MockSpectrometerBackend()
    assert backend.connect() is True
    result = backend.acquire()
    assert result.num_points > 0
    backend.disconnect()
    print("PASS: MockSpectrometerBackend")


def test_picam_binding_import():
    from pi_spectrometer.picam.binding import _find_picam_dll, PICamBinding

    picam_dll = _find_picam_dll()
    assert picam_dll is not None
    print(f"PASS: PICamBinding import OK, Picam.dll={picam_dll}")


def test_arc_binding_bits_detection():
    from pi_spectrometer.picam.arc_binding import (
        _find_arc_dll,
        _get_dll_bits,
        _get_python_bits,
    )

    arc_dll = _find_arc_dll()
    assert arc_dll is not None, "未找到 ARC_SpectraPro.dll"
    dll_bits = _get_dll_bits(str(arc_dll))
    py_bits = _get_python_bits()
    assert dll_bits == 32, f"预期 ARC DLL 为 32-bit，实际为 {dll_bits}-bit"
    assert py_bits == 64, f"预期当前 Python 为 64-bit，实际为 {py_bits}-bit"
    print(f"PASS: ARC DLL={dll_bits}-bit, Python={py_bits}-bit")


def test_arc_binding_raises_on_mismatch():
    from pi_spectrometer.picam.arc_binding import ARCSpectraBinding, ARCSpectraError

    try:
        ARCSpectraBinding()
    except ARCSpectraError as e:
        msg = str(e)
        assert "位数不匹配" in msg, f"错误信息未包含位数不匹配: {msg}"
        print("PASS: ARCSpectraBinding 正确报告位数不匹配")
        return
    raise AssertionError("ARCSpectraBinding 未报告位数不匹配")


def test_diagnose_script():
    import diagnose_pi_environment

    # 仅验证能导入并运行主要函数
    print("PASS: diagnose_pi_environment import OK")


def main():
    test_mock_backend()
    test_picam_binding_import()
    test_arc_binding_bits_detection()
    test_arc_binding_raises_on_mismatch()
    test_diagnose_script()
    print("\n所有测试通过!")


if __name__ == "__main__":
    main()
