"""分析 ARC SDK DLL 的导出函数（不依赖第三方库）。"""
import os
import struct
import sys

DLLS = [
    "ARC_Instrument.dll",
    "ARC_Spectra.dll",
    "ARC_SpectraPro.dll",
]


def pe_export_names(path: str):
    """读取 PE 文件的导出函数名列表。"""
    with open(path, "rb") as f:
        data = f.read()
    size = len(data)

    if data[:2] != b"MZ":
        raise ValueError("不是有效的 PE/Windows 可执行文件")

    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset + 4 > size or data[pe_offset : pe_offset + 4] != b"PE\x00\x00":
        raise ValueError("PE 头签名错误")

    machine = struct.unpack_from("<H", data, pe_offset + 4)[0]
    arch = "x64" if machine == 0x8664 else "x86"

    num_sections = struct.unpack_from("<H", data, pe_offset + 6)[0]
    optional_header_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
    optional_header_offset = pe_offset + 24
    section_table_offset = optional_header_offset + optional_header_size

    if section_table_offset + num_sections * 40 > size:
        raise ValueError("节表超出文件范围")

    magic = struct.unpack_from("<H", data, optional_header_offset)[0]
    is_64 = magic == 0x20B

    # DataDirectory[0] = Export Directory
    export_dir_entry_offset = optional_header_offset + (112 if is_64 else 96)
    export_dir_rva = struct.unpack_from("<I", data, export_dir_entry_offset)[0]

    if export_dir_rva == 0:
        return arch, 0, []

    def rva_to_file_offset(rva):
        for i in range(num_sections):
            sec_offset = section_table_offset + i * 40
            v_addr = struct.unpack_from("<I", data, sec_offset + 12)[0]
            p_offset = struct.unpack_from("<I", data, sec_offset + 20)[0]
            v_size = struct.unpack_from("<I", data, sec_offset + 16)[0]
            if v_addr <= rva < v_addr + v_size:
                return rva - v_addr + p_offset
        return None

    export_dir_offset = rva_to_file_offset(export_dir_rva)
    if export_dir_offset is None or export_dir_offset + 40 > size:
        raise ValueError("无法解析导出目录")

    num_funcs = struct.unpack_from("<I", data, export_dir_offset + 24)[0]
    num_names = struct.unpack_from("<I", data, export_dir_offset + 28)[0]
    names_rva = struct.unpack_from("<I", data, export_dir_offset + 36)[0]

    names = []
    if num_names > 0 and names_rva:
        names_offset = rva_to_file_offset(names_rva)
        if names_offset is not None and names_offset + num_names * 4 <= size:
            for i in range(num_names):
                name_rva = struct.unpack_from("<I", data, names_offset + i * 4)[0]
                name_offset = rva_to_file_offset(name_rva)
                if name_offset is None or name_offset >= size:
                    continue
                end = data.find(b"\x00", name_offset)
                if end == -1:
                    end = size
                try:
                    names.append(data[name_offset:end].decode("utf-8", errors="ignore"))
                except Exception:
                    pass

    return arch, num_funcs, names


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for name in DLLS:
        path = os.path.join(base_dir, name)
        print(f"\n=== {name} ===")
        if not os.path.exists(path):
            print(f"文件不存在: {path}")
            continue
        try:
            arch, count, names = pe_export_names(path)
            print(f"架构: {arch}, 导出函数总数: {count}, 具名导出: {len(names)}")
            names.sort()
            for n in names[:80]:
                print(f"  {n}")
            if len(names) > 80:
                print(f"  ... 还有 {len(names)-80} 个函数")
        except Exception as e:
            print(f"分析失败: {e}")


if __name__ == "__main__":
    main()
