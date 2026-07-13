"""
验证测量工作流相关默认配置已切换到真实实验室环境。

覆盖：
  1. config_angle_repair_fixed.py 中照明串口为 COM20
  2. config_angle_repair_fixed.py 中硬件模式为 real
  3. config_angle_repair_fixed.py 中 YOLO-OBB 默认模型路径指向 .\vision\best_wan12.2.pt
  4. 主程序 0_measurement_workflow_real_virtual_same_detection_6.25.py 的 fallback 默认值同步更新
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path
from types import ModuleType

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

import config_angle_repair_fixed as cfg_mod

MAIN_SOURCE = AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_6.25.py"


def _load_module_at(path: Path, name: str) -> ModuleType:
    """动态加载指定路径的 Python 模块，避免与已导入模块命名冲突。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _assert_default_config(config_mod: ModuleType, label: str) -> None:
    """断言给定配置模块的默认值符合真实实验室环境要求。"""
    cfg = config_mod.DEFAULT_CONFIG

    assert cfg["light_port"] == "COM20", (
        f"[{label}] light_port 应为 COM20，实际为 {cfg['light_port']}"
    )
    assert cfg.get("hardware_mode") == "real", (
        f"[{label}] hardware_mode 应为 real，实际为 {cfg.get('hardware_mode')}"
    )

    configured = Path(cfg["angle_model_path"]).expanduser()
    expected = AUTOZOOM_ROOT / "vision" / "best_wan12.2.pt"
    if not configured.is_absolute():
        configured = AUTOZOOM_ROOT / configured
    assert configured.resolve() == expected.resolve(), (
        f"[{label}] angle_model_path 解析结果应为 {expected.resolve()}，"
        f"实际为 {configured.resolve()}"
    )


def test_default_light_port_com20() -> None:
    """照明光串口默认值应为 COM20。"""
    assert cfg_mod.DEFAULT_CONFIG["light_port"] == "COM20", (
        f"light_port 应为 COM20，实际为 {cfg_mod.DEFAULT_CONFIG['light_port']}"
    )
    print("PASS: default_light_port_com20")


def test_default_hardware_mode_real() -> None:
    """硬件模式默认值应为 real。"""
    assert cfg_mod.DEFAULT_CONFIG.get("hardware_mode") == "real", (
        f"hardware_mode 应为 real，实际为 {cfg_mod.DEFAULT_CONFIG.get('hardware_mode')}"
    )
    print("PASS: default_hardware_mode_real")


def test_default_angle_model_path() -> None:
    """YOLO-OBB 默认模型路径应解析为 Utils/AutoZoom/vision/best_wan12.2.pt。"""
    configured = Path(cfg_mod.DEFAULT_CONFIG["angle_model_path"]).expanduser()
    expected = AUTOZOOM_ROOT / "vision" / "best_wan12.2.pt"

    if not configured.is_absolute():
        configured = AUTOZOOM_ROOT / configured

    assert configured.resolve() == expected.resolve(), (
        f"angle_model_path 解析结果应为 {expected.resolve()}，实际为 {configured.resolve()}"
    )

    if not expected.exists():
        print(
            f"WARN: 模型文件不存在：{expected}，真实测试前请将 best_wan12.2.pt 放置到该路径"
        )
    else:
        print(f"INFO: 模型文件存在：{expected}")

    print("PASS: default_angle_model_path")


def test_utils_config_defaults_sync() -> None:
    """Utils/AutoZoom/Utils/config_angle_repair_fixed.py 的默认值应与主配置一致。"""
    utils_cfg_path = AUTOZOOM_ROOT / "Utils" / "config_angle_repair_fixed.py"
    assert utils_cfg_path.exists(), f"重复配置文件不存在：{utils_cfg_path}"

    utils_cfg_mod = _load_module_at(utils_cfg_path, "utils_config_angle_repair_fixed")
    _assert_default_config(utils_cfg_mod, "Utils/config")
    print("PASS: utils_config_defaults_sync")


def test_main_source_fallback_strings() -> None:
    """主程序源码中的 fallback 默认值应已更新。"""
    assert MAIN_SOURCE.exists(), f"主程序源码不存在：{MAIN_SOURCE}"
    source = MAIN_SOURCE.read_text(encoding="utf-8")

    checks = [
        ('hardware_mode fallback', '_cfg("hardware_mode", "real")'),
        ('light_port fallback', '_cfg("light_port", "COM20")'),
        ('angle_model_path fallback', r'_cfg("angle_model_path", r".\vision\best_wan12.2.pt")'),
        ('build_config_from_ui hardware_mode or real', 'hardware_mode=str(self.hardware_mode_var.get()).strip() or "real"'),
    ]

    for name, needle in checks:
        assert needle in source, f"主程序源码中未找到 {name}：{needle}"

    print("PASS: main_source_fallback_strings")


if __name__ == "__main__":
    tests = [
        test_default_light_port_com20,
        test_default_hardware_mode_real,
        test_default_angle_model_path,
        test_utils_config_defaults_sync,
        test_main_source_fallback_strings,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            failed += 1
            print(f"FAIL: {test.__name__}: {e}")
            traceback.print_exc()

    if failed == 0:
        print("\nAll tests passed!")
    else:
        print(f"\n{failed}/{len(tests)} tests failed.")
        sys.exit(1)
