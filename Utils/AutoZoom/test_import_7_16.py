"""临时测试脚本：验证 7_16 版本可导入且设备选择配置生效。"""

import sys
from pathlib import Path

# 添加 PI 适配器路径
PROJECT_PI = Path(__file__).resolve().parent / "Utils" / "AutoZoom" / "Utils" / "PrincetonInstruments" / "Project"
if str(PROJECT_PI) not in sys.path:
    sys.path.insert(0, str(PROJECT_PI))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "workflow_7_16", "0_measurement_workflow_real_virtual_same_detection_7_16.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["workflow_7_16"] = mod  # 注册模块名，解决 @dataclass 问题
spec.loader.exec_module(mod)

print("Import OK")
print("backend default:", mod.DEFAULT_CONFIG.get("spectrometer_backend"))
print("cfg has backend:", hasattr(mod.MeasurementConfig(), "spectrometer_backend"))

# 验证 mock 后端能创建适配器
cfg = mod.MeasurementConfig(spectrometer_backend="mock")
print("mock cfg backend:", cfg.spectrometer_backend)
