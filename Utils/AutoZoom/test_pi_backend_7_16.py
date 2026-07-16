"""临时测试脚本：验证 7_16 版本 PI mock 后端能完成启动-就绪-采集闭环。"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_PI = SCRIPT_DIR / "Utils" / "PrincetonInstruments" / "Project"
if str(PROJECT_PI) not in sys.path:
    sys.path.insert(0, str(PROJECT_PI))

import importlib.util
workflow_path = SCRIPT_DIR / "0_measurement_workflow_real_virtual_same_detection_7_16.py"
spec = importlib.util.spec_from_file_location(
    "workflow_7_16", str(workflow_path)
)
mod = importlib.util.module_from_spec(spec)
sys.modules["workflow_7_16"] = mod
spec.loader.exec_module(mod)

print("Import OK")

cfg = mod.MeasurementConfig(
    spectrometer_backend="picam_demo",
    picam_exposure=0.05,
    picam_temperature=-20.0,
    picam_roi_width=1024,
    picam_roi_height=256,
)

wf = mod.MeasurementWorkflow(cfg, on_log=print)

print("\n--- start_tcp_server ---")
result = wf.start_tcp_server()
print("start result:", result)

print("\n--- wait_labview_ready ---")
ready = wf.wait_labview_ready()
print("ready:", ready)

print("\n--- request_labview_spectrum ---")
spec_result = wf.request_labview_spectrum(cycle_index=1)
print("spectrum ok:", spec_result.get("ok"))
print("num_points:", spec_result.get("num_points"))
print("raw_peak:", spec_result.get("raw_peak"))
print("fit_peak:", spec_result.get("fit_peak"))

print("\n--- close tcp_server ---")
wf.tcp_server.close()
wf.tcp_server = None
print("Closed")
