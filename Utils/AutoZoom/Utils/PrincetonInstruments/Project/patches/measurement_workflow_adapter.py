"""现有测量工作流的最小改动补丁。

目标：让 0_measurement_workflow_real_virtual_same_detection_6.25.py
      在不启动 LabVIEW TCP 的情况下，直接通过 PICam SDK 采集光谱。

本文件提供：
  1. MeasurementWorkflow 扩展类：新增 `use_picam_direct` 开关
  2. _create_labview_tcp_server 的替代实现
  3. 在原始文件中的最小修改说明
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from patches.pi_spectrometer_adapter import PISpectrometerAdapter


# ---------------------------------------------------------------------------
# 修改说明（请在 0_measurement_workflow_real_virtual_same_detection_6.25.py 中执行）
# ---------------------------------------------------------------------------
SETUP_INSTRUCTIONS = """
修改步骤（最小侵入式）：

1. 在文件顶部新增导入：

    from pi_spectrometer.patches.measurement_workflow_adapter import patch_measurement_workflow

2. 在 MeasurementWorkflow.__init__ 末尾调用 patch：

    patch_measurement_workflow(self)

3. 在 DEFAULT_CONFIG 中新增：

    'spectrometer_backend': 'labview_tcp',   # 选项: 'labview_tcp', 'picam', 'picam_demo', 'mock'
    'picam_dll_path': None,
    'picam_camera_index': 0,
    'picam_exposure': 0.1,
    'picam_temperature': -25.0,

4. （可选）在 run_full_measurement 开头跳过 TCP 等待：

    if self.config.get('spectrometer_backend', 'labview_tcp') == 'labview_tcp':
        wait_labview_ready()
    else:
        self._start_picam_backend()

完成。工作流将根据配置自动选择 LabVIEW TCP 或直接 PI 控制。
"""


def create_picam_adapter_from_config(config: Dict[str, Any]) -> PISpectrometerAdapter:
    """从配置字典创建 PI 适配器。"""
    backend_type = config.get("spectrometer_backend", "picam_demo")
    # 移除 labview_tcp 前缀，保留 picam / demo / mock
    if backend_type == "picam_demo":
        backend_type = "demo"
    elif backend_type == "picam":
        backend_type = "picam"
    elif backend_type == "mock":
        backend_type = "mock"
    else:
        raise ValueError(f"不支持的 PI 后端类型: {backend_type}")

    return PISpectrometerAdapter(
        backend_type=backend_type,
        output_dir=config.get("tcp_output_dir", "labview_csv_output"),
        dll_path=config.get("picam_dll_path") or None,
        camera_index=int(config.get("picam_camera_index", 0)),
        exposure=float(config.get("picam_exposure", 0.1)),
        sensor_temperature=float(config.get("picam_temperature", -25.0)),
    )


def patch_measurement_workflow(workflow_obj: Any) -> None:
    """在 MeasurementWorkflow 实例上绑定 PI 直接控制方法。

    该方法通过 monkey-patch 方式扩展 workflow 实例，不修改原类文件。
    """

    def _create_picam_server(self) -> PISpectrometerAdapter:
        """替代 _create_labview_tcp_server。"""
        adapter = create_picam_adapter_from_config(self.config)
        adapter.start_server_blocking()
        return adapter

    def _start_picam_backend(self) -> Dict[str, Any]:
        """启动 PI 后端并等待就绪。"""
        if not hasattr(self, "_picam_adapter") or self._picam_adapter is None:
            self._picam_adapter = _create_picam_server(self)
        return self._picam_adapter.wait_for_ready()

    def _request_picam_spectrum(self, cycle_index: int) -> Dict[str, Any]:
        """替代 request_labview_spectrum 的 PI 直接版本。"""
        if not hasattr(self, "_picam_adapter") or self._picam_adapter is None:
            raise RuntimeError("PI 后端尚未启动")
        return self._picam_adapter.request_measure(index=cycle_index)

    def _close_picam_backend(self) -> None:
        """关闭 PI 后端。"""
        adapter = getattr(self, "_picam_adapter", None)
        if adapter is not None:
            adapter.close()
            self._picam_adapter = None

    # 绑定到实例
    workflow_obj._create_picam_server = lambda: _create_picam_server(workflow_obj)
    workflow_obj._start_picam_backend = lambda: _start_picam_backend(workflow_obj)
    workflow_obj._request_picam_spectrum = lambda idx: _request_picam_spectrum(workflow_obj, idx)
    workflow_obj._close_picam_backend = lambda: _close_picam_backend(workflow_obj)


def decide_spectrum_method(workflow_obj: Any) -> str:
    """根据配置决定使用哪种光谱采集方法。"""
    backend = workflow_obj.config.get("spectrometer_backend", "labview_tcp")
    if backend in ("picam", "picam_demo", "mock"):
        return "picam_direct"
    return "labview_tcp"


# ---------------------------------------------------------------------------
# 可直接运行的独立示例
# ---------------------------------------------------------------------------
def standalone_demo():
    """不依赖原有工作流的独立闭环示例。"""
    config = {
        "spectrometer_backend": "mock",
        "tcp_output_dir": "pi_csv_output",
        "picam_exposure": 0.05,
        "picam_temperature": -20.0,
    }

    adapter = create_picam_adapter_from_config(config)
    adapter.start_server_blocking()
    ready = adapter.wait_for_ready()
    print("READY:", ready)

    for i in range(3):
        result = adapter.request_measure(index=i + 1)
        print(f"采集 {i+1}: ok={result['ok']}, 点数={result['num_points']}, "
              f"峰值={result.get('fit_peak')}")

    adapter.close()


if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)
    print("\n--- 独立示例 ---")
    standalone_demo()
