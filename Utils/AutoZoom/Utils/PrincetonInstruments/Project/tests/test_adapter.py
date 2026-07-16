"""工作流适配器测试。"""

import sys
from pathlib import Path

import pytest

from patches.measurement_workflow_adapter import (
    create_picam_adapter_from_config,
    patch_measurement_workflow,
)


class FakeWorkflow:
    """模拟 MeasurementWorkflow 的最小结构。"""

    def __init__(self, config):
        self.config = config


def test_create_picam_adapter_mock():
    config = {
        "spectrometer_backend": "mock",
        "tcp_output_dir": "test_output",
        "picam_exposure": 0.05,
    }
    adapter = create_picam_adapter_from_config(config)
    assert adapter.backend_type == "mock"
    assert adapter.exposure == pytest.approx(0.05)


def test_create_picam_adapter_demo():
    config = {"spectrometer_backend": "picam_demo", "tcp_output_dir": "test_output"}
    adapter = create_picam_adapter_from_config(config)
    assert adapter.backend_type == "demo"


def test_create_picam_adapter_unsupported():
    config = {"spectrometer_backend": "labview_tcp"}
    with pytest.raises(ValueError):
        create_picam_adapter_from_config(config)


def test_adapter_full_lifecycle_mock(tmp_path):
    config = {
        "spectrometer_backend": "mock",
        "tcp_output_dir": str(tmp_path),
        "picam_exposure": 0.01,
    }
    adapter = create_picam_adapter_from_config(config)
    adapter.start_server_blocking()

    ready = adapter.wait_for_ready()
    assert ready["ok"] is True

    result = adapter.request_measure(index=1)
    assert result["ok"] is True
    assert result["num_points"] > 0
    assert len(result["values"]) == result["num_points"]
    assert result.get("csv_path") is not None

    adapter.close()
    assert not adapter.is_connected


def test_patch_measurement_workflow():
    config = {
        "spectrometer_backend": "mock",
        "tcp_output_dir": "test_output",
        "picam_exposure": 0.01,
    }
    workflow = FakeWorkflow(config)
    patch_measurement_workflow(workflow)

    assert hasattr(workflow, "_create_picam_server")
    assert hasattr(workflow, "_start_picam_backend")
    assert hasattr(workflow, "_request_picam_spectrum")
    assert hasattr(workflow, "_close_picam_backend")

    ready = workflow._start_picam_backend()
    assert ready["ok"] is True

    result = workflow._request_picam_spectrum(1)
    assert result["ok"] is True
    assert result["num_points"] > 0

    workflow._close_picam_backend()
