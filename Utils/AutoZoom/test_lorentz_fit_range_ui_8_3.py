from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent / "0_measurement_workflow_real_virtual_same_detection_8_3.py"


def _load_module():
    module_name = "measurement_workflow_8_3_for_lorentz_range_test"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_filter_xy_by_x_range_keeps_only_manual_fit_range():
    module = _load_module()
    gui = object.__new__(module.MeasurementWorkflowGUI)

    xs, ys = gui._filter_xy_by_x_range(
        [514.0, 515.0, 516.0, 517.0, 518.0],
        [10.0, 20.0, 30.0, 20.0, 10.0],
        515.0,
        517.0,
    )

    assert xs == [515.0, 516.0, 517.0]
    assert ys == [20.0, 30.0, 20.0]


def test_lorentz_fit_uses_filtered_range_and_reports_quality():
    module = _load_module()
    gui = object.__new__(module.MeasurementWorkflowGUI)

    x = np.linspace(510.0, 522.0, 401)
    y = 1210.0 + 105.0 * 0.98 / (4.0 * (x - 516.44) ** 2 + 0.98 ** 2)
    y += 3.0 * np.sin((x - 510.0) * 1.7)

    filtered_x, filtered_y = gui._filter_xy_by_x_range(
        [float(v) for v in x],
        [float(v) for v in y],
        513.44,
        519.44,
    )
    result = gui._fit_lorentzian_curve(filtered_x, filtered_y)

    assert result is not None
    params = result["params"]
    quality = result["fit_quality"]
    assert abs(params["c"] - 516.44) < 0.05
    assert 0.7 < params["w"] < 1.3
    assert quality["n_points"] == len(filtered_x)
    assert math.isfinite(quality["r2"])
    assert quality["r2"] > 0.9
