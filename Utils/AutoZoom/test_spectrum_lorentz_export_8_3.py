from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent / "0_measurement_workflow_real_virtual_same_detection_8_3.py"


def _load_module():
    module_name = "measurement_workflow_8_3_for_spectrum_lorentz_export_test"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cycle_spectrum_csv_writes_median_lorentz_fit_column_and_plot_peak():
    module = _load_module()
    with tempfile.TemporaryDirectory(prefix="spectrum_lorentz_export_") as tmpdir:
        cfg = module.MeasurementConfig()
        cfg.save_root = tmpdir
        cfg.lorentz_fit_x_min_nm = 513.0
        cfg.lorentz_fit_x_max_nm = 519.0
        wf = module.MeasurementWorkflow(cfg)
        wf.begin_new_run_session()
        paths = wf.build_save_path(1)

        x = np.linspace(510.0, 522.0, 121)
        y = 1210.0 + 105.0 * 0.98 / (4.0 * (x - 516.44) ** 2 + 0.98 ** 2)
        ctx = {
            "x_axis_values": [float(v) for v in x],
            "raw_values": [float(v) for v in y],
            "raw_filtered_values": [float(v) for v in y],
            "raw_median_values": [float(v) for v in y],
            "fit_values": [float(v) for v in y],
            "save_angle_deg": 25.5,
            "fit_peak": float(np.max(y)),
        }

        wf.save_cycle_result(
            cycle_index=1,
            paths=paths,
            angle_before_result={},
            angle_after_result={},
            labview_result={},
            context_snapshot=ctx,
            append_plot_point=True,
        )

        with open(paths["spectrum_csv"], "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        assert "median_filtered_lorentz_fit_y" in rows[0]
        inside = [r for r in rows if 513.0 <= float(r["wavelength"]) <= 519.0]
        outside = [r for r in rows if float(r["wavelength"]) < 513.0 or float(r["wavelength"]) > 519.0]
        assert any(r["median_filtered_lorentz_fit_y"] for r in inside)
        assert all(r["median_filtered_lorentz_fit_y"] == "" for r in outside)
        assert len(wf.plot_points) == 1
        assert wf.plot_points[0]["median_filtered_lorentz_fit_peak"] is not None
        assert wf.plot_points[0]["median_filtered_lorentz_fit_peak_x"] is not None


class _FakeStatusVar:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value


class _FakeTree:
    def __init__(self):
        self.rows = {"existing": {"values": (0, 25.5, 1300.0), "tags": ()}}
        self.next_id = 0
        self.deleted = []

    def get_children(self):
        return list(self.rows.keys())

    def item(self, item, option=None):
        data = self.rows[item]
        if option == "tags":
            return data.get("tags", ())
        return data

    def delete(self, item):
        self.deleted.append(item)
        self.rows.pop(item, None)

    def insert(self, parent, index, values=(), tags=()):
        self.next_id += 1
        item = f"row_{self.next_id}"
        self.rows[item] = {"values": values, "tags": tags}
        return item

    def cget(self, name):
        if name == "height":
            return 10
        raise KeyError(name)


def test_angle_fit_list_empty_refresh_keeps_existing_rows_unless_forced():
    module = _load_module()
    gui = object.__new__(module.MeasurementWorkflowGUI)
    gui.angle_fit_tree = _FakeTree()
    gui.angle_fit_list_status_var = _FakeStatusVar()

    gui.update_angle_fit_xy_list([])
    assert "existing" in gui.angle_fit_tree.rows
    assert "未清空" in gui.angle_fit_list_status_var.value

    gui.update_angle_fit_xy_list([], allow_empty_clear=True)
    assert "existing" not in gui.angle_fit_tree.rows
