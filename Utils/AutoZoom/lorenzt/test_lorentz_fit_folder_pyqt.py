from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent / "lorentz_fit_folder_pyqt.py"


def _load_module():
    module_name = "lorentz_fit_folder_pyqt_for_test"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_spectrum_csv(path: Path, center: float) -> None:
    x = np.linspace(498.0, 534.0, 1024)
    y = 1210.0 + 100.0 * 1.1 / (4.0 * (x - center) ** 2 + 1.1 ** 2)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "wavelength", "raw_y", "threshold_filtered_y", "median_filtered_y"])
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            writer.writerow([i, f"{x_i:.6f}", f"{y_i:.6f}", f"{y_i:.6f}", f"{y_i:.6f}"])


def test_process_folder_outputs_curves_plots_and_summary():
    module = _load_module()
    with tempfile.TemporaryDirectory(prefix="lorentz_pyqt_folder_") as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        _write_spectrum_csv(input_dir / "cycle_0001_spectrum.csv", center=516.4)
        _write_spectrum_csv(input_dir / "cycle_0002_spectrum.csv", center=517.2)

        cfg = module.LorentzFitConfig(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            x_column="wavelength",
            y_column="median_filtered_y",
            x_min=513.0,
            x_max=520.0,
        )
        results = module.process_folder(cfg)

        assert len(results) == 2
        assert all(r.ok for r in results)
        assert abs(results[0].c - 516.4) < 0.05
        assert abs(results[1].c - 517.2) < 0.05
        for result in results:
            assert Path(result.curve_csv).exists()
            assert Path(result.params_json).exists()
            assert Path(result.plot_png).exists()
        assert (output_dir / "lorentz_fit_summary.csv").exists()
        assert (output_dir / "lorentz_fit_summary.png").exists()
