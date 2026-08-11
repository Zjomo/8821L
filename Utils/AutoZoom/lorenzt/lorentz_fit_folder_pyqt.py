from __future__ import annotations

import csv
import json
import math
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lorentzian_model(x: Any, y0: float, z: float, w: float, c: float):
    x_arr = np.asarray(x, dtype=float)
    return y0 + z * w / (4.0 * (x_arr - c) ** 2 + w ** 2)


@dataclass
class LorentzFitConfig:
    input_dir: str
    output_dir: str
    x_column: str = "wavelength"
    y_column: str = "median_filtered_y"
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    recursive: bool = False


@dataclass
class LorentzFitResult:
    source_csv: str
    output_dir: str
    ok: bool
    reason: str
    x_column: str = ""
    y_column: str = ""
    n_points: int = 0
    fit_x_min: Optional[float] = None
    fit_x_max: Optional[float] = None
    y0: Optional[float] = None
    z: Optional[float] = None
    w: Optional[float] = None
    c: Optional[float] = None
    peak_height_z_over_w: Optional[float] = None
    r2: Optional[float] = None
    rmse: Optional[float] = None
    curve_csv: str = ""
    params_json: str = ""
    plot_png: str = ""


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _normalize_header(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV 没有表头")
        headers = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return headers, rows


def _find_column(headers: List[str], preferred: str, aliases: Iterable[str]) -> Optional[str]:
    normalized = {_normalize_header(h): h for h in headers}
    preferred_norm = _normalize_header(preferred)
    if preferred_norm in normalized:
        return normalized[preferred_norm]
    for alias in aliases:
        alias_norm = _normalize_header(alias)
        if alias_norm in normalized:
            return normalized[alias_norm]
    return None


def read_spectrum_csv(path: Path, x_column: str = "wavelength", y_column: str = "median_filtered_y") -> Tuple[np.ndarray, np.ndarray, str, str]:
    headers, rows = _read_csv_rows(path)
    x_col = _find_column(headers, x_column, ["wavelength", "lambda", "x", "x_axis", "index", "波长", "中心波长"])
    y_col = _find_column(headers, y_column, ["median_filtered_y", "raw_median_values", "raw_y", "threshold_filtered_y", "intensity", "y", "value"])
    if y_col is None:
        raise ValueError(f"找不到可用 y 列，期望列={y_column}，实际列={headers}")

    xs: List[float] = []
    ys: List[float] = []
    for idx, row in enumerate(rows):
        y = _safe_float(row.get(y_col))
        if y is None:
            continue
        if x_col is None:
            x = float(idx)
        else:
            x = _safe_float(row.get(x_col))
            if x is None:
                x = float(idx)
        xs.append(float(x))
        ys.append(float(y))

    if len(xs) < 4:
        raise ValueError(f"有效光谱点数不足：{len(xs)}")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), (x_col or "index"), y_col


def fit_lorentzian_curve(
    x: np.ndarray,
    y: np.ndarray,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> Dict[str, Any]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if x.size < 4:
        raise ValueError(f"有效拟合点数不足：{x.size}")
    if x_min is not None and x_max is not None and x_min > x_max:
        x_min, x_max = x_max, x_min
    mask = np.ones_like(x, dtype=bool)
    if x_min is not None:
        mask &= x >= float(x_min)
    if x_max is not None:
        mask &= x <= float(x_max)
    x_fit_src = x[mask]
    y_fit_src = y[mask]
    if x_fit_src.size < 4:
        raise ValueError(f"指定范围内有效拟合点数不足：{x_fit_src.size}")

    order = np.argsort(x_fit_src)
    x_fit_src = x_fit_src[order]
    y_fit_src = y_fit_src[order]
    fit_x_min = float(np.nanmin(x_fit_src))
    fit_x_max = float(np.nanmax(x_fit_src))
    if fit_x_max <= fit_x_min:
        raise ValueError("拟合范围无有效跨度")

    y_base = float(np.nanpercentile(y_fit_src, 10))
    y_max = float(np.nanmax(y_fit_src))
    c0 = float(x_fit_src[int(np.nanargmax(y_fit_src))])
    w0 = max(float((fit_x_max - fit_x_min) / 8.0), 1e-6)
    z0 = max((y_max - y_base) * w0, 1e-6)
    lower = [float(np.nanmin(y_fit_src) - abs(y_max - y_base) * 5.0 - 1.0), 0.0, 1e-6, fit_x_min]
    upper = [float(y_max + abs(y_max - y_base) * 5.0 + 1.0), 1e12, max((fit_x_max - fit_x_min) * 2.0, 1e-6), fit_x_max]
    popt, _ = curve_fit(
        lorentzian_model,
        x_fit_src,
        y_fit_src,
        p0=[y_base, z0, w0, c0],
        bounds=(lower, upper),
        maxfev=100000,
    )
    y_pred_src = lorentzian_model(x_fit_src, *popt)
    residual = y_fit_src - y_pred_src
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_fit_src - np.nanmean(y_fit_src)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    rmse = float(math.sqrt(ss_res / float(x_fit_src.size))) if x_fit_src.size else None
    y_pred_all = np.full_like(x, np.nan, dtype=float)
    y_pred_all[mask] = lorentzian_model(x[mask], *popt)

    return {
        "params": {
            "y0": float(popt[0]),
            "z": float(popt[1]),
            "w": float(popt[2]),
            "c": float(popt[3]),
            "peak_height_z_over_w": float(popt[1] / popt[2]) if float(popt[2]) != 0 else None,
        },
        "quality": {
            "n_points": int(x_fit_src.size),
            "fit_x_min": fit_x_min,
            "fit_x_max": fit_x_max,
            "r2": r2,
            "rmse": rmse,
        },
        "x": x,
        "y": y,
        "mask": mask,
        "y_fit": y_pred_all,
    }


def _write_curve_csv(path: Path, x: np.ndarray, y: np.ndarray, y_fit: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "lorentz_fit_y", "residual"])
        for x_i, y_i, fit_i in zip(x, y, y_fit):
            if np.isfinite(fit_i):
                writer.writerow([f"{x_i:.10g}", f"{y_i:.10g}", f"{fit_i:.10g}", f"{(y_i - fit_i):.10g}"])
            else:
                writer.writerow([f"{x_i:.10g}", f"{y_i:.10g}", "", ""])


def _write_plot(path: Path, source_name: str, x: np.ndarray, y: np.ndarray, y_fit: np.ndarray, params: Dict[str, Any], quality: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6), dpi=140)
    plt.plot(x, y, linewidth=1.0, label="spectrum")
    fit_mask = np.isfinite(y_fit)
    if np.any(fit_mask):
        plt.plot(x[fit_mask], y_fit[fit_mask], linewidth=2.0, label=f"Lorentz c={params['c']:.6g}, w={params['w']:.6g}")
        plt.axvline(float(params["c"]), linestyle="--", linewidth=1.0)
    plt.title(f"Lorentz fit - {source_name}\nR²={quality.get('r2')}, RMSE={quality.get('rmse')}")
    plt.xlabel("wavelength(nm) / index")
    plt.ylabel("intensity")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def process_spectrum_file(path: Path, output_root: Path, config: LorentzFitConfig) -> LorentzFitResult:
    file_out = output_root / path.stem
    file_out.mkdir(parents=True, exist_ok=True)
    try:
        x, y, x_col, y_col = read_spectrum_csv(path, config.x_column, config.y_column)
        fit = fit_lorentzian_curve(x, y, config.x_min, config.x_max)
        params = fit["params"]
        quality = fit["quality"]

        curve_csv = file_out / f"{path.stem}_lorentz_fit_curve.csv"
        params_json = file_out / f"{path.stem}_lorentz_fit_params.json"
        plot_png = file_out / f"{path.stem}_lorentz_fit.png"
        _write_curve_csv(curve_csv, fit["x"], fit["y"], fit["y_fit"])
        payload = {
            "source_csv": str(path),
            "x_column": x_col,
            "y_column": y_col,
            "model": "y0 + z*w/(4*(x-c)^2+w^2)",
            "requested_range": {"x_min": config.x_min, "x_max": config.x_max},
            "params": params,
            "quality": quality,
            "curve_csv": str(curve_csv),
            "plot_png": str(plot_png),
        }
        params_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_plot(plot_png, path.name, fit["x"], fit["y"], fit["y_fit"], params, quality)
        return LorentzFitResult(
            source_csv=str(path),
            output_dir=str(file_out),
            ok=True,
            reason="ok",
            x_column=x_col,
            y_column=y_col,
            n_points=int(quality["n_points"]),
            fit_x_min=quality["fit_x_min"],
            fit_x_max=quality["fit_x_max"],
            y0=params["y0"],
            z=params["z"],
            w=params["w"],
            c=params["c"],
            peak_height_z_over_w=params["peak_height_z_over_w"],
            r2=quality["r2"],
            rmse=quality["rmse"],
            curve_csv=str(curve_csv),
            params_json=str(params_json),
            plot_png=str(plot_png),
        )
    except Exception as exc:
        err_json = file_out / f"{path.stem}_lorentz_fit_error.json"
        err_json.write_text(
            json.dumps({"source_csv": str(path), "error": str(exc), "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return LorentzFitResult(source_csv=str(path), output_dir=str(file_out), ok=False, reason=str(exc), params_json=str(err_json))


def _iter_input_csvs(input_dir: Path, recursive: bool = False) -> List[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    files = []
    for path in input_dir.glob(pattern):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if "lorentz_fit" in lower or lower.startswith("summary"):
            continue
        files.append(path)
    return sorted(files)


def write_summary(output_root: Path, results: List[LorentzFitResult]) -> Tuple[Path, Path]:
    summary_csv = output_root / "lorentz_fit_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(asdict(LorentzFitResult("", "", False, "")).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    summary_png = output_root / "lorentz_fit_summary.png"
    ok_results = [r for r in results if r.ok]
    if ok_results:
        x = np.arange(len(ok_results))
        labels = [Path(r.source_csv).stem for r in ok_results]
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), dpi=140, sharex=True)
        for ax, attr, title in [
            (axes[0], "c", "center c"),
            (axes[1], "w", "width w/FWHM"),
            (axes[2], "z", "z"),
            (axes[3], "r2", "R²"),
        ]:
            values = [getattr(r, attr) for r in ok_results]
            ax.plot(x, values, marker="o")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(summary_png)
        plt.close(fig)
    return summary_csv, summary_png


def process_folder(config: LorentzFitConfig, progress: Optional[Callable[[str], None]] = None) -> List[LorentzFitResult]:
    input_dir = Path(config.input_dir).resolve()
    output_root = Path(config.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"输入文件夹不存在：{input_dir}")
    files = _iter_input_csvs(input_dir, recursive=config.recursive)
    if not files:
        raise ValueError(f"输入文件夹没有可处理 CSV：{input_dir}")
    results: List[LorentzFitResult] = []
    for i, path in enumerate(files, start=1):
        if progress:
            progress(f"[{i}/{len(files)}] 处理：{path.name}")
        result = process_spectrum_file(path, output_root, config)
        results.append(result)
        if progress:
            progress(f"    {'OK' if result.ok else 'FAIL'}：{result.reason}")
    summary_csv, summary_png = write_summary(output_root, results)
    if progress:
        progress(f"汇总 CSV：{summary_csv}")
        progress(f"汇总图：{summary_png}")
    return results


class _FitWorker:
    pass


def _run_gui() -> int:
    from PyQt5 import QtCore, QtWidgets

    class FitWorker(QtCore.QThread):
        log = QtCore.pyqtSignal(str)
        done = QtCore.pyqtSignal(list)
        failed = QtCore.pyqtSignal(str)

        def __init__(self, cfg: LorentzFitConfig):
            super().__init__()
            self.cfg = cfg

        def run(self):
            try:
                results = process_folder(self.cfg, progress=self.log.emit)
                self.done.emit([asdict(r) for r in results])
            except Exception as exc:
                self.failed.emit(str(exc))

    class MainWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("批量光谱 Lorentz 拟合")
            self.resize(900, 620)
            self.worker: Optional[FitWorker] = None

            self.input_edit = QtWidgets.QLineEdit()
            self.output_edit = QtWidgets.QLineEdit()
            self.x_col_edit = QtWidgets.QLineEdit("wavelength")
            self.y_col_edit = QtWidgets.QLineEdit("median_filtered_y")
            self.x_min_edit = QtWidgets.QLineEdit()
            self.x_max_edit = QtWidgets.QLineEdit()
            self.recursive_check = QtWidgets.QCheckBox("递归处理子文件夹")
            self.run_btn = QtWidgets.QPushButton("开始批量拟合")
            self.progress = QtWidgets.QProgressBar()
            self.log_box = QtWidgets.QTextEdit()
            self.log_box.setReadOnly(True)

            form = QtWidgets.QGridLayout()
            form.addWidget(QtWidgets.QLabel("输入文件夹"), 0, 0)
            form.addWidget(self.input_edit, 0, 1)
            btn_in = QtWidgets.QPushButton("选择")
            form.addWidget(btn_in, 0, 2)
            form.addWidget(QtWidgets.QLabel("输出文件夹"), 1, 0)
            form.addWidget(self.output_edit, 1, 1)
            btn_out = QtWidgets.QPushButton("选择")
            form.addWidget(btn_out, 1, 2)
            form.addWidget(QtWidgets.QLabel("X列"), 2, 0)
            form.addWidget(self.x_col_edit, 2, 1)
            form.addWidget(QtWidgets.QLabel("Y列"), 3, 0)
            form.addWidget(self.y_col_edit, 3, 1)
            form.addWidget(QtWidgets.QLabel("拟合范围 min"), 4, 0)
            form.addWidget(self.x_min_edit, 4, 1)
            form.addWidget(QtWidgets.QLabel("拟合范围 max"), 5, 0)
            form.addWidget(self.x_max_edit, 5, 1)
            form.addWidget(self.recursive_check, 6, 1)

            layout = QtWidgets.QVBoxLayout(self)
            layout.addLayout(form)
            layout.addWidget(self.run_btn)
            layout.addWidget(self.progress)
            layout.addWidget(self.log_box)

            btn_in.clicked.connect(self.pick_input_dir)
            btn_out.clicked.connect(self.pick_output_dir)
            self.run_btn.clicked.connect(self.start_fit)

        def pick_input_dir(self):
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输入文件夹")
            if path:
                self.input_edit.setText(path)
                if not self.output_edit.text().strip():
                    self.output_edit.setText(str(Path(path) / "lorentz_fit_output"))

        def pick_output_dir(self):
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出文件夹")
            if path:
                self.output_edit.setText(path)

        def _optional_float(self, text: str) -> Optional[float]:
            s = str(text).strip()
            return None if not s else float(s)

        def start_fit(self):
            cfg = LorentzFitConfig(
                input_dir=self.input_edit.text().strip(),
                output_dir=self.output_edit.text().strip(),
                x_column=self.x_col_edit.text().strip() or "wavelength",
                y_column=self.y_col_edit.text().strip() or "median_filtered_y",
                x_min=self._optional_float(self.x_min_edit.text()),
                x_max=self._optional_float(self.x_max_edit.text()),
                recursive=bool(self.recursive_check.isChecked()),
            )
            self.log_box.clear()
            self.progress.setRange(0, 0)
            self.run_btn.setEnabled(False)
            self.worker = FitWorker(cfg)
            self.worker.log.connect(self.append_log)
            self.worker.done.connect(self.on_done)
            self.worker.failed.connect(self.on_failed)
            self.worker.start()

        def append_log(self, text: str):
            self.log_box.append(text)

        def on_done(self, rows: list):
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            self.run_btn.setEnabled(True)
            ok_count = sum(1 for row in rows if row.get("ok"))
            self.append_log(f"完成：成功 {ok_count}/{len(rows)}")

        def on_failed(self, text: str):
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.run_btn.setEnabled(True)
            self.append_log(f"失败：{text}")

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(_run_gui())
