from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent / "0_measurement_workflow_real_virtual_same_detection_8_3.py"


def _load_module():
    module_name = "measurement_workflow_8_3_for_cache_test"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(self, data):
        self._data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._data)


class _FakeOBB:
    def __init__(self):
        self.xyxyxyxy = _FakeTensor([[[10, 10], [40, 10], [40, 30], [10, 30]]])
        self.conf = _FakeTensor([0.97])
        self.cls = _FakeTensor([1])

    def __len__(self):
        return 1


class _FakeResult:
    def __init__(self):
        self.obb = _FakeOBB()
        self.names = {1: "box"}

    def plot(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)


class _FakeModel:
    def predict(self, **kwargs):
        return [_FakeResult()]


def test_detect_step7_passes_save_overlay_flag():
    module = _load_module()
    cfg = module.MeasurementConfig()
    wf = module.MeasurementWorkflow(cfg)
    captured = {}

    def fake_detect_angle_once(*, label, allow_fail=True, save_overlay=True):
        captured["label"] = label
        captured["allow_fail"] = allow_fail
        captured["save_overlay"] = save_overlay
        return {"ok": True, "angle_deg": 12.3, "angle_source": "yolo_obb_long_edge", "reason": "ok"}

    wf.detect_angle_once = fake_detect_angle_once
    result = wf.detect_step7_yolo_obb_angle_once(label="step7_test", allow_fail=False, save_overlay=False)

    assert result["ok"] is True
    assert captured["label"] == "step7_test"
    assert captured["allow_fail"] is False
    assert captured["save_overlay"] is False


def test_yolo_obb_detector_respects_save_flags_without_writing_files():
    module = _load_module()
    cfg = module.MeasurementConfig()
    cfg.capture_area = [0, 0, 64, 64]
    cfg.save_root = tempfile.mkdtemp(prefix="yolo_cache_test_")
    wf = module.MeasurementWorkflow(cfg)
    wf.run_session_dir = Path(cfg.save_root) / "run"
    wf.run_session_dir.mkdir(parents=True, exist_ok=True)
    wf.angle_module = _FakeModel()
    wf._capture_yolo_obb_frame_bgr = lambda: (np.zeros((64, 64, 3), dtype=np.uint8), (0, 0, 64, 64))

    result = wf._run_angle_detector_once_raw(label="cache_test", save_overlay=False, save_raw=False, save_meta=False)

    out_dir = wf._yolo_obb_output_dir()
    assert result["ok"] is True
    assert result["raw_image_path"] == ""
    assert result["overlay_image_path"] == ""
    assert result["meta_path"] == ""
    assert not list(out_dir.glob("*_raw.png"))
    assert not list(out_dir.glob("*_YOLO_OBB_overlay.png"))
    assert not list(out_dir.glob("*_YOLO_OBB_meta.json"))


def test_yolo_obb_detector_exports_raw_image_and_angle_csv_for_overlay_detection():
    module = _load_module()
    cfg = module.MeasurementConfig()
    cfg.capture_area = [0, 0, 64, 64]
    cfg.save_root = tempfile.mkdtemp(prefix="yolo_angle_export_test_")
    wf = module.MeasurementWorkflow(cfg)
    wf.run_session_dir = Path(cfg.save_root) / "run"
    wf.run_session_dir.mkdir(parents=True, exist_ok=True)
    wf.angle_module = _FakeModel()
    wf._capture_yolo_obb_frame_bgr = lambda: (np.zeros((64, 64, 3), dtype=np.uint8), (0, 0, 64, 64))

    result = wf._run_angle_detector_once_raw(label="export_test", save_overlay=True, save_meta=False)

    assert result["ok"] is True
    assert result["raw_image_path"]
    assert result["overlay_image_path"]
    assert result["angle_csv_path"]
    assert Path(result["raw_image_path"]).exists()
    assert Path(result["overlay_image_path"]).exists()
    assert Path(result["angle_csv_path"]).exists()


def test_remove_yolo_obb_redundant_cache_files_keeps_overlay_and_csv():
    module = _load_module()
    with tempfile.TemporaryDirectory(prefix="yolo_cache_cleanup_") as tmpdir:
        cache_dir = Path(tmpdir)
        (cache_dir / "a_raw.png").write_bytes(b"raw")
        (cache_dir / "step7_realtime_final_overlay_raw.png").write_bytes(b"final_raw")
        (cache_dir / "a_YOLO_OBB_meta.json").write_text("{}", encoding="utf-8")
        (cache_dir / "a_YOLO_OBB_overlay.png").write_bytes(b"overlay")
        (cache_dir / "step7_realtime_angle_records.csv").write_text("csv", encoding="utf-8")

        removed = module.MeasurementWorkflow._remove_yolo_obb_redundant_cache_files(cache_dir)

        assert removed == {"raw": 1, "meta": 1}
        assert not (cache_dir / "a_raw.png").exists()
        assert (cache_dir / "step7_realtime_final_overlay_raw.png").exists()
        assert not (cache_dir / "a_YOLO_OBB_meta.json").exists()
        assert (cache_dir / "a_YOLO_OBB_overlay.png").exists()
        assert (cache_dir / "step7_realtime_angle_records.csv").exists()
