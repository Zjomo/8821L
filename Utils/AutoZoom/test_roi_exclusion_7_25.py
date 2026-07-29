"""
测试 ROI 排除功能：
  1. 多边形 ROI 转 mask 正确性；
  2. ROI 排除基本功能；
  3. 空 ROI 不改变原 mask；
  4. 多个 ROI 多边形取并集；
  5. 越界 ROI 顶点不抛异常；
  6. CalibrationState ROI 字段序列化 round-trip；
  7. RuntimeConfig 默认空 ROI；
  8. ABCSegmenter._apply_exclude_roi 正确应用 ROI。

运行方式：
    python test_roi_exclusion_7_25.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))


def _import_module(module_path: Path, module_name: str) -> ModuleType:
    """动态导入本地 Python 文件。"""
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _import_roi_exclusion() -> ModuleType:
    try:
        import logic.roi_exclusion as module
        return module
    except Exception:
        module = _import_module(AUTOZOOM_ROOT / "logic" / "roi_exclusion.py", "logic.roi_exclusion")
        sys.modules.setdefault("logic", ModuleType("logic"))
        sys.modules["logic.roi_exclusion"] = module
        return module


def _import_logic_strict_c() -> ModuleType:
    try:
        import logic.actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix_strict_c as module
        return module
    except Exception:
        _import_roi_exclusion()
        return _import_module(
            AUTOZOOM_ROOT / "logic" / "actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix_strict_c.py",
            "logic.actual_nano_boundary_following_sam2VideoAB_xyStep_detailed_pathfix_strict_c",
        )


def _import_workflow() -> ModuleType:
    try:
        import measurement_workflow_7_25 as module
        return module
    except Exception:
        return _import_module(
            AUTOZOOM_ROOT / "0_measurement_workflow_real_virtual_same_detection_7_25.py",
            "measurement_workflow_7_25",
        )


# ========================================================================
# roi_exclusion.py 纯函数测试
# ========================================================================


def test_polygons_to_mask_convex() -> None:
    """凸多边形 ROI 转 mask 应完全填充内部。"""
    roi = _import_roi_exclusion()
    h, w = 100, 120
    poly = [(10, 10), (50, 10), (50, 50), (10, 50)]
    mask = roi.polygons_to_mask((h, w), [poly])
    assert mask is not None
    assert mask.shape == (h, w)
    assert mask.dtype == bool
    # 内部点应被填充
    assert mask[30, 30]
    # 外部点不应被填充
    assert not mask[5, 5]
    assert not mask[60, 60]
    print("PASS: polygons_to_mask_convex")


def test_polygons_to_mask_empty() -> None:
    """空 ROI 列表应返回 None。"""
    roi = _import_roi_exclusion()
    assert roi.polygons_to_mask((50, 50), []) is None
    assert roi.polygons_to_mask((50, 50), None) is None
    print("PASS: polygons_to_mask_empty")


def test_apply_roi_exclusion_basic() -> None:
    """ROI 排除应正确挖空 mask。"""
    roi = _import_roi_exclusion()
    mask = np.zeros((50, 60), dtype=bool)
    mask[10:40, 10:50] = True
    exclude = np.zeros((50, 60), dtype=bool)
    exclude[20:30, 20:40] = True
    result = roi.apply_roi_exclusion(mask, exclude, inplace=False)
    # 排除区域内应为 False
    assert not result[25, 25]
    # 排除区域外仍应为 True
    assert result[15, 15]
    assert result[35, 45]
    # 原 mask 不应被修改
    assert mask[25, 25]
    print("PASS: apply_roi_exclusion_basic")


def test_apply_roi_exclusion_inplace() -> None:
    """inplace=True 时应直接修改输入 mask。"""
    roi = _import_roi_exclusion()
    mask = np.ones((20, 20), dtype=bool)
    exclude = np.zeros((20, 20), dtype=bool)
    exclude[5:15, 5:15] = True
    result = roi.apply_roi_exclusion(mask, exclude, inplace=True)
    assert result is mask
    assert not mask[10, 10]
    print("PASS: apply_roi_exclusion_inplace")


def test_apply_roi_exclusion_none() -> None:
    """exclude_mask 为 None 时返回原 mask。"""
    roi = _import_roi_exclusion()
    mask = np.ones((10, 10), dtype=bool)
    result = roi.apply_roi_exclusion(mask, None, inplace=False)
    assert np.array_equal(result, mask)
    print("PASS: apply_roi_exclusion_none")


def test_multiple_rois_union() -> None:
    """多个 ROI 多边形应取并集排除。"""
    roi = _import_roi_exclusion()
    poly1 = [(5, 5), (15, 5), (15, 15), (5, 15)]
    poly2 = [(30, 30), (45, 30), (45, 45), (30, 45)]
    mask = roi.polygons_to_mask((50, 50), [poly1, poly2])
    assert mask[10, 10]
    assert mask[35, 35]
    assert not mask[25, 25]
    print("PASS: multiple_rois_union")


def test_roi_out_of_bounds() -> None:
    """越界 ROI 顶点不应抛异常，且有效部分仍被填充。"""
    roi = _import_roi_exclusion()
    poly = [(-10, -10), (100, 5), (5, 100)]
    mask = roi.polygons_to_mask((50, 50), [poly])
    assert mask is not None
    # 三角形在图像内的部分应被填充
    assert mask[10, 10]
    print("PASS: roi_out_of_bounds")


def test_normalize_roi_polygons() -> None:
    """归一化应兼容 list/tuple/dict/numpy 等多种格式。"""
    roi = _import_roi_exclusion()
    raw = [
        [(0, 0), (10, 0), (10, 10)],
        [[20, 20], [30, 20], [30, 30]],
        [{"x": 40, "y": 40}, {"x": 50, "y": 40}, {"x": 50, "y": 50}],
    ]
    polys = roi.normalize_roi_polygons(raw)
    assert len(polys) == 3
    assert all(len(p) == 3 for p in polys)
    assert all(isinstance(p[0], tuple) for p in polys)
    print("PASS: normalize_roi_polygons")


# ========================================================================
# CalibrationState / RuntimeConfig ROI 字段测试
# ========================================================================


def test_runtime_config_default_roi() -> None:
    """RuntimeConfig 默认 exclude_roi_polygons 为空列表。"""
    logic = _import_logic_strict_c()
    cfg = logic.RuntimeConfig()
    assert hasattr(cfg, "exclude_roi_polygons")
    assert cfg.exclude_roi_polygons == []
    print("PASS: runtime_config_default_roi")


def test_calibration_state_roi_round_trip() -> None:
    """CalibrationState 的 ROI 字段应支持 dict round-trip。"""
    wf = _import_workflow()
    state = wf.CalibrationState()
    polys = [
        [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0]],
        [[100.0, 100.0], [150.0, 100.0], [150.0, 150.0], [100.0, 150.0]],
    ]
    state.exclude_roi_polygons = polys
    data = state.to_dict()
    assert "exclude_roi_polygons" in data
    assert data["exclude_roi_polygons"] == polys

    restored = wf.CalibrationState.from_dict(data)
    assert restored.exclude_roi_polygons == polys
    print("PASS: calibration_state_roi_round_trip")


def test_calibration_state_roi_normalization() -> None:
    """from_dict 应能归一化多种 ROI 描述格式。"""
    wf = _import_workflow()
    raw = {
        "exclude_roi_polygons": [
            [(0, 0), (10, 0), (10, 10)],
            [{"x": 20, "y": 20}, {"x": 30, "y": 20}, {"x": 30, "y": 30}],
        ]
    }
    state = wf.CalibrationState.from_dict(raw)
    assert len(state.exclude_roi_polygons) == 2
    assert state.exclude_roi_polygons[0] == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    assert state.exclude_roi_polygons[1] == [[20.0, 20.0], [30.0, 20.0], [30.0, 30.0]]
    print("PASS: calibration_state_roi_normalization")


# ========================================================================
# SAM2ABCSegmenter ROI 集成测试（mock predictor，无需 GPU）
# ========================================================================


def test_segmenter_apply_exclude_roi() -> None:
    """SAM2ABCSegmenter._apply_exclude_roi 应正确挖空 mask。"""
    logic = _import_logic_strict_c()
    # 构造一个 segmenter 实例，但把 predictor 替换为 mock，避免加载 SAM2 权重。
    with patch.object(logic.SAM2ABCSegmenter, "__init__", lambda self, **kwargs: None):
        segmenter = logic.SAM2ABCSegmenter()
        segmenter.exclude_roi_mask = None

        mask = np.ones((40, 40), dtype=bool)
        mask[5:35, 5:35] = True
        # 无 ROI 时不改变 mask
        result = segmenter._apply_exclude_roi(mask.copy(), inplace=False)
        assert np.array_equal(result, mask)

        # 设置 ROI 后应挖空
        exclude = np.zeros((40, 40), dtype=bool)
        exclude[15:25, 15:25] = True
        segmenter.exclude_roi_mask = exclude
        result = segmenter._apply_exclude_roi(mask.copy(), inplace=False)
        assert not result[20, 20]
        assert result[10, 10]
    print("PASS: segmenter_apply_exclude_roi")


def test_segmenter_set_exclude_roi_mask() -> None:
    """SAM2ABCSegmenter.set_exclude_roi_mask 应能正确从多边形生成 mask。"""
    logic = _import_logic_strict_c()
    with patch.object(logic.SAM2ABCSegmenter, "__init__", lambda self, **kwargs: None):
        segmenter = logic.SAM2ABCSegmenter()
        segmenter.exclude_roi_mask = None
        poly = [[(5, 5), (25, 5), (25, 25), (5, 25)]]
        segmenter.set_exclude_roi_mask(polygons=poly, image_shape_hw=(40, 40))
        assert segmenter.exclude_roi_mask is not None
        assert segmenter.exclude_roi_mask.shape == (40, 40)
        assert segmenter.exclude_roi_mask[15, 15]
        assert not segmenter.exclude_roi_mask[35, 35]
    print("PASS: segmenter_set_exclude_roi_mask")


# ========================================================================
# 整体 runner
# ========================================================================


def run_all_tests() -> None:
    test_polygons_to_mask_convex()
    test_polygons_to_mask_empty()
    test_apply_roi_exclusion_basic()
    test_apply_roi_exclusion_inplace()
    test_apply_roi_exclusion_none()
    test_multiple_rois_union()
    test_roi_out_of_bounds()
    test_normalize_roi_polygons()
    test_runtime_config_default_roi()
    test_calibration_state_roi_round_trip()
    test_calibration_state_roi_normalization()
    test_segmenter_apply_exclude_roi()
    test_segmenter_set_exclude_roi_mask()
    print("\nAll ROI exclusion tests passed.")


if __name__ == "__main__":
    run_all_tests()
