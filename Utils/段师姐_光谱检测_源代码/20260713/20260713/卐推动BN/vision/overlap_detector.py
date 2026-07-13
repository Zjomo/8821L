# -*- coding: utf-8 -*-
"""
SAM2 + 颜色检测计算重合区域面积。

流程：
1. 固定区域屏幕截图。
2. 第一帧手动点击 A、B、C，SAM2 分割 A/B/C。
3. 根据 B 的 SAM2 mask 得到 B ROI。
4. 第一次在图上点击重合区域颜色样本，保存 HSV 颜色配置。
5. 后续截图继续用 SAM2 分割 B，只在 B ROI 且受 B mask 约束的区域内做颜色检测。
6. 输出重合颜色区域面积 px^2 / um^2，并保存可视化结果。
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAM2_REPO_ROOT = PROJECT_ROOT / "sam2-main"
if SAM2_REPO_ROOT.exists() and str(SAM2_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO_ROOT))

from vision.screen_capture import FixedRegionScreenCapture, CaptureArea


Vec2 = Tuple[float, float]
Rect = Tuple[int, int, int, int]  # x1, y1, x2, y2


# ============================================================
# 1. 基础图像工具
# ============================================================

def imread_unicode(path: Union[str, Path]) -> np.ndarray:
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def imwrite_unicode(path: Union[str, Path], image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    buf.tofile(str(path))


def list_images(image_dir: Union[str, Path]) -> List[str]:
    image_dir = Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(str(p) for p in image_dir.iterdir() if p.suffix.lower() in exts)


def rect_clip(rect: Rect, image_shape: Tuple[int, int, int]) -> Rect:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = rect
    return (
        max(0, min(w - 1, int(x1))),
        max(0, min(h - 1, int(y1))),
        max(0, min(w - 1, int(x2))),
        max(0, min(h - 1, int(y2))),
    )


def mask_to_expanded_roi(mask_bool: np.ndarray, pad: int = 10) -> Optional[Rect]:
    ys, xs = np.where(mask_bool > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = mask_bool.shape[:2]
    x1 = max(0, int(xs.min()) - int(pad))
    y1 = max(0, int(ys.min()) - int(pad))
    x2 = min(w - 1, int(xs.max()) + int(pad))
    y2 = min(h - 1, int(ys.max()) + int(pad))
    return x1, y1, x2, y2


def mask_center(mask_bool: np.ndarray) -> Vec2:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    m = cv2.moments(mask_u8)
    if abs(m["m00"]) < 1e-8:
        ys, xs = np.where(mask_bool > 0)
        if len(xs) == 0:
            return 0.0, 0.0
        return float(xs.mean()), float(ys.mean())
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def clean_mask(mask_bin: np.ndarray, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    out = mask_bin.copy()

    if open_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

    if close_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    return out


# ============================================================
# 2. SAM2 分割 A/B/C
# ============================================================

class SAM2ABCSegmenter:
    def __init__(
        self,
        sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_checkpoint: Union[str, Path] = SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
        device: str = "cuda",
    ):
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:
            raise ImportError(
                "SAM2 没有正确安装。请确认可以导入:\n"
                "from sam2.build_sam import build_sam2\n"
                "from sam2.sam2_image_predictor import SAM2ImagePredictor"
            ) from e

        self.torch = torch
        sam2_model = build_sam2(
            sam2_cfg,
            str(sam2_checkpoint),
            device=device,
        )
        self.predictor = SAM2ImagePredictor(sam2_model)

    def segment_abc(
        self,
        image_bgr: np.ndarray,
        a_point: Vec2,
        b_point: Vec2,
        c_point: Vec2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        a_mask = self._predict_mask(a_point, [b_point, c_point])
        b_mask = self._predict_mask(b_point, [a_point, c_point])
        c_mask = self._predict_mask(c_point, [a_point, b_point])
        return a_mask, b_mask, c_mask

    def _predict_mask(
        self,
        positive_point: Vec2,
        negative_points: Optional[List[Vec2]] = None,
    ) -> np.ndarray:
        points = [[positive_point[0], positive_point[1]]]
        labels = [1]

        for p in negative_points or []:
            points.append([p[0], p[1]])
            labels.append(0)

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        with self.torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=True,
            )

        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM2 point prompt 没有返回 mask。")

        best_idx = int(np.argmax(np.asarray(scores).reshape(-1)))
        return masks[best_idx].astype(bool)


def select_abc_points_interactively(
    image_bgr: np.ndarray,
    window_name: str = "SAM2 init: click A then B then C",
    scale: float = 0.85,
) -> Tuple[Vec2, Vec2, Vec2]:
    if scale <= 0:
        scale = 1.0

    h, w = image_bgr.shape[:2]
    show_w = int(w * scale)
    show_h = int(h * scale)
    display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)
    points_display: List[Tuple[int, int]] = []

    def redraw() -> np.ndarray:
        canvas = display.copy()
        cv2.putText(
            canvas,
            "Click A, then B, then C. R=reset, ESC=cancel.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        labels = ("A", "B", "C")
        colors = ((0, 255, 255), (0, 128, 255), (255, 255, 0))
        for i, p in enumerate(points_display):
            cv2.circle(canvas, p, 6, colors[i], -1)
            cv2.putText(
                canvas,
                labels[i],
                (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                colors[i],
                2,
                cv2.LINE_AA,
            )
        return canvas

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_display) < 3:
            points_display.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, show_w, show_h)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, redraw())
        key = cv2.waitKey(30) & 0xFF

        if len(points_display) >= 3:
            break
        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("用户取消了 A/B/C 点击初始化。")
        if key in (ord("r"), ord("R")):
            points_display.clear()

    cv2.destroyWindow(window_name)
    a_disp, b_disp, c_disp = points_display
    return (
        (float(a_disp[0] / scale), float(a_disp[1] / scale)),
        (float(b_disp[0] / scale), float(b_disp[1] / scale)),
        (float(c_disp[0] / scale), float(c_disp[1] / scale)),
    )


# ============================================================
# 3. 颜色样本配置
# ============================================================

def select_color_sample_point(
    image_bgr: np.ndarray,
    b_mask: Optional[np.ndarray] = None,
    window_name: str = "Select overlap color sample",
    scale: float = 0.85,
) -> Tuple[int, int]:
    if scale <= 0:
        scale = 1.0

    h, w = image_bgr.shape[:2]
    show_w = int(w * scale)
    show_h = int(h * scale)
    display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)

    if b_mask is not None:
        overlay = display.copy()
        b_show = cv2.resize(
            b_mask.astype(np.uint8),
            (show_w, show_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        overlay[b_show] = (0, 128, 255)
        display = cv2.addWeighted(overlay, 0.25, display, 0.75, 0)

    clicked: Dict[str, Optional[Tuple[int, int]]] = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (int(x), int(y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, show_w, show_h)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        show = display.copy()
        cv2.putText(
            show,
            "Click overlap color sample point. ENTER/SPACE=confirm, ESC=cancel.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if clicked["pt"] is not None:
            cv2.circle(show, clicked["pt"], 6, (0, 255, 255), 2)

        cv2.imshow(window_name, show)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 32) and clicked["pt"] is not None:
            break
        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("用户取消了颜色样本点选择。")

    cv2.destroyWindow(window_name)
    sx, sy = clicked["pt"]
    return int(round(sx / scale)), int(round(sy / scale))


def compute_sample_hsv_stats_from_point(
    image_bgr: np.ndarray,
    sample_point: Tuple[int, int],
    patch_radius: int = 3,
) -> Dict[str, Any]:
    h, w = image_bgr.shape[:2]
    cx, cy = sample_point

    x1 = max(0, int(cx) - patch_radius)
    y1 = max(0, int(cy) - patch_radius)
    x2 = min(w - 1, int(cx) + patch_radius)
    y2 = min(h - 1, int(cy) + patch_radius)

    patch = image_bgr[y1:y2 + 1, x1:x2 + 1]
    if patch.size == 0:
        raise RuntimeError("样本点邻域为空。")

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return {
        "sample_point": [int(cx), int(cy)],
        "sample_patch_rect": [int(x1), int(y1), int(x2), int(y2)],
        "h_mean": float(np.mean(hsv[:, :, 0])),
        "s_mean": float(np.mean(hsv[:, :, 1])),
        "v_mean": float(np.mean(hsv[:, :, 2])),
        "patch_radius": int(patch_radius),
    }


def save_sample_config(
    save_path: Union[str, Path],
    sample_stats: Dict[str, Any],
    h_tol: int = 15,
    s_tol: int = 60,
    v_tol: int = 60,
) -> None:
    cfg = dict(sample_stats)
    cfg["h_tol"] = int(h_tol)
    cfg["s_tol"] = int(s_tol)
    cfg["v_tol"] = int(v_tol)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_sample_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = ["h_mean", "s_mean", "v_mean", "h_tol", "s_tol", "v_tol"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"颜色配置缺少字段: {missing}")

    return cfg


# ============================================================
# 4. B ROI 内颜色检测
# ============================================================

def hsv_range_mask(
    hsv_img: np.ndarray,
    h_mean: float,
    s_mean: float,
    v_mean: float,
    h_tol: int,
    s_tol: int,
    v_tol: int,
) -> np.ndarray:
    lower_h = int(round(h_mean - h_tol))
    upper_h = int(round(h_mean + h_tol))
    lower_s = max(0, int(round(s_mean - s_tol)))
    upper_s = min(255, int(round(s_mean + s_tol)))
    lower_v = max(0, int(round(v_mean - v_tol)))
    upper_v = min(255, int(round(v_mean + v_tol)))

    if lower_h < 0:
        mask1 = cv2.inRange(
            hsv_img,
            np.array([0, lower_s, lower_v], dtype=np.uint8),
            np.array([upper_h, upper_s, upper_v], dtype=np.uint8),
        )
        mask2 = cv2.inRange(
            hsv_img,
            np.array([180 + lower_h, lower_s, lower_v], dtype=np.uint8),
            np.array([179, upper_s, upper_v], dtype=np.uint8),
        )
        return cv2.bitwise_or(mask1, mask2)

    if upper_h > 179:
        mask1 = cv2.inRange(
            hsv_img,
            np.array([lower_h, lower_s, lower_v], dtype=np.uint8),
            np.array([179, upper_s, upper_v], dtype=np.uint8),
        )
        mask2 = cv2.inRange(
            hsv_img,
            np.array([0, lower_s, lower_v], dtype=np.uint8),
            np.array([upper_h - 180, upper_s, upper_v], dtype=np.uint8),
        )
        return cv2.bitwise_or(mask1, mask2)

    return cv2.inRange(
        hsv_img,
        np.array([lower_h, lower_s, lower_v], dtype=np.uint8),
        np.array([upper_h, upper_s, upper_v], dtype=np.uint8),
    )


def choose_best_component(
    mask_candidate: np.ndarray,
    sample_center_in_roi: Optional[Tuple[float, float]] = None,
    min_area: int = 30,
    prefer_sample_center: bool = True,
) -> Tuple[np.ndarray, int, List[Dict[str, Any]]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_candidate,
        connectivity=8,
    )

    best_label = None
    best_score = None
    candidates: List[Dict[str, Any]] = []

    sx, sy = sample_center_in_roi if sample_center_in_roi is not None else (0.0, 0.0)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        cx, cy = centroids[label]
        dist = math.hypot(float(cx) - sx, float(cy) - sy)

        if prefer_sample_center and sample_center_in_roi is not None:
            score = dist - 0.02 * area
        else:
            score = -area

        item = {
            "label": int(label),
            "area": int(area),
            "cx": float(cx),
            "cy": float(cy),
            "dist_to_sample": float(dist),
            "score": float(score),
        }
        candidates.append(item)

        if best_score is None or score < best_score:
            best_score = score
            best_label = label

    selected = np.zeros_like(mask_candidate, dtype=np.uint8)
    if best_label is None:
        return selected, 0, candidates

    selected[labels == best_label] = 255
    return selected, int(np.sum(selected > 0)), candidates


def detect_overlap_area_in_b_roi(
    image_bgr: np.ndarray,
    b_mask: np.ndarray,
    sample_cfg: Dict[str, Any],
    pad: int = 10,
    min_area: int = 30,
    open_k: int = 3,
    close_k: int = 5,
    um_per_px: Optional[float] = None,
) -> Dict[str, Any]:
    roi_rect = mask_to_expanded_roi(b_mask, pad=pad)
    if roi_rect is None:
        return {"success": False, "reason": "B mask 为空，无法生成 ROI。"}

    x1, y1, x2, y2 = rect_clip(roi_rect, image_bgr.shape)
    roi_bgr = image_bgr[y1:y2 + 1, x1:x2 + 1].copy()
    roi_b_mask = (b_mask[y1:y2 + 1, x1:x2 + 1].astype(np.uint8) * 255)

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask_color = hsv_range_mask(
        roi_hsv,
        h_mean=float(sample_cfg["h_mean"]),
        s_mean=float(sample_cfg["s_mean"]),
        v_mean=float(sample_cfg["v_mean"]),
        h_tol=int(sample_cfg["h_tol"]),
        s_tol=int(sample_cfg["s_tol"]),
        v_tol=int(sample_cfg["v_tol"]),
    )

    mask_candidate = cv2.bitwise_and(mask_color, mask_color, mask=roi_b_mask)
    mask_candidate = clean_mask(mask_candidate, open_k=open_k, close_k=close_k)

    sample_center_in_roi = None
    if "sample_point" in sample_cfg:
        spx, spy = sample_cfg["sample_point"]
        sample_center_in_roi = (float(spx) - x1, float(spy) - y1)

    selected_mask_roi, area_px, candidates_info = choose_best_component(
        mask_candidate=mask_candidate,
        sample_center_in_roi=sample_center_in_roi,
        min_area=min_area,
        prefer_sample_center=True,
    )

    area_um2 = None
    if um_per_px is not None:
        area_um2 = float(area_px) * (float(um_per_px) ** 2)

    contours, _ = cv2.findContours(
        selected_mask_roi,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour_roi = max(contours, key=cv2.contourArea) if contours else None

    return {
        "success": True,
        "roi_rect": (x1, y1, x2, y2),
        "roi_b_mask": roi_b_mask,
        "mask_color": mask_color,
        "mask_candidate": mask_candidate,
        "selected_mask_roi": selected_mask_roi,
        "selected_contour_roi": contour_roi,
        "area_px": int(area_px),
        "area_um2": area_um2,
        "candidates_info": candidates_info,
    }


# ============================================================
# 5. 可视化
# ============================================================

def draw_result_visualization(
    image_bgr: np.ndarray,
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    c_mask: np.ndarray,
    result: Dict[str, Any],
    sample_cfg: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    vis = image_bgr.copy()

    overlay = vis.copy()
    overlay[a_mask] = (0, 255, 255)
    overlay[b_mask] = (0, 128, 255)
    overlay[c_mask] = (255, 255, 0)
    vis = cv2.addWeighted(overlay, 0.22, vis, 0.78, 0)

    if not result.get("success", False):
        cv2.putText(
            vis,
            f"Detection failed: {result.get('reason', '')}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    x1, y1, x2, y2 = result["roi_rect"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    roi_view = vis[y1:y2 + 1, x1:x2 + 1]

    candidate_color = np.zeros_like(roi_view)
    candidate_color[:, :, 1] = result["mask_candidate"]
    roi_view[:] = cv2.addWeighted(roi_view, 1.0, candidate_color, 0.25, 0)

    final_color = np.zeros_like(roi_view)
    final_color[:, :, 2] = result["selected_mask_roi"]
    roi_view[:] = cv2.addWeighted(roi_view, 1.0, final_color, 0.65, 0)

    contour_roi = result.get("selected_contour_roi")
    if contour_roi is not None:
        cv2.drawContours(roi_view, [contour_roi], -1, (255, 255, 255), 2)

    if sample_cfg is not None and "sample_point" in sample_cfg:
        spx, spy = sample_cfg["sample_point"]
        cv2.circle(vis, (int(spx), int(spy)), 5, (0, 255, 255), 2)

    text = f"Overlap area = {result.get('area_px', 0)} px^2"
    cv2.putText(
        vis,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if result.get("area_um2") is not None:
        text2 = f"Overlap area = {result['area_um2']:.6f} um^2"
        cv2.putText(
            vis,
            text2,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis


# ============================================================
# 6. 检测器封装
# ============================================================

class SAM2ColorOverlapDetector:
    def __init__(
        self,
        sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_checkpoint: Union[str, Path] = SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
        sam2_device: str = "cuda",
        sample_config_path: Union[str, Path] = "sample_config.json",
        output_dir: Union[str, Path] = "overlap_results_sam2",
        capture_area: CaptureArea = (116, 98, 1112, 886),
        save_capture_image: bool = True,
        h_tol: int = 15,
        s_tol: int = 60,
        v_tol: int = 60,
        patch_radius: int = 3,
        roi_pad: int = 10,
        min_area: int = 30,
        open_k: int = 3,
        close_k: int = 5,
        um_per_px: Optional[float] = None,
        click_scale: float = 0.85,
    ):
        self.segmenter = SAM2ABCSegmenter(
            sam2_cfg=sam2_cfg,
            sam2_checkpoint=sam2_checkpoint,
            device=sam2_device,
        )

        self.sample_config_path = Path(sample_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capture_dir = self.output_dir / "captured_frames"
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        self.capture_area = capture_area
        self.capturer = FixedRegionScreenCapture(
            capture_area=self.capture_area,
            output_dir=self.capture_dir,
            save_image=save_capture_image,
        )

        self.h_tol = int(h_tol)
        self.s_tol = int(s_tol)
        self.v_tol = int(v_tol)
        self.patch_radius = int(patch_radius)
        self.roi_pad = int(roi_pad)
        self.min_area = int(min_area)
        self.open_k = int(open_k)
        self.close_k = int(close_k)
        self.um_per_px = um_per_px
        self.click_scale = float(click_scale)

        self.abc_points: Optional[Tuple[Vec2, Vec2, Vec2]] = None
        self.sample_cfg: Optional[Dict[str, Any]] = None

        if self.sample_config_path.exists():
            self.sample_cfg = load_sample_config(self.sample_config_path)

    def capture_once(self) -> Dict[str, Any]:
        return self.capturer.capture_and_save()

    def initialize_points(self, image_bgr: np.ndarray) -> Tuple[Vec2, Vec2, Vec2]:
        self.abc_points = select_abc_points_interactively(
            image_bgr=image_bgr,
            scale=self.click_scale,
        )
        return self.abc_points

    def segment_current_image(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.abc_points is None:
            self.initialize_points(image_bgr)

        a_point, b_point, c_point = self.abc_points
        return self.segmenter.segment_abc(
            image_bgr=image_bgr,
            a_point=a_point,
            b_point=b_point,
            c_point=c_point,
        )

    def initialize_sample_if_needed(
        self,
        image_bgr: np.ndarray,
        b_mask: np.ndarray,
        force_reselect: bool = False,
    ) -> Dict[str, Any]:
        if self.sample_cfg is not None and not force_reselect:
            return self.sample_cfg

        sample_point = select_color_sample_point(
            image_bgr=image_bgr,
            b_mask=b_mask,
            scale=self.click_scale,
        )
        sample_stats = compute_sample_hsv_stats_from_point(
            image_bgr=image_bgr,
            sample_point=sample_point,
            patch_radius=self.patch_radius,
        )
        save_sample_config(
            save_path=self.sample_config_path,
            sample_stats=sample_stats,
            h_tol=self.h_tol,
            s_tol=self.s_tol,
            v_tol=self.v_tol,
        )
        self.sample_cfg = load_sample_config(self.sample_config_path)
        return self.sample_cfg

    def detect_image(
        self,
        image_bgr: np.ndarray,
        force_reselect_sample: bool = False,
    ) -> Dict[str, Any]:
        a_mask, b_mask, c_mask = self.segment_current_image(image_bgr)
        sample_cfg = self.initialize_sample_if_needed(
            image_bgr=image_bgr,
            b_mask=b_mask,
            force_reselect=force_reselect_sample,
        )

        result = detect_overlap_area_in_b_roi(
            image_bgr=image_bgr,
            b_mask=b_mask,
            sample_cfg=sample_cfg,
            pad=self.roi_pad,
            min_area=self.min_area,
            open_k=self.open_k,
            close_k=self.close_k,
            um_per_px=self.um_per_px,
        )

        result.update({
            "a_mask": a_mask,
            "b_mask": b_mask,
            "c_mask": c_mask,
            "sample_cfg": sample_cfg,
            "a_center": mask_center(a_mask),
            "b_center": mask_center(b_mask),
            "c_center": mask_center(c_mask),
        })
        return result

    def process_image_path(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        force_reselect_sample: bool = False,
    ) -> Dict[str, Any]:
        image_path = Path(image_path)
        image_bgr = imread_unicode(image_path)
        result = self.detect_image(
            image_bgr=image_bgr,
            force_reselect_sample=force_reselect_sample,
        )

        out_dir = Path(output_dir) if output_dir is not None else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        vis = draw_result_visualization(
            image_bgr=image_bgr,
            a_mask=result["a_mask"],
            b_mask=result["b_mask"],
            c_mask=result["c_mask"],
            result=result,
            sample_cfg=result.get("sample_cfg"),
        )
        vis_path = out_dir / f"{image_path.stem}_overlap_vis.png"
        imwrite_unicode(vis_path, vis)

        result["image_path"] = str(image_path)
        result["vis_path"] = str(vis_path)
        return result

    def detect_once_from_screen(
        self,
        force_reselect_sample: bool = False,
    ) -> Dict[str, Any]:
        capture_result = self.capture_once()
        if not capture_result.get("ok", False):
            return {
                "success": False,
                "reason": "capture_failed",
                "error": capture_result.get("error"),
                "capture_result": capture_result,
            }

        image_rgb = capture_result.get("image")
        if image_rgb is None:
            return {
                "success": False,
                "reason": "captured_image_is_none",
                "capture_result": capture_result,
            }

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        result = self.detect_image(
            image_bgr=image_bgr,
            force_reselect_sample=force_reselect_sample,
        )

        timestamp = capture_result.get("timestamp") or time.strftime("%Y%m%d_%H%M%S")
        vis = draw_result_visualization(
            image_bgr=image_bgr,
            a_mask=result["a_mask"],
            b_mask=result["b_mask"],
            c_mask=result["c_mask"],
            result=result,
            sample_cfg=result.get("sample_cfg"),
        )

        vis_path = self.output_dir / f"overlap_screen_{timestamp}.png"
        imwrite_unicode(vis_path, vis)

        result["capture_result"] = capture_result
        result["image_path"] = capture_result.get("image_path")
        result["vis_path"] = str(vis_path)
        result["capture_area"] = self.capture_area
        result["timestamp"] = timestamp
        return result


# ============================================================
# 7. 屏幕截图检测
# ============================================================

def write_screen_results_csv(
    rows: List[Dict[str, Any]],
    csv_path: Union[str, Path],
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "success",
                "reason",
                "area_px",
                "area_um2",
                "roi_rect",
                "image_path",
                "vis_path",
                "a_center",
                "b_center",
                "c_center",
                "timestamp",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_screen_loop(
    num_frames: int = 1,
    output_dir: Union[str, Path] = "overlap_results_sam2",
    sample_config_path: Union[str, Path] = "sample_config.json",
    capture_area: CaptureArea = (116, 98, 1112, 886),
    sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
    sam2_checkpoint: Union[str, Path] = SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
    sam2_device: str = "cuda",
    h_tol: int = 15,
    s_tol: int = 60,
    v_tol: int = 60,
    patch_radius: int = 3,
    roi_pad: int = 10,
    min_area: int = 30,
    open_k: int = 3,
    close_k: int = 5,
    um_per_px: Optional[float] = None,
    loop_interval_s: float = 0.5,
    save_capture_image: bool = True,
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    run_dir = output_dir / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    detector = SAM2ColorOverlapDetector(
        sam2_cfg=sam2_cfg,
        sam2_checkpoint=sam2_checkpoint,
        sam2_device=sam2_device,
        sample_config_path=sample_config_path,
        output_dir=run_dir,
        capture_area=capture_area,
        save_capture_image=save_capture_image,
        h_tol=h_tol,
        s_tol=s_tol,
        v_tol=v_tol,
        patch_radius=patch_radius,
        roi_pad=roi_pad,
        min_area=min_area,
        open_k=open_k,
        close_k=close_k,
        um_per_px=um_per_px,
    )

    rows: List[Dict[str, Any]] = []
    for idx in range(int(num_frames)):
        print(f"[{idx + 1}/{num_frames}] capture and detect overlap area")
        result = detector.detect_once_from_screen(force_reselect_sample=False)

        row = {
            "index": idx + 1,
            "success": result.get("success"),
            "reason": result.get("reason", ""),
            "area_px": result.get("area_px", 0),
            "area_um2": result.get("area_um2"),
            "roi_rect": result.get("roi_rect"),
            "image_path": result.get("image_path"),
            "vis_path": result.get("vis_path"),
            "a_center": result.get("a_center"),
            "b_center": result.get("b_center"),
            "c_center": result.get("c_center"),
            "timestamp": result.get("timestamp"),
        }
        rows.append(row)

        write_screen_results_csv(rows, run_dir / "overlap_area_results.csv")

        if idx < int(num_frames) - 1:
            time.sleep(float(loop_interval_s))

    print(f"[DONE] results saved to: {run_dir}")
    return rows


# ============================================================
# 8. 示例入口
# ============================================================

def main() -> None:
    output_dir = "overlap_results_sam2"
    sample_config_path = "sample_config.json"

    run_screen_loop(
        num_frames=1,
        output_dir=output_dir,
        sample_config_path=sample_config_path,
        capture_area=(116, 98, 1112, 886),
        sam2_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_checkpoint=SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
        sam2_device="cuda",
        h_tol=15,
        s_tol=60,
        v_tol=60,
        patch_radius=3,
        roi_pad=10,
        min_area=30,
        open_k=3,
        close_k=5,
        um_per_px=None,
        loop_interval_s=0.5,
        save_capture_image=True,
    )


if __name__ == "__main__":
    main()
