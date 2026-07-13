# vision/angle_detect.py
from __future__ import annotations

import math
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

import cv2
import numpy as np

CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAM2_REPO_ROOT = PROJECT_ROOT / "sam2-main"
if SAM2_REPO_ROOT.exists() and str(SAM2_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO_ROOT))

from vision.screen_capture import FixedRegionScreenCapture, CaptureArea

logger = logging.getLogger(__name__)

Vec2 = Tuple[float, float]
Edge = Tuple[Vec2, Vec2]


def _mask_to_bbox_xyxy(mask_bool: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("mask 为空，无法计算 bbox。")
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _mask_center(mask_bool: np.ndarray) -> Vec2:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    m = cv2.moments(mask_u8)
    if abs(m["m00"]) < 1e-8:
        x1, y1, x2, y2 = _mask_to_bbox_xyxy(mask_bool)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ap = p - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-8:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(ap, ab) / ab2)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _largest_contour_points(mask_bool: np.ndarray) -> np.ndarray:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("B mask 中没有找到轮廓。")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1.0:
        raise RuntimeError("B mask 轮廓面积过小。")
    return contour.reshape(-1, 2).astype(np.float32)


def _polygon_edges_from_contour(
    contour_points: np.ndarray,
    epsilon_ratio: float,
) -> List[Edge]:
    contour = contour_points.reshape(-1, 1, 2).astype(np.float32)
    peri = float(cv2.arcLength(contour, True))
    epsilon = max(1.0, epsilon_ratio * peri)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)
    if len(approx) < 2:
        raise RuntimeError("B 轮廓近似边数量不足。")
    pts: List[Vec2] = [(float(x), float(y)) for x, y in approx]
    return list(zip(pts, pts[1:] + pts[:1]))


def _edge_mid(edge: Edge) -> Vec2:
    a, b = edge
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _edge_length(edge: Edge) -> float:
    a, b = edge
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _choose_edge(
    edges: List[Edge],
    mode: str,
    reference_point: Optional[Vec2] = None,
) -> Edge:
    mode = mode.lower().strip()

    if mode == "left":
        return min(edges, key=lambda e: _edge_mid(e)[0])
    if mode == "right":
        return max(edges, key=lambda e: _edge_mid(e)[0])
    if mode == "top":
        return min(edges, key=lambda e: _edge_mid(e)[1])
    if mode == "bottom":
        return max(edges, key=lambda e: _edge_mid(e)[1])
    if mode == "longest":
        return max(edges, key=_edge_length)
    if mode == "nearest_to_a":
        if reference_point is None:
            raise ValueError("b_edge_mode='nearest_to_a' 需要 A 的中心点。")
        p = np.array(reference_point, dtype=np.float32)
        return min(
            edges,
            key=lambda e: _point_segment_distance(
                p,
                np.array(e[0], dtype=np.float32),
                np.array(e[1], dtype=np.float32),
            ),
        )

    raise ValueError(
        f"unsupported b_edge_mode={mode}. "
        "Use left/right/top/bottom/longest/nearest_to_a."
    )


def _collect_edge_points(
    contour_points: np.ndarray,
    edge: Edge,
    distance_px: float,
    min_points: int,
) -> np.ndarray:
    a = np.array(edge[0], dtype=np.float32)
    b = np.array(edge[1], dtype=np.float32)
    distances = np.array(
        [_point_segment_distance(p, a, b) for p in contour_points],
        dtype=np.float32,
    )
    selected = contour_points[distances <= float(distance_px)]

    if len(selected) >= min_points:
        return selected

    order = np.argsort(distances)
    take = min(max(min_points, 2), len(contour_points))
    return contour_points[order[:take]]


def _fit_line_total_least_squares(points: np.ndarray) -> Dict[str, Any]:
    if points is None or len(points) < 2:
        raise RuntimeError("用于拟合的边缘点数量不足。")

    pts = points.astype(np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    dx, dy = float(direction[0]), float(direction[1])

    if dx < 0:
        dx, dy = -dx, -dy

    angle_deg = math.degrees(math.atan2(dy, dx))
    if angle_deg < 0:
        angle_deg += 180.0
    if angle_deg >= 180.0:
        angle_deg -= 180.0

    residuals = np.abs(centered @ np.array([-dy, dx], dtype=np.float64))
    return {
        "angle_deg": float(angle_deg),
        "center": (float(centroid[0]), float(centroid[1])),
        "direction": (float(dx), float(dy)),
        "rms_error_px": float(np.sqrt(np.mean(residuals * residuals))),
        "num_points": int(len(points)),
    }


def _draw_result(
    image_rgb: np.ndarray,
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    c_mask: np.ndarray,
    edge_points: np.ndarray,
    fit: Dict[str, Any],
    save_path: Path,
) -> None:
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = img.copy()
    overlay[a_mask] = (0, 255, 255)
    overlay[b_mask] = (0, 128, 255)
    overlay[c_mask] = (255, 255, 0)
    img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

    for x, y in edge_points.astype(int):
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

    cx, cy = fit["center"]
    dx, dy = fit["direction"]
    length = 180.0
    p1 = (int(cx - dx * length), int(cy - dy * length))
    p2 = (int(cx + dx * length), int(cy + dy * length))
    cv2.line(img, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(
        img,
        f"B edge angle={fit['angle_deg']:.2f} deg",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)


class SAM2ABCSegmenter:
    def __init__(
        self,
        sam2_cfg: str,
        sam2_checkpoint: str,
        device: str = "cuda",
    ):
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:
            raise ImportError(
                "SAM2 没有正确安装。请确认可以导入：\n"
                "from sam2.build_sam import build_sam2\n"
                "from sam2.sam2_image_predictor import SAM2ImagePredictor"
            ) from e

        self.torch = torch
        sam2_model = build_sam2(sam2_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def segment_with_points(
        self,
        image_rgb: np.ndarray,
        a_point: Vec2,
        b_point: Vec2,
        c_point: Vec2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def select_a_b_c_points_interactively(
    image_rgb: np.ndarray,
    window_name: str = "SAM2 angle init: click A then B then C",
    scale: float = 0.85,
) -> Tuple[Vec2, Vec2, Vec2]:
    if scale <= 0:
        scale = 1.0

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    show_w = int(w * scale)
    show_h = int(h * scale)
    display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)
    points_display: List[Tuple[int, int]] = []

    def redraw() -> np.ndarray:
        canvas = display.copy()
        cv2.putText(
            canvas,
            "Click A, then B, then C. Press R to reset, ESC to cancel.",
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
            raise RuntimeError("用户取消了 A/B/C 初始化。")
        if key in (ord("r"), ord("R")):
            points_display.clear()

    cv2.destroyWindow(window_name)
    a_disp, b_disp, c_disp = points_display
    return (
        (float(a_disp[0] / scale), float(a_disp[1] / scale)),
        (float(b_disp[0] / scale), float(b_disp[1] / scale)),
        (float(c_disp[0] / scale), float(c_disp[1] / scale)),
    )


class ScreenAngleDetector:
    """
    屏幕捕获 + SAM2 分割 A/B/C + B 指定边缘角度拟合。

    当前角度来源：
        1. 截图；
        2. 第一帧交互点击 A、B、C；
        3. SAM2 分割 A、B、C；
        4. 从 B mask 的指定边缘提取轮廓点；
        5. 对边缘点做正交最小二乘直线拟合，输出角度。
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        capture_area: CaptureArea = (116, 98, 1112, 886),
        output_dir: Union[str, Path] = "outputs/sam2_b_edge_angle",
        save_image: bool = True,
        device: Optional[str] = None,
        num: int = 0,
        cw: int = 0,
        use_second_detect: bool = True,
        sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_checkpoint: Union[str, Path] = SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
        sam2_device: str = "cuda",
        b_edge_mode: str = "longest",
        contour_epsilon_ratio: float = 0.006,
        edge_point_distance_px: float = 4.0,
        min_fit_points: int = 20,
        init_window_scale: float = 0.85,
    ):
        self.model_path = str(model_path) if model_path is not None else None
        self.capture_area = capture_area
        self.output_dir = Path(output_dir) / time.strftime("run_%Y%m%d_%H%M%S")
        self.save_image = bool(save_image)
        self.device = device
        self.num = int(num)
        self.cw = int(cw)
        self.use_second_detect = bool(use_second_detect)

        self.b_edge_mode = str(b_edge_mode)
        self.contour_epsilon_ratio = float(contour_epsilon_ratio)
        self.edge_point_distance_px = float(edge_point_distance_px)
        self.min_fit_points = int(min_fit_points)
        self.init_window_scale = float(init_window_scale)

        self.capture_dir = self.output_dir / "captured_frames"
        self.result_dir = self.output_dir / "angle_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.capturer = FixedRegionScreenCapture(
            capture_area=self.capture_area,
            output_dir=self.capture_dir,
            save_image=self.save_image,
        )

        self.segmenter = SAM2ABCSegmenter(
            sam2_cfg=sam2_cfg,
            sam2_checkpoint=str(sam2_checkpoint),
            device=sam2_device,
        )

        self.initialized_points: Optional[Tuple[Vec2, Vec2, Vec2]] = None
        self.last_capture_result: Optional[Dict[str, Any]] = None
        self.last_image_np: Optional[np.ndarray] = None
        self.last_image_path: Optional[str] = None
        self.last_angle_deg: Optional[float] = None
        self.last_result: Optional[Dict[str, Any]] = None

    def capture_once(self) -> Dict[str, Any]:
        result = self.capturer.capture_and_save()
        self.last_capture_result = result
        if result.get("ok", False):
            self.last_image_np = result.get("image")
            self.last_image_path = result.get("image_path")
        else:
            self.last_image_np = None
            self.last_image_path = None
        return result

    def detect_angle_from_image(self, image_np: np.ndarray) -> Optional[float]:
        result = self.detect_angle_from_image_detail(image_np)
        return result.get("angle_deg") if result.get("ok", False) else None

    def detect_angle_from_image_detail(self, image_np: np.ndarray) -> Dict[str, Any]:
        if self.initialized_points is None:
            self.initialized_points = select_a_b_c_points_interactively(
                image_rgb=image_np,
                scale=self.init_window_scale,
            )

        a_point, b_point, c_point = self.initialized_points
        a_mask, b_mask, c_mask = self.segmenter.segment_with_points(
            image_rgb=image_np,
            a_point=a_point,
            b_point=b_point,
            c_point=c_point,
        )

        a_center = _mask_center(a_mask)
        contour_points = _largest_contour_points(b_mask)
        edges = _polygon_edges_from_contour(contour_points, self.contour_epsilon_ratio)
        target_edge = _choose_edge(
            edges,
            mode=self.b_edge_mode,
            reference_point=a_center,
        )
        edge_points = _collect_edge_points(
            contour_points=contour_points,
            edge=target_edge,
            distance_px=self.edge_point_distance_px,
            min_points=self.min_fit_points,
        )
        fit = _fit_line_total_least_squares(edge_points)
        angle_deg = float(fit["angle_deg"])

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_image_path = None
        if self.save_image:
            result_image_path = self.result_dir / f"b_edge_angle_{timestamp}.png"
            _draw_result(
                image_rgb=image_np,
                a_mask=a_mask,
                b_mask=b_mask,
                c_mask=c_mask,
                edge_points=edge_points,
                fit=fit,
                save_path=result_image_path,
            )

        self.last_angle_deg = angle_deg
        return {
            "ok": True,
            "angle_deg": angle_deg,
            "reason": "ok",
            "b_edge_mode": self.b_edge_mode,
            "b_target_edge": target_edge,
            "fit": fit,
            "num_edge_points": int(len(edge_points)),
            "result_image_path": str(result_image_path) if result_image_path else None,
        }

    def detect_once(self) -> Optional[float]:
        result = self.detect_once_detail()
        if result.get("ok", False):
            return result.get("angle_deg")
        return None

    def detect_once_detail(self) -> Dict[str, Any]:
        capture_result = self.capture_once()
        if not capture_result.get("ok", False):
            result = {
                "ok": False,
                "angle_deg": None,
                "reason": "capture_failed",
                "error": capture_result.get("error"),
                "image_path": None,
                "capture_area": self.capture_area,
                "timestamp": capture_result.get("timestamp"),
                "capture_result": capture_result,
            }
            self.last_result = result
            return result

        image_np = capture_result.get("image")
        if image_np is None:
            result = {
                "ok": False,
                "angle_deg": None,
                "reason": "captured_image_is_none",
                "image_path": capture_result.get("image_path"),
                "capture_area": self.capture_area,
                "timestamp": capture_result.get("timestamp"),
                "capture_result": capture_result,
            }
            self.last_result = result
            return result

        try:
            angle_result = self.detect_angle_from_image_detail(image_np)
            result = {
                **angle_result,
                "image_path": capture_result.get("image_path"),
                "capture_area": self.capture_area,
                "timestamp": capture_result.get("timestamp"),
                "shape": capture_result.get("shape"),
                "capture_result": capture_result,
            }
            self.last_result = result
            return result

        except Exception as e:
            logger.exception("屏幕捕获后的 SAM2 B 边缘角度检测失败")
            result = {
                "ok": False,
                "angle_deg": None,
                "reason": "angle_detect_exception",
                "error": str(e),
                "image_path": capture_result.get("image_path"),
                "capture_area": self.capture_area,
                "timestamp": capture_result.get("timestamp"),
                "shape": capture_result.get("shape"),
                "capture_result": capture_result,
            }
            self.last_result = result
            return result


def get_angle_from_screen_once(
    model_path: Optional[Union[str, Path]] = None,
    capture_area: CaptureArea = (116, 98, 1112, 886),
    output_dir: Union[str, Path] = "outputs/sam2_b_edge_angle",
    save_image: bool = True,
    device: Optional[str] = None,
    num: int = 0,
    cw: int = 0,
    use_second_detect: bool = True,
    b_edge_mode: str = "longest",
) -> Optional[float]:
    detector = ScreenAngleDetector(
        model_path=model_path,
        capture_area=capture_area,
        output_dir=output_dir,
        save_image=save_image,
        device=device,
        num=num,
        cw=cw,
        use_second_detect=use_second_detect,
        b_edge_mode=b_edge_mode,
    )
    return detector.detect_once()


def get_angle_from_screen_once_detail(
    model_path: Optional[Union[str, Path]] = None,
    capture_area: CaptureArea = (116, 98, 1112, 886),
    output_dir: Union[str, Path] = "outputs/sam2_b_edge_angle",
    save_image: bool = True,
    device: Optional[str] = None,
    num: int = 0,
    cw: int = 0,
    use_second_detect: bool = True,
    b_edge_mode: str = "longest",
) -> Dict[str, Any]:
    detector = ScreenAngleDetector(
        model_path=model_path,
        capture_area=capture_area,
        output_dir=output_dir,
        save_image=save_image,
        device=device,
        num=num,
        cw=cw,
        use_second_detect=use_second_detect,
        b_edge_mode=b_edge_mode,
    )
    return detector.detect_once_detail()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    detector = ScreenAngleDetector(
        capture_area=(116, 98, 1112, 886),
        output_dir="outputs/sam2_b_edge_angle",
        save_image=True,
        sam2_device="cuda",
        b_edge_mode="longest",
    )

    result = detector.detect_once_detail()
    print("========== SAM2 B 边缘角度检测结果 ==========")
    print(f"ok          = {result.get('ok')}")
    print(f"angle_deg   = {result.get('angle_deg')}")
    print(f"reason      = {result.get('reason')}")
    print(f"image_path  = {result.get('image_path')}")
    print(f"result_path = {result.get('result_image_path')}")
