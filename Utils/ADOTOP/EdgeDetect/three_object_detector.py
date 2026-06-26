import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import ImageGrab  # type: ignore
except Exception:  # pragma: no cover
    ImageGrab = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WINDOW_ORIG = "原图"
WINDOW_SEG = "分割图"
WINDOW_HELP = "操作说明"


@dataclass
class Detection:
    label: str
    score: float
    contour: List[List[int]]
    bbox: List[float]
    center: List[float]
    angle: Optional[float] = None
    area: float = 0.0
    perimeter: float = 0.0


@dataclass
class TemplateSample:
    label: str
    contour: List[List[int]]


@dataclass
class FrameResult:
    detections: Dict[str, Optional[Detection]]
    binary: np.ndarray
    contours: List[np.ndarray]


COLOR_MAP = {
    "circle": (0, 255, 255),
    "rod": (0, 255, 0),
    "rotator": (255, 0, 255),
    "manual": (0, 128, 255),
    "roi": (255, 255, 0),
}


LABEL_KEYS = {ord("1"): "circle", ord("2"): "rod", ord("3"): "rotator"}


def list_images(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def angle_from_min_area_rect(rect) -> float:
    (_, _), (w, h), angle = rect
    if w < h:
        angle += 90.0
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return float(angle)


def contour_bbox(contour: np.ndarray) -> List[float]:
    x, y, w, h = cv2.boundingRect(contour)
    return [float(x), float(y), float(w), float(h)]


def contour_center(contour: np.ndarray) -> List[float]:
    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-6:
        x, y, w, h = cv2.boundingRect(contour)
        return [float(x + w / 2.0), float(y + h / 2.0)]
    return [float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])]


def contour_to_list(contour: np.ndarray) -> List[List[int]]:
    return contour.reshape(-1, 2).astype(int).tolist()


def list_to_contour(points: List[List[int]]) -> np.ndarray:
    return np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 自动让前景尽量为白色
    if np.mean(binary == 255) > 0.65:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def split_overlapped_mask(frame: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """对粘连目标做一次轻量分割，尽量拆开重叠轮廓。"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 2:
        return binary

    big_components = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] > 0.08 * binary.shape[0] * binary.shape[1]]
    if not big_components:
        return binary

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers <= 2:
        return binary

    markers = markers + 1
    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers[unknown == 255] = 0
    ws_img = frame.copy()
    cv2.watershed(ws_img, markers)

    result = np.zeros_like(binary)
    for idx in range(2, markers.max() + 1):
        result[markers == idx] = 255
    if cv2.countNonZero(result) > 0:
        return result
    return binary


def extract_contours(frame: np.ndarray, binary: np.ndarray) -> List[np.ndarray]:
    split_mask = split_overlapped_mask(frame, binary)
    contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def load_templates(template_path: Path) -> Dict[str, List[np.ndarray]]:
    if not template_path.exists():
        return {"circle": [], "rod": [], "rotator": []}
    with open(template_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result = {"circle": [], "rod": [], "rotator": []}
    for label, items in raw.items():
        if label not in result:
            continue
        for item in items:
            contour = np.asarray(item["contour"], dtype=np.int32).reshape(-1, 1, 2)
            result[label].append(contour)
    return result


def save_templates(template_path: Path, templates: Dict[str, List[np.ndarray]]) -> None:
    data = {
        label: [{"label": label, "contour": contour_to_list(contour)} for contour in contours]
        for label, contours in templates.items()
    }
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def contour_metrics(contour: np.ndarray) -> Dict[str, float]:
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / max(1.0, min(w, h))
    circularity = 4.0 * math.pi * area / max(1.0, peri * peri)
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    rect_aspect = max(rw, rh) / max(1.0, min(rw, rh))
    return {
        "area": area,
        "peri": peri,
        "aspect": aspect,
        "circularity": circularity,
        "rect_aspect": rect_aspect,
        "fill_ratio": area / max(1.0, float(w * h)),
        "angle": angle_from_min_area_rect(rect),
    }


def contour_similarity(contour: np.ndarray, template: np.ndarray) -> float:
    return float(cv2.matchShapes(contour, template, cv2.CONTOURS_MATCH_I1, 0.0))


def geometric_label_score(contour: np.ndarray) -> Dict[str, float]:
    m = contour_metrics(contour)
    circle_score = max(0.0, 1.2 - abs(m["circularity"] - 1.0)) + max(0.0, 1.8 - abs(m["aspect"] - 1.0))
    rod_score = max(0.0, m["rect_aspect"] - 1.0) + max(0.0, m["fill_ratio"] - 0.2)
    rotator_score = m["area"] * (1.0 + max(0.0, 1.0 - abs(m["aspect"] - 1.0) * 0.15))
    return {"circle": circle_score, "rod": rod_score, "rotator": rotator_score}


def classify_contours(
    contours: List[np.ndarray],
    image_shape: Tuple[int, int, int],
    templates: Optional[Dict[str, List[np.ndarray]]] = None,
) -> Dict[str, Optional[Detection]]:
    h, w = image_shape[:2]
    img_area = float(h * w)
    candidates = []

    for idx, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        if area < max(120.0, img_area * 0.0003):
            continue
        peri = float(cv2.arcLength(contour, True))
        if peri <= 1e-6:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        metrics = contour_metrics(contour)
        candidates.append(
            {
                "idx": idx,
                "contour": contour,
                "area": area,
                "peri": peri,
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "center": contour_center(contour),
                **metrics,
            }
        )

    result: Dict[str, Optional[Detection]] = {"circle": None, "rod": None, "rotator": None}
    if not candidates:
        return result

    assigned = set()

    def make_detection(label: str, c: dict, score: float) -> Detection:
        return Detection(
            label=label,
            score=float(score),
            contour=contour_to_list(c["contour"]),
            bbox=c["bbox"],
            center=c["center"],
            angle=c["angle"] if label == "rod" else None,
            area=c["area"],
            perimeter=c["peri"],
        )

    # 先用模板匹配做优先判别
    if templates:
        template_used = set()
        for label in ["circle", "rod", "rotator"]:
            if not templates.get(label):
                continue
            best = None
            best_score = 1e9
            for c in candidates:
                if c["idx"] in assigned:
                    continue
                score = min(contour_similarity(c["contour"], t) for t in templates[label])
                if score < best_score:
                    best_score = score
                    best = c
            if best is not None:
                threshold = 0.28 if label == "circle" else 0.35 if label == "rod" else 0.45
                if best_score <= threshold:
                    assigned.add(best["idx"])
                    template_used.add(label)
                    result[label] = make_detection(label, best, 1.0 - best_score)

    # 圆底
    if result["circle"] is None:
        circle_scores = [
            (
                max(0.0, c["circularity"]) * 0.7
                + max(0.0, 1.5 - abs(c["aspect"] - 1.0)) * 0.2
                + max(0.0, 1.5 - abs(c["rect_aspect"] - 1.0)) * 0.1,
                c,
            )
            for c in candidates
            if c["idx"] not in assigned and c["circularity"] > 0.45 and c["aspect"] < 2.2
        ]
        if circle_scores:
            _, best = max(circle_scores, key=lambda x: x[0])
            assigned.add(best["idx"])
            result["circle"] = make_detection("circle", best, best["circularity"])

    # 长条
    if result["rod"] is None:
        rod_scores = []
        for c in candidates:
            if c["idx"] in assigned:
                continue
            if c["rect_aspect"] >= 2.1 or c["aspect"] >= 2.1:
                score = c["rect_aspect"] * 0.75 + c["fill_ratio"] * 0.25
                rod_scores.append((score, c))
        if rod_scores:
            _, best = max(rod_scores, key=lambda x: x[0])
            assigned.add(best["idx"])
            result["rod"] = make_detection("rod", best, best["rect_aspect"])

    # 旋转体
    if result["rotator"] is None:
        rotator_scores = []
        for c in candidates:
            if c["idx"] in assigned:
                continue
            if c["area"] >= img_area * 0.001:
                score = c["area"] * (1.0 + max(0.0, 1.0 - abs(c["aspect"] - 1.0) * 0.2))
                rotator_scores.append((score, c))
        if rotator_scores:
            _, best = max(rotator_scores, key=lambda x: x[0])
            assigned.add(best["idx"])
            result["rotator"] = make_detection("rotator", best, best["area"])

    return result


def draw_detection(image: np.ndarray, det: Detection) -> np.ndarray:
    out = image.copy()
    pts = list_to_contour(det.contour)
    color = COLOR_MAP.get(det.label, (255, 255, 0))
    cv2.drawContours(out, [pts], -1, color, 2)
    x, y, w, h = det.bbox
    cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
    cx, cy = det.center
    cv2.circle(out, (int(cx), int(cy)), 4, color, -1)
    text = det.label
    if det.angle is not None:
        text += f" | angle={det.angle:.1f} deg"
    cv2.putText(out, text, (int(x), max(20, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def build_segmentation_view(frame: np.ndarray, binary: np.ndarray, detections: Dict[str, Optional[Detection]]) -> np.ndarray:
    seg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for det in detections.values():
        if det is not None:
            seg = draw_detection(seg, det)
    return seg


class ManualAnnotator:
    def __init__(self, template_path: Path):
        self.template_path = template_path
        self.templates = load_templates(template_path)
        self.current_label = "circle"
        self.drawing = False
        self.start_pt: Optional[Tuple[int, int]] = None
        self.end_pt: Optional[Tuple[int, int]] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_binary: Optional[np.ndarray] = None
        self.window_name = "手动标注"

    def set_frame(self, frame: np.ndarray):
        self.current_frame = frame.copy()
        self.current_binary = preprocess_frame(frame)

    def on_mouse(self, event, x, y, flags, param):
        if self.current_frame is None or self.current_binary is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.end_pt = (x, y)
            self._commit_annotation()

    def _commit_annotation(self):
        assert self.current_frame is not None and self.current_binary is not None
        assert self.start_pt is not None and self.end_pt is not None
        x1, y1 = self.start_pt
        x2, y2 = self.end_pt
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        if x_max - x_min < 8 or y_max - y_min < 8:
            return

        roi_bin = self.current_binary[y_min:y_max, x_min:x_max]
        contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        contour = max(contours, key=cv2.contourArea)
        contour = contour + np.array([[[x_min, y_min]]], dtype=np.int32)
        self.templates.setdefault(self.current_label, []).append(contour)
        save_templates(self.template_path, self.templates)
        print(f"已保存模板: {self.current_label} / {self.template_path}")

    def build_canvas(self) -> np.ndarray:
        assert self.current_frame is not None
        canvas = self.current_frame.copy()
        if self.current_binary is not None:
            overlay = cv2.cvtColor(self.current_binary, cv2.COLOR_GRAY2BGR)
            canvas = cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0)
        if self.drawing and self.start_pt and self.end_pt:
            color = COLOR_MAP.get(self.current_label, COLOR_MAP["manual"])
            cv2.rectangle(canvas, self.start_pt, self.end_pt, color, 2)
        cv2.putText(
            canvas,
            f"标注模式={self.current_label} | 1圆底 2长条 3旋转体 | 回车保存模板 | c清空模板 | q退出",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return canvas

    def run(self, frame: np.ndarray):
        self.set_frame(frame)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        while True:
            cv2.imshow(self.window_name, self.build_canvas())
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key in LABEL_KEYS:
                self.current_label = LABEL_KEYS[key]
            elif key == ord("c"):
                self.templates = {"circle": [], "rod": [], "rotator": []}
                save_templates(self.template_path, self.templates)
            elif key in (13, 10):
                save_templates(self.template_path, self.templates)
                print(f"模板已刷新: {self.template_path}")
        cv2.destroyWindow(self.window_name)


class SegmentationApp:
    def __init__(self, args):
        self.args = args
        self.input_path = Path(args.input)
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = self.output_dir / "templates.json"
        self.templates = load_templates(self.template_path)
        self.annotator = ManualAnnotator(self.template_path)
        self.capture = None
        self.source_type = None
        self.frame_count = 0
        self.current_idx = 0
        self.paused = False
        self.recording = False
        self.roi = None
        self.screen_roi = None
        self.last_frame = None
        self.last_result = None
        self._updating_seek = False
        self.seek_window = "视频进度"
        self.status = ""
        self.screen_rect = None

    def start(self):
        if self.input_path.is_dir():
            files = list_images(self.input_path)
            if not files:
                raise RuntimeError(f"目录中未找到图片: {self.input_path}")
            self.source_type = "image"
            self._run_image_mode(files[0])
            return

        suffix = self.input_path.suffix.lower()
        if suffix in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}:
            self.source_type = "video"
            self._run_video_mode()
            return

        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            self.source_type = "image"
            frame = cv2.imread(str(self.input_path))
            if frame is None:
                raise RuntimeError(f"无法读取图片: {self.input_path}")
            self._run_image_mode(self.input_path, frame)
            return

        if self.input_path.name.lower() == "screen":
            self.source_type = "screen"
            self._run_screen_mode()
            return

        raise RuntimeError("不支持的输入类型。请提供图片、视频文件，或使用 screen 作为屏幕模式。")

    def _select_roi(self, frame: np.ndarray, title: str) -> Optional[Tuple[int, int, int, int]]:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(title, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(title)
        if roi[2] <= 0 or roi[3] <= 0:
            return None
        return tuple(map(int, roi))

    def _capture_screen(self) -> np.ndarray:
        if ImageGrab is None:
            raise RuntimeError("当前环境缺少 Pillow/ImageGrab，无法进行屏幕截取。")
        img = ImageGrab.grab()
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return frame

    def _run_image_mode(self, path: Path, frame: Optional[np.ndarray] = None):
        if frame is None:
            frame = cv2.imread(str(path))
        assert frame is not None
        self.last_frame = frame
        if self.roi is None:
            self.roi = self._select_roi(frame, "请选择ROI（图片）")
        if self.roi is None:
            self.roi = (0, 0, frame.shape[1], frame.shape[0])
        self._interactive_loop(frame)

    def _run_video_mode(self):
        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.input_path}")
        self.capture = cap
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.current_idx = 0
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("无法读取视频首帧")
        self.last_frame = frame
        if self.roi is None:
            self.roi = self._select_roi(frame, "请选择ROI（视频首帧）")
        if self.roi is None:
            self.roi = (0, 0, frame.shape[1], frame.shape[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._interactive_loop(mode="video")
        cap.release()

    def _run_screen_mode(self):
        frame = self._capture_screen()
        self.last_frame = frame
        if self.screen_roi is None:
            self.screen_roi = self._select_roi(frame, "请选择屏幕ROI")
        if self.screen_roi is None:
            self.screen_roi = (0, 0, frame.shape[1], frame.shape[0])
        self._interactive_loop(mode="screen")

    def _get_frame(self, mode: str):
        if mode == "video":
            assert self.capture is not None
            if not self.paused:
                if self.frame_count > 0:
                    seek = cv2.getTrackbarPos("进度", self.seek_window)
                    if not self._updating_seek and abs(seek - self.current_idx) > 1:
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, seek)
                        self.current_idx = seek
                ok, frame = self.capture.read()
                if not ok:
                    return None, None
                self.current_idx = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                return frame, self.current_idx
            return self.last_frame, self.current_idx

        if mode == "screen":
            return self._capture_screen(), 0

        return self.last_frame, 0

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        roi = self.screen_roi if self.source_type == "screen" else self.roi
        if roi is None:
            return frame
        x, y, w, h = roi
        x = max(0, x)
        y = max(0, y)
        return frame[y : y + h, x : x + w].copy()

    def _analyze_frame(self, frame: np.ndarray) -> FrameResult:
        binary = preprocess_frame(frame)
        contours = extract_contours(frame, binary)
        detections = classify_contours(contours, frame.shape, self.templates)
        return FrameResult(detections=detections, binary=binary, contours=contours)

    def _compose_view(self, frame: np.ndarray, result: FrameResult, fps: float, idx: int) -> np.ndarray:
        orig = frame.copy()
        seg = build_segmentation_view(frame, result.binary, result.detections)
        for det in result.detections.values():
            if det is not None and det.label == "rod" and det.angle is not None:
                cv2.line(seg, (0, seg.shape[0] // 2), (seg.shape[1], seg.shape[0] // 2), (255, 255, 255), 1)
                cv2.putText(seg, f"rod angle={det.angle:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break

        info = [
            f"source={self.source_type}",
            f"frame={idx}",
            f"fps={fps:.1f}",
            f"paused={self.paused}",
            f"recording={self.recording}",
            f"label={self.annotator.current_label}",
            "keys: q退出 space暂停/继续 r记录 m标注 e重选ROI s保存快照 p视频拖动",
        ]
        if self.roi:
            info.append(f"ROI={self.roi}")
        for i, text in enumerate(info):
            cv2.putText(orig, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(seg, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        height = max(orig.shape[0], seg.shape[0])
        if orig.shape[0] != height:
            orig = cv2.copyMakeBorder(orig, 0, height - orig.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if seg.shape[0] != height:
            seg = cv2.copyMakeBorder(seg, 0, height - seg.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        canvas = np.hstack([orig, seg])
        cv2.putText(canvas, "原图", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, "分割图", (orig.shape[1] + 10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return canvas

    def _save_snapshot(self, frame: np.ndarray, result: FrameResult, idx: int):
        ts = int(time.time() * 1000)
        img_path = self.output_dir / f"snapshot_{ts}_{idx}.png"
        json_path = self.output_dir / f"snapshot_{ts}_{idx}.json"
        canvas = build_segmentation_view(frame, result.binary, result.detections)
        cv2.imwrite(str(img_path), canvas)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "frame_index": idx,
                    "roi": self.roi if self.source_type != "screen" else self.screen_roi,
                    "detections": {k: asdict(v) if v is not None else None for k, v in result.detections.items()},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _interactive_loop(self, frame: Optional[np.ndarray] = None, mode: Optional[str] = None):
        mode = mode or self.source_type
        if frame is not None:
            self.last_frame = frame
        assert mode is not None
        cv2.namedWindow(WINDOW_ORIG, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_SEG, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_HELP, cv2.WINDOW_NORMAL)
        if mode == "video" and self.frame_count > 1:
            cv2.namedWindow(self.seek_window, cv2.WINDOW_NORMAL)
            cv2.createTrackbar("进度", self.seek_window, 0, self.frame_count - 1, lambda v: None)

        while True:
            frame, idx = self._get_frame(mode)
            if frame is None:
                break
            self.last_frame = frame
            crop = self._crop_roi(frame)
            result = self._analyze_frame(crop)
            self.last_result = result
            fps = 0.0
            canvas = self._compose_view(crop, result, fps, idx)
            orig = canvas[:, : canvas.shape[1] // 2]
            seg = canvas[:, canvas.shape[1] // 2 :]
            cv2.imshow(WINDOW_ORIG, orig)
            cv2.imshow(WINDOW_SEG, seg)
            if mode == "video" and self.frame_count > 0:
                cv2.imshow(self.seek_window, np.zeros((1, 1, 3), dtype=np.uint8))
            help_img = np.zeros((260, 940, 3), dtype=np.uint8)
            lines = [
                "操作流程：1 选择视频/屏幕 -> 2 选择ROI -> 3 按 m 进行手动标注 -> 4 按 r 开始/停止记录",
                "键位：q退出 | space暂停/继续 | e重新选择ROI | s保存快照 | m手动模板标注 | 1/2/3切换标注类别",
                "视频模式可拖动进度条；屏幕模式会持续抓取当前屏幕ROI区域的实时画面。",
                "重叠目标处理：先轮廓分离，再用模板匹配辅助分类；长条角度使用最小外接矩形的长边方向。",
            ]
            for i, text in enumerate(lines):
                cv2.putText(help_img, text, (10, 35 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(WINDOW_HELP, help_img)

            if mode == "video" and self.frame_count > 0:
                self._updating_seek = True
                cv2.setTrackbarPos("进度", self.seek_window, max(0, idx))
                self._updating_seek = False

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                self.paused = not self.paused
            elif key == ord("r"):
                self.recording = not self.recording
            elif key == ord("s"):
                self._save_snapshot(crop, result, idx)
            elif key == ord("m"):
                self.annotator.current_label = self.annotator.current_label or "circle"
                self.annotator.run(crop)
                self.templates = load_templates(self.template_path)
            elif key == ord("e"):
                self.roi = None
                self.screen_roi = None
                if mode == "video" and self.capture is not None:
                    ok, new_frame = self.capture.read()
                    if ok:
                        self.roi = self._select_roi(new_frame, "重新选择ROI")
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
                elif mode == "screen":
                    new_frame = self._capture_screen()
                    self.screen_roi = self._select_roi(new_frame, "重新选择屏幕ROI")
            elif key in LABEL_KEYS:
                self.annotator.current_label = LABEL_KEYS[key]
            elif key == ord("p") and mode == "video" and self.capture is not None and self.frame_count > 0:
                # 保留为视频拖动提示，不额外占用其它按键
                pass

        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="三对象识别 + 视频/屏幕ROI + 手动模板标注 UI")
    parser.add_argument(
        "--input",
        type=str,
        default=r"e:\jupyter file\2_Optics\8821L\Utils\ADOTOP\EdgeDetect\Fore_BackGround_471_images",
        help="图片目录、图片文件、视频文件，或 screen（屏幕模式）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"e:\jupyter file\2_Optics\8821L\Utils\ADOTOP\EdgeDetect\outputs",
        help="输出目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = SegmentationApp(args)
    app.start()


if __name__ == "__main__":
    main()
