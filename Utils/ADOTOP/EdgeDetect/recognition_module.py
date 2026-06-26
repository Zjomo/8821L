import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLOR_MAP = {
    "circle": (0, 255, 255),
    "rod": (0, 255, 0),
    "rotator": (255, 0, 255),
    "manual": (0, 128, 255),
    "roi": (255, 255, 0),
}
LABEL_KEYS = {ord("1"): "circle", ord("2"): "rod", ord("3"): "rotator"}

MIN_CONTOUR_AREA_RATIO = 0.00025
MIN_CONTOUR_AREA_ABS = 100.0
CIRCLE_CIRCULARITY_MIN = 0.52
CIRCLE_ASPECT_MAX = 2.0
ROD_ASPECT_MIN = 2.0
ROTATOR_AREA_RATIO = 0.0008
TEMPLATE_THRESHOLDS = {"circle": 0.25, "rod": 0.32, "rotator": 0.40}
CIRCLE_SCORE_WEIGHTS = (0.70, 0.20, 0.10)
ROD_SCORE_WEIGHTS = (0.80, 0.20)
ROTATOR_SCORE_WEIGHTS = (1.0, 0.2)


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
class FrameResult:
    detections: Dict[str, Optional[Detection]]
    binary: np.ndarray
    contours: List[np.ndarray]


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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground_ratio = float(np.mean(binary == 255))
    if foreground_ratio > 0.65:
        binary = cv2.bitwise_not(binary)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel5, iterations=1)
    return binary


def split_overlapped_mask(frame: np.ndarray, binary: np.ndarray) -> np.ndarray:
    nonzero = cv2.countNonZero(binary)
    if nonzero == 0:
        return binary

    total_area = binary.shape[0] * binary.shape[1]
    if nonzero < total_area * 0.01:
        return binary

    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 2:
        return binary

    largest_area = max(stats[1:, cv2.CC_STAT_AREA]) if num > 1 else 0
    if largest_area < total_area * 0.04:
        return binary

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() <= 1e-6:
        return binary

    _, sure_fg = cv2.threshold(dist, 0.42 * dist.max(), 255, 0)
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
    return result if cv2.countNonZero(result) > 0 else binary


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
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / max(1.0, hull_area)
    ellipse_ratio = 0.0
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            (ea, eb) = ellipse[1]
            ellipse_ratio = max(ea, eb) / max(1.0, min(ea, eb))
        except cv2.error:
            ellipse_ratio = 0.0
    return {
        "area": area,
        "peri": peri,
        "aspect": aspect,
        "circularity": circularity,
        "rect_aspect": rect_aspect,
        "fill_ratio": area / max(1.0, float(w * h)),
        "angle": angle_from_min_area_rect(rect),
        "solidity": solidity,
        "ellipse_ratio": ellipse_ratio,
    }


def contour_similarity(contour: np.ndarray, template: np.ndarray) -> float:
    return float(cv2.matchShapes(contour, template, cv2.CONTOURS_MATCH_I1, 0.0))


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
        if area < max(MIN_CONTOUR_AREA_ABS, img_area * MIN_CONTOUR_AREA_RATIO):
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

    if templates:
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
            if best is not None and best_score <= TEMPLATE_THRESHOLDS[label]:
                assigned.add(best["idx"])
                result[label] = make_detection(label, best, 1.0 - best_score)

    if result["circle"] is None:
        circle_scores = []
        for c in candidates:
            if c["idx"] in assigned:
                continue
            if c["circularity"] >= CIRCLE_CIRCULARITY_MIN and c["aspect"] <= CIRCLE_ASPECT_MAX:
                score = (
                    c["circularity"] * CIRCLE_SCORE_WEIGHTS[0]
                    + max(0.0, 1.6 - abs(c["aspect"] - 1.0)) * CIRCLE_SCORE_WEIGHTS[1]
                    + max(0.0, 1.5 - abs(c["solidity"] - 1.0)) * CIRCLE_SCORE_WEIGHTS[2]
                )
                circle_scores.append((score, c))
        if circle_scores:
            _, best = max(circle_scores, key=lambda x: x[0])
            assigned.add(best["idx"])
            result["circle"] = make_detection("circle", best, best["circularity"])

    if result["rod"] is None:
        rod_scores = []
        for c in candidates:
            if c["idx"] in assigned:
                continue
            if c["rect_aspect"] >= ROD_ASPECT_MIN:
                rod_shape = max(c["rect_aspect"], c["ellipse_ratio"] if c["ellipse_ratio"] > 0 else c["rect_aspect"])
                score = rod_shape * ROD_SCORE_WEIGHTS[0] + c["fill_ratio"] * ROD_SCORE_WEIGHTS[1]
                rod_scores.append((score, c))
        if rod_scores:
            _, best = max(rod_scores, key=lambda x: x[0])
            assigned.add(best["idx"])
            result["rod"] = make_detection("rod", best, best["rect_aspect"])

    if result["rotator"] is None:
        rotator_scores = []
        for c in candidates:
            if c["idx"] in assigned:
                continue
            if c["area"] >= img_area * ROTATOR_AREA_RATIO:
                compact = max(0.0, 1.5 - abs(c["aspect"] - 1.0))
                score = c["area"] * ROTATOR_SCORE_WEIGHTS[0] + compact * ROTATOR_SCORE_WEIGHTS[1]
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


def build_segmentation_view(binary: np.ndarray, detections: Dict[str, Optional[Detection]]) -> np.ndarray:
    seg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for det in detections.values():
        if det is not None:
            seg = draw_detection(seg, det)
    return seg
