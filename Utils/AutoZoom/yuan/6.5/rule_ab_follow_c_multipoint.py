# actual_nano_boundary_following_yoloA_sam2BC.py
from __future__ import annotations

import csv
import math
import sys
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAM2_REPO_ROOT = PROJECT_ROOT / "sam2-main"
if SAM2_REPO_ROOT.exists() and str(SAM2_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO_ROOT))

from vision.screen_capture import FixedRegionScreenCapture
from control.stage34 import Stage34


# ============================================================
# 0. 日志
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. 基本配置
# ============================================================

Vec2 = Tuple[float, float]
CaptureArea = Tuple[int, int, int, int]


@dataclass
class RuntimeConfig:
    # --------------------------------------------------------
    # 固定屏幕截图区域
    # --------------------------------------------------------
    capture_area: CaptureArea = (116, 98, 1112, 886)

    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------
    output_dir: str = "outputs/actual_nano_boundary_following_sam2ABC"
    save_annotated_image: bool = True
    save_csv: bool = True

    # --------------------------------------------------------
    # 兼容旧 YOLO 配置字段。当前主流程已经完全绕过 YOLO，
    # A/B/C 均由 SAM2 第一帧点击初始化并逐帧更新。
    # --------------------------------------------------------
    yolo_a_model_path: str = "vision/best_nano_lightspot.pt"
    class_name_a: str = "nano"
    yolo_conf_threshold: float = 0.35

    # 兼容旧 YOLO 配置字段，当前主流程不使用。
    use_last_a_center_to_select: bool = True

    # --------------------------------------------------------
    # SAM2：第一帧点击 A/B/C，后续用上一帧 bbox + center 更新
    # 下面路径需要改成你的 SAM2 配置和权重路径。
    # --------------------------------------------------------
    sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_checkpoint: str = str(SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt")
    sam2_device: str = "cuda"

    # 第一帧手动点击窗口缩放比例
    init_window_scale: float = 0.85

    # SAM2 后续帧质量检查
    sam2_min_area_px: float = 50.0
    sam2_area_ratio_min: float = 0.35
    sam2_area_ratio_max: float = 2.80
    sam2_center_jump_max_px: float = 180.0

    # 如果 SAM2 当前帧失败，是否使用上一帧 mask 兜底
    # 为了避免错误运动，建议正式实验设 False；
    # 调试时可以设 True。
    use_last_mask_when_sam2_failed: bool = False

    # --------------------------------------------------------
    # mask 后处理
    # --------------------------------------------------------
    min_mask_area_px: float = 50.0
    contour_approx_epsilon_ratio: float = 0.006

    # --------------------------------------------------------
    # B 目标边选择方式
    # nearest/left/right/top/bottom/longest
    # --------------------------------------------------------
    b_target_edge_mode: str = "nearest"

    # --------------------------------------------------------
    # A-C 距离控制，单位 px
    # --------------------------------------------------------
    a_c_target_clearance: float = 90.0
    a_c_min_clearance: float = 80.0
    a_c_max_clearance: float = 100.0
    a_c_inside_collision_check: bool = True

    # A-C 距离修正方向模式：
    #   "horizontal_from_c_center"：A-C 太近/太远时，优先按 A 与 C 中心的左右关系修正；
    #       适合你的当前实验画面：A 需要在 C 左侧保持间隙，所以太近时应先向 LEFT 远离 C。
    #   "nearest_normal"：使用 A 到 C 最近边界的法向方向修正；
    #       如果 C 的最近边是水平边，就可能输出 UP/DOWN。
    #   "auto"：先尝试水平修正，水平关系不明显时退回 nearest_normal。
    a_c_correction_mode: str = "horizontal_from_c_center"

    # --------------------------------------------------------
    # A-B 覆盖暂停控制
    # --------------------------------------------------------
    # 新规则不再使用 A-B 距离控制。A 沿 C 边界运动并维持 A-C 间隙；
    # 只有当 A mask 与 B mask 发生覆盖时，Stage 停止不动，只继续检测 B 目标边角度。
    # 保留 a_b_min_dist / a_b_max_dist 只是为了兼容旧日志字段。
    a_b_min_dist: float = 10.0
    a_b_max_dist: float = 35.0
    a_b_overlap_stop: bool = True
    a_b_overlap_min_area_px: float = 1.0

    # --------------------------------------------------------
    # 沿 C 边界跟随方向
    # 如果发现沿边界方向反了，把 1 改成 -1。
    # --------------------------------------------------------
    follow_c_direction: int = 1

    # --------------------------------------------------------
    # 混合控制权重
    # --------------------------------------------------------
    boundary_tangent_weight: float = 1.00
    boundary_clearance_correction_weight: float = 0.65
    boundary_ab_correction_weight: float = 0.45

    # --------------------------------------------------------
    # Stage34 参数
    # --------------------------------------------------------
    stage_conn: str = "27267878"
    stage_x_channel: int = 3
    stage_y_channel: int = 4
    stage_default_velocity: int = 10
    stage_default_acceleration: int = 10
    stage_default_max_voltage: int = 100
    stage_step_x: int = 100
    stage_step_y: int = 100

    # 如果图像方向和实际运动方向相反，改这里
    stage_x_sign: int = 1
    stage_y_sign: int = 1

    # 每次规则动作对应的实际位移步数
    action_step: int = 100

    # 是否等待 Stage 运动完成
    wait_stage_move: bool = True
    stage_move_timeout_s: Optional[float] = 5.0

    # --------------------------------------------------------
    # 闭环循环参数
    # --------------------------------------------------------
    loop_interval_s: float = 0.15
    max_cycles: int = 500
    max_lost_frames: int = 10

    # 调试视觉时先设 False，确认方向正确后再设 True
    enable_stage: bool = False


# ============================================================
# 2. 向量和几何工具
# ============================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def vec_add(a: Vec2, b: Vec2) -> Vec2:
    return a[0] + b[0], a[1] + b[1]


def vec_sub(a: Vec2, b: Vec2) -> Vec2:
    return a[0] - b[0], a[1] - b[1]


def vec_mul(a: Vec2, s: float) -> Vec2:
    return a[0] * s, a[1] * s


def vec_dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def vec_len(a: Vec2) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


def vec_norm(a: Vec2) -> Vec2:
    l = vec_len(a)
    if l < 1e-8:
        return 0.0, 0.0
    return a[0] / l, a[1] / l


def polygon_edges(points: List[Vec2]) -> List[Tuple[Vec2, Vec2]]:
    if len(points) < 2:
        return []
    return list(zip(points, points[1:] + points[:1]))


def polygon_centroid_simple(points: List[Vec2]) -> Vec2:
    if not points:
        return 0.0, 0.0
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    return sx / len(points), sy / len(points)


def distance_point_to_segment(p: Vec2, a: Vec2, b: Vec2) -> Tuple[float, Vec2, float]:
    ab = vec_sub(b, a)
    ap = vec_sub(p, a)
    ab2 = vec_dot(ab, ab)

    if ab2 < 1e-8:
        return vec_len(vec_sub(p, a)), a, 0.0

    t = vec_dot(ap, ab) / ab2
    t_clamped = clamp(t, 0.0, 1.0)
    proj = vec_add(a, vec_mul(ab, t_clamped))
    return vec_len(vec_sub(p, proj)), proj, t_clamped


def point_in_polygon(p: Vec2, poly_points: List[Vec2]) -> bool:
    x, y = p
    inside = False
    n = len(poly_points)

    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = poly_points[i]
        xj, yj = poly_points[j]

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def compute_point_to_polygon_boundary_contact(
    p: Vec2,
    poly_points: List[Vec2],
    inside_collision_check: bool = True,
) -> Dict[str, Any]:
    best = None

    for c1, c2 in polygon_edges(poly_points):
        dist, proj, t = distance_point_to_segment(p, c1, c2)

        item = {
            "dist": dist,
            "effective_dist": dist,
            "a_contact": p,
            "c_contact": proj,
            "c_edge": (c1, c2),
            "c_edge_dir": vec_norm(vec_sub(c2, c1)),
            "c_t": t,
        }

        if best is None or item["dist"] < best["dist"]:
            best = item

    if best is None:
        raise RuntimeError("C polygon has no valid edge.")

    collision_like = False
    if inside_collision_check and point_in_polygon(p, poly_points):
        collision_like = True

    boundary_to_a = vec_norm(vec_sub(p, best["c_contact"]))

    if vec_len(boundary_to_a) < 1e-8 or collision_like:
        c_center = polygon_centroid_simple(poly_points)
        boundary_to_a = vec_norm(vec_sub(p, c_center))

    if vec_len(boundary_to_a) < 1e-8:
        boundary_to_a = (1.0, 0.0)

    best["boundary_to_a"] = boundary_to_a
    best["collision_like"] = collision_like

    if collision_like:
        best["effective_dist"] = 0.0

    return best


def compute_point_to_segment_contact(p: Vec2, a: Vec2, b: Vec2) -> Dict[str, Any]:
    dist, proj, t = distance_point_to_segment(p, a, b)

    if dist > 1e-8:
        edge_to_a = vec_norm(vec_sub(p, proj))
    else:
        edge_to_a = (1.0, 0.0)

    return {
        "dist": dist,
        "a_contact": p,
        "b_contact": proj,
        "b_edge": (a, b),
        "b_edge_dir": vec_norm(vec_sub(b, a)),
        "b_t": t,
        "edge_to_a": edge_to_a,
    }


def vector_to_action(v: Vec2) -> int:
    """
    0 = STAY
    1 = UP
    2 = DOWN
    3 = LEFT
    4 = RIGHT
    """
    x, y = v

    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return 0

    if abs(x) >= abs(y):
        return 4 if x > 0 else 3
    else:
        return 2 if y > 0 else 1


ACTION_NAMES = {
    0: "STAY",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
}

ACTION_TO_DIRECTION = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}


# ============================================================
# 3. 数据结构
# ============================================================

@dataclass
class ADetection:
    bbox_xyxy: Tuple[float, float, float, float]
    center: Vec2
    confidence: float
    mask: Optional[np.ndarray] = None


@dataclass
class FlakeGeometry:
    name: str
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    mask: np.ndarray
    contour: List[Vec2]
    center: Vec2
    area: float
    source: str = ""


@dataclass
class SceneGeometry:
    a: FlakeGeometry
    b: FlakeGeometry
    c: FlakeGeometry
    b_target_edge: Tuple[Vec2, Vec2]
    timestamp: float
    image_shape: Tuple[int, int, int]


@dataclass
class PointPromptSet:
    """
    单个目标的 SAM2 点提示。

    positive_points: 多个正点，表示这些点属于同一个目标。
    negative_points: 多个负点，表示这些点不属于该目标。
    """
    positive_points: List[Vec2]
    negative_points: List[Vec2]


# ============================================================
# 4. mask / contour 工具
# ============================================================

def mask_to_bbox_xyxy(mask_bool: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask_bool)

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("mask 为空，无法计算 bbox。")

    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def bbox_to_mask(
    bbox_xyxy: Tuple[float, float, float, float],
    image_shape_hw: Tuple[int, int],
) -> np.ndarray:
    h, w = image_shape_hw
    x1, y1, x2, y2 = bbox_xyxy

    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w - 1, round(x2))))
    y2 = int(max(0, min(h - 1, round(y2))))

    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2 + 1, x1:x2 + 1] = True
    return mask


def mask_center(mask_bool: np.ndarray) -> Vec2:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    m = cv2.moments(mask_u8)

    if abs(m["m00"]) < 1e-8:
        x1, y1, x2, y2 = mask_to_bbox_xyxy(mask_bool)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def mask_to_largest_contour_polygon(
    mask_bool: np.ndarray,
    min_area: float = 50.0,
    epsilon_ratio: float = 0.006,
) -> Tuple[List[Vec2], float, Vec2]:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError("mask 中没有找到有效轮廓。")

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))

    if area < min_area:
        raise RuntimeError(f"mask 轮廓面积过小: area={area:.2f}")

    peri = float(cv2.arcLength(contour, True))
    epsilon = max(1.0, epsilon_ratio * peri)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    pts: List[Vec2] = []
    for p in approx.reshape(-1, 2):
        pts.append((float(p[0]), float(p[1])))

    if len(pts) < 3:
        raise RuntimeError("轮廓点数小于 3，无法构成多边形。")

    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-8:
        center = polygon_centroid_simple(pts)
    else:
        center = (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))

    return pts, area, center


def edge_mid(edge: Tuple[Vec2, Vec2]) -> Vec2:
    a, b = edge
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def choose_b_target_edge(
    b_polygon: List[Vec2],
    mode: str = "left",
    reference_point: Optional[Vec2] = None,
) -> Tuple[Vec2, Vec2]:
    edges = polygon_edges(b_polygon)

    if not edges:
        raise RuntimeError("B 轮廓没有有效边。")

    mode = mode.lower().strip()

    if mode == "nearest":
        if reference_point is None:
            raise ValueError("b_target_edge_mode='nearest' requires reference_point.")
        return min(
            edges,
            key=lambda e: distance_point_to_segment(reference_point, e[0], e[1])[0],
        )

    if mode == "longest":
        return max(edges, key=lambda e: vec_len(vec_sub(e[1], e[0])))

    if mode == "left":
        return min(edges, key=lambda e: edge_mid(e)[0])

    if mode == "right":
        return max(edges, key=lambda e: edge_mid(e)[0])

    if mode == "top":
        return min(edges, key=lambda e: edge_mid(e)[1])

    if mode == "bottom":
        return max(edges, key=lambda e: edge_mid(e)[1])

    raise ValueError(
        f"unsupported b_target_edge_mode={mode}. "
        "Use nearest/left/right/top/bottom/longest."
    )


def edge_angle_deg(edge: Tuple[Vec2, Vec2]) -> Optional[float]:
    """返回边在图像坐标系中的方向角，范围 0~180 deg。"""
    try:
        p1, p2 = edge
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        return float(math.degrees(math.atan2(dy, dx)) % 180.0)
    except Exception:
        return None


def mask_overlap_area(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """计算两个 bool mask 的覆盖像素面积。"""
    if mask_a is None or mask_b is None:
        return 0.0
    if mask_a.shape != mask_b.shape:
        return 0.0
    return float(np.logical_and(mask_a.astype(bool), mask_b.astype(bool)).sum())


# ============================================================
# 5. YOLO：只检测 A
# ============================================================

class YoloADetector:
    """
    YOLO 只负责检测 A。

    支持：
        1. YOLO detect：输出 A bbox；
        2. YOLO-seg：输出 A bbox + mask。

    如果 YOLO 输出多个 A：
        - 第一帧选择置信度最高；
        - 后续帧可选择离上一帧 A 最近的目标。
    """

    def __init__(
        self,
        model_path: str,
        class_name_a: str = "A",
        conf_threshold: float = 0.35,
        use_last_center_to_select: bool = True,
    ):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise ImportError(
                "未安装 ultralytics。请先执行: pip install ultralytics"
            ) from e

        self.model = YOLO(model_path)
        self.class_name_a = str(class_name_a)
        self.conf_threshold = float(conf_threshold)
        self.use_last_center_to_select = bool(use_last_center_to_select)

    def infer_a(
        self,
        image_rgb: np.ndarray,
        last_a_center: Optional[Vec2] = None,
    ) -> ADetection:
        h, w = image_rgb.shape[:2]

        results = self.model.predict(
            source=image_rgb,
            conf=self.conf_threshold,
            verbose=False,
        )

        if not results:
            raise RuntimeError("YOLO 没有返回结果。")

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            raise RuntimeError("YOLO 没有检测到任何目标。")

        names = r.names
        candidates: List[Dict[str, Any]] = []

        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].detach().cpu().item())
            conf = float(r.boxes.conf[i].detach().cpu().item())
            class_name = str(names.get(cls_id, cls_id))

            if class_name.lower().strip() != self.class_name_a.lower().strip():
                continue

            xyxy = r.boxes.xyxy[i].detach().cpu().numpy().astype(float)
            x1, y1, x2, y2 = map(float, xyxy)
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            candidates.append(
                {
                    "index": i,
                    "confidence": conf,
                    "bbox_xyxy": (x1, y1, x2, y2),
                    "center": center,
                }
            )

        if not candidates:
            available = []
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].detach().cpu().item())
                available.append(str(names.get(cls_id, cls_id)))
            raise RuntimeError(
                f"YOLO 没有检测到 A 类别: {self.class_name_a}。"
                f"当前检测类别: {sorted(set(available))}"
            )

        if (
            self.use_last_center_to_select
            and last_a_center is not None
            and len(candidates) > 1
        ):
            best = min(
                candidates,
                key=lambda d: vec_len(vec_sub(d["center"], last_a_center)),
            )
        else:
            best = max(candidates, key=lambda d: d["confidence"])

        mask = None

        if r.masks is not None:
            mask_data = r.masks.data[best["index"]].detach().cpu().numpy()
            mask_u8 = (mask_data > 0.5).astype(np.uint8)

            if mask_u8.shape[:2] != (h, w):
                mask_u8 = cv2.resize(
                    mask_u8,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

            mask = mask_u8.astype(bool)

        return ADetection(
            bbox_xyxy=best["bbox_xyxy"],
            center=best["center"],
            confidence=float(best["confidence"]),
            mask=mask,
        )


# ============================================================
# 6. SAM2：第一帧点击 B/C，后续 bbox + center prompt 更新
# ============================================================

class SAM2ABCSegmenter:
    """
    SAM2 负责 A/B/C 的分割和身份绑定。

    第一帧：
        用户点击 A 内部一点；
        用户点击 B 内部一点；
        用户点击 C 内部一点；
        SAM2 分割 A、B 和 C。

    后续帧：
        用上一帧 A bbox + A center 作为 prompt 更新 A；
        用上一帧 B bbox + B center 作为 prompt 更新 B；
        用上一帧 C bbox + C center 作为 prompt 更新 C。

    注意：
        这里使用 SAM2ImagePredictor 的逐帧方案，
        不是 SAM2 video predictor。
        优点是容易接入你现在的固定区域截图闭环。
    """

    def __init__(
        self,
        sam2_cfg: str,
        sam2_checkpoint: str,
        device: str = "cuda",
        min_area_px: float = 50.0,
        area_ratio_min: float = 0.35,
        area_ratio_max: float = 2.80,
        center_jump_max_px: float = 180.0,
        use_last_mask_when_failed: bool = False,
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

        sam2_model = build_sam2(
            sam2_cfg,
            sam2_checkpoint,
            device=device,
        )

        self.predictor = SAM2ImagePredictor(sam2_model)

        self.min_area_px = float(min_area_px)
        self.area_ratio_min = float(area_ratio_min)
        self.area_ratio_max = float(area_ratio_max)
        self.center_jump_max_px = float(center_jump_max_px)
        self.use_last_mask_when_failed = bool(use_last_mask_when_failed)

        self.initialized = False

        self.a_init_point: Optional[Vec2] = None
        self.b_init_point: Optional[Vec2] = None
        self.c_init_point: Optional[Vec2] = None

        # 多点提示记录：第一帧可以为 A/B/C 分别标记多个正点和多个负点。
        self.a_init_positive_points: List[Vec2] = []
        self.a_init_negative_points: List[Vec2] = []
        self.b_init_positive_points: List[Vec2] = []
        self.b_init_negative_points: List[Vec2] = []
        self.c_init_positive_points: List[Vec2] = []
        self.c_init_negative_points: List[Vec2] = []

        self.last_a_mask: Optional[np.ndarray] = None
        self.last_b_mask: Optional[np.ndarray] = None
        self.last_c_mask: Optional[np.ndarray] = None

        self.last_a_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_b_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_c_bbox: Optional[Tuple[float, float, float, float]] = None

        self.last_a_center: Optional[Vec2] = None
        self.last_b_center: Optional[Vec2] = None
        self.last_c_center: Optional[Vec2] = None

    def initialize_with_points(
        self,
        image_rgb: np.ndarray,
        a_point: Optional[Vec2] = None,
        b_point: Optional[Vec2] = None,
        c_point: Optional[Vec2] = None,
        a_positive_points: Optional[List[Vec2]] = None,
        a_negative_points: Optional[List[Vec2]] = None,
        b_positive_points: Optional[List[Vec2]] = None,
        b_negative_points: Optional[List[Vec2]] = None,
        c_positive_points: Optional[List[Vec2]] = None,
        c_negative_points: Optional[List[Vec2]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        用多个正点/负点初始化 A/B/C。

        兼容旧调用：如果只传 a_point/b_point/c_point，则自动转成每个目标一个正点，
        并把另外两个目标点作为负点。
        """
        if a_positive_points is None:
            if a_point is None:
                raise ValueError("A 至少需要一个正点。")
            a_positive_points = [a_point]
        if b_positive_points is None:
            if b_point is None:
                raise ValueError("B 至少需要一个正点。")
            b_positive_points = [b_point]
        if c_positive_points is None:
            if c_point is None:
                raise ValueError("C 至少需要一个正点。")
            c_positive_points = [c_point]

        # 旧模式下自动把其他目标点作为负点；新模式下用户可以自己提供负点。
        if a_negative_points is None:
            a_negative_points = []
            if b_positive_points:
                a_negative_points.append(b_positive_points[0])
            if c_positive_points:
                a_negative_points.append(c_positive_points[0])
        if b_negative_points is None:
            b_negative_points = []
            if a_positive_points:
                b_negative_points.append(a_positive_points[0])
            if c_positive_points:
                b_negative_points.append(c_positive_points[0])
        if c_negative_points is None:
            c_negative_points = []
            if a_positive_points:
                c_negative_points.append(a_positive_points[0])
            if b_positive_points:
                c_negative_points.append(b_positive_points[0])

        self.a_init_point = a_positive_points[0]
        self.b_init_point = b_positive_points[0]
        self.c_init_point = c_positive_points[0]
        self.a_init_positive_points = list(a_positive_points)
        self.a_init_negative_points = list(a_negative_points)
        self.b_init_positive_points = list(b_positive_points)
        self.b_init_negative_points = list(b_negative_points)
        self.c_init_positive_points = list(c_positive_points)
        self.c_init_negative_points = list(c_negative_points)

        self.predictor.set_image(image_rgb)

        a_mask = self._predict_mask_by_points(
            positive_points=a_positive_points,
            negative_points=a_negative_points,
        )
        b_mask = self._predict_mask_by_points(
            positive_points=b_positive_points,
            negative_points=b_negative_points,
        )
        c_mask = self._predict_mask_by_points(
            positive_points=c_positive_points,
            negative_points=c_negative_points,
        )

        self._basic_mask_check("A", a_mask)
        self._basic_mask_check("B", b_mask)
        self._basic_mask_check("C", c_mask)

        self.last_a_mask = a_mask
        self.last_b_mask = b_mask
        self.last_c_mask = c_mask

        self.last_a_bbox = mask_to_bbox_xyxy(a_mask)
        self.last_b_bbox = mask_to_bbox_xyxy(b_mask)
        self.last_c_bbox = mask_to_bbox_xyxy(c_mask)

        self.last_a_center = mask_center(a_mask)
        self.last_b_center = mask_center(b_mask)
        self.last_c_center = mask_center(c_mask)

        self.initialized = True

        logger.info(
            "SAM2 A/B/C 多点初始化完成: "
            "A_pos=%d,A_neg=%d; B_pos=%d,B_neg=%d; C_pos=%d,C_neg=%d; "
            "A_center=(%.1f, %.1f), B_center=(%.1f, %.1f), C_center=(%.1f, %.1f)",
            len(a_positive_points),
            len(a_negative_points),
            len(b_positive_points),
            len(b_negative_points),
            len(c_positive_points),
            len(c_negative_points),
            self.last_a_center[0],
            self.last_a_center[1],
            self.last_b_center[0],
            self.last_b_center[1],
            self.last_c_center[0],
            self.last_c_center[1],
        )

        return a_mask, b_mask, c_mask

    def infer_abc(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.initialized:
            raise RuntimeError("SAM2ABCSegmenter 尚未初始化，请先点击 A、B 和 C。")

        if self.last_a_bbox is None or self.last_b_bbox is None or self.last_c_bbox is None:
            raise RuntimeError("SAM2 A/B/C 上一帧 bbox 为空，无法继续跟踪。")

        self.predictor.set_image(image_rgb)

        try:
            a_pos = self._sample_positive_points_from_mask(self.last_a_mask, self.last_a_center, max_points=3)
            b_pos = self._sample_positive_points_from_mask(self.last_b_mask, self.last_b_center, max_points=3)
            c_pos = self._sample_positive_points_from_mask(self.last_c_mask, self.last_c_center, max_points=4)

            a_mask = self._predict_mask_by_box_and_points(
                box_xyxy=self.last_a_bbox,
                positive_points=a_pos,
                negative_points=[self.last_b_center, self.last_c_center],
            )
            b_mask = self._predict_mask_by_box_and_points(
                box_xyxy=self.last_b_bbox,
                positive_points=b_pos,
                negative_points=[self.last_a_center, self.last_c_center],
            )
            c_mask = self._predict_mask_by_box_and_points(
                box_xyxy=self.last_c_bbox,
                positive_points=c_pos,
                negative_points=[self.last_a_center, self.last_b_center],
            )

            self._validate_mask_update(
                name="A",
                new_mask=a_mask,
                old_mask=self.last_a_mask,
                old_center=self.last_a_center,
            )

            self._validate_mask_update(
                name="B",
                new_mask=b_mask,
                old_mask=self.last_b_mask,
                old_center=self.last_b_center,
            )

            self._validate_mask_update(
                name="C",
                new_mask=c_mask,
                old_mask=self.last_c_mask,
                old_center=self.last_c_center,
            )

            self.last_a_mask = a_mask
            self.last_b_mask = b_mask
            self.last_c_mask = c_mask

            self.last_a_bbox = mask_to_bbox_xyxy(a_mask)
            self.last_b_bbox = mask_to_bbox_xyxy(b_mask)
            self.last_c_bbox = mask_to_bbox_xyxy(c_mask)

            self.last_a_center = mask_center(a_mask)
            self.last_b_center = mask_center(b_mask)
            self.last_c_center = mask_center(c_mask)

            return a_mask, b_mask, c_mask

        except Exception as e:
            logger.error("SAM2 A/B/C 更新失败: %s", e)

            if (
                self.use_last_mask_when_failed
                and self.last_a_mask is not None
                and self.last_b_mask is not None
                and self.last_c_mask is not None
            ):
                logger.warning("使用上一帧 A/B/C mask 兜底。")
                return self.last_a_mask, self.last_b_mask, self.last_c_mask

            raise

    def _predict_mask_by_point(self, point: Vec2) -> np.ndarray:
        return self._predict_mask_by_points(
            positive_points=[point],
            negative_points=None,
        )

    @staticmethod
    def _clean_points(points: Optional[List[Optional[Vec2]]]) -> List[Vec2]:
        out: List[Vec2] = []
        for p in points or []:
            if p is None:
                continue
            out.append((float(p[0]), float(p[1])))
        return out

    def _predict_mask_by_points(
        self,
        positive_point: Optional[Vec2] = None,
        positive_points: Optional[List[Optional[Vec2]]] = None,
        negative_points: Optional[List[Optional[Vec2]]] = None,
    ) -> np.ndarray:
        """
        SAM2 点提示分割。

        支持：
            1. 旧接口 positive_point=单个正点；
            2. 新接口 positive_points=多个正点；
            3. negative_points=多个负点。

        多个正点/负点只用于分割同一个目标；A/B/C 仍然分别调用本函数。
        """
        pos = self._clean_points(positive_points)
        if positive_point is not None:
            pos.insert(0, (float(positive_point[0]), float(positive_point[1])))

        neg = self._clean_points(negative_points)

        if not pos:
            raise ValueError("SAM2 点提示至少需要一个正点。")

        points: List[List[float]] = []
        labels: List[int] = []

        for p in pos:
            points.append([p[0], p[1]])
            labels.append(1)

        for p in neg:
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

    def _sample_positive_points_from_mask(
        self,
        mask_bool: Optional[np.ndarray],
        center: Optional[Vec2],
        max_points: int = 3,
    ) -> List[Vec2]:
        """
        后续帧自动从上一帧 mask 内采样多个正点，减少单中心点漂移导致的分割失败。
        """
        points: List[Vec2] = []
        if center is not None:
            points.append((float(center[0]), float(center[1])))

        if mask_bool is None:
            return points

        ys, xs = np.where(mask_bool.astype(bool))
        if len(xs) == 0:
            return points

        # 用 bbox 内的几个代表位置找最近的 mask 内点。
        x1, y1, x2, y2 = mask_to_bbox_xyxy(mask_bool.astype(bool))
        candidate_targets = [
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            (x1 + 0.30 * (x2 - x1), y1 + 0.30 * (y2 - y1)),
            (x1 + 0.70 * (x2 - x1), y1 + 0.70 * (y2 - y1)),
            (x1 + 0.30 * (x2 - x1), y1 + 0.70 * (y2 - y1)),
            (x1 + 0.70 * (x2 - x1), y1 + 0.30 * (y2 - y1)),
        ]

        for tx, ty in candidate_targets:
            if len(points) >= max_points:
                break
            d2 = (xs.astype(float) - float(tx)) ** 2 + (ys.astype(float) - float(ty)) ** 2
            idx = int(np.argmin(d2))
            p = (float(xs[idx]), float(ys[idx]))
            # 避免重复点太近
            if all(vec_len(vec_sub(p, q)) > 3.0 for q in points):
                points.append(p)

        return points[:max_points]

    def _predict_mask_by_box_and_point(
        self,
        box_xyxy: Tuple[float, float, float, float],
        point: Optional[Vec2],
        negative_points: Optional[List[Optional[Vec2]]] = None,
    ) -> np.ndarray:
        return self._predict_mask_by_box_and_points(
            box_xyxy=box_xyxy,
            positive_points=[] if point is None else [point],
            negative_points=negative_points,
        )

    def _predict_mask_by_box_and_points(
        self,
        box_xyxy: Tuple[float, float, float, float],
        positive_points: Optional[List[Optional[Vec2]]] = None,
        negative_points: Optional[List[Optional[Vec2]]] = None,
    ) -> np.ndarray:
        box = np.array(box_xyxy, dtype=np.float32)

        pos = self._clean_points(positive_points)
        neg = self._clean_points(negative_points)

        if pos or neg:
            points: List[List[float]] = []
            labels: List[int] = []

            for p in pos:
                points.append([p[0], p[1]])
                labels.append(1)

            for p in neg:
                points.append([p[0], p[1]])
                labels.append(0)

            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)
        else:
            point_coords = None
            point_labels = None

        with self.torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM2 box+multi-point prompt 没有返回 mask。")

        best_idx = int(np.argmax(np.asarray(scores).reshape(-1)))
        return masks[best_idx].astype(bool)

    def _basic_mask_check(self, name: str, mask: np.ndarray) -> None:
        area = float(mask.sum())

        if area < self.min_area_px:
            raise RuntimeError(f"{name} mask 面积过小: area={area:.1f}")

    def _validate_mask_update(
        self,
        name: str,
        new_mask: np.ndarray,
        old_mask: Optional[np.ndarray],
        old_center: Optional[Vec2],
    ) -> None:
        self._basic_mask_check(name, new_mask)

        if old_mask is None:
            return

        old_area = float(old_mask.sum())
        new_area = float(new_mask.sum())

        if old_area > 1.0:
            ratio = new_area / old_area
            if ratio < self.area_ratio_min or ratio > self.area_ratio_max:
                raise RuntimeError(
                    f"{name} mask 面积突变: "
                    f"old={old_area:.1f}, new={new_area:.1f}, ratio={ratio:.2f}"
                )

        if old_center is not None:
            new_center = mask_center(new_mask)
            jump = vec_len(vec_sub(new_center, old_center))

            if jump > self.center_jump_max_px:
                raise RuntimeError(
                    f"{name} mask 中心跳变过大: jump={jump:.1f} px"
                )


# ============================================================
# 7. 第一帧点击 A/B/C 初始化
# ============================================================

def select_a_b_c_points_interactively(
    image_rgb: np.ndarray,
    window_name: str = "SAM2 multi-point init",
    scale: float = 0.85,
) -> Tuple[PointPromptSet, PointPromptSet, PointPromptSet]:
    """
    在第一帧图像上为 A/B/C 分别标记多个正点和多个负点。

    操作方式：
        当前目标依次为 A -> B -> C。
        左键：添加当前目标正点 positive point。
        右键：添加当前目标负点 negative point。
        N 或 Enter：完成当前目标，进入下一个目标。
        R：清空当前目标的点。
        Backspace：回到上一个目标。
        ESC：取消。

    注意：
        多个正点/负点是为了把“同一个目标”分得更准；
        A/B/C 仍然会分别调用 SAM2 三次，避免多个目标被粘成一个 mask。
    """
    if scale <= 0:
        scale = 1.0

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    show_w = int(w * scale)
    show_h = int(h * scale)

    display = cv2.resize(
        image_bgr,
        (show_w, show_h),
        interpolation=cv2.INTER_AREA,
    )

    names = ["A", "B", "C"]
    colors = {
        "A": (0, 255, 255),
        "B": (0, 128, 255),
        "C": (255, 255, 0),
    }
    prompt_sets: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        name: {"pos": [], "neg": []} for name in names
    }
    current_idx = 0

    def redraw() -> np.ndarray:
        canvas = display.copy()
        current_name = names[current_idx]

        instructions = [
            "SAM2 multi-point init",
            f"Current: {current_name} | Left=positive, Right=negative, N/Enter=next, R=reset current, Backspace=previous, ESC=cancel",
            "Tip: For each object, click several positive points inside it; click negative points on nearby non-target regions.",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(
                canvas,
                text,
                (20, 28 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        y_panel = 105
        for obj_i, name in enumerate(names):
            prefix = ">" if obj_i == current_idx else " "
            count_text = (
                f"{prefix}{name}: pos={len(prompt_sets[name]['pos'])}, "
                f"neg={len(prompt_sets[name]['neg'])}"
            )
            cv2.putText(
                canvas,
                count_text,
                (20, y_panel + obj_i * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                colors[name],
                2,
                cv2.LINE_AA,
            )

        for name in names:
            color = colors[name]
            for j, p in enumerate(prompt_sets[name]["pos"]):
                cv2.circle(canvas, p, 6, color, -1)
                cv2.putText(
                    canvas,
                    f"{name}+{j + 1}",
                    (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            for j, p in enumerate(prompt_sets[name]["neg"]):
                cv2.circle(canvas, p, 7, (0, 0, 255), 2)
                cv2.line(canvas, (p[0] - 5, p[1] - 5), (p[0] + 5, p[1] + 5), (0, 0, 255), 2)
                cv2.line(canvas, (p[0] - 5, p[1] + 5), (p[0] + 5, p[1] - 5), (0, 0, 255), 2)
                cv2.putText(
                    canvas,
                    f"{name}-{j + 1}",
                    (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        return canvas

    def on_mouse(event, x, y, flags, param):
        current_name = names[current_idx]
        if event == cv2.EVENT_LBUTTONDOWN:
            prompt_sets[current_name]["pos"].append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            prompt_sets[current_name]["neg"].append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, show_w, show_h)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, redraw())
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("用户取消了 A/B/C 多点初始化。")

        if key in (ord("r"), ord("R")):
            name = names[current_idx]
            prompt_sets[name]["pos"].clear()
            prompt_sets[name]["neg"].clear()

        if key in (8, 127):  # Backspace
            current_idx = max(0, current_idx - 1)

        if key in (13, 10, ord("n"), ord("N")):
            name = names[current_idx]
            if len(prompt_sets[name]["pos"]) == 0:
                logger.warning("%s 至少需要 1 个正点。", name)
                continue
            if current_idx < len(names) - 1:
                current_idx += 1
            else:
                break

    cv2.destroyWindow(window_name)

    def to_original(points_disp: List[Tuple[int, int]]) -> List[Vec2]:
        return [(float(x / scale), float(y / scale)) for x, y in points_disp]

    a_prompt = PointPromptSet(
        positive_points=to_original(prompt_sets["A"]["pos"]),
        negative_points=to_original(prompt_sets["A"]["neg"]),
    )
    b_prompt = PointPromptSet(
        positive_points=to_original(prompt_sets["B"]["pos"]),
        negative_points=to_original(prompt_sets["B"]["neg"]),
    )
    c_prompt = PointPromptSet(
        positive_points=to_original(prompt_sets["C"]["pos"]),
        negative_points=to_original(prompt_sets["C"]["neg"]),
    )

    logger.info(
        "用户多点初始化: A(pos=%d,neg=%d), B(pos=%d,neg=%d), C(pos=%d,neg=%d)",
        len(a_prompt.positive_points),
        len(a_prompt.negative_points),
        len(b_prompt.positive_points),
        len(b_prompt.negative_points),
        len(c_prompt.positive_points),
        len(c_prompt.negative_points),
    )

    return a_prompt, b_prompt, c_prompt


# ============================================================
# 8. A/B/C 构造 SceneGeometry
# ============================================================

def build_scene_from_abc_sam2(
    image_rgb: np.ndarray,
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    c_mask: np.ndarray,
    cfg: RuntimeConfig,
) -> SceneGeometry:
    # --------------------------------------------------------
    # A：来自 SAM2
    # --------------------------------------------------------
    a_contour, a_area, a_center = mask_to_largest_contour_polygon(
        a_mask,
        min_area=cfg.min_mask_area_px,
        epsilon_ratio=cfg.contour_approx_epsilon_ratio,
    )

    flake_a = FlakeGeometry(
        name="A",
        class_name="A",
        confidence=1.0,
        bbox_xyxy=mask_to_bbox_xyxy(a_mask),
        mask=a_mask,
        contour=a_contour,
        center=a_center,
        area=a_area,
        source="sam2",
    )

    # --------------------------------------------------------
    # B：来自 SAM2
    # --------------------------------------------------------
    b_contour, b_area, b_center = mask_to_largest_contour_polygon(
        b_mask,
        min_area=cfg.min_mask_area_px,
        epsilon_ratio=cfg.contour_approx_epsilon_ratio,
    )

    flake_b = FlakeGeometry(
        name="B",
        class_name="B",
        confidence=1.0,
        bbox_xyxy=mask_to_bbox_xyxy(b_mask),
        mask=b_mask,
        contour=b_contour,
        center=b_center,
        area=b_area,
        source="sam2",
    )

    # --------------------------------------------------------
    # C：来自 SAM2
    # --------------------------------------------------------
    c_contour, c_area, c_center = mask_to_largest_contour_polygon(
        c_mask,
        min_area=cfg.min_mask_area_px,
        epsilon_ratio=cfg.contour_approx_epsilon_ratio,
    )

    flake_c = FlakeGeometry(
        name="C",
        class_name="C",
        confidence=1.0,
        bbox_xyxy=mask_to_bbox_xyxy(c_mask),
        mask=c_mask,
        contour=c_contour,
        center=c_center,
        area=c_area,
        source="sam2",
    )

    b_edge = choose_b_target_edge(
        flake_b.contour,
        mode=cfg.b_target_edge_mode,
        reference_point=flake_a.center,
    )

    return SceneGeometry(
        a=flake_a,
        b=flake_b,
        c=flake_c,
        b_target_edge=b_edge,
        timestamp=time.time(),
        image_shape=image_rgb.shape,
    )


# ============================================================
# 9. 实际闭环控制器
# ============================================================

class ActualNanoBoundaryFollower:
    """
    实际实验闭环：

        固定区域截图
            ↓
        第一帧人工点击 A/B/C，SAM2 分割并绑定
            ↓
        后续帧 SAM2 用上一帧 bbox+center 更新 A/B/C
            ↓
        构造 SceneGeometry
            ↓
        rule_policy 输出动作
            ↓
        Stage34 控制 A 运动
    """

    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg

        base_output_dir = Path(cfg.output_dir)
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
        self.output_dir = base_output_dir / run_name
        suffix = 1
        while self.output_dir.exists():
            self.output_dir = base_output_dir / f"{run_name}_{suffix:02d}"
            suffix += 1
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_dir = self.output_dir / "annotated_frames"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "actual_nano_boundary_log.csv"

        self.capturer = FixedRegionScreenCapture(
            capture_area=cfg.capture_area,
            output_dir=self.output_dir / "captured_frames",
            save_image=False,
        )

        self.abc_segmenter = SAM2ABCSegmenter(
            sam2_cfg=cfg.sam2_cfg,
            sam2_checkpoint=cfg.sam2_checkpoint,
            device=cfg.sam2_device,
            min_area_px=cfg.sam2_min_area_px,
            area_ratio_min=cfg.sam2_area_ratio_min,
            area_ratio_max=cfg.sam2_area_ratio_max,
            center_jump_max_px=cfg.sam2_center_jump_max_px,
            use_last_mask_when_failed=cfg.use_last_mask_when_sam2_failed,
        )

        self.stage: Optional[Stage34] = None

        if cfg.enable_stage:
            self.stage = Stage34(
                conn=cfg.stage_conn,
                x_channel=cfg.stage_x_channel,
                y_channel=cfg.stage_y_channel,
                default_velocity=cfg.stage_default_velocity,
                default_acceleration=cfg.stage_default_acceleration,
                default_max_voltage=cfg.stage_default_max_voltage,
                step_x=cfg.stage_step_x,
                step_y=cfg.stage_step_y,
                x_sign=cfg.stage_x_sign,
                y_sign=cfg.stage_y_sign,
                auto_enable=True,
            )
            logger.info("Stage34 已连接。")
        else:
            logger.warning("enable_stage=False：当前只计算动作，不实际移动。")

        self.history: List[Dict[str, Any]] = []
        self.cycle_index = 0
        self.lost_count = 0

    # --------------------------------------------------------
    # 初始化 A/B/C
    # --------------------------------------------------------

    def initialize_abc_with_first_frame(self) -> None:
        logger.info("开始第一帧 A/B/C 点击初始化。")

        image_rgb = self.capturer.capture()

        a_prompt, b_prompt, c_prompt = select_a_b_c_points_interactively(
            image_rgb=image_rgb,
            window_name="SAM2 init: multi positive/negative points for A/B/C",
            scale=self.cfg.init_window_scale,
        )

        a_mask, b_mask, c_mask = self.abc_segmenter.initialize_with_points(
            image_rgb=image_rgb,
            a_positive_points=a_prompt.positive_points,
            a_negative_points=a_prompt.negative_points,
            b_positive_points=b_prompt.positive_points,
            b_negative_points=b_prompt.negative_points,
            c_positive_points=c_prompt.positive_points,
            c_negative_points=c_prompt.negative_points,
        )

        # 初始化图保存，便于检查 A/B/C 是否分割正确
        if self.cfg.save_annotated_image:
            init_path = self.image_dir / "sam2_init_ABC.png"
            self.save_abc_init_image(
                image_rgb=image_rgb,
                a_mask=a_mask,
                b_mask=b_mask,
                c_mask=c_mask,
                a_prompt=a_prompt,
                b_prompt=b_prompt,
                c_prompt=c_prompt,
                save_path=init_path,
            )
            logger.info("A/B/C 初始化检查图已保存: %s", init_path)

    def save_abc_init_image(
        self,
        image_rgb: np.ndarray,
        a_mask: np.ndarray,
        b_mask: np.ndarray,
        c_mask: np.ndarray,
        a_prompt: PointPromptSet,
        b_prompt: PointPromptSet,
        c_prompt: PointPromptSet,
        save_path: Path,
    ) -> None:
        img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        overlay = img.copy()
        overlay[a_mask] = (0, 255, 255)
        overlay[b_mask] = (0, 128, 255)
        overlay[c_mask] = (255, 255, 0)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

        def draw_prompt_set(prompt: PointPromptSet, name: str, color: Tuple[int, int, int]) -> None:
            for i, p in enumerate(prompt.positive_points):
                cv2.circle(img, (int(p[0]), int(p[1])), 6, color, -1)
                cv2.putText(
                    img,
                    f"{name}+{i + 1}",
                    (int(p[0]) + 8, int(p[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            for i, p in enumerate(prompt.negative_points):
                x, y = int(p[0]), int(p[1])
                cv2.circle(img, (x, y), 7, (0, 0, 255), 2)
                cv2.line(img, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
                cv2.line(img, (x - 5, y + 5), (x + 5, y - 5), (0, 0, 255), 2)
                cv2.putText(
                    img,
                    f"{name}-{i + 1}",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        draw_prompt_set(a_prompt, "A", (0, 255, 255))
        draw_prompt_set(b_prompt, "B", (0, 128, 255))
        draw_prompt_set(c_prompt, "C", (255, 255, 0))

        cv2.imwrite(str(save_path), img)

    # --------------------------------------------------------
    # 几何状态
    # --------------------------------------------------------

    def get_boundary_rule_status(self, scene: SceneGeometry) -> Dict[str, Any]:
        """
        新规则状态：
            1. 计算 A-C 边界距离，用于保持 A 与 C 既不太近也不太远；
            2. 计算 A 与 B 的 mask 覆盖面积。只要覆盖面积超过阈值，Stage 停止；
            3. 持续计算 B 目标边角度，方便主程序/日志监测。

        注意：新规则不再根据 A-B 距离决定靠近或远离 B。
        """
        ac = compute_point_to_polygon_boundary_contact(
            p=scene.a.center,
            poly_points=scene.c.contour,
            inside_collision_check=self.cfg.a_c_inside_collision_check,
        )

        b1, b2 = scene.b_target_edge
        ab_contact = compute_point_to_segment_contact(scene.a.center, b1, b2)

        c_clearance = float(ac["effective_dist"])
        ab_dist = float(ab_contact["dist"])
        ab_overlap_area = mask_overlap_area(scene.a.mask, scene.b.mask)
        ab_overlap = bool(
            self.cfg.a_b_overlap_stop
            and ab_overlap_area >= float(self.cfg.a_b_overlap_min_area_px)
        )
        b_edge_angle = edge_angle_deg(scene.b_target_edge)

        if ac.get("collision_like", False) or c_clearance < self.cfg.a_c_min_clearance:
            ac_status = "ac_too_near"
        elif c_clearance > self.cfg.a_c_max_clearance:
            ac_status = "ac_too_far"
        else:
            ac_status = "ac_ok"

        if ab_overlap:
            ab_status = "ab_overlap_stop"
        else:
            ab_status = "ab_no_overlap"

        return {
            "ac": ac,
            "ab_contact": ab_contact,
            "ab_dist": ab_dist,
            "ab_overlap_area": ab_overlap_area,
            "ab_overlap": ab_overlap,
            "b_edge_angle_deg": b_edge_angle,
            "c_clearance": c_clearance,
            "ac_status": ac_status,
            "ab_status": ab_status,
            "contact_valid": ac_status == "ac_ok" and not ab_overlap,
            "distance_status": f"{ac_status}|{ab_status}",
        }

    def rule_policy(self, scene: SceneGeometry) -> int:
        """
        新运动规则：
            - 不考虑 A-B 距离；
            - 若 A 与 B 的 mask 有覆盖，Stage 不动，只监测 B 边角度；
            - 若没有覆盖，A 沿 C 的局部切向运动一圈；
            - 同时用法向修正维持 A-C 距离在 [min, max] 附近。
        """
        status = self.get_boundary_rule_status(scene)

        if status.get("ab_overlap", False):
            return 0

        ac = status["ac"]
        c_clearance = status["c_clearance"]

        c_edge_dir = ac["c_edge_dir"]
        away_from_c = ac["boundary_to_a"]
        toward_c = vec_mul(away_from_c, -1.0)
        tangent = vec_mul(c_edge_dir, float(self.cfg.follow_c_direction))

        def horizontal_away_from_c() -> Vec2:
            """
            按 A 与 C 中心的左右关系给出远离 C 的水平修正方向。
            图像坐标中 x 增大为向右：
                A 在 C 左侧 -> 远离 C = LEFT；靠近 C = RIGHT。
                A 在 C 右侧 -> 远离 C = RIGHT；靠近 C = LEFT。
            """
            dx = float(scene.a.center[0] - scene.c.center[0])
            if abs(dx) < 1e-6:
                return away_from_c
            return (-1.0, 0.0) if dx < 0 else (1.0, 0.0)

        correction_mode = str(getattr(self.cfg, "a_c_correction_mode", "horizontal_from_c_center")).lower().strip()
        if correction_mode == "horizontal_from_c_center":
            ac_away_vec = horizontal_away_from_c()
        elif correction_mode == "auto":
            dx = abs(float(scene.a.center[0] - scene.c.center[0]))
            ac_away_vec = horizontal_away_from_c() if dx > 2.0 else away_from_c
        else:
            ac_away_vec = away_from_c
        ac_toward_vec = vec_mul(ac_away_vec, -1.0)

        # A-C 防碰撞：最高优先级。
        # 注意：这里不再直接使用最近边界法向 away_from_c，
        # 因为在你的画面中最近边界可能是 C 的上下边，导致策略输出 UP/DOWN；
        # 当前默认使用 A/C 中心水平关系，所以 A 在 C 左侧且太近时会输出 LEFT。
        if ac.get("collision_like", False) or c_clearance < self.cfg.a_c_min_clearance:
            return vector_to_action(ac_away_vec)

        # A-C 太远：靠近 C 边界
        if c_clearance > self.cfg.a_c_max_clearance:
            return vector_to_action(ac_toward_vec)

        # 距离合适：沿 C 切向运动，同时按 target clearance 做小修正。
        desired = vec_mul(tangent, self.cfg.boundary_tangent_weight)
        clearance_error = c_clearance - self.cfg.a_c_target_clearance

        if clearance_error > 0:
            # 当前比目标远，稍微靠近 C
            desired = vec_add(
                desired,
                vec_mul(ac_toward_vec, self.cfg.boundary_clearance_correction_weight),
            )
        elif clearance_error < 0:
            # 当前比目标近，稍微远离 C
            desired = vec_add(
                desired,
                vec_mul(ac_away_vec, self.cfg.boundary_clearance_correction_weight),
            )

        return vector_to_action(desired)

    # --------------------------------------------------------
    # Stage34 执行动作
    # --------------------------------------------------------

    def execute_action(self, action_id: int) -> None:
        direction = ACTION_TO_DIRECTION[action_id]

        if action_id == 0:
            logger.info("[Stage34] action=STAY，不移动。")
            return

        if self.stage is None:
            logger.info(
                "[DRY-RUN] action=%s, direction=%s, step=%d",
                ACTION_NAMES[action_id],
                direction,
                self.cfg.action_step,
            )
            return

        logger.info(
            "[Stage34] 执行动作: action=%s, direction=%s, step=%d",
            ACTION_NAMES[action_id],
            direction,
            self.cfg.action_step,
        )

        self.stage.execute_rule_action(
            action_id=action_id,
            direction=direction,
            step=self.cfg.action_step,
            wait=self.cfg.wait_stage_move,
            timeout=self.cfg.stage_move_timeout_s,
        )

    # --------------------------------------------------------
    # 单轮闭环
    # --------------------------------------------------------

    def capture_and_build_scene(self) -> Tuple[np.ndarray, SceneGeometry]:
        image_rgb = self.capturer.capture()

        a_mask, b_mask, c_mask = self.abc_segmenter.infer_abc(image_rgb)

        scene = build_scene_from_abc_sam2(
            image_rgb=image_rgb,
            a_mask=a_mask,
            b_mask=b_mask,
            c_mask=c_mask,
            cfg=self.cfg,
        )

        return image_rgb, scene

    def run_one_cycle(self) -> bool:
        self.cycle_index += 1

        try:
            image_rgb, scene = self.capture_and_build_scene()
            self.lost_count = 0

        except Exception as e:
            self.lost_count += 1

            logger.error(
                "[Cycle %04d] 视觉检测/分割失败: %s, lost_count=%d/%d",
                self.cycle_index,
                e,
                self.lost_count,
                self.cfg.max_lost_frames,
            )

            self.record_lost_frame(error=str(e))

            try:
                if self.stage is not None:
                    self.stage.stop_all()
            except Exception:
                pass

            if self.lost_count >= self.cfg.max_lost_frames:
                logger.error("连续视觉失败次数过多，停止闭环。")
                return False

            return True

        status = self.get_boundary_rule_status(scene)
        action_id = self.rule_policy(scene)
        action_name = ACTION_NAMES[action_id]
        direction = ACTION_TO_DIRECTION[action_id]

        logger.info(
            "[Cycle %04d] "
            "A=(%.1f, %.1f), B=(%.1f, %.1f), C=(%.1f, %.1f), "
            "A-C=%.2f[%s], AB_overlap=%.1f[%s], B_edge_angle=%s, action=%s",
            self.cycle_index,
            scene.a.center[0],
            scene.a.center[1],
            scene.b.center[0],
            scene.b.center[1],
            scene.c.center[0],
            scene.c.center[1],
            status["c_clearance"],
            status["ac_status"],
            status["ab_overlap_area"],
            status["ab_status"],
            status.get("b_edge_angle_deg"),
            action_name,
        )

        if self.cfg.save_annotated_image:
            self.save_annotated_image(
                image_rgb=image_rgb,
                scene=scene,
                status=status,
                action_name=action_name,
            )

        self.record_cycle(
            scene=scene,
            status=status,
            action_id=action_id,
            action_name=action_name,
            direction=direction,
        )

        # A/B 覆盖时：立即停止控制器，不发送运动，只持续检测角度。
        if status.get("ab_overlap", False):
            try:
                if self.stage is not None:
                    self.stage.stop_all()
            except Exception:
                pass
            logger.info("[Cycle %04d] A/B 发生覆盖，Stage停止，只检测 B 角度。", self.cycle_index)
            return True

        self.execute_action(action_id)

        return True

    # --------------------------------------------------------
    # 保存日志
    # --------------------------------------------------------

    def record_lost_frame(self, error: str) -> None:
        self.history.append(
            {
                "time": time.time(),
                "cycle": self.cycle_index,
                "ok": False,
                "error": error,
                "action_id": None,
                "action_name": None,
                "direction": None,
            }
        )

        if self.cfg.save_csv:
            self.flush_csv()

    def record_cycle(
        self,
        scene: SceneGeometry,
        status: Dict[str, Any],
        action_id: int,
        action_name: str,
        direction: str,
    ) -> None:
        ac = status["ac"]
        ab_contact = status["ab_contact"]
        b1, b2 = scene.b_target_edge

        stage_x, stage_y = (None, None)

        if self.stage is not None:
            try:
                stage_x, stage_y = self.stage.get_position()
            except Exception:
                stage_x, stage_y = (None, None)

        row = {
            "time": time.time(),
            "cycle": self.cycle_index,
            "ok": True,

            "a_x": scene.a.center[0],
            "a_y": scene.a.center[1],
            "a_conf": scene.a.confidence,
            "a_area": scene.a.area,
            "a_source": scene.a.source,

            "b_x": scene.b.center[0],
            "b_y": scene.b.center[1],
            "b_conf": scene.b.confidence,
            "b_area": scene.b.area,
            "b_source": scene.b.source,

            "c_x": scene.c.center[0],
            "c_y": scene.c.center[1],
            "c_conf": scene.c.confidence,
            "c_area": scene.c.area,
            "c_source": scene.c.source,

            "b_edge_x1": b1[0],
            "b_edge_y1": b1[1],
            "b_edge_x2": b2[0],
            "b_edge_y2": b2[1],

            "ac_clearance": status["c_clearance"],
            "ac_raw_dist": ac["dist"],
            "ac_collision_like": ac.get("collision_like", False),
            "ac_status": status["ac_status"],

            "ab_edge_dist": status["ab_dist"],
            "ab_overlap_area": status.get("ab_overlap_area"),
            "ab_overlap": status.get("ab_overlap"),
            "b_edge_angle_deg": status.get("b_edge_angle_deg"),
            "ab_status": status["ab_status"],

            "ac_c_contact_x": ac["c_contact"][0],
            "ac_c_contact_y": ac["c_contact"][1],
            "ab_b_contact_x": ab_contact["b_contact"][0],
            "ab_b_contact_y": ab_contact["b_contact"][1],

            "distance_status": status["distance_status"],
            "contact_valid": status["contact_valid"],

            "action_id": action_id,
            "action_name": action_name,
            "direction": direction,

            "stage_x": stage_x,
            "stage_y": stage_y,
        }

        self.history.append(row)

        if self.cfg.save_csv:
            self.flush_csv()

    def flush_csv(self) -> None:
        if not self.history:
            return

        fieldnames = sorted(
            set(k for row in self.history for k in row.keys())
        )

        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    # --------------------------------------------------------
    # 保存标注图
    # --------------------------------------------------------

    def save_annotated_image(
        self,
        image_rgb: np.ndarray,
        scene: SceneGeometry,
        status: Dict[str, Any],
        action_name: str,
    ) -> None:
        img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # mask overlay
        overlay = img.copy()

        overlay[scene.a.mask] = (0, 255, 255)
        overlay[scene.b.mask] = (0, 128, 255)
        overlay[scene.c.mask] = (255, 255, 0)

        img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)

        def draw_poly(
            points: List[Vec2],
            color: Tuple[int, int, int],
            thickness: int = 2,
        ) -> None:
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                img,
                [pts],
                isClosed=True,
                color=color,
                thickness=thickness,
            )

        def draw_center(
            p: Vec2,
            color: Tuple[int, int, int],
            label: str,
        ) -> None:
            cv2.circle(img, (int(p[0]), int(p[1])), 5, color, -1)
            cv2.putText(
                img,
                label,
                (int(p[0]) + 6, int(p[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        draw_poly(scene.a.contour, (0, 255, 255), 2)
        draw_poly(scene.b.contour, (0, 128, 255), 2)
        draw_poly(scene.c.contour, (255, 255, 0), 2)

        draw_center(scene.a.center, (0, 255, 255), "A-SAM2")
        draw_center(scene.b.center, (0, 128, 255), "B-SAM2")
        draw_center(scene.c.center, (255, 255, 0), "C-SAM2")

        # B 目标边
        b1, b2 = scene.b_target_edge
        cv2.line(
            img,
            (int(b1[0]), int(b1[1])),
            (int(b2[0]), int(b2[1])),
            (0, 0, 255),
            4,
        )

        # A-C 最近距离线
        ac = status["ac"]
        cv2.line(
            img,
            (int(scene.a.center[0]), int(scene.a.center[1])),
            (int(ac["c_contact"][0]), int(ac["c_contact"][1])),
            (0, 255, 0) if status["ac_status"] == "ac_ok" else (0, 0, 255),
            2,
        )

        # A-B 最近距离线
        ab = status["ab_contact"]
        cv2.line(
            img,
            (int(scene.a.center[0]), int(scene.a.center[1])),
            (int(ab["b_contact"][0]), int(ab["b_contact"][1])),
            (0, 0, 255) if status.get("ab_overlap", False) else (255, 0, 255),
            2,
        )

        text_lines = [
            f"cycle={self.cycle_index}",
            f"A-C={status['c_clearance']:.2f} [{status['ac_status']}]",
            f"AB_overlap={status.get('ab_overlap_area', 0):.1f} [{status['ab_status']}]",
            f"B_edge_angle={status.get('b_edge_angle_deg')}",
            f"action={action_name}",
            f"A={scene.a.source}, B={scene.b.source}, C={scene.c.source}",
        ]

        x0, y0 = 15, 25
        for i, text in enumerate(text_lines):
            cv2.putText(
                img,
                text,
                (x0, y0 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        timestamp_ms = int(time.time() * 1000)
        safe_action = str(action_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
        save_path = self.image_dir / f"cycle_{self.cycle_index:06d}_{timestamp_ms}_{safe_action}.png"
        cv2.imwrite(str(save_path), img)

    # --------------------------------------------------------
    # 主循环
    # --------------------------------------------------------

    def run(self) -> None:
        logger.info("开始实际纳米片边界跟随闭环。")

        try:
            self.initialize_abc_with_first_frame()

            for _ in range(self.cfg.max_cycles):
                ok = self.run_one_cycle()

                if not ok:
                    break

                time.sleep(self.cfg.loop_interval_s)

        except KeyboardInterrupt:
            logger.warning("用户中断。")

        finally:
            self.close()

    def close(self) -> None:
        logger.info("准备关闭实际闭环程序。")

        if self.cfg.save_csv:
            self.flush_csv()
            logger.info("CSV 已保存: %s", self.csv_path)

        if self.stage is not None:
            try:
                self.stage.stop_all()
            except Exception:
                pass

            try:
                self.stage.close()
            except Exception:
                pass

            logger.info("Stage34 已关闭。")


# ============================================================
# 10. main
# ============================================================

def main():
    cfg = RuntimeConfig(
        # ----------------------------------------------------
        # 固定区域截图
        # ----------------------------------------------------
        capture_area=(116, 98, 1112, 886),

        # ----------------------------------------------------
        # YOLO 旧配置字段。当前主流程不使用 YOLO。
        # ----------------------------------------------------
        yolo_a_model_path="vision/best_wan12.2.pt",
        class_name_a="Nano",
        yolo_conf_threshold=0.5,

        # ----------------------------------------------------
        # SAM2 配置和权重路径
        # 需要改成你的实际路径。
        # ----------------------------------------------------
        sam2_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_checkpoint=str(SAM2_REPO_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"),
        sam2_device="cuda",

        # 第一帧手动点击 A、B、C
        init_window_scale=0.85,

        # B 目标边选择
        b_target_edge_mode="nearest",

        # A-C 距离控制
        a_c_target_clearance=90.0,
        a_c_min_clearance=70.0,
        a_c_max_clearance=90.0,

        # A-B 距离控制
        a_b_min_dist=25.0,
        a_b_max_dist=50.0,

        # C 边界跟随方向
        follow_c_direction=1,

        # Stage34 参数
        stage_conn="27267878",
        stage_x_channel=3,
        stage_y_channel=4,
        stage_default_velocity=5.0,
        stage_default_acceleration=5.0,
        stage_default_max_voltage=100,
        stage_step_x=100,
        stage_step_y=100,
        stage_x_sign=1,
        stage_y_sign=1,

        # 每次规则动作的位移步数
        action_step=100,

        # 调试时先 False，只看保存图片和日志；
        # 确认方向正确后再改 True。
        enable_stage=False,

        max_cycles=500,
        loop_interval_s=0.15,
    )

    follower = ActualNanoBoundaryFollower(cfg)
    follower.run()


if __name__ == "__main__":
    main()
