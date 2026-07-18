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
    # 第一帧 SAM2 分割结果确认
    # --------------------------------------------------------
    # True：用户确认第一帧 A/B/C 分割结果后，才允许执行运动。
    # False：兼容旧流程，SAM2 初始化后直接进入运动。
    confirm_first_frame_segmentation: bool = True

    # 确认窗口缩放比例
    confirm_window_scale: float = 0.85

    # --------------------------------------------------------
    # C 轮廓拟合方式
    # --------------------------------------------------------
    # "quadrilateral"：将 C 的 SAM2 mask 再拟合成四边形，推荐。
    # "contour"：保留原来的 SAM2 多边形轮廓。
    c_contour_mode: str = "quadrilateral"

    # C 四边形拟合方式：
    # "min_area_rect"：最稳定，一定输出 4 个点，推荐。
    # "approx_first"：先 approxPolyDP，若不是 4 点再退回 minAreaRect。
    c_quadrilateral_method: str = "min_area_rect"

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


def order_quad_points_clockwise(points: np.ndarray) -> List[Vec2]:
    """
    将 4 个点按顺时针顺序排列，保证 polygon_edges() 得到稳定的四条边。
    图像坐标中 y 向下，排序只用于边连接，不改变几何含义。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]
    return [(float(x), float(y)) for x, y in ordered]


def mask_to_quadrilateral_contour(
    mask_bool: np.ndarray,
    min_area: float = 50.0,
    epsilon_ratio: float = 0.02,
    method: str = "min_area_rect",
) -> Tuple[List[Vec2], float, Vec2]:
    """
    将 SAM2 得到的 C mask 再拟合成四边形轮廓。

    method="min_area_rect"：
        对 SAM2 边缘毛刺、不规则凹凸更稳定；一定输出 4 个点。
    method="approx_first"：
        先用 approxPolyDP 尝试得到真实四边形；如果不是 4 个点，
        自动退回 minAreaRect。
    """
    mask_u8 = (mask_bool.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError("C mask 中没有找到有效轮廓，无法拟合四边形。")

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))

    if area < min_area:
        raise RuntimeError(f"C mask 轮廓面积过小，无法拟合四边形: area={area:.2f}")

    method = str(method).lower().strip()

    if method == "approx_first":
        peri = float(cv2.arcLength(contour, True))
        epsilon = max(1.0, float(epsilon_ratio) * peri)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            pts = order_quad_points_clockwise(approx.reshape(4, 2))
            m = cv2.moments(contour)
            if abs(m["m00"]) < 1e-8:
                center = polygon_centroid_simple(pts)
            else:
                center = (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))
            return pts, area, center

    # 默认/兜底：最小外接旋转矩形。
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    pts = order_quad_points_clockwise(box)

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

        # C 固定参考：第一次 C 分割后保存所有 C 像素位置，后续不再更新 C。
        self.static_c_mask: Optional[np.ndarray] = None
        self.static_c_bbox: Optional[Tuple[float, float, float, float]] = None
        self.static_c_center: Optional[Vec2] = None
        self.static_c_pixel_yx: Optional[np.ndarray] = None

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
        初始化 A/B/C。

        按当前实验逻辑：
            1. A 只用一个正点分割，不使用负点，也不使用多个正点；
            2. B 只用一个正点分割，不使用负点，也不使用多个正点；
            3. C 保留多个正点 + 多个负点分割；
            4. 第一次 C 分割得到的 mask 会被固定保存，后续运动控制始终使用这个
               初始 C mask 的所有像素位置，不再逐帧重新分割/更新 C。
        """
        # A/B：只保留第一个正点，彻底忽略 A/B 的负点和多正点。
        if a_point is None:
            if a_positive_points:
                a_point = a_positive_points[0]
            else:
                raise ValueError("A 需要 1 个正点。")

        if b_point is None:
            if b_positive_points:
                b_point = b_positive_points[0]
            else:
                raise ValueError("B 需要 1 个正点。")

        # C：保留多个正点；如果旧调用只传 c_point，则退化为 C 单正点。
        if c_positive_points is None:
            if c_point is None:
                raise ValueError("C 至少需要 1 个正点。")
            c_positive_points = [c_point]

        if len(c_positive_points) <= 0:
            raise ValueError("C 至少需要 1 个正点。")

        self.a_init_point = a_point
        self.b_init_point = b_point
        self.c_init_point = c_positive_points[0]

        self.a_init_positive_points = [a_point]
        self.a_init_negative_points = []
        self.b_init_positive_points = [b_point]
        self.b_init_negative_points = []
        c_negative_points = list(c_negative_points or [])

        self.c_init_positive_points = list(c_positive_points)
        self.c_init_negative_points = list(c_negative_points)

        self.predictor.set_image(image_rgb)

        # A/B：单正点，无负点。
        a_mask = self._predict_mask_by_points(
            positive_points=[a_point],
            negative_points=None,
        )
        b_mask = self._predict_mask_by_points(
            positive_points=[b_point],
            negative_points=None,
        )

        # C：多个正点 + 多个负点。
        c_mask = self._predict_mask_by_points(
            positive_points=c_positive_points,
            negative_points=c_negative_points,
        )

        self._basic_mask_check("A", a_mask)
        self._basic_mask_check("B", b_mask)
        self._basic_mask_check("C", c_mask)

        self.last_a_mask = a_mask
        self.last_b_mask = b_mask

        self.last_a_bbox = mask_to_bbox_xyxy(a_mask)
        self.last_b_bbox = mask_to_bbox_xyxy(b_mask)

        self.last_a_center = mask_center(a_mask)
        self.last_b_center = mask_center(b_mask)

        # C 固定：记录第一次 C 分割得到的所有像素位置。
        self.last_c_mask = c_mask.copy()
        self.last_c_bbox = mask_to_bbox_xyxy(c_mask)
        self.last_c_center = mask_center(c_mask)

        self.static_c_mask = c_mask.copy()
        self.static_c_bbox = self.last_c_bbox
        self.static_c_center = self.last_c_center
        self.static_c_pixel_yx = np.column_stack(np.where(self.static_c_mask.astype(bool)))

        self.initialized = True

        logger.info(
            "SAM2 初始化完成: "
            "A=single positive point, B=single positive point, "
            "C=%d positive points + %d negative points fixed; "
            "A_center=(%.1f, %.1f), B_center=(%.1f, %.1f), "
            "C_static_center=(%.1f, %.1f), C_static_pixels=%d",
            len(c_positive_points),
            len(c_negative_points),
            self.last_a_center[0],
            self.last_a_center[1],
            self.last_b_center[0],
            self.last_b_center[1],
            self.static_c_center[0],
            self.static_c_center[1],
            int(self.static_c_mask.sum()),
        )

        return a_mask, b_mask, c_mask

    def infer_abc(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        后续帧更新 A/B，但 C 不再重新分割。

        A/B：
            用上一帧 bbox + 单个中心正点更新，且不使用负点。
        C：
            直接返回第一次初始化时保存的 static_c_mask。
            因此 A 沿 C 运动时，所有 A-C 距离、C 边界、C 轮廓都来自第一次记住的 C 像素位置。
        """
        if not self.initialized:
            raise RuntimeError("SAM2ABCSegmenter 尚未初始化。")

        if self.last_a_bbox is None or self.last_b_bbox is None:
            raise RuntimeError("SAM2 A/B 上一帧 bbox 为空，无法继续跟踪。")

        if self.static_c_mask is None:
            raise RuntimeError("static_c_mask 为空：第一次 C 分割像素没有被记录。")

        self.predictor.set_image(image_rgb)

        try:
            # A/B：单中心正点 + 上一帧 bbox，无负点。
            a_mask = self._predict_mask_by_box_and_point(
                box_xyxy=self.last_a_bbox,
                point=self.last_a_center,
                negative_points=None,
            )
            b_mask = self._predict_mask_by_box_and_point(
                box_xyxy=self.last_b_bbox,
                point=self.last_b_center,
                negative_points=None,
            )

            c_mask = self.static_c_mask.copy()

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

            self.last_a_mask = a_mask
            self.last_b_mask = b_mask

            self.last_a_bbox = mask_to_bbox_xyxy(a_mask)
            self.last_b_bbox = mask_to_bbox_xyxy(b_mask)

            self.last_a_center = mask_center(a_mask)
            self.last_b_center = mask_center(b_mask)

            # C 始终保持第一次分割结果，不更新 bbox/center/mask。
            self.last_c_mask = c_mask
            self.last_c_bbox = self.static_c_bbox
            self.last_c_center = self.static_c_center

            return a_mask, b_mask, c_mask

        except Exception as e:
            logger.error("SAM2 A/B 更新失败: %s", e)

            if (
                self.use_last_mask_when_failed
                and self.last_a_mask is not None
                and self.last_b_mask is not None
                and self.static_c_mask is not None
            ):
                logger.warning("使用上一帧 A/B mask + 初始固定 C mask 兜底。")
                return self.last_a_mask, self.last_b_mask, self.static_c_mask.copy()

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
    window_name: str = "SAM2 init: A/B single point, C multi positive + negative",
    scale: float = 0.85,
) -> Tuple[PointPromptSet, PointPromptSet, PointPromptSet]:
    """
    第一帧初始化点选择。

    当前规则：
        A：只点击 1 个正点；
        B：只点击 1 个正点；
        C：点击多个正点 + 多个负点；
        A/B 不使用负点，C 使用负点辅助排除非 C 区域。

    操作方式：
        当前目标依次为 A -> B -> C。
        左键：
            - A/B：添加正点，只保留最后一次点击的 1 个点；
            - C：添加 C 正点，可以添加多个。
        右键：
            - A/B：忽略；
            - C：添加 C 负点，可以添加多个，用于排除 A/B/背景等非 C 区域。
        N 或 Enter：完成当前目标，进入下一个目标。
        R：清空当前目标。
        Backspace：回到上一个目标。
        ESC：取消。
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
            "SAM2 init",
            f"Current: {current_name} | Left=positive, Right=C negative only, N/Enter=next, R=reset, Backspace=previous, ESC=cancel",
            "A/B: one positive point only. C: multiple positive points and multiple negative points.",
        ]
        for i, text_line in enumerate(instructions):
            cv2.putText(
                canvas,
                text_line,
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
            if name in ("A", "B"):
                count_text = f"{prefix}{name}: single positive={len(prompt_sets[name]['pos'])}/1"
            else:
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

            # 只显示 C 的负点。A/B 负点模式已删除。
            if name == "C":
                for j, p in enumerate(prompt_sets[name]["neg"]):
                    x, y = int(p[0]), int(p[1])
                    cv2.circle(canvas, (x, y), 7, (0, 0, 255), 2)
                    cv2.line(canvas, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
                    cv2.line(canvas, (x - 5, y + 5), (x + 5, y - 5), (0, 0, 255), 2)
                    cv2.putText(
                        canvas,
                        f"C-{j + 1}",
                        (x + 8, y - 8),
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
            if current_name in ("A", "B"):
                # A/B 删除多正点+负点模式，只保留一个正点。
                prompt_sets[current_name]["pos"] = [(x, y)]
                prompt_sets[current_name]["neg"].clear()
            elif current_name == "C":
                # C 保留多个正点。
                prompt_sets[current_name]["pos"].append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if current_name == "C":
                # C 补充多个负点，用于排除 A/B/背景等非 C 区域。
                prompt_sets[current_name]["neg"].append((x, y))
            else:
                logger.info("A/B 不使用负点；当前右键点击已忽略。")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, show_w, show_h)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, redraw())
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("用户取消了 A/B/C 初始化。")

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

    a_pos = to_original(prompt_sets["A"]["pos"])
    b_pos = to_original(prompt_sets["B"]["pos"])
    c_pos = to_original(prompt_sets["C"]["pos"])
    c_neg = to_original(prompt_sets["C"]["neg"])

    a_prompt = PointPromptSet(
        positive_points=a_pos[:1],
        negative_points=[],
    )
    b_prompt = PointPromptSet(
        positive_points=b_pos[:1],
        negative_points=[],
    )
    c_prompt = PointPromptSet(
        positive_points=c_pos,
        negative_points=c_neg,
    )

    logger.info(
        "用户初始化: A(single_pos=%d), B(single_pos=%d), C(pos=%d, neg=%d, fixed static mask after segmentation)",
        len(a_prompt.positive_points),
        len(b_prompt.positive_points),
        len(c_prompt.positive_points),
        len(c_prompt.negative_points),
    )

    return a_prompt, b_prompt, c_prompt


def confirm_first_frame_segmentation_interactively(
    image_rgb: np.ndarray,
    scene: SceneGeometry,
    window_name: str = "Confirm SAM2 first frame segmentation",
    scale: float = 0.85,
) -> bool:
    """
    显示第一帧 SAM2 分割结果，让用户确认。

    按键：
        Y / Enter：确认，继续运动；
        R：不确认，重新点击 A/B/C；
        ESC / Q：取消程序。
    """
    if scale <= 0:
        scale = 1.0

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img = image_bgr.copy()
    overlay = img.copy()

    # OpenCV 使用 BGR 颜色。
    overlay[scene.a.mask.astype(bool)] = (0, 255, 255)      # A：黄色
    overlay[scene.b.mask.astype(bool)] = (0, 128, 255)      # B：橙色
    overlay[scene.c.mask.astype(bool)] = (255, 255, 0)      # C：青色
    img = cv2.addWeighted(overlay, 0.32, img, 0.68, 0)

    def draw_poly(points: List[Vec2], color: Tuple[int, int, int], thickness: int = 2) -> None:
        if not points:
            return
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    def draw_center(p: Vec2, color: Tuple[int, int, int], label: str) -> None:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, color, -1)
        cv2.putText(
            img,
            label,
            (int(p[0]) + 8, int(p[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    draw_poly(scene.a.contour, (0, 255, 255), 2)
    draw_poly(scene.b.contour, (0, 128, 255), 2)
    # C 的拟合四边形用粗线显示，方便判断四边形是否正确。
    draw_poly(scene.c.contour, (255, 255, 0), 4)

    draw_center(scene.a.center, (0, 255, 255), "A")
    draw_center(scene.b.center, (0, 128, 255), "B")
    draw_center(scene.c.center, (255, 255, 0), "C-quad")

    # B 的目标边用红线显示。
    b1, b2 = scene.b_target_edge
    cv2.line(
        img,
        (int(b1[0]), int(b1[1])),
        (int(b2[0]), int(b2[1])),
        (0, 0, 255),
        4,
    )

    text_lines = [
        "Confirm SAM2 first frame segmentation",
        "Y / Enter = accept and start motion",
        "R = reject and re-click A/B/C",
        "ESC / Q = cancel",
        f"A area={scene.a.area:.1f}, B area={scene.b.area:.1f}, C area={scene.c.area:.1f}",
        f"C source={scene.c.source}, C contour points={len(scene.c.contour)}",
    ]

    for i, line in enumerate(text_lines):
        cv2.putText(
            img,
            line,
            (20, 32 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    h, w = img.shape[:2]
    show_w = max(1, int(w * scale))
    show_h = max(1, int(h * scale))
    display = cv2.resize(img, (show_w, show_h), interpolation=cv2.INTER_AREA)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, show_w, show_h)

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("y"), ord("Y"), 13, 10):
            cv2.destroyWindow(window_name)
            return True

        if key in (ord("r"), ord("R")):
            cv2.destroyWindow(window_name)
            return False

        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(window_name)
            raise RuntimeError("用户取消了第一帧 SAM2 分割确认。")


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
    # C：来自 SAM2。C 一般近似四边形，因此这里可以把 SAM2 mask
    #    再拟合为四边形轮廓，后续 A-C 最近边、C 切向运动都基于
    #    这个四边形，而不是基于毛刺较多的原始 mask 边界。
    # --------------------------------------------------------
    c_contour_mode = str(getattr(cfg, "c_contour_mode", "quadrilateral")).lower().strip()
    if c_contour_mode == "quadrilateral":
        c_contour, c_area, c_center = mask_to_quadrilateral_contour(
            c_mask,
            min_area=cfg.min_mask_area_px,
            epsilon_ratio=max(0.01, float(cfg.contour_approx_epsilon_ratio) * 3.0),
            method=getattr(cfg, "c_quadrilateral_method", "min_area_rect"),
        )
        c_source = "sam2_quad"
    else:
        c_contour, c_area, c_center = mask_to_largest_contour_polygon(
            c_mask,
            min_area=cfg.min_mask_area_px,
            epsilon_ratio=cfg.contour_approx_epsilon_ratio,
        )
        c_source = "sam2_contour"

    flake_c = FlakeGeometry(
        name="C",
        class_name="C",
        confidence=1.0,
        bbox_xyxy=mask_to_bbox_xyxy(c_mask),
        mask=c_mask,
        contour=c_contour,
        center=c_center,
        area=c_area,
        source=c_source,
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
        logger.info(
            "开始第一帧 A/B/C 初始化：A/B 单正点；C 多正点+负点；"
            "SAM2 分割后需要用户确认；确认后才允许运动；C mask 固定；C 轮廓可拟合为四边形。"
        )

        image_rgb = self.capturer.capture()

        while True:
            a_prompt, b_prompt, c_prompt = select_a_b_c_points_interactively(
                image_rgb=image_rgb,
                window_name="SAM2 init: A/B single positive, C multi pos+neg fixed",
                scale=self.cfg.init_window_scale,
            )

            a_mask, b_mask, c_mask = self.abc_segmenter.initialize_with_points(
                image_rgb=image_rgb,
                a_positive_points=a_prompt.positive_points,
                a_negative_points=[],
                b_positive_points=b_prompt.positive_points,
                b_negative_points=[],
                c_positive_points=c_prompt.positive_points,
                c_negative_points=c_prompt.negative_points,
            )

            # 先构造 scene。这里会根据 cfg.c_contour_mode 把 C mask 转成
            # 普通轮廓或四边形轮廓。
            scene = build_scene_from_abc_sam2(
                image_rgb=image_rgb,
                a_mask=a_mask,
                b_mask=b_mask,
                c_mask=c_mask,
                cfg=self.cfg,
            )

            # 保存候选初始化图，便于复查。
            if self.cfg.save_annotated_image:
                init_path = self.image_dir / "sam2_init_candidate_AB_single_C_pos_neg.png"
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
                logger.info("A/B/C 初始化候选检查图已保存: %s", init_path)

            # 用户确认第一帧分割。确认前不会进入运动循环。
            if bool(getattr(self.cfg, "confirm_first_frame_segmentation", True)):
                accepted = confirm_first_frame_segmentation_interactively(
                    image_rgb=image_rgb,
                    scene=scene,
                    window_name="Confirm SAM2 first frame segmentation",
                    scale=float(getattr(self.cfg, "confirm_window_scale", self.cfg.init_window_scale)),
                )
                if not accepted:
                    logger.warning("用户拒绝当前 SAM2 第一帧分割结果，重新点击 A/B/C。")
                    continue

            # 只有用户确认后，才保存第一次 C 分割得到的完整像素位置。
            # 后续 C 作为固定参考，不再被新的 SAM2 结果覆盖。
            self.save_static_c_reference(c_mask)

            # 保存确认后的叠加图。这里用 save_annotated_image() 会同时画出 C 四边形轮廓。
            if self.cfg.save_annotated_image:
                status = self.get_boundary_rule_status(scene)
                self.save_annotated_image(
                    image_rgb=image_rgb,
                    scene=scene,
                    status=status,
                    action_name="INIT_CONFIRMED_NO_MOTION",
                )

            logger.info(
                "第一帧 SAM2 分割已确认：A/B/C 绑定完成；C source=%s, C contour points=%d；现在允许进入运动循环。",
                scene.c.source,
                len(scene.c.contour),
            )
            break

    def save_static_c_reference(self, c_mask: np.ndarray) -> None:
        """
        保存第一次 C 分割得到的固定像素位置。

        保存内容：
            1. static_c_mask.png：C 初始 mask；
            2. static_c_pixels_yx.npy：所有 C 像素坐标，格式为 [y, x]；
            3. static_c_reference.csv：所有 C 像素坐标表。
        """
        static_dir = self.output_dir / "static_c_reference"
        static_dir.mkdir(parents=True, exist_ok=True)

        c_bool = c_mask.astype(bool)
        yx = np.column_stack(np.where(c_bool))

        mask_path = static_dir / "static_c_mask.png"
        npy_path = static_dir / "static_c_pixels_yx.npy"
        csv_path = static_dir / "static_c_reference.csv"

        cv2.imwrite(str(mask_path), (c_bool.astype(np.uint8) * 255))
        np.save(str(npy_path), yx.astype(np.int32))

        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "y", "x"])
            for idx, (y, x) in enumerate(yx):
                writer.writerow([idx, int(y), int(x)])

        logger.info(
            "固定 C 参考已保存: mask=%s, pixels=%s, csv=%s, count=%d",
            mask_path,
            npy_path,
            csv_path,
            int(len(yx)),
        )

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

            1. 不考虑 A-B 距离；
            2. 如果 A 与 B 的 mask 有覆盖：
                  Stage 不动，只监测 B 边角度；
            3. 如果 A-C 太近：
                  A 远离 C；
            4. 如果 A-C 太远：
                  A 靠近 C；
            5. 如果 A-C OK：
                  A 沿 C 的最近边切向运动。

        说明：
            C 的轮廓可以先由 SAM2 mask 拟合成四边形；因此这里的
            ac["c_edge_dir"] 是 A 当前最近的那条 C 四边形边的切向方向。
            follow_c_direction=1 / -1 用来反转沿边方向。

        判据：
            A-C < a_c_min_clearance       -> 远离 C
            A-C > a_c_max_clearance       -> 靠近 C
            a_c_min <= A-C <= a_c_max     -> 沿 C 最近边切向运动
        """
        status = self.get_boundary_rule_status(scene)

        # --------------------------------------------------------
        # 1. A/B 覆盖：停止运动，只检测 B 边角度
        # --------------------------------------------------------
        if status.get("ab_overlap", False):
            return 0

        ac = status["ac"]
        c_clearance = float(status["c_clearance"])

        away_from_c = ac["boundary_to_a"]

        def horizontal_away_from_c() -> Vec2:
            """
            按 A 与 C 中心的左右关系给出远离 C 的水平修正方向。

            图像坐标中 x 增大为向右：
                A 在 C 左侧 -> 远离 C = LEFT；靠近 C = RIGHT
                A 在 C 右侧 -> 远离 C = RIGHT；靠近 C = LEFT
            """
            dx = float(scene.a.center[0] - scene.c.center[0])
            if abs(dx) < 1e-6:
                return away_from_c
            return (-1.0, 0.0) if dx < 0 else (1.0, 0.0)

        def b_edge_normal_for_ac_ok() -> Vec2:
            """
            AC_OK 时的运动方向：B 相对于 A 最近边的垂直方向。

            b_target_edge 已经由 choose_b_target_edge(..., reference_point=A.center)
            选择为 B 相对于 A 的目标边/最近边。

            对边方向 t=(tx, ty)，其两个垂直方向为：
                n1=(-ty, tx), n2=(ty, -tx)

            这里用 follow_c_direction 控制取 n1 还是 n2：
                follow_c_direction= 1 -> n1
                follow_c_direction=-1 -> n2

            如果后续发现 AC_OK 时应该向右而不是向左，
            只需要把 GUI/配置中的 follow_c_direction 改成 -1。
            """
            b1, b2 = scene.b_target_edge
            t = vec_norm(vec_sub(b2, b1))

            if vec_len(t) < 1e-8:
                # 兜底：如果 B 边无效，就沿 A 指向 B 最近边的方向运动。
                ab_contact = status.get("ab_contact") or {}
                b_contact = ab_contact.get("b_contact", scene.b.center)
                v = vec_norm(vec_sub(b_contact, scene.a.center))
                return v if vec_len(v) >= 1e-8 else (-1.0, 0.0)

            if int(getattr(self.cfg, "follow_c_direction", 1)) >= 0:
                n = (-t[1], t[0])
            else:
                n = (t[1], -t[0])

            return vec_norm(n)

        correction_mode = str(
            getattr(self.cfg, "a_c_correction_mode", "horizontal_from_c_center")
        ).lower().strip()

        if correction_mode == "horizontal_from_c_center":
            ac_away_vec = horizontal_away_from_c()
        elif correction_mode == "auto":
            dx = abs(float(scene.a.center[0] - scene.c.center[0]))
            ac_away_vec = horizontal_away_from_c() if dx > 2.0 else away_from_c
        else:
            ac_away_vec = away_from_c

        ac_toward_vec = vec_mul(ac_away_vec, -1.0)

        # --------------------------------------------------------
        # 2. A-C 太近：远离 C
        # --------------------------------------------------------
        if ac.get("collision_like", False) or c_clearance < self.cfg.a_c_min_clearance:
            return vector_to_action(ac_away_vec)

        # --------------------------------------------------------
        # 3. A-C 太远：靠近 C
        # --------------------------------------------------------
        if c_clearance > self.cfg.a_c_max_clearance:
            return vector_to_action(ac_toward_vec)

        # --------------------------------------------------------
        # 4. A-C OK：沿 C 最近边的切向运动。
        #    如果 C contour_mode="quadrilateral"，这里就是沿 C 四边形的
        #    当前最近边运动；如果方向反了，修改 follow_c_direction 即可。
        # --------------------------------------------------------
        c_tangent = ac.get("c_edge_dir", (0.0, 0.0))
        if int(getattr(self.cfg, "follow_c_direction", 1)) < 0:
            c_tangent = vec_mul(c_tangent, -1.0)
        if vec_len(c_tangent) < 1e-8:
            return 0
        return vector_to_action(c_tangent)


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
