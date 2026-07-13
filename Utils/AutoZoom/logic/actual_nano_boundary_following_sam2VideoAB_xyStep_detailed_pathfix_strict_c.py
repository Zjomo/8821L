# actual_nano_boundary_following_yoloA_sam2BC.py
from __future__ import annotations

import csv
import json
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
from control.stage34_wait_done_then_pause import Stage34


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
    # "body_edge"：沿 C 主体边缘做四边形/少量顶点多边形近似，推荐。
    # "min_area_rect"：最小外接旋转矩形，不贴合主体边缘，仅作显式兜底。
    c_quadrilateral_method: str = "body_edge"

    # --------------------------------------------------------
    # 固定 C 地图复用 / 加载
    # --------------------------------------------------------
    # True：在同一个 GUI 进程内，第一次确认好的 C mask/contour 会一直复用。
    reuse_static_c_map_in_memory: bool = True

    # True：如果 GUI 已关闭又重新打开，则从磁盘加载之前确认好的 C map。
    load_static_c_map_if_exists: bool = True

    # True：第一次确认 C 后，把 C mask、全部像素位置、主体轮廓保存到磁盘。
    save_static_c_map_library: bool = True

    # True：忽略内存/磁盘已有 C map，强制重新标注 A/B/C 并覆盖保存新的 C map。
    force_reselect_c_each_run: bool = False

    # 固定 C map 的名字。不同样品/不同区域建议改不同名字。
    static_c_map_name: str = "default_static_c_map"

    # 固定 C map 保存目录。为空时默认使用 output_dir/_static_c_map_library/static_c_map_name。
    static_c_map_dir: str = ""

    # --------------------------------------------------------
    # B 目标边选择方式
    # nearest/left/right/top/bottom/longest
    # --------------------------------------------------------
    b_target_edge_mode: str = "nearest"

    # --------------------------------------------------------
    # A 后续帧实时分割：选择最像上一帧的候选 mask
    # --------------------------------------------------------
    # A 的形状基本不变、像素位置会变化，因此不能只按 SAM2 score 选候选 mask。
    # True：对 SAM2 返回的多个候选 A mask 进行时序稳定评分，选择最像上一帧 A 的结果。
    a_use_temporal_mask_selection: bool = True

    # 综合评分权重。score 只作为弱参考；shape IoU / bbox 形状约束 / C 泄漏硬拒绝是主约束。
    a_temporal_score_weight: float = 0.20
    a_temporal_iou_weight: float = 0.00
    a_temporal_shape_iou_weight: float = 0.00
    a_temporal_area_weight: float = 0.45
    a_temporal_center_weight: float = 0.45
    a_temporal_aspect_weight: float = 0.00
    a_temporal_bbox_size_weight: float = 0.00
    a_temporal_c_leak_weight: float = 0.00

    # A 可以移动，但中心突跳越大得分越低；值越小，约束越严格。
    a_temporal_center_sigma_px: float = 80.0

    # A 形状基本不变，因此面积、宽高比、bbox 尺寸都要严格约束。
    a_temporal_area_ratio_min: float = 0.35
    a_temporal_area_ratio_max: float = 2.80
    a_temporal_min_shape_iou: float = 0.00
    a_temporal_min_aspect_similarity: float = 0.00
    a_temporal_min_bbox_size_similarity: float = 0.00

    # A 不应大面积吸入 C 的独有区域；但允许上一帧 A/C 已经真实接近或接触的区域。
    # C 独有区域定义为 static_C - last_A。
    a_temporal_max_c_unique_leak_ratio: float = 1.00

    # 如果所有候选都很差，直接退回上一帧 A，避免错误 mask 继续传递。
    a_temporal_fallback_to_last_if_bad: bool = True
    a_temporal_min_accept_score: float = 0.05

    # --------------------------------------------------------
    # B 后续帧实时分割：选择最像上一帧的候选 mask
    # --------------------------------------------------------
    # B 的形状基本不变、像素位置会变化，因此不能只按 SAM2 score 选候选 mask。
    # True：对 SAM2 返回的多个候选 B mask 进行时序稳定评分，选择最像上一帧 B 的结果。
    b_use_temporal_mask_selection: bool = True

    # 综合评分权重。score 只作为弱参考；shape IoU / bbox 形状约束 / C 泄漏硬拒绝是主约束。
    b_temporal_score_weight: float = 0.20
    b_temporal_iou_weight: float = 0.00
    b_temporal_shape_iou_weight: float = 0.00
    b_temporal_area_weight: float = 0.45
    b_temporal_center_weight: float = 0.45
    b_temporal_aspect_weight: float = 0.00
    b_temporal_bbox_size_weight: float = 0.00
    b_temporal_c_leak_weight: float = 0.00

    # B 可以移动，但中心突跳越大得分越低；值越小，约束越严格。
    b_temporal_center_sigma_px: float = 80.0

    # B 形状基本不变，因此面积、宽高比、bbox 尺寸都要严格约束。
    b_temporal_area_ratio_min: float = 0.35
    b_temporal_area_ratio_max: float = 2.80
    b_temporal_min_shape_iou: float = 0.00
    b_temporal_min_aspect_similarity: float = 0.00
    b_temporal_min_bbox_size_similarity: float = 0.00

    # B 不能大面积吸入 C 的独有区域；但允许真实 B/C 重合区域。
    # C 独有区域定义为 static_C - last_B。
    b_temporal_max_c_unique_leak_ratio: float = 1.00

    # 如果所有候选都很差，直接退回上一帧 B，避免错误 mask 继续传递。
    b_temporal_fallback_to_last_if_bad: bool = True
    b_temporal_min_accept_score: float = 0.05

    # 后续帧搜索框外扩：避免 A/B 发生位移后仍用上一帧 bbox 太窄导致 prompt 错位。
    temporal_search_margin_px: float = 60.0

    # True：后续帧不再从旧 mask 中采样一堆旧正点，只使用“外扩 bbox + 上一帧中心点”。
    # 旧 mask 像素用于候选筛选，不直接作为硬提示；这样更符合“形状不变但位置会变”。
    temporal_use_center_point_only: bool = True

    # --------------------------------------------------------
    # A/B 位置跟踪：形状不变，但位置会移动
    # --------------------------------------------------------
    # True：先用上一帧图像块在当前帧中做模板匹配，估计 A/B 当前中心；
    # 再用“预测 bbox + 预测中心点”提示 SAM2。这样不会一直用上一帧旧中心。
    temporal_position_match_enable: bool = True
    temporal_position_match_margin_px: float = 220.0
    temporal_position_match_min_score: float = 0.15
    # False：默认使用第一帧人工确认后的初始化模板定位 A/B，避免上一帧错误模板污染后续帧。
    temporal_position_match_use_last_image_template: bool = True
    # True：位置匹配失败或候选 mask 质量不足时，不更新 A/B，不执行运动；等待下一帧或用户重新点击。
    temporal_stop_on_track_fail: bool = True

    # --------------------------------------------------------
    # A/B 推动过程实时跟踪模式
    # --------------------------------------------------------
    # True：优先使用 SAM2 video predictor 做 A/B 连续帧传播跟踪。
    # 失败时自动回退到上一帧局部模板 + SAM2 ImagePredictor refine。
    use_sam2_video_tracking: bool = True

    # 当前实现采用 two-frame streaming：
    #   上一帧已确认图像作为 anchor frame 0；
    #   当前截图作为 frame 1；
    #   在 frame 0 注册 A/B obj_id，再 propagate 到 frame 1。
    # 这样适合实时截图闭环，不需要提前准备完整视频。
    sam2_video_tracking_mode: str = "two_frame_anchor"
    sam2_video_temp_dir: str = "outputs/sam2_video_tracking_tmp"
    sam2_video_a_obj_id: int = 1
    sam2_video_b_obj_id: int = 2
    sam2_video_prompt_max_points: int = 5

    # video tracking 输出仍然只用面积与中心连续性做门控，允许自转。
    sam2_video_area_ratio_min: float = 0.35
    sam2_video_area_ratio_max: float = 2.80
    sam2_video_center_jump_max_px: float = 220.0
    sam2_video_fallback_to_template: bool = True

    # --------------------------------------------------------
    # A-C 距离控制，单位 px
    # --------------------------------------------------------
    a_c_target_clearance: float = 90.0
    a_c_min_clearance: float = 80.0
    a_c_max_clearance: float = 100.0
    a_c_inside_collision_check: bool = True

    # A-C 距离修正方向模式：
    #   "horizontal_from_c_center"：A-C 太近/太远时，优先按 A 与 C 中心的左右关系修正；
    #       适合 A 只在 C 左/右侧保持间隙的画面；如果 A 可能在 C 上/下方，不推荐。
    #   "nearest_normal"：使用 A 到 C 最近边界的法向方向修正；
    #       如果 C 的最近边是水平边，就可能输出 UP/DOWN。
    #   "auto"：先尝试水平修正，水平关系不明显时退回 nearest_normal。
    a_c_correction_mode: str = "nearest_normal"

    # --------------------------------------------------------
    # A-B 覆盖记录
    # --------------------------------------------------------
    # 当前版本不再使用 A-B 覆盖作为停止条件。
    # A/B overlap 只作为日志、CSV、可视化状态记录；
    # 运动决策始终由 A-C 距离和 C 最近边切向决定。
    # 保留 a_b_min_dist / a_b_max_dist 只是为了兼容旧日志字段。
    a_b_min_dist: float = 10.0
    a_b_max_dist: float = 35.0
    a_b_overlap_stop: bool = False
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

    # 3/4 通道可单独设置速度、加速度、最大电压。
    # 若底层 Stage34 支持 setup_channel 或 dev.setup_drive，会按通道单独下发。
    stage_ch3_velocity: int = 1
    stage_ch3_acceleration: int = 1
    stage_ch3_max_voltage: int = 50
    stage_ch4_velocity: int = 1
    stage_ch4_acceleration: int = 1
    stage_ch4_max_voltage: int = 50

    # 单次动作距离/步数。
    # 新规则：
    #   LEFT / RIGHT 使用 stage_step_x，运动完成后按 stage_ch3_pause_after_move_s 停顿；
    #   UP   / DOWN  使用 stage_step_y，运动完成后按 stage_ch4_pause_after_move_s 停顿。
    # action_step 只作为旧兼容兜底值，不再优先决定每次动作距离。
    stage_step_x: int = 20
    stage_step_y: int = 20

    # 如果图像方向和实际运动方向相反，改这里
    stage_x_sign: int = 1
    stage_y_sign: int = 1

    # 旧兼容字段：当 stage_step_x / stage_step_y 非法时才作为兜底步数。
    action_step: int = 100

    # 是否等待 Stage 运动完成
    wait_stage_move: bool = True
    stage_move_timeout_s: Optional[float] = 5.0

    # --------------------------------------------------------
    # CH3/CH4 运动节拍控制
    # --------------------------------------------------------
    # 需求：
    #   action=LEFT/RIGHT：CH3 一次性运动 stage_step_x，然后强制停顿 stage_ch3_pause_after_move_s。
    #   action=UP/DOWN   ：CH4 一次性运动 stage_step_y，然后强制停顿 stage_ch4_pause_after_move_s。
    # 停顿发生在一次完整 Stage 指令之后，不是在累计 action_step 多次之后。
    stage_ch3_pause_after_move_s: float = 2.0
    stage_ch3_force_wait_before_pause: bool = True
    stage_ch3_stop_after_move: bool = True

    stage_ch4_pause_after_move_s: float = 2.0
    stage_ch4_force_wait_before_pause: bool = True
    stage_ch4_stop_after_move: bool = True

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

    注意：这个函数只用于兜底的 minAreaRect 四边形。
    对于沿主体边缘的 approxPolyDP 结果，应该保留 OpenCV 返回的轮廓顺序，
    因为该顺序本身就是沿着 C 主体边界走的顺序。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]
    return [(float(x), float(y)) for x, y in ordered]


def contour_points_to_vec2_list(points: np.ndarray) -> List[Vec2]:
    """把 OpenCV 轮廓点转换为 Vec2 列表，并保留轮廓顺序。"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in pts]


def polygon_area_abs(points: List[Vec2]) -> float:
    """计算多边形面积绝对值。"""
    if len(points) < 3:
        return 0.0
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(arr)))


def simplify_contour_along_body_edge(
    contour: np.ndarray,
    contour_area: float,
    epsilon_ratio: float = 0.02,
    target_vertices: int = 4,
    max_vertices: int = 8,
) -> List[Vec2]:
    """
    沿 C 主体真实边缘做多边形近似，而不是做最小外接矩形。

    逻辑：
        1. 对原始最大轮廓 contour 做 approxPolyDP；
        2. 优先寻找 4 个点的近似轮廓；
        3. 如果没有稳定的 4 点结果，就返回 5~8 点的主体边缘多边形；
        4. 所有返回点都来自轮廓近似，沿着 C 主体边缘走，不使用外接矩形。

    这样得到的 C.contour 可以是四边形，也可以是少量顶点的主体边缘多边形。
    后续 polygon_edges()、A-C 最近边、C 切向运动都可以直接使用。
    """
    peri = float(cv2.arcLength(contour, True))
    if peri < 1e-8:
        raise RuntimeError("C 轮廓周长过小，无法做主体边缘近似。")

    base = max(0.001, float(epsilon_ratio))

    # 从小到大尝试：小 epsilon 贴边更精细，大 epsilon 顶点更少。
    eps_candidates = [
        base * 0.50,
        base * 0.75,
        base,
        base * 1.25,
        base * 1.50,
        base * 2.00,
        base * 2.50,
        base * 3.00,
        0.006,
        0.008,
        0.010,
        0.012,
        0.015,
        0.018,
        0.020,
        0.025,
        0.030,
        0.035,
        0.040,
        0.050,
        0.060,
        0.080,
    ]

    # 去重并排序，保证搜索稳定。
    eps_candidates = sorted(set(float(x) for x in eps_candidates if float(x) > 0))

    candidates: List[Dict[str, Any]] = []

    for er in eps_candidates:
        epsilon = max(1.0, er * peri)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = contour_points_to_vec2_list(approx.reshape(-1, 2))

        if len(pts) < 3:
            continue

        poly_area = polygon_area_abs(pts)
        if poly_area < 1e-8:
            continue

        area_ratio = poly_area / max(contour_area, 1e-8)

        # 只保留面积没有严重失真的近似。
        # approxPolyDP 的顶点来自主体边缘，面积可能略大或略小。
        if area_ratio < 0.45 or area_ratio > 1.80:
            continue

        candidates.append(
            {
                "eps_ratio": er,
                "pts": pts,
                "n": len(pts),
                "area_ratio": area_ratio,
                "area_error": abs(area_ratio - 1.0),
            }
        )

    if not candidates:
        raise RuntimeError("无法从 C 主体轮廓得到有效的边缘近似多边形。")

    # 1) 最优先：4 点主体边缘近似。
    quad_candidates = [c for c in candidates if c["n"] == int(target_vertices)]
    if quad_candidates:
        # 优先选择面积最接近原 contour 的 4 点近似。
        best = min(quad_candidates, key=lambda c: (c["area_error"], c["eps_ratio"]))
        return best["pts"]

    # 2) 次优先：5~8 点主体边缘多边形。
    polygon_candidates = [
        c for c in candidates
        if int(target_vertices) < c["n"] <= int(max_vertices)
    ]
    if polygon_candidates:
        # 面积接近优先；如果面积接近，则顶点少优先。
        best = min(polygon_candidates, key=lambda c: (c["area_error"], c["n"], c["eps_ratio"]))
        return best["pts"]

    # 3) 如果顶点仍太多，选择顶点数量最接近 max_vertices 且面积合理的结果。
    #    这仍然是沿主体边缘的多边形，不是外接矩形。
    best = min(candidates, key=lambda c: (abs(c["n"] - int(max_vertices)), c["area_error"], c["eps_ratio"]))
    return best["pts"]


def mask_to_quadrilateral_contour(
    mask_bool: np.ndarray,
    min_area: float = 50.0,
    epsilon_ratio: float = 0.02,
    method: str = "body_edge",
) -> Tuple[List[Vec2], float, Vec2]:
    """
    将 SAM2 得到的 C mask 转成沿 C 主体边缘走的四边形/多边形。

    推荐 method="body_edge"：
        - 使用 C 的最大真实轮廓；
        - 用 approxPolyDP 沿主体边缘近似；
        - 优先返回 4 点四边形；
        - 如果 4 点会明显失真，则返回 5~8 点主体边缘多边形；
        - 不使用 minAreaRect，因此不是外接矩形。

    method="min_area_rect" 仅作为显式兜底模式保留：
        - 会得到最小外接旋转矩形；
        - 不贴合 C 主体边缘；
        - 只有你明确设置 c_quadrilateral_method="min_area_rect" 时才会使用。
    """
    mask_u8 = (mask_bool.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError("C mask 中没有找到有效轮廓，无法拟合主体边缘。")

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))

    if area < min_area:
        raise RuntimeError(f"C mask 轮廓面积过小，无法拟合主体边缘: area={area:.2f}")

    method = str(method).lower().strip()

    if method in ("body_edge", "body_polygon", "approx_first", "approx", "poly", "quadrilateral"):
        pts = simplify_contour_along_body_edge(
            contour=contour,
            contour_area=area,
            epsilon_ratio=float(epsilon_ratio),
            target_vertices=4,
            max_vertices=8,
        )

        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-8:
            center = polygon_centroid_simple(pts)
        else:
            center = (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))

        return pts, area, center

    if method == "min_area_rect":
        # 仅在用户显式要求时使用。该结果是外接矩形，不是主体边缘。
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        pts = order_quad_points_clockwise(box)

        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-8:
            center = polygon_centroid_simple(pts)
        else:
            center = (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))

        return pts, area, center

    raise ValueError(
        f"unsupported c_quadrilateral_method={method}. "
        "Use body_edge/body_polygon/approx_first/min_area_rect."
    )

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


def mask_iou(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> float:
    """两个 mask 的 IoU，范围 0~1。"""
    if mask_a is None or mask_b is None:
        return 0.0
    if mask_a.shape != mask_b.shape:
        return 0.0

    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 1e-8:
        return 0.0
    return float(inter / union)


def mask_area_similarity(candidate_mask: np.ndarray, reference_mask: Optional[np.ndarray]) -> float:
    """面积相似度，范围 0~1，1 表示面积完全一致。"""
    if reference_mask is None:
        return 0.0
    cand_area = float(candidate_mask.astype(bool).sum())
    ref_area = float(reference_mask.astype(bool).sum())
    if cand_area <= 1e-8 or ref_area <= 1e-8:
        return 0.0
    ratio = cand_area / ref_area
    ratio = min(ratio, 1.0 / max(ratio, 1e-8))
    return float(max(0.0, min(1.0, ratio)))


def center_similarity(candidate_mask: np.ndarray, reference_center: Optional[Vec2], sigma_px: float = 80.0) -> float:
    """中心相似度，范围 0~1。B 可以移动，但中心跳变越大得分越低。"""
    if reference_center is None:
        return 0.0
    try:
        c = mask_center(candidate_mask.astype(bool))
    except Exception:
        return 0.0
    d = vec_len(vec_sub(c, reference_center))
    sigma = max(1.0, float(sigma_px))
    return float(math.exp(-d / sigma))


def c_unique_leak_ratio(
    candidate_b_mask: np.ndarray,
    static_c_mask: Optional[np.ndarray],
    last_b_mask: Optional[np.ndarray],
) -> float:
    """
    候选 mask 泄漏到 C 独有区域的比例。

    reference_mask = last_A 或 last_B。
    这样允许上一帧已经真实重合的区域，不允许突然吸入 static_C - reference_mask 的区域。
    """
    if static_c_mask is None or last_b_mask is None:
        return 0.0
    if candidate_b_mask.shape != static_c_mask.shape or candidate_b_mask.shape != last_b_mask.shape:
        return 0.0

    cand = candidate_b_mask.astype(bool)
    c_unique = np.logical_and(static_c_mask.astype(bool), np.logical_not(last_b_mask.astype(bool)))
    cand_area = float(cand.sum())
    if cand_area <= 1e-8:
        return 1.0
    leak = float(np.logical_and(cand, c_unique).sum())
    return float(leak / cand_area)


def translate_mask_nearest(mask_bool: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """平移 mask：dx>0 向右，dy>0 向下。"""
    h, w = mask_bool.shape[:2]
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    shifted = cv2.warpAffine(
        mask_bool.astype(np.uint8),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted.astype(bool)


def center_aligned_shape_iou(
    candidate_mask: np.ndarray,
    reference_mask: Optional[np.ndarray],
    reference_center: Optional[Vec2] = None,
) -> float:
    """
    中心对齐后的形状 IoU。
    允许目标整体平移，但要求候选 mask 的形状与上一帧 reference_mask 相似。
    """
    if reference_mask is None:
        return 0.0
    if candidate_mask.shape != reference_mask.shape:
        return 0.0
    try:
        cand_center = mask_center(candidate_mask.astype(bool))
        ref_center = reference_center if reference_center is not None else mask_center(reference_mask.astype(bool))
    except Exception:
        return 0.0
    dx = float(ref_center[0]) - float(cand_center[0])
    dy = float(ref_center[1]) - float(cand_center[1])
    aligned = translate_mask_nearest(candidate_mask.astype(bool), dx=dx, dy=dy)
    return mask_iou(aligned, reference_mask.astype(bool))


def mask_area_ratio(candidate_mask: np.ndarray, reference_mask: Optional[np.ndarray]) -> float:
    if reference_mask is None:
        return 1.0
    ca = float(candidate_mask.astype(bool).sum())
    ra = float(reference_mask.astype(bool).sum())
    if ca <= 1e-8 or ra <= 1e-8:
        return 0.0
    return float(ca / ra)


def bbox_width_height(mask_bool: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = mask_to_bbox_xyxy(mask_bool.astype(bool))
    return max(1.0, float(x2 - x1 + 1.0)), max(1.0, float(y2 - y1 + 1.0))


def bbox_aspect_similarity(candidate_mask: np.ndarray, reference_mask: Optional[np.ndarray]) -> float:
    """bbox 宽高比相似度，1 表示完全一致。"""
    if reference_mask is None:
        return 0.0
    try:
        cw, ch = bbox_width_height(candidate_mask.astype(bool))
        rw, rh = bbox_width_height(reference_mask.astype(bool))
    except Exception:
        return 0.0
    cand_ratio = cw / max(ch, 1e-8)
    ref_ratio = rw / max(rh, 1e-8)
    r = cand_ratio / max(ref_ratio, 1e-8)
    r = min(r, 1.0 / max(r, 1e-8))
    return float(max(0.0, min(1.0, r)))


def bbox_size_similarity(candidate_mask: np.ndarray, reference_mask: Optional[np.ndarray]) -> float:
    """bbox 宽、高尺寸相似度，1 表示宽高均完全一致。"""
    if reference_mask is None:
        return 0.0
    try:
        cw, ch = bbox_width_height(candidate_mask.astype(bool))
        rw, rh = bbox_width_height(reference_mask.astype(bool))
    except Exception:
        return 0.0
    sim_w = min(cw / max(rw, 1e-8), rw / max(cw, 1e-8))
    sim_h = min(ch / max(rh, 1e-8), rh / max(ch, 1e-8))
    return float(max(0.0, min(1.0, 0.5 * (sim_w + sim_h))))


def expand_bbox_xyxy(
    bbox_xyxy: Tuple[float, float, float, float],
    margin_px: float,
    image_shape_hw: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    h, w = image_shape_hw
    x1, y1, x2, y2 = bbox_xyxy
    m = max(0.0, float(margin_px))
    return (
        float(max(0.0, x1 - m)),
        float(max(0.0, y1 - m)),
        float(min(w - 1.0, x2 + m)),
        float(min(h - 1.0, y2 + m)),
    )


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
        a_use_temporal_mask_selection: bool = True,
        a_temporal_score_weight: float = 0.35,
        a_temporal_iou_weight: float = 0.00,
        a_temporal_shape_iou_weight: float = 0.00,
        a_temporal_area_weight: float = 0.45,
        a_temporal_center_weight: float = 0.45,
        a_temporal_aspect_weight: float = 0.00,
        a_temporal_bbox_size_weight: float = 0.00,
        a_temporal_c_leak_weight: float = 0.00,
        a_temporal_center_sigma_px: float = 80.0,
        a_temporal_area_ratio_min: float = 0.35,
        a_temporal_area_ratio_max: float = 2.80,
        a_temporal_min_shape_iou: float = 0.00,
        a_temporal_min_aspect_similarity: float = 0.00,
        a_temporal_min_bbox_size_similarity: float = 0.00,
        a_temporal_max_c_unique_leak_ratio: float = 1.00,
        a_temporal_fallback_to_last_if_bad: bool = True,
        a_temporal_min_accept_score: float = 0.05,
        b_use_temporal_mask_selection: bool = True,
        b_temporal_score_weight: float = 0.20,
        b_temporal_iou_weight: float = 0.00,
        b_temporal_shape_iou_weight: float = 0.00,
        b_temporal_area_weight: float = 0.45,
        b_temporal_center_weight: float = 0.45,
        b_temporal_aspect_weight: float = 0.00,
        b_temporal_bbox_size_weight: float = 0.00,
        b_temporal_c_leak_weight: float = 0.00,
        b_temporal_center_sigma_px: float = 80.0,
        b_temporal_area_ratio_min: float = 0.35,
        b_temporal_area_ratio_max: float = 2.80,
        b_temporal_min_shape_iou: float = 0.00,
        b_temporal_min_aspect_similarity: float = 0.00,
        b_temporal_min_bbox_size_similarity: float = 0.00,
        b_temporal_max_c_unique_leak_ratio: float = 1.00,
        b_temporal_fallback_to_last_if_bad: bool = True,
        b_temporal_min_accept_score: float = 0.05,
        temporal_search_margin_px: float = 60.0,
        temporal_use_center_point_only: bool = True,
        temporal_position_match_enable: bool = True,
        temporal_position_match_margin_px: float = 220.0,
        temporal_position_match_min_score: float = 0.15,
        temporal_position_match_use_last_image_template: bool = True,
        temporal_stop_on_track_fail: bool = True,
        use_sam2_video_tracking: bool = True,
        sam2_video_tracking_mode: str = "two_frame_anchor",
        sam2_video_temp_dir: str = "outputs/sam2_video_tracking_tmp",
        sam2_video_a_obj_id: int = 1,
        sam2_video_b_obj_id: int = 2,
        sam2_video_prompt_max_points: int = 5,
        sam2_video_area_ratio_min: float = 0.35,
        sam2_video_area_ratio_max: float = 2.80,
        sam2_video_center_jump_max_px: float = 220.0,
        sam2_video_fallback_to_template: bool = True,
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

        # SAM2 video predictor 是可选增强：如果当前 SAM2 安装版本不支持，
        # 程序会自动回退到上一帧局部模板 + ImagePredictor refine。
        self.video_predictor = None
        self.video_predictor_available = False
        try:
            from sam2.build_sam import build_sam2_video_predictor
            self.video_predictor = build_sam2_video_predictor(
                sam2_cfg,
                sam2_checkpoint,
                device=device,
            )
            self.video_predictor_available = True
            logger.info("SAM2 video predictor 初始化成功，将优先用于 A/B 连续帧跟踪。")
        except Exception as e:
            logger.warning(
                "SAM2 video predictor 初始化失败，将回退到局部模板+ImagePredictor。原因：%s",
                e,
            )

        self.min_area_px = float(min_area_px)
        self.area_ratio_min = float(area_ratio_min)
        self.area_ratio_max = float(area_ratio_max)
        self.center_jump_max_px = float(center_jump_max_px)
        self.use_last_mask_when_failed = bool(use_last_mask_when_failed)

        # A 后续帧候选 mask 稳定选择参数。
        self.a_use_temporal_mask_selection = bool(a_use_temporal_mask_selection)
        self.a_temporal_score_weight = float(a_temporal_score_weight)
        self.a_temporal_iou_weight = float(a_temporal_iou_weight)
        self.a_temporal_shape_iou_weight = float(a_temporal_shape_iou_weight)
        self.a_temporal_area_weight = float(a_temporal_area_weight)
        self.a_temporal_center_weight = float(a_temporal_center_weight)
        self.a_temporal_aspect_weight = float(a_temporal_aspect_weight)
        self.a_temporal_bbox_size_weight = float(a_temporal_bbox_size_weight)
        self.a_temporal_c_leak_weight = float(a_temporal_c_leak_weight)
        self.a_temporal_center_sigma_px = float(a_temporal_center_sigma_px)
        self.a_temporal_area_ratio_min = float(a_temporal_area_ratio_min)
        self.a_temporal_area_ratio_max = float(a_temporal_area_ratio_max)
        self.a_temporal_min_shape_iou = float(a_temporal_min_shape_iou)
        self.a_temporal_min_aspect_similarity = float(a_temporal_min_aspect_similarity)
        self.a_temporal_min_bbox_size_similarity = float(a_temporal_min_bbox_size_similarity)
        self.a_temporal_max_c_unique_leak_ratio = float(a_temporal_max_c_unique_leak_ratio)
        self.a_temporal_fallback_to_last_if_bad = bool(a_temporal_fallback_to_last_if_bad)
        self.a_temporal_min_accept_score = float(a_temporal_min_accept_score)
        self.last_a_temporal_debug: Dict[str, Any] = {}

        # B 后续帧候选 mask 稳定选择参数。
        self.b_use_temporal_mask_selection = bool(b_use_temporal_mask_selection)
        self.b_temporal_score_weight = float(b_temporal_score_weight)
        self.b_temporal_iou_weight = float(b_temporal_iou_weight)
        self.b_temporal_shape_iou_weight = float(b_temporal_shape_iou_weight)
        self.b_temporal_area_weight = float(b_temporal_area_weight)
        self.b_temporal_center_weight = float(b_temporal_center_weight)
        self.b_temporal_aspect_weight = float(b_temporal_aspect_weight)
        self.b_temporal_bbox_size_weight = float(b_temporal_bbox_size_weight)
        self.b_temporal_c_leak_weight = float(b_temporal_c_leak_weight)
        self.b_temporal_center_sigma_px = float(b_temporal_center_sigma_px)
        self.b_temporal_area_ratio_min = float(b_temporal_area_ratio_min)
        self.b_temporal_area_ratio_max = float(b_temporal_area_ratio_max)
        self.b_temporal_min_shape_iou = float(b_temporal_min_shape_iou)
        self.b_temporal_min_aspect_similarity = float(b_temporal_min_aspect_similarity)
        self.b_temporal_min_bbox_size_similarity = float(b_temporal_min_bbox_size_similarity)
        self.b_temporal_max_c_unique_leak_ratio = float(b_temporal_max_c_unique_leak_ratio)
        self.b_temporal_fallback_to_last_if_bad = bool(b_temporal_fallback_to_last_if_bad)
        self.b_temporal_min_accept_score = float(b_temporal_min_accept_score)
        self.temporal_search_margin_px = float(temporal_search_margin_px)
        self.temporal_use_center_point_only = bool(temporal_use_center_point_only)
        self.temporal_position_match_enable = bool(temporal_position_match_enable)
        self.temporal_position_match_margin_px = float(temporal_position_match_margin_px)
        self.temporal_position_match_min_score = float(temporal_position_match_min_score)
        self.temporal_position_match_use_last_image_template = bool(temporal_position_match_use_last_image_template)
        self.temporal_stop_on_track_fail = bool(temporal_stop_on_track_fail)

        self.use_sam2_video_tracking = bool(use_sam2_video_tracking)
        self.sam2_video_tracking_mode = str(sam2_video_tracking_mode)
        self.sam2_video_temp_dir = Path(str(sam2_video_temp_dir))
        self.sam2_video_a_obj_id = int(sam2_video_a_obj_id)
        self.sam2_video_b_obj_id = int(sam2_video_b_obj_id)
        self.sam2_video_prompt_max_points = int(max(1, sam2_video_prompt_max_points))
        self.sam2_video_area_ratio_min = float(sam2_video_area_ratio_min)
        self.sam2_video_area_ratio_max = float(sam2_video_area_ratio_max)
        self.sam2_video_center_jump_max_px = float(sam2_video_center_jump_max_px)
        self.sam2_video_fallback_to_template = bool(sam2_video_fallback_to_template)
        self.last_video_tracking_debug: Dict[str, Any] = {}

        self.last_b_temporal_debug: Dict[str, Any] = {}

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

        # 第一帧确认后的形状模板。A/B 可以移动，但不允许形状逐帧漂移。
        self.template_a_mask: Optional[np.ndarray] = None
        self.template_b_mask: Optional[np.ndarray] = None

        # 第一帧人工确认后的固定模板。默认用这些模板在当前帧定位 A/B，
        # 不使用上一帧模板，避免“上一帧错了 -> 下一帧继续错”的漂移链路。
        self.init_image_rgb: Optional[np.ndarray] = None
        self.init_a_mask: Optional[np.ndarray] = None
        self.init_b_mask: Optional[np.ndarray] = None
        self.init_a_bbox: Optional[Tuple[float, float, float, float]] = None
        self.init_b_bbox: Optional[Tuple[float, float, float, float]] = None
        self.init_a_center: Optional[Vec2] = None
        self.init_b_center: Optional[Vec2] = None
        self.init_a_template_rgb: Optional[np.ndarray] = None
        self.init_b_template_rgb: Optional[np.ndarray] = None

        # 上一帧原始图像。仅在 temporal_position_match_use_last_image_template=True 时作为模板来源。
        self.last_image_rgb: Optional[np.ndarray] = None

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

    def _crop_template_by_bbox(
        self,
        image_rgb: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float],
        margin: int = 8,
    ) -> np.ndarray:
        """按 bbox 裁剪模板图像块。用于后续帧 A/B 位置匹配。"""
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        x1i = int(max(0, round(x1) - int(margin)))
        y1i = int(max(0, round(y1) - int(margin)))
        x2i = int(min(w - 1, round(x2) + int(margin)))
        y2i = int(min(h - 1, round(y2) + int(margin)))
        if x2i <= x1i or y2i <= y1i:
            raise RuntimeError("模板 bbox 无效，无法裁剪模板。")
        return image_rgb[y1i:y2i + 1, x1i:x2i + 1].copy()

    def _save_initial_ab_templates(self, image_rgb: np.ndarray) -> None:
        """
        保存第一帧人工确认后的 A/B 初始形状和初始图像模板。

        后续帧默认使用 init_a_template/init_b_template 定位当前位置，
        而不是使用上一帧图像块作为模板，防止错误分割结果污染后续跟踪。
        """
        if self.last_a_mask is None or self.last_b_mask is None:
            raise RuntimeError("保存初始化模板失败：last_a_mask 或 last_b_mask 为空。")
        if self.last_a_bbox is None or self.last_b_bbox is None:
            raise RuntimeError("保存初始化模板失败：last_a_bbox 或 last_b_bbox 为空。")

        self.init_image_rgb = image_rgb.copy()
        self.init_a_mask = self.last_a_mask.astype(bool).copy()
        self.init_b_mask = self.last_b_mask.astype(bool).copy()
        self.init_a_bbox = tuple(float(x) for x in self.last_a_bbox)
        self.init_b_bbox = tuple(float(x) for x in self.last_b_bbox)
        self.init_a_center = mask_center(self.init_a_mask)
        self.init_b_center = mask_center(self.init_b_mask)
        self.init_a_template_rgb = self._crop_template_by_bbox(image_rgb, self.init_a_bbox, margin=8)
        self.init_b_template_rgb = self._crop_template_by_bbox(image_rgb, self.init_b_bbox, margin=8)

        logger.info(
            "已保存 A/B 初始化模板：A_template=%s, B_template=%s, A_center=(%.1f, %.1f), B_center=(%.1f, %.1f)",
            None if self.init_a_template_rgb is None else self.init_a_template_rgb.shape,
            None if self.init_b_template_rgb is None else self.init_b_template_rgb.shape,
            self.init_a_center[0], self.init_a_center[1],
            self.init_b_center[0], self.init_b_center[1],
        )

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

        修改后的规则：
            1. A 支持多个正点 + 多个负点；
            2. B 支持多个正点 + 多个负点；
            3. C 支持多个正点 + 多个负点；
            4. B 和 C 允许真实重合，因此不要把 B/C 重合区点成负点；
            5. 第一次 C 分割得到的 mask 会被固定保存，后续运动控制始终使用
               初始 C mask 的所有像素位置，不再逐帧重新分割/更新 C。
        """
        # 兼容旧接口：如果只传 a_point/b_point/c_point，则退化为单正点。
        if a_positive_points is None:
            if a_point is None:
                raise ValueError("A 至少需要 1 个正点。")
            a_positive_points = [a_point]

        if b_positive_points is None:
            if b_point is None:
                raise ValueError("B 至少需要 1 个正点。")
            b_positive_points = [b_point]

        if c_positive_points is None:
            if c_point is None:
                raise ValueError("C 至少需要 1 个正点。")
            c_positive_points = [c_point]

        a_positive_points = self._clean_points(a_positive_points)
        a_negative_points = self._clean_points(a_negative_points)
        b_positive_points = self._clean_points(b_positive_points)
        b_negative_points = self._clean_points(b_negative_points)
        c_positive_points = self._clean_points(c_positive_points)
        c_negative_points = self._clean_points(c_negative_points)

        if len(a_positive_points) <= 0:
            raise ValueError("A 至少需要 1 个正点。")
        if len(b_positive_points) <= 0:
            raise ValueError("B 至少需要 1 个正点。")
        if len(c_positive_points) <= 0:
            raise ValueError("C 至少需要 1 个正点。")

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

        # A/B/C：全部支持多个正点 + 多个负点。
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
        self.template_a_mask = a_mask.copy()
        self.template_b_mask = b_mask.copy()

        self.last_a_bbox = mask_to_bbox_xyxy(a_mask)
        self.last_b_bbox = mask_to_bbox_xyxy(b_mask)

        self.last_a_center = mask_center(a_mask)
        self.last_b_center = mask_center(b_mask)
        self._save_initial_ab_templates(image_rgb)

        # C 固定：记录第一次 C 分割得到的所有像素位置。
        self.last_c_mask = c_mask.copy()
        self.last_c_bbox = mask_to_bbox_xyxy(c_mask)
        self.last_c_center = mask_center(c_mask)

        self.static_c_mask = c_mask.copy()
        self.static_c_bbox = self.last_c_bbox
        self.static_c_center = self.last_c_center
        self.static_c_pixel_yx = np.column_stack(np.where(self.static_c_mask.astype(bool)))

        self.last_image_rgb = image_rgb.copy()

        self.initialized = True

        logger.info(
            "SAM2 初始化完成: "
            "A=%d positive + %d negative, "
            "B=%d positive + %d negative, "
            "C=%d positive + %d negative fixed; "
            "A_center=(%.1f, %.1f), B_center=(%.1f, %.1f), "
            "C_static_center=(%.1f, %.1f), C_static_pixels=%d",
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
            self.static_c_center[0],
            self.static_c_center[1],
            int(self.static_c_mask.sum()),
        )

        return a_mask, b_mask, c_mask

    def initialize_ab_with_static_c_map(
        self,
        image_rgb: np.ndarray,
        static_c_mask: np.ndarray,
        a_positive_points: Optional[List[Vec2]] = None,
        a_negative_points: Optional[List[Vec2]] = None,
        b_positive_points: Optional[List[Vec2]] = None,
        b_negative_points: Optional[List[Vec2]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用已经确认/加载的固定 C map，只重新点击并分割 A、B。

        用途：
            1. GUI 未关闭：复用内存中的 C 像素和轮廓；
            2. GUI 已关闭：从磁盘加载之前确认的 C 像素和轮廓；
            3. 后续每次实验只需要重新确认 A、B 的像素位置。
        """
        a_positive_points = self._clean_points(a_positive_points)
        a_negative_points = self._clean_points(a_negative_points)
        b_positive_points = self._clean_points(b_positive_points)
        b_negative_points = self._clean_points(b_negative_points)

        if len(a_positive_points) <= 0:
            raise ValueError("A 至少需要 1 个正点。")
        if len(b_positive_points) <= 0:
            raise ValueError("B 至少需要 1 个正点。")

        c_mask = static_c_mask.astype(bool).copy()
        self._basic_mask_check("C_static", c_mask)

        self.a_init_point = a_positive_points[0]
        self.b_init_point = b_positive_points[0]
        self.c_init_point = mask_center(c_mask)

        self.a_init_positive_points = list(a_positive_points)
        self.a_init_negative_points = list(a_negative_points)
        self.b_init_positive_points = list(b_positive_points)
        self.b_init_negative_points = list(b_negative_points)

        self.predictor.set_image(image_rgb)

        a_mask = self._predict_mask_by_points(
            positive_points=a_positive_points,
            negative_points=a_negative_points,
        )
        b_mask = self._predict_mask_by_points(
            positive_points=b_positive_points,
            negative_points=b_negative_points,
        )

        self._basic_mask_check("A", a_mask)
        self._basic_mask_check("B", b_mask)

        self.last_a_mask = a_mask
        self.last_b_mask = b_mask
        self.template_a_mask = a_mask.copy()
        self.template_b_mask = b_mask.copy()
        self.last_c_mask = c_mask.copy()

        self.last_a_bbox = mask_to_bbox_xyxy(a_mask)
        self.last_b_bbox = mask_to_bbox_xyxy(b_mask)
        self.last_c_bbox = mask_to_bbox_xyxy(c_mask)

        self.last_a_center = mask_center(a_mask)
        self.last_b_center = mask_center(b_mask)
        self._save_initial_ab_templates(image_rgb)
        self.last_c_center = mask_center(c_mask)

        self.static_c_mask = c_mask.copy()
        self.static_c_bbox = self.last_c_bbox
        self.static_c_center = self.last_c_center
        self.static_c_pixel_yx = np.column_stack(np.where(self.static_c_mask.astype(bool)))

        self.last_image_rgb = image_rgb.copy()

        self.initialized = True

        logger.info(
            "SAM2 A/B 初始化完成，并复用固定 C map: "
            "A=%d positive + %d negative, "
            "B=%d positive + %d negative, "
            "C_static_pixels=%d",
            len(a_positive_points),
            len(a_negative_points),
            len(b_positive_points),
            len(b_negative_points),
            int(self.static_c_mask.sum()),
        )

        return a_mask, b_mask, c_mask

    def _estimate_bbox_by_template_match(
        self,
        name: str,
        image_rgb: np.ndarray,
        last_bbox: Tuple[float, float, float, float],
        last_center: Optional[Vec2],
    ) -> Tuple[Tuple[float, float, float, float], Vec2, Dict[str, Any]]:
        """
        用模板匹配估计 A/B 在当前帧的新位置。

        默认策略：使用第一帧人工确认后的 init template，而不是上一帧 template。
        原因：上一帧如果已经分割/定位错误，继续使用上一帧模板会把错误传递到后续帧。

        如果局部搜索分数低，会在整幅当前图像中做一次全局兜底搜索；仍低于阈值时，
        返回上一帧 bbox/center，并把 track_ok=False 写入 debug，由 infer_abc 阻止更新和运动。
        """
        if not self.temporal_position_match_enable:
            c = last_center if last_center is not None else ((last_bbox[0] + last_bbox[2]) / 2.0, (last_bbox[1] + last_bbox[3]) / 2.0)
            return last_bbox, c, {"enabled": False, "track_ok": True, "reason": "disabled"}

        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = last_bbox
        old_center = last_center if last_center is not None else ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        use_last_template = bool(self.temporal_position_match_use_last_image_template)

        template_rgb: Optional[np.ndarray] = None
        template_source = ""

        if use_last_template and self.last_image_rgb is not None:
            x1i = int(max(0, min(w - 1, round(x1))))
            y1i = int(max(0, min(h - 1, round(y1))))
            x2i = int(max(0, min(w - 1, round(x2))))
            y2i = int(max(0, min(h - 1, round(y2))))
            if x2i > x1i + 4 and y2i > y1i + 4:
                template_rgb = self.last_image_rgb[y1i:y2i + 1, x1i:x2i + 1]
                template_source = "last_frame_bbox"

        if template_rgb is None or template_rgb.size <= 0:
            if str(name).upper() == "A":
                template_rgb = self.init_a_template_rgb
                template_source = "init_A_template"
            elif str(name).upper() == "B":
                template_rgb = self.init_b_template_rgb
                template_source = "init_B_template"

        if template_rgb is None or template_rgb.size <= 0:
            return last_bbox, old_center, {
                "enabled": True,
                "track_ok": False,
                "fallback": True,
                "reason": "no_valid_template",
                "template_source": template_source,
            }

        th, tw = template_rgb.shape[:2]
        if th < 5 or tw < 5:
            return last_bbox, old_center, {
                "enabled": True,
                "track_ok": False,
                "fallback": True,
                "reason": "template_too_small",
                "template_source": template_source,
            }

        def _match_in_window(sx1: int, sy1: int, sx2: int, sy2: int, tag: str) -> Optional[Dict[str, Any]]:
            sx1 = int(max(0, min(w - 1, sx1)))
            sy1 = int(max(0, min(h - 1, sy1)))
            sx2 = int(max(0, min(w - 1, sx2)))
            sy2 = int(max(0, min(h - 1, sy2)))
            if sx2 <= sx1 or sy2 <= sy1:
                return None
            search_rgb = image_rgb[sy1:sy2 + 1, sx1:sx2 + 1]
            sh, sw = search_rgb.shape[:2]
            if sh < th + 2 or sw < tw + 2:
                return None
            tmpl = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
            srch = cv2.cvtColor(search_rgb, cv2.COLOR_RGB2GRAY)
            tmpl = cv2.equalizeHist(tmpl)
            srch = cv2.equalizeHist(srch)
            try:
                res = cv2.matchTemplate(srch, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
            except Exception as e:
                return {"error": f"matchTemplate_failed:{e}", "score": -1.0, "tag": tag}
            bx = sx1 + int(max_loc[0])
            by = sy1 + int(max_loc[1])
            new_x1 = float(bx)
            new_y1 = float(by)
            new_x2 = float(min(w - 1, bx + tw - 1))
            new_y2 = float(min(h - 1, by + th - 1))
            new_bbox = (new_x1, new_y1, new_x2, new_y2)
            new_center = ((new_x1 + new_x2) / 2.0, (new_y1 + new_y2) / 2.0)
            return {
                "tag": tag,
                "score": float(max_val),
                "bbox": new_bbox,
                "center": new_center,
                "window": (sx1, sy1, sx2, sy2),
            }

        margin = float(self.temporal_position_match_margin_px)
        local = _match_in_window(
            int(round(x1 - margin)),
            int(round(y1 - margin)),
            int(round(x2 + margin)),
            int(round(y2 + margin)),
            tag="local",
        )

        # 使用上一帧局部模板时，目标应当只在上一帧附近搜索，避免整图匹配跳到相似背景。
        # 只有在显式使用初始化模板时，才允许局部失败后做整图兜底。
        global_match = None
        if (not use_last_template) and ((local is None) or (float(local.get("score", -1.0)) < float(self.temporal_position_match_min_score))):
            global_match = _match_in_window(0, 0, w - 1, h - 1, tag="global")

        candidates = [m for m in (local, global_match) if m is not None and "bbox" in m]
        if not candidates:
            return last_bbox, old_center, {
                "enabled": True,
                "track_ok": False,
                "fallback": True,
                "reason": "no_match_candidate",
                "template_source": template_source,
            }

        best = max(candidates, key=lambda d: float(d.get("score", -1.0)))
        score = float(best.get("score", -1.0))
        track_ok = score >= float(self.temporal_position_match_min_score)

        if not track_ok:
            return last_bbox, old_center, {
                "enabled": True,
                "track_ok": False,
                "fallback": True,
                "reason": "low_template_score",
                "score": score,
                "min_score": float(self.temporal_position_match_min_score),
                "template_source": template_source,
                "local_score": None if local is None else float(local.get("score", -1.0)),
                "global_score": None if global_match is None else float(global_match.get("score", -1.0)),
            }

        new_bbox = best["bbox"]
        new_center = best["center"]
        shift = vec_sub(new_center, old_center)

        return new_bbox, new_center, {
            "enabled": True,
            "track_ok": True,
            "fallback": False,
            "score": score,
            "min_score": float(self.temporal_position_match_min_score),
            "template_source": template_source,
            "match_tag": best.get("tag"),
            "old_center": old_center,
            "pred_center": new_center,
            "shift_x": float(shift[0]),
            "shift_y": float(shift[1]),
            "search_margin_px": float(margin),
            "local_score": None if local is None else float(local.get("score", -1.0)),
            "global_score": None if global_match is None else float(global_match.get("score", -1.0)),
        }


    def _sample_video_prompt_points_from_mask(
        self,
        mask_bool: np.ndarray,
        center: Optional[Vec2],
        max_points: int = 5,
    ) -> np.ndarray:
        """
        从上一帧 A/B mask 内采样少量正点，作为 SAM2 video predictor 在 anchor frame 上的 obj prompt。
        这些点只用于注册上一帧中的目标身份；后续当前帧由 video predictor propagate。
        """
        mask = mask_bool.astype(bool)
        ys, xs = np.where(mask)
        points: List[Vec2] = []

        if center is not None:
            cx, cy = float(center[0]), float(center[1])
            h, w = mask.shape[:2]
            xi = int(max(0, min(w - 1, round(cx))))
            yi = int(max(0, min(h - 1, round(cy))))
            if mask[yi, xi]:
                points.append((cx, cy))

        if len(xs) > 0:
            # 选取 bbox 内几个分位点附近的 mask 像素，增强注册稳定性。
            idxs = np.linspace(0, len(xs) - 1, num=min(int(max_points) * 3, len(xs))).astype(int)
            for idx in idxs:
                p = (float(xs[idx]), float(ys[idx]))
                if p not in points:
                    points.append(p)
                if len(points) >= int(max_points):
                    break

        if not points and center is not None:
            points.append((float(center[0]), float(center[1])))

        if not points:
            c = mask_center(mask)
            points.append(c)

        return np.asarray(points[: int(max_points)], dtype=np.float32)

    def _video_add_object_prompt(
        self,
        inference_state: Any,
        frame_idx: int,
        obj_id: int,
        mask_bool: np.ndarray,
        center: Optional[Vec2],
        bbox_xyxy: Optional[Tuple[float, float, float, float]],
    ) -> None:
        """兼容不同 SAM2 版本的 add_new_points_or_box 调用。"""
        if self.video_predictor is None:
            raise RuntimeError("SAM2 video predictor 不可用。")

        points = self._sample_video_prompt_points_from_mask(
            mask_bool=mask_bool,
            center=center,
            max_points=self.sam2_video_prompt_max_points,
        )
        labels = np.ones((len(points),), dtype=np.int32)
        box = None
        if bbox_xyxy is not None:
            box = np.asarray(bbox_xyxy, dtype=np.float32)

        # SAM2.1 常见签名：add_new_points_or_box(inference_state, frame_idx, obj_id, points, labels, box)
        try:
            self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                points=points,
                labels=labels,
                box=box,
            )
            return
        except TypeError:
            pass

        # 某些版本不支持 box 参数。
        try:
            self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                points=points,
                labels=labels,
            )
            return
        except TypeError:
            pass

        # 更旧版本可能叫 add_new_points。
        if hasattr(self.video_predictor, "add_new_points"):
            self.video_predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                points=points,
                labels=labels,
            )
            return

        raise RuntimeError("当前 SAM2 video predictor 不支持 add_new_points_or_box/add_new_points 接口。")

    def _extract_video_masks_for_frame(
        self,
        out_obj_ids: Any,
        out_mask_logits: Any,
        image_shape_hw: Tuple[int, int],
    ) -> Dict[int, np.ndarray]:
        """把 SAM2 video predictor 输出转成 {obj_id: bool mask}。"""
        h, w = image_shape_hw
        if hasattr(out_mask_logits, "detach"):
            arr = out_mask_logits.detach().cpu().numpy()
        else:
            arr = np.asarray(out_mask_logits)

        # 常见形状：[N, 1, H, W] 或 [N, H, W]
        if arr.ndim == 4:
            arr = arr[:, 0, :, :]
        elif arr.ndim == 3:
            pass
        elif arr.ndim == 2:
            arr = arr[None, :, :]
        else:
            raise RuntimeError(f"无法解析 SAM2 video mask logits 形状: {arr.shape}")

        masks: Dict[int, np.ndarray] = {}
        obj_ids = list(out_obj_ids)
        for i, oid in enumerate(obj_ids):
            if i >= arr.shape[0]:
                continue
            m = arr[i] > 0.0
            if m.shape[:2] != (h, w):
                m = cv2.resize(
                    m.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            masks[int(oid)] = m.astype(bool)
        return masks

    def _validate_video_tracking_mask(
        self,
        name: str,
        new_mask: np.ndarray,
        old_mask: Optional[np.ndarray],
        old_center: Optional[Vec2],
    ) -> Dict[str, Any]:
        """
        SAM2 video tracking 的轻量质量门控：只看面积连续性和中心连续性。
        允许目标自转，因此不再强制 shape-IoU / bbox 宽高比一致。
        """
        self._basic_mask_check(name, new_mask)
        debug: Dict[str, Any] = {"ok": True, "name": name}

        new_area = float(new_mask.astype(bool).sum())
        debug["new_area"] = new_area

        if old_mask is not None:
            old_area = float(old_mask.astype(bool).sum())
            debug["old_area"] = old_area
            if old_area > 1.0:
                ratio = new_area / old_area
                debug["area_ratio"] = float(ratio)
                if ratio < self.sam2_video_area_ratio_min or ratio > self.sam2_video_area_ratio_max:
                    debug["ok"] = False
                    debug["reason"] = (
                        f"{name} video mask 面积突变: old={old_area:.1f}, "
                        f"new={new_area:.1f}, ratio={ratio:.2f}, "
                        f"allowed=[{self.sam2_video_area_ratio_min}, {self.sam2_video_area_ratio_max}]"
                    )
                    return debug

        if old_center is not None:
            new_center = mask_center(new_mask)
            jump = vec_len(vec_sub(new_center, old_center))
            debug["old_center"] = old_center
            debug["new_center"] = new_center
            debug["center_jump_px"] = float(jump)
            if jump > self.sam2_video_center_jump_max_px:
                debug["ok"] = False
                debug["reason"] = (
                    f"{name} video mask 中心跳变过大: "
                    f"jump={jump:.1f}px > {self.sam2_video_center_jump_max_px:.1f}px"
                )
                return debug

        return debug

    def _infer_ab_by_sam2_video_two_frame(
        self,
        image_rgb: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """
        实时 two-frame SAM2 video tracking：
            frame 0 = 上一帧已确认图像；
            frame 1 = 当前截图；
            在 frame 0 注册 A/B obj_id，并 propagate 到 frame 1。

        这比逐帧 ImagePredictor 更适合 A/B 同时发生平移和自转的情况。
        """
        debug: Dict[str, Any] = {
            "enabled": bool(self.use_sam2_video_tracking),
            "mode": self.sam2_video_tracking_mode,
            "video_predictor_available": bool(self.video_predictor_available),
        }

        if not self.use_sam2_video_tracking:
            debug["reason"] = "disabled"
            self.last_video_tracking_debug = debug
            return None
        if not self.video_predictor_available or self.video_predictor is None:
            debug["reason"] = "video_predictor_unavailable"
            self.last_video_tracking_debug = debug
            return None
        if self.last_image_rgb is None:
            debug["reason"] = "last_image_rgb_is_none"
            self.last_video_tracking_debug = debug
            return None
        if self.last_a_mask is None or self.last_b_mask is None:
            debug["reason"] = "last_a_or_b_mask_is_none"
            self.last_video_tracking_debug = debug
            return None
        if self.last_a_bbox is None or self.last_b_bbox is None:
            debug["reason"] = "last_a_or_b_bbox_is_none"
            self.last_video_tracking_debug = debug
            return None

        # 运行中 GUI 参数同步时，sam2_video_temp_dir 可能被写成 str。
        # 这里强制转成 Path，避免 "'str' object has no attribute 'mkdir'"。
        tmp_dir = Path(str(self.sam2_video_temp_dir))
        self.sam2_video_temp_dir = tmp_dir
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # 清空临时目录，确保 video predictor 只看到这两帧。
        for old_file in tmp_dir.glob("*.jpg"):
            try:
                old_file.unlink()
            except Exception:
                pass

        frame0 = tmp_dir / "000000.jpg"
        frame1 = tmp_dir / "000001.jpg"
        cv2.imwrite(str(frame0), cv2.cvtColor(self.last_image_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame1), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        try:
            inference_state = self.video_predictor.init_state(video_path=str(tmp_dir))
        except Exception as e:
            debug["reason"] = f"init_state_failed:{e}"
            self.last_video_tracking_debug = debug
            return None

        try:
            self._video_add_object_prompt(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=self.sam2_video_a_obj_id,
                mask_bool=self.last_a_mask,
                center=self.last_a_center,
                bbox_xyxy=self.last_a_bbox,
            )
            self._video_add_object_prompt(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=self.sam2_video_b_obj_id,
                mask_bool=self.last_b_mask,
                center=self.last_b_center,
                bbox_xyxy=self.last_b_bbox,
            )
        except Exception as e:
            debug["reason"] = f"add_prompt_failed:{e}"
            self.last_video_tracking_debug = debug
            return None

        out_masks_by_id: Dict[int, np.ndarray] = {}
        try:
            with self.torch.inference_mode():
                for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                    if int(out_frame_idx) == 1:
                        out_masks_by_id = self._extract_video_masks_for_frame(
                            out_obj_ids=out_obj_ids,
                            out_mask_logits=out_mask_logits,
                            image_shape_hw=image_rgb.shape[:2],
                        )
                        break
        except Exception as e:
            debug["reason"] = f"propagate_failed:{e}"
            self.last_video_tracking_debug = debug
            return None
        finally:
            try:
                if hasattr(self.video_predictor, "reset_state"):
                    self.video_predictor.reset_state(inference_state)
            except Exception:
                pass

        a_mask = out_masks_by_id.get(int(self.sam2_video_a_obj_id))
        b_mask = out_masks_by_id.get(int(self.sam2_video_b_obj_id))
        if a_mask is None or b_mask is None:
            debug["reason"] = f"missing_obj_mask: ids={list(out_masks_by_id.keys())}"
            self.last_video_tracking_debug = debug
            return None

        a_q = self._validate_video_tracking_mask("A", a_mask, self.last_a_mask, self.last_a_center)
        b_q = self._validate_video_tracking_mask("B", b_mask, self.last_b_mask, self.last_b_center)
        debug["A_quality"] = a_q
        debug["B_quality"] = b_q

        if not a_q.get("ok", False):
            debug["reason"] = a_q.get("reason", "A_quality_failed")
            self.last_video_tracking_debug = debug
            return None
        if not b_q.get("ok", False):
            debug["reason"] = b_q.get("reason", "B_quality_failed")
            self.last_video_tracking_debug = debug
            return None

        debug["ok"] = True
        debug["reason"] = "sam2_video_tracking_success"
        self.last_video_tracking_debug = debug
        return a_mask.astype(bool), b_mask.astype(bool), debug

    def infer_abc(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        后续帧更新 A/B，但 C 不再重新分割。

        A/B：
            用上一帧 bbox + 从上一帧 mask 自动采样的多个正点更新。
            同时对 A/B 增加自动负点：
                - A 的负点：上一帧 B 区域、固定 C 区域中不属于 A 的采样点；
                - B 的负点：固定 C 中不属于上一帧 B 的 C 独有区域采样点。
            这样可以降低 B 把 C 独有区域一起吸进去的概率。

        C：
            直接返回第一次初始化时保存的 static_c_mask。
        """
        if not self.initialized:
            raise RuntimeError("SAM2ABCSegmenter 尚未初始化。")

        if self.last_a_bbox is None or self.last_b_bbox is None:
            raise RuntimeError("SAM2 A/B 上一帧 bbox 为空，无法继续跟踪。")

        if self.static_c_mask is None:
            raise RuntimeError("static_c_mask 为空：第一次 C 分割像素没有被记录。")

        # --------------------------------------------------------
        # 优先使用 SAM2 video predictor 跟踪 A/B。
        # video tracking 适合 A/B 平移 + 自转；质量门控只看面积和中心连续性。
        # 如果 video predictor 不可用或当前帧质量不合格，则按配置回退到
        # 上一帧局部模板 + SAM2 ImagePredictor refine。
        # --------------------------------------------------------
        video_result = self._infer_ab_by_sam2_video_two_frame(image_rgb)
        if video_result is not None:
            a_mask, b_mask, video_debug = video_result
            c_mask = self.static_c_mask.copy()

            self.last_a_mask = a_mask
            self.last_b_mask = b_mask
            self.last_a_bbox = mask_to_bbox_xyxy(a_mask)
            self.last_b_bbox = mask_to_bbox_xyxy(b_mask)
            self.last_a_center = mask_center(a_mask)
            self.last_b_center = mask_center(b_mask)

            self.last_a_temporal_debug = {"mode": "sam2_video", "video": video_debug}
            self.last_b_temporal_debug = {"mode": "sam2_video", "video": video_debug}

            # 当前帧成为下一帧的 anchor frame。
            self.last_image_rgb = image_rgb.copy()

            self.last_c_mask = c_mask
            self.last_c_bbox = self.static_c_bbox
            self.last_c_center = self.static_c_center

            return a_mask, b_mask, c_mask

        if self.use_sam2_video_tracking and not self.sam2_video_fallback_to_template:
            raise RuntimeError(f"SAM2 video tracking 失败且禁止回退: {self.last_video_tracking_debug}")

        # 回退方案：上一帧局部模板定位 + ImagePredictor refine。
        self.predictor.set_image(image_rgb)

        try:
            # --------------------------------------------------------
            # 先估计 A/B 当前帧的位置，再把预测中心/预测 bbox 交给 SAM2。
            # 只看 A/B：C 不参与 A/B 的实时分割提示和质量判断。
            # --------------------------------------------------------
            a_pred_bbox, a_pred_center, a_track_debug = self._estimate_bbox_by_template_match(
                name="A",
                image_rgb=image_rgb,
                last_bbox=self.last_a_bbox,
                last_center=self.last_a_center,
            )
            b_pred_bbox, b_pred_center, b_track_debug = self._estimate_bbox_by_template_match(
                name="B",
                image_rgb=image_rgb,
                last_bbox=self.last_b_bbox,
                last_center=self.last_b_center,
            )

            if self.temporal_stop_on_track_fail:
                if a_track_debug.get("enabled", False) and not bool(a_track_debug.get("track_ok", True)):
                    raise RuntimeError(f"A 位置模板匹配失败，拒绝更新 A/B 并暂停本帧运动: {a_track_debug}")
                if b_track_debug.get("enabled", False) and not bool(b_track_debug.get("track_ok", True)):
                    raise RuntimeError(f"B 位置模板匹配失败，拒绝更新 A/B 并暂停本帧运动: {b_track_debug}")

            a_pos = [a_pred_center]
            b_pos = [b_pred_center]

            # 不使用 C 作为 A/B 负点；也不使用旧 mask 内的大量旧像素点作为当前帧硬提示。
            a_neg: List[Vec2] = []
            b_neg: List[Vec2] = []

            a_prompt_box = expand_bbox_xyxy(
                a_pred_bbox,
                margin_px=self.temporal_search_margin_px,
                image_shape_hw=image_rgb.shape[:2],
            )
            b_prompt_box = expand_bbox_xyxy(
                b_pred_bbox,
                margin_px=self.temporal_search_margin_px,
                image_shape_hw=image_rgb.shape[:2],
            )

            if self.a_use_temporal_mask_selection:
                a_mask, a_debug = self._predict_a_mask_by_box_and_points_temporal(
                    box_xyxy=a_prompt_box,
                    positive_points=a_pos,
                    negative_points=a_neg,
                    reference_mask=self.last_a_mask,
                    reference_center=self.last_a_center,
                    static_c_mask=None,
                )
                a_debug["position_match"] = a_track_debug
                self.last_a_temporal_debug = a_debug
            else:
                a_mask = self._predict_mask_by_box_and_points(
                    box_xyxy=a_prompt_box,
                    positive_points=a_pos,
                    negative_points=a_neg,
                )
                self.last_a_temporal_debug = {"enabled": False, "position_match": a_track_debug}

            if self.b_use_temporal_mask_selection:
                b_mask, b_debug = self._predict_b_mask_by_box_and_points_temporal(
                    box_xyxy=b_prompt_box,
                    positive_points=b_pos,
                    negative_points=b_neg,
                    reference_mask=self.last_b_mask,
                    reference_center=self.last_b_center,
                    static_c_mask=None,
                )
                b_debug["position_match"] = b_track_debug
                self.last_b_temporal_debug = b_debug
            else:
                b_mask = self._predict_mask_by_box_and_points(
                    box_xyxy=b_prompt_box,
                    positive_points=b_pos,
                    negative_points=b_neg,
                )
                self.last_b_temporal_debug = {"enabled": False, "position_match": b_track_debug}

            if self.temporal_stop_on_track_fail:
                if self.last_a_temporal_debug.get("fallback_to_last_a", False):
                    raise RuntimeError(f"A temporal 候选质量不足，拒绝更新并暂停本帧运动: {self.last_a_temporal_debug.get('fallback_reason')}")
                if self.last_b_temporal_debug.get("fallback_to_last_b", False):
                    raise RuntimeError(f"B temporal 候选质量不足，拒绝更新并暂停本帧运动: {self.last_b_temporal_debug.get('fallback_reason')}")

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

            # 更新上一帧图像，用于下一帧模板匹配定位。
            self.last_image_rgb = image_rgb.copy()

            # C 始终保持第一次分割/加载结果，不更新 bbox/center/mask。
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

    def _sample_points_from_mask(
        self,
        mask_bool: Optional[np.ndarray],
        max_points: int = 6,
    ) -> List[Vec2]:
        """
        从任意 bool mask 中均匀采样若干点，用作自动负点或辅助点。
        """
        if mask_bool is None:
            return []

        ys, xs = np.where(mask_bool.astype(bool))
        if len(xs) == 0:
            return []

        x1, y1, x2, y2 = mask_to_bbox_xyxy(mask_bool.astype(bool))
        candidate_targets = [
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            (x1 + 0.20 * (x2 - x1), y1 + 0.20 * (y2 - y1)),
            (x1 + 0.80 * (x2 - x1), y1 + 0.20 * (y2 - y1)),
            (x1 + 0.20 * (x2 - x1), y1 + 0.80 * (y2 - y1)),
            (x1 + 0.80 * (x2 - x1), y1 + 0.80 * (y2 - y1)),
            (x1 + 0.50 * (x2 - x1), y1 + 0.20 * (y2 - y1)),
            (x1 + 0.50 * (x2 - x1), y1 + 0.80 * (y2 - y1)),
            (x1 + 0.20 * (x2 - x1), y1 + 0.50 * (y2 - y1)),
            (x1 + 0.80 * (x2 - x1), y1 + 0.50 * (y2 - y1)),
        ]

        points: List[Vec2] = []
        for tx, ty in candidate_targets:
            if len(points) >= max_points:
                break
            d2 = (xs.astype(float) - float(tx)) ** 2 + (ys.astype(float) - float(ty)) ** 2
            idx = int(np.argmin(d2))
            p = (float(xs[idx]), float(ys[idx]))
            if all(vec_len(vec_sub(p, q)) > 5.0 for q in points):
                points.append(p)

        return points[:max_points]

    def _score_temporal_candidate_mask(
        self,
        name: str,
        candidate_mask: np.ndarray,
        sam2_score: float,
        reference_mask: Optional[np.ndarray],
        reference_center: Optional[Vec2],
        static_c_mask: Optional[np.ndarray],
        template_mask: Optional[np.ndarray] = None,
        template_center: Optional[Vec2] = None,
        score_weight: float = 0.20,
        iou_weight: float = 0.00,
        shape_iou_weight: float = 0.00,
        area_weight: float = 0.45,
        center_weight: float = 0.45,
        aspect_weight: float = 0.00,
        bbox_size_weight: float = 0.00,
        c_leak_weight: float = 0.0,
        center_sigma_px: float = 80.0,
        area_ratio_min: float = 0.35,
        area_ratio_max: float = 2.80,
        min_shape_iou: float = 0.0,
        min_aspect_similarity: float = 0.0,
        min_bbox_size_similarity: float = 0.0,
        max_c_unique_leak_ratio: float = 1.0,
    ) -> Dict[str, Any]:
        """
        A/B 通用候选 mask 评分：适用于“目标会平移 + 自转”的情况。

        这一版不再用“第一帧 shape-IoU / bbox 宽高比 / bbox 尺寸”做硬约束，
        因为目标自转后这些量可能明显变化。分割质量门控只保留：
            1. 面积连续性：candidate_area / last_area 不能突变；
            2. 中心连续性：candidate_center 距离上一帧/预测位置不能突跳；
            3. SAM2 score 作为弱参考。

        C 不参与 A/B 的分割质量判断。
        """
        cand = candidate_mask.astype(bool)
        ref = reference_mask.astype(bool) if reference_mask is not None else None

        cand_area = float(cand.sum())
        ref_area = float(ref.sum()) if ref is not None else 0.0
        area_ratio = mask_area_ratio(cand, ref) if ref is not None else 1.0

        raw_score = float(sam2_score)
        score_norm = float(max(0.0, min(1.0, raw_score)))

        # 这些指标仍然记录到 debug，但不再作为硬拒绝条件。
        raw_iou = mask_iou(cand, ref)
        shape_iou = center_aligned_shape_iou(cand, ref, reference_center)
        area_sim = mask_area_similarity(cand, ref)
        center_sim = center_similarity(cand, reference_center, sigma_px=center_sigma_px)
        aspect_sim = bbox_aspect_similarity(cand, ref)
        bbox_size_sim = bbox_size_similarity(cand, ref)
        leak_ratio = 0.0

        tmpl = template_mask.astype(bool) if template_mask is not None else None
        template_shape_iou = center_aligned_shape_iou(cand, tmpl, template_center) if tmpl is not None else shape_iou
        template_area_ratio = mask_area_ratio(cand, tmpl) if tmpl is not None else area_ratio
        template_area_sim = mask_area_similarity(cand, tmpl) if tmpl is not None else area_sim
        template_aspect_sim = bbox_aspect_similarity(cand, tmpl) if tmpl is not None else aspect_sim
        template_bbox_size_sim = bbox_size_similarity(cand, tmpl) if tmpl is not None else bbox_size_sim

        hard_reject = False
        reject_reasons: List[str] = []

        if ref_area > 1e-8 and (area_ratio < float(area_ratio_min) or area_ratio > float(area_ratio_max)):
            hard_reject = True
            reject_reasons.append(f"area_ratio_out({area_ratio:.3f})")

        center_jump = None
        if reference_center is not None:
            try:
                cand_center = mask_center(cand)
                center_jump = vec_len(vec_sub(cand_center, reference_center))
                # 使用已有 center_jump_max_px 做硬门控：目标可移动，但不能一帧突跳过大。
                if center_jump > float(self.center_jump_max_px):
                    hard_reject = True
                    reject_reasons.append(f"center_jump_high({center_jump:.1f})")
            except Exception as e:
                hard_reject = True
                reject_reasons.append(f"center_compute_failed({e})")

        total = (
            float(score_weight) * score_norm
            + float(area_weight) * area_sim
            + float(center_weight) * center_sim
            + float(iou_weight) * raw_iou
            + float(shape_iou_weight) * shape_iou
            + float(aspect_weight) * aspect_sim
            + float(bbox_size_weight) * bbox_size_sim
        )

        if hard_reject:
            total -= 100.0

        return {
            "score_total": float(total),
            "hard_reject": bool(hard_reject),
            "reject_reason": ";".join(reject_reasons),
            "sam2_score": raw_score,
            "score_norm": score_norm,
            "iou": float(raw_iou),
            "shape_iou": float(shape_iou),
            "template_shape_iou": float(template_shape_iou),
            "area_similarity": float(area_sim),
            "template_area_similarity": float(template_area_sim),
            "center_similarity": float(center_sim),
            "center_jump_px": None if center_jump is None else float(center_jump),
            "aspect_similarity": float(aspect_sim),
            "bbox_size_similarity": float(bbox_size_sim),
            "template_aspect_similarity": float(template_aspect_sim),
            "template_bbox_size_similarity": float(template_bbox_size_sim),
            "c_unique_leak_ratio": float(leak_ratio),
            "area_ratio": float(area_ratio),
            "template_area_ratio": float(template_area_ratio),
            "candidate_area": cand_area,
            "reference_area": ref_area,
            "name": name,
        }

    def _score_a_candidate_mask(
        self,
        candidate_mask: np.ndarray,
        sam2_score: float,
        reference_mask: Optional[np.ndarray],
        reference_center: Optional[Vec2],
        static_c_mask: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        return self._score_temporal_candidate_mask(
            name="A",
            candidate_mask=candidate_mask,
            sam2_score=sam2_score,
            reference_mask=reference_mask,
            reference_center=reference_center,
            static_c_mask=static_c_mask,
            template_mask=self.template_a_mask,
            template_center=self.init_a_center,
            score_weight=self.a_temporal_score_weight,
            iou_weight=self.a_temporal_iou_weight,
            shape_iou_weight=self.a_temporal_shape_iou_weight,
            area_weight=self.a_temporal_area_weight,
            center_weight=self.a_temporal_center_weight,
            aspect_weight=self.a_temporal_aspect_weight,
            bbox_size_weight=self.a_temporal_bbox_size_weight,
            c_leak_weight=self.a_temporal_c_leak_weight,
            center_sigma_px=self.a_temporal_center_sigma_px,
            area_ratio_min=self.a_temporal_area_ratio_min,
            area_ratio_max=self.a_temporal_area_ratio_max,
            min_shape_iou=self.a_temporal_min_shape_iou,
            min_aspect_similarity=self.a_temporal_min_aspect_similarity,
            min_bbox_size_similarity=self.a_temporal_min_bbox_size_similarity,
            max_c_unique_leak_ratio=self.a_temporal_max_c_unique_leak_ratio,
        )

    def _predict_a_mask_by_box_and_points_temporal(
        self,
        box_xyxy: Tuple[float, float, float, float],
        positive_points: Optional[List[Optional[Vec2]]] = None,
        negative_points: Optional[List[Optional[Vec2]]] = None,
        reference_mask: Optional[np.ndarray] = None,
        reference_center: Optional[Vec2] = None,
        static_c_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        A 专用后续帧 SAM2 分割。

        SAM2 返回多个候选 mask 时，不直接选最高 SAM2 score，
        而是选“最像上一帧 A”的候选 mask。
        """
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
            raise RuntimeError("SAM2 A temporal prompt 没有返回 mask。")

        scores_arr = np.asarray(scores).reshape(-1)
        candidates: List[Dict[str, Any]] = []
        for i, m in enumerate(masks):
            cand_mask = m.astype(bool)
            sam2_score = float(scores_arr[i]) if i < len(scores_arr) else 0.0
            info = self._score_a_candidate_mask(
                candidate_mask=cand_mask,
                sam2_score=sam2_score,
                reference_mask=reference_mask,
                reference_center=reference_center,
                static_c_mask=static_c_mask,
            )
            info["index"] = int(i)
            candidates.append(info)

        best = max(candidates, key=lambda d: d["score_total"])
        best_idx = int(best["index"])
        best_score = float(best["score_total"])

        debug = {
            "enabled": True,
            "best_index": best_idx,
            "best_score_total": best_score,
            "best": best,
            "candidates": candidates,
            "num_candidates": len(candidates),
        }

        if (
            self.a_temporal_fallback_to_last_if_bad
            and reference_mask is not None
            and best_score < self.a_temporal_min_accept_score
        ):
            debug["fallback_to_last_a"] = True
            debug["fallback_reason"] = (
                f"best_score_total={best_score:.3f} < "
                f"min_accept={self.a_temporal_min_accept_score:.3f}"
            )
            logger.warning(
                "A temporal mask 质量过低，使用上一帧 A 兜底: %s",
                debug["fallback_reason"],
            )
            return reference_mask.astype(bool).copy(), debug

        debug["fallback_to_last_a"] = False
        return masks[best_idx].astype(bool), debug


    def _score_b_candidate_mask(
        self,
        candidate_mask: np.ndarray,
        sam2_score: float,
        reference_mask: Optional[np.ndarray],
        reference_center: Optional[Vec2],
        static_c_mask: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        return self._score_temporal_candidate_mask(
            name="B",
            candidate_mask=candidate_mask,
            sam2_score=sam2_score,
            reference_mask=reference_mask,
            reference_center=reference_center,
            static_c_mask=static_c_mask,
            template_mask=self.template_b_mask,
            template_center=self.init_b_center,
            score_weight=self.b_temporal_score_weight,
            iou_weight=self.b_temporal_iou_weight,
            shape_iou_weight=self.b_temporal_shape_iou_weight,
            area_weight=self.b_temporal_area_weight,
            center_weight=self.b_temporal_center_weight,
            aspect_weight=self.b_temporal_aspect_weight,
            bbox_size_weight=self.b_temporal_bbox_size_weight,
            c_leak_weight=self.b_temporal_c_leak_weight,
            center_sigma_px=self.b_temporal_center_sigma_px,
            area_ratio_min=self.b_temporal_area_ratio_min,
            area_ratio_max=self.b_temporal_area_ratio_max,
            min_shape_iou=self.b_temporal_min_shape_iou,
            min_aspect_similarity=self.b_temporal_min_aspect_similarity,
            min_bbox_size_similarity=self.b_temporal_min_bbox_size_similarity,
            max_c_unique_leak_ratio=self.b_temporal_max_c_unique_leak_ratio,
        )

    def _predict_b_mask_by_box_and_points_temporal(
        self,
        box_xyxy: Tuple[float, float, float, float],
        positive_points: Optional[List[Optional[Vec2]]] = None,
        negative_points: Optional[List[Optional[Vec2]]] = None,
        reference_mask: Optional[np.ndarray] = None,
        reference_center: Optional[Vec2] = None,
        static_c_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        B 专用后续帧 SAM2 分割。

        与普通 _predict_mask_by_box_and_points 不同：
        SAM2 返回多个候选 mask 时，不直接选最高 SAM2 score，
        而是选“最像上一帧 B”的候选 mask。
        """
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
            raise RuntimeError("SAM2 B temporal prompt 没有返回 mask。")

        scores_arr = np.asarray(scores).reshape(-1)
        candidates: List[Dict[str, Any]] = []
        for i, m in enumerate(masks):
            cand_mask = m.astype(bool)
            sam2_score = float(scores_arr[i]) if i < len(scores_arr) else 0.0
            info = self._score_b_candidate_mask(
                candidate_mask=cand_mask,
                sam2_score=sam2_score,
                reference_mask=reference_mask,
                reference_center=reference_center,
                static_c_mask=static_c_mask,
            )
            info["index"] = int(i)
            candidates.append(info)

        best = max(candidates, key=lambda d: d["score_total"])
        best_idx = int(best["index"])
        best_score = float(best["score_total"])

        debug = {
            "enabled": True,
            "best_index": best_idx,
            "best_score_total": best_score,
            "best": best,
            "candidates": candidates,
            "num_candidates": len(candidates),
        }

        if (
            self.b_temporal_fallback_to_last_if_bad
            and reference_mask is not None
            and best_score < self.b_temporal_min_accept_score
        ):
            debug["fallback_to_last_b"] = True
            debug["fallback_reason"] = (
                f"best_score_total={best_score:.3f} < "
                f"min_accept={self.b_temporal_min_accept_score:.3f}"
            )
            logger.warning(
                "B temporal mask 质量过低，使用上一帧 B 兜底: %s",
                debug["fallback_reason"],
            )
            return reference_mask.astype(bool).copy(), debug

        debug["fallback_to_last_b"] = False
        return masks[best_idx].astype(bool), debug


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
    window_name: str = "SAM2 init: A/B/C multi positive + negative",
    scale: float = 0.85,
) -> Tuple[PointPromptSet, PointPromptSet, PointPromptSet]:
    """
    第一帧初始化点选择。

    修改后的规则：
        A：多个正点 + 多个负点；
        B：多个正点 + 多个负点；
        C：多个正点 + 多个负点。

    操作方式：
        当前目标依次为 A -> B -> C。
        左键：给当前目标添加正点。
        右键：给当前目标添加负点。
        N 或 Enter：完成当前目标，进入下一个目标。
        R：清空当前目标。
        Z：撤销当前目标最后一个点，优先撤销负点，没有负点时撤销正点。
        Backspace：回到上一个目标。
        ESC：取消。

    点提示原则：
        - 正点：点在“确定属于当前目标”的内部区域，尽量不要点边界。
        - 负点：点在“确定不属于当前目标”的区域。
        - B/C 有真实重合时，不要把 B/C 重合区域点成 B 或 C 的负点。
          B 负点应点在 C 独有区域或背景；C 负点应点在 A 或背景。
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
        "A": (0, 255, 255),    # 黄色
        "B": (0, 128, 255),    # 橙色
        "C": (255, 255, 0),    # 青色
    }

    prompt_sets: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        name: {"pos": [], "neg": []} for name in names
    }
    current_idx = 0

    def redraw() -> np.ndarray:
        canvas = display.copy()
        current_name = names[current_idx]

        instructions = [
            "SAM2 init: A/B/C all use positive and negative points",
            f"Current: {current_name} | Left=positive, Right=negative, N/Enter=next, R=reset, Z=undo, Backspace=previous, ESC=cancel",
            "Do NOT put negative points on real B/C overlap. Use negatives only on regions that definitely do not belong to current object.",
        ]
        for i, text_line in enumerate(instructions):
            cv2.putText(
                canvas,
                text_line,
                (20, 28 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        y_panel = 110
        for obj_i, name in enumerate(names):
            prefix = ">" if obj_i == current_idx else " "
            count_text = (
                f"{prefix}{name}: pos={len(prompt_sets[name]['pos'])}, "
                f"neg={len(prompt_sets[name]['neg'])}"
            )
            cv2.putText(
                canvas,
                count_text,
                (20, y_panel + obj_i * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
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
                x, y = int(p[0]), int(p[1])
                cv2.circle(canvas, (x, y), 7, (0, 0, 255), 2)
                cv2.line(canvas, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
                cv2.line(canvas, (x - 5, y + 5), (x + 5, y - 5), (0, 0, 255), 2)
                cv2.putText(
                    canvas,
                    f"{name}-{j + 1}",
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
            raise RuntimeError("用户取消了 A/B/C 初始化。")

        if key in (ord("r"), ord("R")):
            name = names[current_idx]
            prompt_sets[name]["pos"].clear()
            prompt_sets[name]["neg"].clear()

        if key in (ord("z"), ord("Z")):
            name = names[current_idx]
            if prompt_sets[name]["neg"]:
                prompt_sets[name]["neg"].pop()
            elif prompt_sets[name]["pos"]:
                prompt_sets[name]["pos"].pop()

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
        "用户初始化: A(pos=%d, neg=%d), B(pos=%d, neg=%d), C(pos=%d, neg=%d, fixed static mask after segmentation)",
        len(a_prompt.positive_points),
        len(a_prompt.negative_points),
        len(b_prompt.positive_points),
        len(b_prompt.negative_points),
        len(c_prompt.positive_points),
        len(c_prompt.negative_points),
    )

    return a_prompt, b_prompt, c_prompt


def select_a_b_points_interactively(
    image_rgb: np.ndarray,
    window_name: str = "SAM2 init: A/B only, reuse static C map",
    scale: float = 0.85,
) -> Tuple[PointPromptSet, PointPromptSet]:
    """
    复用固定 C map 时，只需要重新选择 A、B 的正点和负点。

    操作方式：
        当前目标依次为 A -> B。
        左键：给当前目标添加正点。
        右键：给当前目标添加负点。
        N 或 Enter：完成当前目标，进入下一个目标。
        R：清空当前目标。
        Z：撤销当前目标最后一个点，优先撤销负点。
        Backspace：回到上一个目标。
        ESC：取消。
    """
    if scale <= 0:
        scale = 1.0

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    show_w = int(w * scale)
    show_h = int(h * scale)
    display = cv2.resize(image_bgr, (show_w, show_h), interpolation=cv2.INTER_AREA)

    names = ["A", "B"]
    colors = {"A": (0, 255, 255), "B": (0, 128, 255)}
    prompt_sets: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        name: {"pos": [], "neg": []} for name in names
    }
    current_idx = 0

    def redraw() -> np.ndarray:
        canvas = display.copy()
        current_name = names[current_idx]
        instructions = [
            "SAM2 init: reuse confirmed C map, click A/B only",
            f"Current: {current_name} | Left=positive, Right=negative, N/Enter=next, R=reset, Z=undo, Backspace=previous, ESC=cancel",
            "C is loaded from confirmed static map; do not re-click C unless force_reselect_c_each_run=True.",
        ]
        for i, text_line in enumerate(instructions):
            cv2.putText(canvas, text_line, (20, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2, cv2.LINE_AA)

        y_panel = 110
        for obj_i, name in enumerate(names):
            prefix = ">" if obj_i == current_idx else " "
            count_text = f"{prefix}{name}: pos={len(prompt_sets[name]['pos'])}, neg={len(prompt_sets[name]['neg'])}"
            cv2.putText(canvas, count_text, (20, y_panel + obj_i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors[name], 2, cv2.LINE_AA)

        for name in names:
            color = colors[name]
            for j, p in enumerate(prompt_sets[name]["pos"]):
                cv2.circle(canvas, p, 6, color, -1)
                cv2.putText(canvas, f"{name}+{j + 1}", (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            for j, p in enumerate(prompt_sets[name]["neg"]):
                x, y = int(p[0]), int(p[1])
                cv2.circle(canvas, (x, y), 7, (0, 0, 255), 2)
                cv2.line(canvas, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
                cv2.line(canvas, (x - 5, y + 5), (x + 5, y - 5), (0, 0, 255), 2)
                cv2.putText(canvas, f"{name}-{j + 1}", (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
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
            raise RuntimeError("用户取消了 A/B 初始化。")
        if key in (ord("r"), ord("R")):
            name = names[current_idx]
            prompt_sets[name]["pos"].clear()
            prompt_sets[name]["neg"].clear()
        if key in (ord("z"), ord("Z")):
            name = names[current_idx]
            if prompt_sets[name]["neg"]:
                prompt_sets[name]["neg"].pop()
            elif prompt_sets[name]["pos"]:
                prompt_sets[name]["pos"].pop()
        if key in (8, 127):
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

    logger.info(
        "用户初始化 A/B: A(pos=%d, neg=%d), B(pos=%d, neg=%d)，C 使用固定 map",
        len(a_prompt.positive_points), len(a_prompt.negative_points),
        len(b_prompt.positive_points), len(b_prompt.negative_points),
    )
    return a_prompt, b_prompt

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
    draw_center(scene.c.center, (255, 255, 0), "C-body")

    # B 的目标边用红线显示。
    b1, b2 = scene.b_target_edge
    cv2.line(
        img,
        (int(b1[0]), int(b1[1])),
        (int(b2[0]), int(b2[1])),
        (0, 0, 255),
        4,
    )

    ab_overlap = mask_overlap_area(scene.a.mask, scene.b.mask)
    bc_overlap = mask_overlap_area(scene.b.mask, scene.c.mask)
    ac_overlap = mask_overlap_area(scene.a.mask, scene.c.mask)

    text_lines = [
        "Confirm SAM2 first frame segmentation",
        "Y / Enter = accept and start motion",
        "R = reject and re-click A/B/C",
        "ESC / Q = cancel",
        f"A area={scene.a.area:.1f}, B area={scene.b.area:.1f}, C area={scene.c.area:.1f}",
        f"overlap: AB={ab_overlap:.1f}, BC={bc_overlap:.1f}, AC={ac_overlap:.1f}",
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
            method=getattr(cfg, "c_quadrilateral_method", "body_edge"),
        )
        c_source = "sam2_body_edge"
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
    _SHARED_STATIC_C_MAPS: Dict[str, Dict[str, Any]] = {}

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
        self.base_output_dir = base_output_dir
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
            a_use_temporal_mask_selection=cfg.a_use_temporal_mask_selection,
            a_temporal_score_weight=cfg.a_temporal_score_weight,
            a_temporal_iou_weight=cfg.a_temporal_iou_weight,
            a_temporal_shape_iou_weight=cfg.a_temporal_shape_iou_weight,
            a_temporal_area_weight=cfg.a_temporal_area_weight,
            a_temporal_center_weight=cfg.a_temporal_center_weight,
            a_temporal_aspect_weight=cfg.a_temporal_aspect_weight,
            a_temporal_bbox_size_weight=cfg.a_temporal_bbox_size_weight,
            a_temporal_c_leak_weight=0.0,
            a_temporal_center_sigma_px=cfg.a_temporal_center_sigma_px,
            a_temporal_area_ratio_min=cfg.a_temporal_area_ratio_min,
            a_temporal_area_ratio_max=cfg.a_temporal_area_ratio_max,
            a_temporal_min_shape_iou=cfg.a_temporal_min_shape_iou,
            a_temporal_min_aspect_similarity=cfg.a_temporal_min_aspect_similarity,
            a_temporal_min_bbox_size_similarity=cfg.a_temporal_min_bbox_size_similarity,
            a_temporal_max_c_unique_leak_ratio=1.0,
            a_temporal_fallback_to_last_if_bad=cfg.a_temporal_fallback_to_last_if_bad,
            a_temporal_min_accept_score=cfg.a_temporal_min_accept_score,
            b_use_temporal_mask_selection=cfg.b_use_temporal_mask_selection,
            b_temporal_score_weight=cfg.b_temporal_score_weight,
            b_temporal_iou_weight=cfg.b_temporal_iou_weight,
            b_temporal_shape_iou_weight=cfg.b_temporal_shape_iou_weight,
            b_temporal_area_weight=cfg.b_temporal_area_weight,
            b_temporal_center_weight=cfg.b_temporal_center_weight,
            b_temporal_aspect_weight=cfg.b_temporal_aspect_weight,
            b_temporal_bbox_size_weight=cfg.b_temporal_bbox_size_weight,
            b_temporal_c_leak_weight=0.0,
            b_temporal_center_sigma_px=cfg.b_temporal_center_sigma_px,
            b_temporal_area_ratio_min=cfg.b_temporal_area_ratio_min,
            b_temporal_area_ratio_max=cfg.b_temporal_area_ratio_max,
            b_temporal_min_shape_iou=cfg.b_temporal_min_shape_iou,
            b_temporal_min_aspect_similarity=cfg.b_temporal_min_aspect_similarity,
            b_temporal_min_bbox_size_similarity=cfg.b_temporal_min_bbox_size_similarity,
            b_temporal_max_c_unique_leak_ratio=1.0,
            b_temporal_fallback_to_last_if_bad=cfg.b_temporal_fallback_to_last_if_bad,
            b_temporal_min_accept_score=cfg.b_temporal_min_accept_score,
            temporal_search_margin_px=cfg.temporal_search_margin_px,
            temporal_use_center_point_only=cfg.temporal_use_center_point_only,
            temporal_position_match_enable=cfg.temporal_position_match_enable,
            temporal_position_match_margin_px=cfg.temporal_position_match_margin_px,
            temporal_position_match_min_score=cfg.temporal_position_match_min_score,
            temporal_position_match_use_last_image_template=cfg.temporal_position_match_use_last_image_template,
        )

        self.stage: Optional[Stage34] = None

        # 当前运行所使用的固定 C 地图。
        # 如果 GUI 没关，可以来自 _SHARED_STATIC_C_MAPS；如果 GUI 关了，可以从磁盘加载。
        self.static_c_map_record: Optional[Dict[str, Any]] = None
        self.static_c_map_contour: Optional[List[Vec2]] = None
        self.static_c_map_center: Optional[Vec2] = None
        self.static_c_map_area: Optional[float] = None
        self.static_c_map_source: str = ""

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
            self._configure_stage_channels_if_supported()
            logger.info("Stage34 已连接。")
        else:
            logger.warning("enable_stage=False：当前只计算动作，不实际移动。")

        self.history: List[Dict[str, Any]] = []
        self.cycle_index = 0
        self.lost_count = 0

    # --------------------------------------------------------
    # Stage / 固定 C 地图工具
    # --------------------------------------------------------

    def _configure_stage_channels_if_supported(self) -> None:
        """
        尝试分别设置 3/4 通道速度、加速度、最大电压。

        如果 4 通道 velocity/acceleration 已经为 1 但仍过快，建议优先：
            1. 降低 action_step / stage_step_x / stage_step_y；
            2. 适当降低对应通道 max_voltage；
            3. 增大 loop_interval_s。
        """
        if self.stage is None:
            return

        ch3_v = int(getattr(self.cfg, "stage_ch3_velocity", self.cfg.stage_default_velocity))
        ch3_a = int(getattr(self.cfg, "stage_ch3_acceleration", self.cfg.stage_default_acceleration))
        ch3_mv = int(getattr(self.cfg, "stage_ch3_max_voltage", self.cfg.stage_default_max_voltage))
        ch4_v = int(getattr(self.cfg, "stage_ch4_velocity", self.cfg.stage_default_velocity))
        ch4_a = int(getattr(self.cfg, "stage_ch4_acceleration", self.cfg.stage_default_acceleration))
        ch4_mv = int(getattr(self.cfg, "stage_ch4_max_voltage", self.cfg.stage_default_max_voltage))

        try:
            if hasattr(self.stage, "setup_channel"):
                self.stage.setup_channel(channel=3, max_voltage=ch3_mv, velocity=ch3_v, acceleration=ch3_a)
                self.stage.setup_channel(channel=4, max_voltage=ch4_mv, velocity=ch4_v, acceleration=ch4_a)
                logger.info(
                    "已分别设置 Stage34: CH3(v=%s,a=%s,V=%s), CH4(v=%s,a=%s,V=%s)",
                    ch3_v, ch3_a, ch3_mv, ch4_v, ch4_a, ch4_mv,
                )
                return

            dev = getattr(self.stage, "dev", None)
            if dev is not None and hasattr(dev, "setup_drive"):
                dev.setup_drive(max_voltage=ch3_mv, velocity=ch3_v, acceleration=ch3_a, channel=3)
                dev.setup_drive(max_voltage=ch4_mv, velocity=ch4_v, acceleration=ch4_a, channel=4)
                logger.info(
                    "已通过 stage.dev 分别设置 CH3/CH4: CH3(v=%s,a=%s,V=%s), CH4(v=%s,a=%s,V=%s)",
                    ch3_v, ch3_a, ch3_mv, ch4_v, ch4_a, ch4_mv,
                )
                return

            logger.warning("当前 Stage34 对象没有 setup_channel/setup_drive 接口，无法单独设置 CH3/CH4 电压。")
        except Exception as e:
            logger.warning("分别设置 CH3/CH4 速度/加速度/电压失败: %s", e)

    def _static_c_map_key(self) -> str:
        name = str(getattr(self.cfg, "static_c_map_name", "default_static_c_map")).strip() or "default_static_c_map"
        custom = str(getattr(self.cfg, "static_c_map_dir", "")).strip()
        if custom:
            try:
                return f"{str(Path(custom).resolve())}::{name}"
            except Exception:
                return f"{custom}::{name}"
        return f"{str(self.base_output_dir.resolve())}::{name}"

    def _static_c_map_dir(self) -> Path:
        """
        返回固定 C map 目录。

        支持两种用法：
            1. static_c_map_dir 直接指向旧版 static_c_reference 文件夹；
            2. static_c_map_dir 指向 run_xxx 文件夹，内部包含 static_c_reference；
            3. static_c_map_dir 为空时，使用稳定库目录
               output_dir/_static_c_map_library/static_c_map_name。
        """
        custom = str(getattr(self.cfg, "static_c_map_dir", "")).strip().strip('"').strip("'")
        if custom:
            p = Path(custom)
            if (p / "static_c_mask.png").exists():
                return p
            if (p / "static_c_reference" / "static_c_mask.png").exists():
                return p / "static_c_reference"
            return p

        name = str(getattr(self.cfg, "static_c_map_name", "default_static_c_map")).strip() or "default_static_c_map"
        return self.base_output_dir / "_static_c_map_library" / name

    def _find_latest_static_c_reference_dir(self) -> Optional[Path]:
        """当没有显式选择目录时，在 output_dir 下寻找最近一次旧版 static_c_reference。"""
        try:
            candidates = []
            for mask_path in self.base_output_dir.rglob("static_c_mask.png"):
                d = mask_path.parent
                if d.name == "static_c_reference" or (d / "static_c_pixels_yx.npy").exists():
                    candidates.append(d)
            if not candidates:
                return None
            candidates.sort(key=lambda x: (x / "static_c_mask.png").stat().st_mtime, reverse=True)
            return candidates[0]
        except Exception:
            return None

    def _record_static_c_map(self, record: Dict[str, Any]) -> None:
        self.static_c_map_record = record
        self.static_c_map_contour = [(float(x), float(y)) for x, y in record["contour"]]
        self.static_c_map_center = (float(record["center"][0]), float(record["center"][1]))
        self.static_c_map_area = float(record.get("area", float(record["mask"].sum())))
        self.static_c_map_source = str(record.get("source", "static_c_map"))

        if bool(getattr(self.cfg, "reuse_static_c_map_in_memory", True)):
            ActualNanoBoundaryFollower._SHARED_STATIC_C_MAPS[self._static_c_map_key()] = record

    def _get_shared_static_c_map(self) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self.cfg, "reuse_static_c_map_in_memory", True)):
            return None
        rec = ActualNanoBoundaryFollower._SHARED_STATIC_C_MAPS.get(self._static_c_map_key())
        if rec is not None:
            logger.info("已从当前 GUI 进程内存复用固定 C map: %s", self._static_c_map_key())
        return rec

    def load_static_c_map_from_disk(self) -> Optional[Dict[str, Any]]:
        """
        从磁盘加载固定 C map。

        兼容两类目录：
            1. 新版库目录：static_c_map_meta.json + static_c_mask.png + static_c_pixels_yx.npy；
            2. 旧版 static_c_reference 目录：static_c_mask.png + static_c_pixels_yx.npy + static_c_reference.csv。

        旧版目录没有 static_c_map_meta.json 时，会从 static_c_mask.png 重新计算
        C 的主体边缘轮廓、中心、bbox，并自动补写 meta 文件。
        """
        if not bool(getattr(self.cfg, "load_static_c_map_if_exists", True)):
            return None

        static_dir = self._static_c_map_dir()
        custom_static_c_dir = str(getattr(self.cfg, "static_c_map_dir", "") or "").strip().strip('"').strip("'")
        if not (static_dir / "static_c_mask.png").exists():
            # 关键修复：如果主程序已经显式传入 static_c_map_dir，尤其是完整测量的
            # full_calibration_static_c，则绝不能回退到 output_dir 下最近的旧 C。
            # 否则会重新加载 RuleAB/RuleAC 单独测试区域残留的 C。
            if custom_static_c_dir:
                logger.error("显式指定的 static_c_map_dir 中没有 static_c_mask.png，禁止回退到最近旧 C: %s", static_dir)
                return None
            latest = self._find_latest_static_c_reference_dir()
            if latest is not None:
                logger.info("未在指定目录找到 C map，自动使用最近的 static_c_reference: %s", latest)
                static_dir = latest

        meta_path = static_dir / "static_c_map_meta.json"
        mask_path = static_dir / "static_c_mask.png"
        pixels_path = static_dir / "static_c_pixels_yx.npy"

        if not mask_path.exists():
            logger.info("未找到固定 C mask: %s", mask_path)
            return None

        try:
            mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_u8 is None:
                raise RuntimeError(f"无法读取 C mask: {mask_path}")
            mask = mask_u8 > 0

            meta: Dict[str, Any] = {}
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)

            if meta.get("contour") is not None and meta.get("center") is not None:
                contour = [(float(x), float(y)) for x, y in meta["contour"]]
                center = (float(meta["center"][0]), float(meta["center"][1]))
                area = float(meta.get("area", float(mask.sum())))
                bbox = tuple(meta.get("bbox", mask_to_bbox_xyxy(mask)))
                source = str(meta.get("source", "static_c_map_loaded"))
            else:
                # 旧版 static_c_reference 只有 mask/npy/csv：从 mask 重新计算 C 主体边缘轮廓。
                contour, area, center = mask_to_quadrilateral_contour(
                    mask,
                    min_area=float(getattr(self.cfg, "min_mask_area_px", 50.0)),
                    epsilon_ratio=max(0.012, float(getattr(self.cfg, "contour_approx_epsilon_ratio", 0.006)) * 3.0),
                    method=str(getattr(self.cfg, "c_quadrilateral_method", "body_edge")),
                )
                bbox = mask_to_bbox_xyxy(mask)
                if static_dir.name == "full_calibration_static_c":
                    source = "full_calibration_static_c"
                else:
                    source = "static_c_map_loaded_from_legacy_reference"

                # 自动补写 meta，后续加载更快且轮廓固定一致。
                try:
                    auto_meta = {
                        "created_time": time.time(),
                        "static_c_map_name": str(getattr(self.cfg, "static_c_map_name", "default_static_c_map")),
                        "bbox": list(bbox),
                        "center": list(center),
                        "area": float(area),
                        "source": source,
                        "contour": [[float(x), float(y)] for x, y in contour],
                        "mask_path": str(mask_path),
                        "pixels_path": str(pixels_path),
                        "note": "auto-generated from legacy static_c_reference folder",
                    }
                    with meta_path.open("w", encoding="utf-8") as f:
                        json.dump(auto_meta, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning("自动补写 static_c_map_meta.json 失败，不影响本次加载: %s", e)

            pixels_yx = np.load(str(pixels_path)) if pixels_path.exists() else np.column_stack(np.where(mask))

            record = {
                "mask": mask,
                "pixels_yx": pixels_yx.astype(np.int32),
                "contour": contour,
                "center": center,
                "bbox": tuple(float(v) for v in bbox),
                "area": float(area),
                "source": source,
                "meta_path": str(meta_path),
                "mask_path": str(mask_path),
                "static_dir": str(static_dir),
            }
            self._record_static_c_map(record)
            logger.info("已从磁盘加载固定 C map: %s", static_dir)
            return record
        except Exception as e:
            logger.warning("加载固定 C map 失败，将重新标注 C: %s", e)
            return None

    def get_reusable_static_c_map(self) -> Optional[Dict[str, Any]]:
        if bool(getattr(self.cfg, "force_reselect_c_each_run", False)):
            logger.info("force_reselect_c_each_run=True：忽略已有 C map，重新标注 C。")
            return None
        return self._get_shared_static_c_map() or self.load_static_c_map_from_disk()

    def save_confirmed_static_c_map(self, scene: SceneGeometry, c_mask: np.ndarray, image_rgb: Optional[np.ndarray] = None) -> None:
        """保存用户确认后的 C map 到内存和磁盘，供后续只点击 A/B 使用。"""
        c_bool = c_mask.astype(bool).copy()
        pixels_yx = np.column_stack(np.where(c_bool)).astype(np.int32)
        record = {
            "mask": c_bool,
            "pixels_yx": pixels_yx,
            "contour": [(float(x), float(y)) for x, y in scene.c.contour],
            "center": (float(scene.c.center[0]), float(scene.c.center[1])),
            "bbox": tuple(float(v) for v in scene.c.bbox_xyxy),
            "area": float(scene.c.area),
            "source": "static_c_map_confirmed",
        }
        self._record_static_c_map(record)

        if not bool(getattr(self.cfg, "save_static_c_map_library", True)):
            return

        static_dir = self._static_c_map_dir()
        static_dir.mkdir(parents=True, exist_ok=True)
        mask_path = static_dir / "static_c_mask.png"
        pixels_path = static_dir / "static_c_pixels_yx.npy"
        csv_path = static_dir / "static_c_reference.csv"
        meta_path = static_dir / "static_c_map_meta.json"
        preview_path = static_dir / "static_c_preview.png"

        cv2.imwrite(str(mask_path), c_bool.astype(np.uint8) * 255)
        np.save(str(pixels_path), pixels_yx)

        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "y", "x"])
            for idx, (y, x) in enumerate(pixels_yx):
                writer.writerow([idx, int(y), int(x)])

        meta = {
            "created_time": time.time(),
            "static_c_map_name": str(getattr(self.cfg, "static_c_map_name", "default_static_c_map")),
            "capture_area": list(getattr(self.cfg, "capture_area", [])),
            "image_shape": list(scene.image_shape),
            "bbox": list(record["bbox"]),
            "center": list(record["center"]),
            "area": record["area"],
            "source": record["source"],
            "contour": [[float(x), float(y)] for x, y in record["contour"]],
            "mask_path": str(mask_path),
            "pixels_path": str(pixels_path),
            "csv_path": str(csv_path),
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if image_rgb is not None:
            img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            overlay = img.copy()
            overlay[c_bool] = (255, 255, 0)
            img = cv2.addWeighted(overlay, 0.30, img, 0.70, 0)
            pts = np.asarray(record["contour"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (255, 255, 0), 3)
            cv2.imwrite(str(preview_path), img)

        logger.info("确认 C map 已保存到库: %s", static_dir)

    def apply_static_c_map_to_scene(self, scene: SceneGeometry) -> SceneGeometry:
        """把 scene.c 替换为固定 C map 的 contour/center/area，确保每帧 C 轮廓不变。"""
        if self.static_c_map_record is None:
            return scene

        rec = self.static_c_map_record
        scene.c.mask = rec["mask"].astype(bool).copy()
        scene.c.contour = [(float(x), float(y)) for x, y in rec["contour"]]
        scene.c.center = (float(rec["center"][0]), float(rec["center"][1]))
        scene.c.area = float(rec.get("area", float(scene.c.mask.sum())))
        scene.c.bbox_xyxy = tuple(float(v) for v in rec.get("bbox", mask_to_bbox_xyxy(scene.c.mask)))
        scene.c.source = str(rec.get("source", "static_c_map"))
        return scene

    # --------------------------------------------------------
    # 初始化 A/B/C
    # --------------------------------------------------------


    def _normalize_preloaded_points(self, value: Any) -> List[Vec2]:
        """把主程序/标定包注入的点统一转成 [(x,y), ...]。"""
        out: List[Vec2] = []
        if value is None:
            return out
        try:
            # 兼容 {"positive": [...], "negative": [...]} 误传进来的情况。
            if isinstance(value, dict):
                value = value.get("positive", [])
        except Exception:
            pass
        try:
            for p in value:
                try:
                    out.append((float(p[0]), float(p[1])))
                except Exception:
                    continue
        except Exception:
            return []
        return out

    def _first_nonempty_points(self, *names: str) -> List[Vec2]:
        """从 follower 和 cfg 的多个候选属性名中取第一组非空点。"""
        for obj in (self, getattr(self, "cfg", None), getattr(self, "abc_segmenter", None)):
            if obj is None:
                continue
            for name in names:
                try:
                    pts = self._normalize_preloaded_points(getattr(obj, name, None))
                    if pts:
                        return pts
                except Exception:
                    continue
        return []

    def _get_preloaded_prompt_sets(self) -> Tuple[PointPromptSet, PointPromptSet, PointPromptSet]:
        """
        从主 GUI 完整标定注入的字段中读取 A/B/C 正点负点。

        支持主程序注入的多种字段名：
            preloaded_a_positive_points / rule_ab_a_positive_points / a_positive_points / initial_a_positive_points
            preloaded_b_positive_points / rule_ab_b_positive_points / b_positive_points / initial_b_positive_points
            preloaded_c_positive_points / global_c_positive_points / c_positive_points / initial_c_positive_points
        这样外部模块不会再无视完整标定包而弹出 A/B/C 交互窗口。
        """
        a_pos = self._first_nonempty_points(
            "preloaded_a_positive_points", "rule_ab_a_positive_points", "a_positive_points", "initial_a_positive_points"
        )
        a_neg = self._first_nonempty_points(
            "preloaded_a_negative_points", "rule_ab_a_negative_points", "a_negative_points", "initial_a_negative_points"
        )
        b_pos = self._first_nonempty_points(
            "preloaded_b_positive_points", "rule_ab_b_positive_points", "b_positive_points", "initial_b_positive_points"
        )
        b_neg = self._first_nonempty_points(
            "preloaded_b_negative_points", "rule_ab_b_negative_points", "b_negative_points", "initial_b_negative_points"
        )
        c_pos = self._first_nonempty_points(
            "preloaded_c_positive_points", "global_c_positive_points", "c_positive_points", "initial_c_positive_points"
        )
        c_neg = self._first_nonempty_points(
            "preloaded_c_negative_points", "global_c_negative_points", "c_negative_points", "initial_c_negative_points"
        )
        return (
            PointPromptSet(positive_points=a_pos, negative_points=a_neg),
            PointPromptSet(positive_points=b_pos, negative_points=b_neg),
            PointPromptSet(positive_points=c_pos, negative_points=c_neg),
        )

    def _should_skip_interactive_from_preload(self) -> bool:
        """判断完整测量时是否应禁止弹出人工点选/确认窗口。"""
        for obj in (self, getattr(self, "cfg", None), getattr(self, "abc_segmenter", None)):
            if obj is None:
                continue
            for name in (
                "skip_interactive_selection",
                "skip_first_frame_confirmation",
                "use_precalibration",
                "use_full_calibration",
            ):
                try:
                    if bool(getattr(obj, name, False)):
                        return True
                except Exception:
                    pass
        return False

    def initialize_abc_with_calibration(
        self,
        calibration_state: Any = None,
        a_positive_points: Optional[List[Vec2]] = None,
        a_negative_points: Optional[List[Vec2]] = None,
        b_positive_points: Optional[List[Vec2]] = None,
        b_negative_points: Optional[List[Vec2]] = None,
        c_positive_points: Optional[List[Vec2]] = None,
        c_negative_points: Optional[List[Vec2]] = None,
        static_c_map_dir: str = "",
        **kwargs: Any,
    ) -> None:
        """
        非交互式初始化 A/B/C。

        这是给主 GUI 的“完整标定包”使用的入口：
            - 有 static C map 时：只用预加载 A/B 点分割 A/B，C 从 static map 读取；
            - 没有 static C map 时：用预加载 A/B/C 点一次性分割 A/B/C，并保存 static C；
            - 全程不调用 select_a_b_points_interactively / select_a_b_c_points_interactively；
            - 默认也不弹确认窗口。
        """
        if calibration_state is not None:
            try:
                data = calibration_state.to_dict() if hasattr(calibration_state, "to_dict") else calibration_state
                if isinstance(data, dict):
                    a_positive_points = a_positive_points or data.get("rule_ab_a_positive_points")
                    a_negative_points = a_negative_points or data.get("rule_ab_a_negative_points")
                    b_positive_points = b_positive_points or data.get("rule_ab_b_positive_points") or data.get("angle_b_positive_points")
                    b_negative_points = b_negative_points or data.get("rule_ab_b_negative_points") or data.get("angle_b_negative_points")
                    c_positive_points = c_positive_points or data.get("global_c_positive_points")
                    c_negative_points = c_negative_points or data.get("global_c_negative_points")
                    static_c_map_dir = static_c_map_dir or str(data.get("static_c_map_dir", "") or "")
            except Exception:
                pass

        # 参数显式传入时，先写入 self/cfg，后面的统一读取函数就能拿到。
        explicit_attrs = {
            "preloaded_a_positive_points": a_positive_points,
            "preloaded_a_negative_points": a_negative_points,
            "preloaded_b_positive_points": b_positive_points,
            "preloaded_b_negative_points": b_negative_points,
            "preloaded_c_positive_points": c_positive_points,
            "preloaded_c_negative_points": c_negative_points,
        }
        for k, v in explicit_attrs.items():
            if v is None:
                continue
            try:
                setattr(self, k, self._normalize_preloaded_points(v))
                setattr(self.cfg, k, self._normalize_preloaded_points(v))
            except Exception:
                pass

        if static_c_map_dir:
            try:
                self.cfg.static_c_map_dir = str(static_c_map_dir)
                self.cfg.load_static_c_map_if_exists = True
                self.cfg.force_reselect_c_each_run = False
            except Exception:
                pass

        image_rgb = self.capturer.capture()
        reusable_c = self.get_reusable_static_c_map()
        a_prompt, b_prompt, c_prompt = self._get_preloaded_prompt_sets()

        if not a_prompt.positive_points:
            raise RuntimeError("完整标定注入失败：缺少 A 正点，不能非交互初始化 RuleAB。")
        if not b_prompt.positive_points:
            raise RuntimeError("完整标定注入失败：缺少 B 正点，不能非交互初始化 RuleAB。")

        # 情况 1：已有 static C map。用 A/B 点 + 固定 C 初始化，不弹 A/B 点选窗口。
        if reusable_c is not None:
            self._record_static_c_map(reusable_c)
            logger.info(
                "[完整标定] 使用预加载 A/B 点 + static C map 初始化：A+=%d, B+=%d, C_pixels=%d。",
                len(a_prompt.positive_points),
                len(b_prompt.positive_points),
                int(reusable_c["mask"].sum()),
            )
            a_mask, b_mask, c_mask = self.abc_segmenter.initialize_ab_with_static_c_map(
                image_rgb=image_rgb,
                static_c_mask=reusable_c["mask"],
                a_positive_points=a_prompt.positive_points,
                a_negative_points=a_prompt.negative_points,
                b_positive_points=b_prompt.positive_points,
                b_negative_points=b_prompt.negative_points,
            )
            scene = build_scene_from_abc_sam2(
                image_rgb=image_rgb,
                a_mask=a_mask,
                b_mask=b_mask,
                c_mask=c_mask,
                cfg=self.cfg,
            )
            scene = self.apply_static_c_map_to_scene(scene)
            if self.cfg.save_annotated_image:
                init_path = self.image_dir / "sam2_init_preloaded_AB_reuse_C.png"
                empty_c_prompt = PointPromptSet(positive_points=[], negative_points=[])
                self.save_abc_init_image(
                    image_rgb=image_rgb,
                    a_mask=a_mask,
                    b_mask=b_mask,
                    c_mask=c_mask,
                    a_prompt=a_prompt,
                    b_prompt=b_prompt,
                    c_prompt=empty_c_prompt,
                    save_path=init_path,
                )
                status = self.get_boundary_rule_status(scene)
                self.save_annotated_image(
                    image_rgb=image_rgb,
                    scene=scene,
                    status=status,
                    action_name="INIT_PRELOADED_AB_REUSE_C_NO_POPUP",
                )
            logger.info("[完整标定] RuleAB 已完成非交互 A/B + static C 初始化。")
            return

        # 情况 2：没有 static C map。必须有 C 正点，用 A/B/C 点初始化，不弹 A/B/C 点选窗口。
        if not c_prompt.positive_points:
            raise RuntimeError(
                "完整标定注入失败：没有可复用 static C map，且缺少 C 正点，不能非交互初始化 RuleAB。"
            )

        logger.info(
            "[完整标定] 使用预加载 A/B/C 点初始化：A+=%d, B+=%d, C+=%d。",
            len(a_prompt.positive_points),
            len(b_prompt.positive_points),
            len(c_prompt.positive_points),
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
        scene = build_scene_from_abc_sam2(
            image_rgb=image_rgb,
            a_mask=a_mask,
            b_mask=b_mask,
            c_mask=c_mask,
            cfg=self.cfg,
        )
        self.save_static_c_reference(c_mask)
        self.save_confirmed_static_c_map(scene=scene, c_mask=c_mask, image_rgb=image_rgb)
        scene = self.apply_static_c_map_to_scene(scene)
        if self.cfg.save_annotated_image:
            init_path = self.image_dir / "sam2_init_preloaded_ABC.png"
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
            status = self.get_boundary_rule_status(scene)
            self.save_annotated_image(
                image_rgb=image_rgb,
                scene=scene,
                status=status,
                action_name="INIT_PRELOADED_ABC_NO_POPUP",
            )
        logger.info("[完整标定] RuleAB 已完成非交互 A/B/C 初始化。")

    def initialize_abc_with_first_frame(self) -> None:
        logger.info(
            "开始初始化：如果已有确认 C map，则只点击 A/B；否则点击 A/B/C 并确认保存 C map。"
        )

        # 关键修复：完整测量已经加载标定包时，主程序会向 follower/cfg 注入
        # preloaded_a/b/c_positive_points 和 skip_interactive_selection。
        # 旧版这里无条件继续调用 select_a_b_points_interactively()/select_a_b_c_points_interactively()，
        # 所以仍然弹出 A/B 或 A/B/C 标点窗口。
        # 现在先检查预加载点；只要 A/B 点存在，且有 static C 或 C 点，就直接非交互初始化。
        try:
            a_prompt0, b_prompt0, c_prompt0 = self._get_preloaded_prompt_sets()
            reusable_probe = self.get_reusable_static_c_map()
            has_preloaded_enough = bool(a_prompt0.positive_points and b_prompt0.positive_points and (reusable_probe is not None or c_prompt0.positive_points))
            if has_preloaded_enough or self._should_skip_interactive_from_preload():
                self.initialize_abc_with_calibration()
                return
        except Exception as e:
            # 如果明确要求跳过交互，却非交互初始化失败，则直接报错，不能退回弹窗。
            if self._should_skip_interactive_from_preload():
                raise
            logger.warning("预加载完整标定初始化失败，将退回交互式初始化: %s", e)

        image_rgb = self.capturer.capture()
        reusable_c = self.get_reusable_static_c_map()

        # ----------------------------------------------------
        # 情况 1：已有固定 C map。只重新点击 A、B。
        # ----------------------------------------------------
        if reusable_c is not None:
            self._record_static_c_map(reusable_c)
            logger.info(
                "使用已有固定 C map：source=%s, C_pixels=%d, C_contour_points=%d。后续只需要点击 A/B。",
                reusable_c.get("source"),
                int(reusable_c["mask"].sum()),
                len(reusable_c.get("contour", [])),
            )

            while True:
                a_prompt, b_prompt = select_a_b_points_interactively(
                    image_rgb=image_rgb,
                    window_name="SAM2 init: click A/B only, reuse confirmed C map",
                    scale=self.cfg.init_window_scale,
                )

                a_mask, b_mask, c_mask = self.abc_segmenter.initialize_ab_with_static_c_map(
                    image_rgb=image_rgb,
                    static_c_mask=reusable_c["mask"],
                    a_positive_points=a_prompt.positive_points,
                    a_negative_points=a_prompt.negative_points,
                    b_positive_points=b_prompt.positive_points,
                    b_negative_points=b_prompt.negative_points,
                )

                scene = build_scene_from_abc_sam2(
                    image_rgb=image_rgb,
                    a_mask=a_mask,
                    b_mask=b_mask,
                    c_mask=c_mask,
                    cfg=self.cfg,
                )
                scene = self.apply_static_c_map_to_scene(scene)

                if self.cfg.save_annotated_image:
                    init_path = self.image_dir / "sam2_init_candidate_AB_reuse_C.png"
                    empty_c_prompt = PointPromptSet(positive_points=[], negative_points=[])
                    self.save_abc_init_image(
                        image_rgb=image_rgb,
                        a_mask=a_mask,
                        b_mask=b_mask,
                        c_mask=c_mask,
                        a_prompt=a_prompt,
                        b_prompt=b_prompt,
                        c_prompt=empty_c_prompt,
                        save_path=init_path,
                    )
                    logger.info("A/B 初始化候选检查图已保存: %s", init_path)

                if bool(getattr(self.cfg, "confirm_first_frame_segmentation", True)):
                    accepted = confirm_first_frame_segmentation_interactively(
                        image_rgb=image_rgb,
                        scene=scene,
                        window_name="Confirm A/B segmentation with loaded static C map",
                        scale=float(getattr(self.cfg, "confirm_window_scale", self.cfg.init_window_scale)),
                    )
                    if not accepted:
                        logger.warning("用户拒绝当前 A/B 分割结果，保持 C map 不变并重新点击 A/B。")
                        continue

                if self.cfg.save_annotated_image:
                    status = self.get_boundary_rule_status(scene)
                    self.save_annotated_image(
                        image_rgb=image_rgb,
                        scene=scene,
                        status=status,
                        action_name="INIT_CONFIRMED_AB_REUSE_C_NO_MOTION",
                    )

                logger.info(
                    "A/B 分割已确认，继续使用固定 C map：C source=%s, C contour points=%d。现在允许进入运动循环。",
                    scene.c.source,
                    len(scene.c.contour),
                )
                break
            return

        # ----------------------------------------------------
        # 情况 2：没有可复用 C map。第一次必须点击 A/B/C，确认后保存 C map。
        # ----------------------------------------------------
        logger.info(
            "未找到可复用固定 C map：需要第一帧点击 A/B/C；确认后会保存 C 像素和轮廓供后续复用。"
        )

        while True:
            a_prompt, b_prompt, c_prompt = select_a_b_c_points_interactively(
                image_rgb=image_rgb,
                window_name="SAM2 init: A/B/C multi pos+neg fixed",
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

            scene = build_scene_from_abc_sam2(
                image_rgb=image_rgb,
                a_mask=a_mask,
                b_mask=b_mask,
                c_mask=c_mask,
                cfg=self.cfg,
            )

            if self.cfg.save_annotated_image:
                init_path = self.image_dir / "sam2_init_candidate_ABC_pos_neg.png"
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

            self.save_static_c_reference(c_mask)
            self.save_confirmed_static_c_map(scene=scene, c_mask=c_mask, image_rgb=image_rgb)
            scene = self.apply_static_c_map_to_scene(scene)

            if self.cfg.save_annotated_image:
                status = self.get_boundary_rule_status(scene)
                self.save_annotated_image(
                    image_rgb=image_rgb,
                    scene=scene,
                    status=status,
                    action_name="INIT_CONFIRMED_ABC_SAVE_C_MAP_NO_MOTION",
                )

            logger.info(
                "第一帧 SAM2 分割已确认：A/B/C 绑定完成；C source=%s, C contour points=%d；C map 已保存，现在允许进入运动循环。",
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
            2. A 与 B 的 mask 是否覆盖只记录，不再作为停止条件；
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
        # 1. A/B 覆盖不再作为停止条件。
        #    status 中仍保留 ab_overlap / ab_overlap_area，用于日志和可视化。
        # --------------------------------------------------------

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
            getattr(self.cfg, "a_c_correction_mode", "nearest_normal")
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
        #    如果 C contour_mode="quadrilateral"，这里就是沿 C 主体边缘四边形/多边形的
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

    def execute_action(self, action_id: int) -> Dict[str, Any]:
        direction = ACTION_TO_DIRECTION[action_id]

        if action_id == 0:
            info = {
                "action_id": int(action_id),
                "action_name": ACTION_NAMES.get(action_id, "STAY"),
                "direction": direction,
                "channel_name": "none",
                "channel": 0,
                "move_step": 0,
                "pause_s": 0.0,
                "voltage": 0,
                "velocity": 0,
                "acceleration": 0,
                "dry_run": self.stage is None,
            }
            logger.info("[Stage34] action=STAY，不移动。")
            return info

        # action_id=3/4 分别对应 LEFT/RIGHT，也就是 stage_x_channel，默认 CH3。
        # action_id=1/2 分别对应 UP/DOWN，也就是 stage_y_channel，默认 CH4。
        # 需求：
        #   CH3：不论 LEFT / RIGHT，每次只执行一次动作，动作结束后强制停顿 2 s。
        #   CH4：不论 UP / DOWN，每次只执行一次动作，动作结束后强制停顿 2 s。
        is_ch3_horizontal_action = action_id in (3, 4)
        is_ch4_vertical_action = action_id in (1, 2)

        ch3_pause_s = float(getattr(self.cfg, "stage_ch3_pause_after_move_s", 0.0) or 0.0)
        ch3_force_wait = bool(getattr(self.cfg, "stage_ch3_force_wait_before_pause", True))
        ch3_stop_after = bool(getattr(self.cfg, "stage_ch3_stop_after_move", True))

        ch4_pause_s = float(getattr(self.cfg, "stage_ch4_pause_after_move_s", 0.0) or 0.0)
        ch4_force_wait = bool(getattr(self.cfg, "stage_ch4_force_wait_before_pause", True))
        ch4_stop_after = bool(getattr(self.cfg, "stage_ch4_stop_after_move", True))

        need_pause = False
        pause_s = 0.0
        force_wait = bool(self.cfg.wait_stage_move)
        stop_after = False
        pause_tag = ""

        # --------------------------------------------------------
        # 新运动节拍规则：
        #   LEFT / RIGHT -> CH3：一次性运动 stage_step_x，然后停顿 CH3_pause。
        #   UP   / DOWN  -> CH4：一次性运动 stage_step_y，然后停顿 CH4_pause。
        #
        # 注意：这里不再优先使用 action_step。
        # action_step 只作为 stage_step_x / stage_step_y 非法时的兜底兼容值。
        # 因此 GUI 中：
        #   CH3速度=20、CH3加速度=20、CH3电压=100、X步数=100、CH3停顿=4
        # 表示 CH3 每次判断为 LEFT/RIGHT 时，一次运动 100 distance，然后停 4 s。
        # --------------------------------------------------------
        fallback_step = int(getattr(self.cfg, "action_step", 0) or 0)
        if fallback_step <= 0:
            fallback_step = 1

        if is_ch3_horizontal_action:
            move_step = int(getattr(self.cfg, "stage_step_x", fallback_step) or fallback_step)
            if move_step <= 0:
                move_step = fallback_step
            pause_s = ch3_pause_s
            need_pause = pause_s > 0
            force_wait = ch3_force_wait or force_wait
            stop_after = ch3_stop_after
            pause_tag = "CH3"
            channel_id = int(getattr(self.cfg, "stage_x_channel", 3))
            voltage = int(getattr(self.cfg, "stage_ch3_max_voltage", getattr(self.cfg, "stage_default_max_voltage", 0)))
            velocity = int(getattr(self.cfg, "stage_ch3_velocity", getattr(self.cfg, "stage_default_velocity", 0)))
            acceleration = int(getattr(self.cfg, "stage_ch3_acceleration", getattr(self.cfg, "stage_default_acceleration", 0)))
        elif is_ch4_vertical_action:
            move_step = int(getattr(self.cfg, "stage_step_y", fallback_step) or fallback_step)
            if move_step <= 0:
                move_step = fallback_step
            pause_s = ch4_pause_s
            need_pause = pause_s > 0
            force_wait = ch4_force_wait or force_wait
            stop_after = ch4_stop_after
            pause_tag = "CH4"
            channel_id = int(getattr(self.cfg, "stage_y_channel", 4))
            voltage = int(getattr(self.cfg, "stage_ch4_max_voltage", getattr(self.cfg, "stage_default_max_voltage", 0)))
            velocity = int(getattr(self.cfg, "stage_ch4_velocity", getattr(self.cfg, "stage_default_velocity", 0)))
            acceleration = int(getattr(self.cfg, "stage_ch4_acceleration", getattr(self.cfg, "stage_default_acceleration", 0)))
        else:
            move_step = fallback_step
            channel_id = 0
            voltage = 0
            velocity = 0
            acceleration = 0

        exec_info: Dict[str, Any] = {
            "action_id": int(action_id),
            "action_name": ACTION_NAMES.get(action_id, f"ACTION_{action_id}"),
            "direction": direction,
            "channel_name": pause_tag or "unknown",
            "channel": int(channel_id),
            "move_step": int(move_step),
            "pause_s": float(pause_s),
            "voltage": int(voltage),
            "velocity": int(velocity),
            "acceleration": int(acceleration),
            "dry_run": self.stage is None,
            "wait": bool(force_wait or self.cfg.wait_stage_move),
            "stop_after": bool(stop_after),
        }

        if self.stage is None:
            logger.info(
                "[DRY-RUN] action=%s, direction=%s, channel=%s, move_step=%d, pause=%.3f s",
                ACTION_NAMES[action_id],
                direction,
                pause_tag or "unknown",
                move_step,
                pause_s,
            )
            if need_pause and pause_s > 0:
                logger.info(
                    "[DRY-RUN][Stage34-%s] action=%s 后模拟停顿 %.3f s",
                    pause_tag,
                    ACTION_NAMES[action_id],
                    pause_s,
                )
                time.sleep(pause_s)
            return exec_info

        logger.info(
            "[Stage34] 执行动作: action=%s, direction=%s, channel=%s, move_step=%d, pause=%.3f s",
            ACTION_NAMES[action_id],
            direction,
            pause_tag or "unknown",
            move_step,
            pause_s,
        )

        # 对需要“动一次停一下”的通道，强制 wait=True，
        # 避免 wait=False 时刚发命令就 stop_all，导致实际位移不完整。
        wait_flag = bool(self.cfg.wait_stage_move)
        if need_pause and force_wait:
            wait_flag = True

        self.stage.execute_rule_action(
            action_id=action_id,
            direction=direction,
            step=move_step,
            wait=wait_flag,
            timeout=self.cfg.stage_move_timeout_s,
        )

        if need_pause and pause_s > 0:
            if stop_after:
                try:
                    self.stage.stop_all()
                    logger.info("[Stage34-%s] action=%s 后已 stop_all。", pause_tag, ACTION_NAMES[action_id])
                except Exception as e:
                    logger.warning("[Stage34-%s] action=%s 后 stop_all 失败: %s", pause_tag, ACTION_NAMES[action_id], e)

            logger.info(
                "[Stage34-%s] action=%s 已完成；强制停顿 %.3f s 后再进入下一帧。",
                pause_tag,
                ACTION_NAMES[action_id],
                pause_s,
            )
            time.sleep(pause_s)

        return exec_info

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
        scene = self.apply_static_c_map_to_scene(scene)

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
            "c_contour_points": len(scene.c.contour),
            "static_c_map_name": str(getattr(self.cfg, "static_c_map_name", "default_static_c_map")),

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
            "action_step": int(getattr(self.cfg, "action_step", 0)),

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
        stage_default_velocity=1,
        stage_default_acceleration=1,
        stage_default_max_voltage=50,
        stage_ch3_velocity=1,
        stage_ch3_acceleration=1,
        stage_ch3_max_voltage=50,
        stage_ch4_velocity=1,
        stage_ch4_acceleration=1,
        stage_ch4_max_voltage=50,
        stage_step_x=20,
        stage_step_y=20,
        stage_x_sign=1,
        stage_y_sign=1,

        # 每次规则动作的位移步数
        action_step=20,

        # 固定 C map：第一次确认后保存；后续运行只需重新点击 A/B。
        static_c_map_name="default_static_c_map",
        reuse_static_c_map_in_memory=True,
        load_static_c_map_if_exists=True,
        save_static_c_map_library=True,
        force_reselect_c_each_run=False,

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
