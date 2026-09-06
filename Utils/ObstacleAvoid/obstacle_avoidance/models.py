"""领域模型与坐标契约（PLAN 第 1 节）。

所有几何以像素为主单位，物理由 ``CoordinateTransform`` 换算；
所有对象可 JSON 序列化并带时间戳/置信度/来源帧。
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


# ---------------------------------------------------------------- errors
class ConfigError(ValueError):
    """任务配置非法（越界、clearance 不足等），规划层拒绝。"""


class FailureReason(str, Enum):
    NONE = "none"
    OUT_OF_BOUNDS = "out_of_bounds"          # 起点/终点/区域在衬底可行域外
    LOW_CLEARANCE = "low_clearance"          # 与障碍/边界 clearance 不足
    NO_SAFE_PATH = "no_safe_path"            # 无可行路径
    DETECTION_UNCERTAIN = "detection_uncertain"
    COMM_TIMEOUT = "comm_timeout"
    ESTOP = "estop"
    PAUSED = "paused"
    MAX_STEPS = "max_steps_exceeded"
    SPOT_SLIP = "spot_slip"                   # 光斑-球失位（滑移超限）


class RunState(str, Enum):
    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    TRACKING = "TRACKING"
    PLANNING = "PLANNING"
    MOVING = "MOVING"
    VERIFYING = "VERIFYING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"
    FAULT = "FAULT"
    DETECTION_UNCERTAIN = "DETECTION_UNCERTAIN"


# ---------------------------------------------------------------- geometry
def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > p[1]) != (yj > p[1]):
            x_int = (xj - xi) * (p[1] - yi) / (yj - yi + 1e-12) + xi
            if p[0] < x_int:
                inside = not inside
        j = i
    return inside


def dist_point_segment(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def dist_point_polygon_boundary(p: Point, poly: Sequence[Point]) -> float:
    return min(dist_point_segment(p, poly[i], poly[(i + 1) % len(poly)])
               for i in range(len(poly)))


# ---------------------------------------------------------------- transform
@dataclass
class CoordinateTransform:
    """像素 <-> 物理(mm)。方向一致，仅比例换算；往返误差 <= 1 px。"""

    px_per_mm: float = 100.0

    def to_mm(self, p_px: Point) -> Point:
        return (p_px[0] / self.px_per_mm, p_px[1] / self.px_per_mm)

    def to_px(self, p_mm: Point) -> Point:
        return (p_mm[0] * self.px_per_mm, p_mm[1] * self.px_per_mm)

    def mm_to_px_len(self, v_mm: float) -> float:
        return v_mm * self.px_per_mm

    def px_to_mm_len(self, v_px: float) -> float:
        return v_px / self.px_per_mm

    # ---- serialization
    def to_dict(self) -> dict:
        return {"px_per_mm": self.px_per_mm}

    @classmethod
    def from_dict(cls, d: dict) -> "CoordinateTransform":
        return cls(px_per_mm=float(d["px_per_mm"]))


# ---------------------------------------------------------------- regions
@dataclass
class SubstrateRegion:
    """衬底多边形 + 安全边界（内缩 margin 后为可行域）。"""

    polygon: List[Point]
    safety_margin_px: float = 0.0

    def contains(self, p: Point) -> bool:
        return point_in_polygon(p, self.polygon)

    def dist_to_boundary(self, p: Point) -> float:
        return dist_point_polygon_boundary(p, self.polygon)

    def is_feasible(self, p: Point, clearance_px: float = 0.0) -> bool:
        """可行 = 在衬底内且距边界 >= max(safety_margin, clearance)。"""
        need = max(self.safety_margin_px, clearance_px)
        return self.contains(p) and self.dist_to_boundary(p) >= need - 1e-9

    def to_dict(self) -> dict:
        return {"polygon": [list(p) for p in self.polygon],
                "safety_margin_px": self.safety_margin_px}

    @classmethod
    def from_dict(cls, d: dict) -> "SubstrateRegion":
        return cls(polygon=[(float(x), float(y)) for x, y in d["polygon"]],
                   safety_margin_px=float(d.get("safety_margin_px", 0.0)))


@dataclass
class Obstacle:
    """障碍：圆形或多边形；``inflation_px`` 由碰撞模型统一计算。"""

    kind: str                      # "circle" | "polygon"
    center: Optional[Point] = None
    radius: float = 0.0
    polygon: Optional[List[Point]] = None
    obstacle_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def blocks(self, p: Point, clearance_px: float) -> bool:
        if self.kind == "circle":
            return math.hypot(p[0] - self.center[0], p[1] - self.center[1]) \
                <= self.radius + clearance_px
        if point_in_polygon(p, self.polygon):
            return True
        return dist_point_polygon_boundary(p, self.polygon) <= clearance_px

    def clearance(self, p: Point) -> float:
        """点 p 到该障碍表面的距离（内部为负）。"""
        if self.kind == "circle":
            return math.hypot(p[0] - self.center[0], p[1] - self.center[1]) - self.radius
        if point_in_polygon(p, self.polygon):
            return -dist_point_polygon_boundary(p, self.polygon)
        return dist_point_polygon_boundary(p, self.polygon)

    def signature(self) -> str:
        if self.kind == "circle":
            return f"{self.obstacle_id}:c:{self.center[0]:.2f}:{self.center[1]:.2f}:{self.radius:.2f}"
        pts = ";".join(f"{x:.2f},{y:.2f}" for x, y in self.polygon)
        return f"{self.obstacle_id}:p:{pts}"

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "obstacle_id": self.obstacle_id}
        if self.kind == "circle":
            d.update({"center": list(self.center), "radius": self.radius})
        else:
            d.update({"polygon": [list(p) for p in self.polygon]})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Obstacle":
        if d["kind"] == "circle":
            return cls(kind="circle",
                       center=(float(d["center"][0]), float(d["center"][1])),
                       radius=float(d["radius"]),
                       obstacle_id=d.get("obstacle_id", uuid.uuid4().hex[:8]))
        return cls(kind="polygon",
                   polygon=[(float(x), float(y)) for x, y in d["polygon"]],
                   obstacle_id=d.get("obstacle_id", uuid.uuid4().hex[:8]))


@dataclass
class GoalRegion:
    """终点区域（圆形）。complete 条件：球心距 center <= radius - ball_radius。"""

    center: Point
    radius_px: float

    def contains_center(self, p: Point, ball_radius_px: float = 0.0) -> bool:
        return math.hypot(p[0] - self.center[0], p[1] - self.center[1]) \
            <= self.radius_px - ball_radius_px + 1e-9

    def to_dict(self) -> dict:
        return {"center": list(self.center), "radius_px": self.radius_px}

    @classmethod
    def from_dict(cls, d: dict) -> "GoalRegion":
        return cls(center=(float(d["center"][0]), float(d["center"][1])),
                   radius_px=float(d["radius_px"]))


# ---------------------------------------------------------------- particles
@dataclass
class Particle:
    track_id: int
    position_px: Point
    radius_px: float
    confidence: float = 1.0
    frame_id: int = -1
    velocity_px_s: Point = (0.0, 0.0)
    history_px: List[Point] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"track_id": self.track_id, "position_px": list(self.position_px),
                "radius_px": self.radius_px, "confidence": self.confidence,
                "frame_id": self.frame_id}

    @classmethod
    def from_dict(cls, d: dict) -> "Particle":
        return cls(track_id=int(d["track_id"]),
                   position_px=(float(d["position_px"][0]), float(d["position_px"][1])),
                   radius_px=float(d["radius_px"]),
                   confidence=float(d.get("confidence", 1.0)),
                   frame_id=int(d.get("frame_id", -1)))


# ---------------------------------------------------------------- snapshot
@dataclass
class WorkspaceSnapshot:
    """一次感知的世界状态：衬底 + 障碍 + 粒子 + 时间戳/帧号/置信度。"""

    frame_id: int
    timestamp: float
    substrate: SubstrateRegion
    obstacles: List[Obstacle]
    particles: List[Particle]
    transform: CoordinateTransform
    frame_size: Tuple[int, int] = (640, 480)
    overall_confidence: float = 1.0
    uncertain: bool = False
    uncertain_reason: str = ""

    def particle(self, track_id: int) -> Optional[Particle]:
        for p in self.particles:
            if p.track_id == track_id:
                return p
        return None

    def obstacle_signature(self) -> str:
        return "|".join(sorted(o.signature() for o in self.obstacles))

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id, "timestamp": self.timestamp,
            "substrate": self.substrate.to_dict(),
            "obstacles": [o.to_dict() for o in self.obstacles],
            "particles": [p.to_dict() for p in self.particles],
            "transform": self.transform.to_dict(),
            "frame_size": list(self.frame_size),
            "overall_confidence": self.overall_confidence,
            "uncertain": self.uncertain, "uncertain_reason": self.uncertain_reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkspaceSnapshot":
        return cls(
            frame_id=int(d["frame_id"]), timestamp=float(d["timestamp"]),
            substrate=SubstrateRegion.from_dict(d["substrate"]),
            obstacles=[Obstacle.from_dict(o) for o in d["obstacles"]],
            particles=[Particle.from_dict(p) for p in d["particles"]],
            transform=CoordinateTransform.from_dict(d["transform"]),
            frame_size=tuple(d.get("frame_size", (640, 480))),
            overall_confidence=float(d.get("overall_confidence", 1.0)),
            uncertain=bool(d.get("uncertain", False)),
            uncertain_reason=str(d.get("uncertain_reason", "")),
        )


# ---------------------------------------------------------------- commands
@dataclass
class StageCommand:
    """一条电机位移命令，可追溯到 waypoint 与视觉帧（PLAN 第 4 节验收）。"""

    seq: int
    timestamp: float
    dx_mm: float
    dy_mm: float
    task_id: str
    track_id: int
    waypoint_index: int
    frame_id: int
    source_plan_version: int

    def to_dict(self) -> dict:
        return {"seq": self.seq, "timestamp": self.timestamp,
                "dx_mm": self.dx_mm, "dy_mm": self.dy_mm,
                "task_id": self.task_id, "track_id": self.track_id,
                "waypoint_index": self.waypoint_index, "frame_id": self.frame_id,
                "source_plan_version": self.source_plan_version}


def now() -> float:
    return time.time()


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
