"""ROI 与区域划分配置（固定镜头视野 + 桌面分区）。

用途：在固定镜头（视频源）上选定 ROI 作为工作视野，并在视野内划分
目标区（组装终点）/ 障碍区（静态禁区），持久化为 JSON，供闭环场景
（video03）与 UI 实时检测使用。

坐标约定：ROI 与区域均使用视频绝对像素坐标；区域必须完全位于 ROI 内。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

Rect = Tuple[int, int, int, int]  # (x, y, w, h)

ZONE_KINDS = ("goal", "obstacle", "free")


class LayoutValidationError(ValueError):
    """Invalid simulation/motor task layout."""


def rect_contains_rect(outer, inner, margin: float = 0.0) -> bool:
    """Return True when the complete inner rectangle is inside outer."""
    ox, oy, ow, oh = [float(v) for v in outer]
    ix, iy, iw, ih = [float(v) for v in inner]
    return (ix >= ox + margin and iy >= oy + margin and
            ix + iw <= ox + ow - margin and
            iy + ih <= oy + oh - margin)


def validate_sim_layout(layout: dict, mode: str = "oa") -> list[list[int]]:
    """Validate a multi-ground layout and return ball indexes per ground.

    Every drawable object must be fully contained by exactly one substrate.
    A ground containing balls must have exactly one task target for the
    selected mode (point for ``oa``, rectangle for ``ag``).
    """
    if mode not in ("oa", "ag"):
        raise LayoutValidationError(f"unsupported task mode: {mode}")
    grounds = layout.get("grounds") or []
    balls = layout.get("balls") or []
    obstacles = layout.get("obstacles") or []
    if not grounds:
        raise LayoutValidationError("at least one substrate ground is required")

    def owners(rect):
        return [i for i, g in enumerate(grounds)
                if rect_contains_rect(g, rect)]

    for index, rect in enumerate(grounds):
        if len(rect) != 4 or rect[2] <= 0 or rect[3] <= 0:
            raise LayoutValidationError(f"substrate {index} has invalid rectangle")

    groups = [[] for _ in grounds]
    for kind, items in (("ball", balls), ("obstacle", obstacles)):
        for index, rect in enumerate(items):
            if len(rect) != 4 or rect[2] <= 0 or rect[3] <= 0:
                raise LayoutValidationError(f"{kind} {index} has invalid rectangle")
            own = owners(rect)
            if len(own) != 1:
                raise LayoutValidationError(
                    f"{kind} {index} must be fully inside exactly one substrate")
            if kind == "ball":
                groups[own[0]].append(index)

    goals = layout.get("ground_goals") or []
    ranges = layout.get("ground_goal_ranges") or []
    if len(goals) > len(grounds) or len(ranges) > len(grounds):
        raise LayoutValidationError("target mapping count cannot exceed substrate count")
    for i, g in enumerate(grounds):
        goal = goals[i] if i < len(goals) else None
        region = ranges[i] if i < len(ranges) else None
        if groups[i]:
            if mode == "oa" and goal is None:
                raise LayoutValidationError(
                    f"substrate {i} must have exactly one target point")
            if mode == "ag" and region is None:
                raise LayoutValidationError(
                    f"substrate {i} must have exactly one target range")
            if mode == "oa" and region is not None:
                raise LayoutValidationError(
                    f"substrate {i} must have only one target point in avoidance mode")
            if mode == "ag" and goal is not None:
                raise LayoutValidationError(
                    f"substrate {i} must have only one target range in assembly mode")
        if goal is not None:
            if len(goal) != 2 or not (g[0] <= goal[0] <= g[0] + g[2] and
                                      g[1] <= goal[1] <= g[1] + g[3]):
                raise LayoutValidationError(f"target point for substrate {i} is outside ground")
        if region is not None and not rect_contains_rect(g, region):
            raise LayoutValidationError(f"target range for substrate {i} is outside ground")
    return groups


@dataclass
class Zone:
    """视野内一个命名区域：goal=目标/组装区，obstacle=障碍区，free=自由区。"""

    name: str
    kind: str                     # "goal" | "obstacle" | "free"
    rect: Rect                    # 视频绝对坐标 (x, y, w, h)

    def center(self) -> Tuple[float, float]:
        return (self.rect[0] + self.rect[2] / 2, self.rect[1] + self.rect[3] / 2)

    def contains(self, p: Tuple[float, float]) -> bool:
        x, y, w, h = self.rect
        return x <= p[0] <= x + w and y <= p[1] <= y + h


@dataclass
class RoiConfig:
    """固定镜头配置：ROI（工作视野）+ 视野内区域划分。"""

    roi: Rect
    zones: List[Zone] = field(default_factory=list)
    video: Optional[str] = None   # 来源视频（记录用）
    px_per_mm: float = 100.0
    # 边界安全间隙(px)：球心距衬底边界的最小允许值；障碍碰撞不受影响。
    # 默认 39 = 球半径25 + 光斑6 + 裕量8（原安全行为）；可降至 1。
    edge_clearance_px: float = 39.0

    # ---------------- validation
    def validate(self, video_size: Optional[Tuple[int, int]] = None) -> None:
        x, y, w, h = self.roi
        if w <= 0 or h <= 0:
            raise ValueError(f"ROI 尺寸非法: {self.roi}")
        if video_size:
            vw, vh = video_size
            if x < 0 or y < 0 or x + w > vw or y + h > vh:
                raise ValueError(
                    f"ROI 超出视频范围 {video_size}: {self.roi}")
        for z in self.zones:
            zx, zy, zw, zh = z.rect
            if z.kind not in ZONE_KINDS:
                raise ValueError(f"区域类型非法: {z.kind}")
            if zw <= 0 or zh <= 0:
                raise ValueError(f"区域尺寸非法: {z.name} {z.rect}")
            if zx < x or zy < y or zx + zw > x + w or zy + zh > y + h:
                raise ValueError(
                    f"区域 {z.name} 超出 ROI {self.roi}: {z.rect}")
        if sum(1 for z in self.zones if z.kind == "goal") > 1:
            raise ValueError("目标区最多一个")

    def goal_zone(self) -> Optional[Zone]:
        return next((z for z in self.zones if z.kind == "goal"), None)

    def obstacle_zones(self) -> List[Zone]:
        return [z for z in self.zones if z.kind == "obstacle"]

    # ---------------- persistence
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"roi": list(self.roi),
                       "video": self.video,
                       "px_per_mm": self.px_per_mm,
                       "edge_clearance_px": self.edge_clearance_px,
                       "zones": [asdict(z) for z in self.zones]},
                      f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "RoiConfig":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        cfg = cls(roi=tuple(d["roi"]),                  # type: ignore[arg-type]
                  video=d.get("video"),
                  px_per_mm=float(d.get("px_per_mm", 100.0)),
                  edge_clearance_px=float(d.get("edge_clearance_px", 39.0)),
                  zones=[Zone(name=z["name"], kind=z["kind"],
                              rect=tuple(z["rect"]))       # type: ignore[arg-type]
                         for z in d.get("zones", [])])
        cfg.validate()
        return cfg
