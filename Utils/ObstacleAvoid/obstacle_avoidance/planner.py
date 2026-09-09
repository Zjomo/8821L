"""规划层（PLAN 第 3 节）：occupancy grid + A*，带膨胀碰撞与 clearance 审计。

- 碰撞模型：inflation = ball_radius + spot_radius + safety_margin。
- 起点终点越界/低 clearance -> 拒绝（OUT_OF_BOUNDS / LOW_CLEARANCE）。
- 无路可走 -> NO_SAFE_PATH，绝不直接下发电机命令。
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .models import (ConfigError, FailureReason, Obstacle, Point,
                     SubstrateRegion, WorkspaceSnapshot, point_in_polygon)


@dataclass
class CollisionModel:
    ball_radius_px: float = 12.0
    spot_radius_px: float = 6.0
    safety_margin_px: float = 4.0

    @property
    def inflation_px(self) -> float:
        return self.ball_radius_px + self.spot_radius_px + self.safety_margin_px


@dataclass
class PlanConfig:
    grid_res_px: float = 2.0
    model: CollisionModel = field(default_factory=CollisionModel)
    # 衬底边界最小间隙（球心距边界）；None=用模型全膨胀（默认安全行为）。
    # 障碍碰撞始终用全膨胀，不受此参数影响。
    # 无论用户如何配置，间隙下限为球半径（碰撞体积下限：球心更近
    # 即球体越过衬底边界），见 edge_clearance。
    edge_clearance_px: Optional[float] = None

    @property
    def edge_clearance(self) -> float:
        floor = self.model.ball_radius_px
        if self.edge_clearance_px is not None:
            return max(floor, float(self.edge_clearance_px))
        return self.model.inflation_px


@dataclass
class PlanResult:
    success: bool
    waypoints_px: List[Point] = field(default_factory=list)
    length_px: float = 0.0
    min_clearance_px: float = float("inf")
    plan_version: int = 0
    failure_reason: Optional[FailureReason] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {"success": self.success,
                "waypoints_px": [[x, y] for x, y in self.waypoints_px],
                "length_px": self.length_px,
                "min_clearance_px": (None if self.min_clearance_px == float("inf")
                                     else self.min_clearance_px),
                "plan_version": self.plan_version,
                "failure_reason": (self.failure_reason.value
                                   if self.failure_reason else None),
                "detail": self.detail}


class GridPlanner:
    """栅格 A* 规划器。每次 ``plan`` 版本号 +1，输出可审计路径。"""

    def __init__(self, config: Optional[PlanConfig] = None) -> None:
        self.config = config or PlanConfig()
        self.version = 0

    # -------------------------------------------------- occupancy
    def build_occupancy(self, snap: WorkspaceSnapshot,
                        extra_obstacles: Sequence[Obstacle] = ()
                        ) -> Tuple[np.ndarray, float]:
        """返回 (blocked[h,w] bool, cell_size)。blocked=True 不可通行。"""
        cs = self.config.grid_res_px
        w, h = snap.frame_size
        nx, ny = int(math.ceil(w / cs)), int(math.ceil(h / cs))
        blocked = np.zeros((ny, nx), dtype=bool)
        infl = self.config.model.inflation_px
        sub = snap.substrate
        xs = (np.arange(nx) + 0.5) * cs
        ys = (np.arange(ny) + 0.5) * cs
        X, Y = np.meshgrid(xs, ys)
        # 衬底内部（含边界膨胀：与 check_point 的 is_feasible(infl) 语义一致）
        poly = np.array(sub.polygon, dtype=np.float64)
        inside = _points_in_polygon(X, Y, poly)
        # 距衬底边界（边界间隙独立于障碍膨胀）
        dist_b = _dist_to_polygon_boundary(X, Y, poly)
        blocked[~inside] = True
        blocked[dist_b < max(sub.safety_margin_px,
                             self.config.edge_clearance)] = True
        # 障碍
        for ob in list(snap.obstacles) + list(extra_obstacles):
            if ob.kind == "circle":
                d = np.hypot(X - ob.center[0], Y - ob.center[1])
                blocked[d <= ob.radius + infl] = True
            else:
                opoly = np.array(ob.polygon, dtype=np.float64)
                pin = _points_in_polygon(X, Y, opoly)
                dout = _dist_to_polygon_boundary(X, Y, opoly)
                blocked[pin | (dout <= infl)] = True
        return blocked, cs

    # -------------------------------------------------- checks
    def check_point(self, snap: WorkspaceSnapshot, p: Point,
                    extra_obstacles: Sequence[Obstacle] = ()
                    ) -> Optional[FailureReason]:
        """点是否可作为路径端点；非法返回原因。"""
        infl = self.config.model.inflation_px
        edge = self.config.edge_clearance
        if not snap.substrate.is_feasible(p, clearance_px=edge):
            if not snap.substrate.contains(p):
                return FailureReason.OUT_OF_BOUNDS
            return FailureReason.LOW_CLEARANCE
        for ob in list(snap.obstacles) + list(extra_obstacles):
            if ob.blocks(p, infl):
                return FailureReason.LOW_CLEARANCE
        return None

    def clearance_at(self, snap: WorkspaceSnapshot, p: Point,
                     extra_obstacles: Sequence[Obstacle] = ()) -> float:
        c = snap.substrate.dist_to_boundary(p) - snap.substrate.safety_margin_px
        for ob in list(snap.obstacles) + list(extra_obstacles):
            c = min(c, ob.clearance(p))
        return c

    # -------------------------------------------------- plan
    def plan(self, snap: WorkspaceSnapshot, start: Point, goal: Point,
             extra_obstacles: Sequence[Obstacle] = ()) -> PlanResult:
        self.version += 1
        for name, p in (("start", start), ("goal", goal)):
            reason = self.check_point(snap, p, extra_obstacles)
            if reason is not None:
                return PlanResult(success=False, plan_version=self.version,
                                  failure_reason=reason,
                                  detail=f"{name} point rejected: {reason.value}")
        blocked, cs = self.build_occupancy(snap, extra_obstacles)
        ny, nx = blocked.shape
        s = self._nearest_free(blocked, *self._to_cell(start, cs))
        g = self._nearest_free(blocked, *self._to_cell(goal, cs))
        if s is None or g is None:
            return PlanResult(success=False, plan_version=self.version,
                              failure_reason=FailureReason.NO_SAFE_PATH,
                              detail="no free cell near start/goal")
        cells = self._astar(blocked, s, g)
        if cells is None:
            return PlanResult(success=False, plan_version=self.version,
                              failure_reason=FailureReason.NO_SAFE_PATH,
                              detail="A* found no path")
        pts = [(c[0] * cs + cs / 2, c[1] * cs + cs / 2) for c in cells]
        pts[0] = start
        pts[-1] = goal
        pts = self._simplify(pts, snap, extra_obstacles)
        length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        min_cl = self._min_clearance(pts, snap, extra_obstacles)
        return PlanResult(success=True, waypoints_px=pts, length_px=length,
                          min_clearance_px=min_cl, plan_version=self.version)

    # -------------------------------------------------- internals
    @staticmethod
    def _to_cell(p: Point, cs: float) -> Tuple[int, int]:
        return int(p[0] / cs), int(p[1] / cs)

    @staticmethod
    def _nearest_free(blocked: np.ndarray, cx: int, cy: int,
                      max_radius: int = 25) -> Optional[Tuple[int, int]]:
        ny, nx = blocked.shape
        if 0 <= cx < nx and 0 <= cy < ny and not blocked[cy, cx]:
            return (cx, cy)
        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    x, y = cx + dx, cy + dy
                    if 0 <= x < nx and 0 <= y < ny and not blocked[y, x]:
                        return (x, y)
        return None

    @staticmethod
    def _astar(blocked: np.ndarray, s: Tuple[int, int], g: Tuple[int, int]
               ) -> Optional[List[Tuple[int, int]]]:
        ny, nx = blocked.shape
        if blocked[g[1], g[0]]:
            return None
        openq = [(0.0, s)]
        came = {s: None}
        cost = {s: 0.0}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = came[cur]
                return path[::-1]
            x, y = cur
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxp, nyp = x + dx, y + dy
                if not (0 <= nxp < nx and 0 <= nyp < ny) or blocked[nyp, nxp]:
                    continue
                if dx != 0 and dy != 0:  # 防对角穿缝
                    if blocked[y, nxp] or blocked[nyp, x]:
                        continue
                step = math.hypot(dx, dy)
                nc = cost[cur] + step
                nxt = (nxp, nyp)
                if nc < cost.get(nxt, float("inf")):
                    cost[nxt] = nc
                    came[nxt] = cur
                    h = max(abs(nxp - g[0]), abs(nyp - g[1])) + \
                        0.41421356 * min(abs(nxp - g[0]), abs(nyp - g[1]))
                    heapq.heappush(openq, (nc + h, nxt))
        return None

    def _line_free(self, a: Point, b: Point, snap: WorkspaceSnapshot,
                   extra_obstacles: Sequence[Obstacle]) -> bool:
        cs = self.config.grid_res_px
        n = max(2, int(math.dist(a, b) / cs) + 1)
        infl = self.config.model.inflation_px
        edge = self.config.edge_clearance
        for i in range(n + 1):
            t = i / n
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if not snap.substrate.is_feasible(p, clearance_px=edge):
                return False
            for ob in list(snap.obstacles) + list(extra_obstacles):
                if ob.blocks(p, infl):
                    return False
        return True

    def _simplify(self, pts: List[Point], snap: WorkspaceSnapshot,
                  extra_obstacles: Sequence[Obstacle]) -> List[Point]:
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._line_free(pts[i], pts[j], snap, extra_obstacles):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def _min_clearance(self, pts: List[Point], snap: WorkspaceSnapshot,
                       extra_obstacles: Sequence[Obstacle]) -> float:
        cs = self.config.grid_res_px
        mc = float("inf")
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            n = max(2, int(math.dist(a, b) / cs) + 1)
            for k in range(n + 1):
                t = k / n
                p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                mc = min(mc, self.clearance_at(snap, p, extra_obstacles))
        return mc


# ---------------------------------------------------------------- numpy helpers
def _points_in_polygon(X: np.ndarray, Y: np.ndarray,
                       poly: np.ndarray) -> np.ndarray:
    """向量化射线法。poly: (n,2)。"""
    inside = np.zeros(X.shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = (yi > Y) != (yj > Y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (xj - xi) * (Y - yi) / (yj - yi + 1e-12) + xi
        inside ^= cond & (X < x_int)
        j = i
    return inside


def _dist_to_polygon_boundary(X: np.ndarray, Y: np.ndarray,
                              poly: np.ndarray) -> np.ndarray:
    d = np.full(X.shape, float("inf"))
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 <= 1e-12:
            dd = np.hypot(X - ax, Y - ay)
        else:
            t = np.clip(((X - ax) * dx + (Y - ay) * dy) / seg2, 0.0, 1.0)
            dd = np.hypot(X - (ax + t * dx), Y - (ay + t * dy))
        d = np.minimum(d, dd)
    return d
