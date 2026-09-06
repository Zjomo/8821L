"""仿真世界（PLAN 第 1/4 节回滚点）：合成帧渲染 + dryrun stage 适配。

- 渲染约定：背景 0、衬底 ~200、障碍 ~40、粒子 ~255（软边）。
- ``DryRunStage`` 实现 XYStageProtocol 语义：move_by 只记账并驱动仿真球移动，
  不触碰任何真实设备；所有命令可追溯。
- 支持动态障碍（移动/新增）与遮挡开关，用于 OA-05 / OA-06。
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import (CoordinateTransform, Obstacle, Particle, Point,
                     StageCommand, SubstrateRegion, WorkspaceSnapshot)


class StageError(RuntimeError):
    """通信/驱动故障（HW-02）。"""


class XYStageProtocol:
    """位移台最小接口；真实驱动实现同签名即可接入。"""

    def move_by(self, dx_mm: float, dy_mm: float) -> bool:  # pragma: no cover
        raise NotImplementedError

    @property
    def last_command(self) -> Optional[StageCommand]:  # pragma: no cover
        raise NotImplementedError


class DryRunStage(XYStageProtocol):
    """dryrun 位移台：记录命令并回调仿真世界移动指定球。"""

    def __init__(self, on_move: Callable[[float, float], None],
                 latency_s: float = 0.0) -> None:
        self._on_move = on_move
        self._latency = latency_s
        self._last: Optional[StageCommand] = None
        self.commands: List[StageCommand] = []
        self._seq = 0
        self.timeout_s: Optional[float] = None  # 设置后模拟通信超时
        self._lock = threading.Lock()

    def move_by(self, dx_mm: float, dy_mm: float,
                task_id: str = "", track_id: int = -1, waypoint_index: int = -1,
                frame_id: int = -1, plan_version: int = -1) -> bool:
        if self.timeout_s is not None:
            raise StageError(f"stage communication timeout "
                             f"(simulated, timeout={self.timeout_s}s)")
        if self._latency > 0:
            time.sleep(self._latency)
        with self._lock:
            self._seq += 1
            cmd = StageCommand(seq=self._seq, timestamp=time.time(),
                               dx_mm=dx_mm, dy_mm=dy_mm, task_id=task_id,
                               track_id=track_id, waypoint_index=waypoint_index,
                               frame_id=frame_id,
                               source_plan_version=plan_version)
            self.commands.append(cmd)
            self._last = cmd
        self._on_move(dx_mm, dy_mm)
        return True

    @property
    def last_command(self) -> Optional[StageCommand]:
        return self._last


class SimWorld:
    """合成场景：衬底 + 障碍（可动态）+ 球（可遮挡），渲染 BGR 帧。"""

    BG, SUB, OBS, PART = 0, 200, 40, 255

    def __init__(self,
                 substrate_polygon: Sequence[Point],
                 obstacles: Sequence[Obstacle],
                 particles: Sequence[Tuple[Point, float]],  # (center, radius)
                 frame_size: Tuple[int, int] = (640, 480),
                 px_per_mm: float = 100.0) -> None:
        self.frame_size = frame_size
        self.transform = CoordinateTransform(px_per_mm=px_per_mm)
        self.substrate = SubstrateRegion(
            polygon=[(float(x), float(y)) for x, y in substrate_polygon])
        self.obstacles: List[Obstacle] = list(obstacles)
        self.particles: Dict[int, Particle] = {}
        for i, (pos, r) in enumerate(particles, start=1):
            self.particles[i] = Particle(track_id=i, position_px=pos, radius_px=r)
        self.frame_counter = 0
        # 场景脚本（每帧回调，用于动态障碍/遮挡）
        self._per_frame_hooks: List[Callable[[int], None]] = []
        self.occluded: Dict[int, bool] = {}

    # ---------------- scripting
    def on_frame(self, hook: Callable[[int], None]) -> None:
        self._per_frame_hooks.append(hook)

    def add_obstacle(self, ob: Obstacle) -> None:
        self.obstacles.append(ob)

    def move_obstacle(self, obstacle_id: str, new_center: Point) -> None:
        for ob in self.obstacles:
            if ob.obstacle_id == obstacle_id:
                ob.center = new_center
                return
        raise KeyError(obstacle_id)

    def set_occluded(self, track_id: int, value: bool) -> None:
        self.occluded[track_id] = value

    # ---------------- snapshot / render
    def snapshot(self) -> WorkspaceSnapshot:
        self.frame_counter += 1
        for hook in self._per_frame_hooks:
            hook(self.frame_counter)
        return WorkspaceSnapshot(
            frame_id=self.frame_counter, timestamp=time.time(),
            substrate=self.substrate, obstacles=list(self.obstacles),
            particles=[p for p in self.particles.values()],
            transform=self.transform, frame_size=self.frame_size)

    def render(self) -> np.ndarray:
        w, h = self.frame_size
        frame = np.full((h, w), self.BG, dtype=np.uint8)
        poly = np.array(self.substrate.polygon, dtype=np.int32)
        cv2_fill(frame, poly, self.SUB)
        for ob in self.obstacles:
            if ob.kind == "circle":
                cv2_fill_circle(frame, ob.center, ob.radius, self.OBS)
            else:
                cv2_fill(frame, np.array(ob.polygon, dtype=np.int32), self.OBS)
        for pid, p in self.particles.items():
            if self.occluded.get(pid):
                continue
            cv2_fill_circle_soft(frame, p.position_px, p.radius_px, self.PART)
        bgr = cv2_gray_to_bgr(frame)
        return bgr

    def move_particle(self, track_id: int, dx_px: float, dy_px: float) -> None:
        p = self.particles[track_id]
        p.position_px = (p.position_px[0] + dx_px, p.position_px[1] + dy_px)
        p.history_px.append(p.position_px)

    def particle_position(self, track_id: int) -> Point:
        return self.particles[track_id].position_px

    def make_stage(self, track_id: int) -> DryRunStage:
        """返回绑定指定球的 dryrun stage；mm -> px 转换在此完成。"""

        def _move(dx_mm: float, dy_mm: float) -> None:
            self.move_particle(track_id,
                               dx_mm * self.transform.px_per_mm,
                               dy_mm * self.transform.px_per_mm)

        return DryRunStage(_move)


# ---------------------------------------------------------------- cv helpers (避免顶部强依赖命名冲突)
def cv2_fill(frame: np.ndarray, poly: np.ndarray, value: int) -> None:
    import cv2
    cv2.fillPoly(frame, [poly], int(value))


def cv2_fill_circle(frame: np.ndarray, center: Point, radius: float,
                    value: int) -> None:
    import cv2
    cv2.circle(frame, (int(round(center[0])), int(round(center[1]))),
               int(round(radius)), int(value), -1)


def cv2_fill_circle_soft(frame: np.ndarray, center: Point, radius: float,
                         value: int) -> None:
    """球：中心亮、边缘较快衰减（0.15r 过渡带），模拟光斑推球亮度。"""
    import cv2
    x0, x1 = int(center[0] - radius - 2), int(center[0] + radius + 3)
    y0, y1 = int(center[1] - radius - 2), int(center[1] + radius + 3)
    h, w = frame.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - center[0], yy - center[1])
    patch = np.clip((radius - d) / max(radius * 0.15, 1e-6), 0, 1)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.maximum(
        roi, patch * value).astype(np.uint8)


def cv2_gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
