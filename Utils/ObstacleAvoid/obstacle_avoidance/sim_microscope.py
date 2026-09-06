"""显微镜仿真世界（"仿真模拟"模式，复用 microscope-master 模拟设备）。

帧源与位移台来自 API/microscope-master 的模拟设备：
- SampleAwareCamera（simulator_app/devices.py）：程序合成组织切片样本，
  视野中心跟随位移台 X/Y（0.5µm/px，传感器 800x600），Z 轴模拟离焦，
  曝光/增益/读出噪声拟真 —— 即"仿真镜头"。
- SQLiteStage：X/Y/Z 三轴模拟位移台（µm，SQLite 持久化，重启恢复）。

坐标契约（与 VideoWorld 一致，闭环控制器无感知）：
- stage 命令 (dx,dy) mm -> 球在画面中平移 +(dx,dy)*px_per_mm（台位取反）；
  px_per_mm = 1000/像素尺寸 = 2000 px/mm（0.5µm/px）。
- 球/静态障碍定义在样本像素坐标（世界系），随台位移动在画面中真实移动。
- 球为白色圆盘（255），ClassicDetector 阈值 235 直接可检；YOLO 后端亦可用。
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .controller import ControllerConfig, ObstacleAvoidController
from .models import (CoordinateTransform, GoalRegion, Obstacle, Point,
                     SubstrateRegion, WorkspaceSnapshot)
from .motion import (AxisMotionConfig, MotionConfig, MotionConfigError,
                      VirtualXYZStage)
from .planner import GridPlanner
from .simulator import DryRunStage, StageError
from .vision import ClassicDetector, VisionPipeline
from .video_sim import _initial_detect, _nudge_to_feasible

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MICRO_REPO = os.path.join(PROJECT_ROOT, "API", "microscope-master",
                          "microscope-master")
SIM_APP = os.path.join(PROJECT_ROOT, "API", "microscope-master",
                       "simulator_app")
SIM_DB = os.path.join(PROJECT_ROOT, "artifacts", "sim_microscope.db")

WINDOW = (800, 600)          # SampleAwareCamera 传感器 (w, h)
SAMPLE_CENTER = (1500.0, 1000.0)   # 合成样本 3000x2000 的中心（样本 px）

# 仿真显微镜图层类别（microscope-master DEFAULT_SAMPLE_SPEC 同名）：
# 每类可指定 count（数量）/ shape（形状）/ label（类别标签前缀）/
# size（基准尺寸 px）/ custom（自定义多边形顶点 "x,y x,y ..."，shape=custom 时生效）。
# shape 可选：ellipse / blob / rect / triangle / star / custom（任意多边形）。
SAMPLE_CATEGORIES = ("ground", "mask", "obstacle")

# 默认场景布局：球为样本 px（初始台位在样本中心 -> win = sample - (1100, 700)），
# 静态障碍为窗口 px（相机固定）。
#   ball0 (320,450) -> 静态盘 (480,435) 正挡直线 -> goal (620,450)，强制绕行；
#   ball1 (550,220) 随台位漂移作为动态障碍。
# 球半径 12px 与 CollisionModel.ball_radius_px 一致（视觉检测 = 规划膨胀模型）。
DEFAULT_BALLS = [((1420.0, 1150.0), 12.0), ((1650.0, 920.0), 12.0)]
DEFAULT_OBSTACLES = [(480.0, 435.0, 55.0)]
GOAL = (620.0, 450.0)        # 终点（窗口坐标）
TARGET_HINT = (320.0, 450.0)  # 目标球初始位置提示（窗口坐标）

# 镜头图层合法形状（microscope-master _unit_shape 支持集）
_VALID_SHAPES = {"ellipse", "blob", "rect", "triangle", "star", "custom"}


def _ensure_paths() -> None:
    # Import the simulator as a package so callers do not depend on the
    # historical ``sys.path`` layout.  MICRO_REPO remains available for the
    # python-microscope dependency resolver in simulator_app.devices.
    for p in (MICRO_REPO, os.path.dirname(SIM_APP)):
        if not os.path.isdir(p):
            raise RuntimeError(f"microscope-master 不存在: {p}")
        if p not in sys.path:
            sys.path.insert(0, p)


def _import_sim_devices():
    """Lazy import of the reusable microscope simulator package."""
    _ensure_paths()
    from simulator_app import (Database, SampleAwareCamera, SQLiteStage)
    return SQLiteStage, SampleAwareCamera, Database


class SimMicroscopeWorld:
    """仿真模拟世界：microscope-master 相机+位移台 + 叠加球/障碍。

    sample_spec 可配置仿真镜头样本图层（掩码 mask / 衬底 ground /
    障碍物 obstacle），每类支持 count（数量）/ shape（形状，含 custom
    任意多边形）/ label（类别标签）/ size（基准尺寸）/ custom（顶点串）。
    """

    def __init__(self,
                 balls: Sequence[Tuple[Point, float]] = DEFAULT_BALLS,
                 static_obstacles: Sequence[Tuple[float, float, float]] =
                 DEFAULT_OBSTACLES,
                 db_path: str = SIM_DB,
                 stage_limits_um: Tuple[float, float] = (-1000.0, 1000.0),
                 sample_spec: Optional[Mapping[str, Mapping]] = None) -> None:
        SQLiteStage, SampleAwareCamera, Database = _import_sim_devices()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db = Database(db_path)
        lim = stage_limits_um
        self.micro_stage = SQLiteStage(
            self._db, {"x": (lim[0], lim[1]), "y": (lim[0], lim[1]),
                       "z": (-50.0, 50.0)})
        self.cam = SampleAwareCamera(self.micro_stage, self._db)
        if sample_spec:
            self._apply_sample_spec(sample_spec)
        self.cam.enable()
        self.cam.set_fast_preview(True)   # 仿真闭环跳过曝光等待
        # 台位 (0,0)µm 对准样本左上角；移到样本中心，使默认球/障碍布局
        # 落在初始视野内（SQLite 持久化，重复设置幂等）
        cx_um = SAMPLE_CENTER[0] * self.cam._pixel_size
        cy_um = SAMPLE_CENTER[1] * self.cam._pixel_size
        self.micro_stage.move_to({"x": cx_um, "y": cy_um})

        self.window = WINDOW
        self.pixel_size_um = float(self.cam._pixel_size)   # 0.5 µm/px
        self.transform = CoordinateTransform(
            px_per_mm=1000.0 / self.pixel_size_um)         # 2000 px/mm
        self.substrate = SubstrateRegion(
            polygon=[(10.0, 10.0), (self.window[0] - 10.0, 10.0),
                     (self.window[0] - 10.0, self.window[1] - 10.0),
                     (10.0, self.window[1] - 10.0)],
            safety_margin_px=4.0)
        self._balls = [(float(p[0]), float(p[1]), float(r))
                       for p, r in balls]
        self._obstacles = [(float(x), float(y), float(r))
                           for x, y, r in static_obstacles]
        self._stage_limits_um = lim
        self.motion_stage = VirtualXYZStage(
            MotionConfig(axes={
                "x": AxisMotionConfig("x", lim[0], lim[1],
                                       steps_per_unit=1.0,
                                       max_speed=100.0, acceleration=200.0),
                "y": AxisMotionConfig("y", lim[0], lim[1],
                                       steps_per_unit=1.0,
                                       max_speed=100.0, acceleration=200.0),
                "z": AxisMotionConfig("z", -50.0, 50.0,
                                       steps_per_unit=1.0,
                                       max_speed=20.0, acceleration=40.0),
            }),
            initial_position=dict(self.micro_stage.position),
            on_move=lambda pos: self.micro_stage.move_to(pos),
        )
        self.pipeline: Optional[VisionPipeline] = None
        self.target_track_id: int = -1
        self.frame_counter = 0
        self._last_frame: Optional[np.ndarray] = None
        self._closed = False

    # ---------------- 视野几何
    def make_detector(self):
        """返回适配本世界帧的检测器（UI 实时检测/调试用）。"""
        return _SimLensDetector(self.substrate, min_particle_radius_px=8.0)

    def _apply_sample_spec(self, sample_spec: Mapping[str, Mapping]) -> None:
        """应用镜头图层配置（掩码/衬底/障碍物），按类别/字段与当前配置合并。

        每类字段：count / shape / label / size / custom（未指定的沿用当前值）。
        set_sample_spec 以随机 seed 重建样本并持久化（含 seed）——应用后
        布局确定，后续启动加载同一布局；重新应用才会重新随机布点。
        """
        for cat, cfg in sample_spec.items():
            if cat not in SAMPLE_CATEGORIES:
                raise ValueError(
                    f"sample_spec 未知图层类别 {cat!r}，"
                    f"可选: {', '.join(SAMPLE_CATEGORIES)}")
            shape = str((cfg or {}).get("shape", ""))
            if shape and shape not in _VALID_SHAPES:
                raise ValueError(
                    f"sample_spec[{cat}].shape 非法: {shape!r}，"
                    f"可选: {', '.join(sorted(_VALID_SHAPES))}")
        current = self.cam.get_sample_spec()
        merged = {cat: {**current.get(cat, {}), **(sample_spec.get(cat) or {})}
                  for cat in SAMPLE_CATEGORIES}
        self.cam.set_sample_spec(merged)
        # generate_sample 在全样本(3000x2000)随机布点，单视野(800x600)仅覆盖
        # 约 8% —— 重试布点直到初始视野内至少可见 1 个图层对象，
        # 保证配置对"仿真镜头"可见（重试上限兜底极小概率场景）。
        vx0 = SAMPLE_CENTER[0] - WINDOW[0] / 2.0
        vy0 = SAMPLE_CENTER[1] - WINDOW[1] / 2.0
        for _ in range(20):
            objs = self.cam.get_sample_objects()
            if any(vx0 <= o["cx"] <= vx0 + WINDOW[0]
                   and vy0 <= o["cy"] <= vy0 + WINDOW[1] for o in objs):
                break
            self.cam.set_sample_spec(merged)

    def _view_origin(self) -> Point:
        """视野左上角在样本像素坐标中的位置（窗口中心 = 台位）。"""
        px = self.micro_stage.position
        return (px["x"] / self.pixel_size_um - self.window[0] / 2.0,
                px["y"] / self.pixel_size_um - self.window[1] / 2.0)

    def _to_window(self, p: Point) -> Point:
        ox, oy = self._view_origin()
        return (p[0] - ox, p[1] - oy)

    # ---------------- sensor
    def render(self) -> np.ndarray:
        """触发相机取一帧（BGR），叠加静态障碍与球（样本坐标->窗口坐标）。"""
        self.cam.trigger()
        gray = self.cam._fetch_data()   # noqa: SLF001 - 与 simulator_app 同用法
        if gray is None:
            raise RuntimeError("sim camera returned no frame")
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, r in self._obstacles:
            wx, wy = self._to_window((x, y))
            if -r < wx < self.window[0] + r and -r < wy < self.window[1] + r:
                cv2.circle(frame, (int(wx), int(wy)), int(r), (30, 30, 30), -1)
        for i, (x, y, r) in enumerate(self._balls):
            wx, wy = self._to_window((x, y))
            if -r < wx < self.window[0] + r and -r < wy < self.window[1] + r:
                cv2.circle(frame, (int(wx), int(wy)), int(r), (255, 255, 255), -1)
                hl = max(2, int(r) // 5)
                cv2.circle(frame, (int(wx) - int(r) // 4,
                                   int(wy) - int(r) // 4),
                           hl, (255, 255, 255), -1)
                cv2.putText(frame, str(i), (int(wx) - 8, int(wy) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        self._last_frame = frame
        return frame

    # ---------------- stage
    def make_stage(self) -> DryRunStage:
        def _move(dx_mm: float, dy_mm: float) -> None:
            # 契约：命令 (dx,dy) -> 球画面位移 +(dx,dy)*ppm。
            # 相机视野中心 = 台位，故台位需向反方向移动。
            try:
                self.motion_stage.move_by({"x": -dx_mm * 1000.0,
                                           "y": -dy_mm * 1000.0},
                                          source="controller")
            except MotionConfigError as exc:
                # Preserve the existing XYStageProtocol error contract.
                raise StageError(str(exc)) from exc

        return DryRunStage(_move)

    # ---------------- snapshot（滞后一帧：由上一视觉结果构造）
    def bind_pipeline(self, pipeline: VisionPipeline) -> None:
        self.pipeline = pipeline

    def snapshot(self) -> WorkspaceSnapshot:
        self.frame_counter += 1
        particles = (self.pipeline.tracker.active_particles()
                     if self.pipeline is not None else [])
        obstacles: List[Obstacle] = []
        for x, y, r in self._obstacles:   # 相机固定（窗口坐标）
            obstacles.append(Obstacle(kind="circle", center=(x, y),
                                      radius=r, obstacle_id="sim-obs"))
        for p in particles:
            if p.track_id == self.target_track_id:
                continue  # 目标球不是障碍
            obstacles.append(Obstacle(kind="circle",
                                      center=p.position_px,
                                      radius=float(p.radius_px),
                                      obstacle_id=f"ball-{p.track_id}"))
        return WorkspaceSnapshot(
            frame_id=self.frame_counter, timestamp=time.time(),
            substrate=self.substrate, obstacles=obstacles,
            particles=particles, transform=self.transform,
            frame_size=self.window)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.cam.disable()
        except Exception:  # noqa: BLE001
            pass
        finally:
            close_db = getattr(self._db, "close", None)
            if callable(close_db):
                close_db()
            self._closed = True


class _SimLensDetector(ClassicDetector):
    """显微仿真检测器（ClassicDetector 协议适配）：

    - 衬底：显微样本为暗色组织纹理，ClassicDetector 的亮衬底阈值法
      （SimWorld 合成帧约定）不适用 —— 与 YoloDetector 同策略，
      衬底 = 视野内缩矩形（工作域 = 视野）。
    - 障碍：暗背景会污染暗区阈值检测，返回空；静态障碍由
      SimMicroscopeWorld 快照层注入，动态障碍由其他球构造。
    - 球检测：白盘 255 阈值法直接可用（继承）。
    """

    def __init__(self, substrate: SubstrateRegion, **kwargs) -> None:
        super().__init__(**kwargs)
        self._substrate = substrate

    def detect_substrate(self, frame: np.ndarray) -> SubstrateRegion:
        return self._substrate

    def detect_obstacles(self, frame: np.ndarray,
                         substrate: Optional[SubstrateRegion]) -> List[Obstacle]:
        return []


def build_sim_scenario(balls: Sequence[Tuple[Point, float]] = DEFAULT_BALLS,
                       static_obstacles: Sequence[Tuple[float, float, float]] =
                       DEFAULT_OBSTACLES,
                       goal: Point = GOAL, hint: Optional[Point] = TARGET_HINT,
                       cfg: Optional[ControllerConfig] = None,
                       sample_spec: Optional[Mapping[str, Mapping]] = None,
                       layout: Optional[Mapping] = None):
    """返回 (world, run)。sim01：显微镜仿真闭环（目标球 -> 终点绕障）。

    sample_spec：仿真镜头图层配置（掩码/衬底/障碍物），见 SimMicroscopeWorld。
    layout：UI ROI 划分的场景布局（窗口坐标，键均可选，未画项用默认值）：
      - balls [[x,y,w,h],..]：圆球框 -> 样本坐标球（框中心=球心，min(w,h)/2=半径）；
      - grounds [[x,y,w,h],..]：衬底框 -> 各框包围盒作为可行域多边形；
      - obstacles [[x,y,w,h],..]：障碍框 -> 相机固定圆；
      - goal [x,y]：目标点（窗口坐标）。
    """
    balls = list(balls)
    obstacles = list(static_obstacles)
    if layout:
        gx0 = SAMPLE_CENTER[0] - WINDOW[0] / 2.0
        gy0 = SAMPLE_CENTER[1] - WINDOW[1] / 2.0
        lb = layout.get("balls") or []
        if lb:
            balls = [((x + w / 2.0 + gx0, y + h / 2.0 + gy0), min(w, h) / 2.0)
                     for x, y, w, h in lb]
            hint = (lb[0][0] + lb[0][2] / 2.0, lb[0][1] + lb[0][3] / 2.0)
        lo = layout.get("obstacles") or []
        if lo:
            obstacles = [(x + w / 2.0, y + h / 2.0, min(w, h) / 2.0)
                         for x, y, w, h in lo]
        lg = layout.get("goal")
        if lg:
            goal = (float(lg[0]), float(lg[1]))
    world = SimMicroscopeWorld(balls=balls, static_obstacles=obstacles,
                               sample_spec=sample_spec)
    if layout and layout.get("grounds"):
        gs = layout["grounds"]
        x0 = min(g[0] for g in gs)
        y0 = min(g[1] for g in gs)
        x1 = max(g[0] + g[2] for g in gs)
        y1 = max(g[1] + g[3] for g in gs)
        if x1 - x0 > 20 and y1 - y0 > 20:
            world.substrate = SubstrateRegion(
                polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                safety_margin_px=4.0)
    goal_region = GoalRegion(center=goal, radius_px=25)
    detector = _SimLensDetector(world.substrate, min_particle_radius_px=8.0)

    def run(world=world, detector=detector, goal=goal_region, hint=hint,
            cfg=cfg, rep=None, stage_factory=None, stage_sink=None):
        from .models import FailureReason   # noqa: PLC0415
        from .vision import ParticleTracker  # noqa: PLC0415
        planner = GridPlanner()
        cfg = cfg or ControllerConfig(
            max_step_mm=0.05,      # 0.05mm = 100px @2000px/mm
            tolerance_px=8.0, stable_frames=3, max_iterations=400,
            max_track_jump_px=250.0)
        # 一步位移 100px > tracker 默认关联半径 40px，必须放宽，
        # 否则目标 track 停留原地（coast）-> 滑移误报
        pipeline = VisionPipeline(
            detector, tracker=ParticleTracker(max_jump_px=220.0))
        stage = (stage_factory() if stage_factory is not None
                 else world.make_stage())
        if stage_sink is not None:
            stage_sink.append(stage)
        try:
            snap, tid = _initial_detect(world, detector, hint,
                                        pipeline=pipeline)
            reason = planner.check_point(snap, goal.center)
            if reason is FailureReason.LOW_CLEARANCE:
                fixed = _nudge_to_feasible(planner, snap, goal.center)
                if fixed is not None:
                    if rep is not None:
                        rep.log("goal_adjusted", from_px=list(goal.center),
                                to_px=list(fixed))
                    goal = GoalRegion(center=fixed, radius_px=goal.radius_px)
            ctl = ObstacleAvoidController(stage, world.pipeline,
                                          planner, cfg, rep)
            return ctl.run(snap, tid, goal, task_id="sim01",
                           get_frame=world.render, get_snapshot=world.snapshot)
        finally:
            world.close()

    return world, run
