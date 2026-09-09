"""真实视频闭环仿真（YOLO 后端 + 既有安全控制器）。

模型约定（与真实位移台一致）：
- 相机固定，位移台移动样品 -> 命令 (dx,dy) 使视野内容平移 +(dx,dy)，
  即目标球在画面中移动 +(dx,dy)·ppm，其余球作为动态障碍同向漂移。
- 每次取帧推进一帧视频（50fps 实拍序列，含真实纹理/噪声/微小漂移）。
- 视频中未被模型检出的物体（如失焦小球）不进入碰撞模型——已在场景
  选择时规避其邻域，报告中的 clearance 不包含未建模物体。

场景：
- video01：目标球（画面左下）-> 终点，另一检出球作为动态障碍。
- video02：在 video01 基础上叠加用户标注静态障碍（深色圆盘，渲染可见），
  强制绕行，验证真实图像上的重规划。
"""
from __future__ import annotations

import math
import os
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .controller import ControllerConfig, ObstacleAvoidController
from .models import (CoordinateTransform, GoalRegion, Obstacle, Point,
                     SubstrateRegion, WorkspaceSnapshot)
from .planner import CollisionModel, GridPlanner, PlanConfig
from .roi_zones import RoiConfig, Zone
from .simulator import DryRunStage
from .vision import (VisionPipeline, YoloDetector, get_shared_detector)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(PROJECT_ROOT, "weight", "Duan_best.pt")
VIDEO = os.path.join(PROJECT_ROOT, "Dataset", "v2.mp4")
DEFAULT_ROI_CONFIG = os.path.join(PROJECT_ROOT, "artifacts", "roi_config.json")

WINDOW = (480, 420)          # 仿真视野 (w, h)，为窗口平移保留行程
WINDOW_OFFSET = (257.0, 57.0)  # 视野左上角在视频坐标中的初始位置
GOAL = (430.0, 100.0)        # 终点（窗口坐标，距衬底边界 >= 膨胀半径）
TARGET_HINT = (265.0, 235.0)  # 目标球初始位置提示（窗口坐标，用于选 track）
PX_PER_MM = 100.0            # 0.3mm/步 -> 30px/步，低于跟踪跳变阈值


class VideoWorld:
    """视频驱动的闭环世界：裁剪窗口=相机视野，stage 平移窗口。"""

    def __init__(self, video_path: str = VIDEO,
                 window: Tuple[int, int] = WINDOW,
                 offset: Point = WINDOW_OFFSET,
                 px_per_mm: float = PX_PER_MM,
                 static_obstacles: Sequence[Obstacle] = ()) -> None:
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        self.window = window
        self.offset = [float(offset[0]), float(offset[1])]
        self.transform = CoordinateTransform(px_per_mm=px_per_mm)
        self.substrate = SubstrateRegion(
            polygon=[(10.0, 10.0),
                     (window[0] - 10.0, 10.0),
                     (window[0] - 10.0, window[1] - 10.0),
                     (10.0, window[1] - 10.0)],
            safety_margin_px=4.0)
        self.static_obstacles: List[Obstacle] = list(static_obstacles)
        self.pipeline: Optional[VisionPipeline] = None  # run 前注入
        self.target_track_id: int = -1                  # 初始检测后设置
        self.frame_counter = 0
        self._last_frame: Optional[np.ndarray] = None
        self._eof = False

    # ---------------- sensor
    def _read_video_frame(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:  # 循环播放
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("video read failed")
        ox, oy = int(self.offset[0]), int(self.offset[1])
        w, h = self.window
        vh, vw = frame.shape[:2]
        ox = max(0, min(vw - w - 1, ox))
        oy = max(0, min(vh - h - 1, oy))
        self.offset = [float(ox), float(oy)]
        return frame[oy:oy + h, ox:ox + w].copy()

    def render(self) -> np.ndarray:
        """推进一帧视频并裁剪窗口；叠加用户标注障碍（深色圆盘）。"""
        frame = self._read_video_frame().copy()
        for ob in self.static_obstacles:
            if ob.kind == "circle":
                cv2.circle(frame, (int(ob.center[0]), int(ob.center[1])),
                           int(ob.radius), (30, 30, 30), -1)
            else:
                poly = np.array(ob.polygon, dtype=np.int32)
                cv2.fillPoly(frame, [poly], (30, 30, 30))
        self._last_frame = frame
        return frame

    # ---------------- stage
    def set_window(self, window: Tuple[int, int],
                   offset: Tuple[float, float]) -> None:
        """重设固定视野（UI 划定 ROI 后调用），并同步衬底可行域。"""
        self.window = (int(window[0]), int(window[1]))
        self.offset = [float(offset[0]), float(offset[1])]
        self.substrate = SubstrateRegion(
            polygon=[(10.0, 10.0),
                     (self.window[0] - 10.0, 10.0),
                     (self.window[0] - 10.0, self.window[1] - 10.0),
                     (10.0, self.window[1] - 10.0)],
            safety_margin_px=4.0)

    def shift_window(self, dx_px: float, dy_px: float) -> None:
        """窗口反向平移：固定物在画面中平移 +(dx_px, dy_px)。"""
        self.offset[0] -= dx_px
        self.offset[1] -= dy_px

    def make_stage(self) -> DryRunStage:
        def _move(dx_mm: float, dy_mm: float) -> None:
            self.shift_window(dx_mm * self.transform.px_per_mm,
                              dy_mm * self.transform.px_per_mm)

        return DryRunStage(_move)

    def particle_position(self, track_id: int) -> Point:
        if self.pipeline is None:
            raise KeyError(track_id)
        for item in self.pipeline.tracker.active_particles():
            if item.track_id == track_id:
                return item.position_px
        raise KeyError(track_id)

    # ---------------- snapshot（滞后一帧：由上一视觉结果构造）
    def bind_pipeline(self, pipeline: VisionPipeline) -> None:
        self.pipeline = pipeline

    def snapshot(self) -> WorkspaceSnapshot:
        self.frame_counter += 1
        particles = (self.pipeline.tracker.active_particles()
                     if self.pipeline is not None else [])
        obstacles: List[Obstacle] = list(self.static_obstacles)
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


def _video_size(video_path: str = VIDEO) -> Tuple[int, int]:
    """视频原始尺寸 (w, h)。"""
    cap = cv2.VideoCapture(video_path)
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


def camera_source(index: int):
    """真实相机帧源：返回 read() -> Optional[np.ndarray] 的可调用。

    附带 close() 属性用于释放相机（UI 退出/停止实时检测时调用）。
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {index}")

    def _read() -> Optional[np.ndarray]:
        ok, frame = cap.read()
        return frame if ok else None

    def _close() -> None:
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass

    _read.close = _close
    return _read


def screen_source(region=None, monitor: int = 1):
    """屏幕捕获帧源：截取某个显示屏的一部分区域。

    region=(x,y,w,h) 为虚拟桌面全局坐标；None 时取整块显示器。
    返回 read() -> Optional[np.ndarray]（BGR）可调用，附带 close()。
    """
    try:
        import mss
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mss 未安装: pip install mss") from exc
    if region is not None:
        if len(region) != 4 or int(region[2]) <= 0 or int(region[3]) <= 0:
            raise RuntimeError(f"screen region 非法: {region}")
    sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
    if region is None:
        mons = sct.monitors
        m = mons[monitor] if 0 < monitor < len(mons) else mons[1]
        region = (int(m["left"]), int(m["top"]),
                  int(m["width"]), int(m["height"]))
    x, y, w, h = (int(v) for v in region)
    box = {"left": x, "top": y, "width": w, "height": h}

    def _read() -> Optional[np.ndarray]:
        img = np.asarray(sct.grab(box))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _close() -> None:
        try:
            sct.close()
        except Exception:  # noqa: BLE001
            pass

    _read.close = _close
    return _read


class CameraWorld:
    """电机模式世界：真实相机帧源 + ROI 视野 + YOLO/classic 检测。

    与 VideoWorld 的区别：不模拟图像平移（真实位移台负责移动样品），
    stage 由外部注入（SerialXYStage）。frame_source 可注入用于测试。
    """

    def __init__(self, frame_source, window: Tuple[int, int],
                 offset: Tuple[float, float] = (0.0, 0.0),
                 static_obstacles: Sequence[Obstacle] = (),
                 px_per_mm: float = 100.0) -> None:
        self._source = frame_source
        self.window = (int(window[0]), int(window[1]))
        self.offset = [float(offset[0]), float(offset[1])]
        self.static_obstacles = list(static_obstacles)
        self.transform = CoordinateTransform(px_per_mm=px_per_mm)
        self.substrate = SubstrateRegion(
            polygon=[(10.0, 10.0),
                     (self.window[0] - 10.0, 10.0),
                     (self.window[0] - 10.0, self.window[1] - 10.0),
                     (10.0, self.window[1] - 10.0)],
            safety_margin_px=4.0)
        self.pipeline = None
        self.target_track_id: Optional[int] = None
        self._frame_counter = 0

    def bind_pipeline(self, pipeline: VisionPipeline) -> None:
        self.pipeline = pipeline

    def close(self) -> None:
        """释放底层相机资源（frame_source 需附带 close 属性）。"""
        close = getattr(self._source, "close", None)
        if callable(close):
            close()

    def set_window(self, window: Tuple[int, int],
                   offset: Tuple[float, float]) -> None:
        """重设固定视野（UI 划定 ROI 后调用），并同步衬底可行域。"""
        self.window = (int(window[0]), int(window[1]))
        self.offset = [float(offset[0]), float(offset[1])]
        self.substrate = SubstrateRegion(
            polygon=[(10.0, 10.0),
                     (self.window[0] - 10.0, 10.0),
                     (self.window[0] - 10.0, self.window[1] - 10.0),
                     (10.0, self.window[1] - 10.0)],
            safety_margin_px=4.0)

    def render(self) -> Optional[np.ndarray]:
        frame = self._source()
        if frame is None:
            return None
        x, y = int(self.offset[0]), int(self.offset[1])
        w, h = self.window
        H, W = frame.shape[:2]
        x0, y0 = max(0, min(x, W - 1)), max(0, min(y, H - 1))
        x1, y1 = min(W, x0 + w), min(H, y0 + h)
        crop = frame[y0:y1, x0:x1]
        if crop.shape[0] != h or crop.shape[1] != w:
            crop = cv2.copyMakeBorder(crop, 0, max(0, h - crop.shape[0]),
                                      0, max(0, w - crop.shape[1]),
                                      cv2.BORDER_CONSTANT)
        return crop

    def particle_position(self, track_id: int) -> Point:
        if self.pipeline is None:
            raise KeyError(track_id)
        for item in self.pipeline.tracker.active_particles():
            if item.track_id == track_id:
                return item.position_px
        raise KeyError(track_id)

    def snapshot(self) -> WorkspaceSnapshot:
        self._frame_counter += 1
        particles = (self.pipeline.tracker.active_particles()
                     if self.pipeline is not None else [])
        obstacles: List[Obstacle] = list(self.static_obstacles)
        for p in particles:
            if p.track_id == self.target_track_id:
                continue
            obstacles.append(Obstacle(kind="circle", center=p.position_px,
                                      radius=float(p.radius_px),
                                      obstacle_id=f"ball-{p.track_id}"))
        return WorkspaceSnapshot(
            frame_id=self._frame_counter, timestamp=time.time(),
            substrate=self.substrate, obstacles=obstacles,
            particles=particles, transform=self.transform,
            frame_size=self.window)


def _initial_detect(world: VideoWorld, detector: YoloDetector,
                    hint: Optional[Point] = None,
                    pipeline: Optional[VisionPipeline] = None
                    ) -> Tuple[WorkspaceSnapshot, int]:
    """首帧检测：建立初始快照并选择目标 track（hint 最近者，无 hint 取最大球）。

    pipeline 可注入自定义 tracker（如 sim 场景需放宽 max_jump_px）。
    初始快照由 world.snapshot() 构建——含用户标注静态障碍（障碍区/仿真
    障碍），保证首次规划即绕行（此前仅含检出粒子，导致障碍不生效）。
    """
    frame = world.render()
    pipeline = pipeline or VisionPipeline(detector)
    world.bind_pipeline(pipeline)
    vis = pipeline.process(frame, 1)
    if not vis.particles:
        raise RuntimeError(
            f"首帧未检测到球（视野 {world.window[0]}x{world.window[1]}）："
            "请确认画面中有球、球未被障碍/图层遮挡、对比度足够（阈值 235），"
            "且光照正常")
    if hint is not None:
        target = min(vis.particles,
                     key=lambda p: math.dist(p.position_px, hint))
    else:  # 区域配置模式：取最靠近视野中心的球（确定性，且离边界最远）
        center = (world.window[0] / 2, world.window[1] / 2)
        target = min(vis.particles,
                     key=lambda p: math.dist(p.position_px, center))
    world.target_track_id = target.track_id
    snap = world.snapshot()   # 静态障碍 + 非目标粒子一并计入
    if not snap.particles:    # snapshot 与 vis 不同源时兜底
        snap = WorkspaceSnapshot(
            frame_id=1, timestamp=time.time(),
            substrate=world.substrate,
            obstacles=[Obstacle(kind="circle", center=p.position_px,
                                radius=float(p.radius_px),
                                obstacle_id=f"ball-{p.track_id}")
                       for p in vis.particles if p.track_id != target.track_id],
            particles=vis.particles, transform=world.transform,
            frame_size=world.window)
    return snap, target.track_id


def build_video_scenario(task: str = "video01", weights: str = WEIGHTS,
                         video: str = VIDEO, config: Optional["RoiConfig"] = None,
                         motor: Optional[dict] = None,
                         cfg_overrides: Optional[dict] = None,
                         task_mode: str = "oa",
                         controller_sink: Optional[dict] = None):
    """返回 (world, run_fn)，与 cli.build_scenario 契约一致。

    task=video03：从 RoiConfig 构建——ROI=固定视野，goal 区=终点，
    obstacle 区=静态障碍（用户标注），目标球取最靠近视野中心者。

    motor 非 None 时进入电机模式：真实相机帧源（motor["camera_index"]）+
    SerialXYStage（motor["port"]/["baudrate"]/["confirmed"]）。
    cfg_overrides 用于测试/调参覆盖 ControllerConfig 字段。
    """
    if not os.path.isfile(weights):
        raise SystemExit(f"weights not found: {weights}")
    if motor is None and not os.path.isfile(video):
        raise SystemExit(f"video not found: {video}")

    if task == "video03":
        cfg_path = None
        if config is None:
            cfg_path = DEFAULT_ROI_CONFIG
            if not os.path.isfile(cfg_path):
                raise SystemExit(
                    f"video03 需要区域配置文件: {cfg_path} "
                    "(在 UI 中用'实时检测+区域划分'生成，或 --roi-config 指定)")
            config = RoiConfig.load(cfg_path)
        gz = config.goal_zone()
        if gz is None:
            src = cfg_path or config.video or "内存配置"
            raise RuntimeError(
                f"区域配置缺少目标区(goal): {src} —— "
                "请在 UI 用'目标区'模式画框后保存配置，再运行 video03")
        window = (int(config.roi[2]), int(config.roi[3]))
        offset = (float(config.roi[0]), float(config.roi[1]))
        goal_pt = gz.center()
        goal = GoalRegion(center=(goal_pt[0] - offset[0], goal_pt[1] - offset[1]),
                          radius_px=max(12.0, min(gz.rect[2:4]) * 0.3))

        def _zone_poly_win(z: Zone) -> List[Point]:
            x, y, w, h = z.rect
            return [(x - offset[0], y - offset[1]),
                    (x - offset[0] + w, y - offset[1]),
                    (x - offset[0] + w, y - offset[1] + h),
                    (x - offset[0], y - offset[1] + h)]

        def _zone_obstacle_win(z: Zone) -> Obstacle:
            """区域 -> 障碍：圆形 zone 生成圆障碍（rect 为外接正方形），
            Free 自由多边形 zone 顶点首尾连通生成多边形障碍。"""
            if getattr(z, "shape", "rect") == "circle":
                cx, cy = z.center()
                return Obstacle(kind="circle",
                                center=(cx - offset[0], cy - offset[1]),
                                radius=z.radius(), obstacle_id=z.name)
            if getattr(z, "shape", "rect") == "free" and z.points:
                return Obstacle(kind="polygon",
                                polygon=[(px - offset[0], py - offset[1])
                                         for px, py in z.points],
                                obstacle_id=z.name)
            return Obstacle(kind="polygon", polygon=_zone_poly_win(z),
                            obstacle_id=z.name)

        static_obs = [_zone_obstacle_win(z)
                      for z in config.obstacle_zones()]
        px_per_mm = config.px_per_mm
        hint = None
        world = VideoWorld(video_path=video, window=window, offset=offset,
                           px_per_mm=px_per_mm, static_obstacles=static_obs)
    else:
        static_obs: List[Obstacle] = []
        if task == "video02":  # 用户标注静态障碍：位于 video01 直线路径中点
            mx, my = (TARGET_HINT[0] + GOAL[0]) / 2, (TARGET_HINT[1] + GOAL[1]) / 2
            static_obs = [Obstacle(kind="circle", center=(mx, my), radius=55.0,
                                   obstacle_id="obs-annotated")]
        world = VideoWorld(video_path=video, static_obstacles=static_obs)
        goal = GoalRegion(center=GOAL, radius_px=20)
        hint = TARGET_HINT

    detector = get_shared_detector(weights)   # 进程级单例，避免重复加载权重
    model = CollisionModel(ball_radius_px=25.0)   # 实测球半径 ~25px
    edge = getattr(config, "edge_clearance_px", None) if task == "video03" else None
    planner = GridPlanner(PlanConfig(model=model, edge_clearance_px=edge))
    # 电机/视频世界：stage 命令 (dx,dy) -> 窗口平移，目标球在画面中移动 +(dx,dy)
    # （与 shift_window 实测一致；真实机型图像位移符号需现场标定，
    #   电机模式经 motor["ball_shift_sign"] 覆盖）
    cfg = ControllerConfig(max_step_mm=0.30, tolerance_px=8.0,
                           stable_frames=3, max_iterations=300,
                           max_track_jump_px=90.0)
    # 双模型同步：AG 驻点间距/容量校验用 cfg.model，规划膨胀用 planner
    # 模型——必须同一实例，否则（如 motor 模式）环形驻点按 12px 间距
    # 排布而实际球 25px，聚拢完成后球体互相重叠
    cfg.model = model
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    # ---- 电机模式：真实相机 + 真实位移台
    default_stage_factory = None
    if motor is not None:
        if task != "video03":
            raise SystemExit("motor 模式当前仅支持 video03（需要 ROI/区域配置）")
        # 帧源可注入（屏幕区域捕获）；默认真实相机
        src = motor.get("frame_source")
        if src is None:
            src = camera_source(int(motor["camera_index"]))
        world = CameraWorld(
            frame_source=src,
            window=window, offset=offset,
            static_obstacles=static_obs, px_per_mm=px_per_mm)
        if motor.get("driver", "picomotor") == "serial":
            default_stage_factory = lambda: SerialXYStage(   # noqa: E731
                port=motor["port"], baudrate=int(motor.get("baudrate", 115200)),
                cmd_template=motor.get("cmd_template",
                                       "G91 G1 X{dx:.4f} Y{dy:.4f}\n"),
                max_step_mm=float(motor.get("max_step_mm", 0.30)),
                confirmed=bool(motor.get("confirmed", False)))
        elif motor.get("driver", "picomotor") == "kinesis":
            from .stages import KinesisKIM101Stage
            default_stage_factory = lambda: KinesisKIM101Stage(
                serial_no=motor.get("serial_no", ""),
                # Keep loading reports/configurations written by the old UI.
                serial_by_axis=motor.get("serial_by_axis"),
                steps_per_mm=float(motor.get("steps_per_mm", 1000.0)),
                steps_per_mm_by_axis=motor.get("steps_per_mm_by_axis"),
                speed_steps=motor.get("speed_steps"),
                max_step_mm=float(motor.get("max_step_mm", 0.30)),
                confirmed=bool(motor.get("confirmed", False)))
        else:   # 8742/8743 Picomotor（默认驱动）
            from .stages import PicoMotorStage
            default_stage_factory = lambda: PicoMotorStage(  # noqa: E731
                conn=int(motor.get("conn", 0)),
                x_axis=int(motor.get("x_axis", 1)),
                y_axis=int(motor.get("y_axis", 2)),
                z_axis=int(motor.get("z_axis", 3)),
                steps_per_mm=float(motor.get("steps_per_mm", 1000.0)),
                steps_per_mm_by_axis=motor.get("steps_per_mm_by_axis"),
                speed_steps=motor.get("speed_steps"),
                axes_sign=tuple(motor.get("axes_sign", (1.0, 1.0))),
                max_step_mm=float(motor.get("max_step_mm", 0.30)),
                confirmed=bool(motor.get("confirmed", False)))
        # 真实机型：图像位移符号（相机/衬底取向）需现场标定；
        # 校准方法：下发已知小步，观测目标球图像位移方向，反号则设 -1
        if "ball_shift_sign" in motor:
            cfg.ball_shift_sign = int(motor["ball_shift_sign"])

    def run(world=world, detector=detector, goal=goal, planner=planner,
            cfg=cfg, hint=hint, rep=None, stage_factory=None,
            stage_sink=None):
        from .models import FailureReason
        factory = stage_factory or default_stage_factory
        # Assembly creates one stage per ball through AggregationPlanner.  Do
        # not open an extra controller here, otherwise the USB handle remains
        # locked when the first per-ball stage is created.
        stage = (None if task_mode == "ag" and factory is not None
                 else (factory() if factory is not None else world.make_stage()))
        if stage is not None and stage_sink is not None:
            stage_sink.append(stage)
        try:
            if (motor is not None and motor.get("algorithm") == "Alg2"
                    and hasattr(stage, "prepare_focus")):
                # Alg2 fixed-beam operation explicitly selects the Z focus
                # before the shared XY controller starts.
                stage.prepare_focus(
                    z_safe_um=float(motor.get("z_safe_um", 5.0)),
                    z_focus_um=float(motor.get("z_focus_um", 0.0)))
            snap, tid = _initial_detect(world, detector, hint)
            # goal 自动微调：目标区中心 clearance 不足但仍在衬底内时，
            # 就近挪到第一个可行点（安全层拒绝之前最后一道用户体验防线）
            reason = planner.check_point(snap, goal.center)
            if reason is FailureReason.LOW_CLEARANCE:
                fixed = _nudge_to_feasible(planner, snap, goal.center)
                if fixed is not None:
                    if rep is not None:
                        rep.log("goal_adjusted", from_px=list(goal.center),
                                to_px=list(fixed))
                    goal = GoalRegion(center=fixed, radius_px=goal.radius_px)
            if task_mode == "ag":
                from .aggregation import AggregationConfig, AggregationPlanner
                def aggregation_stage_factory():
                    next_stage = factory() if factory is not None else world.make_stage()
                    if stage_sink is not None:
                        stage_sink.append(next_stage)
                    if (motor is not None and motor.get("algorithm") == "Alg2"
                            and hasattr(next_stage, "prepare_focus")):
                        next_stage.prepare_focus(
                            z_safe_um=float(motor.get("z_safe_um", 5.0)),
                            z_focus_um=float(motor.get("z_focus_um", 0.0)))
                    return next_stage
                ag = AggregationPlanner(
                    planner, world.pipeline,
                    AggregationConfig(required_count=len(snap.particles),
                                      controller=cfg),
                    rep, controller_sink=controller_sink,
                    stage_factory=(aggregation_stage_factory
                                   if motor is not None else None))
                return ag.run(world, snap, goal, task_id=task)
            ctl = ObstacleAvoidController(stage, world.pipeline,
                                          planner, cfg, rep)
            if controller_sink is not None:
                controller_sink["controller"] = ctl
            return ctl.run(snap, tid, goal, task_id=task,
                           get_frame=world.render, get_snapshot=world.snapshot)
        finally:
            close = getattr(stage, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

    return world, run


def _nudge_to_feasible(planner: GridPlanner, snap: WorkspaceSnapshot,
                       p: Point, max_r: float = 150.0,
                       step: float = 10.0) -> Optional[Point]:
    """环绕搜索最近可行点；仅在衬底内微调，找不到返回 None。"""
    if not snap.substrate.contains(p):
        return None
    best = None
    r = step
    while r <= max_r:
        n = max(8, int(2 * math.pi * r / step))
        for i in range(n):
            a = 2 * math.pi * i / n
            cand = (p[0] + r * math.cos(a), p[1] + r * math.sin(a))
            if planner.check_point(snap, cand) is None:
                if best is None or math.dist(cand, p) < math.dist(best, p):
                    best = cand
        if best is not None:
            return best
        r += step
    return best
