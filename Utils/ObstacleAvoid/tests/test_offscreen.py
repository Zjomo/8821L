"""需求2 验收：始终记录画面中所有元素位置；元素离开画面范围时不得直接
`ABORTED【FAIL】`，而是用台账外推位置继续定位运动。

覆盖：
  ① PositionLedger：离屏元素仍保有位置记录，并按速度外推（in_frame=False）；
  ② ParticleTracker：丢检期间 track 不立刻删除，位置继续外推；
  ③ VisionPipeline：allow_offscreen=True 时离屏不算检测失败（不 uncertain）；
  ④ 控制器离屏元素选择：用外推位置继续定位（in_frame=False 才采用）；
  ⑤ GridPlanner.direct_plan：终点/起点/衬底离屏时给直行兜底计划；
  ⑥ 端到端：运行中目标点离开画面范围 -> 不 ABORTED(NO_SAFE_PATH)，
     记录 element_offscreen 事件且电机命令持续下发（定位运动不停止）。
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from obstacle_avoidance.controller import (ControllerConfig,  # noqa: E402
                                           ObstacleAvoidController)
from obstacle_avoidance.models import (CoordinateTransform,  # noqa: E402
                                       FailureReason, GoalRegion, Particle,
                                       RunState, SubstrateRegion,
                                       WorkspaceSnapshot)
from obstacle_avoidance.planner import (CollisionModel,  # noqa: E402
                                        GridPlanner, PlanConfig)
from obstacle_avoidance.reporter import RunReporter  # noqa: E402
from obstacle_avoidance.simulator import SimWorld  # noqa: E402
from obstacle_avoidance.vision import (ClassicDetector,  # noqa: E402
                                       ParticleTracker, PositionLedger,
                                       VisionPipeline)

R = CollisionModel().ball_radius_px


# ---------------------------------------------------------------- 单元：台账
def test_position_ledger_keeps_offscreen_positions():
    """① 台账始终记录元素位置；离屏后按速度外推且标记 in_frame=False。"""
    led = PositionLedger(max_age_frames=30)
    led.record([Particle(track_id=5, position_px=(600.0, 240.0),
                         radius_px=12.0, frame_id=4,
                         velocity_px_s=(40.0, 0.0))],
               4, (640, 480))
    out = led.predict(7, known_ids=[], frame_size=(640, 480))
    assert len(out) == 1
    assert out[0].track_id == 5
    assert out[0].position_px == (720.0, 240.0)   # 600 + 40 * 3 帧
    assert out[0].in_frame is False               # 已离开画面范围
    # 台账条目（最后已知位置）仍在
    assert led.entry(5) is not None
    assert led.entry(5).position == (600.0, 240.0)
    # 本帧已跟踪到的元素不重复外推
    assert led.predict(7, known_ids=[5], frame_size=(640, 480)) == []
    # 超出最大年龄 -> 台账失效（此后才允许回到中止路径）
    assert led.predict(4 + 31, [], (640, 480)) == []


def test_tracker_keeps_track_alive_and_extrapolates():
    """② keep_lost_frames 内丢检不丢 ID，位置按速度继续外推。"""
    tracker = ParticleTracker(max_jump_px=200.0, max_lost_frames=3,
                              keep_lost_frames=30)
    out = tracker.update([((100.0, 100.0), 12.0, 0.9)], 1)
    tid = out[0].track_id
    tracker.update([((120.0, 100.0), 12.0, 0.9)], 2)   # 建立速度 20px/帧
    last_x = 120.0
    for fid in range(3, 13):                            # 连续 10 帧丢检
        out = tracker.update([], fid)
        assert len(out) == 1, "keep_lost_frames 内不应删除 track"
        assert out[0].track_id == tid
        assert out[0].position_px[0] > last_x - 1e-9, "位置应继续外推"
        last_x = out[0].position_px[0]
    assert last_x > 180.0
    # 旧行为（默认 coast 3 帧）仍然保留
    legacy = ParticleTracker(max_jump_px=200.0, max_lost_frames=3)
    legacy.update([((100.0, 100.0), 12.0, 0.9)], 1)
    for fid in range(2, 7):
        legacy.update([], fid)
    assert legacy.update([], 7) == []


# ---------------------------------------------------------------- 单元：视觉
def _ball_world(center=(600.0, 240.0)):
    return SimWorld(
        substrate_polygon=[(40, 40), (600, 40), (600, 440), (40, 440)],
        obstacles=[], particles=[(center, 12.0)], frame_size=(640, 480))


def test_pipeline_offscreen_is_not_uncertain():
    """③ allow_offscreen=True：球离开画面范围 -> 不判 uncertain，位置可外推。"""
    world = _ball_world()
    pipe = VisionPipeline(ClassicDetector(),
                          ParticleTracker(max_jump_px=200.0))
    res = pipe.process(world.render(), 1, allow_offscreen=True)
    assert len(res.particles) == 1
    tid = res.particles[0].track_id
    # 球向右快速移动，随后移出画面（不再被检测到）
    world.move_particle(tid, 30.0, 0.0)
    pipe.process(world.render(), 2, allow_offscreen=True)
    world.move_particle(tid, 30.0, 0.0)                # 球心 660 已在画面外
    res = pipe.process(world.render(), 3, allow_offscreen=True)
    assert res.uncertain is False, "离屏不是检测失败，不应中止"
    assert res.uncertain_reason.startswith("offscreen")
    assert res.offscreen_ids, "应记录离屏元素 ID"
    assert any(not p.in_frame for p in res.particles)
    assert any(p.track_id == tid for p in res.particles), "台账仍保有该球位置"


def test_pipeline_without_offscreen_flag_still_safe():
    """③b 关闭开关时保持原有安全语义：画面内丢检仍判 uncertain（不盲动）。"""
    world = _ball_world()
    pipe = VisionPipeline(ClassicDetector(),
                          ParticleTracker(max_jump_px=200.0))
    res = pipe.process(world.render(), 1)
    tid = res.particles[0].track_id
    world.move_particle(tid, 30.0, 0.0)
    pipe.process(world.render(), 2)
    world.move_particle(tid, 30.0, 0.0)
    res = pipe.process(world.render(), 3)              # 默认 allow_offscreen=False
    assert res.uncertain is True
    assert res.offscreen_ids == []


def test_controller_picks_offscreen_particle_by_extrapolation():
    """④ 控制器离屏元素选择：只用 in_frame=False 的外推位置。"""
    off = Particle(track_id=7, position_px=(700.0, 240.0), radius_px=12.0,
                   confidence=0.3, frame_id=5, in_frame=False)
    fresh = Particle(track_id=7, position_px=(100.0, 100.0), radius_px=12.0,
                     confidence=0.9, frame_id=6, in_frame=True)
    pick = ObstacleAvoidController._offscreen_particle
    assert pick([off], 7, (100.0, 100.0)) is off
    # track_id 不匹配时退化为"离参考位置最近的外推元素"
    assert pick([off], 99, (0.0, 0.0)) is off
    # 画面内元素不参与外推（保持原有中止/不确定语义）
    assert pick([fresh], 7, (100.0, 100.0)) is None
    assert pick([], 7, (100.0, 100.0)) is None


# ---------------------------------------------------------------- 单元：规划
def _snap(frame_size=(640, 480), dx=0.0, dy=0.0):
    return WorkspaceSnapshot(
        frame_id=0, timestamp=0.0,
        substrate=SubstrateRegion(
            polygon=[(40 + dx, 40 + dy), (600 + dx, 40 + dy),
                     (600 + dx, 440 + dy), (40 + dx, 440 + dy)],
            safety_margin_px=6.0),
        obstacles=[], particles=[],
        transform=CoordinateTransform(px_per_mm=100.0),
        frame_size=frame_size)


def test_direct_plan_fallback_for_offscreen_elements():
    """⑤ 终点/起点/衬底离屏 -> 直行兜底；全部在画面内 -> 不兜底。"""
    planner = GridPlanner(PlanConfig())
    snap = _snap()
    assert planner.direct_plan(snap, (200.0, 200.0), (400.0, 300.0)) is None

    goal_off = planner.direct_plan(snap, (200.0, 200.0), (700.0, 300.0))
    assert goal_off is not None and goal_off.success
    assert goal_off.waypoints_px == [(200.0, 200.0), (700.0, 300.0)]
    assert "goal" in goal_off.detail
    # clearance 视为无限（兜底路径不做栅格避让判定，避免被 LOW_CLEARANCE 中止）
    assert goal_off.min_clearance_px == float("inf")

    start_off = planner.direct_plan(snap, (-50.0, 200.0), (300.0, 300.0))
    assert start_off is not None and "start" in start_off.detail

    # 衬底整体移出画面（栅格边界失效）
    sub_off = planner.direct_plan(_snap(dx=-2000.0), (200.0, 200.0),
                                  (400.0, 300.0))
    assert sub_off is not None and sub_off.detail.startswith("offscreen direct step")


# ---------------------------------------------------------------- 端到端
def test_oa_goal_offscreen_keeps_moving(qapp=None):
    """⑥ 目标点离屏：不 ABORTED(NO_SAFE_PATH)，记录事件且继续下发命令。"""
    world = SimWorld(
        substrate_polygon=[(40, 40), (600, 40), (600, 440), (40, 440)],
        obstacles=[], particles=[((150.0, 240.0), R)], frame_size=(640, 480))
    goal = GoalRegion(center=(480.0, 240.0), radius_px=20)   # 全程在画面内
    world.on_frame(lambda fid: setattr(goal, "center", (700.0, 240.0))
                   if fid >= 4 else None)                     # 运行中移出画面
    rep = RunReporter(None)
    snap = world.snapshot()
    stage = world.make_stage(1)
    ctl = ObstacleAvoidController(
        stage, VisionPipeline(expected_radius_px=R), GridPlanner(),
        ControllerConfig(max_iterations=30), rep)
    try:
        result = ctl.run(snap, 1, goal, task_id="offscreen",
                         get_frame=world.render, get_snapshot=world.snapshot)
    finally:
        rep.close()
    offs = [e for e in rep.events if e.get("event") == "element_offscreen"]
    assert offs, "目标点离屏应记录 element_offscreen 事件（直行兜底）"
    assert any(e.get("role") == "goal" for e in offs)
    assert result.final_state != RunState.ABORTED, \
        f"离屏不应直接中止: {result.failure_reason} {result.detail}"
    assert result.failure_reason != FailureReason.NO_SAFE_PATH
    assert len(result.stage_commands) >= 15, "离屏后定位运动必须继续"
    # 球被持续向右推出画面（真值位置超过画面右边界）
    assert world.particle_position(1)[0] > 640.0
    assert math.isclose(world.particle_position(1)[1], 240.0, abs_tol=1e-6)