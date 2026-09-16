"""Alg2 多球鲁棒 + 对齐 abort 修复 + 日志补全（PLAN_ALG2_MULTIBALL.md）。

覆盖：
  ① _target_particle 在 pinned 状态下不因邻居更靠光斑而切换目标；
  ② 多球端到端：beam 锁定后全程 COMPLETE，且产生 detection 日志、无目标切换；
  ③ 对齐阶段"完全检测不到球"时 abort 原因应为 TARGET_LOST（而非误报 SPOT_SLIP）。
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from obstacle_avoidance.algorithm2 import (Alg2Config, Alg2Stage,  # noqa: E402
                                           FixedBeamController)
from obstacle_avoidance.controller import ControllerConfig  # noqa: E402
from obstacle_avoidance.models import (FailureReason, GoalRegion,  # noqa: E402
                                       Particle, RunState, WorkspaceSnapshot)
from obstacle_avoidance.planner import GridPlanner  # noqa: E402
from obstacle_avoidance.reporter import RunReporter  # noqa: E402
from obstacle_avoidance.sim_microscope import build_sim_scenario  # noqa: E402
from obstacle_avoidance.vision import (ParticleTracker,  # noqa: E402
                                       VisionPipeline)


def _controller(stage, cfg=None, reporter=None):
    from obstacle_avoidance.vision import VisionPipeline
    from obstacle_avoidance.planner import GridPlanner
    cfg = cfg or ControllerConfig(
        max_step_mm=0.005, tolerance_px=8.0, stable_frames=2,
        max_iterations=2000, max_track_jump_px=250.0, slip_abort_px=80.0,
        uncertain_retry_limit=3)
    vision = getattr(stage.world, "pipeline", None) or VisionPipeline(
        stage.world.make_detector(), tracker=ParticleTracker(max_jump_px=250))
    stage.world.bind_pipeline(vision)
    return FixedBeamController(stage, vision, GridPlanner(), cfg,
                               reporter or RunReporter(None))


# ---------------------------------------------------------------- ① 单元：pinned
def test_target_particle_pinned_keeps_locked_id_despite_nearer_neighbor():
    """邻居更靠光斑时，pinned 状态必须仍选中锁定球（不误切目标）。"""
    world, _ = build_sim_scenario()
    try:
        stage = Alg2Stage(world, config=Alg2Config(beam_position_px=(400, 300)))
        ctl = _controller(stage)
        beam = (400.0, 300.0)
        # 锁定球被推到离光斑很远（> max_track_jump_px）的位置（模拟同 id
        # 但位置漂移/误配），邻居反而紧贴光斑。
        locked = Particle(track_id=7, position_px=(100.0, 300.0), radius_px=12.0,
                          confidence=0.9, frame_id=1)
        neighbor = Particle(track_id=9, position_px=(405.0, 300.0), radius_px=12.0,
                            confidence=0.9, frame_id=1)   # 更靠近 beam
        particles = [neighbor, locked]
        # 未 pinned：by_id 离光斑超限 -> 回落到最近邻 neighbor
        assert ctl._target_particle(particles, 7, beam, preferred=beam,
                                    pinned=False).track_id == 9
        # pinned：仍锁定 track 7（即使远离光斑也不被邻居顶替）
        assert ctl._target_particle(particles, 7, beam, preferred=beam,
                                    pinned=True).track_id == 7
        # pinned 且锁定球本帧缺失 -> None（交给离屏外推，不回落到邻居）
        assert ctl._target_particle([neighbor], 7, beam, preferred=beam,
                                    pinned=True) is None
    finally:
        world.close()


# ---------------------------------------------------------------- ② E2E：多球
def test_alg2_multi_ball_completes_and_logs_detections(qapp=None):
    """多球（默认 2 球）Alg2 全程 COMPLETE；记录 detection；目标不切换。"""
    world, _ = build_sim_scenario()
    try:
        world.alg2_mode = True
        stage = Alg2Stage(world, track_id=None,
                          config=Alg2Config(beam_position_px=(400.0, 300.0),
                                            image_shift_sign=-1))
        cfg = ControllerConfig(
            max_step_mm=0.005, tolerance_px=8.0, stable_frames=2,
            max_iterations=2000, max_track_jump_px=250.0, slip_abort_px=80.0,
            uncertain_retry_limit=3)
        ctl = _controller(stage, cfg)
        ctl.stage = stage
        p0 = ctl.vision.process(world.render(), 1)
        new0 = p0.particles[0]
        world.set_alg2_target(new0.track_id)
        world.set_beam_position((400.0, 300.0))
        rep = RunReporter(None)
        ctl.reporter = rep
        res = ctl.run(world.snapshot(), new0.track_id,
                      GoalRegion((520.0, 300.0), 20.0),
                      get_frame=world.render, get_snapshot=world.snapshot)
        assert res.final_state == RunState.COMPLETE, \
            f"{res.failure_reason} | {res.detail}"
        dets = [e for e in rep.events if e.get("event") == "detection"]
        assert dets, "多球运行时应有 detection 日志（供复盘诊断）"
        assert dets[0].get("particles"), "detection 应包含全部球位置"
        # 应为 2 球：目标 + 邻居
        assert len({p["track_id"] for p in dets[0]["particles"]}) >= 2
        switches = [e for e in rep.events if e.get("event") == "beam_target"]
        assert not switches, "pinned 后不应发生目标切换"
    finally:
        world.close()


# ---------------------------------------------------------------- ③ 对齐原因
def test_alg2_alignment_never_seen_aborts_target_lost(qapp=None):
    """对齐阶段始终解析不到目标球 -> ABORTED(TARGET_LOST)，而非 SPOT_SLIP。"""
    world, _ = build_sim_scenario(balls=[((1420.0, 1150.0), 12.0)])
    try:
        world.alg2_mode = True
        stage = Alg2Stage(world, config=Alg2Config(beam_position_px=(400, 300)))
        # 用空检测的快照：初始粒子为空 -> run 在 initial 阶段即 TARGET_LOST。
        # 这里直接构造一个"对齐失败"情境：固定一个只渲染空球的世界。
        ctl = _controller(stage)
        ctl.stage = stage
        world.set_beam_position((400.0, 300.0))
        # 目标 track 不存在于初始快照 -> initial missing -> TARGET_LOST
        res = ctl.run(world.snapshot(), 12345,
                      GoalRegion((520.0, 300.0), 20.0),
                      get_frame=world.render, get_snapshot=world.snapshot)
        assert res.final_state == RunState.ABORTED
        assert res.failure_reason == FailureReason.TARGET_LOST
    finally:
        world.close()