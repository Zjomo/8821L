"""Alg2 多球：只移动离标定点最近的球 + UI 最终位置更新（需求1/2）。

覆盖：
  ① `WorkerThread._alg2_nearest_ball_idx` 选出离光斑最近者（非默认 0 号球）；
  ② 标定点未配置时退化为 ball_idx[0]，且只对给定子集生效；
  ③ Alg2 单球路径在存在第二个球时仍把目标球移动到目标点（COMPLETE），
     由于多球 Alg2 改走单球路径，其最终位置会被回写 UI。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from obstacle_avoidance.qt_compat import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _wt(qapp) -> "object":
    from obstacle_avoidance.app import WorkerThread
    return WorkerThread("sim01", {}, parent=None)  # 不 start()，仅作方法宿主


# ---------------------------------------------------------------- ① 选择最近球
def test_alg2_nearest_ball_prefers_beam_nearest(qapp):
    wt = _wt(qapp)
    wt._motion_params = {"beam_position_px": (500.0, 300.0)}
    balls = [[0, 0, 20, 20], [100, 100, 20, 20]]
    # 球0 中心 (10,10)，球1 中心 (110,110)；标定点在 (500,300) -> 球1 更近
    assert wt._alg2_nearest_ball_idx([0, 1], balls) == 1
    # 标定点移向球0 附近 -> 选球0
    wt._motion_params = {"beam_position_px": (11.0, 10.0)}
    assert wt._alg2_nearest_ball_idx([0, 1], balls) == 0


def test_alg2_nearest_ball_subset_and_fallback(qapp):
    wt = _wt(qapp)
    wt._motion_params = {}
    balls = [[0, 0, 20, 20], [100, 100, 20, 20], [200, 200, 20, 20]]
    # 未配置标定点 -> 返回子集第一个（即使 0 号不在子集）
    assert wt._alg2_nearest_ball_idx([1, 2], balls) == 1
    # 只考虑给定子集，不与子集外球比较
    wt._motion_params = {"beam_position_px": (210.0, 210.0)}
    assert wt._alg2_nearest_ball_idx([0, 2], balls) == 2


# ---------------------------------------------------------------- ③ 目标球到位
def test_alg2_single_oa_reaches_goal_with_second_ball(qapp):
    """存在第二个球时，Alg2 仍把目标球移到目标点（多球改走单球路径）。"""
    from obstacle_avoidance.app import WorkerThread
    from obstacle_avoidance.models import RunState
    from obstacle_avoidance.reporter import RunReporter
    from obstacle_avoidance.sim_microscope import build_sim_scenario
    from obstacle_avoidance.vision import ParticleTracker, VisionPipeline
    world, _ = build_sim_scenario()
    init = None
    try:
        world.alg2_mode = True
        try:
            init = dict(world.motion_stage.position)  # 记录初始，close 前还原
        except Exception:  # noqa: BLE001
            init = None
        det = world.make_detector()
        vp = VisionPipeline(det, tracker=ParticleTracker(max_jump_px=220))
        world.bind_pipeline(vp)
        parts = vp.process(world.render(), 1).particles
        assert len(parts) >= 2, f"期望至少 2 球，got {len(parts)}"
        target, other = sorted(parts, key=lambda p: p.position_px[0])[:2]
        ball_rect = [int(target.position_px[0] - 6),
                     int(target.position_px[1] - 6), 12, 12]
        # 目标点在远离其它球/静态障碍的空旷区（靠左），避免 LOW_CLEARANCE
        goal = (120.0, target.position_px[1])
        wt = WorkerThread("sim01", {}, parent=None)
        wt._algorithm = "Alg2"
        wt._motion_params = {
            "beam_position_px": (target.position_px[0],
                                 target.position_px[1]),
            "step_mm": 0.005,
            "speed_ums": 500.0,
            "accel_ums2": 600.0,
            "steps_per_mm": 33333.0,
        }
        rep = RunReporter(None)
        res = wt._run_single_oa(world, rep, ball_rect, goal)
        assert res is not None, "Alg2 应成功移动目标球"
        assert res.final_state == RunState.COMPLETE, (
            f"{res.failure_reason} | {res.detail}")
    finally:
        # 还原共享 SQLite 台位位置，避免留下分数 origin 污染后续测试
        try:
            if init:
                world.motion_stage.move_to(init)
        except Exception:  # noqa: BLE001
            pass
        world.close()