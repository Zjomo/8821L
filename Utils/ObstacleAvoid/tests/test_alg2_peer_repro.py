"""复现报告：Alg2 多球 TRACKING 起始 low_clearance（start point rejected）。

模拟 app._run_single_oa 传 peer_balls 的真实路径，验证 peer 半径扣减修复后
不再因 start(光束驻点) 被判 LOW_CLEARANCE、目标球可达 COMPLETE。
"""
import math
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


def _three_ball_layout():
    """球3(目标)起始贴近光束(400,300)，球1/球2 为 peer 的布局（窗口坐标）。"""
    return {
        "balls": [
            [394, 294, 12, 12],   # 球3 目标，贴近 beam (400,300)
            [188, 543, 12, 12],   # 球1 peer
            [200, 301, 12, 12],   # 球2 peer
        ],
        # 衬底覆盖整个可行域
        "grounds": [[20, 20, 760, 560]],
        # 障碍放在远离光束的空旷区，保证不干预 start
        "obstacles": [[680, 420, 60, 60]],
    }


def test_alg2_three_ball_peer_not_low_clearance(qapp):
    from obstacle_avoidance.app import WorkerThread
    from obstacle_avoidance.models import RunState
    from obstacle_avoidance.reporter import RunReporter
    from obstacle_avoidance.sim_microscope import build_sim_scenario, SIM_PIXEL_SIZE
    from obstacle_avoidance.vision import ParticleTracker, VisionPipeline

    world, _ = build_sim_scenario(layout=_three_ball_layout())
    init = None
    try:
        world.alg2_mode = True
        try:
            init = dict(world.motion_stage.position)
        except Exception:  # noqa: BLE001
            init = None
        det = world.make_detector()
        vp = VisionPipeline(det, tracker=ParticleTracker(max_jump_px=220))
        world.bind_pipeline(vp)
        parts = vp.process(world.render(), 1).particles
        assert len(parts) >= 3, f"期望至少 3 球，got {len(parts)}"
        target = min(parts, key=lambda p: math.dist(p.position_px, (400.0, 300.0)))
        peers = [p for p in parts if p.track_id != target.track_id]
        ball_rect = [int(target.position_px[0] - 6),
                     int(target.position_px[1] - 6), 12, 12]
        # 目标点选在 peer 环绕之外的空旷区
        goal = (500.0, 200.0)
        peer_balls = [(p.position_px[0], p.position_px[1], p.radius_px)
                      for p in peers]
        wt = WorkerThread("sim01", {}, parent=None)
        wt._algorithm = "Alg2"
        wt._motion_params = {
            "beam_position_px": (400.0, 300.0),
            "step_mm": 0.005, "speed_ums": 500.0, "accel_ums2": 600.0,
            "steps_per_mm": 33333.0,
        }
        rep = RunReporter(None)
        res = wt._run_single_oa(world, rep, ball_rect, goal, peer_balls)
        assert res is not None, "Alg2 应成功移动目标球"
        assert res.final_state == RunState.COMPLETE, (
            f"{res.failure_reason} | {res.detail}")
    finally:
        try:
            if init:
                world.motion_stage.move_to(init)
        except Exception:  # noqa: BLE001
            pass
        world.close()