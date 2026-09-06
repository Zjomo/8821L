"""YOLO 后端 + 真实视频闭环仿真测试（weight/Duan_best.pt 可用时启用）。"""
import os

import numpy as np
import pytest

from obstacle_avoidance import video_sim

WEIGHTS = video_sim.WEIGHTS
VIDEO = video_sim.VIDEO
_needs_yolo = pytest.mark.skipif(
    not (os.path.isfile(WEIGHTS) and os.path.isfile(VIDEO)),
    reason="需要 weight/Duan_best.pt 与 Dataset/测试视频.mp4")


@_needs_yolo
def test_yolo_detector_protocol():
    """YoloDetector 与 ClassicDetector 同协议：可直接注入 VisionPipeline。"""
    from obstacle_avoidance.vision import VisionPipeline, YoloDetector
    world = video_sim.VideoWorld()
    det = YoloDetector(WEIGHTS)
    frame = world.render()
    sub = det.detect_substrate(frame)
    assert sub is not None and len(sub.polygon) == 4
    dets, amb = det.detect_particles(frame)
    assert len(dets) >= 2
    for (cx, cy), r, c in dets:
        assert 0 <= cx <= world.window[0] and 0 <= cy <= world.window[1]
        assert 10 <= r <= 45          # 实测大球 ~25px，小球 ~13px
        assert c >= 0.25
    assert amb == []
    pipe = VisionPipeline(det)
    vis = pipe.process(frame, 1)
    assert not vis.uncertain and len(vis.particles) >= 2


@_needs_yolo
def test_video_world_window_shift():
    """stage 命令 (mm) -> 窗口平移 (px)，方向与真实位移台一致。"""
    world = video_sim.VideoWorld()
    f1 = world.render().copy()
    stage = world.make_stage()
    stage.move_by(0.30, -0.10)                 # mm
    assert world.offset == pytest.approx(
        [video_sim.WINDOW_OFFSET[0] - 30.0,
         video_sim.WINDOW_OFFSET[1] + 10.0])   # 100 px/mm
    f2 = world.render()
    assert f1.shape == f2.shape == (world.window[1], world.window[0], 3)
    assert (f1 != f2).any()                    # 视野内容确实平移


@_needs_yolo
def test_video01_closed_loop_complete():
    """真实图像 + YOLO 检测的闭环避障：COMPLETE、误差达标、命令可审计。"""
    from obstacle_avoidance.reporter import RunReporter, replay, summarize
    prefix = "artifacts/test_video01"
    if os.path.exists(prefix + ".jsonl"):
        os.remove(prefix + ".jsonl")   # RunReporter 为追加模式，先清理
    rep = RunReporter(prefix + ".jsonl")
    world, run = video_sim.build_video_scenario("video01")
    result = run(rep=rep)
    rep.close()
    assert result.final_state.value == "COMPLETE"
    assert result.final_error_px <= 8.0
    assert 1 <= len(result.stage_commands) <= 60
    s = summarize(replay(prefix + ".jsonl"))
    assert s["stage_command_count"] == len(result.stage_commands)


@_needs_yolo
def test_video02_static_obstacle_bypass():
    """video02：用户标注静态障碍横跨直线路径 -> 绕行到达（OA-02 真实图版）。"""
    world, run = video_sim.build_video_scenario("video02")
    assert len(world.static_obstacles) == 1
    result = run()
    assert result.final_state.value == "COMPLETE"
    assert result.plan is not None and result.plan.success
    # 路径必须绕开标注障碍（膨胀后任一 waypoint 不落入障碍圆内）
    ob = world.static_obstacles[0]
    infl = 25.0 + 6.0 + 4.0
    for wp in result.plan.waypoints_px:
        assert ((wp[0] - ob.center[0]) ** 2 + (wp[1] - ob.center[1]) ** 2) \
            > (ob.radius + infl) ** 2 * 0.9
