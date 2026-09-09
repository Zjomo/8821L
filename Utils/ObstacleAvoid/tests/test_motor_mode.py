"""电机模式测试：SerialXYStage 门控/指令/限位 + 失位检测 + CameraWorld。"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from obstacle_avoidance import video_sim
from obstacle_avoidance.roi_zones import RoiConfig, Zone
from obstacle_avoidance.simulator import StageError


def test_virtual_xyz_profile_limits_and_telemetry():
    from obstacle_avoidance.motion import (AxisMotionConfig, MotionConfig,
                                           MotionLimitError, MotionStateError,
                                           VirtualXYZStage)
    stage = VirtualXYZStage(MotionConfig(axes={
        "x": AxisMotionConfig("x", -10, 10, steps_per_unit=100,
                               max_speed=20, acceleration=40),
        "y": AxisMotionConfig("y", -10, 10, steps_per_unit=100,
                               max_speed=20, acceleration=40),
        "z": AxisMotionConfig("z", -2, 2, steps_per_unit=10,
                               max_speed=5, acceleration=10),
    }))
    t = stage.move_by({"x": 2.5, "z": -0.5}, source="test")
    assert t.steps == {"x": 250, "y": 0, "z": -5}
    assert t.position_after == {"x": 2.5, "y": 0.0, "z": -0.5}
    assert t.duration_s > 0 and t.peak_speed["x"] > 0
    with pytest.raises(MotionLimitError):
        stage.move_by({"x": 20})
    assert stage.position["x"] == pytest.approx(2.5)
    stage.stop()
    with pytest.raises(MotionStateError):
        stage.move_by({"x": 1})
    stage.enable()
    stage.home()
    assert stage.position == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_sim_microscope_xyz_z_changes_focus(tmp_path):
    from obstacle_avoidance.sim_microscope import SimMicroscopeWorld
    world = SimMicroscopeWorld(db_path=str(tmp_path / "stage.db"))
    try:
        world.cam.enable()
        world.cam.set_fast_preview(True)
        world.cam.trigger()
        focused = world.cam._fetch_data().copy()
        world.motion_stage.move_by({"z": 20.0}, source="test-z")
        world.cam.trigger()
        defocused = world.cam._fetch_data().copy()
        assert world.motion_stage.position["z"] == pytest.approx(20.0)
        assert float(np.var(defocused)) < float(np.var(focused))
        assert world.motion_stage.last_telemetry.steps["z"] == 20
    finally:
        world.close()


# ---------------- SerialXYStage（无真实硬件）
class FakeSerial:
    def __init__(self, reply=b"ok\n"):
        self.written = b""
        self._reply = reply

    def reset_input_buffer(self):
        pass

    def write(self, data):
        self.written += data
        return len(data)

    def flush(self):
        pass

    def read(self, n):
        r = self._reply
        self._reply = b""
        return r

    def close(self):
        pass


def _make_stage(monkeypatch, reply=b"ok\n", **kw):
    from obstacle_avoidance.stages import SerialXYStage
    fake = FakeSerial(reply=reply)
    monkeypatch.setattr(SerialXYStage, "_open",
                        staticmethod(lambda *a, **k: fake))
    stage = SerialXYStage(port="COM3", confirmed=True, timeout_s=0.2,
                          settle_s=0.0, **kw)
    return stage, fake


def test_serial_stage_requires_confirm():
    from obstacle_avoidance.stages import SerialXYStage
    with pytest.raises(StageError, match="NOT confirmed"):
        SerialXYStage(port="COM3", confirmed=False)


def test_serial_stage_command_format_and_limits(monkeypatch):
    stage, fake = _make_stage(monkeypatch)
    assert stage.move_by(0.3, -0.2, task_id="t", waypoint_index=2)
    assert b"G91 G1 X0.3000 Y-0.2000\n" in fake.written
    assert stage.position_mm == [pytest.approx(0.3), pytest.approx(-0.2)]
    assert stage.last_command.waypoint_index == 2
    # 单步超限
    with pytest.raises(StageError, match="max_step_mm"):
        stage.move_by(2.0, 0.0)
    # 软限位（放宽单步限制，仅触发 workspace 检查）
    stage_w, _ = _make_stage(monkeypatch, max_step_mm=500.0)
    with pytest.raises(StageError, match="workspace"):
        stage_w.move_by(200.0, 0.0)
    # 方向标定
    stage2, fake2 = _make_stage(monkeypatch, axes_sign=(-1.0, 1.0))
    stage2.move_by(0.1, 0.0)
    assert b"X-0.1000" in fake2.written


def test_serial_stage_ack_timeout(monkeypatch):
    stage, _ = _make_stage(monkeypatch, reply=b"")
    with pytest.raises(StageError, match="ack timeout"):
        stage.move_by(0.1, 0.0)


# ---------------- 失位（spot_slip）检测：video03 + 缩水 stage
_needs_yolo = pytest.mark.skipif(
    not (video_sim and __import__("os").path.isfile(video_sim.WEIGHTS)
         and __import__("os").path.isfile(video_sim.VIDEO)),
    reason="需要权重与测试视频")


@_needs_yolo
def test_controller_aborts_on_spot_slip():
    """球只跟随命令的 30%（滑移 70%）-> 记滑移事件并安全停止。"""
    from obstacle_avoidance.models import FailureReason
    from obstacle_avoidance.reporter import RunReporter
    from obstacle_avoidance.simulator import DryRunStage

    cfg = RoiConfig(roi=(60, 300, 520, 420), video=video_sim.VIDEO)
    cfg.zones = [Zone(name="goal_1", kind="goal", rect=(320, 620, 100, 80))]
    cfg.validate(video_sim._video_size())
    world, run = video_sim.build_video_scenario(
        "video03", config=cfg,
        cfg_overrides={"slip_threshold_px": 5.0, "slip_abort_px": 1000.0,
                       "slip_max_events": 2})

    def slippery_factory():
        ppm = world.transform.px_per_mm
        return DryRunStage(lambda dx, dy: world.shift_window(
            dx * ppm * 0.3, dy * ppm * 0.3))

    rep = RunReporter(None)
    result = run(rep=rep, stage_factory=slippery_factory)
    events = rep.events  # path=None 时不落盘，直接用内存事件
    slips = [e for e in events if e.get("event") == "spot_slip"]
    rep.close()
    assert result.final_state.value == "ABORTED"
    assert result.failure_reason == FailureReason.SPOT_SLIP
    assert len(slips) >= 2


@_needs_yolo
def test_controller_no_false_slip_in_normal_run():
    """正常闭环（球完全跟随）不应产生 spot_slip 事件。"""
    from obstacle_avoidance.reporter import RunReporter
    cfg = RoiConfig(roi=(60, 300, 520, 420), video=video_sim.VIDEO)
    cfg.zones = [Zone(name="goal_1", kind="goal", rect=(320, 620, 100, 80))]
    cfg.validate(video_sim._video_size())
    world, run = video_sim.build_video_scenario("video03", config=cfg)
    rep = RunReporter(None)
    result = run(rep=rep)
    slips = [e for e in rep.events if e.get("event") == "spot_slip"]
    rep.close()
    assert result.final_state.value == "COMPLETE"
    assert slips == []


# ---------------- CameraWorld（frame_source 可注入）
def test_camera_world_crop_and_snapshot():
    from obstacle_avoidance.vision import ClassicDetector, VisionPipeline

    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.circle(frame, (350, 250), 22, (255, 255, 255), -1)   # 视频坐标球心
    world = video_sim.CameraWorld(frame_source=lambda: frame,
                                  window=(480, 420), offset=(100, 100),
                                  px_per_mm=100.0)
    img = world.render()
    assert img.shape[:2] == (420, 480)
    # 球在窗口坐标 (250, 150)
    world.bind_pipeline(VisionPipeline(detector=ClassicDetector(),
                                       expected_radius_px=22))
    world.pipeline.process(world.render(), 1)  # 喂帧 -> tracker 更新
    snap = world.snapshot()
    assert len(snap.particles) >= 1
    p = min(snap.particles, key=lambda p: math.dist(p.position_px, (250, 150)))
    assert math.dist(p.position_px, (250, 150)) <= 6


import cv2  # noqa: E402

# ---------------- PicoMotorStage（8742/8743，fake pylablib 设备）
class FakePicoDev:
    """记录 move_by/wait_move/stop 调用，模拟 pylablib Picomotor8742。"""
    def __init__(self, fail=False):
        self.moves = []          # [(axis, steps)]
        self.stops = []
        self.veLOCITY = None
        self.fail = fail

    def move_by(self, axis, steps, **kw):
        if self.fail:
            raise IOError("USB error")
        self.moves.append((axis, int(steps)))

    def wait_move(self, axis, **kw):
        pass

    def stop(self, axis="all", immediate=False, **kw):
        self.stops.append((axis, immediate))

    def setup_velocity(self, axis, speed=None, accel=None, **kw):
        self.veLOCITY = (axis, speed)

    def close(self):
        pass


def _make_pico(monkeypatch, dev=None, **kw):
    from obstacle_avoidance.stages import PicoMotorStage
    dev = dev or FakePicoDev()
    monkeypatch.setattr(PicoMotorStage, "_open",
                        staticmethod(lambda conn, timeout: dev))
    return PicoMotorStage(confirmed=True, **kw), dev


def test_picomotor_requires_confirm():
    from obstacle_avoidance.stages import PicoMotorStage
    with pytest.raises(StageError, match="NOT confirmed"):
        PicoMotorStage(confirmed=False)


def test_picomotor_mm_to_steps_conversion(monkeypatch):
    stage, dev = _make_pico(monkeypatch, steps_per_mm=1000.0,
                            x_axis=1, y_axis=2)
    stage.move_by(0.10, -0.05, task_id="t")
    # dx_mm=+0.10 -> 100 steps axis1; dy_mm=-0.05 -> -50 steps axis2
    assert dev.moves == [(1, 100), (2, -50)]


def test_picomotor_skips_zero_axis(monkeypatch):
    stage, dev = _make_pico(monkeypatch, steps_per_mm=1000.0)
    stage.move_by(0.20, 0.0)
    assert dev.moves == [(1, 200)]     # dy=0 不发指令


def test_picomotor_step_and_workspace_limits(monkeypatch):
    stage, _ = _make_pico(monkeypatch, steps_per_mm=1000.0, max_step_mm=1.0)
    with pytest.raises(StageError, match="max_step_mm"):
        stage.move_by(2.0, 0.0)
    stage_w, _ = _make_pico(monkeypatch, steps_per_mm=1000.0,
                            max_step_mm=500.0, workspace_mm=(1.0, 1.0))
    with pytest.raises(StageError, match="workspace"):
        stage_w.move_by(200.0, 0.0)


def test_picomotor_driver_error_wrapped(monkeypatch):
    stage, _ = _make_pico(monkeypatch, dev=FakePicoDev(fail=True),
                          steps_per_mm=1000.0)
    with pytest.raises(StageError, match="picomotor move failed"):
        stage.move_by(0.10, 0.0)


def test_picomotor_estop_and_axes_sign(monkeypatch):
    stage, dev = _make_pico(monkeypatch, steps_per_mm=1000.0,
                            axes_sign=(-1.0, 1.0))
    stage.move_by(0.10, 0.0)
    assert dev.moves == [(1, -100)]    # 方向标定生效
    stage.stop_all()
    assert dev.stops == [("all", True)]


def test_picomotor_xyz_stage_maps_profiled_axes(monkeypatch):
    """XYZ panel moves use the detected channel mapping and µm calibration."""
    from obstacle_avoidance.stages import PicoMotorStage, PicoMotorXYZStage
    dev = FakePicoDev()
    monkeypatch.setattr(PicoMotorStage, "_open",
                        staticmethod(lambda conn, timeout: dev))
    stage = PicoMotorXYZStage(
        confirmed=True, x_axis=4, y_axis=2, z_axis=1,
        steps_per_mm=1000.0,
        profiles={"x": {"steps_per_unit": 2.0},
                  "z": {"steps_per_unit": 0.5}})
    stage.move_by({"x": 3.0}, source="test")
    stage.move_by({"z": -4.0}, source="test")
    assert dev.moves == [(4, 6), (1, -2)]


def test_kinesis_uses_one_controller_serial(monkeypatch):
    from obstacle_avoidance.stages import KinesisKIM101Stage, KinesisXYZStage

    class Device:
        def __init__(self, serial):
            self.serial = serial
            self.position = 0
            self.moves = []
            self.connected = False

        def Connect(self, serial):
            self.connected = True
        def Disconnect(self):
            self.connected = False
        def IsSettingsInitialized(self): return True
        def WaitForSettingsInitialized(self, _timeout): pass
        def StartPolling(self, _period): pass
        def StopPolling(self): pass
        def EnableDevice(self): pass
        def GetDeviceInfo(self):
            return SimpleNamespace(Description=f"KIM101-{self.serial}")
        def GetPosition(self, _channel): return self.position
        def MoveTo(self, _channel, position, _timeout):
            self.position = int(position)
            self.moves.append(self.position)
        def SetPositionAs(self, _channel, position): self.position = int(position)
        def Stop(self, _channel): pass

    devices = {"K1": Device("K1")}
    channels = SimpleNamespace(Channel1=1)
    api = {
        "manager": SimpleNamespace(
            BuildDeviceList=lambda: None,
            GetDeviceList=lambda: list(devices)),
        "motor": SimpleNamespace(
            CreateKCubeInertialMotor=lambda serial: devices[str(serial)]),
        "channels": channels,
        "settings": SimpleNamespace(),
    }
    monkeypatch.setattr(KinesisKIM101Stage, "_load_api",
                        classmethod(lambda cls: api))
    records = KinesisKIM101Stage.detect_devices()
    assert [record["serial"] for record in records] == ["K1"]
    stage = KinesisXYZStage(
        serial_no="K1",
        profiles={"x": {"steps_per_unit": 2.0}}, confirmed=True)
    stage.move_by({"x": 3.0}, source="test")
    assert devices["K1"].moves == [6]
