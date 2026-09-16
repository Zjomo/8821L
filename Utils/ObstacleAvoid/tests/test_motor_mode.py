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


# ---------------- 『衬底区域』接入电机场景/识别
def test_motor_scenario_uses_manual_substrate_zone(monkeypatch):
    """电机模式：手动『衬底区域』= 可行域，并作为识别的衬底（停用自动衬底）。"""
    cfg = RoiConfig(
        roi=(0, 0, 320, 240),
        zones=[Zone(name="goal_1", kind="goal", rect=(240, 180, 40, 40)),
               Zone(name="substrate_1", kind="substrate",
                    rect=(20, 20, 240, 180))])
    monkeypatch.setattr(video_sim, "get_shared_detector", lambda w: object())
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    world, _run = video_sim.build_video_scenario(
        "video03", config=cfg,
        motor={"frame_source": lambda: frame, "driver": "picomotor",
               "confirmed": True},
        task_mode="oa")
    assert world.substrate_manual
    assert [tuple(p) for p in world.substrate.polygon] == \
        [(20.0, 20.0), (260.0, 20.0), (260.0, 200.0), (20.0, 200.0)]
    snap = world.snapshot()
    assert snap.substrate.contains((100.0, 100.0))
    assert not snap.substrate.contains((300.0, 230.0))
    # 未标注衬底时保持默认（ROI 内缩 10px）
    cfg.zones = [z for z in cfg.zones if z.kind != "substrate"]
    plain, _run = video_sim.build_video_scenario(
        "video03", config=cfg,
        motor={"frame_source": lambda: frame, "driver": "picomotor",
               "confirmed": True},
        task_mode="oa")
    assert not plain.substrate_manual
    assert plain.substrate.contains((12.0, 12.0))


def test_auto_recognition_pipeline_passes_manual_substrate():
    from obstacle_avoidance.auto_recognition import (AutoRecognitionConfig,
                                                     AutoRecognitionPipeline)
    polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    pipe = AutoRecognitionPipeline(
        particle_detector=object(), manual_substrate_polygon=polygon,
        config=AutoRecognitionConfig(auto_substrate=False))
    assert pipe.recognizer.manual_substrate_polygon == polygon
    assert pipe.recognizer.config.auto_substrate is False


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


class _FakeKinesisSettings:
    """Kinesis 设置树替身：记录 StepRate / StepAcceleration 写入。"""

    def __init__(self):
        class Channel:
            def __init__(self):
                self.StepRate = 0
                self.StepAcceleration = 0

        class Drive:
            def __init__(self):
                self._channel = Channel()

            def Channel(self, _channel):
                return self._channel

        self.Drive = Drive()

    def snapshot(self):
        channel = self.Drive.Channel(1)
        return (channel.StepRate, channel.StepAcceleration)


def _fake_kinesis(monkeypatch):
    """搭一个假 KIM101：返回 (设备, 设置树, 工厂)。"""
    from obstacle_avoidance.stages import KinesisKIM101Stage

    settings = _FakeKinesisSettings()
    device = SimpleNamespace(
        moves=[],
        applied=[],
        Connect=lambda serial: None,
        Disconnect=lambda: None,
        IsSettingsInitialized=lambda: True,
        WaitForSettingsInitialized=lambda timeout: None,
        StartPolling=lambda period: None,
        StopPolling=lambda: None,
        EnableDevice=lambda: None,
        GetPosition=lambda channel: 0,
        MoveTo=lambda channel, position, timeout: device.moves.append(
            (channel, position)),
        SetPositionAs=lambda channel, position: None,
        Stop=lambda channel: None,
        GetInertialMotorConfiguration=lambda serial: object())

    def _set_settings(tree, persist, reload_):
        device.applied.append(settings.snapshot())
        assert (persist, reload_) == (True, True)

    device.SetSettings = _set_settings
    api = {
        "manager": SimpleNamespace(BuildDeviceList=lambda: None,
                                   GetDeviceList=lambda: []),
        "motor": SimpleNamespace(
            CreateKCubeInertialMotor=lambda serial: device),
        "channels": SimpleNamespace(Channel1=1),
        "settings": SimpleNamespace(
            GetSettings=lambda config: settings),
    }
    monkeypatch.setattr(KinesisKIM101Stage, "_load_api",
                        classmethod(lambda cls: api))
    return device, settings


def test_kinesis_downlinks_step_rate_and_acceleration(monkeypatch):
    """「位移速度/位移加速度」必须真正写进 KIM101（StepRate/StepAcceleration）。"""
    from obstacle_avoidance.stages import KinesisKIM101Stage

    device, settings = _fake_kinesis(monkeypatch)
    stage = KinesisKIM101Stage(serial_no="K1", confirmed=True,
                               steps_per_mm=1000.0, max_step_mm=0.30,
                               speed_steps=200, accel_steps=2000)
    assert device.applied == [(200, 2000)]
    # 运行期改速度/加速度：None 表示保持当前值
    assert stage.apply_motion_profile(speed_steps=500) is True
    assert device.applied[-1] == (500, 2000)
    assert settings.Drive.Channel(1).StepRate == 500


def test_kinesis_motion_profile_untouched_by_default(monkeypatch):
    """两项都留 0（不改）时不得写设置——避免覆盖控制器出厂参数。"""
    from obstacle_avoidance.stages import KinesisKIM101Stage

    device, _ = _fake_kinesis(monkeypatch)
    stage = KinesisKIM101Stage(serial_no="K1", confirmed=True,
                               steps_per_mm=1000.0)
    assert device.applied == []
    assert stage.apply_motion_profile() is False


def test_kinesis_single_step_limit_matches_max_step_mm(monkeypatch):
    """单步位移(step)→mm 的限位由驱动兜底：超限直接拒绝，不静默走大步。"""
    from obstacle_avoidance.stages import KinesisKIM101Stage

    device, _ = _fake_kinesis(monkeypatch)
    stage = KinesisKIM101Stage(serial_no="K1", confirmed=True,
                               steps_per_mm=1000.0, max_step_mm=0.005)
    stage.move_by(0.005, 0.0)          # 5 step，正好到限
    assert device.moves == [(1, 5)]
    with pytest.raises(StageError, match="max_step_mm"):
        stage.move_by(0.006, 0.0)      # 6 step，超限


# ---------------- KIM101 通道自动映射（按序列号连接后探测）
class _FakeChannelSettings:
    def __init__(self):
        self.StepRate = 0
        self.StepAcceleration = 0


class _FakeMultiChannelSettings:
    """多通道设置树替身：按通道号记录 StepRate / StepAcceleration。"""

    def __init__(self):
        self.channels = {}

        class _Drive:
            def __init__(self, outer):
                self._outer = outer

            def Channel(self, number):
                return self._outer.channel(int(number))

        self.Drive = _Drive(self)

    def channel(self, number):
        return self.channels.setdefault(int(number), _FakeChannelSettings())


def _fake_multichannel_kinesis(monkeypatch, available=(1, 2, 3)):
    """假 4 通道 KIM101：只有 ``available`` 里的通道能读到位置。"""
    from obstacle_avoidance.stages import KinesisKIM101Stage

    settings = _FakeMultiChannelSettings()
    device = SimpleNamespace(
        moves=[], zeroed=[], stopped=[], applied=[],
        Connect=lambda serial: None,
        Disconnect=lambda: None,
        IsSettingsInitialized=lambda: True,
        WaitForSettingsInitialized=lambda timeout: None,
        StartPolling=lambda period: None,
        StopPolling=lambda: None,
        EnableDevice=lambda: None,
        MoveTo=lambda channel, position, timeout: device.moves.append(
            (int(channel), int(position))),
        SetPositionAs=lambda channel, position: device.zeroed.append(
            int(channel)),
        Stop=lambda channel: device.stopped.append(int(channel)),
        GetInertialMotorConfiguration=lambda serial: object())

    def _position(channel):
        if int(channel) not in available:
            raise RuntimeError(f"channel {int(channel)} has no motor")
        return 0

    device.GetPosition = _position

    def _set_settings(tree, persist, reload_):
        device.applied.append((persist, reload_))

    device.SetSettings = _set_settings
    api = {
        "manager": SimpleNamespace(BuildDeviceList=lambda: None,
                                   GetDeviceList=lambda: []),
        "motor": SimpleNamespace(
            CreateKCubeInertialMotor=lambda serial: device),
        "channels": SimpleNamespace(Channel1=1, Channel2=2,
                                    Channel3=3, Channel4=4),
        "settings": SimpleNamespace(GetSettings=lambda config: settings),
    }
    monkeypatch.setattr(KinesisKIM101Stage, "_load_api",
                        classmethod(lambda cls: api))
    return device, settings


def test_kinesis_auto_maps_xyz_to_controller_channels(monkeypatch):
    """按序列号连接后自动把 X/Y/Z 映射到控制器的 3 个可用通道。"""
    from obstacle_avoidance.stages import KinesisKIM101Stage

    device, _ = _fake_multichannel_kinesis(monkeypatch)
    stage = KinesisKIM101Stage(serial_no="K1", confirmed=True,
                               steps_per_mm=1000.0, max_step_mm=0.30)
    assert stage.single_axis is False
    assert stage.axis_channels() == {"x": 1, "y": 2, "z": 3}
    assert stage.channel_map_desc() == "X=Channel1 Y=Channel2 Z=Channel3"
    stage.move_by(0.01, 0.02)                      # 10 step / 20 step
    assert device.moves == [(1, 10), (2, 20)]      # 两个轴打到不同通道
    stage.set_zero("y")
    assert device.zeroed == [2]
    stage.stop_all()
    assert device.stopped == [1, 2, 3]


def test_kinesis_motion_profile_written_to_every_mapped_channel(monkeypatch):
    """速度/加速度必须写进映射到的每个通道，而不是只写 Channel1。"""
    from obstacle_avoidance.stages import KinesisKIM101Stage

    device, settings = _fake_multichannel_kinesis(monkeypatch)
    KinesisKIM101Stage(serial_no="K1", confirmed=True, steps_per_mm=1000.0,
                       speed_steps=200, accel_steps=2000)
    assert [(settings.channel(n).StepRate, settings.channel(n).StepAcceleration)
            for n in (1, 2, 3)] == [(200, 2000)] * 3
    assert settings.channel(4).StepRate == 0        # 未映射的通道不动
    assert device.applied == [(True, True)]         # 只下发一次设置


def test_kinesis_degrades_to_single_axis_when_one_channel(monkeypatch):
    """只探测到一个通道时退化为单轴，而不是报错或映射到空通道。"""
    from obstacle_avoidance.stages import KinesisKIM101Stage

    device, _ = _fake_multichannel_kinesis(monkeypatch, available=(1,))
    stage = KinesisKIM101Stage(serial_no="K1", confirmed=True,
                               steps_per_mm=1000.0)
    assert stage.single_axis is True
    assert stage.axis_channels() == {"x": 1, "y": 1, "z": 1}
    assert stage.channel_map_desc() == "Channel1（单轴）"
    stage.move_by(0.01, 0.02)
    assert device.moves == [(1, 10), (1, 20)]
    stage.stop_all()
    assert device.stopped == [1]
