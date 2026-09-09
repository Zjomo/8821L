"""屏幕区域帧源测试：mss 捕获尺寸 + CameraWorld 集成。"""
import pytest

from obstacle_avoidance import video_sim
from obstacle_avoidance.qt_compat import QtCore

mss = pytest.importorskip("mss", reason="mss 未安装，跳过屏幕捕获测试")


def test_screen_source_frame_size():
    """区域捕获帧尺寸 = (h, w, 3) BGR。"""
    src = video_sim.screen_source(region=(0, 0, 320, 240))
    try:
        frame = src()
        assert frame is not None
        assert frame.shape == (240, 320, 3)
    finally:
        src.close()


def test_screen_source_invalid_region():
    with pytest.raises(RuntimeError):
        video_sim.screen_source(region=(0, 0, 0, 0))


def test_camera_world_with_screen_source():
    """CameraWorld 可直接挂载屏幕帧源（窗口裁剪 / close 透传）。"""
    src = video_sim.screen_source(region=(0, 0, 400, 300))
    try:
        world = video_sim.CameraWorld(frame_source=src,
                                      window=(200, 150),
                                      offset=(50.0, 50.0))
        frame = world.render()
        assert frame is not None
        assert frame.shape[:2] == (150, 200)
        assert callable(getattr(src, "close", None))
    finally:
        src.close()


class _FakeVirtual:
    """Qt 虚拟桌面几何桩（x/y/width/height 接口）。"""

    def __init__(self, x, y, w, h):
        self._x, self._y, self._w, self._h = x, y, w, h

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._w

    def height(self):
        return self._h


MON_1920 = {"left": 0, "top": 0, "width": 1920, "height": 1080}


def test_physical_region_dpi_unaware_scaling():
    """系统缩放 125%（DPI-unaware Qt DPR=1）：按全局逻辑/物理比值换算。"""
    from obstacle_avoidance.app import ScreenRegionOverlay
    rect = QtCore.QRect(100, 100, 400, 300)
    monitors = [dict(MON_1920), dict(MON_1920)]
    virtual = _FakeVirtual(0, 0, 1536, 864)   # 1920/1.25, 1080/1.25
    r = ScreenRegionOverlay._physical_region(rect, None, monitors=monitors,
                                             virtual=virtual)
    assert r == (125, 125, 500, 375)          # 旧 DPR 逻辑会错给 (100,100,400,300)


def test_physical_region_dpi_aware_identity():
    """DPI-aware（逻辑=物理）：区域原样换算。"""
    from obstacle_avoidance.app import ScreenRegionOverlay
    rect = QtCore.QRect(100, 100, 400, 300)
    monitors = [dict(MON_1920), dict(MON_1920)]
    virtual = _FakeVirtual(0, 0, 1920, 1080)
    r = ScreenRegionOverlay._physical_region(rect, None, monitors=monitors,
                                             virtual=virtual)
    assert r == (100, 100, 400, 300)


def test_physical_region_clamped_to_monitor():
    """矩形跨到第二台显示器：裁剪到中心点所在显示器边界内。"""
    from obstacle_avoidance.app import ScreenRegionOverlay
    mon2 = {"left": 1920, "top": 0, "width": 1920, "height": 1080}
    rect = QtCore.QRect(1800, 100, 400, 300)  # 中心 (2000,250) 在 mon2
    monitors = [{"left": 0, "top": 0, "width": 3840, "height": 1080},
                dict(MON_1920), mon2]
    virtual = _FakeVirtual(0, 0, 3840, 1080)
    r = ScreenRegionOverlay._physical_region(rect, None, monitors=monitors,
                                             virtual=virtual)
    assert r[0] >= 1920                        # 起点夹到第二屏内
    assert r[2] <= 1920 and r[3] <= 1080


def test_ensure_live_screen_region_reset(tmp_path):
    """新选屏幕区域后 _ensure_live：重置旧 ROI 为全画幅；同区域复用世界。"""
    qt = pytest.importorskip("PyQt5.QtWidgets")
    from obstacle_avoidance.app import MainWindow
    from obstacle_avoidance.roi_zones import RoiConfig

    app = qt.QApplication.instance() or qt.QApplication([])
    win = MainWindow()
    win.mode_sel.setCurrentIndex(1)    # 电机模式
    win.src_combo.setCurrentIndex(1)   # 帧源=屏幕区域
    win._screen_region = (0, 0, 320, 240)
    win._live_src_region = None
    win._live_world = None
    world = win._ensure_live()
    try:
        assert isinstance(world, video_sim.CameraWorld)
        assert win._cam_frame_size == (320, 240)
        assert win._live_src_region == (0, 0, 320, 240)
        assert win._roi_defaulted                 # 提示重画 ROI
        assert win._roi_cfg.roi == (0, 0, 320, 240)  # 旧配置不沿用
        # 同区域：直接复用，不重建
        assert win._ensure_live() is world
        # 换新区域：重建 + 再次重置 ROI
        win._screen_region = (10, 10, 200, 150)
        win._roi_cfg = RoiConfig(roi=(5, 5, 100, 100))
        win._live_world = None
        world2 = win._ensure_live()
        assert world2 is not world
        assert win._roi_cfg.roi == (0, 0, 200, 150)
        assert win._roi_defaulted
    finally:
        win.on_live_toggled(False)
        world.close()
        win.close()
