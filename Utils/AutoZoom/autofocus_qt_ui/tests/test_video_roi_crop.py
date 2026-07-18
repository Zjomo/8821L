"""
离线视频 ROI 裁剪模块测试。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from autofocus_qt_ui.video_roi_crop import VideoRoiCropper, _clamp_roi, _guess_output_path

pytestmark = pytest.mark.skipif(cv2 is None, reason="需要 opencv-python")


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """生成一个 64x48、10 帧的合成视频。"""
    path = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    for i in range(10):
        # 每帧绘制不同颜色，便于验证帧顺序
        frame = np.full((48, 64, 3), (i * 20, i * 10, 255 - i * 20), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_clamp_roi_inside_frame() -> None:
    """ROI 完全在帧内时保持不变。"""
    assert _clamp_roi((5, 5, 10, 10), (48, 64, 3)) == (5, 5, 10, 10)


def test_clamp_roi_out_of_bounds() -> None:
    """ROI 越界时应被限制在帧范围内。"""
    assert _clamp_roi((50, 40, 100, 100), (48, 64, 3)) == (50, 40, 14, 8)


def test_clamp_roi_negative_origin() -> None:
    """ROI 起点为负时应被修正为 0。"""
    assert _clamp_roi((-5, -5, 20, 20), (48, 64, 3)) == (0, 0, 20, 20)


def test_guess_output_path() -> None:
    """自动生成输出路径应包含 ROI 信息。"""
    path = _guess_output_path("/tmp/video.mp4", (10, 20, 100, 80))
    assert path.name == "video_roi_10_20_100_80.mp4"


def test_crop_frame(sample_video: Path) -> None:
    """单帧裁剪应返回正确尺寸与像素。"""
    cap = cv2.VideoCapture(str(sample_video))
    ret, frame = cap.read()
    cap.release()
    assert ret

    cropped = VideoRoiCropper.crop_frame(frame, (10, 8, 20, 16))
    assert cropped.shape == (16, 20, 3)
    # 左上角像素应与原图对应位置一致
    np.testing.assert_array_equal(cropped[0, 0], frame[8, 10])


def test_get_video_info(sample_video: Path) -> None:
    """视频信息读取应正确。"""
    cropper = VideoRoiCropper()
    info = cropper.get_video_info(sample_video)
    assert info["fps"] == pytest.approx(10.0, abs=0.1)
    assert info["frame_count"] == 10
    assert info["width"] == 64
    assert info["height"] == 48


def test_crop_video(sample_video: Path, tmp_path: Path) -> None:
    """完整裁剪流程应输出正确尺寸与帧数的视频。"""
    cropper = VideoRoiCropper()
    output = cropper.crop_video(
        input_path=sample_video,
        output_path=tmp_path / "cropped.mp4",
        roi=(10, 8, 20, 16),
    )

    assert output.exists()

    cap = cv2.VideoCapture(str(output))
    assert cap.isOpened()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert width == 20
    assert height == 16
    assert frame_count == 10

    # 读取首帧验证像素（允许编码带来的微小误差）
    ret, frame = cap.read()
    cap.release()
    assert ret
    np.testing.assert_allclose(
        frame[0, 0], np.array([0, 0, 255], dtype=np.uint8), atol=10
    )


def test_crop_video_invalid_roi(sample_video: Path) -> None:
    """无效 ROI（宽/高为 0）应抛出异常。"""
    cropper = VideoRoiCropper()
    with pytest.raises(ValueError):
        cropper.crop_video(sample_video, roi=(0, 0, 0, 0))


def test_crop_video_progress_callback(sample_video: Path, tmp_path: Path) -> None:
    """进度回调应按顺序递增。"""
    progress_values: list[tuple[int, int]] = []

    def on_progress(current: int, total: int) -> None:
        progress_values.append((current, total))

    cropper = VideoRoiCropper(on_progress=on_progress)
    cropper.crop_video(
        input_path=sample_video,
        output_path=tmp_path / "progress.avi",
        roi=(0, 0, 32, 24),
    )

    assert len(progress_values) == 10
    currents = [p[0] for p in progress_values]
    totals = [p[1] for p in progress_values]
    assert currents == list(range(1, 11))
    assert all(t == 10 for t in totals)


class TestVideoRoiCropWorker:
    """需要 Qt 的 Worker 测试。"""

    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def qt_app(cls):
        """确保 QCoreApplication 已创建。"""
        from autofocus_qt_ui.qt_compat import QCoreApplication

        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])
        yield app

    def test_worker_emits_progress_and_finished(self, sample_video: Path, tmp_path: Path) -> None:
        """Worker 应正确发射进度与完成信号。"""
        from autofocus_qt_ui.video_roi_crop_worker import VideoRoiCropWorker

        worker = VideoRoiCropWorker()
        worker.configure(
            input_path=sample_video,
            output_path=tmp_path / "worker_out.avi",
            roi=(0, 0, 32, 24),
        )

        logs: list[str] = []
        progress_values: list[tuple[int, int]] = []
        finished_result: tuple[bool, str] | None = None

        worker.log.connect(lambda msg: logs.append(msg))
        worker.progress.connect(lambda c, t: progress_values.append((c, t)))

        def on_finished(ok: bool, message: str) -> None:
            nonlocal finished_result
            finished_result = (ok, message)

        worker.finished.connect(on_finished)
        worker.run()

        assert finished_result is not None
        assert finished_result[0] is True
        assert Path(finished_result[1]).exists()
        assert len(progress_values) == 10
        assert progress_values[-1] == (10, 10)
        assert any("完成" in msg for msg in logs)

    def test_worker_invalid_input(self) -> None:
        """Worker 对无效输入应发射失败信号。"""
        from autofocus_qt_ui.video_roi_crop_worker import VideoRoiCropWorker

        worker = VideoRoiCropWorker()
        worker.configure(input_path="/non/existent/video.mp4", roi=(0, 0, 10, 10))

        finished_result: tuple[bool, str] | None = None

        def on_finished(ok: bool, message: str) -> None:
            nonlocal finished_result
            finished_result = (ok, message)

        worker.finished.connect(on_finished)
        worker.run()

        assert finished_result is not None
        assert finished_result[0] is False
