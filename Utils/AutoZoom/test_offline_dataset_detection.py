"""
离线数据集检测模块测试。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
from PIL import Image

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from offline_dataset_detection import OfflineDatasetDetector, OfflineDetectionResult
from Focus.config import AutofocusConfig
from autofocus_qt_ui.qt_compat import QApplication
from autofocus_qt_ui.app import AutofocusMainWindow

app = QApplication.instance() or QApplication(sys.argv)



def _create_test_images(folder: Path, names: List[str], size: tuple = (100, 100)) -> List[Path]:
    """生成若干张纯色测试图片。"""
    paths: List[Path] = []
    for i, name in enumerate(names):
        arr = np.full((*size, 3), 50 + i * 40, dtype=np.uint8)
        path = folder / name
        Image.fromarray(arr).save(str(path))
        paths.append(path)
    return paths


def test_list_image_paths() -> None:
    """list_image_paths 应正确列出文件夹中的图片。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        _create_test_images(folder, ["a.jpg", "b.png"])
        (folder / "c.txt").write_text("not an image", encoding="utf-8")

        cfg = AutofocusConfig()
        detector = OfflineDatasetDetector(cfg)
        paths = detector.list_image_paths(folder)

        assert len(paths) == 2, f"应列出 2 张图片，实际 {len(paths)}"
        assert all(p.suffix.lower() in {".jpg", ".png"} for p in paths)
        assert paths[0].name == "a.jpg"
        print("PASS: list_image_paths")


def test_select_reference_default() -> None:
    """未指定基准图时默认选择第一张。"""
    cfg = AutofocusConfig()
    detector = OfflineDatasetDetector(cfg)
    paths = [Path("ref.jpg"), Path("img1.jpg")]
    ref = detector.select_reference(paths)
    assert ref == paths[0], "默认基准图应为第一张"
    print("PASS: select_reference_default")


def test_select_reference_manual() -> None:
    """手动指定基准图时应返回指定图片。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        paths = _create_test_images(folder, ["img0.jpg", "ref.jpg", "img1.jpg"])

        cfg = AutofocusConfig()
        detector = OfflineDatasetDetector(cfg)
        ref = detector.select_reference(paths, reference_path=paths[1])
        assert ref == paths[1], "手动指定基准图应被使用"
        print("PASS: select_reference_manual")


def test_annotate_image() -> None:
    """annotate_image 应在图像正上方添加文字。"""
    cfg = AutofocusConfig()
    detector = OfflineDatasetDetector(cfg)

    image = np.full((100, 200, 3), 128, dtype=np.uint8)
    annotated = detector.annotate_image(image, "FS: 0.95")

    assert annotated.shape == image.shape
    # 顶部区域应被修改（出现非背景色）
    top_region_before = image[:40, :, :]
    top_region_after = annotated[:40, :, :]
    assert not np.array_equal(top_region_before, top_region_after)
    print("PASS: annotate_image")


def test_run_with_folder() -> None:
    """使用文件夹输入时，应输出标注后的图片。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        _create_test_images(input_dir, ["ref.jpg", "img1.jpg", "img2.jpg"])

        cfg = AutofocusConfig(focus_roi=(0, 0, 100, 100))
        detector = OfflineDatasetDetector(cfg)
        output_dir, results = detector.run(
            input_path=input_dir,
            output_dir=Path(tmpdir) / "output",
        )

        assert output_dir.exists()
        # 基准图 + 2 张目标图都应保存
        assert len(results) == 3
        kept = [r for r in results if not r.deleted]
        assert len(kept) == 3
        assert all(r.annotated_path is not None for r in kept)
        print("PASS: run_with_folder")


def test_run_deletes_zero_score() -> None:
    """FocusScore 为 0 的图片不应出现在输出文件夹中。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        paths = _create_test_images(input_dir, ["ref.jpg", "img1.jpg", "img2.jpg"])

        cfg = AutofocusConfig(focus_roi=(0, 0, 100, 100))
        detector = OfflineDatasetDetector(cfg)

        ref_path = paths[0]
        zero_path = paths[1]

        def _patched_compute_score(image_rgb: np.ndarray) -> float:
            # 模拟 img1 为 0 分，其余为 1 分
            return 0.0 if detector._current_path == zero_path else 1.0

        original_read = detector._read_image_rgb

        def _patched_read(path):
            detector._current_path = Path(path)
            return original_read(path)

        with patch.object(detector, "compute_score", _patched_compute_score):
            with patch.object(detector, "_read_image_rgb", _patched_read):
                output_dir, results = detector.run(
                    input_path=input_dir,
                    output_dir=Path(tmpdir) / "output",
                )

        deleted = [r for r in results if r.deleted]
        kept = [r for r in results if not r.deleted]
        assert len(deleted) == 1, f"应删除 1 张 0 分图片，实际 {len(deleted)}"
        assert deleted[0].image_path == zero_path
        assert len(kept) == 2, f"应保留 2 张图片，实际 {len(kept)}"
        assert all(r.annotated_path is not None for r in kept)
        print("PASS: run_deletes_zero_score")


def test_extract_video_frames(tmp_path: Path = None) -> None:
    """从测试视频中按间隔提取帧。"""
    import cv2

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        fps = 10
        size = (100, 100)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, size)
        if not writer.isOpened():
            print("SKIP: test_extract_video_frames (无法创建测试视频)")
            return

        # 写入 5 秒 = 50 帧
        for i in range(50):
            frame = np.full((*size, 3), i % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        cfg = AutofocusConfig()
        detector = OfflineDatasetDetector(cfg)
        frames_dir = Path(tmpdir) / "frames"
        frame_paths = detector.extract_video_frames(
            video_path, frames_dir, interval_seconds=3.0
        )

        # 5 秒视频，每 3 秒一帧，应提取 0s、3s 两帧
        assert len(frame_paths) == 2, f"应提取 2 帧，实际 {len(frame_paths)}"
        assert all(p.exists() for p in frame_paths)
        print("PASS: extract_video_frames")


def test_run_with_video() -> None:
    """使用视频输入时，应提取帧、评分、标注并输出。"""
    import cv2

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        fps = 10
        size = (100, 100)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, size)
        if not writer.isOpened():
            print("SKIP: test_run_with_video (无法创建测试视频)")
            return

        # 写入 4 秒 = 40 帧
        for i in range(40):
            frame = np.full((*size, 3), 100 + i, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        cfg = AutofocusConfig(focus_roi=(0, 0, 100, 100))
        detector = OfflineDatasetDetector(cfg)
        output_dir, results = detector.run(
            input_path=video_path,
            output_dir=Path(tmpdir) / "output",
            interval_seconds=2.0,
        )

        # 4 秒视频，每 2 秒一帧，共 2 帧（0s、2s），其中一张为参考，一张为目标
        assert output_dir.exists()
        assert len(results) == 2
        kept = [r for r in results if not r.deleted]
        assert len(kept) == 2
        print("PASS: run_with_video")


def test_ui_has_offline_detection_controls() -> None:
    """主窗口应包含离线数据集检测控件。"""
    window = AutofocusMainWindow()
    try:
        assert hasattr(window, "offline_group"), "UI 应包含离线检测分组"
        assert hasattr(window, "offline_input_edit"), "UI 应包含离线输入框"
        assert hasattr(window, "offline_output_edit"), "UI 应包含离线输出框"
        assert hasattr(window, "offline_reference_edit"), "UI 应包含离线基准图框"
        assert hasattr(window, "offline_interval_spin"), "UI 应包含离线抽帧间隔"
        assert hasattr(window, "offline_run_btn"), "UI 应包含离线运行按钮"
        assert window.offline_interval_spin.value() == 3.0
        print("PASS: UI has offline detection controls")
    finally:
        window.close()
        QApplication.processEvents()


def main() -> None:
    test_list_image_paths()
    test_select_reference_default()
    test_select_reference_manual()
    test_annotate_image()
    test_run_with_folder()
    test_run_deletes_zero_score()
    test_extract_video_frames()
    test_run_with_video()
    test_ui_has_offline_detection_controls()
    print("\n所有离线数据集检测测试通过!")


if __name__ == "__main__":
    main()
