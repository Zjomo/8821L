"""
离线数据集检测模块。

功能：
  1. 输入为视频时，按固定间隔（默认每 3 秒）提取帧并保存为图片；
  2. 输入为文件夹时，直接读取其中的图片；
  3. 手动选择基准图（未选择则默认使用第一张图片）建立 FocusScore 参考；
  4. 对后续图片计算 FocusScore_ratio，并将得分标注在图像正上方；
  5. FocusScore 为 0 的图片不进入输出文件夹（从中间结果中删除）；
  6. 输出到新建的文件夹，内部仅保留标注后的非零分图片。

使用示例：
    from offline_dataset_detection import OfflineDatasetDetector
    from Focus.config import AutofocusConfig

    cfg = AutofocusConfig(focus_roi=(0, 0, 300, 300))
    detector = OfflineDatasetDetector(cfg)
    detector.run(
        input_path=r"path/to/video.mp4",
        output_dir=r"path/to/output",
        reference_path=None,  # 默认使用第一张图作为基准
    )
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    cv2 = None

AUTOZOOM_ROOT = Path(__file__).resolve().parent

from Focus.config import AutofocusConfig
from Focus.metrics import FocusMetricsCalculator
from Focus.scorer import FocusScorer

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


@dataclass
class OfflineDetectionResult:
    """离线数据集检测单张图片的处理结果。"""

    image_path: Path
    score: Optional[float]
    annotated_path: Optional[Path]
    deleted: bool


class OfflineDatasetDetector:
    """离线数据集检测器：提取帧/读取图片 → 建立参考 → 评分 → 标注 → 输出。"""

    def __init__(
        self,
        cfg: Optional[AutofocusConfig] = None,
        scorer: Optional[FocusScorer] = None,
        metrics_calc: Optional[FocusMetricsCalculator] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        初始化检测器。

        参数
        ----------
        cfg : AutofocusConfig, optional
            聚焦配置，默认使用 Focus 模块默认配置。
        scorer : FocusScorer, optional
            外部传入的评分器；为 None 时根据 cfg 新建。
        metrics_calc : FocusMetricsCalculator, optional
            外部传入的指标计算器；为 None 时根据 cfg 新建。
        on_log : callable, optional
            日志回调函数。
        """
        self.cfg = cfg or AutofocusConfig()
        self.metrics_calc = metrics_calc or FocusMetricsCalculator(self.cfg)
        self.scorer = scorer or FocusScorer(self.cfg, self.metrics_calc)
        self.on_log = on_log or logger.info

    # ------------------------------------------------------------------
    # 输入解析
    # ------------------------------------------------------------------

    @staticmethod
    def is_video(path: Union[str, Path]) -> bool:
        """判断路径是否为支持的视频文件。"""
        return Path(path).suffix.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def is_image_folder(path: Union[str, Path]) -> bool:
        """判断路径是否为包含图片的文件夹。"""
        p = Path(path)
        if not p.is_dir():
            return False
        return any(
            child.suffix.lower() in IMAGE_EXTENSIONS
            for child in p.iterdir()
            if child.is_file()
        )

    def extract_video_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        interval_seconds: float = 3.0,
    ) -> List[Path]:
        """
        从视频中按间隔提取帧并保存为图片。

        参数
        ----------
        video_path : str | Path
            视频文件路径。
        output_dir : str | Path
            帧输出目录。
        interval_seconds : float, default 3.0
            提取间隔（秒）。

        返回
        -------
        List[Path]
            提取出的帧文件路径列表，按文件名排序。
        """
        if cv2 is None:
            raise ImportError("需要安装 opencv-python：pip install opencv-python")

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(fps * interval_seconds))
        frame_index = 0
        saved_paths: List[Path] = []

        self.on_log(
            f"[离线检测] 开始提取视频帧：{video_path.name}, "
            f"fps={fps:.2f}, 间隔={interval_seconds}s"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                time_stamp = frame_index / fps
                save_path = output_dir / f"frame_{time_stamp:08.3f}s.jpg"
                cv2.imwrite(str(save_path), frame)
                saved_paths.append(save_path)

            frame_index += 1

        cap.release()
        saved_paths.sort()
        self.on_log(f"[离线检测] 共提取 {len(saved_paths)} 帧到 {output_dir}")
        return saved_paths

    def list_image_paths(self, folder_path: Union[str, Path]) -> List[Path]:
        """
        列出文件夹内所有支持的图片路径，并按文件名排序。

        参数
        ----------
        folder_path : str | Path
            图片文件夹路径。

        返回
        -------
        List[Path]
            图片路径列表。
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"文件夹不存在：{folder}")

        paths = [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        paths.sort(key=lambda p: p.name)
        self.on_log(f"[离线检测] 从文件夹读取到 {len(paths)} 张图片：{folder}")
        return paths

    # ------------------------------------------------------------------
    # 参考图与 FocusScore
    # ------------------------------------------------------------------

    def select_reference(
        self,
        image_paths: List[Path],
        reference_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        选择基准图片。

        参数
        ----------
        image_paths : List[Path]
            候选图片路径列表。
        reference_path : str | Path, optional
            手动指定的基准图路径。为 None 时默认使用第一张图片。

        返回
        -------
        Path
            选定的基准图路径。
        """
        if not image_paths:
            raise ValueError("没有可用的图片来选择基准图")

        if reference_path is not None:
            ref = Path(reference_path)
            if not ref.is_file():
                raise ValueError(f"指定的基准图不存在：{ref}")
            self.on_log(f"[离线检测] 使用手动指定的基准图：{ref.name}")
            return ref

        self.on_log(f"[离线检测] 未指定基准图，默认使用第一张：{image_paths[0].name}")
        return image_paths[0]

    def build_reference(self, reference_path: Union[str, Path]) -> None:
        """
        以选定的基准图建立 FocusScore 参考。

        参数
        ----------
        reference_path : str | Path
            基准图路径。
        """
        image_rgb = self._read_image_rgb(reference_path)
        self.scorer.build_reference_from_image(
            image_rgb=image_rgb, output_root=None, on_log=self.on_log
        )
        self.on_log(f"[离线检测] 基准图建立完成：{Path(reference_path).name}")

    def compute_score(self, image_rgb: np.ndarray) -> Optional[float]:
        """
        计算单张 RGB 图像的 FocusScore_ratio。

        参数
        ----------
        image_rgb : np.ndarray
            RGB 图像数组。

        返回
        -------
        float | None
            FocusScore_ratio；参考未就绪或计算失败时返回 None。
        """
        if not self.scorer.focus_reference_ready:
            raise RuntimeError("尚未建立 FocusScore 参考，请先调用 build_reference()")

        roi = self.metrics_calc.clamp_roi(tuple(self.cfg.focus_roi), image_rgb.shape)
        x, y, rw, rh = roi
        roi_rgb = image_rgb[y : y + rh, x : x + rw].copy()
        roi_metrics = self.metrics_calc.compute_for_image(roi_rgb)
        score, _ = self.scorer.score_ratio(roi_metrics)
        return score

    # ------------------------------------------------------------------
    # 图像标注
    # ------------------------------------------------------------------

    @staticmethod
    def annotate_image(
        image_rgb: np.ndarray,
        text: str,
        font_size: int = 28,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        margin: int = 8,
    ) -> np.ndarray:
        """
        在图像正上方标注文字。

        参数
        ----------
        image_rgb : np.ndarray
            RGB 图像。
        text : str
            要标注的文字。
        font_size : int, default 28
            字体大小。
        text_color : tuple, default (255, 255, 255)
            文字颜色（RGB）。
        bg_color : tuple, default (0, 0, 0)
            文字背景色（RGB）。
        margin : int, default 8
            文字与背景边缘的间距。

        返回
        -------
        np.ndarray
            标注后的 RGB 图像。
        """
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype("msyh.ttc", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        img_w, img_h = pil_image.size
        x = (img_w - text_w) // 2
        y = margin

        draw.rectangle(
            [x - margin, y - margin, x + text_w + margin, y + text_h + margin],
            fill=bg_color,
        )
        draw.text((x, y), text, fill=text_color, font=font)

        return np.array(pil_image)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        reference_path: Optional[Union[str, Path]] = None,
        interval_seconds: float = 3.0,
        delete_zero_score: bool = True,
    ) -> Tuple[Path, List[OfflineDetectionResult]]:
        """
        执行离线数据集检测完整流程。

        参数
        ----------
        input_path : str | Path
            输入视频文件或图片文件夹路径。
        output_dir : str | Path, optional
            输出目录。为 None 时自动在输入路径旁创建。
        reference_path : str | Path, optional
            手动指定的基准图路径。为 None 时默认使用第一张图片。
        interval_seconds : float, default 3.0
            视频帧提取间隔（秒）。
        delete_zero_score : bool, default True
            是否删除 FocusScore 为 0 的图片。

        返回
        -------
        Tuple[Path, List[OfflineDetectionResult]]
            输出目录路径与每张图片的处理结果列表。
        """
        input_path = Path(input_path)

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = input_path.parent / f"{input_path.stem}_annotated_{timestamp}"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 准备图片列表
        if self.is_video(input_path):
            work_dir = output_dir / ".work" / "frames"
            image_paths = self.extract_video_frames(
                input_path, work_dir, interval_seconds
            )
        elif self.is_image_folder(input_path):
            image_paths = self.list_image_paths(input_path)
        else:
            raise ValueError(
                f"输入既不是支持的视频文件，也不是包含图片的文件夹：{input_path}"
            )

        if not image_paths:
            raise ValueError("未找到任何可处理的图片")

        # 选择基准图并建立参考
        ref_path = self.select_reference(image_paths, reference_path)
        self.build_reference(ref_path)

        results: List[OfflineDetectionResult] = []
        for img_path in image_paths:
            if img_path == ref_path:
                # 基准图本身也计算并标注，方便对比
                label = "reference"
            else:
                label = "target"

            try:
                image_rgb = self._read_image_rgb(img_path)
                score = self.compute_score(image_rgb)
            except Exception as exc:
                self.on_log(f"[离线检测] 处理 {img_path.name} 失败：{exc}")
                results.append(
                    OfflineDetectionResult(
                        image_path=img_path,
                        score=None,
                        annotated_path=None,
                        deleted=False,
                    )
                )
                continue

            score_str = f"{score:.4f}" if score is not None else "None"
            text = f"FocusScore: {score_str}"
            annotated = self.annotate_image(image_rgb, text)

            if score == 0.0 and delete_zero_score:
                self.on_log(
                    f"[离线检测] {img_path.name} 得分为 0，已删除"
                )
                if img_path.parent == (output_dir / ".work" / "frames"):
                    img_path.unlink(missing_ok=True)
                results.append(
                    OfflineDetectionResult(
                        image_path=img_path,
                        score=score,
                        annotated_path=None,
                        deleted=True,
                    )
                )
                continue

            out_path = output_dir / f"{img_path.stem}_score{score_str}{img_path.suffix}"
            Image.fromarray(annotated).save(str(out_path))
            results.append(
                OfflineDetectionResult(
                    image_path=img_path,
                    score=score,
                    annotated_path=out_path,
                    deleted=False,
                )
            )
            self.on_log(
                f"[离线检测] {label:8s} {img_path.name}: FocusScore={score_str}"
            )

        # 清理临时帧目录（若从视频提取）
        work_dir = output_dir / ".work"
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

        self.on_log(f"[离线检测] 完成，输出目录：{output_dir}")
        return output_dir, results

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _read_image_rgb(path: Union[str, Path]) -> np.ndarray:
        """读取图片并返回 RGB 数组。"""
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"图片不存在：{path}")

        pil_image = Image.open(path).convert("RGB")
        return np.array(pil_image)


def main() -> None:
    """命令行入口示例。"""
    import argparse

    parser = argparse.ArgumentParser(description="离线数据集 FocusScore 检测")
    parser.add_argument("input", help="输入视频文件或图片文件夹路径")
    parser.add_argument("--output", "-o", help="输出目录路径")
    parser.add_argument("--reference", "-r", help="手动指定的基准图路径")
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=3.0,
        help="视频帧提取间隔（秒），默认 3",
    )
    parser.add_argument(
        "--roi",
        help="聚焦 ROI，格式 x,y,w,h，默认使用配置文件中的 focus_roi",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cfg = AutofocusConfig()
    if args.roi:
        parts = [int(p.strip()) for p in args.roi.split(",")]
        if len(parts) != 4:
            raise ValueError("ROI 格式应为 x,y,w,h")
        cfg.focus_roi = tuple(parts)

    detector = OfflineDatasetDetector(cfg)
    output_dir, results = detector.run(
        input_path=args.input,
        output_dir=args.output,
        reference_path=args.reference,
        interval_seconds=args.interval,
    )

    kept = sum(1 for r in results if not r.deleted)
    deleted = sum(1 for r in results if r.deleted)
    print(f"\n输出目录：{output_dir}")
    print(f"总计处理：{len(results)} 张，保留 {kept} 张，删除 {deleted} 张")


if __name__ == "__main__":
    main()
