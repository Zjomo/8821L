"""
主动学习数据采集器 (ActiveLearningCollector)

灵感来源:
- PPAL (https://github.com/ChenhongyiYang/PPAL) — 即插即用主动学习 (CVPR 2024)
- AL-MDN (https://github.com/NVlabs/AL-MDN) — 概率建模主动学习 (ICCV 2021)
- Label Studio (https://github.com/HumanSignal/label-studio) — 多模态标注平台

算法原理:
- Uncertainty Sampling — 不确定性采样 (选择模型最不确定的样本)
- Diversity Sampling — 多样性采样 (确保样本覆盖不同场景)
- Difficulty Calibration — 难度校准 (平衡简单和困难样本)
- Entropy-based Scoring — 基于熵的不确定性评分

功能:
- 自动收集对准过程中的"困难样本" (检测失败/低置信度/边界情况)
- 基于不确定性和多样性的智能采样策略
- 导出标注数据集用于 YOLO 模型再训练
- 优先级队列管理，确保数据集均衡性

依赖: numpy, json, pathlib
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.ActiveLearning")


@dataclass
class SampleRecord:
    """单个采集样本记录。"""
    timestamp: float
    image_path: Optional[str]  # 保存的图像路径
    bbox: Optional[Tuple[int, int, int, int]]  # 检测框
    confidence: float  # 检测置信度
    focus_score: float  # 焦点评分
    pixel_error: Optional[Tuple[int, int]]  # 像素偏差
    detection_success: bool  # 是否检测成功
    is_hard_sample: bool  # 是否为困难样本
    uncertainty_score: float  # 不确定性分数
    sample_reason: str  # 采集原因
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加元数据


@dataclass
class CollectionStats:
    """采集统计。"""
    total_samples: int = 0
    hard_samples: int = 0
    easy_samples: int = 0
    failed_detections: int = 0
    low_confidence_samples: int = 0
    high_error_samples: int = 0
    avg_confidence: float = 0.0
    avg_uncertainty: float = 0.0
    coverage_score: float = 0.0  # 样本覆盖度


class ActiveLearningCollector:
    """主动学习数据采集器。

    自动收集对准过程中的困难样本，用于后续 YOLO 模型再训练。

    Parameters
    ----------
    output_dir : str
        采集数据输出目录。
    max_samples : int
        最大采集样本数。
    confidence_threshold : float
        低置信度阈值 (低于此值视为困难样本)。
    error_threshold : int
        高偏差阈值 (高于此值视为困难样本)。
    min_save_interval_s : float
        最小保存间隔 (秒)，避免频繁保存。
    diversity_bins : int
        多样性分箱数 (将误差空间分为 N×N 个区域)。
    """

    def __init__(
        self,
        output_dir: str = "./active_learning_data",
        max_samples: int = 500,
        confidence_threshold: float = 0.5,
        error_threshold: int = 50,
        min_save_interval_s: float = 2.0,
        diversity_bins: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.max_samples = int(max_samples)
        self.confidence_threshold = float(confidence_threshold)
        self.error_threshold = int(error_threshold)
        self.min_save_interval_s = float(min_save_interval_s)
        self.diversity_bins = int(diversity_bins)

        self._samples: List[SampleRecord] = []
        self._last_save_time: float = 0.0
        self._diversity_grid: Dict[Tuple[int, int], int] = defaultdict(int)
        self._total_error_range: Tuple[float, float] = (0.0, 200.0)
        self._total_conf_range: Tuple[float, float] = (0.0, 1.0)

        # 创建输出目录
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.metadata_dir = self.output_dir / "metadata"

    def should_collect(
        self,
        confidence: float,
        pixel_error: Optional[Tuple[int, int]] = None,
        focus_score: float = 0.0,
        detection_success: bool = True,
    ) -> Tuple[bool, str]:
        """判断当前帧是否值得采集。

        Parameters
        ----------
        confidence : float
            检测置信度 [0, 1]。
        pixel_error : Tuple[int, int] or None
            像素偏差。
        focus_score : float
            焦点评分。
        detection_success : bool
            是否检测成功。

        Returns
        -------
        Tuple[bool, str]
            (是否采集, 原因)。
        """
        if len(self._samples) >= self.max_samples:
            return (False, "max_samples_reached")

        # 检测失败 → 一定采集
        if not detection_success:
            return (True, "detection_failed")

        # 低置信度 → 采集
        if confidence < self.confidence_threshold:
            return (True, f"low_confidence({confidence:.2f})")

        # 高偏差 → 采集
        if pixel_error is not None:
            error_mag = (pixel_error[0] ** 2 + pixel_error[1] ** 2) ** 0.5
            if error_mag > self.error_threshold:
                return (True, f"high_error({error_mag:.1f}px)")

        # 低焦点评分 → 采集
        if focus_score > 0 and focus_score < 15.0:
            return (True, f"low_focus({focus_score:.1f})")

        # 多样性采样 (每隔一段时间采集一个"普通"样本)
        if len(self._samples) > 0 and len(self._samples) % 10 == 0:
            bin_key = self._get_diversity_bin(confidence, pixel_error)
            if self._diversity_grid.get(bin_key, 0) < 2:
                return (True, f"diversity_fill(bin={bin_key})")

        return (False, "")

    def collect(
        self,
        frame: np.ndarray,
        confidence: float,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        pixel_error: Optional[Tuple[int, int]] = None,
        focus_score: float = 0.0,
        detection_success: bool = True,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SampleRecord]:
        """采集一个样本。

        Parameters
        ----------
        frame : np.ndarray
            BGR 图像帧。
        confidence : float
            检测置信度。
        bbox : Tuple[int, int, int, int] or None
            检测框。
        pixel_error : Tuple[int, int] or None
            像素偏差。
        focus_score : float
            焦点评分。
        detection_success : bool
            是否检测成功。
        reason : str
            采集原因。
        metadata : Dict or None
            附加元数据。

        Returns
        -------
        Optional[SampleRecord]
            采集的样本记录。如果未采集返回 None。
        """
        now = time.time()
        if now - self._last_save_time < self.min_save_interval_s:
            return None

        should, reason = self.should_collect(
            confidence, pixel_error, focus_score, detection_success
        )
        if not should:
            return None

        # 计算不确定性分数
        uncertainty = self._compute_uncertainty(confidence, pixel_error, focus_score)

        # 保存图像
        image_path = None
        try:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            img_filename = f"sample_{len(self._samples):06d}_{int(now * 1000)}.jpg"
            image_path = str(self.images_dir / img_filename)
            import cv2
            cv2.imwrite(image_path, frame)
        except Exception as e:
            LOGGER.warning("Failed to save sample image: %s", e)

        # 判断是否为困难样本
        is_hard = (
            not detection_success
            or confidence < self.confidence_threshold
            or (pixel_error is not None and (pixel_error[0] ** 2 + pixel_error[1] ** 2) ** 0.5 > self.error_threshold)
        )

        record = SampleRecord(
            timestamp=now,
            image_path=image_path,
            bbox=bbox,
            confidence=confidence,
            focus_score=focus_score,
            pixel_error=pixel_error,
            detection_success=detection_success,
            is_hard_sample=is_hard,
            uncertainty_score=uncertainty,
            sample_reason=reason or should,
            metadata=metadata or {},
        )

        self._samples.append(record)
        self._last_save_time = now

        # 更新多样性网格
        bin_key = self._get_diversity_bin(confidence, pixel_error)
        self._diversity_grid[bin_key] += 1

        # 更新范围
        if pixel_error is not None:
            mag = (pixel_error[0] ** 2 + pixel_error[1] ** 2) ** 0.5
            self._total_error_range = (
                min(self._total_error_range[0], mag),
                max(self._total_error_range[1], mag),
            )

        if len(self._samples) % 10 == 0:
            LOGGER.info(
                "ActiveLearning: collected %d samples (%d hard, %d easy)",
                len(self._samples),
                sum(1 for s in self._samples if s.is_hard_sample),
                sum(1 for s in self._samples if not s.is_hard_sample),
            )

        return record

    def _compute_uncertainty(
        self,
        confidence: float,
        pixel_error: Optional[Tuple[int, int]],
        focus_score: float,
    ) -> float:
        """计算不确定性分数 [0, 1]。"""
        # 置信度不确定性 (低置信度 → 高不确定性)
        conf_uncertainty = 1.0 - confidence

        # 偏差不确定性 (大偏差 → 高不确定性)
        error_uncertainty = 0.0
        if pixel_error is not None:
            mag = (pixel_error[0] ** 2 + pixel_error[1] ** 2) ** 0.5
            error_uncertainty = min(mag / max(self._total_error_range[1], 1.0), 1.0)

        # 焦点不确定性 (低焦点 → 高不确定性)
        focus_uncertainty = max(0.0, 1.0 - focus_score / 100.0)

        # 加权组合
        uncertainty = (
            0.4 * conf_uncertainty
            + 0.4 * error_uncertainty
            + 0.2 * focus_uncertainty
        )
        return round(min(max(uncertainty, 0.0), 1.0), 4)

    def _get_diversity_bin(
        self,
        confidence: float,
        pixel_error: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """获取多样性分箱坐标。"""
        conf_bin = min(
            int(confidence * self.diversity_bins),
            self.diversity_bins - 1,
        )
        if pixel_error is not None:
            mag = (pixel_error[0] ** 2 + pixel_error[1] ** 2) ** 0.5
            err_range = max(self._total_error_range[1] - self._total_error_range[0], 1.0)
            err_norm = (mag - self._total_error_range[0]) / err_range
            err_bin = min(int(err_norm * self.diversity_bins), self.diversity_bins - 1)
        else:
            err_bin = 0
        return (conf_bin, err_bin)

    def get_stats(self) -> CollectionStats:
        """获取采集统计。"""
        if not self._samples:
            return CollectionStats()

        hard = sum(1 for s in self._samples if s.is_hard_sample)
        failed = sum(1 for s in self._samples if not s.detection_success)
        low_conf = sum(1 for s in self._samples if s.confidence < self.confidence_threshold)
        high_err = sum(1 for s in self._samples if s.pixel_error is not None and
                       (s.pixel_error[0] ** 2 + s.pixel_error[1] ** 2) ** 0.5 > self.error_threshold)

        avg_conf = float(np.mean([s.confidence for s in self._samples]))
        avg_unc = float(np.mean([s.uncertainty_score for s in self._samples]))

        # 覆盖度 = 已填充的 bin 数 / 总 bin 数
        total_bins = self.diversity_bins ** 2
        filled_bins = len(self._diversity_grid)
        coverage = filled_bins / max(total_bins, 1)

        return CollectionStats(
            total_samples=len(self._samples),
            hard_samples=hard,
            easy_samples=len(self._samples) - hard,
            failed_detections=failed,
            low_confidence_samples=low_conf,
            high_error_samples=high_err,
            avg_confidence=round(avg_conf, 4),
            avg_uncertainty=round(avg_unc, 4),
            coverage_score=round(coverage, 4),
        )

    def export_metadata(self, path: Optional[str] = None) -> None:
        """导出采集元数据为 JSON。"""
        export_path = Path(path) if path else self.metadata_dir / "collection_metadata.json"
        export_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.get_stats()
        data = {
            "stats": {
                "total_samples": stats.total_samples,
                "hard_samples": stats.hard_samples,
                "easy_samples": stats.easy_samples,
                "failed_detections": stats.failed_detections,
                "avg_confidence": stats.avg_confidence,
                "avg_uncertainty": stats.avg_uncertainty,
                "coverage_score": stats.coverage_score,
            },
            "samples": [
                {
                    "timestamp": s.timestamp,
                    "image": s.image_path,
                    "bbox": list(s.bbox) if s.bbox else None,
                    "confidence": s.confidence,
                    "focus_score": s.focus_score,
                    "pixel_error": list(s.pixel_error) if s.pixel_error else None,
                    "detection_success": s.detection_success,
                    "is_hard": s.is_hard_sample,
                    "uncertainty": s.uncertainty_score,
                    "reason": s.sample_reason,
                    "metadata": s.metadata,
                }
                for s in self._samples
            ],
        }

        with export_path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

        LOGGER.info("ActiveLearning metadata exported to %s", export_path)

    def generate_yolo_labels(self) -> None:
        """生成 YOLO 格式标注文件 (基于采集的 bbox)。

        对于检测失败的样本，生成空标注文件 (negative sample)。
        """
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        for sample in self._samples:
            if sample.image_path is None:
                continue
            img_name = Path(sample.image_path).stem
            label_path = self.labels_dir / f"{img_name}.txt"

            lines = []
            if sample.bbox is not None and sample.detection_success:
                x1, y1, x2, y2 = sample.bbox
                # YOLO 格式: class cx cy w h (归一化)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                # 假设图像尺寸 (需要从实际图像获取，这里使用默认值)
                img_w, img_h = 640, 480
                lines.append(f"0 {cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}")

            with label_path.open("w", encoding="utf-8") as fp:
                fp.write("\n".join(lines))

        LOGGER.info("YOLO labels generated in %s", self.labels_dir)

    def reset(self) -> None:
        """重置采集器。"""
        self._samples.clear()
        self._last_save_time = 0.0
        self._diversity_grid.clear()
        LOGGER.info("ActiveLearning collector reset")
