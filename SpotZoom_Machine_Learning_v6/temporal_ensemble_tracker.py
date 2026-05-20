"""
Temporal Ensemble Tracker - 时序集成追踪器

Inspired by:
- ByteTrack (ifzhang): Low-confidence detection recovery
- BoT-SORT (mikel-brostrom): Camera motion compensation and tracking
- DeepTrack2 (DeepTrackAI): Particle tracking with temporal consistency

Core Innovation:
- 多帧时序集成: 融合历史检测结果提升鲁棒性
- 低置信度恢复: 恢复被过滤的低分检测
- 运动预测: 基于历史轨迹预测下一帧位置
- 置信度衰减: 历史检测的置信度随时间衰减
- 纯 numpy+cv2 实现，零外部依赖
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TemporalEnsembleConfig:
    """时序集成追踪器配置"""
    # 历史帧窗口大小
    history_length: int = 10
    # 低置信度恢复阈值
    low_conf_recovery_threshold: float = 0.15
    # 置信度衰减因子 (每帧)
    confidence_decay: float = 0.85
    # 运动预测权重
    prediction_weight: float = 0.3
    # 最大预测距离 (像素)
    max_prediction_distance: float = 20.0
    # 速度平滑窗口
    velocity_smooth_window: int = 5
    # 是否启用低置信度恢复
    low_conf_recovery_enabled: bool = True
    # 是否启用运动预测
    motion_prediction_enabled: bool = True
    # 关联距离阈值 (像素)
    association_distance: float = 30.0


@dataclass
class Track:
    """追踪轨迹"""
    positions: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=50))
    confidences: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    velocities: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=10))
    age: int = 0
    missed_frames: int = 0
    id: int = 0

    @property
    def last_position(self) -> Optional[Tuple[float, float]]:
        return self.positions[-1] if self.positions else None

    @property
    def last_confidence(self) -> float:
        return self.confidences[-1] if self.confidences else 0.0

    @property
    def avg_velocity(self) -> Tuple[float, float]:
        if not self.velocities:
            return (0.0, 0.0)
        vx = np.mean([v[0] for v in self.velocities])
        vy = np.mean([v[1] for v in self.velocities])
        return (float(vx), float(vy))


@dataclass
class TemporalEnsembleResult:
    """时序集成追踪结果"""
    center: Optional[Tuple[float, float]]
    confidence: float
    track_age: int
    is_recovered: bool  # 是否为恢复检测
    predicted_position: Optional[Tuple[float, float]]
    velocity: Tuple[float, float]
    ensemble_size: int  # 参与集成的帧数


class TemporalEnsembleTracker:
    """时序集成追踪器

    通过融合多帧检测结果和运动预测，
    提供鲁棒的光斑追踪。

    Inspired by ByteTrack's low-confidence recovery and
    BoT-SORT's motion compensation approach.
    """

    def __init__(self, config: Optional[TemporalEnsembleConfig] = None):
        self.config = config or TemporalEnsembleConfig()
        self._tracks: List[Track] = []
        self._next_id: int = 1
        self._detection_history: Deque[List[Tuple[float, float, float]]] = deque(
            maxlen=self.config.history_length
        )

    def reset(self) -> None:
        """重置追踪器"""
        self._tracks.clear()
        self._next_id = 1
        self._detection_history.clear()

    def _predict_position(self, track: Track) -> Optional[Tuple[float, float]]:
        """基于运动模型预测下一帧位置"""
        if not self.config.motion_prediction_enabled:
            return None

        if len(track.positions) < 2:
            return None

        vx, vy = track.avg_velocity
        last = track.last_position
        if last is None:
            return None

        pred_x = last[0] + vx
        pred_y = last[1] + vy

        # 限制预测距离
        dx = pred_x - last[0]
        dy = pred_y - last[1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist > self.config.max_prediction_distance:
            scale = self.config.max_prediction_distance / dist
            pred_x = last[0] + dx * scale
            pred_y = last[1] + dy * scale

        return (pred_x, pred_y)

    def _associate_detection(
        self, detection: Tuple[float, float, float], tracks: List[Track]
    ) -> Optional[Track]:
        """将检测关联到已有轨迹

        Args:
            detection: (x, y, confidence)
            tracks: 候选轨迹列表

        Returns:
            匹配的轨迹或 None
        """
        dx_det, dy_det, _ = detection

        best_track = None
        best_score = -np.inf

        for track in tracks:
            if not track.positions:
                continue

            pred = self._predict_position(track)
            ref = pred if pred is not None else track.last_position
            if ref is None:
                continue

            # 距离分数
            dist = np.sqrt((dx_det - ref[0]) ** 2 + (dy_det - ref[1]) ** 2)
            if dist > self.config.association_distance:
                continue

            dist_score = 1.0 - dist / self.config.association_distance

            # 时间衰减分数
            time_score = self.config.confidence_decay ** track.missed_frames

            score = dist_score * time_score

            if score > best_score:
                best_score = score
                best_track = track

        return best_track

    def _recover_low_confidence(
        self, frame: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """尝试恢复低置信度检测

        使用历史轨迹预测位置，在预测位置附近搜索。

        Args:
            frame: 当前帧图像

        Returns:
            恢复的检测 (x, y, confidence) 或 None
        """
        if not self.config.low_conf_recovery_enabled:
            return None

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        for track in self._tracks:
            if track.missed_frames > 3:
                continue

            pred = self._predict_position(track)
            if pred is None:
                continue

            px, py = int(pred[0]), int(pred[1])
            h, w = gray.shape

            # 在预测位置附近搜索
            search_radius = 15
            y_min = max(0, py - search_radius)
            y_max = min(h, py + search_radius)
            x_min = max(0, px - search_radius)
            x_max = min(w, px + search_radius)

            region = gray[y_min:y_max, x_min:x_max]
            if region.size == 0:
                continue

            # 寻找最亮点
            _, max_val, _, max_loc = cv2.minMaxLoc(region)
            recovered_x = max_loc[0] + x_min
            recovered_y = max_loc[1] + y_min

            # 检查是否在合理范围内
            dist = np.sqrt((recovered_x - pred[0]) ** 2 + (recovered_y - pred[1]) ** 2)
            if dist < search_radius:
                # 恢复的置信度基于历史置信度和距离
                base_conf = track.last_confidence * (self.config.confidence_decay ** track.missed_frames)
                dist_factor = 1.0 - dist / search_radius
                recovered_conf = base_conf * dist_factor * 0.5

                if recovered_conf > self.config.low_conf_recovery_threshold:
                    return (recovered_x, recovered_y, recovered_conf)

        return None

    def update(
        self,
        detections: List[Tuple[float, float, float]],
        frame: Optional[np.ndarray] = None,
    ) -> TemporalEnsembleResult:
        """更新追踪器

        Args:
            detections: 当前帧检测结果 [(x, y, confidence), ...]
            frame: 当前帧图像 (用于低置信度恢复)

        Returns:
            TemporalEnsembleResult 追踪结果
        """
        self._detection_history.append(detections)

        # 按置信度排序
        sorted_dets = sorted(detections, key=lambda d: d[2], reverse=True)

        matched_tracks = set()
        new_tracks = []

        # 关联检测到轨迹
        for det in sorted_dets:
            track = self._associate_detection(det, self._tracks)
            if track is not None:
                # 更新轨迹
                if track.positions:
                    vx = det[0] - track.last_position[0]
                    vy = det[1] - track.last_position[1]
                    track.velocities.append((vx, vy))

                track.positions.append((det[0], det[1]))
                track.confidences.append(det[2])
                track.missed_frames = 0
                track.age += 1
                matched_tracks.add(track.id)
            else:
                # 创建新轨迹
                new_track = Track(
                    positions=deque([(det[0], det[1])], maxlen=50),
                    confidences=deque([det[2]], maxlen=50),
                    id=self._next_id,
                    age=1,
                )
                self._next_id += 1
                new_tracks.append(new_track)

        # 更新未匹配的轨迹
        for track in self._tracks:
            if track.id not in matched_tracks:
                track.missed_frames += 1
                track.age += 1

        # 添加新轨迹
        self._tracks.extend(new_tracks)

        # 移除过期轨迹
        self._tracks = [t for t in self._tracks if t.missed_frames <= 5]

        # 低置信度恢复
        is_recovered = False
        if not sorted_dets and frame is not None:
            recovery = self._recover_low_confidence(frame)
            if recovery is not None:
                sorted_dets.append(recovery)
                is_recovered = True

        # 选择最佳结果
        if not self._tracks:
            return TemporalEnsembleResult(
                center=None, confidence=0.0, track_age=0,
                is_recovered=False, predicted_position=None,
                velocity=(0.0, 0.0), ensemble_size=0,
            )

        # 选择置信度最高的活跃轨迹
        active_tracks = [t for t in self._tracks if t.missed_frames == 0]
        if not active_tracks:
            active_tracks = self._tracks

        best_track = max(active_tracks, key=lambda t: t.last_confidence * (self.config.confidence_decay ** t.missed_frames))

        # 时序集成: 加权平均历史位置
        if len(best_track.positions) > 1:
            weights = []
            positions = []
            for i, (pos, conf) in enumerate(zip(
                list(best_track.positions),
                list(best_track.confidences)
            )):
                w = conf * (self.config.confidence_decay ** (len(best_track.positions) - 1 - i))
                weights.append(w)
                positions.append(pos)

            total_w = sum(weights)
            if total_w > 0:
                ensemble_x = sum(p[0] * w for p, w in zip(positions, weights)) / total_w
                ensemble_y = sum(p[1] * w for p, w in zip(positions, weights)) / total_w
                ensemble_conf = sum(c * w for c, w in zip(
                    list(best_track.confidences), weights
                )) / total_w
            else:
                ensemble_x, ensemble_y = best_track.last_position or (0, 0)
                ensemble_conf = best_track.last_confidence

            # 混合运动预测
            pred = self._predict_position(best_track)
            if pred is not None:
                pw = self.config.prediction_weight
                ensemble_x = (1 - pw) * ensemble_x + pw * pred[0]
                ensemble_y = (1 - pw) * ensemble_y + pw * pred[1]
        else:
            ensemble_x, ensemble_y = best_track.last_position or (0, 0)
            ensemble_conf = best_track.last_confidence

        return TemporalEnsembleResult(
            center=(ensemble_x, ensemble_y),
            confidence=float(ensemble_conf),
            track_age=best_track.age,
            is_recovered=is_recovered,
            predicted_position=self._predict_position(best_track),
            velocity=best_track.avg_velocity,
            ensemble_size=len(best_track.positions),
        )

    def get_diagnostics(self) -> dict:
        """获取追踪器诊断信息"""
        return {
            "num_tracks": len(self._tracks),
            "active_tracks": sum(1 for t in self._tracks if t.missed_frames == 0),
            "history_length": len(self._detection_history),
            "total_ids_issued": self._next_id - 1,
        }
