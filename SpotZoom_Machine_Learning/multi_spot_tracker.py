"""
多光斑关联追踪器 (MultiSpotTracker)

灵感来源:
- BoxMOT/ByteTrack — 多目标追踪框架 (基于检测的多目标追踪)
- Hungarian Algorithm — 匈牙利算法 (Kuhn-Munkres, 二部图最优匹配)
- SORT (Simple Online and Realtime Tracking) — 简单在线实时追踪
- DeepSORT 关联机制 — 基于距离的级联匹配

算法原理:
- Nearest-Neighbor Association — 最近邻关联 (欧氏距离)
- Hungarian Algorithm (简化版) — 贪心匹配实现
- Track Lifecycle Management — 轨迹生命周期管理 (创建/更新/删除)
- Inter-Spot Distance Matrix — 光斑间距离矩阵 (多光束对准)

功能:
- 跨帧追踪多个光斑，分配唯一 track_id
- 处理光斑的出现与消失 (新轨迹创建 / 旧轨迹删除)
- 计算光斑间距离用于多光束对准分析
- 估计每个光斑的运动速度

依赖: numpy, logging
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.MultiSpotTracker")


@dataclass
class TrackedSpot:
    """被追踪的光斑对象。"""
    track_id: int                       # 唯一轨迹 ID
    center: Tuple[float, float]         # 当前中心坐标 (x, y)
    velocity: Tuple[float, float]       # 估计速度 (vx, vy) 像素/帧
    confidence: float                   # 关联置信度 [0, 1]
    age: int                            # 轨迹存在帧数
    last_seen: int                      # 最后一次被关联的帧号
    hit_count: int = 0                  # 总关联命中次数
    position_history: List[Tuple[float, float]] = field(
        default_factory=list,
    )  # 位置历史


@dataclass
class TrackReport:
    """多光斑追踪报告 (每帧输出)。"""
    active_tracks: List[TrackedSpot]    # 当前活跃轨迹
    new_tracks: int                     # 本帧新增轨迹数
    lost_tracks: int                    # 本帧丢失轨迹数
    matched_count: int                  # 本帧成功关联数
    unmatched_detections: int           # 未关联的检测数
    inter_spot_distances: Dict[Tuple[int, int], float]  # 光斑间距离矩阵
    frame_number: int                   # 当前帧号


class MultiSpotTracker:
    """多光斑关联追踪器。

    使用最近邻距离关联 + 贪心匹配实现多光斑跨帧追踪。
    每帧输入检测结果列表，输出追踪报告。

    Parameters
    ----------
    max_distance : float
        最大关联距离 (像素)。超过此距离的检测-轨迹对不进行关联。
    max_age : int
        轨迹最大丢失帧数。超过后删除轨迹。
    min_hits : int
        轨迹确认所需的最小连续命中次数。
    max_history_length : int
        每条轨迹保存的最大位置历史长度。
    """

    def __init__(
        self,
        max_distance: float = 50.0,
        max_age: int = 10,
        min_hits: int = 3,
        max_history_length: int = 100,
    ):
        self.max_distance = float(max_distance)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.max_history_length = int(max_history_length)

        self._tracks: Dict[int, TrackedSpot] = {}
        self._next_id: int = 0
        self._frame_count: int = 0

    def update(
        self,
        detections: List[Tuple[float, float, float]],
    ) -> TrackReport:
        """用新的检测结果更新追踪器。

        Parameters
        ----------
        detections : List[Tuple[float, float, float]]
            检测结果列表，每个元素为 (x, y, confidence)。

        Returns
        -------
        TrackReport
            本帧追踪报告。
        """
        self._frame_count += 1
        frame_num = self._frame_count

        # 转换为 numpy 数组
        if not detections:
            det_array = np.empty((0, 2), dtype=np.float64)
            det_confs = np.empty(0, dtype=np.float64)
        else:
            det_array = np.array([[d[0], d[1]] for d in detections], dtype=np.float64)
            det_confs = np.array([d[2] if len(d) > 2 else 1.0 for d in detections], dtype=np.float64)

        # 获取活跃轨迹
        active_ids = list(self._tracks.keys())
        matched_det_indices: set = set()
        matched_track_ids: set = set()
        new_track_count = 0

        if len(active_ids) > 0 and len(detections) > 0:
            # 构建轨迹位置矩阵
            track_positions = np.array([
                [self._tracks[tid].center[0], self._tracks[tid].center[1]]
                for tid in active_ids
            ], dtype=np.float64)

            # 计算距离矩阵
            # dist_matrix[i, j] = 轨迹 i 与检测 j 之间的欧氏距离
            diff = track_positions[:, np.newaxis, :] - det_array[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

            # 贪心匹配 (按距离从小到大排序)
            matches: List[Tuple[int, int, float]] = []
            flat_indices = np.argsort(dist_matrix, axis=None)

            for idx in flat_indices:
                ti = int(idx // len(detections))
                dj = int(idx % len(detections))
                if ti >= len(active_ids):
                    continue

                dist = dist_matrix[ti, dj]
                if dist > self.max_distance:
                    break  # 后续距离更大，无需继续

                if ti in matched_track_ids or dj in matched_det_indices:
                    continue

                matches.append((active_ids[ti], dj, dist))
                matched_track_ids.add(ti)
                matched_det_indices.add(dj)

            # 更新匹配的轨迹
            for track_id, det_idx, dist in matches:
                self._update_track(track_id, det_array[det_idx], det_confs[det_idx], dist, frame_num)

        # 创建新轨迹 (未匹配的检测)
        for dj in range(len(detections)):
            if dj not in matched_det_indices:
                self._create_track(det_array[dj], det_confs[dj], frame_num)
                new_track_count += 1

        # 删除过期轨迹
        lost_count = 0
        expired_ids = []
        for tid, track in self._tracks.items():
            if frame_num - track.last_seen > self.max_age:
                expired_ids.append(tid)
                lost_count += 1

        for tid in expired_ids:
            del self._tracks[tid]

        # 计算光斑间距离
        active_tracks = self._get_confirmed_tracks()
        inter_distances = self._compute_inter_spot_distances(active_tracks)

        report = TrackReport(
            active_tracks=active_tracks,
            new_tracks=new_track_count,
            lost_tracks=lost_count,
            matched_count=len(matched_det_indices),
            unmatched_detections=len(detections) - len(matched_det_indices),
            inter_spot_distances=inter_distances,
            frame_number=frame_num,
        )

        LOGGER.debug(
            "MultiSpot: frame=%d, active=%d, matched=%d, new=%d, lost=%d",
            frame_num, len(active_tracks), report.matched_count,
            new_track_count, lost_count,
        )

        return report

    def _create_track(
        self,
        position: np.ndarray,
        confidence: float,
        frame_num: int,
    ) -> None:
        """创建新轨迹。

        Parameters
        ----------
        position : np.ndarray
            检测位置 [x, y]。
        confidence : float
            检测置信度。
        frame_num : int
            当前帧号。
        """
        track_id = self._next_id
        self._next_id += 1

        track = TrackedSpot(
            track_id=track_id,
            center=(float(position[0]), float(position[1])),
            velocity=(0.0, 0.0),
            confidence=float(confidence),
            age=1,
            last_seen=frame_num,
            hit_count=1,
            position_history=[(float(position[0]), float(position[1]))],
        )
        self._tracks[track_id] = track

        LOGGER.debug("MultiSpot: 创建新轨迹 id=%d, pos=(%.1f, %.1f)",
                     track_id, position[0], position[1])

    def _update_track(
        self,
        track_id: int,
        position: np.ndarray,
        confidence: float,
        distance: float,
        frame_num: int,
    ) -> None:
        """更新已有轨迹。

        Parameters
        ----------
        track_id : int
            轨迹 ID。
        position : np.ndarray
            新检测位置 [x, y]。
        confidence : float
            检测置信度。
        distance : float
            关联距离。
        frame_num : int
            当前帧号。
        """
        track = self._tracks[track_id]
        old_cx, old_cy = track.center

        # 更新速度 (指数移动平均)
        new_cx = float(position[0])
        new_cy = float(position[1])
        alpha = 0.7  # 速度平滑因子
        track.velocity = (
            alpha * (new_cx - old_cx) + (1 - alpha) * track.velocity[0],
            alpha * (new_cy - old_cy) + (1 - alpha) * track.velocity[1],
        )

        # 更新位置
        track.center = (new_cx, new_cy)
        track.confidence = float(confidence)
        track.age += 1
        track.last_seen = frame_num
        track.hit_count += 1

        # 更新位置历史
        track.position_history.append((new_cx, new_cy))
        if len(track.position_history) > self.max_history_length:
            track.position_history = track.position_history[-self.max_history_length:]

    def _get_confirmed_tracks(self) -> List[TrackedSpot]:
        """获取已确认的活跃轨迹 (命中次数 >= min_hits)。

        Returns
        -------
        List[TrackedSpot]
            已确认的轨迹列表。
        """
        return [
            track for track in self._tracks.values()
            if track.hit_count >= self.min_hits
        ]

    @staticmethod
    def _compute_inter_spot_distances(
        tracks: List[TrackedSpot],
    ) -> Dict[Tuple[int, int], float]:
        """计算所有光斑对之间的欧氏距离。

        Parameters
        ----------
        tracks : List[TrackedSpot]
            活跃轨迹列表。

        Returns
        -------
        Dict[Tuple[int, int], float]
            {(track_id_i, track_id_j): distance} 距离字典。
            键为排序后的 ID 对 (i < j)。
        """
        distances: Dict[Tuple[int, int], float] = {}
        n = len(tracks)
        for i in range(n):
            for j in range(i + 1, n):
                dx = tracks[i].center[0] - tracks[j].center[0]
                dy = tracks[i].center[1] - tracks[j].center[1]
                dist = float(np.sqrt(dx ** 2 + dy ** 2))
                key = (
                    min(tracks[i].track_id, tracks[j].track_id),
                    max(tracks[i].track_id, tracks[j].track_id),
                )
                distances[key] = round(dist, 2)
        return distances

    def get_active_tracks(self) -> List[TrackedSpot]:
        """获取当前所有活跃轨迹 (含未确认的)。

        Returns
        -------
        List[TrackedSpot]
            活跃轨迹列表。
        """
        return list(self._tracks.values())

    def get_track_by_id(self, track_id: int) -> Optional[TrackedSpot]:
        """根据 ID 获取轨迹。

        Parameters
        ----------
        track_id : int
            轨迹 ID。

        Returns
        -------
        TrackedSpot or None
            轨迹对象。如果不存在返回 None。
        """
        return self._tracks.get(track_id)

    @property
    def active_count(self) -> int:
        """当前活跃轨迹数。"""
        return len(self._tracks)

    @property
    def frame_number(self) -> int:
        """当前帧号。"""
        return self._frame_count

    def reset(self) -> None:
        """重置追踪器，清除所有轨迹。"""
        self._tracks.clear()
        self._next_id = 0
        self._frame_count = 0
        LOGGER.info("MultiSpot: 追踪器已重置")
