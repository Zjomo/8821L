"""
注意力光斑跟踪器 (Attention Spot Tracker)

基于 Transformer 注意力机制的多帧光斑跟踪模块。
使用自注意力和交叉注意力实现多光斑上下文感知和时间一致性跟踪。

灵感来源:
- TrackFormer (mikel-brostrom/TrackFormer): Transformer 目标跟踪
  (https://github.com/mikel-brostrom/TrackFormer)
- Vision Transformer (ViT): 注意力机制用于视觉任务
- DETR (facebookresearch/detr): 端到端目标检测 Transformer
  (https://github.com/facebookresearch/detr)

算法原理:
  1. 位置编码: 为每个光斑位置添加空间位置信息
  2. 自注意力: 多光斑之间的上下文交互
     Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
  3. 交叉注意力: 当前帧与历史帧之间的时间关联
  4. 置信度加权: 基于注意力权重融合多假设跟踪结果
  5. 注意力热力图: 可视化诊断跟踪质量

外部依赖: numpy, cv2 (可选, 用于可视化)
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class PositionalEncodingType(Enum):
    """位置编码类型枚举。"""
    SINUSOIDAL = "sinusoidal"   # 正弦位置编码
    LEARNED = "learned"         # 可学习位置编码 (随机初始化)
    RELATIVE = "relative"       # 相对位置编码


@dataclass
class Track:
    """单条跟踪轨迹。"""
    track_id: int = -1                     # 轨迹 ID
    positions: List[np.ndarray] = field(default_factory=list)  # 位置历史
    confidences: List[float] = field(default_factory=list)     # 置信度历史
    attention_weights: List[float] = field(default_factory=list)  # 注意力权重历史
    active: bool = True                    # 是否活跃
    last_updated: int = 0                  # 最后更新帧号
    total_distance: float = 0.0            # 累计移动距离


@dataclass
class AttentionTrackResult:
    """注意力跟踪结果。"""
    tracks: Dict[int, Track] = field(default_factory=dict)  # 所有轨迹
    current_positions: np.ndarray = None     # 当前帧位置 (N, 2)
    current_confidences: np.ndarray = None   # 当前帧置信度 (N,)
    attention_heatmap: np.ndarray = None     # 注意力热力图
    cross_attention_map: np.ndarray = None   # 交叉注意力图
    n_active_tracks: int = 0                 # 活跃轨迹数
    n_lost_tracks: int = 0                   # 丢失轨迹数
    frame_id: int = 0                        # 帧号
    processing_time_ms: float = 0.0          # 处理耗时 (ms)


@dataclass
class AttentionTrackerConfig:
    """注意力光斑跟踪器配置。"""
    # 注意力参数
    attention_heads: int = 4                 # 注意力头数
    embed_dim: int = 32                      # 嵌入维度
    temporal_window: int = 5                 # 时间窗口 (帧数)

    # 位置编码
    positional_encoding: PositionalEncodingType = PositionalEncodingType.SINUSOIDAL
    max_spatial_range: float = 100.0         # 最大空间范围 (用于位置编码归一化)

    # 跟踪参数
    max_tracks: int = 50                     # 最大轨迹数
    association_threshold: float = 15.0      # 关联阈值 (像素)
    lost_threshold: int = 10                 # 丢失阈值 (帧)
    new_track_threshold: float = 0.3         # 新轨迹创建阈值

    # 特征参数
    feature_dim: int = 4                     # 特征维度 (x, y, intensity, size)
    value_decay: float = 0.9                 # 值衰减因子

    # 可视化
    heatmap_resolution: int = 64             # 热力图分辨率


class AttentionSpotTracker:
    """注意力光斑跟踪器。

    基于 Transformer 注意力机制的多帧多光斑跟踪。
    支持自注意力 (多光斑上下文) 和交叉注意力 (时间一致性)。

    使用示例:
        tracker = AttentionTrackerConfig()
        for detections in detection_sequence:
            result = tracker.update(detections)
            print(f"Active tracks: {result.n_active_tracks}")
        # 获取注意力热力图
        heatmap = result.attention_heatmap
    """

    def __init__(self, config: Optional[AttentionTrackerConfig] = None):
        self._config = config or AttentionTrackerConfig()
        self._tracks: Dict[int, Track] = {}
        self._next_track_id: int = 0
        self._frame_id: int = 0
        self._history_buffer: List[np.ndarray] = []
        self._pos_encoding_cache: Optional[np.ndarray] = None

        # 初始化注意力权重矩阵
        self._init_attention_weights()

    @property
    def config(self) -> AttentionTrackerConfig:
        return self._config

    def _init_attention_weights(self):
        """初始化注意力权重矩阵。"""
        cfg = self._config
        d_k = cfg.embed_dim // cfg.attention_heads

        # Query, Key, Value 投影矩阵
        scale = np.sqrt(2.0 / cfg.embed_dim)
        self._W_q = np.random.normal(0, scale, (cfg.embed_dim, cfg.embed_dim))
        self._W_k = np.random.normal(0, scale, (cfg.embed_dim, cfg.embed_dim))
        self._W_v = np.random.normal(0, scale, (cfg.embed_dim, cfg.embed_dim))

        # 输出投影
        self._W_o = np.random.normal(0, scale, (cfg.embed_dim, cfg.embed_dim))

        # 交叉注意力权重 (时间)
        self._W_q_cross = np.random.normal(0, scale, (cfg.embed_dim, cfg.embed_dim))
        self._W_k_cross = np.random.normal(0, scale, (cfg.embed_dim, cfg.embed_dim))
        self._W_v_cross = np.random.normal(0, scale, (cfg.embed_dim, cfg.embed_dim))

    def update(
        self,
        detections: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> AttentionTrackResult:
        """更新跟踪器。

        Args:
            detections: 检测结果 (N, 2) 或 (N, 4) [x, y, intensity, size]
            features: 可选特征向量 (N, feature_dim)

        Returns:
            AttentionTrackResult: 跟踪结果
        """
        import time
        t0 = time.perf_counter()

        self._frame_id += 1
        detections = np.asarray(detections, dtype=np.float64)

        if detections.ndim == 1:
            detections = detections.reshape(1, -1)

        n_detections = detections.shape[0]

        # 提取位置和特征
        positions = detections[:, :2]
        if features is None:
            if detections.shape[1] >= 4:
                features = detections[:, :4]
            else:
                features = np.hstack([
                    positions,
                    np.ones((n_detections, 2), dtype=np.float64)
                ])

        # 位置编码
        pos_encoded = self._add_positional_encoding(positions)

        # 嵌入
        embeddings = self._embed(pos_encoded, features)

        # 自注意力 (多光斑上下文)
        self_attn_output, self_attn_weights = self._self_attention(embeddings)

        # 交叉注意力 (时间一致性)
        cross_attn_output, cross_attn_weights = self._cross_attention(self_attn_output)

        # 数据关联
        associated = self._associate_detections(
            cross_attn_output, positions, self_attn_weights
        )

        # 更新轨迹
        self._update_tracks(associated, positions, self_attn_weights)

        # 生成注意力热力图
        heatmap = self._generate_heatmap(positions, self_attn_weights)

        # 统计
        n_active = sum(1 for t in self._tracks.values() if t.active)
        n_lost = sum(1 for t in self._tracks.values() if not t.active)

        # 保存历史
        self._history_buffer.append(positions.copy())
        if len(self._history_buffer) > self._config.temporal_window:
            self._history_buffer.pop(0)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 当前帧结果
        active_positions = np.array([
            t.positions[-1] for t in self._tracks.values() if t.active and t.positions
        ]) if any(t.active and t.positions for t in self._tracks.values()) else np.zeros((0, 2))
        active_confidences = np.array([
            t.confidences[-1] for t in self._tracks.values() if t.active and t.confidences
        ]) if any(t.active and t.confidences for t in self._tracks.values()) else np.zeros(0)

        result = AttentionTrackResult(
            tracks=dict(self._tracks),
            current_positions=active_positions,
            current_confidences=active_confidences,
            attention_heatmap=heatmap,
            cross_attention_map=cross_attn_weights,
            n_active_tracks=n_active,
            n_lost_tracks=n_lost,
            frame_id=self._frame_id,
            processing_time_ms=elapsed_ms,
        )

        logger.debug(
            f"AttentionSpotTracker: frame={self._frame_id}, "
            f"detections={n_detections}, active={n_active}, lost={n_lost}, "
            f"time={elapsed_ms:.1f}ms"
        )

        return result

    def _add_positional_encoding(
        self, positions: np.ndarray
    ) -> np.ndarray:
        """添加位置编码。

        Args:
            positions: 位置 (N, 2)

        Returns:
            位置编码后的特征 (N, embed_dim)
        """
        cfg = self._config
        n = positions.shape[0]
        d = cfg.embed_dim

        if cfg.positional_encoding == PositionalEncodingType.SINUSOIDAL:
            # 正弦位置编码
            pe = np.zeros((n, d), dtype=np.float64)
            div_term = np.exp(
                np.arange(0, d, 2, dtype=np.float64) *
                -(np.log(10000.0) / d)
            )
            # 归一化位置
            norm_pos = positions / (cfg.max_spatial_range + 1e-10)
            pe[:, 0::2] = np.sin(norm_pos[:, 0:1] * div_term[np.newaxis, :])
            pe[:, 1::2] = np.cos(norm_pos[:, 0:1] * div_term[np.newaxis, :])
            # 如果维度足够，添加 y 坐标编码
            if d > 4:
                pe[:, 2::2] = np.sin(norm_pos[:, 1:2] * div_term[:d // 2 - 1][np.newaxis, :])
                pe[:, 3::2] = np.cos(norm_pos[:, 1:2] * div_term[:d // 2 - 1][np.newaxis, :])
            return pe

        elif cfg.positional_encoding == PositionalEncodingType.RELATIVE:
            # 相对位置编码
            pe = np.zeros((n, d), dtype=np.float64)
            if n > 1:
                center = np.mean(positions, axis=0)
                rel_pos = positions - center
                rel_norm = rel_pos / (cfg.max_spatial_range + 1e-10)
                for i in range(n):
                    for j in range(min(d, 4)):
                        pe[i, j] = rel_norm[i, j % 2]
            return pe

        else:  # LEARNED (随机初始化，固定)
            if self._pos_encoding_cache is None or self._pos_encoding_cache.shape[0] < n:
                max_n = max(n, cfg.max_tracks)
                self._pos_encoding_cache = np.random.normal(
                    0, 0.02, (max_n, d)
                ).astype(np.float64)
            return self._pos_encoding_cache[:n].copy()

    def _embed(
        self, pos_encoded: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        """嵌入特征。

        将位置编码和检测特征融合为嵌入向量。

        Args:
            pos_encoded: 位置编码 (N, embed_dim)
            features: 检测特征 (N, feature_dim)

        Returns:
            嵌入向量 (N, embed_dim)
        """
        cfg = self._config
        n = pos_encoded.shape[0]

        # 特征投影到嵌入维度
        if features.shape[1] != cfg.embed_dim:
            # 简单投影: 重复或截断
            if features.shape[1] < cfg.embed_dim:
                projected = np.zeros((n, cfg.embed_dim), dtype=np.float64)
                projected[:, :features.shape[1]] = features
                # 循环填充
                for i in range(features.shape[1], cfg.embed_dim):
                    projected[:, i] = features[:, i % features.shape[1]]
            else:
                projected = features[:, :cfg.embed_dim]
        else:
            projected = features

        # 融合位置编码和特征
        embeddings = pos_encoded + projected * 0.5

        # Layer Normalization (简化版)
        mean = np.mean(embeddings, axis=1, keepdims=True)
        std = np.std(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings = (embeddings - mean) / std

        return embeddings

    def _self_attention(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """多头自注意力。

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

        Args:
            embeddings: 嵌入向量 (N, embed_dim)

        Returns:
            (output, attention_weights)
        """
        cfg = self._config
        n = embeddings.shape[0]
        d_k = cfg.embed_dim // cfg.attention_heads

        if n == 0:
            return embeddings, np.zeros((0, 0))

        # 投影
        Q = embeddings @ self._W_q  # (N, embed_dim)
        K = embeddings @ self._W_k
        V = embeddings @ self._W_v

        # 多头分割
        Q_heads = self._split_heads(Q, cfg.attention_heads)  # (heads, N, d_k)
        K_heads = self._split_heads(K, cfg.attention_heads)
        V_heads = self._split_heads(V, cfg.attention_heads)

        # 注意力计算
        scale = np.sqrt(d_k)
        all_weights = []
        all_outputs = []

        for h in range(cfg.attention_heads):
            scores = Q_heads[h] @ K_heads[h].T / scale  # (N, N)
            weights = self._softmax(scores)  # (N, N)
            output = weights @ V_heads[h]  # (N, d_k)
            all_weights.append(weights)
            all_outputs.append(output)

        # 合并多头
        combined = self._combine_heads(all_outputs)  # (N, embed_dim)
        output = combined @ self._W_o  # (N, embed_dim)

        # 平均注意力权重
        avg_weights = np.mean(all_weights, axis=0)

        return output, avg_weights

    def _cross_attention(
        self, query: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """交叉注意力 (当前帧 vs 历史帧)。

        Args:
            query: 当前帧嵌入 (N, embed_dim)

        Returns:
            (output, attention_weights)
        """
        cfg = self._config

        if not self._history_buffer or len(self._history_buffer) < 1:
            return query, np.zeros((query.shape[0], query.shape[0]))

        # 构建历史键值 (简化: 使用历史位置作为键)
        history_positions = np.vstack(self._history_buffer)  # (T*M, 2)
        history_pe = self._add_positional_encoding(history_positions)

        # 投影
        Q = query @ self._W_q_cross
        K = history_pe @ self._W_k_cross
        V = history_pe @ self._W_v_cross

        d_k = cfg.embed_dim // cfg.attention_heads
        scale = np.sqrt(d_k)

        # 简化: 单头交叉注意力
        scores = Q @ K.T / scale  # (N, T*M)
        weights = self._softmax(scores)  # (N, T*M)
        output = weights @ V  # (N, embed_dim)

        # 残差连接
        output = query + output * 0.5

        return output, weights

    def _split_heads(
        self, x: np.ndarray, n_heads: int
    ) -> np.ndarray:
        """分割多头。"""
        d_k = x.shape[1] // n_heads
        heads = np.zeros((n_heads, x.shape[0], d_k), dtype=np.float64)
        for h in range(n_heads):
            heads[h] = x[:, h * d_k:(h + 1) * d_k]
        return heads

    def _combine_heads(self, heads: List[np.ndarray]) -> np.ndarray:
        """合并多头。"""
        return np.concatenate(heads, axis=1)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """数值稳定的 softmax。"""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)

    def _associate_detections(
        self,
        embeddings: np.ndarray,
        positions: np.ndarray,
        attention_weights: np.ndarray,
    ) -> Dict[int, int]:
        """数据关联: 将检测与现有轨迹匹配。

        使用注意力加权的距离进行关联。

        Args:
            embeddings: 嵌入向量 (N, embed_dim)
            positions: 检测位置 (N, 2)
            attention_weights: 注意力权重 (N, N)

        Returns:
            关联映射 {track_id: detection_index}
        """
        cfg = self._config
        associations = {}

        active_tracks = {
            tid: t for tid, t in self._tracks.items() if t.active
        }

        if not active_tracks or positions.shape[0] == 0:
            return associations

        # 计算检测与轨迹之间的距离
        track_ids = list(active_tracks.keys())
        track_positions = np.array([
            active_tracks[tid].positions[-1] for tid in track_ids
        ])

        # 距离矩阵
        diff = positions[:, np.newaxis, :] - track_positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))

        # 注意力加权距离 (高注意力的检测更可信)
        if attention_weights.shape[0] == positions.shape[0]:
            diag_attn = np.diag(attention_weights)
            weighted_distances = distances / (diag_attn[:, np.newaxis] + 1e-10)
        else:
            weighted_distances = distances

        # 贪心匹配
        used_detections = set()
        used_tracks = set()

        # 按距离排序
        flat_indices = np.argsort(weighted_distances.ravel())
        for idx in flat_indices:
            det_idx = idx // len(track_ids)
            trk_idx = idx % len(track_ids)

            if det_idx in used_detections or trk_idx in used_tracks:
                continue

            if weighted_distances[det_idx, trk_idx] > cfg.association_threshold:
                continue

            track_id = track_ids[trk_idx]
            associations[track_id] = det_idx
            used_detections.add(det_idx)
            used_tracks.add(trk_idx)

        return associations

    def _update_tracks(
        self,
        associations: Dict[int, int],
        positions: np.ndarray,
        attention_weights: np.ndarray,
    ):
        """更新轨迹状态。

        Args:
            associations: 关联映射
            positions: 检测位置
            attention_weights: 注意力权重
        """
        cfg = self._config

        # 更新已关联的轨迹
        for track_id, det_idx in associations.items():
            track = self._tracks[track_id]
            new_pos = positions[det_idx]

            # 计算置信度 (基于注意力权重)
            if attention_weights.shape[0] > det_idx:
                confidence = float(np.mean(attention_weights[det_idx]))
            else:
                confidence = 0.5

            # 更新轨迹
            if track.positions:
                dist = np.linalg.norm(new_pos - track.positions[-1])
                track.total_distance += dist

            track.positions.append(new_pos.copy())
            track.confidences.append(confidence)
            track.attention_weights.append(confidence)
            track.last_updated = self._frame_id
            track.active = True

        # 标记丢失的轨迹
        for track_id, track in self._tracks.items():
            if track_id not in associations and track.active:
                if self._frame_id - track.last_updated > cfg.lost_threshold:
                    track.active = False

        # 创建新轨迹
        associated_dets = set(associations.values())
        for det_idx in range(positions.shape[0]):
            if det_idx not in associated_dets:
                if attention_weights.shape[0] > det_idx:
                    attn_conf = float(np.mean(attention_weights[det_idx]))
                else:
                    attn_conf = 0.5

                if attn_conf >= cfg.new_track_threshold:
                    new_track = Track(
                        track_id=self._next_track_id,
                        positions=[positions[det_idx].copy()],
                        confidences=[attn_conf],
                        attention_weights=[attn_conf],
                        active=True,
                        last_updated=self._frame_id,
                    )
                    self._tracks[self._next_track_id] = new_track
                    self._next_track_id += 1

        # 限制轨迹数量
        if len(self._tracks) > cfg.max_tracks:
            # 删除最旧的非活跃轨迹
            inactive = [
                tid for tid, t in self._tracks.items()
                if not t.active
            ]
            for tid in inactive[:len(self._tracks) - cfg.max_tracks]:
                del self._tracks[tid]

    def _generate_heatmap(
        self,
        positions: np.ndarray,
        attention_weights: np.ndarray,
    ) -> np.ndarray:
        """生成注意力热力图。

        Args:
            positions: 光斑位置 (N, 2)
            attention_weights: 注意力权重 (N, N)

        Returns:
            热力图 (resolution, resolution), float64
        """
        cfg = self._config
        res = cfg.heatmap_resolution
        heatmap = np.zeros((res, res), dtype=np.float64)

        if positions.shape[0] == 0:
            return heatmap

        # 归一化位置到 [0, res)
        x_min, y_min = np.min(positions, axis=0)
        x_max, y_max = np.max(positions, axis=0)
        range_x = x_max - x_min + 1e-10
        range_y = y_max - y_min + 1e-10

        # 绘制每个光斑的注意力
        for i in range(positions.shape[0]):
            px = int((positions[i, 0] - x_min) / range_x * (res - 1))
            py = int((positions[i, 1] - y_min) / range_y * (res - 1))
            px = np.clip(px, 0, res - 1)
            py = np.clip(py, 0, res - 1)

            # 注意力强度
            if attention_weights.shape[0] > i:
                intensity = float(np.mean(attention_weights[i]))
            else:
                intensity = 0.5

            # 高斯扩散
            sigma = 3
            y_coords, x_coords = np.ogrid[:res, :res]
            gaussian = np.exp(-((x_coords - px) ** 2 + (y_coords - py) ** 2) / (2 * sigma ** 2))
            heatmap += gaussian * intensity

        # 归一化
        max_val = np.max(heatmap)
        if max_val > 1e-10:
            heatmap /= max_val

        return heatmap

    def get_tracks(self) -> Dict[int, Track]:
        """获取所有轨迹。"""
        return dict(self._tracks)

    def get_active_tracks(self) -> Dict[int, Track]:
        """获取活跃轨迹。"""
        return {tid: t for tid, t in self._tracks.items() if t.active}

    def reset(self):
        """重置跟踪器状态。"""
        self._tracks.clear()
        self._next_track_id = 0
        self._frame_id = 0
        self._history_buffer.clear()
        self._pos_encoding_cache = None
        self._init_attention_weights()
        logger.info("AttentionSpotTracker: Reset")
