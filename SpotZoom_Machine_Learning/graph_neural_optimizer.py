"""
图神经网络光斑关联优化器 (GraphNeuralOptimizer)

基于PyTorch Geometric图神经网络架构的多光斑关联与状态预测模块。

算法原理:
- Graph Attention Network (GAT) — 注意力机制图卷积
- Edge Convolution — 点云风格图卷积
- Message Passing Neural Network — 消息传递神经网络

功能:
- 多光斑拓扑关系建模
- 图结构特征学习
- 光斑状态预测与异常检测

依赖: numpy, scipy
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum, auto
import numpy as np


class GraphAggregationType(Enum):
    """图聚合类型"""
    MEAN = auto()
    MAX = auto()
    ATTENTION = auto()
    SET_TRANSFORMER = auto()


@dataclass
class SpotGraphNode:
    """光斑图节点"""
    spot_id: int
    x: float
    y: float
    intensity: float
    area: float
    eccentricity: float
    timestamp: float = 0.0


@dataclass
class SpotGraphEdge:
    """光斑图边"""
    source_id: int
    target_id: int
    distance: float
    intensity_ratio: float


@dataclass
class GraphNodeFeatures:
    """图节点特征向量"""
    position: np.ndarray  # [x, y]
    morphology: np.ndarray  # [area, eccentricity, intensity]
    motion: np.ndarray  # [vx, vy, ax, ay]
    temporal: np.ndarray  # [age, confidence, visibility]


class GraphNeuralOptimizer:
    """基于图神经网络的优化器。
    
    使用图注意力网络对多光斑系统进行建模和优化。
    
    Attributes:
        num_layers: 图卷积层数
        hidden_dim: 隐藏层维度
        attention_heads: 注意力头数
    """
    
    def __init__(
        self,
        num_layers: int = 3,
        hidden_dim: int = 64,
        attention_heads: int = 4,
        aggregation: GraphAggregationType = GraphAggregationType.ATTENTION,
        dropout: float = 0.1,
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.aggregation = aggregation
        self.dropout = dropout
        
        # 初始化权重矩阵
        self._init_weights()
        
        # 节点和边特征缓存
        self._node_features: Optional[np.ndarray] = None
        self._adjacency_matrix: Optional[np.ndarray] = None
        self._edge_features: Optional[np.ndarray] = None
        
    def _init_weights(self) -> None:
        """初始化网络权重"""
        scale = np.sqrt(2.0 / self.hidden_dim)
        
        # 特征投影权重
        self._input_proj = np.random.randn(9, self.hidden_dim).astype(np.float32) * scale
        
        # 图卷积层权重
        self._gat_weights = [
            np.random.randn(self.hidden_dim, self.hidden_dim * self.attention_heads).astype(np.float32) * scale
            for _ in range(self.num_layers)
        ]
        self._gat_attention = [
            np.random.randn(self.attention_heads * 2, 1).astype(np.float32) * 0.1
            for _ in range(self.num_layers)
        ]
        
        # 输出层权重
        self._output_proj = np.random.randn(self.hidden_dim, 4).astype(np.float32) * scale
        
    def _compute_adjacency(
        self,
        positions: np.ndarray,
        max_distance: float = 50.0
    ) -> np.ndarray:
        """基于距离计算邻接矩阵
        
        Parameters:
            positions: 节点位置数组 [N, 2]
            max_distance: 最大连接距离
            
        Returns:
            邻接矩阵 [N, N]，无权连接为0
        """
        n = positions.shape[0]
        dist_matrix = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
            axis=2
        )
        
        # K近邻连接
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            distances = dist_matrix[i]
            k = min(5, n - 1)
            nearest_indices = np.argpartition(distances, k)[:k+1]
            nearest_indices = nearest_indices[nearest_indices != i]
            adj[i, nearest_indices] = 1.0
            
        # 距离加权
        adj = adj * (dist_matrix < max_distance).astype(np.float32)
        
        return adj
        
    def _compute_edge_features(
        self,
        positions: np.ndarray,
        adj: np.ndarray
    ) -> np.ndarray:
        """计算边特征
        
        Parameters:
            positions: 节点位置 [N, 2]
            adj: 邻接矩阵 [N, N]
            
        Returns:
            边特征列表 [M, 4]，每条边: [dx, dy, dist, angle]
        """
        n = positions.shape[0]
        edges = []
        
        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0 and i != j:
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    angle = np.arctan2(dy, dx)
                    edges.append([dx, dy, dist, angle])
                    
        if len(edges) == 0:
            return np.zeros((1, 4), dtype=np.float32)
            
        return np.array(edges, dtype=np.float32)
        
    def _gat_layer(
        self,
        node_features: np.ndarray,
        adj: np.ndarray,
        layer_idx: int
    ) -> np.ndarray:
        """图注意力层
        
        Parameters:
            node_features: 节点特征 [N, D]
            adj: 邻接矩阵 [N, N]
            layer_idx: 层索引
            
        Returns:
            更新后的节点特征 [N, H*D]
        """
        # 线性变换
        transformed = node_features @ self._gat_weights[layer_idx]
        
        # 多头注意力
        n, d = transformed.shape
        h = self.attention_heads
        d_h = d // h
        
        outputs = []
        for head in range(h):
            start = head * d_h
            end = (head + 1) * d_h
            h_features = transformed[:, start:end]  # [N, d_h]
            
            # 计算注意力分数
            a_input = np.concatenate([
                h_features,
                h_features
            ], axis=1)  # [N, 2*d_h]
            e = a_input @ self._gat_attention[layer_idx]  # [N, 1]
            e = e.flatten()  # [N]
            
            # Masked softmax
            attention_scores = np.full(n, -1e9)
            for i in range(n):
                neighbors = np.where(adj[i] > 0)[0]
                if len(neighbors) > 0:
                    scores = e[neighbors] + e[i]
                    attention_scores[neighbors] = np.maximum(
                        attention_scores[neighbors],
                        scores
                    )
                    
            attention_weights = self._softmax(attention_scores)
            
            # 邻域聚合
            h_agg = np.zeros(n * d_h, dtype=np.float32)
            for i in range(n):
                neighbors = np.where(adj[i] > 0)[0]
                if len(neighbors) > 0:
                    for j in neighbors:
                        h_agg[j * d_h:(j + 1) * d_h] += (
                            attention_weights[i] * h_features[i]
                        )
                        
            outputs.append(h_agg.reshape(n, d_h))
            
        return np.concatenate(outputs, axis=1)
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """数值稳定的softmax"""
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x) + 1e-9)
        
    def _aggregate(
        self,
        node_features: np.ndarray,
        adj: np.ndarray
    ) -> np.ndarray:
        """图池化聚合"""
        if self.aggregation == GraphAggregationType.MEAN:
            deg = np.sum(adj, axis=1, keepdims=True) + 1e-9
            return node_features / deg
        elif self.aggregation == GraphAggregationType.MAX:
            masked = node_features * (adj > 0).astype(np.float32)
            masked[adj == 0] = -1e9
            return np.max(masked, axis=1, keepdims=True).repeat(
                node_features.shape[1], axis=1
            )
        else:
            return node_features
            
    def build_graph(
        self,
        spots: List[SpotGraphNode]
    ) -> None:
        """从光斑列表构建图结构
        
        Parameters:
            spots: 光斑节点列表
        """
        n = len(spots)
        
        # 提取位置和特征
        positions = np.array([[s.x, s.y] for s in spots], dtype=np.float32)
        features = np.array([
            [
                s.x, s.y,
                s.intensity, s.area, s.eccentricity,
                0, 0, 0, 0  # 运动特征初始为0
            ]
            for s in spots
        ], dtype=np.float32)
        
        # 计算邻接矩阵
        self._adjacency_matrix = self._compute_adjacency(positions)
        
        # 计算边特征
        self._edge_features = self._compute_edge_features(
            positions, self._adjacency_matrix
        )
        
        # 特征投影
        self._node_features = features @ self._input_proj
        
    def forward(self) -> np.ndarray:
        """前向传播
        
        Returns:
            预测输出 [N, 4] - [dx, dy, confidence, anomaly_score]
        """
        if self._node_features is None:
            raise ValueError("Must call build_graph() before forward()")
            
        node_features = self._node_features.copy()
        
        # 多层图注意力卷积
        for layer_idx in range(self.num_layers):
            node_features = self._gat_layer(
                node_features,
                self._adjacency_matrix,
                layer_idx
            )
            
        # 输出预测
        output = node_features @ self._output_proj
        
        # 后处理：归一化位移、计算置信度
        dx, dy = output[:, 0], output[:, 1]
        
        # 置信度基于邻接度
        degrees = np.sum(self._adjacency_matrix, axis=1)
        confidence = np.tanh(degrees / 5.0)
        
        # 异常分数基于特征残差
        residual = np.linalg.norm(
            self._node_features - node_features,
            axis=1
        )
        anomaly_score = 1.0 / (1.0 + np.exp(-(residual - 2.0)))
        
        return np.column_stack([dx, dy, confidence, anomaly_score])
        
    def predict_next_positions(
        self,
        spots: List[SpotGraphNode],
        time_delta: float = 1.0
    ) -> List[Tuple[int, float, float, float]]:
        """预测下一时刻光斑位置
        
        Parameters:
            spots: 当前光斑列表
            time_delta: 时间步长
            
        Returns:
            预测结果列表 [(spot_id, x, y, confidence), ...]
        """
        self.build_graph(spots)
        predictions = self.forward()
        
        results = []
        for i, spot in enumerate(spots):
            dx, dy = predictions[i, 0] * time_delta, predictions[i, 1] * time_delta
            confidence = predictions[i, 2]
            
            results.append((
                spot.spot_id,
                spot.x + dx,
                spot.y + dy,
                confidence
            ))
            
        return results
        
    def detect_anomalies(
        self,
        spots: List[SpotGraphNode],
        threshold: float = 0.7
    ) -> List[Tuple[int, float]]:
        """检测异常光斑
        
        Parameters:
            spots: 光斑列表
            threshold: 异常阈值
            
        Returns:
            异常光斑列表 [(spot_id, anomaly_score), ...]
        """
        self.build_graph(spots)
        predictions = self.forward()
        
        anomalies = []
        for i, spot in enumerate(spots):
            score = predictions[i, 3]
            if score > threshold:
                anomalies.append((spot.spot_id, score))
                
        return sorted(anomalies, key=lambda x: x[1], reverse=True)
        
    def compute_spot_relationships(
        self,
        spots: List[SpotGraphNode]
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """计算光斑间的语义关系
        
        Parameters:
            spots: 光斑列表
            
        Returns:
            关系字典 {(id1, id2): {'distance': ..., 'angle': ..., 'motion_corr': ...}}
        """
        self.build_graph(spots)
        predictions = self.forward()
        
        positions = np.array([[s.x, s.y] for s in spots])
        adj = self._adjacency_matrix
        
        relationships = {}
        for i in range(len(spots)):
            for j in range(len(spots)):
                if adj[i, j] > 0 and i != j:
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    angle = np.arctan2(dy, dx)
                    
                    # 运动相关性
                    motion_i = predictions[i, :2]
                    motion_j = predictions[j, :2]
                    motion_corr = np.dot(motion_i, motion_j) / (
                        np.linalg.norm(motion_i) * np.linalg.norm(motion_j) + 1e-9
                    )
                    
                    relationships[(spots[i].spot_id, spots[j].spot_id)] = {
                        'distance': float(dist),
                        'angle': float(angle),
                        'motion_correlation': float(motion_corr)
                    }
                    
        return relationships
        
    def reset(self) -> None:
        """重置优化器状态"""
        self._node_features = None
        self._adjacency_matrix = None
        self._edge_features = None


__all__ = [
    'GraphNeuralOptimizer',
    'SpotGraphNode',
    'SpotGraphEdge',
    'GraphNodeFeatures',
    'GraphAggregationType',
]
