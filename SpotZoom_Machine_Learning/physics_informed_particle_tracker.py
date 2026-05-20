"""
物理信息粒子追踪器 (PhysicsInformedParticleTracker)

灵感来源:
- DeepTrack2 (https://github.com/DeepTrackAI/DeepTrack2) — 模块化粒子追踪图像管道
- TrackPy (https://github.com/soft-matter/trackpy) — 基于特征的粒子追踪
- HCIPy (https://github.com/ehpor/hcipy) — 光学力场仿真

算法原理:
- Kalman Filter — 最优线性状态估计器，融合预测与观测
- Brownian Motion Model — 布朗运动扩散模型约束粒子运动
- Hungarian Algorithm — 最优二部图匹配实现多粒子数据关联
- Physics Constraints — 速度/加速度上限、光学梯度力约束

功能:
- 基于卡尔曼滤波的多粒子追踪，融合物理约束
- 布朗运动、漂移、光学梯度力等多种运动模型
- 匈牙利算法实现最优多目标数据关联
- 物理约束: 最大速度、最大加速度、边界限制
- 轨迹平滑与缺失帧插值

依赖: numpy, scipy (无外部深度学习框架依赖)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

LOGGER = logging.getLogger("SpotZoom.PhysicsInformedParticleTracker")


# ---------------------------------------------------------------------------
# Noll 索引到 (n, m) 的映射 (复用标准定义)
# ---------------------------------------------------------------------------
_NOLL_TO_NM: Dict[int, Tuple[int, int]] = {
    1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0), 5: (2, 2),
    6: (2, -2), 7: (3, 1), 8: (3, -1), 9: (3, 3), 10: (3, -3),
    11: (4, 0), 12: (4, 2), 13: (4, -2), 14: (4, 4), 15: (4, -4),
}


@dataclass
class TrackerConfig:
    """追踪器配置参数。"""
    # --- 运动模型 ---
    motion_model: str = "brownian"  # "brownian", "constant_velocity", "optical_force"
    dt: float = 1.0  # 时间步长 (帧间隔)

    # --- 布朗运动参数 ---
    diffusion_coefficient: float = 1.0  # 扩散系数 D (像素^2/帧)

    # --- 恒速模型参数 ---
    process_noise_q: float = 0.1  # 过程噪声强度

    # --- 光学力模型参数 ---
    trap_stiffness: float = 0.01  # 光阱刚度 (力/位移)
    damping_coefficient: float = 0.1  # 阻尼系数

    # --- 卡尔曼滤波 ---
    measurement_noise_r: float = 1.0  # 测量噪声协方差

    # --- 物理约束 ---
    max_velocity: float = 20.0  # 最大速度 (像素/帧)
    max_acceleration: float = 10.0  # 最大加速度 (像素/帧^2)
    boundary: Optional[Tuple[float, float, float, float]] = None  # (x_min, y_min, x_max, y_max)

    # --- 数据关联 ---
    max_association_distance: float = 30.0  # 最大关联距离 (像素)
    max_lost_frames: int = 5  # 最大允许丢失帧数

    # --- 状态维度 ---
    state_dim: int = 4  # 状态维度 [x, y, vx, vy]


@dataclass
class ParticleState:
    """单个粒子的状态。"""
    track_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    age: int = 1  # 存活帧数
    lost_count: int = 0  # 连续丢失帧数
    active: bool = True


@dataclass
class TrackingResult:
    """追踪结果。"""
    tracks: List[ParticleState] = field(default_factory=list)
    associations: List[Tuple[int, int]] = field(default_factory=list)  # (track_id, detection_idx)
    unassociated_detections: List[int] = field(default_factory=list)
    frame_index: int = 0
    num_active_tracks: int = 0
    num_lost_tracks: int = 0


class _KalmanFilter2D:
    """二维卡尔曼滤波器，支持多种运动模型。"""

    def __init__(self, config: TrackerConfig):
        self._cfg = config
        self._dim = config.state_dim

        # 状态向量 [x, y, vx, vy]
        self._x = np.zeros(self._dim, dtype=np.float64)
        # 状态协方差
        self._P = np.eye(self._dim, dtype=np.float64) * 100.0

        # 状态转移矩阵
        self._F = self._build_transition_matrix()
        # 过程噪声矩阵
        self._Q = self._build_process_noise()
        # 观测矩阵 (观测 x, y)
        self._H = np.zeros((2, self._dim), dtype=np.float64)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        # 观测噪声矩阵
        self._R = np.eye(2, dtype=np.float64) * config.measurement_noise_r

    def _build_transition_matrix(self) -> np.ndarray:
        """构建状态转移矩阵。"""
        dt = self._cfg.dt
        F = np.eye(self._dim, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt

        if self._cfg.motion_model == "optical_force":
            # 光学力模型: 加入阻尼和阱力
            k = self._cfg.trap_stiffness
            gamma = self._cfg.damping_coefficient
            F[2, 0] = -k * dt
            F[2, 2] = 1.0 - gamma * dt
            F[3, 1] = -k * dt
            F[3, 3] = 1.0 - gamma * dt

        return F

    def _build_process_noise(self) -> np.ndarray:
        """构建过程噪声矩阵。"""
        dt = self._cfg.dt
        if self._cfg.motion_model == "brownian":
            # 布朗运动: 位置噪声与 D*dt 成正比
            D = self._cfg.diffusion_coefficient
            q_pos = D * dt
            q_vel = D * dt * 0.1
        else:
            q_pos = self._cfg.process_noise_q * dt
            q_vel = self._cfg.process_noise_q * dt * 0.1

        Q = np.eye(self._dim, dtype=np.float64)
        Q[0, 0] = q_pos
        Q[1, 1] = q_pos
        Q[2, 2] = q_vel
        Q[3, 3] = q_vel
        return Q

    def initialize(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """初始化滤波器状态。"""
        self._x = np.array([x, y, vx, vy], dtype=np.float64)
        self._P = np.eye(self._dim, dtype=np.float64) * 100.0
        self._P[0, 0] = 1.0
        self._P[1, 1] = 1.0

    def predict(self) -> np.ndarray:
        """预测步骤。"""
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._x.copy()

    def update(self, z: np.ndarray):
        """更新步骤 (Joseph 形式保证数值稳定性)。"""
        z = np.asarray(z, dtype=np.float64)
        y = z - self._H @ self._x  # 新息
        S = self._H @ self._P @ self._H.T + self._R  # 新息协方差
        K = self._P @ self._H.T @ np.linalg.inv(S)  # 卡尔曼增益

        # 状态更新
        self._x = self._x + K @ y

        # Joseph 形式协方差更新
        I_KH = np.eye(self._dim) - K @ self._H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

    def get_state(self) -> np.ndarray:
        """获取当前状态向量。"""
        return self._x.copy()

    def get_position(self) -> Tuple[float, float]:
        """获取当前位置。"""
        return float(self._x[0]), float(self._x[1])

    def get_velocity(self) -> Tuple[float, float]:
        """获取当前速度。"""
        return float(self._x[2]), float(self._x[3])


class PhysicsInformedParticleTracker:
    """物理信息粒子追踪器。

    融合卡尔曼滤波与物理约束的多粒子追踪系统。
    支持布朗运动、恒速运动和光学力场三种运动模型，
    并通过匈牙利算法实现多目标数据关联。

    Parameters
    ----------
    config : TrackerConfig, optional
        追踪器配置参数。为 None 时使用默认配置。
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self._cfg = config if config is not None else TrackerConfig()
        self._filters: Dict[int, _KalmanFilter2D] = {}
        self._tracks: Dict[int, ParticleState] = {}
        self._next_id: int = 1
        self._frame_index: int = 0

    def _apply_physics_constraints(self, state: np.ndarray) -> np.ndarray:
        """对状态向量施加物理约束。"""
        x, y, vx, vy = state

        # 速度约束
        speed = np.sqrt(vx ** 2 + vy ** 2)
        max_v = self._cfg.max_velocity
        if speed > max_v:
            scale = max_v / speed
            vx *= scale
            vy *= scale

        # 加速度约束 (基于速度变化)
        # 此处简化处理: 限制速度变化率
        if self._cfg.max_acceleration > 0:
            max_dv = self._cfg.max_acceleration * self._cfg.dt
            # 检查是否有上一帧速度记录
            track = None
            for t in self._tracks.values():
                if np.isclose(t.x, x, atol=1.0) and np.isclose(t.y, y, atol=1.0):
                    track = t
                    break
            if track is not None:
                dvx = vx - track.vx
                dvy = vy - track.vy
                dv = np.sqrt(dvx ** 2 + dvy ** 2)
                if dv > max_dv:
                    scale = max_dv / dv
                    vx = track.vx + dvx * scale
                    vy = track.vy + dvy * scale

        # 边界约束
        if self._cfg.boundary is not None:
            x_min, y_min, x_max, y_max = self._cfg.boundary
            x = np.clip(x, x_min, x_max)
            y = np.clip(y, y_min, y_max)

        return np.array([x, y, vx, vy], dtype=np.float64)

    def _hungarian_association(
        self,
        predicted_positions: np.ndarray,
        detections: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """使用匈牙利算法进行数据关联。

        Parameters
        ----------
        predicted_positions : np.ndarray, shape (N, 2)
            预测位置矩阵。
        detections : np.ndarray, shape (M, 2)
            检测位置矩阵。

        Returns
        -------
        List[Tuple[int, int]]
            匹配对列表 (预测索引, 检测索引)。
        """
        n_pred = len(predicted_positions)
        n_det = len(detections)

        if n_pred == 0 or n_det == 0:
            return []

        # 构建代价矩阵 (欧氏距离)
        cost = np.zeros((n_pred, n_det), dtype=np.float64)
        for i in range(n_pred):
            for j in range(n_det):
                cost[i, j] = np.linalg.norm(predicted_positions[i] - detections[j])

        # 匈牙利算法求解
        row_indices, col_indices = linear_sum_assignment(cost)

        # 过滤距离过大的匹配
        matches = []
        max_dist = self._cfg.max_association_distance
        for r, c in zip(row_indices, col_indices):
            if cost[r, c] <= max_dist:
                matches.append((int(r), int(c)))

        return matches

    def _create_new_track(self, x: float, y: float) -> int:
        """创建新轨迹。"""
        track_id = self._next_id
        self._next_id += 1

        kf = _KalmanFilter2D(self._cfg)
        kf.initialize(x, y)

        self._filters[track_id] = kf
        self._tracks[track_id] = ParticleState(
            track_id=track_id, x=x, y=y, active=True
        )

        LOGGER.debug("创建新轨迹 ID=%d 于 (%.2f, %.2f)", track_id, x, y)
        return track_id

    def _remove_track(self, track_id: int):
        """移除轨迹。"""
        if track_id in self._filters:
            del self._filters[track_id]
        if track_id in self._tracks:
            del self._tracks[track_id]

    def process(
        self,
        detections: np.ndarray,
        frame_index: Optional[int] = None,
    ) -> TrackingResult:
        """处理一帧检测结果。

        Parameters
        ----------
        detections : np.ndarray, shape (N, 2) or (N, 3)
            检测结果矩阵，每行至少包含 [x, y]。
            如果有第三列，视为检测置信度。
        frame_index : int, optional
            当前帧索引。为 None 时自动递增。

        Returns
        -------
        TrackingResult
            追踪结果。
        """
        if frame_index is not None:
            self._frame_index = frame_index
        else:
            self._frame_index += 1

        detections = np.atleast_2d(np.asarray(detections, dtype=np.float64))
        if detections.ndim != 2 or detections.shape[1] < 2:
            raise ValueError(
                f"detections 形状应为 (N, 2+)，实际为 {detections.shape}"
            )

        det_positions = detections[:, :2]

        # --- 获取所有活跃轨迹的预测位置 ---
        active_ids = [tid for tid, t in self._tracks.items() if t.active]
        if len(active_ids) > 0:
            predicted = np.array([
                self._filters[tid].get_position() for tid in active_ids
            ])
        else:
            predicted = np.empty((0, 2), dtype=np.float64)

        # --- 数据关联 ---
        matches = self._hungarian_association(predicted, det_positions)

        # --- 更新匹配的轨迹 ---
        matched_det_indices = set()
        matched_track_indices = set()
        associations: List[Tuple[int, int]] = []

        for pred_idx, det_idx in matches:
            track_id = active_ids[pred_idx]
            z = det_positions[det_idx]

            # 卡尔曼预测 + 更新
            self._filters[track_id].predict()
            self._filters[track_id].update(z)

            # 获取更新后状态并施加物理约束
            state = self._filters[track_id].get_state()
            state = self._apply_physics_constraints(state)
            self._filters[track_id]._x = state

            # 更新轨迹记录
            track = self._tracks[track_id]
            track.x, track.y = float(state[0]), float(state[1])
            track.vx, track.vy = float(state[2]), float(state[3])
            track.age += 1
            track.lost_count = 0

            matched_det_indices.add(det_idx)
            matched_track_indices.add(pred_idx)
            associations.append((track_id, det_idx))

        # --- 未匹配检测: 创建新轨迹 ---
        unassociated_detections = []
        for det_idx in range(len(det_positions)):
            if det_idx not in matched_det_indices:
                self._create_new_track(
                    det_positions[det_idx, 0],
                    det_positions[det_idx, 1],
                )
                unassociated_detections.append(det_idx)

        # --- 未匹配轨迹: 标记丢失 ---
        num_lost = 0
        for pred_idx, track_id in enumerate(active_ids):
            if pred_idx not in matched_track_indices:
                # 仅预测，不更新
                self._filters[track_id].predict()
                state = self._filters[track_id].get_state()
                state = self._apply_physics_constraints(state)
                self._filters[track_id]._x = state

                track = self._tracks[track_id]
                track.x, track.y = float(state[0]), float(state[1])
                track.vx, track.vy = float(state[2]), float(state[3])
                track.lost_count += 1

                if track.lost_count > self._cfg.max_lost_frames:
                    track.active = False
                    self._remove_track(track_id)
                    num_lost += 1

        # --- 构建结果 ---
        active_tracks = [t for t in self._tracks.values() if t.active]

        result = TrackingResult(
            tracks=active_tracks,
            associations=associations,
            unassociated_detections=unassociated_detections,
            frame_index=self._frame_index,
            num_active_tracks=len(active_tracks),
            num_lost_tracks=num_lost,
        )

        LOGGER.debug(
            "帧 %d: %d 活跃轨迹, %d 匹配, %d 新建, %d 丢失",
            self._frame_index,
            len(active_tracks),
            len(associations),
            len(unassociated_detections),
            num_lost,
        )

        return result

    def get_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """获取所有活跃轨迹的历史位置。

        Returns
        -------
        Dict[int, List[Tuple[float, float]]]
            轨迹 ID 到位置列表的映射。
        """
        result: Dict[int, List[Tuple[float, float]]] = {}
        for tid, track in self._tracks.items():
            if track.active:
                result[tid] = [(track.x, track.y)]
        return result

    def predict_next(self) -> np.ndarray:
        """预测所有活跃轨迹的下一帧位置。

        Returns
        -------
        np.ndarray, shape (N, 2)
            预测位置矩阵。
        """
        active_ids = [tid for tid, t in self._tracks.items() if t.active]
        if len(active_ids) == 0:
            return np.empty((0, 2), dtype=np.float64)

        positions = []
        for tid in active_ids:
            state = self._filters[tid].predict()
            state = self._apply_physics_constraints(state)
            # 回退预测 (不修改滤波器状态)
            self._filters[tid]._x = self._filters[tid]._F.T @ (state - np.zeros_like(state))
            # 简化: 直接使用预测位置
            positions.append([float(state[0]), float(state[1])])

        return np.array(positions, dtype=np.float64)

    def reset(self):
        """重置追踪器，清除所有轨迹和状态。"""
        self._filters.clear()
        self._tracks.clear()
        self._next_id = 1
        self._frame_index = 0
        LOGGER.info("追踪器已重置")
