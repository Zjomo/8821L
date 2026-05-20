"""
光流法光斑运动追踪器 (OpticalFlowTracker)

灵感来源:
- Lucas-Kanade Optical Flow (B. Lucas & T. Kanade, 1981) — 稀疏光流
- Farneback Dense Optical Flow (G. Farneback, 2003) — 密集光流
- OpenCV cv2.calcOpticalFlowPyrLK — 金字塔 Lucas-Kanade 实现
- OpenCV cv2.calcOpticalFlowFarneback — Farneback 密集光流实现

算法原理:
- Pyramid Lucas-Kanade — 多尺度稀疏光流 (特征点追踪)
- Dense Optical Flow — 全帧密集光流场 (运动补偿)
- Vibration Detection — 振动检测 (基于光流幅值统计)
- Motion Compensation — 运动补偿向量生成

功能:
- 使用稀疏光流追踪光斑特征点在帧间的运动
- 使用密集光流生成全局运动场用于运动补偿
- 检测异常振动事件
- 输出像素级位移向量

依赖: numpy, opencv-python (cv2), logging
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("SpotZoom.OpticalFlow")


@dataclass
class FlowResult:
    """光流追踪结果。"""
    displacement: Tuple[float, float]    # 总位移 (dx, dy) 像素
    magnitude: float                     # 位移幅值 (像素)
    flow_field: List[Tuple[float, float, float, float]]  # 稀疏光流点 (x, y, dx, dy)
    mean_flow: float                     # 平均光流幅值
    max_flow: float                      # 最大光流幅值
    is_vibration: bool                   # 是否检测到振动
    flow_angle_deg: float                # 主运动方向角 (度)
    frame_number: int                    # 帧号


class OpticalFlowTracker:
    """光流法光斑运动追踪器。

    结合稀疏光流 (Lucas-Kanade) 和密集光流 (Farneback) 进行帧间运动分析。

    Parameters
    ----------
    win_size : int
        Lucas-Kanade 搜索窗口大小。
    max_level : int
        金字塔最大层数。
    criteria_eps : float
        LK 迭代终止误差阈值。
    criteria_max_count : int
        LK 最大迭代次数。
    vibration_threshold : float
        振动检测阈值 (像素)。平均光流幅值超过此值视为振动。
    vibration_history_length : int
        振动判断所需的历史帧数。
    dense_flow_scale : float
        密集光流缩放因子 (降低分辨率以提升速度)。
    min_feature_points : int
        最少特征点数 (不足时使用密集光流)。
    """

    def __init__(
        self,
        win_size: int = 21,
        max_level: int = 3,
        criteria_eps: float = 0.01,
        criteria_max_count: int = 30,
        vibration_threshold: float = 2.0,
        vibration_history_length: int = 5,
        dense_flow_scale: float = 0.25,
        min_feature_points: int = 4,
    ):
        self.win_size = int(win_size)
        self.max_level = int(max_level)
        self.criteria_eps = float(criteria_eps)
        self.criteria_max_count = int(criteria_max_count)
        self.vibration_threshold = float(vibration_threshold)
        self.vibration_history_length = int(vibration_history_length)
        self.dense_flow_scale = float(dense_flow_scale)
        self.min_feature_points = int(min_feature_points)

        # LK 光流参数
        self._lk_params = dict(
            winSize=(self.win_size, self.win_size),
            maxLevel=self.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      self.criteria_max_count, self.criteria_eps),
        )

        # Farneback 密集光流参数
        self._fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # 状态
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._flow_history: List[float] = []  # 历史光流幅值
        self._last_result: Optional[FlowResult] = None

    def update(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> FlowResult:
        """用前后两帧计算光流并返回运动分析结果。

        Parameters
        ----------
        prev_frame : np.ndarray
            前一帧图像 (BGR 或灰度)。
        curr_frame : np.ndarray
            当前帧图像 (BGR 或灰度)。

        Returns
        -------
        FlowResult
            光流追踪结果。
        """
        self._frame_count += 1

        # 转灰度
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame.copy()

        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame.copy()

        # 尝试稀疏光流
        sparse_result = self._compute_sparse_flow(prev_gray, curr_gray)

        if sparse_result is not None and len(sparse_result[0]) >= self.min_feature_points:
            # 稀疏光流成功
            good_old, good_new, status = sparse_result
            dx_total, dy_total, mean_mag, max_mag, flow_field = \
                self._analyze_sparse_flow(good_old, good_new)
        else:
            # 回退到密集光流
            LOGGER.debug("OpticalFlow: 特征点不足，回退到密集光流")
            dx_total, dy_total, mean_mag, max_mag, flow_field = \
                self._compute_dense_flow(prev_gray, curr_gray)

        magnitude = float(np.sqrt(dx_total ** 2 + dy_total ** 2))

        # 计算主运动方向角
        flow_angle = float(np.degrees(np.arctan2(dy_total, dx_total))) if magnitude > 1e-6 else 0.0

        # 振动检测
        self._flow_history.append(mean_mag)
        if len(self._flow_history) > self.vibration_history_length:
            self._flow_history = self._flow_history[-self.vibration_history_length:]

        is_vibration = self._check_vibration()

        result = FlowResult(
            displacement=(round(dx_total, 3), round(dy_total, 3)),
            magnitude=round(magnitude, 3),
            flow_field=flow_field,
            mean_flow=round(mean_mag, 3),
            max_flow=round(max_mag, 3),
            is_vibration=is_vibration,
            flow_angle_deg=round(flow_angle, 1),
            frame_number=self._frame_count,
        )

        self._last_result = result

        LOGGER.debug(
            "OpticalFlow: frame=%d, disp=(%.2f, %.2f), mag=%.2f, vibration=%s",
            self._frame_count, dx_total, dy_total, magnitude, is_vibration,
        )

        return result

    def _compute_sparse_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """使用 Lucas-Kanade 金字塔光流计算稀疏光流。

        Parameters
        ----------
        prev_gray, curr_gray : np.ndarray
            前后帧灰度图。

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            (prev_points, next_points, status) 或 None (如果特征点不足)。
        """
        # 检测 Shi-Tomasi 角点
        corners = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
        )

        if corners is None or len(corners) < self.min_feature_points:
            return None

        # 计算 LK 光流
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, corners, None, **self._lk_params,
        )

        status = status.flatten()

        # 筛选成功追踪的点并重塑为 (N, 2)
        good_old = corners[status == 1].reshape(-1, 2)
        good_new = next_points[status == 1].reshape(-1, 2)

        if len(good_old) < self.min_feature_points:
            return None

        return (good_old, good_new, status)

    @staticmethod
    def _analyze_sparse_flow(
        prev_points: np.ndarray,
        next_points: np.ndarray,
    ) -> Tuple[float, float, float, float, List[Tuple[float, float, float, float]]]:
        """分析稀疏光流结果。

        Parameters
        ----------
        prev_points, next_points : np.ndarray
            前后帧对应点坐标。

        Returns
        -------
        Tuple[float, float, float, float, List[Tuple[float, float, float, float]]]
            (dx_total, dy_total, mean_magnitude, max_magnitude, flow_field)
        """
        diffs = next_points - prev_points
        magnitudes = np.sqrt(np.sum(diffs ** 2, axis=1))

        # 中值位移 (鲁棒估计，排除异常值)
        dx_total = float(np.median(diffs[:, 0]))
        dy_total = float(np.median(diffs[:, 1]))
        mean_mag = float(np.mean(magnitudes))
        max_mag = float(np.max(magnitudes))

        # 构建光流场列表
        flow_field = []
        for i in range(len(prev_points)):
            flow_field.append((
                float(prev_points[i, 0]),
                float(prev_points[i, 1]),
                float(diffs[i, 0]),
                float(diffs[i, 1]),
            ))

        return (dx_total, dy_total, mean_mag, max_mag, flow_field)

    def _compute_dense_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> Tuple[float, float, float, float, List[Tuple[float, float, float, float]]]:
        """使用 Farneback 密集光流计算全局运动。

        Parameters
        ----------
        prev_gray, curr_gray : np.ndarray
            前后帧灰度图。

        Returns
        -------
        Tuple[float, float, float, float, List[Tuple[float, float, float, float]]]
            (dx_total, dy_total, mean_magnitude, max_magnitude, flow_field)
        """
        # 缩小图像以加速密集光流计算
        h, w = prev_gray.shape
        new_w = max(1, int(w * self.dense_flow_scale))
        new_h = max(1, int(h * self.dense_flow_scale))

        prev_small = cv2.resize(prev_gray, (new_w, new_h))
        curr_small = cv2.resize(curr_gray, (new_w, new_h))

        # 计算 Farneback 密集光流
        flow = cv2.calcOpticalFlowFarneback(
            prev_small, curr_small, None, **self._fb_params,
        )

        # 流场幅值和方向
        magnitudes = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

        # 中值位移
        dx_total = float(np.median(flow[:, :, 0])) / self.dense_flow_scale
        dy_total = float(np.median(flow[:, :, 1])) / self.dense_flow_scale
        mean_mag = float(np.mean(magnitudes)) / self.dense_flow_scale
        max_mag = float(np.max(magnitudes)) / self.dense_flow_scale

        # 采样稀疏光流点 (每隔若干像素取一个)
        flow_field = []
        step = max(1, int(10 / self.dense_flow_scale))
        for y in range(0, new_h, step):
            for x in range(0, new_w, step):
                fx = float(flow[y, x, 0]) / self.dense_flow_scale
                fy = float(flow[y, x, 1]) / self.dense_flow_scale
                flow_field.append((
                    float(x / self.dense_flow_scale),
                    float(y / self.dense_flow_scale),
                    fx,
                    fy,
                ))

        return (dx_total, dy_total, mean_mag, max_mag, flow_field)

    def _check_vibration(self) -> bool:
        """基于历史光流幅值判断是否发生振动。

        振动判定条件:
        1. 最近若干帧的平均光流幅值超过阈值
        2. 光流幅值的标准差较大 (表示方向在变化，非单向运动)

        Returns
        -------
        bool
            是否检测到振动。
        """
        if len(self._flow_history) < self.vibration_history_length:
            return False

        recent = np.array(self._flow_history[-self.vibration_history_length:])
        mean_mag = float(np.mean(recent))
        std_mag = float(np.std(recent))

        # 条件 1: 平均幅值超过阈值
        magnitude_exceeded = mean_mag > self.vibration_threshold

        # 条件 2: 幅值波动较大 (方向在变化)
        cv_ratio = std_mag / max(mean_mag, 1e-6)
        direction_changing = cv_ratio > 0.3

        return bool(magnitude_exceeded and direction_changing)

    def detect_vibration(self) -> bool:
        """获取当前振动检测状态。

        Returns
        -------
        bool
            是否检测到振动。
        """
        if self._last_result is not None:
            return self._last_result.is_vibration
        return False

    def get_compensation_vector(self) -> Optional[Tuple[float, float]]:
        """获取运动补偿向量 (用于反向补偿)。

        Returns
        -------
        Optional[Tuple[float, float]]
            补偿向量 (-dx, -dy)。如果无有效结果返回 None。
        """
        if self._last_result is not None:
            dx, dy = self._last_result.displacement
            return (-dx, -dy)
        return None

    @property
    def frame_number(self) -> int:
        """当前帧号。"""
        return self._frame_count

    @property
    def last_result(self) -> Optional[FlowResult]:
        """最近一次光流结果。"""
        return self._last_result

    def reset(self) -> None:
        """重置追踪器状态。"""
        self._prev_gray = None
        self._prev_points = None
        self._frame_count = 0
        self._flow_history.clear()
        self._last_result = None
        LOGGER.info("OpticalFlow: 追踪器已重置")
