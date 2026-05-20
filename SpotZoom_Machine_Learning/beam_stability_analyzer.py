"""
光束指向稳定性分析器 (BeamStabilityAnalyzer)

灵感来源:
- ISO 11670 — 激光光束指向稳定性测量标准
- pyBeamProfiling (https://github.com/JohnBriggs/pyBeamProfiling) — 光束剖面分析
- Allan Variance — Allan 方差 (IEEE Std 952-1997, 惯性传感器噪声分析)

算法原理:
- RMS Jitter — 均方根抖动 (位置标准差)
- Peak-to-Peak Drift — 峰峰值漂移 (最大-最小位移)
- Allan Deviation — Allan 偏差 (区分不同时间尺度的噪声类型)
- Linear Drift Fit — 线性漂移拟合 (最小二乘法)
- Stability Classification — 稳定性等级分类 (基于 ISO 11670)

功能:
- 实时分析光斑位置时间序列的指向稳定性
- 计算 RMS 抖动、峰峰值漂移、Allan 偏差等关键指标
- 区分短期抖动与长期漂移
- 检测漂移趋势并估计漂移速率
- 输出标准化稳定性报告

依赖: numpy, logging
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.BeamStability")


class StabilityClass(Enum):
    """光束指向稳定性等级 (参考 ISO 11670)。"""
    EXCELLENT = "excellent"  # 优秀: RMS < 0.5 px
    GOOD = "good"            # 良好: RMS < 1.5 px
    FAIR = "fair"            # 一般: RMS < 3.0 px
    POOR = "poor"            # 较差: RMS >= 3.0 px


@dataclass
class StabilityReport:
    """光束指向稳定性分析报告。"""
    rms_jitter_px: float              # RMS 抖动 (像素)
    peak_to_peak_px: float            # 峰峰值漂移 (像素)
    allan_deviation: float            # Allan 偏差 (像素, tau=1 采样间隔)
    drift_rate_px_per_s: float        # 漂移速率 (像素/秒)
    stability_class: StabilityClass   # 稳定性等级
    total_samples: int                # 总采样数
    duration_s: float                 # 总时长 (秒)
    mean_position: Tuple[float, float]  # 平均位置 (x, y)
    linear_drift_x: float             # X 方向线性漂移系数 (px/s)
    linear_drift_y: float             # Y 方向线性漂移系数 (px/s)


class BeamStabilityAnalyzer:
    """光束指向稳定性分析器。

    对光斑位置时间序列进行统计分析，计算指向稳定性指标。
    支持实时流式更新，每帧调用 update() 即可。

    Parameters
    ----------
    max_history : int
        最大历史记录长度。超过后自动丢弃最旧数据。
    min_samples : int
        生成报告所需的最小样本数。
    rms_excellent_threshold : float
        "优秀" 稳定性等级的 RMS 阈值 (像素)。
    rms_good_threshold : float
        "良好" 稳定性等级的 RMS 阈值 (像素)。
    rms_fair_threshold : float
        "一般" 稳定性等级的 RMS 阈值 (像素)。
    """

    def __init__(
        self,
        max_history: int = 1000,
        min_samples: int = 10,
        rms_excellent_threshold: float = 0.5,
        rms_good_threshold: float = 1.5,
        rms_fair_threshold: float = 3.0,
    ):
        self.max_history = int(max_history)
        self.min_samples = int(min_samples)
        self.rms_excellent_threshold = float(rms_excellent_threshold)
        self.rms_good_threshold = float(rms_good_threshold)
        self.rms_fair_threshold = float(rms_fair_threshold)

        # 位置历史: [(x, y, timestamp), ...]
        self._positions: List[Tuple[float, float, float]] = []
        self._start_time: Optional[float] = None

    def update(self, position: Tuple[float, float]) -> None:
        """添加新的位置采样点。

        Parameters
        ----------
        position : Tuple[float, float]
            光斑中心坐标 (x, y)，像素单位。
        """
        now = time.time()
        if self._start_time is None:
            self._start_time = now

        self._positions.append((float(position[0]), float(position[1]), now))

        # 限制历史长度
        if len(self._positions) > self.max_history:
            self._positions = self._positions[-self.max_history:]

    def get_report(self) -> Optional[StabilityReport]:
        """生成当前稳定性分析报告。

        Returns
        -------
        StabilityReport or None
            稳定性报告。如果样本数不足，返回 None。
        """
        if len(self._positions) < self.min_samples:
            LOGGER.debug("BeamStability: 样本数不足 (%d < %d)，跳过报告生成",
                         len(self._positions), self.min_samples)
            return None

        xs = np.array([p[0] for p in self._positions])
        ys = np.array([p[1] for p in self._positions])
        ts = np.array([p[2] for p in self._positions])

        # 1. RMS 抖动
        rms_x = float(np.std(xs))
        rms_y = float(np.std(ys))
        rms_jitter = float(np.sqrt(rms_x ** 2 + rms_y ** 2))

        # 2. 峰峰值漂移
        ptp_x = float(np.ptp(xs))
        ptp_y = float(np.ptp(ys))
        peak_to_peak = float(np.sqrt(ptp_x ** 2 + ptp_y ** 2))

        # 3. Allan 偏差 (tau=1 采样间隔)
        allan_dev = self._compute_allan_deviation(xs, ys)

        # 4. 漂移速率 (线性拟合)
        duration = float(ts[-1] - ts[0])
        if duration < 1e-6:
            duration = 1e-6

        drift_x, drift_y = self._compute_linear_drift(xs, ys, ts)
        drift_rate = float(np.sqrt(drift_x ** 2 + drift_y ** 2))

        # 5. 稳定性等级分类
        stability_class = self._classify_stability(rms_jitter)

        # 6. 平均位置
        mean_pos = (float(np.mean(xs)), float(np.mean(ys)))

        report = StabilityReport(
            rms_jitter_px=round(rms_jitter, 4),
            peak_to_peak_px=round(peak_to_peak, 4),
            allan_deviation=round(allan_dev, 4),
            drift_rate_px_per_s=round(drift_rate, 4),
            stability_class=stability_class,
            total_samples=len(self._positions),
            duration_s=round(duration, 3),
            mean_position=mean_pos,
            linear_drift_x=round(drift_x, 4),
            linear_drift_y=round(drift_y, 4),
        )

        LOGGER.debug(
            "BeamStability: RMS=%.3f px, P2P=%.3f px, Allan=%.3f, class=%s",
            report.rms_jitter_px, report.peak_to_peak_px,
            report.allan_deviation, report.stability_class.value,
        )

        return report

    def _compute_allan_deviation(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> float:
        """计算 Allan 偏差 (tau=1 采样间隔)。

        Allan 偏差用于区分不同时间尺度的噪声成分:
        - 白噪声: Allan 偏差 ~ tau^(-1/2)
        - 随机游走: Allan 偏差 ~ tau^(+1/2)
        - 闪烁噪声: Allan 偏差 ~ 常数

        参考: IEEE Std 952-1997, Allan Variance 定义

        Parameters
        ----------
        xs, ys : np.ndarray
            X/Y 坐标序列。

        Returns
        -------
        float
            tau=1 时的 Allan 偏差 (像素)。
        """
        n = len(xs)
        if n < 3:
            return 0.0

        # 计算差分序列
        dx = np.diff(xs)
        dy = np.diff(ys)

        # 计算差分的差分 (Allan 方差的核心)
        ddx = np.diff(dx)
        ddy = np.diff(dy)

        if len(ddx) == 0:
            return 0.0

        # Allan 方差 = 0.5 * mean((x[n+2] - 2*x[n+1] + x[n])^2)
        allan_var_x = 0.5 * float(np.mean(ddx ** 2))
        allan_var_y = 0.5 * float(np.mean(ddy ** 2))

        # Allan 偏差 = sqrt(Allan 方差)
        return float(np.sqrt(allan_var_x + allan_var_y))

    @staticmethod
    def _compute_linear_drift(
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
    ) -> Tuple[float, float]:
        """通过最小二乘法拟合线性漂移趋势。

        Parameters
        ----------
        xs, ys : np.ndarray
            坐标序列。
        ts : np.ndarray
            时间戳序列。

        Returns
        -------
        Tuple[float, float]
            (drift_x, drift_y) 漂移速率 (像素/秒)。
        """
        n = len(ts)
        if n < 2:
            return (0.0, 0.0)

        # 归一化时间以避免数值问题
        t_norm = ts - ts[0]
        t_mean = np.mean(t_norm)

        # X 方向线性拟合: x = a_x * t + b_x
        # a_x = sum((t - t_mean) * (x - x_mean)) / sum((t - t_mean)^2)
        x_mean = np.mean(xs)
        y_mean = np.mean(ys)

        dt = t_norm - t_mean
        dt_sq_sum = float(np.sum(dt ** 2))

        if dt_sq_sum < 1e-12:
            return (0.0, 0.0)

        drift_x = float(np.sum(dt * (xs - x_mean)) / dt_sq_sum)
        drift_y = float(np.sum(dt * (ys - y_mean)) / dt_sq_sum)

        return (drift_x, drift_y)

    def _classify_stability(self, rms_jitter: float) -> StabilityClass:
        """根据 RMS 抖动分类稳定性等级。

        Parameters
        ----------
        rms_jitter : float
            RMS 抖动 (像素)。

        Returns
        -------
        StabilityClass
            稳定性等级。
        """
        if rms_jitter < self.rms_excellent_threshold:
            return StabilityClass.EXCELLENT
        elif rms_jitter < self.rms_good_threshold:
            return StabilityClass.GOOD
        elif rms_jitter < self.rms_fair_threshold:
            return StabilityClass.FAIR
        else:
            return StabilityClass.POOR

    def get_short_term_rms(self, window: int = 10) -> Optional[float]:
        """获取短期 (最近 N 帧) RMS 抖动。

        Parameters
        ----------
        window : int
            短期窗口大小 (帧数)。

        Returns
        -------
        float or None
            短期 RMS 抖动。如果数据不足返回 None。
        """
        if len(self._positions) < window:
            return None

        recent = self._positions[-window:]
        xs = np.array([p[0] for p in recent])
        ys = np.array([p[1] for p in recent])

        rms = float(np.sqrt(np.std(xs) ** 2 + np.std(ys) ** 2))
        return round(rms, 4)

    def get_long_term_rms(self) -> Optional[float]:
        """获取长期 (全部历史) RMS 抖动。

        Returns
        -------
        float or None
            长期 RMS 抖动。如果数据不足返回 None。
        """
        if len(self._positions) < self.min_samples:
            return None

        xs = np.array([p[0] for p in self._positions])
        ys = np.array([p[1] for p in self._positions])

        rms = float(np.sqrt(np.std(xs) ** 2 + np.std(ys) ** 2))
        return round(rms, 4)

    def get_position_history(self) -> List[Tuple[float, float, float]]:
        """获取完整位置历史记录。

        Returns
        -------
        List[Tuple[float, float, float]]
            [(x, y, timestamp), ...] 位置历史列表。
        """
        return list(self._positions)

    @property
    def sample_count(self) -> int:
        """当前累计采样数。"""
        return len(self._positions)

    def reset(self) -> None:
        """重置分析器，清除所有历史数据。"""
        self._positions.clear()
        self._start_time = None
        LOGGER.info("BeamStability: 分析器已重置")
