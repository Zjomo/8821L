"""
波前误差预测器 (WavefrontPredictor)

灵感来源:
- AOtools / HCIPy — 自适应光学波前预测与校正
- "深度学习无波前探测自适应光学" — LSTM 波前预测研究
- python-control — 预测控制 (Model Predictive Control)

算法原理:
- AR(p) 自回归模型 — 线性自回归时间序列预测
- Least Squares Fitting — 最小二乘法参数估计
- Residual Variance — 残差方差用于置信度评估
- Feedforward Compensation — 前馈补偿减少闭环延迟

功能:
- 基于历史位置误差时间序列预测未来光斑位移
- 对 X/Y 轴分别建立 AR 模型
- 提供前馈补偿向量，降低闭环控制延迟影响
- 实时评估预测置信度

依赖: numpy, logging, dataclasses (无 PyTorch/TensorFlow)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.WavefrontPredictor")


@dataclass
class PredictionStatistics:
    """预测质量统计信息。"""
    mae: float                # 平均绝对误差 (Mean Absolute Error)
    rmse: float               # 均方根误差 (Root Mean Square Error)
    prediction_gain: float    # 预测增益 (相对于朴素预测的改善比例)
    samples: int              # 参与统计的样本数


class WavefrontPredictor:
    """波前误差预测器。

    使用 AR(p) 自回归模型对光斑位置误差时间序列进行建模和预测，
    生成前馈补偿向量以降低闭环对准系统中的延迟效应。

    AR(p) 模型:
        x[t] = a_1 * x[t-1] + a_2 * x[t-2] + ... + a_p * x[t-p] + noise

    对 X 和 Y 轴分别建立独立的 AR 模型，通过最小二乘法在线拟合参数。

    Parameters
    ----------
    history_length : int
        历史数据缓冲区长度。保留最近 N 个采样点用于模型拟合。
    prediction_horizon : int
        默认预测步长 (帧数)。get_compensation() 使用此步长。
    model_order : int
        AR 模型阶数 p。典型值 2~5，值越高可捕获更复杂的动态，
        但需要更多数据且容易过拟合。
    """

    def __init__(
        self,
        history_length: int = 50,
        prediction_horizon: int = 5,
        model_order: int = 3,
    ):
        self.history_length = int(history_length)
        self.prediction_horizon = int(prediction_horizon)
        self.model_order = int(model_order)

        if self.model_order < 1:
            raise ValueError(f"model_order 必须 >= 1，当前值: {self.model_order}")
        if self.history_length < self.model_order + 2:
            raise ValueError(
                f"history_length ({self.history_length}) 必须 >= "
                f"model_order + 2 ({self.model_order + 2})"
            )

        # 历史误差缓冲区
        self._error_x: Deque[float] = deque(maxlen=self.history_length)
        self._error_y: Deque[float] = deque(maxlen=self.history_length)
        self._timestamps: Deque[float] = deque(maxlen=self.history_length)

        # AR 模型参数 (X/Y 轴)
        self._ar_coeffs_x: Optional[np.ndarray] = None
        self._ar_coeffs_y: Optional[np.ndarray] = None

        # 残差统计
        self._residual_var_x: float = 1.0
        self._residual_var_y: float = 1.0
        self._signal_var_x: float = 1.0
        self._signal_var_y: float = 1.0

        # 预测质量跟踪
        self._prediction_errors_x: Deque[float] = deque(maxlen=100)
        self._prediction_errors_y: Deque[float] = deque(maxlen=100)
        self._last_prediction_x: Optional[float] = None
        self._last_prediction_y: Optional[float] = None

        # 补偿增益 (0~1)，用于缩放前馈补偿量
        self._compensation_gain: float = 0.8
        self._min_samples_for_fit: int = self.model_order + 5

        LOGGER.info(
            "WavefrontPredictor: 初始化完成 (history=%d, horizon=%d, order=%d)",
            self.history_length, self.prediction_horizon, self.model_order,
        )

    def update(self, error_x: float, error_y: float, timestamp: Optional[float] = None) -> None:
        """添加新的误差观测值并更新模型。

        Parameters
        ----------
        error_x : float
            X 方向位置误差 (像素)。
        error_y : float
            Y 方向位置误差 (像素)。
        timestamp : float or None
            时间戳。如果为 None，使用系统当前时间。
        """
        if timestamp is None:
            timestamp = time.time()

        # 在添加新数据前，记录上一时刻预测的误差
        self._track_prediction_error(float(error_x), float(error_y))

        # 添加新观测
        self._error_x.append(float(error_x))
        self._error_y.append(float(error_y))
        self._timestamps.append(float(timestamp))

        # 尝试拟合/更新 AR 模型
        if len(self._error_x) >= self._min_samples_for_fit:
            self._fit_models()

        # 模型就绪后，自动生成一步预测缓存，用于下次 update 时评估预测误差
        if self.is_model_ready:
            pred_x = self._predict_axis(self._error_x, self._ar_coeffs_x, 1)
            pred_y = self._predict_axis(self._error_y, self._ar_coeffs_y, 1)
            self._last_prediction_x = pred_x
            self._last_prediction_y = pred_y

        LOGGER.debug(
            "WavefrontPredictor: 更新 error=(%.3f, %.3f), 缓冲区长度=%d",
            error_x, error_y, len(self._error_x),
        )

    def predict(self, steps_ahead: int = 1) -> Tuple[float, float]:
        """预测未来若干步的误差。

        Parameters
        ----------
        steps_ahead : int
            向前预测的步数。默认为 1。

        Returns
        -------
        Tuple[float, float]
            预测的 (error_x, error_y)。如果模型尚未拟合，
            返回 (0.0, 0.0)。
        """
        if steps_ahead < 1:
            steps_ahead = 1

        pred_x = self._predict_axis(self._error_x, self._ar_coeffs_x, steps_ahead)
        pred_y = self._predict_axis(self._error_y, self._ar_coeffs_y, steps_ahead)

        # 缓存预测结果用于后续误差跟踪
        if steps_ahead == 1:
            self._last_prediction_x = pred_x
            self._last_prediction_y = pred_y

        LOGGER.debug(
            "WavefrontPredictor: 预测 %d 步 ahead -> (%.4f, %.4f)",
            steps_ahead, pred_x, pred_y,
        )

        return (pred_x, pred_y)

    def get_compensation(self) -> Tuple[float, float]:
        """获取前馈补偿向量。

        补偿量 = -predicted_error * gain

        将此补偿量叠加到控制输出上，可抵消延迟导致的误差。

        Returns
        -------
        Tuple[float, float]
            补偿向量 (comp_x, comp_y)。
        """
        pred_x, pred_y = self.predict(steps_ahead=self.prediction_horizon)

        confidence = self.get_prediction_confidence()
        # 置信度越低，补偿越保守
        effective_gain = self._compensation_gain * confidence

        comp_x = -pred_x * effective_gain
        comp_y = -pred_y * effective_gain

        LOGGER.debug(
            "WavefrontPredictor: 补偿向量 (%.4f, %.4f), gain=%.3f, confidence=%.3f",
            comp_x, comp_y, effective_gain, confidence,
        )

        return (comp_x, comp_y)

    def get_prediction_confidence(self) -> float:
        """获取当前预测置信度。

        置信度基于信号信噪比 (SNR) 和数据充分性:
        - 数据不足时置信度为 0
        - 残差方差相对于信号方差越小，置信度越高

        Returns
        -------
        float
            置信度 [0, 1]。
        """
        if len(self._error_x) < self._min_samples_for_fit:
            return 0.0

        if self._ar_coeffs_x is None or self._ar_coeffs_y is None:
            return 0.0

        # X 轴 SNR
        if self._signal_var_x > 1e-12:
            snr_x = 1.0 - min(self._residual_var_x / self._signal_var_x, 1.0)
        else:
            snr_x = 0.0

        # Y 轴 SNR
        if self._signal_var_y > 1e-12:
            snr_y = 1.0 - min(self._residual_var_y / self._signal_var_y, 1.0)
        else:
            snr_y = 0.0

        # 综合置信度 (取两轴中较低者)
        confidence = float(np.clip(min(snr_x, snr_y), 0.0, 1.0))

        # 数据量加权: 数据越多越可信
        data_ratio = min(
            len(self._error_x) / (self.history_length * 0.5), 1.0
        )
        confidence *= data_ratio

        return round(confidence, 4)

    def reset(self) -> None:
        """重置预测器，清除所有历史数据和模型参数。"""
        self._error_x.clear()
        self._error_y.clear()
        self._timestamps.clear()
        self._ar_coeffs_x = None
        self._ar_coeffs_y = None
        self._residual_var_x = 1.0
        self._residual_var_y = 1.0
        self._signal_var_x = 1.0
        self._signal_var_y = 1.0
        self._prediction_errors_x.clear()
        self._prediction_errors_y.clear()
        self._last_prediction_x = None
        self._last_prediction_y = None

        LOGGER.info("WavefrontPredictor: 预测器已重置")

    def get_statistics(self) -> dict:
        """获取预测器统计信息。

        Returns
        -------
        dict
            包含以下键的字典:
            - 'x': PredictionStatistics (X 轴)
            - 'y': PredictionStatistics (Y 轴)
            - 'confidence': float (当前置信度)
            - 'model_fitted': bool (模型是否已拟合)
            - 'buffer_usage': float (缓冲区使用率)
            - 'ar_coeffs_x': list (X 轴 AR 系数)
            - 'ar_coeffs_y': list (Y 轴 AR 系数)
        """
        stats_x = self._compute_axis_statistics(self._prediction_errors_x)
        stats_y = self._compute_axis_statistics(self._prediction_errors_y)

        # 计算预测增益 (相对于朴素预测: 假设误差不变)
        gain_x = self._compute_prediction_gain(self._prediction_errors_x, self._error_x)
        gain_y = self._compute_prediction_gain(self._prediction_errors_y, self._error_y)
        overall_gain = max(gain_x, gain_y)

        return {
            'x': PredictionStatistics(
                mae=stats_x['mae'],
                rmse=stats_x['rmse'],
                prediction_gain=gain_x,
                samples=stats_x['samples'],
            ),
            'y': PredictionStatistics(
                mae=stats_y['mae'],
                rmse=stats_y['rmse'],
                prediction_gain=gain_y,
                samples=stats_y['samples'],
            ),
            'confidence': self.get_prediction_confidence(),
            'model_fitted': self._ar_coeffs_x is not None,
            'buffer_usage': round(len(self._error_x) / max(self.history_length, 1), 4),
            'ar_coeffs_x': self._ar_coeffs_x.tolist() if self._ar_coeffs_x is not None else [],
            'ar_coeffs_y': self._ar_coeffs_y.tolist() if self._ar_coeffs_y is not None else [],
            'overall_prediction_gain': round(overall_gain, 4),
        }

    # ======================== 内部方法 ========================

    def _fit_models(self) -> None:
        """使用最小二乘法拟合 AR(p) 模型。

        对 X 和 Y 轴分别拟合独立的 AR 模型:
            x[t] = a_1*x[t-1] + a_2*x[t-2] + ... + a_p*x[t-p]

        构造回归矩阵:
            X = [[x[t-1], x[t-2], ..., x[t-p]],
                 [x[t-2], x[t-3], ..., x[t-p-1]],
                 ...]
            y = [x[t], x[t-1], ...]

        求解: coeffs = (X^T X)^{-1} X^T y
        """
        result_x = self._fit_ar_model(list(self._error_x))
        result_y = self._fit_ar_model(list(self._error_y))

        if result_x is not None:
            self._ar_coeffs_x, self._residual_var_x = result_x
        if result_y is not None:
            self._ar_coeffs_y, self._residual_var_y = result_y

        # 更新信号方差
        if len(self._error_x) > 1:
            arr_x = np.array(list(self._error_x))
            arr_y = np.array(list(self._error_y))
            self._signal_var_x = float(np.var(arr_x))
            self._signal_var_y = float(np.var(arr_y))

        LOGGER.debug(
            "WavefrontPredictor: 模型拟合完成, coeffs_x=%s, coeffs_y=%s",
            self._ar_coeffs_x, self._ar_coeffs_y,
        )

    def _fit_ar_model(self, data: list) -> Optional[Tuple[np.ndarray, float]]:
        """对单轴数据拟合 AR(p) 模型。

        Parameters
        ----------
        data : list
            标量时间序列数据。

        Returns
        -------
        Tuple[np.ndarray, float] or None
            (AR 系数 [a_1, ..., a_p], 残差方差)。数据不足时返回 None。
        """
        n = len(data)
        p = self.model_order

        if n < p + 2:
            return None

        arr = np.array(data, dtype=np.float64)

        # 构造回归矩阵 X 和目标向量 y
        # X[i, j] = arr[n - 1 - i - 1 - j]  (延迟 j+1 步)
        # y[i] = arr[n - 1 - i]
        num_equations = n - p
        X = np.zeros((num_equations, p), dtype=np.float64)
        y = np.zeros(num_equations, dtype=np.float64)

        for i in range(num_equations):
            for j in range(p):
                X[i, j] = arr[n - 1 - i - 1 - j]
            y[i] = arr[n - 1 - i]

        # 最小二乘求解: coeffs = (X^T X)^{-1} X^T y
        try:
            XtX = X.T @ X
            Xty = X.T @ y

            # 添加正则化项防止奇异矩阵
            reg = 1e-6 * np.eye(p, dtype=np.float64)
            XtX_reg = XtX + reg

            coeffs = np.linalg.solve(XtX_reg, Xty)

            # 计算残差方差
            residuals = y - X @ coeffs
            residual_var = float(np.var(residuals)) if len(residuals) > 1 else 1.0
            residual_var = max(residual_var, 1e-12)

            return (coeffs, residual_var)

        except np.linalg.LinAlgError:
            LOGGER.warning("WavefrontPredictor: AR 模型拟合失败 (矩阵奇异)")
            return None

    def _predict_axis(
        self,
        history: Deque[float],
        coeffs: Optional[np.ndarray],
        steps_ahead: int,
    ) -> float:
        """对单轴进行多步预测。

        使用递归预测: 先预测 t+1，再用 t+1 预测 t+2，依此类推。

        Parameters
        ----------
        history : Deque[float]
            历史数据队列。
        coeffs : np.ndarray or None
            AR 系数。
        steps_ahead : int
            预测步数。

        Returns
        -------
        float
            预测值。模型未就绪时返回 0.0。
        """
        if coeffs is None or len(history) < self.model_order:
            return 0.0

        p = len(coeffs)
        # 取最近 p 个值作为初始窗口
        window = list(history)[-p:]
        if len(window) < p:
            return 0.0

        prediction = 0.0
        for _ in range(steps_ahead):
            # x[t] = sum(a_i * x[t-i])
            prediction = float(np.dot(coeffs, window[-p:]))
            # 滑动窗口: 丢弃最旧值，加入预测值
            window.append(prediction)

        return prediction

    def _track_prediction_error(self, actual_x: float, actual_y: float) -> None:
        """记录预测误差用于统计评估。

        将上一时刻的预测值与当前实际值比较，记录误差。

        Parameters
        ----------
        actual_x, actual_y : float
            当前实际误差值。
        """
        if self._last_prediction_x is not None:
            err_x = abs(actual_x - self._last_prediction_x)
            self._prediction_errors_x.append(err_x)

        if self._last_prediction_y is not None:
            err_y = abs(actual_y - self._last_prediction_y)
            self._prediction_errors_y.append(err_y)

    def _compute_axis_statistics(self, errors: Deque[float]) -> dict:
        """计算单轴的预测误差统计。

        Parameters
        ----------
        errors : Deque[float]
            预测绝对误差队列。

        Returns
        -------
        dict
            包含 'mae', 'rmse', 'samples' 的字典。
        """
        if len(errors) == 0:
            return {'mae': 0.0, 'rmse': 0.0, 'samples': 0}

        arr = np.array(list(errors), dtype=np.float64)
        mae = float(np.mean(arr))
        rmse = float(np.sqrt(np.mean(arr ** 2)))

        return {
            'mae': round(mae, 6),
            'rmse': round(rmse, 6),
            'samples': len(errors),
        }

    def _compute_prediction_gain(
        self,
        prediction_errors: Deque[float],
        actual_errors: Deque[float],
    ) -> float:
        """计算预测增益。

        预测增益 = 1 - (预测误差 / 朴素预测误差)

        朴素预测假设下一时刻误差等于当前误差，即朴素预测误差 = |x[t] - x[t-1]|。
        如果增益 > 0，说明 AR 模型优于朴素预测。

        Parameters
        ----------
        prediction_errors : Deque[float]
            AR 模型的预测绝对误差。
        actual_errors : Deque[float]
            实际误差时间序列。

        Returns
        -------
        float
            预测增益。数据不足时返回 0.0。
        """
        if len(prediction_errors) < 5 or len(actual_errors) < 2:
            return 0.0

        # AR 模型的平均预测误差
        pred_list = list(prediction_errors)
        ar_mae = float(np.mean(pred_list))

        # 朴素预测的平均误差: |x[t] - x[t-1]|
        actual_list = list(actual_errors)
        naive_errors = [
            abs(actual_list[i] - actual_list[i - 1])
            for i in range(1, len(actual_list))
        ]

        # 对齐长度: 取最后 len(pred_list) 个朴素误差
        n_compare = min(len(pred_list), len(naive_errors))
        if n_compare < 5:
            return 0.0

        naive_mae = float(np.mean(naive_errors[-n_compare:]))

        if naive_mae < 1e-12:
            return 0.0

        gain = 1.0 - (ar_mae / naive_mae)
        return round(max(gain, 0.0), 4)

    @property
    def is_model_ready(self) -> bool:
        """模型是否已拟合就绪。"""
        return self._ar_coeffs_x is not None and self._ar_coeffs_y is not None

    @property
    def sample_count(self) -> int:
        """当前缓冲区中的样本数。"""
        return len(self._error_x)
