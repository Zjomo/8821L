"""
拉普拉斯自动对焦 (Laplacian Autofocus)

参考开源项目:
  - IRIS Autofocus by bunnie studios (https://github.com/bunnie/iris-audio):
    基于拉普拉斯方差的自适应对焦算法

核心思想:
  从 IRIS 的拉普拉斯方差焦点评估中借鉴，实现一个鲁棒的自动对焦系统。
  通过拉普拉斯算子计算图像高频含量作为焦点评分，并使用曲线拟合
  找到最佳焦点位置。

  在 SpotZoom 场景中:
  - 焦点评分 → 拉普拉斯方差 (值越大越清晰)
  - 焦点搜索 → 在 Z 轴范围内扫描找到最大评分
  - 稳定性处理 → 高斯模糊去除亮斑干扰、等待机械稳定

创新点:
  1. 多尺度拉普拉斯方差评分
  2. 高斯模糊预处理去除亮斑干扰
  3. 二次/高斯曲线拟合精确定位焦点
  4. 机械稳定性检测与等待
  5. 自适应搜索策略 (粗搜 + 精搜)

纯 numpy + cv2 实现，无外部依赖。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AutofocusConfig:
    """自动对焦配置。"""
    # 搜索范围 (Z 位置单位，如步数或微米)
    search_range: Tuple[float, float] = (-500.0, 500.0)
    # 粗搜步长
    coarse_step: float = 50.0
    # 精搜步长
    fine_step: float = 5.0
    # 精搜范围 (粗搜最佳位置附近的范围)
    fine_range: float = 100.0
    # 拉普拉斯核大小
    laplacian_kernel_size: int = 3
    # 高斯模糊核大小 (0 = 不模糊)
    gaussian_blur_size: int = 5
    # 高斯模糊 sigma
    gaussian_blur_sigma: float = 1.5
    # 曲线拟合方法: "quadratic", "gaussian"
    fit_method: str = "gaussian"
    # 机械稳定等待时间 (秒)
    settling_time: float = 0.2
    # 稳定性判断阈值 (评分变化率)
    stability_threshold: float = 0.02
    # 稳定性检查窗口大小
    stability_window: int = 5
    # ROI 中心 (相对于图像中心)
    roi_center: Tuple[int, int] = (0, 0)
    # ROI 大小 (0 = 全图)
    roi_size: Tuple[int, int] = (0, 0)
    # 多尺度分析使能
    multiscale_enabled: bool = True
    # 多尺度金字塔层数
    multiscale_levels: int = 3
    # 最大搜索时间 (秒)
    max_search_time: float = 60.0


@dataclass
class FocusResult:
    """对焦结果。"""
    # 最佳焦点位置
    best_position: float = 0.0
    # 最佳焦点评分
    best_score: float = 0.0
    # 搜索到的所有位置
    positions: List[float] = field(default_factory=list)
    # 对应的评分
    scores: List[float] = field(default_factory=list)
    # 拟合曲线参数
    fit_params: Tuple[float, ...] = field(default_factory=tuple)
    # 是否成功
    success: bool = False
    # 搜索耗时 (秒)
    elapsed_time: float = 0.0
    # 搜索阶段: "coarse", "fine", "done"
    search_phase: str = "done"


class LaplacianAutofocus:
    """拉普拉斯自动对焦器。

    基于拉普拉斯方差焦点评估的自动对焦系统，支持粗搜+精搜
    两阶段策略和曲线拟合精确定位。

    Parameters
    ----------
    config : AutofocusConfig
        对焦配置参数。

    References
    ----------
    .. [1] IRIS Autofocus by bunnie studios:
           https://github.com/bunnie/iris-audio
    .. [2] Pertuz, S. et al. (2013). "Analysis of focus measure
           operators for shape-from-focus." Pattern Recognition, 46(5).
    .. [3] Sun, Y. et al. (2004). " Autofocusing in computer
           microscopy." Microscopy Research and Technique, 65(3).
    """

    def __init__(self, config: Optional[AutofocusConfig] = None) -> None:
        self.config = config or AutofocusConfig()
        self._score_history: List[float] = []
        self._position_history: List[float] = []
        self._search_start_time: float = 0.0

        logger.info(
            f"LaplacianAutofocus 初始化: "
            f"范围=[{self.config.search_range}], "
            f"粗搜步长={self.config.coarse_step}, "
            f"精搜步长={self.config.fine_step}"
        )

    def compute_focus_score(self, image: np.ndarray) -> float:
        """计算焦点评分 (拉普拉斯方差)。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (灰度或彩色)。

        Returns
        -------
        float
            焦点评分 (越大越清晰)。
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.astype(np.float64)

        # ROI 裁剪
        gray = self._extract_roi(gray)

        # 高斯模糊预处理 (去除亮斑干扰)
        if self.config.gaussian_blur_size > 0:
            gray = cv2.GaussianBlur(
                gray,
                (self.config.gaussian_blur_size, self.config.gaussian_blur_size),
                self.config.gaussian_blur_sigma,
            )

        # 多尺度评分
        if self.config.multiscale_enabled:
            score = self._multiscale_score(gray)
        else:
            score = self._single_scale_score(gray)

        return float(score)

    def find_optimal_focus(
        self,
        positions: List[float],
        scores: List[float],
    ) -> FocusResult:
        """从已采集的数据中找到最佳焦点。

        Parameters
        ----------
        positions : List[float]
            Z 位置列表。
        scores : List[float]
            对应的焦点评分列表。

        Returns
        -------
        FocusResult
            对焦结果。
        """
        if len(positions) < 3:
            logger.warning("至少需要 3 个数据点进行曲线拟合")
            best_idx = int(np.argmax(scores))
            return FocusResult(
                best_position=positions[best_idx],
                best_score=scores[best_idx],
                positions=positions,
                scores=scores,
                success=len(positions) > 0,
            )

        pos_arr = np.array(positions)
        score_arr = np.array(scores)

        # 曲线拟合
        if self.config.fit_method == "gaussian":
            params = self._fit_gaussian(pos_arr, score_arr)
        else:
            params = self._fit_quadratic(pos_arr, score_arr)

        # 从拟合曲线找峰值
        if self.config.fit_method == "gaussian":
            # 高斯拟合: A * exp(-(x-mu)^2 / (2*sigma^2)) + C
            if len(params) >= 4:
                best_pos = float(params[1])  # mu
                best_score = float(params[0] + params[3])  # A + C
            else:
                best_idx = int(np.argmax(scores))
                best_pos = positions[best_idx]
                best_score = scores[best_idx]
        else:
            # 二次拟合: ax^2 + bx + c
            if len(params) == 3 and params[0] < 0:
                best_pos = float(-params[1] / (2 * params[0]))
                best_score = float(params[2] - params[1] ** 2 / (4 * params[0]))
            else:
                best_idx = int(np.argmax(scores))
                best_pos = positions[best_idx]
                best_score = scores[best_idx]

        result = FocusResult(
            best_position=best_pos,
            best_score=best_score,
            positions=positions,
            scores=scores,
            fit_params=tuple(float(p) for p in params),
            success=True,
        )

        logger.info(
            f"最佳焦点: 位置={best_pos:.2f}, "
            f"评分={best_score:.2f}, "
            f"拟合参数={params}"
        )
        return result

    def scan_focus_range(
        self,
        image_fn: callable,
        phase: str = "coarse",
    ) -> FocusResult:
        """扫描焦点范围。

        Parameters
        ----------
        image_fn : callable
            图像获取函数，签名为 fn(z_position: float) -> np.ndarray。
        phase : str
            搜索阶段: "coarse" 或 "fine"。

        Returns
        -------
        FocusResult
            扫描结果。
        """
        self._search_start_time = time.time()

        if phase == "coarse":
            step = self.config.coarse_step
            z_start, z_end = self.config.search_range
        elif phase == "fine":
            if not self._position_history:
                logger.error("精搜需要先进行粗搜")
                return FocusResult(success=False)
            coarse_best = self._position_history[np.argmax(self._score_history)]
            step = self.config.fine_step
            z_start = coarse_best - self.config.fine_range / 2
            z_end = coarse_best + self.config.fine_range / 2
        else:
            logger.error(f"未知搜索阶段: {phase}")
            return FocusResult(success=False)

        positions = []
        scores = []

        z = z_start
        while z <= z_end:
            # 超时检查
            if time.time() - self._search_start_time > self.config.max_search_time:
                logger.warning("对焦搜索超时")
                break

            # 获取图像
            try:
                image = image_fn(z)
            except Exception as e:
                logger.error(f"获取图像失败 (z={z}): {e}")
                z += step
                continue

            # 等待机械稳定
            time.sleep(self.config.settling_time)

            # 计算焦点评分
            score = self.compute_focus_score(image)
            positions.append(z)
            scores.append(score)

            self._score_history.append(score)
            self._position_history.append(z)

            logger.debug(f"z={z:.1f}, score={score:.2f}")
            z += step

        result = self.find_optimal_focus(positions, scores)
        result.search_phase = phase
        result.elapsed_time = time.time() - self._search_start_time

        return result

    def is_stable(self, score: float) -> bool:
        """判断系统是否稳定。

        Parameters
        ----------
        score : float
            当前焦点评分。

        Returns
        -------
        bool
            是否稳定。
        """
        self._score_history.append(score)
        window = self.config.stability_window

        if len(self._score_history) < window:
            return False

        recent = self._score_history[-window:]
        mean_score = np.mean(recent)
        if mean_score < 1e-10:
            return True

        # 变异系数
        cv = np.std(recent) / mean_score
        is_stable = cv < self.config.stability_threshold

        return bool(is_stable)

    def reset(self) -> None:
        """重置对焦器状态。"""
        self._score_history.clear()
        self._position_history.clear()
        logger.info("对焦器已重置")

    # ============ 内部方法 ============

    def _extract_roi(self, image: np.ndarray) -> np.ndarray:
        """提取 ROI 区域。"""
        h, w = image.shape
        roi_w, roi_h = self.config.roi_size

        if roi_w <= 0 or roi_h <= 0:
            return image

        cx = w // 2 + self.config.roi_center[0]
        cy = h // 2 + self.config.roi_center[1]
        x1 = max(0, cx - roi_w // 2)
        y1 = max(0, cy - roi_h // 2)
        x2 = min(w, cx + roi_w // 2)
        y2 = min(h, cy + roi_h // 2)

        return image[y1:y2, x1:x2]

    def _single_scale_score(self, image: np.ndarray) -> float:
        """单尺度拉普拉斯方差评分。"""
        laplacian = cv2.Laplacian(
            image, cv2.CV_64F, ksize=self.config.laplacian_kernel_size
        )
        return float(np.var(laplacian))

    def _multiscale_score(self, image: np.ndarray) -> float:
        """多尺度拉普拉斯方差评分。"""
        total_score = 0.0
        current = image.copy()

        for level in range(self.config.multiscale_levels):
            score = self._single_scale_score(current)
            total_score += score / (2 ** level)  # 高层权重降低

            if level < self.config.multiscale_levels - 1:
                # 下采样
                h, w = current.shape[:2]
                new_h, new_w = h // 2, w // 2
                if new_h < 16 or new_w < 16:
                    break
                current = cv2.resize(current, (new_w, new_h))

        return total_score

    def _fit_gaussian(
        self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """高斯曲线拟合。

        模型: y = A * exp(-(x-mu)^2 / (2*sigma^2)) + C
        """
        try:
            # 初始估计
            A0 = float(np.max(y) - np.min(y))
            mu0 = float(x[np.argmax(y)])
            sigma0 = float(np.std(x))
            C0 = float(np.min(y))

            # 线性化拟合 (简化版)
            # 使用加权最小二乘
            y_shifted = y - C0
            y_shifted = np.maximum(y_shifted, 1e-10)
            log_y = np.log(y_shifted / A0 + 1e-10)

            # 二次拟合: log(y/A) = -(x-mu)^2 / (2*sigma^2)
            coeffs = np.polyfit(x, -log_y, 2)
            # coeffs[0] = 1/(2*sigma^2), coeffs[1] = mu/sigma^2, coeffs[2] = -mu^2/(2*sigma^2)

            if coeffs[0] > 0:
                sigma_sq = 1.0 / (2 * coeffs[0])
                mu_fit = coeffs[1] * sigma_sq
                sigma_fit = np.sqrt(sigma_sq)
            else:
                return np.array([A0, mu0, sigma0, C0])

            # 重新估计 A 和 C
            residuals = y - A0 * np.exp(-((x - mu_fit) ** 2) / (2 * sigma_fit ** 2))
            C_fit = float(np.median(residuals))
            A_fit = float(np.max(y) - C_fit)

            return np.array([A_fit, mu_fit, sigma_fit, C_fit])

        except Exception as e:
            logger.warning(f"高斯拟合失败: {e}")
            return np.array([0.0, 0.0, 1.0, 0.0])

    def _fit_quadratic(
        self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """二次曲线拟合。

        模型: y = ax^2 + bx + c
        """
        try:
            coeffs = np.polyfit(x, y, 2)
            return coeffs
        except Exception as e:
            logger.warning(f"二次拟合失败: {e}")
            return np.array([0.0, 0.0, 0.0])
