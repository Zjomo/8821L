"""
鲁棒估计器 (RobustSpotEstimator)

基于鲁棒统计理论的光斑定位与拟合算法。

算法原理:
- RANSAC (Fischler & Bolles 1981) — 随机抽样一致性
- M-estimator (Huber 1964) — 鲁棒 M 估计
- Iteratively Reweighted Least Squares (IRLS) — 迭代重加权最小二乘
- Least Median of Squares (Rousseeuw 1984) — 最小中位数平方
- Modified Z-score — 修正 Z 分数异常值检测

功能:
- RANSAC 鲁棒光斑定位 (抗异常值)
- M-estimator 加权最小二乘
- 渐进最小二乘拟合
- 异常值自动检测与剔除
- 多模型假设验证

依赖: numpy, opencv-python (轮廓提取)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum

import cv2
import numpy as np


class MEstimatorType(Enum):
    """M-estimator 类型枚举。"""
    HUBER = "huber"          # Huber 损失
    TUKEY = "tukey"          # Tukey 双权重
    CAUCHY = "cauchy"        # Cauchy 损失
    WELSCH = "welsch"        # Welsch 损失
    ANDREWS = "andrews"      # Andrews 正弦


@dataclass
class MEstimatorConfig:
    """M-estimator 配置。

    Attributes
    ----------
    estimator_type : MEstimatorType
        M-estimator 类型。
    tuning_constant : float or None
        调谐常数 (None 则自动选择)。
    max_iterations : int
        IRLS 最大迭代次数。
    convergence_threshold : float
        收敛阈值。
    """
    estimator_type: MEstimatorType = MEstimatorType.HUBER
    tuning_constant: Optional[float] = None
    max_iterations: int = 50
    convergence_threshold: float = 1e-6


@dataclass
class RANSACResult:
    """RANSAC 拟合结果。

    Attributes
    ----------
    success : bool
        拟合是否成功。
    center : Tuple[float, float]
        拟合得到的中心坐标 (cx, cy)。
    sigma : float
        拟合得到的高斯 sigma (像素)。
    amplitude : float
        拟合得到的高斯振幅。
    inlier_mask : np.ndarray
        内点掩码 (与输入数据同形状)。
    n_inliers : int
        内点数量。
    inlier_ratio : float
        内点比例。
    n_iterations : int
        实际迭代次数。
    best_model_score : float
        最佳模型得分。
    outlier_indices : List[int]
        异常值索引列表。
    """
    success: bool = False
    center: Tuple[float, float] = (0.0, 0.0)
    sigma: float = 1.0
    amplitude: float = 0.0
    inlier_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    n_iterations: int = 0
    best_model_score: float = 0.0
    outlier_indices: List[int] = field(default_factory=list)


@dataclass
class OutlierReport:
    """异常值报告。

    Attributes
    ----------
    n_total : int
        总数据点数。
    n_outliers : int
        异常值数量。
    outlier_ratio : float
        异常值比例。
    outlier_indices : List[int]
        异常值索引。
    modified_z_scores : np.ndarray
        修正 Z 分数。
    threshold : float
        使用的阈值。
    method : str
        使用的检测方法。
    """
    n_total: int = 0
    n_outliers: int = 0
    outlier_ratio: float = 0.0
    outlier_indices: List[int] = field(default_factory=list)
    modified_z_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    threshold: float = 3.5
    method: str = "modified_z_score"


class RobustSpotEstimator:
    """鲁棒光斑估计器。

    使用 RANSAC 和 M-estimator 等鲁棒统计方法进行光斑定位，
    有效抵抗异常值和噪声干扰。

    Parameters
    ----------
    ransac_threshold : float
        RANSAC 内点判定阈值 (像素)。
    ransac_max_iterations : int
        RANSAC 最大迭代次数。
    ransac_min_samples : int
        RANSAC 最小样本数。
    min_inlier_ratio : float
        最小内点比例。
    mest_config : MEstimatorConfig or None
        M-estimator 配置。
    """

    def __init__(
        self,
        ransac_threshold: float = 2.0,
        ransac_max_iterations: int = 200,
        ransac_min_samples: int = 3,
        min_inlier_ratio: float = 0.3,
        mest_config: Optional[MEstimatorConfig] = None,
    ):
        self.ransac_threshold = float(ransac_threshold)
        self.ransac_max_iterations = int(ransac_max_iterations)
        self.ransac_min_samples = int(ransac_min_samples)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.mest_config = mest_config or MEstimatorConfig()

    def estimate_spot_robust(
        self,
        image: np.ndarray,
        initial_center: Optional[Tuple[int, int]] = None,
        roi_radius: int = 30,
    ) -> RANSACResult:
        """使用 RANSAC 鲁棒估计光斑位置。

        Parameters
        ----------
        image : np.ndarray
            灰度图像。
        initial_center : Tuple[int, int] or None
            初始中心估计 (可选)。
        roi_radius : int
            感兴趣区域半径。

        Returns
        -------
        RANSACResult
            RANSAC 拟合结果。
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = image.astype(np.float64)

        h, w = gray.shape

        # 确定 ROI
        if initial_center is not None:
            cx0, cy0 = initial_center
        else:
            # 使用强度加权质心作为初始估计
            total = gray.sum()
            if total < 1e-6:
                return RANSACResult(success=False)
            yy, xx = np.mgrid[:h, :w]
            cx0 = int(round(float(np.sum(xx * gray) / total)))
            cy0 = int(round(float(np.sum(yy * gray) / total)))

        # 提取 ROI 内的候选点
        yy, xx = np.mgrid[:h, :w]
        dist = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2)
        roi_mask = dist <= roi_radius

        if np.sum(roi_mask) < self.ransac_min_samples:
            return RANSACResult(success=False)

        roi_y = yy[roi_mask].astype(np.float64)
        roi_x = xx[roi_mask].astype(np.float64)
        roi_intensity = gray[roi_mask]

        # 使用 RANSAC 拟合 2D 高斯
        return self._ransac_gaussian_fit(roi_x, roi_y, roi_intensity)

    def estimate_spot_mestimator(
        self,
        points_x: np.ndarray,
        points_y: np.ndarray,
        intensities: np.ndarray,
    ) -> Tuple[Tuple[float, float], float, np.ndarray]:
        """使用 M-estimator 加权拟合光斑中心。

        Parameters
        ----------
        points_x : np.ndarray
            点的 x 坐标。
        points_y : np.ndarray
            点的 y 坐标。
        intensities : np.ndarray
            点的强度值。

        Returns
        -------
        Tuple[Tuple[float, float], float, np.ndarray]
            (中心坐标, sigma, 权重数组)。
        """
        px = np.asarray(points_x, dtype=np.float64)
        py = np.asarray(points_y, dtype=np.float64)
        intens = np.asarray(intensities, dtype=np.float64)

        n = len(px)
        if n < 3:
            return (float(np.mean(px)), float(np.mean(py))), 1.0, np.ones(n)

        # 初始估计 (强度加权质心)
        total = intens.sum()
        if total < 1e-10:
            return (float(np.mean(px)), float(np.mean(py))), 1.0, np.ones(n)

        cx = float(np.sum(px * intens) / total)
        cy = float(np.sum(py * intens) / total)

        # IRLS 迭代
        weights = intens / total
        best_cx, best_cy = cx, cy
        best_sigma = 1.0

        for iteration in range(self.mest_config.max_iterations):
            old_cx, old_cy = cx, cy

            # 加权最小二乘更新
            w_sum = weights.sum()
            if w_sum < 1e-10:
                break

            cx = float(np.sum(weights * px) / w_sum)
            cy = float(np.sum(weights * py) / w_sum)

            # 估计 sigma
            r = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            sigma = float(np.sqrt(np.sum(weights * r ** 2) / w_sum))
            sigma = max(sigma, 0.5)

            # 计算残差并更新权重
            residuals = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            weights = self._compute_mestimator_weights(residuals, sigma)

            # 收敛检查
            if abs(cx - old_cx) < self.mest_config.convergence_threshold and \
               abs(cy - old_cy) < self.mest_config.convergence_threshold:
                break

            best_cx, best_cy = cx, cy
            best_sigma = sigma

        return (best_cx, best_cy), best_sigma, weights

    def detect_outliers(
        self,
        data: np.ndarray,
        method: str = "modified_z_score",
        threshold: float = 3.5,
    ) -> OutlierReport:
        """检测数据中的异常值。

        Parameters
        ----------
        data : np.ndarray
            待检测数据。
        method : str
            检测方法 ('modified_z_score', 'iqr', 'mad')。
        threshold : float
            异常值判定阈值。

        Returns
        -------
        OutlierReport
            异常值报告。
        """
        d = np.asarray(data, dtype=np.float64).flatten()
        n = len(d)
        if n < 3:
            return OutlierReport(n_total=n, method=method, threshold=threshold)

        if method == "modified_z_score":
            return self._detect_outliers_modified_z(d, threshold)
        elif method == "iqr":
            return self._detect_outliers_iqr(d, threshold)
        elif method == "mad":
            return self._detect_outliers_mad(d, threshold)
        else:
            return self._detect_outliers_modified_z(d, threshold)

    def fit_gaussian_robust(
        self,
        points_x: np.ndarray,
        points_y: np.ndarray,
        intensities: np.ndarray,
    ) -> RANSACResult:
        """鲁棒高斯拟合 (RANSAC + M-estimator 精修)。

        先用 RANSAC 剔除异常值，再用 M-estimator 在内点上精修。

        Parameters
        ----------
        points_x : np.ndarray
            点的 x 坐标。
        points_y : np.ndarray
            点的 y 坐标。
        intensities : np.ndarray
            点的强度值。

        Returns
        -------
        RANSACResult
            拟合结果。
        """
        # 第一步: RANSAC 粗拟合
        ransac_result = self._ransac_gaussian_fit(
            np.asarray(points_x, dtype=np.float64),
            np.asarray(points_y, dtype=np.float64),
            np.asarray(intensities, dtype=np.float64),
        )

        if not ransac_result.success:
            return ransac_result

        # 第二步: 在内点上用 M-estimator 精修
        inlier_mask = ransac_result.inlier_mask
        if np.sum(inlier_mask) < 3:
            return ransac_result

        px_in = points_x[inlier_mask]
        py_in = points_y[inlier_mask]
        int_in = intensities[inlier_mask]

        (cx, cy), sigma, weights = self.estimate_spot_mestimator(px_in, py_in, int_in)

        # 更新结果
        ransac_result.center = (cx, cy)
        ransac_result.sigma = sigma

        # 重新计算振幅
        r = np.sqrt((px_in - cx) ** 2 + (py_in - cy) ** 2)
        if sigma > 1e-6:
            gauss_vals = np.exp(-r ** 2 / (2 * sigma ** 2))
            total_w = weights.sum()
            if total_w > 1e-10:
                amplitude = float(np.sum(weights * int_in) / max(np.sum(weights * gauss_vals), 1e-10))
            else:
                amplitude = float(np.max(int_in))
        else:
            amplitude = float(np.max(int_in))

        ransac_result.amplitude = amplitude

        return ransac_result

    # ---- 内部方法 ----

    def _ransac_gaussian_fit(
        self,
        points_x: np.ndarray,
        points_y: np.ndarray,
        intensities: np.ndarray,
    ) -> RANSACResult:
        """RANSAC 2D 高斯拟合的内部实现。"""
        n = len(points_x)
        if n < self.ransac_min_samples:
            return RANSACResult(success=False)

        # 预计算距离
        min_samples = self.ransac_min_samples
        threshold = self.ransac_threshold

        best_inliers = 0
        best_mask = np.zeros(n, dtype=bool)
        best_cx, best_cy = 0.0, 0.0
        best_sigma = 1.0
        best_amplitude = 0.0
        best_score = -float("inf")

        # 自适应最大迭代次数
        max_iter = self.ransac_max_iterations

        for iteration in range(max_iter):
            # 随机采样最小点集
            indices = np.random.choice(n, min(min_samples, n), replace=False)
            sample_x = points_x[indices]
            sample_y = points_y[indices]
            sample_i = intensities[indices]

            # 用采样点估计参数 (强度加权质心 + sigma)
            total_w = sample_i.sum()
            if total_w < 1e-10:
                continue

            cx = float(np.sum(sample_x * sample_i) / total_w)
            cy = float(np.sum(sample_y * sample_i) / total_w)

            r = np.sqrt((sample_x - cx) ** 2 + (sample_y - cy) ** 2)
            sigma = float(np.sqrt(np.sum(sample_i * r ** 2) / total_w))
            sigma = max(sigma, 0.5)

            amplitude = float(np.max(sample_i))

            # 计算内点
            distances = np.sqrt((points_x - cx) ** 2 + (points_y - cy) ** 2)
            # 内点判定: 距离在 3*sigma 内且强度合理
            inlier_mask = (distances < 3 * sigma) & (intensities > amplitude * 0.1)
            n_inliers = int(np.sum(inlier_mask))

            if n_inliers < min_samples:
                continue

            # 评分: 内点数量 * 平均强度
            score = float(n_inliers * np.mean(intensities[inlier_mask]))

            if score > best_score:
                best_score = score
                best_inliers = n_inliers
                best_mask = inlier_mask.copy()
                best_cx = cx
                best_cy = cy
                best_sigma = sigma
                best_amplitude = amplitude

                # 自适应更新迭代次数
                inlier_ratio = n_inliers / n
                if inlier_ratio > 0:
                    expected_iter = int(np.log(0.01) / np.log(1 - inlier_ratio ** min_samples))
                    max_iter = min(max_iter, max(expected_iter, 10))

        if best_inliers < min_samples:
            return RANSACResult(success=False)

        # 用所有内点重新拟合
        inlier_x = points_x[best_mask]
        inlier_y = points_y[best_mask]
        inlier_i = intensities[best_mask]

        total_w = inlier_i.sum()
        if total_w > 1e-10:
            best_cx = float(np.sum(inlier_x * inlier_i) / total_w)
            best_cy = float(np.sum(inlier_y * inlier_i) / total_w)
            r = np.sqrt((inlier_x - best_cx) ** 2 + (inlier_y - best_cy) ** 2)
            best_sigma = float(np.sqrt(np.sum(inlier_i * r ** 2) / total_w))
            best_sigma = max(best_sigma, 0.5)
            best_amplitude = float(np.max(inlier_i))

        # 异常值索引
        outlier_indices = [int(i) for i in range(n) if not best_mask[i]]

        return RANSACResult(
            success=True,
            center=(best_cx, best_cy),
            sigma=best_sigma,
            amplitude=best_amplitude,
            inlier_mask=best_mask,
            n_inliers=best_inliers,
            inlier_ratio=float(best_inliers / n),
            n_iterations=min(iteration + 1, self.ransac_max_iterations),
            best_model_score=best_score,
            outlier_indices=outlier_indices,
        )

    def _compute_mestimator_weights(
        self,
        residuals: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """计算 M-estimator 权重。"""
        r = np.abs(residuals) / max(scale, 1e-10)
        config = self.mest_config

        # 获取调谐常数
        c = config.tuning_constant
        if c is None:
            c = self._default_tuning_constant(config.estimator_type)

        if config.estimator_type == MEstimatorType.HUBER:
            # Huber: w = 1 if r <= c, else c/r
            weights = np.where(r <= c, 1.0, c / np.maximum(r, 1e-10))

        elif config.estimator_type == MEstimatorType.TUKEY:
            # Tukey 双权重: w = (1 - (r/c)^2)^2 if r < c, else 0
            r_c = np.minimum(r / c, 1.0)
            weights = np.where(r < c, (1.0 - r_c ** 2) ** 2, 0.0)

        elif config.estimator_type == MEstimatorType.CAUCHY:
            # Cauchy: w = 1 / (1 + (r/c)^2)
            weights = 1.0 / (1.0 + (r / c) ** 2)

        elif config.estimator_type == MEstimatorType.WELSCH:
            # Welsch: w = exp(-(r/c)^2 / 2)
            weights = np.exp(-0.5 * (r / c) ** 2)

        elif config.estimator_type == MEstimatorType.ANDREWS:
            # Andrews: w = sin(pi*r/c) / (pi*r/c) if r < c, else 0
            r_c = r / c
            weights = np.where(
                r < c,
                np.where(r_c < 1e-10, 1.0, np.sin(np.pi * r_c) / (np.pi * r_c)),
                0.0,
            )
        else:
            weights = np.ones_like(r)

        # 确保权重非负
        weights = np.maximum(weights, 0.0)
        w_sum = weights.sum()
        if w_sum > 1e-10:
            weights /= w_sum

        return weights

    @staticmethod
    def _default_tuning_constant(estimator_type: MEstimatorType) -> float:
        """获取 M-estimator 默认调谐常数。"""
        defaults = {
            MEstimatorType.HUBER: 1.345,    # 95% 渐近效率
            MEstimatorType.TUKEY: 4.685,     # 95% 渐近效率
            MEstimatorType.CAUCHY: 2.385,    # 95% 渐近效率
            MEstimatorType.WELSCH: 2.985,    # 95% 渐近效率
            MEstimatorType.ANDREWS: 1.339,   # 95% 渐近效率
        }
        return defaults.get(estimator_type, 1.345)

    def _detect_outliers_modified_z(
        self,
        data: np.ndarray,
        threshold: float = 3.5,
    ) -> OutlierReport:
        """使用修正 Z 分数检测异常值。"""
        median = float(np.median(data))
        mad = float(np.median(np.abs(data - median)))
        if mad < 1e-10:
            mad = 1e-10

        modified_z = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z) > threshold
        outlier_indices = [int(i) for i in range(len(data)) if outlier_mask[i]]

        return OutlierReport(
            n_total=len(data),
            n_outliers=int(np.sum(outlier_mask)),
            outlier_ratio=float(np.mean(outlier_mask)),
            outlier_indices=outlier_indices,
            modified_z_scores=modified_z,
            threshold=threshold,
            method="modified_z_score",
        )

    def _detect_outliers_iqr(
        self,
        data: np.ndarray,
        threshold: float = 1.5,
    ) -> OutlierReport:
        """使用 IQR 方法检测异常值。"""
        q1 = float(np.percentile(data, 25))
        q3 = float(np.percentile(data, 75))
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr

        outlier_mask = (data < lower) | (data > upper)
        outlier_indices = [int(i) for i in range(len(data)) if outlier_mask[i]]

        return OutlierReport(
            n_total=len(data),
            n_outliers=int(np.sum(outlier_mask)),
            outlier_ratio=float(np.mean(outlier_mask)),
            outlier_indices=outlier_indices,
            threshold=threshold,
            method="iqr",
        )

    def _detect_outliers_mad(
        self,
        data: np.ndarray,
        threshold: float = 3.0,
    ) -> OutlierReport:
        """使用 MAD 方法检测异常值。"""
        median = float(np.median(data))
        mad = float(np.median(np.abs(data - median)))
        if mad < 1e-10:
            mad = 1e-10

        outlier_mask = np.abs(data - median) > threshold * mad
        outlier_indices = [int(i) for i in range(len(data)) if outlier_mask[i]]

        # 转换为修正 Z 分数
        modified_z = 0.6745 * (data - median) / mad

        return OutlierReport(
            n_total=len(data),
            n_outliers=int(np.sum(outlier_mask)),
            outlier_ratio=float(np.mean(outlier_mask)),
            outlier_indices=outlier_indices,
            modified_z_scores=modified_z,
            threshold=threshold,
            method="mad",
        )
