"""
自动系统标定引擎 (AutoCalibrationEngine)

灵感来源:
- slmsuite (https://github.com/holodyne/slmsuite) — 自动傅里叶域标定,
  像素-物理坐标映射, 透镜畸变校正
- python-microscope (https://github.com/python-microscope/python-microscope) —
  设备聚合与自动发现, 多设备协同标定
- OpenCV camera calibration — 棋盘格标定, 仿射变换, 径向畸变模型
- Prandtl-Ishlinskii 滞回模型 — 迟滞非线性建模与补偿

算法原理:
- Affine Transformation — 6 自由度仿射变换 (平移, 旋转, 缩放, 剪切)
- Radial Distortion Model — 径向畸变校正 (k1, k2, k3 Brown-Conrady 模型)
- Least Squares Fitting — 最小二乘法拟合标定参数
- Prandtl-Ishlinskii Play Operator — Play 算子迟滞建模
- Step Response Analysis — 阶跃响应分析 (上升时间, 调节时间, 超调量)

功能:
- 像素坐标到物理坐标 (电机步数) 的自动映射标定
- 仿射变换矩阵估计 (6-DOF)
- 径向畸变系数估计与校正
- 迟滞特性测量与 Prandtl-Ishlinskii 补偿
- 系统阶跃响应分析 (上升时间, 调节时间, 超调量)
- 死区检测与灵敏度估计
- 标定质量评估与验证
- 自动标定流程编排

依赖: numpy, cv2 (可选, 用于辅助标定)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.AutoCalibration")


# ======================== 配置数据类 ========================


@dataclass
class CalibrationConfig:
    """标定配置参数。

    Attributes
    ----------
    grid_points_per_axis : int
        每轴网格标定点数 (如 5 表示 5x5=25 个点)。
    grid_step_size : int
        网格点间电机步距 (步)。
    calibration_settle_time_s : float
        每个标定点稳定等待时间 (秒)。
    max_calibration_time_s : float
        标定最大允许时间 (秒)。
    enable_distortion_correction : bool
        是否启用径向畸变校正。
    enable_hysteresis_compensation : bool
        是否启用迟滞补偿。
    hysteresis_test_points : int
        迟滞测试点数。
    hysteresis_directions : int
        迟滞正向循环次数。
    response_test_amplitude : int
        阶跃响应测试幅值 (电机步数)。
    response_test_points : int
        阶跃响应采样点数。
    min_calibration_points : int
        最少标定点数 (低于此值标定无效)。
    max_rms_error_px : float
        最大允许 RMS 误差 (像素)。
    """
    grid_points_per_axis: int = 5
    grid_step_size: int = 200
    calibration_settle_time_s: float = 0.5
    max_calibration_time_s: float = 300.0
    enable_distortion_correction: bool = True
    enable_hysteresis_compensation: bool = True
    hysteresis_test_points: int = 10
    hysteresis_directions: int = 3
    response_test_amplitude: int = 500
    response_test_points: int = 20
    min_calibration_points: int = 9
    max_rms_error_px: float = 2.0


# ======================== 结果数据类 ========================


@dataclass
class CalibrationResult:
    """标定结果。

    Attributes
    ----------
    affine_matrix : np.ndarray
        3x3 仿射变换矩阵。
    distortion_coefficients : np.ndarray
        径向畸变系数 [k1, k2, k3]。
    hysteresis_model : dict
        迟滞模型参数字典。
    response_curve_data : dict
        响应曲线数据字典。
    rms_error_px : float
        坐标映射 RMS 误差 (像素)。
    calibration_quality_score : float
        标定质量评分 [0, 100]。
    calibration_timestamp : float
        标定时间戳 (Unix 时间)。
    is_valid : bool
        标定结果是否有效。
    """
    affine_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    distortion_coefficients: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    hysteresis_model: dict = field(default_factory=dict)
    response_curve_data: dict = field(default_factory=dict)
    rms_error_px: float = float("inf")
    calibration_quality_score: float = 0.0
    calibration_timestamp: float = 0.0
    is_valid: bool = False


@dataclass
class CalibrationReport:
    """标定详细报告。

    Attributes
    ----------
    coordinate_mapping_accuracy : float
        坐标映射精度 (RMS 像素)。
    hysteresis_residual : float
        迟滞补偿后残余 (像素)。
    response_linearity : float
        响应线性度 [0, 1] (1.0 为完美线性)。
    dead_zone_size : float
        死区大小 (电机步数)。
    sensitivity_px_per_step : float
        灵敏度 (像素/步)。
    recommendations : List[str]
        改进建议列表。
    """
    coordinate_mapping_accuracy: float = float("inf")
    hysteresis_residual: float = float("inf")
    response_linearity: float = 0.0
    dead_zone_size: float = 0.0
    sensitivity_px_per_step: float = 0.0
    recommendations: List[str] = field(default_factory=list)


# ======================== 坐标映射器 ========================


class CoordinateMapper:
    """坐标变换器。

    实现像素坐标与物理坐标 (电机步数) 之间的双向映射。
    包含仿射变换与径向畸变校正。

    Parameters
    ----------
    image_center : Tuple[float, float] or None
        图像中心坐标 (cx, cy)。为 None 时默认 (0, 0)。
    enable_distortion : bool
        是否启用径向畸变校正。
    """

    def __init__(
        self,
        image_center: Optional[Tuple[float, float]] = None,
        enable_distortion: bool = True,
    ):
        self.image_center = np.array(
            image_center if image_center is not None else (0.0, 0.0),
            dtype=np.float64,
        )
        self.enable_distortion = enable_distortion

        # 仿射变换矩阵 (3x3 齐次坐标)
        # [a  b  tx]   像素 = A * 物理
        # [c  d  ty]
        # [0  0   1]
        self._affine = np.eye(3, dtype=np.float64)

        # 径向畸变系数 [k1, k2, k3]
        self._distortion = np.zeros(3, dtype=np.float64)

        # 逆仿射矩阵 (缓存)
        self._affine_inv: Optional[np.ndarray] = None

        # 标定数据点
        self._pixel_points: List[np.ndarray] = []
        self._physical_points: List[np.ndarray] = []

        LOGGER.info(
            "CoordinateMapper: 初始化完成 (distortion=%s, center=%.1f, %.1f)",
            self.enable_distortion, self.image_center[0], self.image_center[1],
        )

    @property
    def affine_matrix(self) -> np.ndarray:
        """获取仿射变换矩阵 (3x3)。"""
        return self._affine.copy()

    @property
    def distortion_coefficients(self) -> np.ndarray:
        """获取径向畸变系数 [k1, k2, k3]。"""
        return self._distortion.copy()

    def add_calibration_point(
        self,
        pixel_x: float,
        pixel_y: float,
        physical_x: float,
        physical_y: float,
    ) -> None:
        """添加标定数据点。

        Parameters
        ----------
        pixel_x : float
            像素 x 坐标。
        pixel_y : float
            像素 y 坐标。
        physical_x : float
            物理 x 坐标 (电机步数)。
        physical_y : float
            物理 y 坐标 (电机步数)。
        """
        self._pixel_points.append(np.array([pixel_x, pixel_y], dtype=np.float64))
        self._physical_points.append(np.array([physical_x, physical_y], dtype=np.float64))

    def clear_calibration_points(self) -> None:
        """清除所有标定数据点。"""
        self._pixel_points.clear()
        self._physical_points.clear()
        self._affine = np.eye(3, dtype=np.float64)
        self._distortion = np.zeros(3, dtype=np.float64)
        self._affine_inv = None
        LOGGER.info("CoordinateMapper: 标定数据已清除")

    def fit_affine(self) -> float:
        """最小二乘法拟合仿射变换矩阵。

        使用所有已添加的标定点，通过最小二乘法估计
        6 自由度仿射变换: 平移 (tx, ty), 旋转, 缩放, 剪切。

        Returns
        -------
        float
            拟合 RMS 误差 (像素)。

        Raises
        ------
        ValueError
            标定点数不足 (至少需要 3 个点)。
        """
        n = len(self._pixel_points)
        if n < 3:
            raise ValueError(
                f"标定点数不足: 需要 >= 3, 实际 {n}"
            )

        # 构建线性方程组: pixel = A * physical
        # [px]   [a b tx] [sx]
        # [py] = [c d ty] [sy]
        #                 [ 1]
        A_mat = np.zeros((2 * n, 6), dtype=np.float64)
        b_vec = np.zeros(2 * n, dtype=np.float64)

        for i in range(n):
            sx, sy = self._physical_points[i]
            px, py = self._pixel_points[i]
            A_mat[2 * i, :] = [sx, sy, 1.0, 0.0, 0.0, 0.0]
            A_mat[2 * i + 1, :] = [0.0, 0.0, 0.0, sx, sy, 1.0]
            b_vec[2 * i] = px
            b_vec[2 * i + 1] = py

        # 最小二乘求解
        params, residuals, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

        a, b, tx, c, d, ty = params
        self._affine = np.array([
            [a, b, tx],
            [c, d, ty],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 缓存逆矩阵
        self._affine_inv = np.linalg.inv(self._affine)

        # 计算 RMS 误差
        rms = self._compute_affine_rms()

        LOGGER.info(
            "CoordinateMapper: 仿射变换拟合完成, %d 个点, RMS=%.3f px",
            n, rms,
        )
        LOGGER.debug(
            "CoordinateMapper: 仿射矩阵=\n%s", self._affine,
        )
        return rms

    def fit_distortion(self, max_iterations: int = 50, tol: float = 1e-8) -> float:
        """迭代估计径向畸变系数。

        假设初始仿射变换已拟合，通过迭代优化径向畸变系数
        k1, k2, k3 使残差最小化。

        畸变模型 (Brown-Conrady):
            r_distorted = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)

        Parameters
        ----------
        max_iterations : int
            最大迭代次数。
        tol : float
            收敛容差。

        Returns
        -------
        float
            畸变校正后 RMS 误差 (像素)。

        Raises
        ------
        ValueError
            标定点数不足或仿射变换未拟合。
        """
        if not self.enable_distortion:
            LOGGER.info("CoordinateMapper: 畸变校正已禁用, 跳过")
            return self._compute_affine_rms()

        n = len(self._pixel_points)
        if n < 6:
            raise ValueError(
                f"畸变标定点数不足: 需要 >= 6, 实际 {n}"
            )

        # 检查仿射矩阵是否为单位矩阵 (未拟合)
        if np.allclose(self._affine, np.eye(3)):
            raise ValueError("请先调用 fit_affine() 拟合仿射变换")

        cx, cy = self.image_center
        k = np.zeros(3, dtype=np.float64)

        prev_rms = float("inf")

        for iteration in range(max_iterations):
            # 对每个标定点，计算去畸变后的像素坐标
            A_mat = np.zeros((2 * n, 3), dtype=np.float64)
            b_vec = np.zeros(2 * n, dtype=np.float64)

            for i in range(n):
                px, py = self._pixel_points[i]
                sx, sy = self._physical_points[i]

                # 通过仿射变换预测无畸变像素坐标
                predicted = self._affine @ np.array([sx, sy, 1.0])
                pred_x, pred_y = predicted[0], predicted[1]

                # 当前畸变坐标 (以图像中心为原点)
                dx = px - cx
                dy = py - cy
                r2 = dx * dx + dy * dy
                r2 = max(r2, 1e-12)  # 避免除零

                # 畸变因子: 1 + k1*r^2 + k2*r^4 + k3*r^6
                dist_factor = 1.0 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3

                # 去畸变坐标
                undist_x = cx + dx / dist_factor
                undist_y = cy + dy / dist_factor

                # 残差
                residual_x = pred_x - undist_x
                residual_y = pred_y - undist_y

                # 雅可比矩阵 (对 k1, k2, k3 的偏导)
                # d(undist_x)/dk_j = -dx * r^(2j) / dist_factor^2
                # d(undist_y)/dk_j = -dy * r^(2j) / dist_factor^2
                for j in range(3):
                    r_pow = r2 ** (j + 1)
                    jac_x = -dx * r_pow / (dist_factor ** 2)
                    jac_y = -dy * r_pow / (dist_factor ** 2)
                    A_mat[2 * i, j] = jac_x
                    A_mat[2 * i + 1, j] = jac_y

                b_vec[2 * i] = residual_x
                b_vec[2 * i + 1] = residual_y

            # 最小二乘更新
            try:
                delta_k, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except np.linalg.LinAlgError:
                LOGGER.warning("CoordinateMapper: 畸变迭代 %d 线性求解失败", iteration)
                break

            k += delta_k

            # 计算当前 RMS
            current_rms = self._compute_distortion_rms(k)

            if abs(prev_rms - current_rms) < tol:
                LOGGER.info(
                    "CoordinateMapper: 畸变迭代收敛于第 %d 次, RMS=%.4f px",
                    iteration + 1, current_rms,
                )
                break

            prev_rms = current_rms

        self._distortion = k

        LOGGER.info(
            "CoordinateMapper: 畸变系数 k1=%.6f, k2=%.6f, k3=%.6f, RMS=%.4f px",
            k[0], k[1], k[2], current_rms,
        )
        return current_rms

    def pixel_to_physical(
        self,
        pixel_x: float,
        pixel_y: float,
    ) -> Tuple[float, float]:
        """像素坐标 -> 物理坐标 (电机步数)。

        Parameters
        ----------
        pixel_x : float
            像素 x 坐标。
        pixel_y : float
            像素 y 坐标。

        Returns
        -------
        Tuple[float, float]
            物理坐标 (physical_x, physical_y)。
        """
        # 去畸变
        if self.enable_distortion and not np.allclose(self._distortion, 0):
            ux, uy = self._undistort_pixel(pixel_x, pixel_y)
        else:
            ux, uy = pixel_x, pixel_y

        # 逆仿射变换
        if self._affine_inv is None:
            self._affine_inv = np.linalg.inv(self._affine)

        pt = self._affine_inv @ np.array([ux, uy, 1.0])
        return float(pt[0]), float(pt[1])

    def physical_to_pixel(
        self,
        physical_x: float,
        physical_y: float,
    ) -> Tuple[float, float]:
        """物理坐标 (电机步数) -> 像素坐标。

        Parameters
        ----------
        physical_x : float
            物理 x 坐标 (电机步数)。
        physical_y : float
            物理 y 坐标 (电机步数)。

        Returns
        -------
        Tuple[float, float]
            像素坐标 (pixel_x, pixel_y)。
        """
        pt = self._affine @ np.array([physical_x, physical_y, 1.0])
        px, py = float(pt[0]), float(pt[1])

        # 添加畸变
        if self.enable_distortion and not np.allclose(self._distortion, 0):
            px, py = self._distort_pixel(px, py)

        return px, py

    def batch_pixel_to_physical(
        self,
        pixel_coords: np.ndarray,
    ) -> np.ndarray:
        """批量像素坐标 -> 物理坐标。

        Parameters
        ----------
        pixel_coords : np.ndarray
            像素坐标数组, 形状 (N, 2)。

        Returns
        -------
        np.ndarray
            物理坐标数组, 形状 (N, 2)。
        """
        result = np.zeros_like(pixel_coords, dtype=np.float64)
        for i in range(len(pixel_coords)):
            result[i, 0], result[i, 1] = self.pixel_to_physical(
                pixel_coords[i, 0], pixel_coords[i, 1],
            )
        return result

    def batch_physical_to_pixel(
        self,
        physical_coords: np.ndarray,
    ) -> np.ndarray:
        """批量物理坐标 -> 像素坐标。

        Parameters
        ----------
        physical_coords : np.ndarray
            物理坐标数组, 形状 (N, 2)。

        Returns
        -------
        np.ndarray
            像素坐标数组, 形状 (N, 2)。
        """
        result = np.zeros_like(physical_coords, dtype=np.float64)
        for i in range(len(physical_coords)):
            result[i, 0], result[i, 1] = self.physical_to_pixel(
                physical_coords[i, 0], physical_coords[i, 1],
            )
        return result

    def assess_accuracy(self) -> float:
        """评估坐标映射精度。

        使用已存储的标定点计算 RMS 映射误差。

        Returns
        -------
        float
            RMS 误差 (像素)。

        Raises
        ------
        ValueError
            无标定数据点。
        """
        if not self._pixel_points:
            raise ValueError("无标定数据点, 无法评估精度")

        if self.enable_distortion and not np.allclose(self._distortion, 0):
            return self._compute_distortion_rms(self._distortion)
        else:
            return self._compute_affine_rms()

    def _compute_affine_rms(self) -> float:
        """计算纯仿射变换的 RMS 误差。"""
        total_sq_error = 0.0
        n = len(self._pixel_points)
        for i in range(n):
            sx, sy = self._physical_points[i]
            predicted = self._affine @ np.array([sx, sy, 1.0])
            pred_x, pred_y = predicted[0], predicted[1]
            actual_x, actual_y = self._pixel_points[i]
            total_sq_error += (pred_x - actual_x) ** 2 + (pred_y - actual_y) ** 2
        return float(np.sqrt(total_sq_error / max(n, 1)))

    def _compute_distortion_rms(self, k: np.ndarray) -> float:
        """计算含畸变校正的 RMS 误差。"""
        cx, cy = self.image_center
        total_sq_error = 0.0
        n = len(self._pixel_points)
        for i in range(n):
            sx, sy = self._physical_points[i]
            predicted = self._affine @ np.array([sx, sy, 1.0])
            pred_x, pred_y = predicted[0], predicted[1]

            px, py = self._pixel_points[i]
            dx = px - cx
            dy = py - cy
            r2 = dx * dx + dy * dy
            r2 = max(r2, 1e-12)

            dist_factor = 1.0 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3
            undist_x = cx + dx / dist_factor
            undist_y = cy + dy / dist_factor

            total_sq_error += (pred_x - undist_x) ** 2 + (pred_y - undist_y) ** 2
        return float(np.sqrt(total_sq_error / max(n, 1)))

    def _undistort_pixel(
        self,
        px: float,
        py: float,
    ) -> Tuple[float, float]:
        """像素坐标去畸变。

        Parameters
        ----------
        px : float
            畸变像素 x。
        py : float
            畸变像素 y。

        Returns
        -------
        Tuple[float, float]
            去畸变后像素坐标。
        """
        cx, cy = self.image_center
        dx = px - cx
        dy = py - cy
        r2 = dx * dx + dy * dy
        r2 = max(r2, 1e-12)

        k1, k2, k3 = self._distortion
        dist_factor = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

        undist_x = cx + dx / dist_factor
        undist_y = cy + dy / dist_factor
        return undist_x, undist_y

    def _distort_pixel(
        self,
        px: float,
        py: float,
    ) -> Tuple[float, float]:
        """像素坐标添加畸变。

        Parameters
        ----------
        px : float
            无畸变像素 x。
        py : float
            无畸变像素 y。

        Returns
        -------
        Tuple[float, float]
            畸变后像素坐标。
        """
        cx, cy = self.image_center
        dx = px - cx
        dy = py - cy
        r2 = dx * dx + dy * dy
        r2 = max(r2, 1e-12)

        k1, k2, k3 = self._distortion
        dist_factor = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 6

        dist_x = cx + dx * dist_factor
        dist_y = cy + dy * dist_factor
        return dist_x, dist_y


# ======================== 迟滞补偿器 ========================


class HysteresisCompensator:
    """迟滞补偿器。

    基于 Prandtl-Ishlinskii (PI) 模型的迟滞非线性建模与补偿。
    使用 Play 算子叠加实现迟滞曲线拟合。

    Parameters
    ----------
    num_play_operators : int
        Play 算子数量 (越多拟合越精确, 但计算量越大)。
    threshold_range : Tuple[float, float]
        Play 算子阈值范围 (最小, 最大)。
    """

    def __init__(
        self,
        num_play_operators: int = 10,
        threshold_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.num_play_operators = int(num_play_operators)
        self.threshold_range = threshold_range

        # Play 算子阈值 (线性分布)
        self._thresholds = np.linspace(
            threshold_range[0],
            threshold_range[1],
            num_play_operators + 1,
        )[1:]  # 去掉 0 阈值

        # Play 算子权重
        self._weights = np.ones(self.num_play_operators, dtype=np.float64)

        # Play 算子状态 (上一次的输出值)
        self._play_states = np.zeros(self.num_play_operators, dtype=np.float64)

        # 初始输入值 (用于 Play 算子初始化)
        self._initial_input: float = 0.0

        # 标定数据
        self._forward_curves: List[np.ndarray] = []
        self._backward_curves: List[np.ndarray] = []
        self._input_positions: Optional[np.ndarray] = None

        # 补偿查找表
        self._compensation_lut: Optional[np.ndarray] = None
        self._lut_inputs: Optional[np.ndarray] = None

        LOGGER.info(
            "HysteresisCompensator: 初始化完成 (operators=%d, range=%.3f-%.3f)",
            self.num_play_operators, threshold_range[0], threshold_range[1],
        )

    @property
    def weights(self) -> np.ndarray:
        """获取 Play 算子权重。"""
        return self._weights.copy()

    @property
    def thresholds(self) -> np.ndarray:
        """获取 Play 算子阈值。"""
        return self._thresholds.copy()

    def add_hysteresis_data(
        self,
        input_positions: np.ndarray,
        forward_output: np.ndarray,
        backward_output: np.ndarray,
    ) -> None:
        """添加迟滞测试数据 (正向/反向响应曲线)。

        Parameters
        ----------
        input_positions : np.ndarray
            输入位置序列 (电机步数), 形状 (N,)。
        forward_output : np.ndarray
            正向行程输出 (像素位置), 形状 (N,)。
        backward_output : np.ndarray
            反向行程输出 (像素位置), 形状 (N,)。
        """
        if len(input_positions) != len(forward_output) or len(input_positions) != len(backward_output):
            raise ValueError("输入位置与输出数组长度不一致")

        if self._input_positions is None:
            self._input_positions = input_positions.copy()
        self._forward_curves.append(forward_output.copy())
        self._backward_curves.append(backward_output.copy())

        LOGGER.info(
            "HysteresisCompensator: 添加迟滞数据, %d 个点, 累计 %d 组",
            len(input_positions), len(self._forward_curves),
        )

    def fit_play_operators(self) -> float:
        """拟合 Prandtl-Ishlinskii Play 算子权重。

        使用最小二乘法拟合 Play 算子权重, 使模型输出
        最佳匹配观测到的迟滞曲线。

        Returns
        -------
        float
            拟合残差 (像素)。

        Raises
        ------
        ValueError
            无迟滞数据。
        """
        if not self._forward_curves:
            raise ValueError("无迟滞数据, 请先调用 add_hysteresis_data()")

        # 合并所有正向/反向数据
        all_inputs = []
        all_outputs = []

        for fwd, bwd in zip(self._forward_curves, self._backward_curves):
            all_inputs.append(self._input_positions)
            all_outputs.append(fwd)
            all_inputs.append(self._input_positions)
            all_outputs.append(bwd)

        inputs = np.concatenate(all_inputs)
        outputs = np.concatenate(all_outputs)

        # 归一化输入到 [0, 1]
        input_min = inputs.min()
        input_max = inputs.max()
        input_range = input_max - input_min
        if input_range < 1e-12:
            raise ValueError("输入范围过小, 无法归一化")

        norm_inputs = (inputs - input_min) / input_range

        # 构建 Play 算子输出矩阵
        # 对每个数据点, 模拟 Play 算子状态
        n_points = len(norm_inputs)
        play_matrix = np.zeros((n_points, self.num_play_operators), dtype=np.float64)

        for j in range(self.num_play_operators):
            r_j = self._thresholds[j]
            state = 0.0
            for i in range(n_points):
                u = norm_inputs[i]
                # Play 算子: F_r[u](t) = max(u(t) - r, min(F_r[u](t-1), u(t) + r))
                state = max(u - r_j, min(state, u + r_j))
                play_matrix[i, j] = state

        # 最小二乘拟合权重
        try:
            self._weights, residuals, _, _ = np.linalg.lstsq(
                play_matrix, outputs, rcond=None,
            )
        except np.linalg.LinAlgError:
            LOGGER.warning("HysteresisCompensator: 最小二乘求解失败, 使用默认权重")
            self._weights = np.ones(self.num_play_operators, dtype=np.float64)
            return float("inf")

        # 计算残差
        predicted = play_matrix @ self._weights
        residual = float(np.sqrt(np.mean((predicted - outputs) ** 2)))

        LOGGER.info(
            "HysteresisCompensator: Play 算子拟合完成, 残差=%.4f px",
            residual,
        )
        LOGGER.debug(
            "HysteresisCompensator: 权重=%s", self._weights,
        )
        return residual

    def compensate(self, input_value: float) -> float:
        """对输入值进行迟滞补偿。

        通过逆 PI 模型计算补偿量, 减小迟滞引起的定位误差。

        Parameters
        ----------
        input_value : float
            原始输入值 (电机步数)。

        Returns
        -------
        float
            补偿后的输入值 (电机步数)。
        """
        if self._input_positions is None:
            LOGGER.warning("HysteresisCompensator: 未标定, 返回原始值")
            return input_value

        # 如果有查找表, 使用查找表插值
        if self._compensation_lut is not None and self._lut_inputs is not None:
            return self._interpolate_compensation(input_value)

        # 否则使用 Play 算子实时补偿
        input_min = self._input_positions.min()
        input_max = self._input_positions.max()
        input_range = input_max - input_min
        if input_range < 1e-12:
            return input_value

        norm_input = (input_value - input_min) / input_range

        # 计算 Play 算子输出
        play_output = 0.0
        for j in range(self.num_play_operators):
            r_j = self._thresholds[j]
            self._play_states[j] = max(
                norm_input - r_j,
                min(self._play_states[j], norm_input + r_j),
            )
            play_output += self._weights[j] * self._play_states[j]

        # 补偿量 = 模型输出 - 理想线性输出
        ideal_output = norm_input * np.sum(self._weights)
        compensation = play_output - ideal_output

        # 将补偿量转换回原始尺度
        compensated = input_value - compensation * input_range

        return float(compensated)

    def generate_compensation_lut(
        self,
        num_lut_points: int = 100,
    ) -> np.ndarray:
        """生成补偿查找表。

        Parameters
        ----------
        num_lut_points : int
            查找表点数。

        Returns
        -------
        np.ndarray
            补偿查找表, 形状 (num_lut_points, 2),
            第一列为输入值, 第二列为补偿后值。
        """
        if self._input_positions is None:
            raise ValueError("无迟滞数据, 请先添加数据并拟合")

        input_min = self._input_positions.min()
        input_max = self._input_positions.max()

        lut_inputs = np.linspace(input_min, input_max, num_lut_points)
        lut_outputs = np.zeros(num_lut_points, dtype=np.float64)

        # 重置 Play 算子状态
        self._play_states = np.zeros(self.num_play_operators, dtype=np.float64)

        for i in range(num_lut_points):
            lut_outputs[i] = self.compensate(lut_inputs[i])

        self._lut_inputs = lut_inputs
        self._compensation_lut = np.column_stack([lut_inputs, lut_outputs])

        LOGGER.info(
            "HysteresisCompensator: 补偿查找表已生成, %d 个点",
            num_lut_points,
        )
        return self._compensation_lut

    def estimate_residual_hysteresis(self) -> float:
        """估计补偿后的残余迟滞。

        Returns
        -------
        float
            残余迟滞量 (像素)。

        Raises
        ------
        ValueError
            无迟滞数据。
        """
        if not self._forward_curves or not self._backward_curves:
            raise ValueError("无迟滞数据")

        # 使用最后一组数据评估
        fwd = self._forward_curves[-1]
        bwd = self._backward_curves[-1]

        # 迟滞宽度 = 正向与反向输出的最大差值
        hysteresis_width = np.max(np.abs(fwd - bwd))

        # 补偿后残余 (粗略估计为迟滞宽度的 10-30%)
        residual = hysteresis_width * 0.2

        LOGGER.info(
            "HysteresisCompensator: 迟滞宽度=%.3f px, 估计残余=%.3f px",
            hysteresis_width, residual,
        )
        return float(residual)

    def reset_state(self) -> None:
        """重置 Play 算子内部状态。"""
        self._play_states = np.zeros(self.num_play_operators, dtype=np.float64)
        LOGGER.info("HysteresisCompensator: 状态已重置")

    def _interpolate_compensation(self, input_value: float) -> float:
        """使用查找表插值计算补偿值。"""
        idx = np.searchsorted(self._lut_inputs, input_value)
        if idx <= 0:
            return float(self._compensation_lut[0, 1])
        if idx >= len(self._lut_inputs):
            return float(self._compensation_lut[-1, 1])

        # 线性插值
        x0 = self._lut_inputs[idx - 1]
        x1 = self._lut_inputs[idx]
        y0 = self._compensation_lut[idx - 1, 1]
        y1 = self._compensation_lut[idx, 1]

        t = (input_value - x0) / (x1 - x0) if abs(x1 - x0) > 1e-12 else 0.0
        return float(y0 + t * (y1 - y0))


# ======================== 响应曲线分析器 ========================


class ResponseCurveAnalyzer:
    """系统响应曲线分析器。

    分析光斑位置对电机步进指令的阶跃响应特性,
    包括上升时间、调节时间、超调量、线性度等。

    Parameters
    ----------
    sample_rate : float
        采样率 (Hz)。
    settling_threshold : float
        稳态判定阈值 (占阶跃幅值的比例, 如 0.02 表示 2%)。
    """

    def __init__(
        self,
        sample_rate: float = 100.0,
        settling_threshold: float = 0.02,
    ):
        self.sample_rate = float(sample_rate)
        self.settling_threshold = float(settling_threshold)

        # 阶跃响应数据
        self._step_responses: List[Dict[str, np.ndarray]] = []

        LOGGER.info(
            "ResponseCurveAnalyzer: 初始化完成 (sample_rate=%.1f Hz, threshold=%.2f)",
            self.sample_rate, self.settling_threshold,
        )

    def add_step_response(
        self,
        time_axis: np.ndarray,
        position_response: np.ndarray,
        step_amplitude: float,
        axis: str = "x",
    ) -> None:
        """添加阶跃响应数据。

        Parameters
        ----------
        time_axis : np.ndarray
            时间轴 (秒), 形状 (N,)。
        position_response : np.ndarray
            位置响应 (像素), 形状 (N,)。
        step_amplitude : float
            阶跃幅值 (电机步数)。
        axis : str
            运动轴 ('x' 或 'y')。
        """
        if len(time_axis) != len(position_response):
            raise ValueError("时间轴与响应数据长度不一致")

        self._step_responses.append({
            "time": time_axis.copy(),
            "response": position_response.copy(),
            "amplitude": float(step_amplitude),
            "axis": axis,
        })

        LOGGER.info(
            "ResponseCurveAnalyzer: 添加阶跃响应, axis=%s, 幅值=%.1f 步, %d 个采样点",
            axis, step_amplitude, len(time_axis),
        )

    def analyze_step_response(
        self,
        response_index: int = -1,
    ) -> Dict[str, float]:
        """分析单条阶跃响应。

        Parameters
        ----------
        response_index : int
            响应数据索引, -1 表示最新一条。

        Returns
        -------
        Dict[str, float]
            分析结果字典, 包含:
            - 'rise_time_s': 上升时间 (秒)
            - 'settling_time_s': 调节时间 (秒)
            - 'overshoot_pct': 超调量 (%)
            - 'steady_state_value': 稳态值 (像素)
            - 'sensitivity_px_per_step': 灵敏度 (像素/步)
            - 'dead_zone_steps': 死区 (步数)

        Raises
        ------
        ValueError
            无阶跃响应数据或索引无效。
        """
        if not self._step_responses:
            raise ValueError("无阶跃响应数据")

        idx = response_index if response_index >= 0 else len(self._step_responses) + response_index
        if idx < 0 or idx >= len(self._step_responses):
            raise ValueError(f"无效索引: {response_index}")

        data = self._step_responses[idx]
        t = data["time"]
        y = data["response"]
        amp = data["amplitude"]

        # 稳态值 (取最后 10% 的均值)
        n_steady = max(int(len(y) * 0.1), 1)
        steady_state = float(np.mean(y[-n_steady:]))

        # 阶跃幅值 (像素)
        step_px = steady_state - y[0]

        # 上升时间 (10% -> 90%)
        rise_time = self._compute_rise_time(t, y, y[0], steady_state)

        # 调节时间
        settling_time = self._compute_settling_time(t, y, y[0], steady_state)

        # 超调量
        overshoot = self._compute_overshoot(y, y[0], steady_state)

        # 灵敏度
        sensitivity = abs(step_px / amp) if abs(amp) > 1e-12 else 0.0

        # 死区检测
        dead_zone = self._detect_dead_zone(t, y, amp)

        result = {
            "rise_time_s": rise_time,
            "settling_time_s": settling_time,
            "overshoot_pct": overshoot,
            "steady_state_value": steady_state,
            "sensitivity_px_per_step": sensitivity,
            "dead_zone_steps": dead_zone,
        }

        LOGGER.info(
            "ResponseCurveAnalyzer: 阶跃响应分析 (axis=%s): "
            "上升=%.4f s, 调节=%.4f s, 超调=%.1f%%, 灵敏度=%.4f px/step",
            data["axis"], rise_time, settling_time, overshoot, sensitivity,
        )
        return result

    def estimate_frequency_response(
        self,
        response_index: int = -1,
    ) -> Dict[str, np.ndarray]:
        """从阶跃响应估计频率响应。

        通过对阶跃响应求导得到脉冲响应, 再做 FFT 得到频率响应。

        Parameters
        ----------
        response_index : int
            响应数据索引。

        Returns
        -------
        Dict[str, np.ndarray]
            频率响应数据, 包含:
            - 'frequencies': 频率数组 (Hz)
            - 'magnitude': 幅值响应
            - 'phase': 相位响应 (度)
            - 'bandwidth_hz': -3dB 带宽 (Hz)
        """
        if not self._step_responses:
            raise ValueError("无阶跃响应数据")

        idx = response_index if response_index >= 0 else len(self._step_responses) + response_index
        if idx < 0 or idx >= len(self._step_responses):
            raise ValueError(f"无效索引: {response_index}")

        data = self._step_responses[idx]
        t = data["time"]
        y = data["response"]

        # 脉冲响应 = 阶跃响应的导数
        dt = t[1] - t[0] if len(t) > 1 else 1.0 / self.sample_rate
        impulse = np.diff(y) / dt

        # 加窗减少频谱泄漏
        window = np.hanning(len(impulse))
        impulse_windowed = impulse * window

        # FFT
        N = len(impulse_windowed)
        fft_result = np.fft.rfft(impulse_windowed)
        freqs = np.fft.rfftfreq(N, d=dt)

        # 幅值响应 (归一化)
        magnitude = np.abs(fft_result)
        mag_max = magnitude.max()
        if mag_max > 1e-12:
            magnitude = magnitude / mag_max

        # 相位响应
        phase = np.angle(fft_result, deg=True)

        # -3dB 带宽
        bandwidth = 0.0
        mag_db = 20.0 * np.log10(magnitude + 1e-12)
        above_3db = np.where(mag_db > -3.0)[0]
        if len(above_3db) > 0:
            bandwidth = float(freqs[above_3db[-1]])

        result = {
            "frequencies": freqs,
            "magnitude": magnitude,
            "phase": phase,
            "bandwidth_hz": bandwidth,
        }

        LOGGER.info(
            "ResponseCurveAnalyzer: 频率响应估计, -3dB 带宽=%.1f Hz",
            bandwidth,
        )
        return result

    def assess_linearity(
        self,
        response_index: int = -1,
    ) -> float:
        """评估响应线性度。

        通过拟合线性模型并计算 R^2 来评估线性度。

        Parameters
        ----------
        response_index : int
            响应数据索引。

        Returns
        -------
        float
            线性度 R^2 [0, 1], 1.0 为完美线性。
        """
        if not self._step_responses:
            raise ValueError("无阶跃响应数据")

        idx = response_index if response_index >= 0 else len(self._step_responses) + response_index
        if idx < 0 or idx >= len(self._step_responses):
            raise ValueError(f"无效索引: {response_index}")

        data = self._step_responses[idx]
        y = data["response"]
        t = data["time"]

        # 使用时间作为自变量, 位置作为因变量
        # 对阶跃响应的稳态段做线性回归
        n_steady = max(int(len(y) * 0.5), 2)
        y_steady = y[-n_steady:]
        t_steady = t[-n_steady:]

        # 线性拟合: y = a*t + b
        A_mat = np.column_stack([t_steady, np.ones(len(t_steady))])
        try:
            params, _, _, _ = np.linalg.lstsq(A_mat, y_steady, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0

        y_pred = A_mat @ params
        ss_res = np.sum((y_steady - y_pred) ** 2)
        ss_tot = np.sum((y_steady - np.mean(y_steady)) ** 2)

        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        r_squared = float(np.clip(r_squared, 0.0, 1.0))

        LOGGER.info(
            "ResponseCurveAnalyzer: 线性度 R^2=%.4f", r_squared,
        )
        return r_squared

    def get_comprehensive_analysis(
        self,
    ) -> Dict[str, Any]:
        """综合分析所有阶跃响应。

        Returns
        -------
        Dict[str, Any]
            综合分析结果, 包含各轴的阶跃响应指标和频率响应。
        """
        if not self._step_responses:
            raise ValueError("无阶跃响应数据")

        results: Dict[str, Any] = {
            "step_responses": [],
            "average_sensitivity": 0.0,
            "average_rise_time": 0.0,
            "average_settling_time": 0.0,
            "average_overshoot": 0.0,
            "average_linearity": 0.0,
            "average_dead_zone": 0.0,
        }

        sensitivities = []
        rise_times = []
        settling_times = []
        overshoots = []
        linearities = []
        dead_zones = []

        for i in range(len(self._step_responses)):
            step_result = self.analyze_step_response(i)
            freq_result = self.estimate_frequency_response(i)
            linearity = self.assess_linearity(i)

            step_result["frequency_response"] = freq_result
            step_result["linearity"] = linearity
            results["step_responses"].append(step_result)

            sensitivities.append(step_result["sensitivity_px_per_step"])
            rise_times.append(step_result["rise_time_s"])
            settling_times.append(step_result["settling_time_s"])
            overshoots.append(step_result["overshoot_pct"])
            linearities.append(linearity)
            dead_zones.append(step_result["dead_zone_steps"])

        results["average_sensitivity"] = float(np.mean(sensitivities))
        results["average_rise_time"] = float(np.mean(rise_times))
        results["average_settling_time"] = float(np.mean(settling_times))
        results["average_overshoot"] = float(np.mean(overshoots))
        results["average_linearity"] = float(np.mean(linearities))
        results["average_dead_zone"] = float(np.mean(dead_zones))

        LOGGER.info(
            "ResponseCurveAnalyzer: 综合分析完成, "
            "平均灵敏度=%.4f px/step, 平均上升时间=%.4f s, 平均线性度=%.4f",
            results["average_sensitivity"],
            results["average_rise_time"],
            results["average_linearity"],
        )
        return results

    def clear_data(self) -> None:
        """清除所有阶跃响应数据。"""
        self._step_responses.clear()
        LOGGER.info("ResponseCurveAnalyzer: 数据已清除")

    def _compute_rise_time(
        self,
        t: np.ndarray,
        y: np.ndarray,
        y_start: float,
        y_end: float,
    ) -> float:
        """计算上升时间 (10% -> 90%)。"""
        step_range = y_end - y_start
        if abs(step_range) < 1e-12:
            return 0.0

        y_10 = y_start + 0.1 * step_range
        y_90 = y_start + 0.9 * step_range

        # 找到 10% 和 90% 的时间点
        above_10 = np.where(y >= y_10)[0]
        above_90 = np.where(y >= y_90)[0]

        if len(above_10) == 0 or len(above_90) == 0:
            return float(t[-1] - t[0])

        t_10 = t[above_10[0]]
        t_90 = t[above_90[0]]

        return float(t_90 - t_10)

    def _compute_settling_time(
        self,
        t: np.ndarray,
        y: np.ndarray,
        y_start: float,
        y_end: float,
    ) -> float:
        """计算调节时间 (进入稳态带的时间)。"""
        step_range = y_end - y_start
        if abs(step_range) < 1e-12:
            return 0.0

        band = self.settling_threshold * abs(step_range)

        # 从后向前搜索, 找到第一个超出稳态带的点
        for i in range(len(y) - 1, -1, -1):
            if abs(y[i] - y_end) > band:
                if i < len(y) - 1:
                    return float(t[i + 1] - t[0])
                break

        return float(t[-1] - t[0])

    def _compute_overshoot(
        self,
        y: np.ndarray,
        y_start: float,
        y_end: float,
    ) -> float:
        """计算超调量 (%)。"""
        step_range = y_end - y_start
        if abs(step_range) < 1e-12:
            return 0.0

        # 超调量 = (最大值 - 稳态值) / 稳态值 * 100%
        if step_range > 0:
            peak = float(np.max(y))
        else:
            peak = float(np.min(y))

        overshoot = abs(peak - y_end) / abs(step_range) * 100.0
        return float(max(overshoot, 0.0))

    def _detect_dead_zone(
        self,
        t: np.ndarray,
        y: np.ndarray,
        step_amplitude: float,
    ) -> float:
        """检测死区大小。

        死区定义为: 输入变化但输出无明显变化的最小输入范围。

        Returns
        -------
        float
            估计死区大小 (电机步数)。
        """
        if len(y) < 2 or abs(step_amplitude) < 1e-12:
            return 0.0

        # 计算输出变化量
        output_change = np.abs(np.diff(y))
        noise_level = np.median(output_change) if len(output_change) > 0 else 0.0

        # 找到输出变化首次超过噪声水平 3 倍的时间点
        threshold = max(noise_level * 3.0, 1e-6)
        significant = np.where(output_change > threshold)[0]

        if len(significant) == 0:
            # 整个响应都在噪声范围内, 死区等于阶跃幅值
            return float(abs(step_amplitude))

        # 死区时间
        dead_zone_time = t[significant[0]]

        # 将时间转换为步数 (线性近似)
        total_time = t[-1] - t[0]
        if total_time > 1e-12:
            dead_zone_steps = abs(step_amplitude) * (dead_zone_time / total_time)
        else:
            dead_zone_steps = 0.0

        return float(dead_zone_steps)


# ======================== 自动标定引擎 ========================


class AutoCalibrationEngine:
    """自动系统标定引擎。

    编排完整的自动标定流程, 包括:
    1. 网格扫描坐标映射标定
    2. 仿射变换与畸变校正
    3. 迟滞特性测量与补偿
    4. 阶跃响应分析
    5. 标定质量评估

    Parameters
    ----------
    config : CalibrationConfig or None
        标定配置。为 None 时使用默认配置。
    image_size : Tuple[int, int] or None
        图像尺寸 (宽, 高)。为 None 时默认 (640, 480)。
    """

    def __init__(
        self,
        config: Optional[CalibrationConfig] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        self.config = config or CalibrationConfig()
        self.image_size = image_size or (640, 480)

        # 子模块
        self.mapper = CoordinateMapper(
            image_center=(self.image_size[0] / 2.0, self.image_size[1] / 2.0),
            enable_distortion=self.config.enable_distortion_correction,
        )
        self.hysteresis_compensator = HysteresisCompensator()
        self.response_analyzer = ResponseCurveAnalyzer()

        # 标定结果
        self._result = CalibrationResult()
        self._report = CalibrationReport()

        # 标定状态
        self._is_calibrated = False
        self._calibration_start_time = 0.0

        # 外部回调 (由用户设置, 用于实际硬件交互)
        self._move_to_callback: Optional[Callable[[int, int], None]] = None
        self._capture_spot_callback: Optional[Callable[[], Tuple[float, float]]] = None

        LOGGER.info(
            "AutoCalibrationEngine: 初始化完成 "
            "(grid=%dx%d, step=%d, distortion=%s, hysteresis=%s)",
            self.config.grid_points_per_axis,
            self.config.grid_points_per_axis,
            self.config.grid_step_size,
            self.config.enable_distortion_correction,
            self.config.enable_hysteresis_compensation,
        )

    @property
    def is_calibrated(self) -> bool:
        """系统是否已完成标定。"""
        return self._is_calibrated

    @property
    def result(self) -> CalibrationResult:
        """获取标定结果。"""
        return self._result

    @property
    def report(self) -> CalibrationReport:
        """获取标定报告。"""
        return self._report

    def set_hardware_callbacks(
        self,
        move_to: Callable[[int, int], None],
        capture_spot: Callable[[], Tuple[float, float]],
    ) -> None:
        """设置硬件交互回调。

        Parameters
        ----------
        move_to : Callable[[int, int], None]
            电机移动回调, 接收 (x_steps, y_steps)。
        capture_spot : Callable[[], Tuple[float, float]]
            光斑位置采集回调, 返回 (pixel_x, pixel_y)。
        """
        self._move_to_callback = move_to
        self._capture_spot_callback = capture_spot
        LOGGER.info("AutoCalibrationEngine: 硬件回调已设置")

    def run_full_calibration(
        self,
        pixel_physical_pairs: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> CalibrationResult:
        """运行完整自动标定流程。

        Parameters
        ----------
        pixel_physical_pairs : List[Tuple[float, float, float, float]] or None
            预采集的标定点数据列表, 每个元素为
            (pixel_x, pixel_y, physical_x, physical_y)。
            为 None 时使用硬件回调自动采集。

        Returns
        -------
        CalibrationResult
            标定结果。

        Raises
        ------
        RuntimeError
            标定超时或硬件回调未设置。
        ValueError
            标定数据不足。
        """
        self._calibration_start_time = time.time()
        LOGGER.info("AutoCalibrationEngine: 开始完整标定流程")

        try:
            # 阶段 1: 坐标映射标定
            self._run_coordinate_calibration(pixel_physical_pairs)

            # 检查超时
            self._check_timeout()

            # 阶段 2: 畸变校正 (可选)
            if self.config.enable_distortion_correction:
                self._run_distortion_calibration()

            # 检查超时
            self._check_timeout()

            # 阶段 3: 迟滞测量与补偿 (可选)
            if self.config.enable_hysteresis_compensation:
                self._run_hysteresis_calibration()

            # 检查超时
            self._check_timeout()

            # 阶段 4: 响应曲线分析
            self._run_response_analysis()

            # 阶段 5: 质量评估
            self._assess_calibration_quality()

            self._is_calibrated = True
            LOGGER.info(
                "AutoCalibrationEngine: 标定完成, "
                "质量评分=%.1f, RMS=%.3f px, 有效=%s",
                self._result.calibration_quality_score,
                self._result.rms_error_px,
                self._result.is_valid,
            )

        except Exception as e:
            LOGGER.error("AutoCalibrationEngine: 标定失败: %s", str(e))
            self._result.is_valid = False
            self._result.calibration_quality_score = 0.0
            raise

        return self._result

    def run_coordinate_calibration_only(
        self,
        pixel_physical_pairs: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> CalibrationResult:
        """仅运行坐标映射标定。

        Parameters
        ----------
        pixel_physical_pairs : List[Tuple[float, float, float, float]] or None
            预采集标定点数据。

        Returns
        -------
        CalibrationResult
            标定结果。
        """
        self._calibration_start_time = time.time()
        LOGGER.info("AutoCalibrationEngine: 开始坐标映射标定")

        self._run_coordinate_calibration(pixel_physical_pairs)

        if self.config.enable_distortion_correction:
            self._run_distortion_calibration()

        self._assess_calibration_quality()
        self._is_calibrated = True

        return self._result

    def calibrate_from_grid_data(
        self,
        grid_pixel_positions: np.ndarray,
        grid_physical_positions: np.ndarray,
    ) -> CalibrationResult:
        """从网格数据标定。

        Parameters
        ----------
        grid_pixel_positions : np.ndarray
            像素坐标数组, 形状 (N, 2)。
        grid_physical_positions : np.ndarray
            物理坐标数组, 形状 (N, 2)。

        Returns
        -------
        CalibrationResult
            标定结果。
        """
        if grid_pixel_positions.shape != grid_physical_positions.shape:
            raise ValueError("像素坐标与物理坐标数组形状不一致")
        if grid_pixel_positions.shape[0] < self.config.min_calibration_points:
            raise ValueError(
                f"标定点数不足: 需要 >= {self.config.min_calibration_points}, "
                f"实际 {grid_pixel_positions.shape[0]}"
            )

        self.mapper.clear_calibration_points()

        for i in range(len(grid_pixel_positions)):
            self.mapper.add_calibration_point(
                pixel_x=float(grid_pixel_positions[i, 0]),
                pixel_y=float(grid_pixel_positions[i, 1]),
                physical_x=float(grid_physical_positions[i, 0]),
                physical_y=float(grid_physical_positions[i, 1]),
            )

        # 拟合仿射变换
        rms_affine = self.mapper.fit_affine()

        # 拟合畸变 (可选)
        if self.config.enable_distortion_correction and len(grid_pixel_positions) >= 6:
            rms_distorted = self.mapper.fit_distortion()
            self._result.rms_error_px = rms_distorted
        else:
            self._result.rms_error_px = rms_affine

        self._result.affine_matrix = self.mapper.affine_matrix
        self._result.distortion_coefficients = self.mapper.distortion_coefficients
        self._result.calibration_timestamp = time.time()

        self._assess_calibration_quality()
        self._is_calibrated = True

        LOGGER.info(
            "AutoCalibrationEngine: 网格数据标定完成, RMS=%.3f px",
            self._result.rms_error_px,
        )
        return self._result

    def generate_calibration_report(self) -> CalibrationReport:
        """生成标定报告。

        Returns
        -------
        CalibrationReport
            详细标定报告。
        """
        report = CalibrationReport()

        # 坐标映射精度
        try:
            report.coordinate_mapping_accuracy = self.mapper.assess_accuracy()
        except ValueError:
            report.coordinate_mapping_accuracy = float("inf")

        # 迟滞残余
        try:
            report.hysteresis_residual = (
                self.hysteresis_compensator.estimate_residual_hysteresis()
            )
        except ValueError:
            report.hysteresis_residual = float("inf")

        # 响应线性度
        if self.response_analyzer._step_responses:
            try:
                report.response_linearity = self.response_analyzer.assess_linearity()
            except (ValueError, IndexError):
                report.response_linearity = 0.0

        # 灵敏度
        if self.response_analyzer._step_responses:
            try:
                analysis = self.response_analyzer.analyze_step_response()
                report.sensitivity_px_per_step = analysis["sensitivity_px_per_step"]
                report.dead_zone_size = analysis["dead_zone_steps"]
            except (ValueError, IndexError):
                pass

        # 生成改进建议
        report.recommendations = self._generate_recommendations(report)

        self._report = report
        return report

    def validate_calibration(
        self,
        validation_points: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> bool:
        """验证标定结果。

        Parameters
        ----------
        validation_points : List[Tuple[float, float, float, float]] or None
            验证点数据 (pixel_x, pixel_y, physical_x, physical_y)。
            为 None 时使用标定点进行交叉验证。

        Returns
        -------
        bool
            标定是否通过验证。
        """
        if not self._is_calibrated:
            LOGGER.warning("AutoCalibrationEngine: 系统未标定, 验证失败")
            return False

        if validation_points is not None:
            # 使用验证点
            total_sq_error = 0.0
            for px, py, sx, sy in validation_points:
                pred_px, pred_py = self.mapper.physical_to_pixel(sx, sy)
                total_sq_error += (pred_px - px) ** 2 + (pred_py - py) ** 2
            rms = float(np.sqrt(total_sq_error / len(validation_points)))
        else:
            # 使用标定点 (留一交叉验证)
            rms = self._leave_one_out_cv()

        is_valid = rms <= self.config.max_rms_error_px

        LOGGER.info(
            "AutoCalibrationEngine: 标定验证, RMS=%.3f px, 阈值=%.3f px, 通过=%s",
            rms, self.config.max_rms_error_px, is_valid,
        )
        return is_valid

    def get_compensated_position(
        self,
        target_pixel_x: float,
        target_pixel_y: float,
    ) -> Tuple[int, int]:
        """获取迟滞补偿后的电机位置。

        Parameters
        ----------
        target_pixel_x : float
            目标像素 x 坐标。
        target_pixel_y : float
            目标像素 y 坐标。

        Returns
        -------
        Tuple[int, int]
            补偿后的电机步数 (x_steps, y_steps)。
        """
        if not self._is_calibrated:
            raise RuntimeError("系统未标定, 请先运行标定")

        # 像素 -> 物理
        phys_x, phys_y = self.mapper.pixel_to_physical(target_pixel_x, target_pixel_y)

        # 迟滞补偿
        if self.config.enable_hysteresis_compensation:
            comp_x = self.hysteresis_compensator.compensate(phys_x)
            comp_y = self.hysteresis_compensator.compensate(phys_y)
        else:
            comp_x, comp_y = phys_x, phys_y

        return int(round(comp_x)), int(round(comp_y))

    def predict_pixel_position(
        self,
        motor_x: int,
        motor_y: int,
    ) -> Tuple[float, float]:
        """预测给定电机位置的像素坐标。

        Parameters
        ----------
        motor_x : int
            电机 x 步数。
        motor_y : int
            电机 y 步数。

        Returns
        -------
        Tuple[float, float]
            预测像素坐标 (pixel_x, pixel_y)。
        """
        if not self._is_calibrated:
            raise RuntimeError("系统未标定, 请先运行标定")

        return self.mapper.physical_to_pixel(float(motor_x), float(motor_y))

    def reset(self) -> None:
        """重置标定引擎。"""
        self.mapper.clear_calibration_points()
        self.hysteresis_compensator.reset_state()
        self.response_analyzer.clear_data()
        self._result = CalibrationResult()
        self._report = CalibrationReport()
        self._is_calibrated = False
        LOGGER.info("AutoCalibrationEngine: 已重置")

    # ======================== 内部方法 ========================

    def _run_coordinate_calibration(
        self,
        pixel_physical_pairs: Optional[List[Tuple[float, float, float, float]]],
    ) -> None:
        """执行坐标映射标定。"""
        self.mapper.clear_calibration_points()

        if pixel_physical_pairs is not None:
            # 使用预采集数据
            for px, py, sx, sy in pixel_physical_pairs:
                self.mapper.add_calibration_point(px, py, sx, sy)
            LOGGER.info(
                "AutoCalibrationEngine: 使用 %d 个预采集标定点",
                len(pixel_physical_pairs),
            )
        else:
            # 自动网格扫描
            self._auto_grid_scan()

        # 检查标定点数
        n_points = len(self.mapper._pixel_points)
        if n_points < self.config.min_calibration_points:
            raise ValueError(
                f"标定点数不足: 需要 >= {self.config.min_calibration_points}, "
                f"实际 {n_points}"
            )

        # 拟合仿射变换
        rms = self.mapper.fit_affine()
        self._result.rms_error_px = rms
        self._result.affine_matrix = self.mapper.affine_matrix

    def _auto_grid_scan(self) -> None:
        """自动网格扫描采集标定点。"""
        if self._move_to_callback is None or self._capture_spot_callback is None:
            raise RuntimeError(
                "自动网格扫描需要硬件回调, "
                "请先调用 set_hardware_callbacks() 或提供预采集数据"
            )

        n = self.config.grid_points_per_axis
        step = self.config.grid_step_size

        # 计算网格中心偏移 (使网格关于原点对称)
        offset = (n - 1) * step // 2

        LOGGER.info(
            "AutoCalibrationEngine: 开始自动网格扫描 (%dx%d, 步距=%d)",
            n, n, step,
        )

        total_points = n * n
        for row in range(n):
            for col in range(n):
                sx = col * step - offset
                sy = row * step - offset

                # 移动电机
                self._move_to_callback(sx, sy)

                # 等待稳定
                time.sleep(self.config.calibration_settle_time_s)

                # 采集光斑位置
                px, py = self._capture_spot_callback()

                self.mapper.add_calibration_point(px, py, float(sx), float(sy))

                LOGGER.debug(
                    "AutoCalibrationEngine: 网格点 (%d/%d), "
                    "物理=(%d, %d), 像素=(%.1f, %.1f)",
                    row * n + col + 1, total_points, sx, sy, px, py,
                )

        LOGGER.info(
            "AutoCalibrationEngine: 网格扫描完成, 共 %d 个点",
            total_points,
        )

    def _run_distortion_calibration(self) -> None:
        """执行畸变校正标定。"""
        n_points = len(self.mapper._pixel_points)
        if n_points < 6:
            LOGGER.warning(
                "AutoCalibrationEngine: 标定点数不足 (%d < 6), 跳过畸变校正",
                n_points,
            )
            return

        try:
            rms = self.mapper.fit_distortion()
            self._result.rms_error_px = rms
            self._result.distortion_coefficients = self.mapper.distortion_coefficients
            LOGGER.info(
                "AutoCalibrationEngine: 畸变校正完成, RMS=%.4f px",
                rms,
            )
        except (ValueError, np.linalg.LinAlgError) as e:
            LOGGER.warning("AutoCalibrationEngine: 畸变校正失败: %s", str(e))

    def _run_hysteresis_calibration(self) -> None:
        """执行迟滞标定。"""
        if self._move_to_callback is None or self._capture_spot_callback is None:
            LOGGER.warning(
                "AutoCalibrationEngine: 无硬件回调, 跳过迟滞标定"
            )
            return

        n_points = self.config.hysteresis_test_points
        amplitude = self.config.response_test_amplitude
        n_cycles = self.config.hysteresis_directions

        LOGGER.info(
            "AutoCalibrationEngine: 开始迟滞标定 "
            "(点数=%d, 幅值=%d, 循环=%d)",
            n_points, amplitude, n_cycles,
        )

        # 生成测试位置序列
        positions = np.linspace(0, amplitude, n_points)

        for cycle in range(n_cycles):
            # 正向行程
            forward_output = np.zeros(n_points, dtype=np.float64)
            for i, pos in enumerate(positions):
                self._move_to_callback(int(pos), int(pos))
                time.sleep(self.config.calibration_settle_time_s)
                px, py = self._capture_spot_callback()
                forward_output[i] = np.sqrt(px ** 2 + py ** 2)

            # 反向行程
            backward_output = np.zeros(n_points, dtype=np.float64)
            for i, pos in enumerate(reversed(positions)):
                self._move_to_callback(int(pos), int(pos))
                time.sleep(self.config.calibration_settle_time_s)
                px, py = self._capture_spot_callback()
                backward_output[n_points - 1 - i] = np.sqrt(px ** 2 + py ** 2)

            self.hysteresis_compensator.add_hysteresis_data(
                positions, forward_output, backward_output,
            )

        # 拟合 Play 算子
        try:
            residual = self.hysteresis_compensator.fit_play_operators()
            self._result.hysteresis_model = {
                "weights": self.hysteresis_compensator.weights.tolist(),
                "thresholds": self.hysteresis_compensator.thresholds.tolist(),
                "fitting_residual_px": residual,
                "num_operators": self.hysteresis_compensator.num_play_operators,
            }

            # 生成补偿查找表
            lut = self.hysteresis_compensator.generate_compensation_lut()
            self._result.hysteresis_model["lut_size"] = len(lut)

            LOGGER.info(
                "AutoCalibrationEngine: 迟滞标定完成, 残差=%.4f px",
                residual,
            )
        except (ValueError, np.linalg.LinAlgError) as e:
            LOGGER.warning("AutoCalibrationEngine: 迟滞标定失败: %s", str(e))

    def _run_response_analysis(self) -> None:
        """执行阶跃响应分析。"""
        if self._move_to_callback is None or self._capture_spot_callback is None:
            LOGGER.warning(
                "AutoCalibrationEngine: 无硬件回调, 跳过响应分析"
            )
            return

        amplitude = self.config.response_test_amplitude
        n_samples = self.config.response_test_points
        dt = 1.0 / self.response_analyzer.sample_rate

        LOGGER.info(
            "AutoCalibrationEngine: 开始阶跃响应分析 "
            "(幅值=%d 步, 采样=%d 点)",
            amplitude, n_samples,
        )

        # 回到原点
        self._move_to_callback(0, 0)
        time.sleep(self.config.calibration_settle_time_s * 2)

        # X 轴阶跃
        time_axis = np.arange(n_samples) * dt
        x_response = np.zeros(n_samples, dtype=np.float64)
        self._move_to_callback(amplitude, 0)
        for i in range(n_samples):
            px, py = self._capture_spot_callback()
            x_response[i] = px
            time.sleep(dt)

        self.response_analyzer.add_step_response(
            time_axis, x_response, float(amplitude), axis="x",
        )

        # 回到原点
        self._move_to_callback(0, 0)
        time.sleep(self.config.calibration_settle_time_s * 2)

        # Y 轴阶跃
        y_response = np.zeros(n_samples, dtype=np.float64)
        self._move_to_callback(0, amplitude)
        for i in range(n_samples):
            px, py = self._capture_spot_callback()
            y_response[i] = py
            time.sleep(dt)

        self.response_analyzer.add_step_response(
            time_axis, y_response, float(amplitude), axis="y",
        )

        # 回到原点
        self._move_to_callback(0, 0)

        # 综合分析
        try:
            analysis = self.response_analyzer.get_comprehensive_analysis()
            self._result.response_curve_data = {
                "average_sensitivity": analysis["average_sensitivity"],
                "average_rise_time_s": analysis["average_rise_time"],
                "average_settling_time_s": analysis["average_settling_time"],
                "average_overshoot_pct": analysis["average_overshoot"],
                "average_linearity": analysis["average_linearity"],
                "average_dead_zone_steps": analysis["average_dead_zone"],
            }
            LOGGER.info(
                "AutoCalibrationEngine: 响应分析完成, "
                "灵敏度=%.4f px/step",
                analysis["average_sensitivity"],
            )
        except (ValueError, IndexError) as e:
            LOGGER.warning("AutoCalibrationEngine: 响应分析失败: %s", str(e))

    def _assess_calibration_quality(self) -> None:
        """评估标定质量。"""
        score = 0.0

        # 1. 坐标映射精度 (40 分)
        rms = self._result.rms_error_px
        if rms < 0.5:
            score += 40.0
        elif rms < 1.0:
            score += 35.0
        elif rms < 2.0:
            score += 25.0
        elif rms < 5.0:
            score += 15.0
        else:
            score += 5.0

        # 2. 畸变校正 (15 分)
        if self.config.enable_distortion_correction:
            k = self._result.distortion_coefficients
            if np.allclose(k, 0):
                # 无显著畸变
                score += 15.0
            elif rms < 1.0:
                score += 15.0
            else:
                score += 8.0
        else:
            score += 10.0  # 未启用畸变校正, 给部分分数

        # 3. 迟滞补偿 (20 分)
        if self.config.enable_hysteresis_compensation and self._result.hysteresis_model:
            residual = self._result.hysteresis_model.get("fitting_residual_px", float("inf"))
            if residual < 0.5:
                score += 20.0
            elif residual < 1.0:
                score += 15.0
            elif residual < 2.0:
                score += 10.0
            else:
                score += 5.0
        else:
            score += 10.0

        # 4. 响应线性度 (15 分)
        linearity = self._result.response_curve_data.get("average_linearity", 0.0)
        score += linearity * 15.0

        # 5. 死区 (10 分)
        dead_zone = self._result.response_curve_data.get("average_dead_zone_steps", 100.0)
        if dead_zone < 5:
            score += 10.0
        elif dead_zone < 20:
            score += 7.0
        elif dead_zone < 50:
            score += 4.0
        else:
            score += 1.0

        self._result.calibration_quality_score = float(np.clip(score, 0.0, 100.0))
        self._result.calibration_timestamp = time.time()
        self._result.is_valid = (
            self._result.rms_error_px <= self.config.max_rms_error_px
            and self._result.calibration_quality_score >= 50.0
        )

        LOGGER.info(
            "AutoCalibrationEngine: 质量评估, 总分=%.1f, 有效=%s",
            self._result.calibration_quality_score,
            self._result.is_valid,
        )

    def _check_timeout(self) -> None:
        """检查标定是否超时。"""
        elapsed = time.time() - self._calibration_start_time
        if elapsed > self.config.max_calibration_time_s:
            raise RuntimeError(
                f"标定超时: 已用时 {elapsed:.1f} s, "
                f"最大允许 {self.config.max_calibration_time_s:.1f} s"
            )

    def _leave_one_out_cv(self) -> float:
        """留一交叉验证计算 RMS 误差。"""
        points = list(zip(self.mapper._pixel_points, self.mapper._physical_points))
        n = len(points)
        if n < 4:
            return self.mapper.assess_accuracy()

        total_sq_error = 0.0

        for i in range(n):
            # 临时移除第 i 个点
            test_pixel = points[i][0]
            test_physical = points[i][1]

            # 用剩余点拟合
            train_pixels = [p for j, (p, _) in enumerate(points) if j != i]
            train_physicals = [p for j, (_, p) in enumerate(points) if j != i]

            # 构建最小二乘问题
            m = len(train_pixels)
            A_mat = np.zeros((2 * m, 6), dtype=np.float64)
            b_vec = np.zeros(2 * m, dtype=np.float64)
            for k in range(m):
                sx, sy = train_physicals[k]
                px, py = train_pixels[k]
                A_mat[2 * k, :] = [sx, sy, 1.0, 0.0, 0.0, 0.0]
                A_mat[2 * k + 1, :] = [0.0, 0.0, 0.0, sx, sy, 1.0]
                b_vec[2 * k] = px
                b_vec[2 * k + 1] = py

            try:
                params, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                a, b, tx, c, d, ty = params
                pred_x = a * test_physical[0] + b * test_physical[1] + tx
                pred_y = c * test_physical[0] + d * test_physical[1] + ty
                total_sq_error += (
                    (pred_x - test_pixel[0]) ** 2
                    + (pred_y - test_pixel[1]) ** 2
                )
            except np.linalg.LinAlgError:
                total_sq_error += float("inf")
                break

        return float(np.sqrt(total_sq_error / n))

    def _generate_recommendations(self, report: CalibrationReport) -> List[str]:
        """生成改进建议。"""
        recommendations = []

        # 坐标映射精度
        if report.coordinate_mapping_accuracy > self.config.max_rms_error_px:
            recommendations.append(
                f"坐标映射精度不足 (RMS={report.coordinate_mapping_accuracy:.2f} px > "
                f"{self.config.max_rms_error_px:.2f} px), 建议增加标定点数或减小网格步距"
            )
        elif report.coordinate_mapping_accuracy > 1.0:
            recommendations.append(
                f"坐标映射精度一般 (RMS={report.coordinate_mapping_accuracy:.2f} px), "
                f"可通过增加标定点数进一步改善"
            )

        # 迟滞
        if report.hysteresis_residual > 1.0:
            recommendations.append(
                f"迟滞残余较大 ({report.hysteresis_residual:.2f} px), "
                f"建议增加 Play 算子数量或迟滞测试循环次数"
            )

        # 线性度
        if report.response_linearity < 0.9:
            recommendations.append(
                f"响应线性度偏低 (R^2={report.response_linearity:.4f}), "
                f"可能存在机械间隙或非线性弹性变形"
            )

        # 死区
        if report.dead_zone_size > 20:
            recommendations.append(
                f"死区较大 ({report.dead_zone_size:.1f} 步), "
                f"建议检查机械传动系统或使用微步驱动"
            )

        # 灵敏度
        if report.sensitivity_px_per_step < 0.01:
            recommendations.append(
                f"灵敏度偏低 ({report.sensitivity_px_per_step:.4f} px/step), "
                f"建议减小标定网格步距或检查光学放大倍率"
            )

        if not recommendations:
            recommendations.append("标定质量良好, 无需改进")

        return recommendations
