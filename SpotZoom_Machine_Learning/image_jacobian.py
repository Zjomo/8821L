"""
图像雅可比视觉伺服控制器 (ImageJacobianController)

灵感来源:
- ViSP (https://github.com/lagadic/visp) — 视觉伺服平台, IBVS 控制律
- OpenCV Visual Servoing (opencv_contrib) — 光流法与特征跟踪
- Chaumette & Hutchinson, "Visual Servo Control", IEEE R&A 2006

算法原理:
- Image Jacobian (Interaction Matrix) — 像素偏差到物理位移的映射
- Online Jacobian Estimation — 在线雅可比矩阵估计 (Broyden 更新)
- Damped Least Squares — 阻尼最小二乘逆 (正则化)
- Adaptive Step Size — 自适应步长控制

功能:
- 建立像素偏差到电机步数的映射模型，替代固定步长
- 在线学习雅可比矩阵，自动适应不同光学系统
- 提供阻尼正则化避免奇异矩阵问题
- 支持增量学习，越用越准

依赖: numpy (无线性代数库之外的外部依赖)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.ImageJacobian")


@dataclass
class JacobianState:
    """雅可比控制器状态。"""
    jacobian: np.ndarray  # 2x2 雅可比矩阵 [[dpx/dsx, dpx/dsy], [dpy/dsx, dpy/dsy]]
    is_calibrated: bool  # 是否已标定
    update_count: int  # 更新次数
    last_pixel_move: Optional[Tuple[float, float]]  # 上次像素位移
    last_stage_move: Optional[Tuple[int, int]]  # 上次电机位移


class ImageJacobianController:
    """图像雅可比视觉伺服控制器。

    通过在线估计图像雅可比矩阵，建立像素偏差到电机步数的
    智能映射关系，实现自适应步长控制。

    Parameters
    ----------
    initial_step_per_px : float
        初始步长/像素比 (未标定时的默认值)。
    damping_factor : float
        阻尼因子 (正则化参数)。值越大越保守。
    learning_rate : float
        Broyden 更新学习率 [0, 1]。
    min_step : int
        最小电机步数。
    max_step : int
        最大电机步数。
    calibration_samples : int
        需要多少次观测后才认为雅可比矩阵已标定。
    """

    def __init__(
        self,
        initial_step_per_px: float = 2.0,
        damping_factor: float = 0.1,
        learning_rate: float = 0.3,
        min_step: int = 50,
        max_step: int = 3000,
        calibration_samples: int = 5,
    ):
        self.initial_step_per_px = float(initial_step_per_px)
        self.damping_factor = float(damping_factor)
        self.learning_rate = float(learning_rate)
        self.min_step = int(min_step)
        self.max_step = int(max_step)
        self.calibration_samples = int(calibration_samples)

        # 初始雅可比矩阵 (对角阵，假设像素与步数线性关系)
        self._J = np.array([
            [initial_step_per_px, 0.0],
            [0.0, initial_step_per_px],
        ], dtype=np.float64)

        self._state = JacobianState(
            jacobian=self._J.copy(),
            is_calibrated=False,
            update_count=0,
            last_pixel_move=None,
            last_stage_move=None,
        )

    @property
    def is_calibrated(self) -> bool:
        """雅可比矩阵是否已标定。"""
        return self._state.is_calibrated

    @property
    def state(self) -> JacobianState:
        """获取当前控制器状态。"""
        return self._state

    def compute_steps(
        self,
        pixel_error_x: int,
        pixel_error_y: int,
    ) -> Tuple[int, int]:
        """根据像素偏差计算电机步数。

        Parameters
        ----------
        pixel_error_x : int
            X 方向像素偏差 (目标 - 当前)。
        pixel_error_y : int
            Y 方向像素偏差 (目标 - 当前)。

        Returns
        -------
        Tuple[int, int]
            (x_steps, y_steps) 电机步数。
        """
        error = np.array([float(pixel_error_x), float(pixel_error_y)], dtype=np.float64)

        if abs(error[0]) < 1 and abs(error[1]) < 1:
            return (0, 0)

        # 阻尼最小二乘逆: J^# = J^T (J J^T + λ²I)^{-1}
        JJT = self._J @ self._J.T
        damped = JJT + self.damping_factor ** 2 * np.eye(2, dtype=np.float64)
        try:
            damped_inv = np.linalg.inv(damped)
        except np.linalg.LinAlgError:
            # 退化时使用初始比例
            x_steps = int(round(self.initial_step_per_px * pixel_error_x))
            y_steps = int(round(self.initial_step_per_px * pixel_error_y))
            return self._clamp(x_steps, y_steps)

        J_pinv = self._J.T @ damped_inv
        stage_move = J_pinv @ error

        x_steps = int(round(stage_move[0]))
        y_steps = int(round(stage_move[1]))

        # 保存用于后续更新
        self._state.last_pixel_move = (float(pixel_error_x), float(pixel_error_y))
        self._state.last_stage_move = (x_steps, y_steps)

        return self._clamp(x_steps, y_steps)

    def update_jacobian(
        self,
        pixel_before: Tuple[int, int],
        pixel_after: Tuple[int, int],
        stage_steps: Tuple[int, int],
    ) -> None:
        """用观测数据更新雅可比矩阵 (Broyden 方法)。

        Parameters
        ----------
        pixel_before : Tuple[int, int]
            移动前的像素位置。
        pixel_after : Tuple[int, int]
            移动后的像素位置。
        stage_steps : Tuple[int, int]
            实际执行的电机步数。
        """
        dp = np.array([
            float(pixel_after[0] - pixel_before[0]),
            float(pixel_after[1] - pixel_before[1]),
        ], dtype=np.float64)

        ds = np.array([float(stage_steps[0]), float(stage_steps[1])], dtype=np.float64)

        ds_norm = np.linalg.norm(ds)
        dp_norm = np.linalg.norm(dp)

        # 过滤无效更新 (步数太小或像素变化太小)
        if ds_norm < 10 or dp_norm < 0.5:
            return

        # Broyden 更新: J_{k+1} = J_k + (dp - J_k ds) ds^T / (ds^T ds)
        predicted_dp = self._J @ ds
        residual = dp - predicted_dp
        ds_dsT = np.outer(ds, ds) / (ds_norm ** 2)
        self._J = self._J + self.learning_rate * residual.reshape(2, 1) * ds.reshape(1, 2) / (ds_norm ** 2)

        self._state.jacobian = self._J.copy()
        self._state.update_count += 1

        if self._state.update_count >= self.calibration_samples and not self._state.is_calibrated:
            self._state.is_calibrated = True
            LOGGER.info(
                "ImageJacobian calibrated after %d updates. J = [[%.3f, %.3f], [%.3f, %.3f]]",
                self._state.update_count,
                self._J[0, 0], self._J[0, 1],
                self._J[1, 0], self._J[1, 1],
            )

    def _clamp(self, x: int, y: int) -> Tuple[int, int]:
        """限幅电机步数。"""
        x = max(-self.max_step, min(self.max_step, x))
        y = max(-self.max_step, min(self.max_step, y))
        # 最小步数 (避免微小抖动)
        if 0 < abs(x) < self.min_step:
            x = self.min_step if x > 0 else -self.min_step
        if 0 < abs(y) < self.min_step:
            y = self.min_step if y > 0 else -self.min_step
        return (x, y)

    def get_jacobian_matrix(self) -> Tuple[float, float, float, float]:
        """获取当前雅可比矩阵元素。

        Returns
        -------
        Tuple[float, float, float, float]
            (J00, J01, J10, J11)
        """
        return (
            float(self._J[0, 0]), float(self._J[0, 1]),
            float(self._J[1, 0]), float(self._J[1, 1]),
        )

    def reset(self) -> None:
        """重置雅可比矩阵到初始状态。"""
        self._J = np.array([
            [self.initial_step_per_px, 0.0],
            [0.0, self.initial_step_per_px],
        ], dtype=np.float64)
        self._state = JacobianState(
            jacobian=self._J.copy(),
            is_calibrated=False,
            update_count=0,
            last_pixel_move=None,
            last_stage_move=None,
        )
        LOGGER.info("ImageJacobian reset to initial state")
