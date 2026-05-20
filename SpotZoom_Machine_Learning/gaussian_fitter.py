"""
二维高斯光束拟合器 (GaussianBeamFitter)

灵感来源:
- pyBeamProfiling (https://github.com/JohnBriggs/pyBeamProfiling) — 二维高斯光束剖面拟合
- Laser-beam-profiler-camera — 激光光束分析
- ISO 13694 — 激光光束剖面测量标准
- ISO 11146 — 激光光束宽度、发散角与束散角测量

算法原理:
- Iterative Levenberg-Marquardt — 阻尼最小二乘迭代拟合 (Gauss-Newton + 阻尼因子)
- Moment-Based Estimation — 基于图像矩的快速参数估计 (零阶/一阶/二阶矩)
- Jacobian Computation — 解析雅可比矩阵计算 (7 参数二维旋转高斯模型)
- M-squared Estimation — M² 光束质量因子估计

功能:
- 对光斑图像区域进行精确的二维旋转高斯拟合
- 提取完整光束参数: 中心、宽度 (sigma_x, sigma_y)、振幅、角度、椭圆度
- 提供快速矩估计模式 (无需迭代，适用于实时场景)
- 计算 FWHM、椭圆度、峰值信噪比、拟合残差等派生指标
- M² 光束质量因子估计
- 拟合有效性检验

依赖: numpy, logging, dataclasses
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.GaussianFitter")

# 物理常数
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # FWHM = 2*sqrt(2*ln2) * sigma ≈ 2.3548


@dataclass
class BeamProfile:
    """拟合得到的光束剖面参数。

    Attributes
    ----------
    center_x : float
        光束中心 x 坐标 (像素，相对于输入图像)。
    center_y : float
        光束中心 y 坐标 (像素，相对于输入图像)。
    sigma_x : float
        x 方向高斯标准差 (像素)。
    sigma_y : float
        y 方向高斯标准差 (像素)。
    amplitude : float
        高斯峰值振幅 (灰度单位)。
    offset : float
        背景偏移量 (灰度单位)。
    angle_deg : float
        高斯椭圆主轴旋转角度 (度)。逆时针为正。
    fwhm_x : float
        x 方向半高全宽 (像素)。
    fwhm_y : float
        y 方向半高全宽 (像素)。
    ellipticity : float
        椭圆度 = min(sigma_x, sigma_y) / max(sigma_x, sigma_y)。1.0 为完美圆形。
    peak_snr : float
        峰值信噪比 = amplitude / noise_std。
    fit_residual : float
        归一化拟合残差 (均方根)。
    converged : bool
        迭代拟合是否收敛。
    """
    center_x: float
    center_y: float
    sigma_x: float
    sigma_y: float
    amplitude: float
    offset: float
    angle_deg: float
    fwhm_x: float
    fwhm_y: float
    ellipticity: float
    peak_snr: float
    fit_residual: float
    converged: bool


class GaussianBeamFitter:
    """二维高斯光束拟合器。

    对光斑图像区域进行精确的二维旋转高斯函数拟合，
    提取光束中心位置、光束宽度、振幅、旋转角度等关键参数。

    支持两种拟合模式:
    - fit(): 完整的 Levenberg-Marquardt 阻尼最小二乘迭代拟合，精度高
    - fit_quick(): 基于图像矩的快速估计，速度快但精度略低

    Parameters
    ----------
    roi_size : int
        感兴趣区域 (ROI) 的默认边长 (像素)。用于从大图中裁剪拟合区域。
    max_iterations : int
        Levenberg-Marquardt 最大迭代次数。
    convergence_threshold : float
        收敛判定阈值 (参数变化量的二范数)。
    """

    def __init__(
        self,
        roi_size: int = 64,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-6,
    ):
        self.roi_size = int(roi_size)
        self.max_iterations = int(max_iterations)
        self.convergence_threshold = float(convergence_threshold)

        # 内部状态
        self._last_params: Optional[np.ndarray] = None
        self._last_residual: float = 0.0
        self._fit_count: int = 0

        LOGGER.debug(
            "GaussianFitter: 初始化完成 (roi_size=%d, max_iter=%d, tol=%.1e)",
            self.roi_size, self.max_iterations, self.convergence_threshold,
        )

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def fit(
        self,
        image: np.ndarray,
        center_hint: Optional[Tuple[float, float]] = None,
    ) -> BeamProfile:
        """对图像进行二维旋转高斯迭代拟合。

        使用 Levenberg-Marquardt 阻尼最小二乘法拟合 7 参数二维高斯模型:
            G(x,y) = A * exp(-(x'^2/(2*sigma_x^2) + y'^2/(2*sigma_y^2))) + offset
        其中 (x', y') 为旋转后的坐标:
            x' =  (x-cx)*cos(theta) + (y-cy)*sin(theta)
            y' = -(x-cx)*sin(theta) + (y-cy)*cos(theta)

        Parameters
        ----------
        image : np.ndarray
            输入图像 (灰度，二维数组)。如果图像大于 roi_size，
            将根据 center_hint 或图像质心裁剪 ROI。
        center_hint : Tuple[float, float] or None
            光束中心位置的先验估计 (x, y)。用于 ROI 裁剪和参数初始化。

        Returns
        -------
        BeamProfile
            拟合得到的光束剖面参数。
        """
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 2:
            raise ValueError(f"输入图像必须是二维数组，当前维度: {image.ndim}")

        # ROI 裁剪
        roi, offset_x, offset_y = self._extract_roi(image, center_hint)
        h, w = roi.shape

        if h < 5 or w < 5:
            LOGGER.warning("GaussianFitter: ROI 太小 (%dx%d)，返回默认值", h, w)
            return self._make_default_profile(offset_x, offset_y)

        # 噪声估计
        noise_std = self._estimate_noise(roi)

        # 初始参数 (基于图像矩)
        params = self._moment_initialization(roi, center_hint)

        LOGGER.debug(
            "GaussianFitter: 初始参数 [cx=%.2f, cy=%.2f, sx=%.2f, sy=%.2f, A=%.1f, off=%.1f, theta=%.2f]",
            params[0], params[1], params[2], params[3], params[4], params[5], params[6],
        )

        # 构建坐标网格
        yy, xx = np.mgrid[:h, :w]
        xx_flat = xx.ravel().astype(np.float64)
        yy_flat = yy.ravel().astype(np.float64)
        data_flat = roi.ravel()

        # Levenberg-Marquardt 迭代
        converged = False
        lam = 1e-3  # 初始阻尼因子
        prev_cost = np.inf

        for iteration in range(self.max_iterations):
            # 计算模型值和残差
            model_flat = self._gaussian_2d_flat(xx_flat, yy_flat, params)
            residuals = data_flat - model_flat

            # 当前代价函数 (残差平方和)
            cost = float(np.dot(residuals, residuals))

            # 计算雅可比矩阵
            J = self._jacobian(xx_flat, yy_flat, params)

            # 法方程: (J^T J + lambda * diag(J^T J)) * delta = J^T * r
            JtJ = J.T @ J
            Jtr = J.T @ residuals

            # 阻尼项: 使用 JtJ 对角线元素
            jtj_diag = np.diag(JtJ).copy()
            jtj_diag[jtj_diag < 1e-12] = 1e-12
            damping = lam * np.diag(jtj_diag)

            # 求解增量
            try:
                delta = np.linalg.solve(JtJ + damping, Jtr)
            except np.linalg.LinAlgError:
                # 矩阵奇异，增大阻尼因子
                lam *= 10.0
                LOGGER.debug("GaussianFitter: 第 %d 次迭代法方程奇异，增大阻尼", iteration)
                continue

            # 尝试更新
            new_params = params + delta

            # 参数约束: sigma 必须为正
            new_params[2] = max(new_params[2], 0.5)
            new_params[3] = max(new_params[3], 0.5)

            # 计算新代价
            new_model = self._gaussian_2d_flat(xx_flat, yy_flat, new_params)
            new_residuals = data_flat - new_model
            new_cost = float(np.dot(new_residuals, new_residuals))

            # 代价改善比例 (用于增益比判定)
            gain_ratio = (cost - new_cost) / max(cost, 1e-12)

            if new_cost < cost:
                # 接受更新，减小阻尼因子
                params = new_params
                improvement = cost - new_cost
                lam = max(lam * 0.1, 1e-10)

                # 收敛判定: 使用相对参数变化量 (消除尺度差异)
                param_scale = np.maximum(np.abs(params), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
                relative_change = float(np.linalg.norm(delta / param_scale))
                if relative_change < self.convergence_threshold:
                    converged = True
                    LOGGER.debug(
                        "GaussianFitter: 第 %d 次迭代收敛 (relative_change=%.2e)",
                        iteration + 1, relative_change,
                    )
                    break

                # 代价改善过小也视为收敛
                if gain_ratio < 1e-10:
                    converged = True
                    LOGGER.debug(
                        "GaussianFitter: 第 %d 次迭代收敛 (cost改善=%.2e)",
                        iteration + 1, gain_ratio,
                    )
                    break
            else:
                # 拒绝更新，增大阻尼因子
                lam = min(lam * 10.0, 1e6)

                # 如果阻尼因子已达到上限，说明已接近最优解，判定收敛
                if lam >= 1e6:
                    converged = True
                    LOGGER.debug(
                        "GaussianFitter: 第 %d 次迭代收敛 (阻尼饱和, cost=%.2f)",
                        iteration + 1, cost,
                    )
                    break

            prev_cost = cost

        # 最终残差
        final_model = self._gaussian_2d_flat(xx_flat, yy_flat, params)
        final_residuals = data_flat - final_model
        rms_residual = float(np.sqrt(np.mean(final_residuals ** 2)))
        signal_range = float(np.max(data_flat) - np.min(data_flat))
        normalized_residual = rms_residual / max(signal_range, 1e-6)

        # 转换为全局坐标
        params[0] += offset_x
        params[1] += offset_y

        # 构建结果
        profile = self._build_profile(params, noise_std, normalized_residual, converged)

        self._last_params = params
        self._last_residual = normalized_residual
        self._fit_count += 1

        LOGGER.info(
            "GaussianFitter: 拟合完成 [%s] center=(%.2f, %.2f) sigma=(%.2f, %.2f) "
            "angle=%.1f° ellipticity=%.3f residual=%.4f",
            "收敛" if converged else "未收敛",
            profile.center_x, profile.center_y,
            profile.sigma_x, profile.sigma_y,
            profile.angle_deg, profile.ellipticity, profile.fit_residual,
        )

        return profile

    def fit_quick(
        self,
        image: np.ndarray,
        center_hint: Optional[Tuple[float, float]] = None,
    ) -> BeamProfile:
        """基于图像矩的快速光束参数估计 (无迭代)。

        使用图像的零阶、一阶、二阶中心矩直接估计高斯参数。
        速度远快于迭代拟合，但对非对称或重叠光斑精度较低。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (灰度，二维数组)。
        center_hint : Tuple[float, float] or None
            光束中心位置的先验估计。如果提供，将用于 ROI 裁剪。

        Returns
        -------
        BeamProfile
            估计的光束剖面参数 (converged 字段始终为 True)。
        """
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 2:
            raise ValueError(f"输入图像必须是二维数组，当前维度: {image.ndim}")

        # ROI 裁剪
        roi, offset_x, offset_y = self._extract_roi(image, center_hint)
        h, w = roi.shape

        if h < 5 or w < 5:
            LOGGER.warning("GaussianFitter[quick]: ROI 太小 (%dx%d)，返回默认值", h, w)
            return self._make_default_profile(offset_x, offset_y)

        noise_std = self._estimate_noise(roi)

        # 图像矩估计
        params = self._moment_initialization(roi, center_hint)

        # 使用二阶矩的协方差矩阵来估计旋转角度
        # 协方差矩阵的特征值/特征向量给出主轴方向
        total = roi.sum()
        if total < 1e-6:
            return self._make_default_profile(offset_x, offset_y)

        yy, xx = np.mgrid[:h, :w]
        cx_local = float(np.sum(xx * roi) / total)
        cy_local = float(np.sum(yy * roi) / total)
        dx = xx - cx_local
        dy = yy - cy_local

        # 二阶中心矩 (协方差矩阵元素)
        mu_xx = float(np.sum(dx * dx * roi) / total)
        mu_yy = float(np.sum(dy * dy * roi) / total)
        mu_xy = float(np.sum(dx * dy * roi) / total)

        # 协方差矩阵的特征分解
        cov_matrix = np.array([[mu_xx, mu_xy], [mu_xy, mu_yy]])
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 确保特征值非负
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # sigma = sqrt(eigenvalue)
        sigma_1 = float(np.sqrt(eigenvalues[1]))  # 较大特征值 -> 较大 sigma
        sigma_2 = float(np.sqrt(eigenvalues[0]))  # 较小特征值 -> 较小 sigma

        # 旋转角度 (特征向量方向)
        angle_rad = float(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        angle_deg = float(np.degrees(angle_rad))

        # 更新参数
        params[0] = cx_local + offset_x
        params[1] = cy_local + offset_y
        params[2] = max(sigma_1, 0.5)
        params[3] = max(sigma_2, 0.5)
        params[6] = angle_deg

        # 快速拟合的残差估计
        yy_full, xx_full = np.mgrid[:h, :w]
        model = self._gaussian_2d(xx_full, yy_full, params[0] - offset_x,
                                   params[1] - offset_y, params[2], params[3],
                                   params[4], params[5], params[6])
        residuals = roi - model
        rms_residual = float(np.sqrt(np.mean(residuals ** 2)))
        signal_range = float(np.max(roi) - np.min(roi))
        normalized_residual = rms_residual / max(signal_range, 1e-6)

        profile = self._build_profile(params, noise_std, normalized_residual, True)

        self._last_params = params
        self._last_residual = normalized_residual
        self._fit_count += 1

        LOGGER.debug(
            "GaussianFitter[quick]: 估计完成 center=(%.2f, %.2f) sigma=(%.2f, %.2f) angle=%.1f°",
            profile.center_x, profile.center_y,
            profile.sigma_x, profile.sigma_y, profile.angle_deg,
        )

        return profile

    def get_beam_quality(self, profile: BeamProfile) -> float:
        """估计光束质量因子 M²。

        M² (M-squared) 是衡量激光光束质量的关键指标。
        M² = 1 表示理想基模高斯光束 (TEM00)。
        M² > 1 表示光束存在高阶模式成分或像差。

        注意: 精确的 M² 测量需要沿传播轴多个位置的束宽数据。
        此方法基于椭圆度和拟合残差提供近似估计，仅供参考。

        Parameters
        ----------
        profile : BeamProfile
            拟合得到的光束剖面参数。

        Returns
        -------
        float
            估计的 M² 值。范围 [1.0, +inf)，1.0 为理想光束。
        """
        # 椭圆度贡献: 椭圆光束的 M² 近似
        ellipticity = profile.ellipticity
        if ellipticity < 1e-6:
            return 10.0  # 极端椭圆，M² 很差

        # 椭圆度对 M² 的贡献 (经验公式)
        # 对于椭圆高斯，有效 M² ≈ 1 / ellipticity (一阶近似)
        m2_ellipticity = 1.0 / max(ellipticity, 0.01)

        # 拟合残差贡献: 残差大说明偏离理想高斯
        # 残差 0 -> M² = 1, 残差 0.5 -> M² ≈ 2
        m2_residual = 1.0 + profile.fit_residual * 2.0

        # 综合估计: 取两个因子的几何平均
        m2_estimate = np.sqrt(m2_ellipticity * m2_residual)

        return float(max(m2_estimate, 1.0))

    def get_ellipticity(self, profile: BeamProfile) -> float:
        """计算光束椭圆度。

        椭圆度定义为 min(sigma_x, sigma_y) / max(sigma_x, sigma_y)。
        值为 1.0 表示完美圆形光束，接近 0 表示高度椭圆。

        Parameters
        ----------
        profile : BeamProfile
            光束剖面参数。

        Returns
        -------
        float
            椭圆度，范围 (0, 1]。
        """
        sx = abs(profile.sigma_x)
        sy = abs(profile.sigma_y)
        if max(sx, sy) < 1e-10:
            return 1.0
        return float(min(sx, sy) / max(sx, sy))

    def is_valid_fit(self, profile: BeamProfile) -> bool:
        """检验拟合结果是否有效且具有物理意义。

        判定标准:
        - 拟合已收敛 (迭代模式)
        - sigma_x, sigma_y 在合理范围内 (0.5 ~ roi_size/2 像素)
        - 振幅为正
        - 椭圆度不过分极端 (> 0.05)
        - 拟合残差可接受 (< 0.5)

        Parameters
        ----------
        profile : BeamProfile
            光束剖面参数。

        Returns
        -------
        bool
            拟合结果是否有效。
        """
        # 收敛性检查
        if not profile.converged:
            LOGGER.debug("GaussianFitter: 拟合未收敛，判定为无效")
            return False

        # sigma 范围检查
        sigma_min = 0.5
        sigma_max = self.roi_size / 2.0
        if profile.sigma_x < sigma_min or profile.sigma_x > sigma_max:
            LOGGER.debug("GaussianFitter: sigma_x=%.2f 超出合理范围 [%.1f, %.1f]",
                         profile.sigma_x, sigma_min, sigma_max)
            return False
        if profile.sigma_y < sigma_min or profile.sigma_y > sigma_max:
            LOGGER.debug("GaussianFitter: sigma_y=%.2f 超出合理范围 [%.1f, %.1f]",
                         profile.sigma_y, sigma_min, sigma_max)
            return False

        # 振幅检查
        if profile.amplitude <= 0:
            LOGGER.debug("GaussianFitter: amplitude=%.2f 非正，判定为无效", profile.amplitude)
            return False

        # 椭圆度检查
        if profile.ellipticity < 0.05:
            LOGGER.debug("GaussianFitter: ellipticity=%.4f 过小，判定为无效", profile.ellipticity)
            return False

        # 残差检查
        if profile.fit_residual > 0.5:
            LOGGER.debug("GaussianFitter: residual=%.4f 过大，判定为无效", profile.fit_residual)
            return False

        return True

    def reset(self) -> None:
        """重置拟合器内部状态。"""
        self._last_params = None
        self._last_residual = 0.0
        self._fit_count = 0
        LOGGER.info("GaussianFitter: 拟合器已重置")

    # ------------------------------------------------------------------
    # 内部方法: ROI 提取
    # ------------------------------------------------------------------

    def _extract_roi(
        self,
        image: np.ndarray,
        center_hint: Optional[Tuple[float, float]],
    ) -> Tuple[np.ndarray, int, int]:
        """从大图中提取 ROI 区域。

        Parameters
        ----------
        image : np.ndarray
            输入图像。
        center_hint : Tuple[float, float] or None
            中心位置提示。

        Returns
        -------
        Tuple[np.ndarray, int, int]
            (roi, offset_x, offset_y) ROI 图像和偏移量。
        """
        h, w = image.shape

        # 如果图像小于 ROI 尺寸，直接使用整个图像
        if h <= self.roi_size and w <= self.roi_size:
            return image.copy(), 0, 0

        # 确定裁剪中心
        if center_hint is not None:
            cx, cy = float(center_hint[0]), float(center_hint[1])
        else:
            # 使用灰度加权质心
            total = image.sum()
            if total < 1e-6:
                cx, cy = w / 2.0, h / 2.0
            else:
                yy, xx = np.mgrid[:h, :w]
                cx = float(np.sum(xx * image) / total)
                cy = float(np.sum(yy * image) / total)

        # 计算裁剪边界
        half = self.roi_size // 2
        x1 = int(round(cx)) - half
        y1 = int(round(cy)) - half
        x2 = x1 + self.roi_size
        y2 = y1 + self.roi_size

        # 边界钳位
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        roi = image[y1:y2, x1:x2].copy()
        return roi, x1, y1

    # ------------------------------------------------------------------
    # 内部方法: 参数初始化 (图像矩)
    # ------------------------------------------------------------------

    def _moment_initialization(
        self,
        roi: np.ndarray,
        center_hint: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        """基于图像矩估计高斯参数初始值。

        Parameters
        ----------
        roi : np.ndarray
            ROI 图像。
        center_hint : Tuple[float, float] or None
            中心位置提示 (ROI 局部坐标)。

        Returns
        -------
        np.ndarray
            参数向量 [cx, cy, sigma_x, sigma_y, amplitude, offset, angle_deg]。
        """
        h, w = roi.shape
        total = roi.sum()

        if total < 1e-6:
            # 无信号，返回默认值
            return np.array([
                w / 2.0, h / 2.0,  # center
                3.0, 3.0,          # sigma
                1.0,               # amplitude
                0.0,               # offset
                0.0,               # angle
            ])

        yy, xx = np.mgrid[:h, :w]

        # 一阶矩 -> 中心
        if center_hint is not None:
            cx = float(center_hint[0])
            cy = float(center_hint[1])
        else:
            cx = float(np.sum(xx * roi) / total)
            cy = float(np.sum(yy * roi) / total)

        # 背景估计 (图像边缘中值)
        border_pixels = np.concatenate([
            roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1],
        ])
        offset = float(np.median(border_pixels))

        # 扣除背景后的信号
        signal = np.maximum(roi - offset, 0.0)
        signal_total = signal.sum()

        if signal_total < 1e-6:
            return np.array([cx, cy, 3.0, 3.0, 1.0, offset, 0.0])

        # 振幅估计
        amplitude = float(signal.max())

        # 二阶矩 -> sigma
        dx = xx - cx
        dy = yy - cy
        mu_xx = float(np.sum(dx * dx * signal) / signal_total)
        mu_yy = float(np.sum(dy * dy * signal) / signal_total)

        sigma_x = max(float(np.sqrt(mu_xx)), 1.0)
        sigma_y = max(float(np.sqrt(mu_yy)), 1.0)

        # 旋转角度初步估计 (从互相关矩)
        mu_xy = float(np.sum(dx * dy * signal) / signal_total)
        angle_rad = 0.5 * np.arctan2(2.0 * mu_xy, mu_xx - mu_yy)
        angle_deg = float(np.degrees(angle_rad))

        return np.array([cx, cy, sigma_x, sigma_y, amplitude, offset, angle_deg])

    # ------------------------------------------------------------------
    # 内部方法: 二维旋转高斯模型
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_2d(
        xx: np.ndarray,
        yy: np.ndarray,
        cx: float,
        cy: float,
        sigma_x: float,
        sigma_y: float,
        amplitude: float,
        offset: float,
        angle_deg: float,
    ) -> np.ndarray:
        """计算二维旋转高斯函数值。

        Parameters
        ----------
        xx, yy : np.ndarray
            坐标网格。
        cx, cy : float
            中心坐标。
        sigma_x, sigma_y : float
            x/y 方向标准差。
        amplitude : float
            峰值振幅。
        offset : float
            背景偏移。
        angle_deg : float
            旋转角度 (度)。

        Returns
        -------
        np.ndarray
            高斯函数值数组。
        """
        theta = np.radians(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        dx = xx - cx
        dy = yy - cy

        # 旋转坐标
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        # 高斯函数
        sx2 = max(sigma_x * sigma_x, 1e-6)
        sy2 = max(sigma_y * sigma_y, 1e-6)

        exponent = -0.5 * (x_rot * x_rot / sx2 + y_rot * y_rot / sy2)
        return amplitude * np.exp(exponent) + offset

    @staticmethod
    def _gaussian_2d_flat(
        xx: np.ndarray,
        yy: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """使用参数向量计算二维旋转高斯函数值 (扁平坐标)。

        Parameters
        ----------
        xx, yy : np.ndarray
            扁平化的坐标数组。
        params : np.ndarray
            [cx, cy, sigma_x, sigma_y, amplitude, offset, angle_deg]。

        Returns
        -------
        np.ndarray
            高斯函数值数组。
        """
        cx, cy, sigma_x, sigma_y, amplitude, offset, angle_deg = params
        theta = np.radians(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        dx = xx - cx
        dy = yy - cy

        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        sx2 = max(sigma_x * sigma_x, 1e-6)
        sy2 = max(sigma_y * sigma_y, 1e-6)

        exponent = -0.5 * (x_rot * x_rot / sx2 + y_rot * y_rot / sy2)
        return amplitude * np.exp(exponent) + offset

    # ------------------------------------------------------------------
    # 内部方法: 雅可比矩阵
    # ------------------------------------------------------------------

    @staticmethod
    def _jacobian(
        xx: np.ndarray,
        yy: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """计算二维旋转高斯模型对 7 个参数的雅可比矩阵。

        参数顺序: [cx, cy, sigma_x, sigma_y, amplitude, offset, angle_deg]

        Parameters
        ----------
        xx, yy : np.ndarray
            扁平化的坐标数组。
        params : np.ndarray
            参数向量。

        Returns
        -------
        np.ndarray
            雅可比矩阵，形状 (N, 7)，N 为像素数。
        """
        cx, cy, sigma_x, sigma_y, amplitude, offset, angle_deg = params

        theta = np.radians(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        dx = xx - cx
        dy = yy - cy

        # 旋转坐标
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        sx2 = max(sigma_x * sigma_x, 1e-6)
        sy2 = max(sigma_y * sigma_y, 1e-6)
        sx3 = sigma_x * sx2
        sy3 = sigma_y * sy2

        # 指数项
        exponent = -0.5 * (x_rot * x_rot / sx2 + y_rot * y_rot / sy2)
        exp_val = np.exp(exponent)

        # 公共因子: amplitude * exp_val
        A_exp = amplitude * exp_val

        # dG/dcx: amplitude * exp * [x_rot*cos_t/sx2 + y_rot*sin_t/sy2]
        # 注意: d(x_rot)/d(cx) = -cos_t, d(y_rot)/d(cx) = sin_t
        dG_dcx = A_exp * (x_rot * cos_t / sx2 + y_rot * sin_t / sy2)

        # dG/dcy: amplitude * exp * [x_rot*sin_t/sx2 - y_rot*cos_t/sy2]
        # 注意: d(x_rot)/d(cy) = -sin_t, d(y_rot)/d(cy) = -cos_t
        dG_dcy = A_exp * (x_rot * sin_t / sx2 - y_rot * cos_t / sy2)

        # dG/d(sigma_x): amplitude * exp * x_rot^2 / sigma_x^3
        dG_dsx = A_exp * x_rot * x_rot / sx3

        # dG/d(sigma_y): amplitude * exp * y_rot^2 / sigma_y^3
        dG_dsy = A_exp * y_rot * y_rot / sy3

        # dG/d(amplitude): exp
        dG_dA = exp_val

        # dG/d(offset): 1
        dG_doff = np.ones_like(xx)

        # dG/d(theta): amplitude * exp * [x_rot*y_rot*(1/sy2 - 1/sx2)]
        # 注意: d(x_rot)/d(theta) = -dx*sin_t + dy*cos_t = y_rot
        #        d(y_rot)/d(theta) = -dx*cos_t - dy*sin_t = -x_rot
        # dG/dtheta = A_exp * [x_rot * (-x_rot)/sx2 + y_rot * x_rot/sy2]
        #           = A_exp * x_rot * y_rot * (1/sy2 - 1/sx2)
        # 角度单位转换: rad -> deg (乘以 pi/180)
        dG_dtheta = A_exp * x_rot * y_rot * (1.0 / sy2 - 1.0 / sx2) * (np.pi / 180.0)

        N = len(xx)
        J = np.empty((N, 7), dtype=np.float64)
        J[:, 0] = dG_dcx
        J[:, 1] = dG_dcy
        J[:, 2] = dG_dsx
        J[:, 3] = dG_dsy
        J[:, 4] = dG_dA
        J[:, 5] = dG_doff
        J[:, 6] = dG_dtheta

        return J

    # ------------------------------------------------------------------
    # 内部方法: 辅助函数
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_noise(image: np.ndarray) -> float:
        """估计图像噪声标准差。

        使用图像中较暗区域的像素来估计噪声水平。

        Parameters
        ----------
        image : np.ndarray
            输入图像。

        Returns
        -------
        float
            噪声标准差估计值。
        """
        # 使用低于 25 百分位的像素估计噪声
        threshold = np.percentile(image, 25)
        dark_pixels = image[image <= threshold]
        if dark_pixels.size < 4:
            return 1.0
        return float(np.std(dark_pixels))

    @staticmethod
    def _build_profile(
        params: np.ndarray,
        noise_std: float,
        normalized_residual: float,
        converged: bool,
    ) -> BeamProfile:
        """从参数向量构建 BeamProfile 对象。

        Parameters
        ----------
        params : np.ndarray
            [cx, cy, sigma_x, sigma_y, amplitude, offset, angle_deg]。
        noise_std : float
            噪声标准差。
        normalized_residual : float
            归一化拟合残差。
        converged : bool
            是否收敛。

        Returns
        -------
        BeamProfile
            光束剖面参数。
        """
        cx, cy, sigma_x, sigma_y, amplitude, offset, angle_deg = params

        # 确保 sigma 为正
        sigma_x = max(abs(sigma_x), 1e-6)
        sigma_y = max(abs(sigma_y), 1e-6)

        # FWHM
        fwhm_x = float(FWHM_FACTOR * sigma_x)
        fwhm_y = float(FWHM_FACTOR * sigma_y)

        # 椭圆度
        if max(sigma_x, sigma_y) < 1e-10:
            ellipticity = 1.0
        else:
            ellipticity = float(min(sigma_x, sigma_y) / max(sigma_x, sigma_y))

        # 峰值信噪比
        peak_snr = float(amplitude / max(noise_std, 1e-6))

        return BeamProfile(
            center_x=round(float(cx), 4),
            center_y=round(float(cy), 4),
            sigma_x=round(float(sigma_x), 4),
            sigma_y=round(float(sigma_y), 4),
            amplitude=round(float(amplitude), 4),
            offset=round(float(offset), 4),
            angle_deg=round(float(angle_deg), 4),
            fwhm_x=round(fwhm_x, 4),
            fwhm_y=round(fwhm_y, 4),
            ellipticity=round(ellipticity, 6),
            peak_snr=round(peak_snr, 4),
            fit_residual=round(normalized_residual, 6),
            converged=converged,
        )

    def _make_default_profile(self, offset_x: int, offset_y: int) -> BeamProfile:
        """创建默认 (无效) 的 BeamProfile。

        Parameters
        ----------
        offset_x, offset_y : int
            ROI 偏移量。

        Returns
        -------
        BeamProfile
            默认光束剖面参数。
        """
        return BeamProfile(
            center_x=float(offset_x + self.roi_size / 2),
            center_y=float(offset_y + self.roi_size / 2),
            sigma_x=0.0,
            sigma_y=0.0,
            amplitude=0.0,
            offset=0.0,
            angle_deg=0.0,
            fwhm_x=0.0,
            fwhm_y=0.0,
            ellipticity=0.0,
            peak_snr=0.0,
            fit_residual=1.0,
            converged=False,
        )
