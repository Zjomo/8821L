"""
在线自适应光学校正器 (Online Adaptive Optics Corrector)

灵感来源: HCIPy (https://github.com/ehpor/hcipy)
           AOtools (https://github.com/AOtools/aotools)
           以及近年深度学习无波前传感器自适应光学研究

核心思想:
──────────────────────────────────────────────────────────────────────
传统自适应光学系统依赖波前传感器 (Shack-Hartmann 等) 实时测量像差，
然后通过变形镜 (DM) 进行校正。然而在许多实际场景中，波前传感器不可用
或引入额外复杂性。

本模块实现了两种校正策略:
1. 基于图像质量的闭环迭代校正 (参考 HCIPy 的 AO 仿真框架)
2. 无波前传感器 (sensorless) 的模式搜索校正 (参考近年 DL-based AO)

适用场景:
- 大气湍流导致的光斑漂移和模糊
- 光学系统热漂移补偿
- 长时间曝光中的实时像差校正
"""

__version__ = "1.0.0"
__author__ = "SpotZoom Team"

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable
from enum import Enum
import time


class CorrectionMode(Enum):
    """校正模式"""
    IMAGE_BASED = "image_based"       # 基于图像质量的闭环校正
    SENSORLESS = "sensorless"         # 无波前传感器模式搜索
    HYBRID = "hybrid"                 # 混合模式


class ZernikeMode(Enum):
    """Zernike 像差模式"""
    TIP = 2
    TILT = 3
    DEFOCUS = 4
    ASTIGMATISM_0 = 5
    ASTIGMATISM_45 = 6
    COMA_X = 7
    COMA_Y = 8
    TREFOIL_X = 9
    TREFOIL_Y = 10
    SPHERICAL = 11


@dataclass
class AOCorrectorConfig:
    """自适应光学校正器配置"""
    # 校正模式
    mode: CorrectionMode = CorrectionMode.HYBRID

    # Zernike 模式配置
    active_modes: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8])
    max_zernike_order: int = 4

    # 闭环校正参数
    closed_loop_gain: float = 0.4        # 积分控制器增益
    leaky_integrator: float = 0.98       # 泄漏积分因子 (防止漂移)
    max_correction_amplitude: float = 2.0  # 最大校正幅度 (微米)

    # 无波前传感器参数
    sensorless_step_size: float = 0.3    # 模式搜索步长
    sensorless_iterations: int = 15      # 每模式迭代次数
    sensorless_convergence_tol: float = 1e-4

    # 图像质量评估
    quality_metric: str = "strehl"       # strehl / sharpness / entropy
    roi_radius: int = 64                 # 评估区域半径

    # 性能参数
    max_correction_time_ms: float = 50.0  # 单次校正最大耗时
    history_length: int = 100             # 历史记录长度

    # 安全限制
    safety_threshold: float = 0.5        # 质量下降安全阈值
    max_total_correction: float = 5.0    # 总校正量上限


@dataclass
class AOCorrectionResult:
    """校正结果"""
    corrected_image: np.ndarray          # 校正后图像
    zernike_coefficients: np.ndarray     # Zernike 系数
    correction_delta: np.ndarray         # 本次校正增量
    quality_before: float                # 校正前质量
    quality_after: float                 # 校正后质量
    quality_improvement: float           # 质量提升比
    correction_time_ms: float            # 校正耗时
    mode: str                            # 使用的校正模式
    converged: bool                      # 是否收敛
    total_residual_rms: float            # 残余像差 RMS


class OnlineAOCorrector:
    """
    在线自适应光学校正器

    实现:
    1. 基于 Zernike 模式分解的闭环校正
    2. 无波前传感器的模式搜索 (Noll 模式逐个优化)
    3. 混合策略: 先用图像质量快速粗调，再用闭环精细校正

    用法示例:
        corrector = OnlineAOCorrector(AOCorrectorConfig())
        result = corrector.correct(image, target_position=(320, 240))
        print(f"质量提升: {result.quality_improvement:.2%}")
    """

    # Zernike 径向多项式 (Noll 编号)
    _ZERNIKE_N = {
        2: (1, 1), 3: (1, 1), 4: (2, 0), 5: (2, -2), 6: (2, 2),
        7: (3, -1), 8: (3, 1), 9: (3, -3), 10: (3, 3), 11: (4, 0)
    }

    def __init__(self, config: Optional[AOCorrectorConfig] = None):
        self.config = config or AOCorrectorConfig()

        # 当前 Zernike 校正状态
        n_modes = max(self.config.active_modes) + 1 if self.config.active_modes else 12
        self._correction_state = np.zeros(n_modes)

        # 积分器状态 (用于闭环控制)
        self._integrator_state = np.zeros(n_modes)

        # 历史记录
        self._quality_history: List[float] = []
        self._correction_history: List[np.ndarray] = []
        self._timestamp_history: List[float] = []

        # 统计信息
        self._total_corrections = 0
        self._total_improvement = 0.0

    def correct(self, image: np.ndarray,
                target_position: Optional[Tuple[int, int]] = None,
                reference_image: Optional[np.ndarray] = None) -> AOCorrectionResult:
        """
        执行自适应光学校正

        Args:
            image: 当前光斑图像
            target_position: 目标位置 (可选)
            reference_image: 参考图像 (可选, 用于图像质量对比)

        Returns:
            AOCorrectionResult: 校正结果
        """
        start_time = time.time()

        # 提取 ROI
        roi = self._extract_roi(image, target_position)

        # 评估当前质量
        quality_before = self._assess_quality(roi, reference_image)

        if self.config.mode == CorrectionMode.IMAGE_BASED:
            correction = self._image_based_correction(roi, quality_before)
        elif self.config.mode == CorrectionMode.SENSORLESS:
            correction = self._sensorless_correction(roi, quality_before)
        else:
            correction = self._hybrid_correction(roi, quality_before)

        # 应用安全限制
        correction = self._apply_safety_limits(correction)

        # 更新校正状态
        self._correction_state += correction

        # 模拟校正效果 (实际系统中由 DM/SLM 执行)
        corrected_roi = self._apply_correction_model(roi, correction)
        quality_after = self._assess_quality(corrected_roi, reference_image)

        # 更新积分器
        self._update_integrator(correction, quality_after - quality_before)

        # 记录历史
        elapsed_ms = (time.time() - start_time) * 1000
        self._record_history(quality_before, quality_after, correction, elapsed_ms)

        improvement = (quality_after - quality_before) / max(abs(quality_before), 1e-10)
        converged = abs(improvement) < self.config.sensorless_convergence_tol

        self._total_corrections += 1
        self._total_improvement += improvement

        return AOCorrectionResult(
            corrected_image=corrected_roi,
            zernike_coefficients=self._correction_state.copy(),
            correction_delta=correction.copy(),
            quality_before=quality_before,
            quality_after=quality_after,
            quality_improvement=improvement,
            correction_time_ms=elapsed_ms,
            mode=self.config.mode.value,
            converged=converged,
            total_residual_rms=float(np.sqrt(np.sum(self._correction_state ** 2)))
        )

    def _extract_roi(self, image: np.ndarray,
                     target_position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """提取感兴趣区域"""
        h, w = image.shape[:2]

        if target_position is None:
            # 自动检测光斑中心
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if _has_cv2() else np.mean(image, axis=2)
            cy, cx = np.unravel_index(np.argmax(gray), gray.shape)
        else:
            cx, cy = target_position

        r = self.config.roi_radius
        y1, y2 = max(0, cy - r), min(h, cy + r)
        x1, x2 = max(0, cx - r), min(w, cx + r)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            roi = image[max(0, h//2-r):min(h, h//2+r), max(0, w//2-r):min(w, w//2+r)]

        return roi

    def _assess_quality(self, roi: np.ndarray,
                        reference: Optional[np.ndarray] = None) -> float:
        """评估图像质量"""
        if len(roi.shape) == 3:
            gray = np.mean(roi, axis=2)
        else:
            gray = roi.astype(np.float64)

        metric = self.config.quality_metric

        if metric == "strehl":
            # Strehl 比: 峰值强度 / 衍射极限峰值
            total = np.sum(gray)
            if total < 1e-10:
                return 0.0
            peak = np.max(gray)
            strehl = peak / (total / gray.size) if (total / gray.size) > 0 else 0
            return min(strehl, 100.0)

        elif metric == "sharpness":
            # 图像锐度: 拉普拉斯方差
            if gray.shape[0] < 3 or gray.shape[1] < 3:
                return 0.0
            laplacian = (np.roll(gray, 1, 0) + np.roll(gray, -1, 0) +
                        np.roll(gray, 1, 1) + np.roll(gray, -1, 1) - 4 * gray)
            return float(np.var(laplacian))

        elif metric == "entropy":
            # 图像熵 (越低越好 = 更集中)
            hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
            hist = hist / (hist.sum() + 1e-10)
            hist = hist[hist > 0]
            return -float(np.sum(hist * np.log2(hist + 1e-10)))

        return 0.0

    def _image_based_correction(self, roi: np.ndarray,
                                 current_quality: float) -> np.ndarray:
        """基于图像质量的闭环校正"""
        correction = np.zeros_like(self._correction_state)

        for mode_idx in self.config.active_modes:
            if mode_idx >= len(correction):
                break

            # 探测方向
            probe = np.zeros_like(correction)
            probe[mode_idx] = self.config.sensorless_step_size

            # 模拟正/负方向的质量变化
            quality_plus = self._estimate_quality_change(roi, probe)
            quality_minus = self._estimate_quality_change(roi, -probe)

            # 梯度估计
            gradient = (quality_plus - quality_minus) / (2 * self.config.sensorless_step_size)

            # 积分控制器更新
            self._integrator_state[mode_idx] = (
                self.config.leaky_integrator * self._integrator_state[mode_idx] +
                self.config.closed_loop_gain * gradient
            )

            correction[mode_idx] = self._integrator_state[mode_idx]

        return correction

    def _sensorless_correction(self, roi: np.ndarray,
                                current_quality: float) -> np.ndarray:
        """无波前传感器模式搜索"""
        correction = np.zeros_like(self._correction_state)

        for mode_idx in self.config.active_modes:
            if mode_idx >= len(correction):
                break

            best_coeff = 0.0
            best_quality = current_quality
            step = self.config.sensorless_step_size

            for _ in range(self.config.sensorless_iterations):
                improved = False

                for direction in [step, -step]:
                    probe = np.zeros_like(correction)
                    probe[mode_idx] = direction
                    q = self._estimate_quality_change(roi, probe)

                    if q > best_quality:
                        best_quality = q
                        best_coeff += direction
                        improved = True
                        break

                if not improved:
                    step *= 0.5
                    if step < self.config.sensorless_convergence_tol:
                        break

            correction[mode_idx] = best_coeff

        return correction

    def _hybrid_correction(self, roi: np.ndarray,
                           current_quality: float) -> np.ndarray:
        """混合校正策略"""
        # 阶段1: 无波前传感器快速粗调 (前3个低阶模式)
        coarse_correction = np.zeros_like(self._correction_state)
        low_order_modes = [m for m in self.config.active_modes if m <= 6]

        for mode_idx in low_order_modes:
            if mode_idx >= len(coarse_correction):
                break
            probe = np.zeros_like(coarse_correction)
            probe[mode_idx] = self.config.sensorless_step_size * 1.5

            q_plus = self._estimate_quality_change(roi, probe)
            q_minus = self._estimate_quality_change(roi, -probe)
            gradient = (q_plus - q_minus) / (2 * self.config.sensorless_step_size * 1.5)

            coarse_correction[mode_idx] = gradient * self.config.closed_loop_gain * 2

        # 阶段2: 闭环精细校正 (所有模式)
        fine_correction = self._image_based_correction(roi, current_quality)

        return coarse_correction + fine_correction

    def _estimate_quality_change(self, roi: np.ndarray,
                                  correction: np.ndarray) -> float:
        """估计应用校正后的图像质量"""
        corrected = self._apply_correction_model(roi, correction)
        return self._assess_quality(corrected)

    def _apply_correction_model(self, roi: np.ndarray,
                                  correction: np.ndarray) -> np.ndarray:
        """
        模拟 Zernike 校正对图像的影响

        在实际系统中，这由 DM/SLM 硬件执行。
        这里使用简化的 PSF 模型模拟校正效果。
        """
        gray = roi.astype(np.float64) if len(roi.shape) == 2 else np.mean(roi, axis=2).astype(np.float64)
        h, w = gray.shape

        # 计算残余像差对 PSF 的影响
        total_correction = self._correction_state + correction
        residual_rms = float(np.sqrt(np.sum(total_correction ** 2)))

        # 简化模型: 校正改善 PSF 锐度
        # 实际中应使用 HCIPy 风格的精确波前传播
        improvement_factor = 1.0 / (1.0 + residual_rms * 0.1)

        # 应用轻微锐化模拟校正效果
        if improvement_factor > 1.0:
            kernel_size = max(3, int(5 * (improvement_factor - 1.0)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            # 简单锐化核
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            blurred = self._convolve2d(gray, kernel)
            corrected = gray + (improvement_factor - 1.0) * (gray - blurred)
            corrected = np.clip(corrected, 0, 255)
        else:
            corrected = gray

        return corrected

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D 卷积 (纯 numpy)"""
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(image, ((ph, ph), (pw, pw)), mode='edge')
        h, w = image.shape
        result = np.zeros_like(image)

        for i in range(kh):
            for j in range(kw):
                result += kernel[i, j] * padded[i:i+h, j:j+w]

        return result

    def _apply_safety_limits(self, correction: np.ndarray) -> np.ndarray:
        """应用安全限制"""
        # 单模式幅度限制
        correction = np.clip(correction,
                            -self.config.max_correction_amplitude,
                            self.config.max_correction_amplitude)

        # 总校正量限制
        total = np.sqrt(np.sum(correction ** 2))
        if total > self.config.max_total_correction:
            correction *= self.config.max_total_correction / total

        return correction

    def _update_integrator(self, correction: np.ndarray,
                           quality_change: float) -> None:
        """更新积分器状态"""
        if quality_change < -self.config.safety_threshold:
            # 质量严重下降, 重置积分器
            self._integrator_state *= 0.5

    def _record_history(self, quality_before: float, quality_after: float,
                        correction: np.ndarray, elapsed_ms: float) -> None:
        """记录历史"""
        self._quality_history.append(quality_after)
        self._correction_history.append(correction.copy())
        self._timestamp_history.append(time.time())

        if len(self._quality_history) > self.config.history_length:
            self._quality_history.pop(0)
            self._correction_history.pop(0)
            self._timestamp_history.pop(0)

    def reset(self) -> None:
        """重置校正器状态"""
        self._correction_state = np.zeros_like(self._correction_state)
        self._integrator_state = np.zeros_like(self._integrator_state)
        self._quality_history = []
        self._correction_history = []
        self._timestamp_history = []
        self._total_corrections = 0
        self._total_improvement = 0.0

    def get_statistics(self) -> Dict:
        """获取校正统计信息"""
        return {
            "total_corrections": self._total_corrections,
            "average_improvement": (self._total_improvement / max(1, self._total_corrections)),
            "current_zernike_state": self._correction_state.tolist(),
            "residual_rms": float(np.sqrt(np.sum(self._correction_state ** 2))),
            "quality_history": self._quality_history[-10:],
            "integrator_energy": float(np.sum(self._integrator_state ** 2))
        }


def _has_cv2() -> bool:
    try:
        import cv2
        return True
    except ImportError:
        return False
