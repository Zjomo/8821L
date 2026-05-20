"""
Coherent Sensitivity Analyzer - 相干灵敏度分析器

Inspired by:
- prysm (brandondube): MTF, PSF, and sensitivity analysis
- HCIPy (ehpor): Wavefront sensitivity matrix computation
- Optiland (optiland): Differentiable optical system analysis

Core Innovation:
- 对准灵敏度矩阵计算: 量化各自由度对光斑位置的影响
- 灵敏度热力图可视化
- 最优校正方向推荐
- 纯 numpy+cv2 实现，零外部依赖
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SensitivityConfig:
    """灵敏度分析器配置"""
    # 分析自由度
    degrees_of_freedom: Tuple[str, ...] = ("x", "y", "z", "intensity")
    # 扰动步长 (像素/单位)
    perturbation_step: float = 1.0
    # 灵敏度矩阵历史窗口
    history_window: int = 100
    # 低灵敏度阈值 (低于此值认为该方向不敏感)
    low_sensitivity_threshold: float = 0.1
    # 高灵敏度阈值 (高于此值认为该方向过于敏感)
    high_sensitivity_threshold: float = 5.0
    # 是否启用自适应步长
    adaptive_step: bool = True


@dataclass
class SensitivityReport:
    """灵敏度分析报告"""
    # 灵敏度矩阵 (DOF x 响应)
    sensitivity_matrix: np.ndarray
    # 各自由度灵敏度
    dof_sensitivities: dict
    # 最敏感方向
    most_sensitive_dof: str
    # 最不敏感方向
    least_sensitive_dof: str
    # 推荐校正方向
    recommended_correction: dict
    # 条件数 (矩阵健康度)
    condition_number: float
    # 是否需要重新校准
    needs_recalibration: bool
    # 灵敏度变化趋势
    sensitivity_trend: dict


class CoherentSensitivityAnalyzer:
    """相干灵敏度分析器

    通过分析光斑响应对各自由度扰动的灵敏度，
    量化对准系统的可控性和最优校正策略。

    Inspired by prysm's sensitivity analysis and HCIPy's
    wavefront sensitivity matrix computation.
    """

    def __init__(self, config: Optional[SensitivityConfig] = None):
        self.config = config or SensitivityConfig()
        self._sensitivity_history: Deque[np.ndarray] = deque(maxlen=self.config.history_window)
        self._response_buffer: dict = {}
        self._perturbation_state: dict = {}
        self._current_step = self.config.perturbation_step

    def reset(self) -> None:
        """重置分析器状态"""
        self._sensitivity_history.clear()
        self._response_buffer.clear()
        self._perturbation_state.clear()

    def _compute_image_response(
        self, ref_image: np.ndarray, perturbed_image: np.ndarray
    ) -> dict:
        """计算图像响应特征

        Args:
            ref_image: 参考图像
            perturbed_image: 扰动后图像

        Returns:
            响应特征字典
        """
        if len(ref_image.shape) == 3:
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_image.copy()

        if len(perturbed_image.shape) == 3:
            pert_gray = cv2.cvtColor(perturbed_image, cv2.COLOR_BGR2GRAY)
        else:
            pert_gray = perturbed_image.copy()

        # 差分图像
        diff = pert_gray.astype(np.float64) - ref_gray.astype(np.float64)

        # 质心偏移
        def centroid(img):
            M = cv2.moments(img)
            if M["m00"] > 1e-6:
                return M["m10"] / M["m00"], M["m01"] / M["m00"]
            return img.shape[1] / 2, img.shape[0] / 2

        ref_cx, ref_cy = centroid(ref_gray)
        pert_cx, pert_cy = centroid(pert_gray)

        # 强度变化
        intensity_change = float(np.mean(diff))

        # 形状变化 (二阶矩)
        ref_var = float(np.var(ref_gray.astype(np.float64)))
        pert_var = float(np.var(pert_gray.astype(np.float64)))
        shape_change = pert_var - ref_var

        return {
            "centroid_dx": pert_cx - ref_cx,
            "centroid_dy": pert_cy - ref_cy,
            "intensity_change": intensity_change,
            "shape_change": shape_change,
            "diff_norm": float(np.linalg.norm(diff)),
        }

    def record_response(
        self,
        dof: str,
        perturbation: float,
        response: dict,
    ) -> None:
        """记录扰动-响应对

        Args:
            dof: 自由度名称
            perturbation: 扰动量
            response: 响应特征
        """
        if dof not in self._response_buffer:
            self._response_buffer[dof] = []

        self._response_buffer[dof].append({
            "perturbation": perturbation,
            "response": response,
        })

        # 保留最近 50 条记录
        if len(self._response_buffer[dof]) > 50:
            self._response_buffer[dof] = self._response_buffer[dof][-50:]

    def compute_sensitivity_matrix(self) -> Optional[SensitivityReport]:
        """计算灵敏度矩阵

        Returns:
            SensitivityReport 或 None (如果数据不足)
        """
        dofs = self.config.degrees_of_freedom
        response_keys = ["centroid_dx", "centroid_dy", "intensity_change", "shape_change"]

        n_dof = len(dofs)
        n_resp = len(response_keys)
        sensitivity_matrix = np.zeros((n_dof, n_resp))

        dof_sensitivities = {}

        for i, dof in enumerate(dofs):
            records = self._response_buffer.get(dof, [])
            if len(records) < 2:
                dof_sensitivities[dof] = 0.0
                continue

            # 线性回归: response = sensitivity * perturbation
            perturbations = np.array([r["perturbation"] for r in records])
            responses = np.array([r["response"][rk] for r in records for rk in response_keys])
            responses = responses.reshape(len(records), n_resp)

            # 对每个响应维度计算灵敏度
            sensitivities = []
            for j in range(n_resp):
                resp_col = responses[:, j]
                p = perturbations
                ss_pp = np.sum(p ** 2)
                if ss_pp > 1e-10:
                    sens = np.sum(p * resp_col) / ss_pp
                else:
                    sens = 0.0
                sensitivities.append(float(sens))
                sensitivity_matrix[i, j] = sens

            # 综合灵敏度: 所有响应维度的 RMS
            dof_sensitivities[dof] = float(np.sqrt(np.mean(np.array(sensitivities) ** 2)))

        # 条件数
        try:
            cond = float(np.linalg.cond(sensitivity_matrix))
        except np.linalg.LinAlgError:
            cond = float("inf")

        # 最敏感/最不敏感方向
        if dof_sensitivities:
            most_sensitive = max(dof_sensitivities, key=dof_sensitivities.get)
            least_sensitive = min(dof_sensitivities, key=dof_sensitivities.get)
        else:
            most_sensitive = "unknown"
            least_sensitive = "unknown"

        # 推荐校正方向
        recommended = {}
        for dof, sens in dof_sensitivities.items():
            if sens > self.config.high_sensitivity_threshold:
                recommended[dof] = "reduce_step"  # 灵敏度过高，减小步长
            elif sens < self.config.low_sensitivity_threshold:
                recommended[dof] = "increase_step"  # 灵敏度太低，增大步长
            else:
                recommended[dof] = "maintain"

        # 是否需要重新校准
        needs_recal = cond > 100 or any(
            s < self.config.low_sensitivity_threshold for s in dof_sensitivities.values()
        )

        # 灵敏度趋势
        self._sensitivity_history.append(sensitivity_matrix.copy())
        trend = {}
        if len(self._sensitivity_history) >= 5:
            recent = np.array(list(self._sensitivity_history)[-5:])
            for i, dof in enumerate(dofs):
                trend_vals = recent[:, i, :]
                trend[dof] = "increasing" if trend_vals[-1].mean() > trend_vals[0].mean() else "stable"

        report = SensitivityReport(
            sensitivity_matrix=sensitivity_matrix,
            dof_sensitivities=dof_sensitivities,
            most_sensitive_dof=most_sensitive,
            least_sensitive_dof=least_sensitive,
            recommended_correction=recommended,
            condition_number=cond,
            needs_recalibration=needs_recal,
            sensitivity_trend=trend,
        )

        return report

    def get_diagnostics(self) -> dict:
        """获取分析器诊断信息"""
        return {
            "num_records_per_dof": {
                dof: len(records)
                for dof, records in self._response_buffer.items()
            },
            "current_step": self._current_step,
            "history_size": len(self._sensitivity_history),
        }
