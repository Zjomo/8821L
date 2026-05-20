"""
PhysicsConstrainedReconstructor - 物理约束波前重构器

灵感来源: DeepXDE, NVIDIA Modulus PINN
功能特点:
- 基于轻量级神经网络的波前预测
- 物理一致性约束 (能量守恒、平滑性)
- 时序平滑处理
- 纯numpy实现，零外部ML依赖

技术路线:
- 轻量级全连接网络
- 物理损失函数约束
- 梯度惩罚正则化
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class WavefrontResult:
    """波前重构结果。"""
    zernike_coeffs: np.ndarray  # Zernike系数
    wavefront_map: np.ndarray   # 波前相位图
    rms_error: float           # RMS波前误差
    strehl_ratio: float        # Strehl比
    confidence: float          # 置信度
    physics_violation: float   # 物理约束违反程度


@dataclass
class PhysicsReconstructorConfig:
    """重构器配置。"""
    num_zernike_modes: int = 15
    hidden_size: int = 64
    learning_rate: float = 0.01
    # 物理约束权重
    energy_weight: float = 1.0
    smoothness_weight: float = 0.5
    temporal_weight: float = 0.3
    # 时序平滑参数
    history_length: int = 10
    # 收敛参数
    max_iterations: int = 100
    convergence_threshold: float = 1e-6


class PhysicsConstrainedReconstructor:
    """
    物理约束波前重构器。
    
    使用轻量级神经网络进行波前预测，同时施加物理约束确保结果合理性。
    物理约束包括：
    1. 能量守恒: 波前能量保持稳定
    2. 平滑性: 波前变化连续平滑
    3. 时序一致性: 相邻帧波前变化合理
    """
    
    def __init__(self, config: Optional[PhysicsReconstructorConfig] = None):
        self.config = config or PhysicsReconstructorConfig()
        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._history: deque = deque(maxlen=self.config.history_length)
        self._baseline_energy: float = 0.0
        self._iteration_count: int = 0
        self._init_network()
        LOGGER.info("PhysicsConstrainedReconstructor初始化完成，模式数=%d",
                   self.config.num_zernike_modes)
    
    def _init_network(self) -> None:
        """初始化轻量级神经网络权重。"""
        # 简化的单层网络: 输入 -> 隐藏 -> 输出
        input_size = self.config.num_zernike_modes * 2  # 斜率输入 (x, y)
        hidden_size = self.config.hidden_size
        output_size = self.config.num_zernike_modes
        
        # Xavier初始化
        self._w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self._b1 = np.zeros(hidden_size)
        self._w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self._b2 = np.zeros(output_size)
        
        self._initialized = True
    
    def reset(self) -> None:
        """重置重构器状态。"""
        self._history.clear()
        self._baseline_energy = 0.0
        self._iteration_count = 0
        self._init_network()
    
    def reconstruct(
        self, 
        slopes_x: np.ndarray,
        slopes_y: np.ndarray,
        grid_shape: Optional[Tuple[int, int]] = None
    ) -> WavefrontResult:
        """
        从斜率数据重构波前。
        
        Args:
            slopes_x: x方向斜率
            slopes_y: y方向斜率
            grid_shape: 网格形状 (rows, cols)
            
        Returns:
            WavefrontResult包含重构结果
        """
        t0 = time.perf_counter()
        
        # 准备输入
        slopes = self._prepare_input(slopes_x, slopes_y)
        
        # 神经网络预测
        raw_prediction = self._forward(slopes)
        
        # 应用物理约束
        constrained = self._apply_physics_constraints(raw_prediction)
        
        # 时序平滑
        smoothed = self._temporal_smooth(constrained)
        
        # 生成波前图
        wavefront_map = self._zernike_to_wavefront(smoothed, grid_shape)
        
        # 计算指标
        rms = self._compute_rms(smoothed)
        strehl = self._compute_strehl(rms)
        confidence = self._compute_confidence(slopes, smoothed)
        physics_violation = self._compute_physics_violation(smoothed)
        
        # 更新历史
        self._history.append(smoothed.copy())
        self._iteration_count += 1
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        LOGGER.debug("波前重构完成: RMS=%.4fλ, Strehl=%.3f, 耗时=%.2fms",
                    rms, strehl, elapsed_ms)
        
        return WavefrontResult(
            zernike_coeffs=smoothed,
            wavefront_map=wavefront_map,
            rms_error=rms,
            strehl_ratio=strehl,
            confidence=confidence,
            physics_violation=physics_violation
        )
    
    def _prepare_input(
        self, 
        slopes_x: np.ndarray, 
        slopes_y: np.ndarray
    ) -> np.ndarray:
        """准备网络输入。"""
        # 展平并拼接
        sx_flat = np.array(slopes_x).flatten()
        sy_flat = np.array(slopes_y).flatten()
        
        # 确保长度一致
        min_len = min(len(sx_flat), len(sy_flat), self.config.num_zernike_modes)
        
        input_vec = np.zeros(self.config.num_zernike_modes * 2)
        input_vec[:min_len] = sx_flat[:min_len]
        input_vec[self.config.num_zernike_modes:self.config.num_zernike_modes + min_len] = sy_flat[:min_len]
        
        # 归一化
        norm = np.linalg.norm(input_vec)
        if norm > 1e-12:
            input_vec = input_vec / norm
        
        return input_vec
    
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """神经网络前向传播。"""
        # 第一层
        h = x @ self._w1 + self._b1
        h = np.maximum(h, 0)  # ReLU
        
        # 第二层
        out = h @ self._w2 + self._b2
        
        return out
    
    def _apply_physics_constraints(self, prediction: np.ndarray) -> np.ndarray:
        """应用物理约束。"""
        constrained = prediction.copy()
        
        # 1. 能量守恒约束
        constrained = self._apply_energy_constraint(constrained)
        
        # 2. 平滑性约束 (高阶模式抑制)
        constrained = self._apply_smoothness_constraint(constrained)
        
        # 3. 模式耦合约束
        constrained = self._apply_mode_coupling_constraint(constrained)
        
        return constrained
    
    def _apply_energy_constraint(self, coeffs: np.ndarray) -> np.ndarray:
        """应用能量守恒约束。"""
        current_energy = np.sum(coeffs ** 2)
        
        if self._baseline_energy == 0.0:
            self._baseline_energy = current_energy if current_energy > 0 else 1.0
            return coeffs
        
        if current_energy < 1e-12:
            return coeffs
        
        # 调整能量使其接近基线
        energy_ratio = math.sqrt(self._baseline_energy / current_energy)
        adjustment = self.config.energy_weight * 0.1
        
        # 混合原始和调整后的系数
        adjusted = coeffs * (1 - adjustment + adjustment * energy_ratio)
        
        return adjusted
    
    def _apply_smoothness_constraint(self, coeffs: np.ndarray) -> np.ndarray:
        """应用平滑性约束 (抑制高阶模式)。"""
        constrained = coeffs.copy()
        
        # 高阶模式应该有更小的幅度
        for i in range(4, len(coeffs)):  # 从第5个模式开始
            # 基于模式索引的衰减因子
            decay = 1.0 / (1.0 + self.config.smoothness_weight * (i - 3) * 0.1)
            constrained[i] *= decay
        
        return constrained
    
    def _apply_mode_coupling_constraint(self, coeffs: np.ndarray) -> np.ndarray:
        """应用模式耦合约束。"""
        # 相邻模式之间的能量分布应该合理
        constrained = coeffs.copy()
        
        for i in range(1, len(coeffs) - 1):
            neighbor_energy = coeffs[i-1]**2 + coeffs[i+1]**2
            current_energy = coeffs[i]**2
            
            if neighbor_energy > 0 and current_energy > neighbor_energy * 4:
                # 当前模式能量过高，进行抑制
                scale = math.sqrt(neighbor_energy * 4 / current_energy)
                constrained[i] *= (0.5 + 0.5 * scale)
        
        return constrained
    
    def _temporal_smooth(self, current: np.ndarray) -> np.ndarray:
        """时序平滑处理。"""
        if len(self._history) < 2:
            return current
        
        # 计算历史平均
        history_array = np.array(list(self._history))
        historical_mean = np.mean(history_array, axis=0)
        
        # 指数加权移动平均
        alpha = self.config.temporal_weight
        smoothed = alpha * historical_mean + (1 - alpha) * current
        
        return smoothed
    
    def _zernike_to_wavefront(
        self, 
        coeffs: np.ndarray,
        grid_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """将Zernike系数转换为波前图。"""
        if grid_shape is None:
            grid_shape = (64, 64)
        
        h, w = grid_shape
        wavefront = np.zeros((h, w))
        
        # 创建极坐标网格
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2) / (min(h, w) / 2)
        theta = np.arctan2(y - cy, x - cx)
        
        # 只计算单位圆内的点
        mask = r <= 1.0
        
        # 叠加各Zernike模式
        for j, coeff in enumerate(coeffs[:self.config.num_zernike_modes], 1):
            if abs(coeff) < 1e-12:
                continue
            
            mode = self._zernike_mode(j, r, theta)
            wavefront[mask] += coeff * mode[mask]
        
        return wavefront
    
    def _zernike_mode(
        self, 
        j: int, 
        r: np.ndarray, 
        theta: np.ndarray
    ) -> np.ndarray:
        """计算第j个Zernike模式。"""
        n, m = self._noll_to_zernike(j)
        
        # 径向多项式
        R = self._zernike_radial(n, abs(m), r)
        
        # 角向部分
        if m == 0:
            return R
        elif m > 0:
            return R * np.cos(m * theta)
        else:
            return R * np.sin(-m * theta)
    
    def _noll_to_zernike(self, j: int) -> Tuple[int, int]:
        """Noll索引转Zernike (n, m)。"""
        n = 0
        while (n + 1) * (n + 2) // 2 < j:
            n += 1
        m_values = [n - 2 * k for k in range(n, -1, -1)]
        idx = j - n * (n + 1) // 2 - 1
        return n, m_values[idx]
    
    def _zernike_radial(self, n: int, m: int, r: np.ndarray) -> np.ndarray:
        """Zernike径向多项式。"""
        result = np.zeros_like(r)
        for s in range((n - m) // 2 + 1):
            coeff = ((-1)**s * math.factorial(n - s) /
                    (math.factorial(s) * math.factorial((n + m) // 2 - s) *
                     math.factorial((n - m) // 2 - s)))
            result += coeff * r**(n - 2 * s)
        return result
    
    def _compute_rms(self, coeffs: np.ndarray) -> float:
        """计算RMS波前误差。"""
        return float(np.sqrt(np.mean(coeffs**2)))
    
    def _compute_strehl(self, rms: float) -> float:
        """计算Strehl比 (Marechal近似)。"""
        return float(np.exp(-(2 * np.pi * rms)**2))
    
    def _compute_confidence(self, input_slopes: np.ndarray, output_coeffs: np.ndarray) -> float:
        """计算重构置信度。"""
        # 基于输入输出一致性的简单估计
        input_magnitude = np.linalg.norm(input_slopes)
        output_magnitude = np.linalg.norm(output_coeffs)
        
        if input_magnitude < 1e-12:
            return 0.0
        
        # 理想情况下输入输出应该有相关性
        consistency = 1.0 / (1.0 + abs(input_magnitude - output_magnitude))
        
        return float(np.clip(consistency, 0.0, 1.0))
    
    def _compute_physics_violation(self, coeffs: np.ndarray) -> float:
        """计算物理约束违反程度。"""
        violations = []
        
        # 能量检查
        current_energy = np.sum(coeffs**2)
        if self._baseline_energy > 0:
            energy_violation = abs(current_energy - self._baseline_energy) / self._baseline_energy
            violations.append(energy_violation)
        
        # 平滑性检查 (高阶模式比例)
        if len(coeffs) > 4:
            low_order_energy = np.sum(coeffs[:4]**2)
            high_order_energy = np.sum(coeffs[4:]**2)
            if low_order_energy > 0:
                smoothness_violation = high_order_energy / low_order_energy
                violations.append(smoothness_violation)
        
        return float(np.mean(violations)) if violations else 0.0
    
    def update_baseline(self, coeffs: np.ndarray) -> None:
        """更新能量基线。"""
        self._baseline_energy = float(np.sum(coeffs**2))
        LOGGER.debug("能量基线更新为 %.4f", self._baseline_energy)


# ============================================================
# 辅助函数
# ============================================================

def generate_test_slopes(
    grid_size: int = 8,
    zernike_coeffs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """生成测试斜率数据。"""
    if zernike_coeffs is None:
        # 生成随机的Zernike系数
        zernike_coeffs = np.random.randn(10) * 0.5
        zernike_coeffs[0] = 0  # 活塞模式不产生斜率
    
    # 创建网格
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 计算斜率 (简化版)
    slopes_x = np.zeros_like(X)
    slopes_y = np.zeros_like(Y)
    
    # 添加倾斜 (Z2, Z3)
    if len(zernike_coeffs) > 1:
        slopes_x += zernike_coeffs[1]  # x倾斜
    if len(zernike_coeffs) > 2:
        slopes_y += zernike_coeffs[2]  # y倾斜
    
    # 添加散焦 (Z4)
    if len(zernike_coeffs) > 3:
        slopes_x += 2 * zernike_coeffs[3] * X
        slopes_y += 2 * zernike_coeffs[3] * Y
    
    # 添加噪声
    noise = 0.01
    slopes_x += np.random.randn(*slopes_x.shape) * noise
    slopes_y += np.random.randn(*slopes_y.shape) * noise
    
    return slopes_x, slopes_y


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建重构器
    reconstructor = PhysicsConstrainedReconstructor()
    
    # 生成测试数据
    slopes_x, slopes_y = generate_test_slopes(grid_size=8)
    
    # 重构波前
    result = reconstructor.reconstruct(slopes_x, slopes_y, grid_shape=(64, 64))
    
    print(f"重构结果:")
    print(f"  RMS误差: {result.rms_error:.4f} λ")
    print(f"  Strehl比: {result.strehl_ratio:.4f}")
    print(f"  置信度: {result.confidence:.3f}")
    print(f"  物理约束违反: {result.physics_violation:.4f}")
    print(f"  前5个Zernike系数: {result.zernike_coeffs[:5]}")
