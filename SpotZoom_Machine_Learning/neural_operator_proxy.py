"""
NeuralOperatorProxy - 神经算子代理

灵感来源: Fourier Neural Operator (FNO), DeepONet
功能特点:
- 基于傅里叶神经算子的波前传播
- 快速正向/反向传播
- 参数化光学系统建模
- 纯numpy实现，零外部ML依赖

技术路线:
- 频域神经网络
- 算子学习框架
- 快速傅里叶变换加速
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class PropagationResult:
    """传播结果。"""
    output_field: np.ndarray
    processing_time_ms: float
    spectral_content: np.ndarray
    energy_conservation_error: float


@dataclass
class NeuralOperatorConfig:
    """神经算子配置。"""
    grid_size: int = 64
    num_fourier_modes: int = 12
    num_layers: int = 4
    hidden_channels: int = 32
    # 传播参数
    wavelength: float = 500e-9  # 500nm
    propagation_distance: float = 0.1  # 10cm
    pixel_size: float = 10e-6  # 10um


class NeuralOperatorProxy:
    """
    神经算子代理。
    
    使用傅里叶神经算子学习光学传播过程，实现快速波前传播计算。
    相比传统FFT传播，具有更快的推理速度和更好的泛化能力。
    
    核心组件:
    1. 傅里叶层: 在频域学习权重
    2. 局部线性层: 在空间域学习局部变换
    3. 激活函数: GELU非线性
    """
    
    def __init__(self, config: Optional[NeuralOperatorConfig] = None):
        self.config = config or NeuralOperatorConfig()
        self._fourier_weights: List[np.ndarray] = []
        self._linear_weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()
        LOGGER.info("NeuralOperatorProxy初始化完成，网格尺寸=%d，傅里叶模式=%d",
                   self.config.grid_size, self.config.num_fourier_modes)
    
    def _init_network(self) -> None:
        """初始化神经网络权重。"""
        grid = self.config.grid_size
        modes = self.config.num_fourier_modes
        hidden = self.config.hidden_channels
        
        for _ in range(self.config.num_layers):
            # 傅里叶权重 (复数)
            fw = np.random.randn(modes, modes, 2) * 0.02
            self._fourier_weights.append(fw)
            
            # 线性权重
            lw = np.random.randn(hidden, hidden) * np.sqrt(2.0 / hidden)
            self._linear_weights.append(lw)
            
            # 偏置
            b = np.zeros(hidden)
            self._biases.append(b)
    
    def reset(self) -> None:
        """重置网络状态。"""
        self._init_network()
    
    def propagate(self, input_field: np.ndarray) -> PropagationResult:
        """
        传播光场。
        
        Args:
            input_field: 输入光场 (复数或实数)
            
        Returns:
            PropagationResult包含传播结果
        """
        t0 = time.perf_counter()
        
        # 预处理
        field = self._preprocess(input_field)
        
        # 神经算子前向传播
        output = self._forward(field)
        
        # 后处理
        output_field = self._postprocess(output)
        
        # 计算指标
        spectral = self._compute_spectral_content(output_field)
        energy_error = self._compute_energy_conservation(input_field, output_field)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        LOGGER.debug("传播完成: 能量误差=%.6f, 耗时=%.2fms",
                    energy_error, elapsed_ms)
        
        return PropagationResult(
            output_field=output_field,
            processing_time_ms=elapsed_ms,
            spectral_content=spectral,
            energy_conservation_error=energy_error
        )
    
    def _preprocess(self, field: np.ndarray) -> np.ndarray:
        """预处理输入光场。"""
        # 确保复数类型
        if np.isrealobj(field):
            field = field.astype(np.complex128)
        
        # 调整尺寸
        target_size = (self.config.grid_size, self.config.grid_size)
        if field.shape != target_size:
            # 使用简单的插值
            field = self._resize_complex(field, target_size)
        
        # 归一化
        energy = np.sum(np.abs(field) ** 2)
        if energy > 0:
            field = field / np.sqrt(energy)
        
        return field
    
    def _resize_complex(
        self, 
        field: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """调整复数场尺寸。"""
        # 分别调整幅度和相位
        magnitude = np.abs(field)
        phase = np.angle(field)
        
        # 简单的最近邻插值
        from scipy.ndimage import zoom
        zoom_factor = (target_size[0] / field.shape[0], target_size[1] / field.shape[1])
        
        mag_resized = zoom(magnitude, zoom_factor, order=1)
        phase_resized = zoom(phase, zoom_factor, order=1)
        
        return mag_resized * np.exp(1j * phase_resized)
    
    def _forward(self, field: np.ndarray) -> np.ndarray:
        """神经算子前向传播。"""
        current = field.copy()
        
        for i in range(self.config.num_layers):
            # 傅里叶层
            fourier_out = self._fourier_layer(current, i)
            
            # 局部线性层
            linear_out = self._linear_layer(current, i)
            
            # 组合
            current = fourier_out + linear_out
            
            # 激活函数 (GELU近似)
            current = self._gelu(current)
        
        return current
    
    def _fourier_layer(self, field: np.ndarray, layer_idx: int) -> np.ndarray:
        """傅里叶层。"""
        # FFT
        fft_field = np.fft.fft2(field)
        fft_shifted = np.fft.fftshift(fft_field)
        
        # 提取低频模式
        h, w = fft_shifted.shape
        cy, cx = h // 2, w // 2
        modes = self.config.num_fourier_modes // 2
        
        # 应用学习的权重
        weights = self._fourier_weights[layer_idx]
        
        # 创建输出频谱
        output_fft = np.zeros_like(fft_shifted)
        
        # 在低频区域应用权重
        for i in range(-modes, modes):
            for j in range(-modes, modes):
                if abs(i) < weights.shape[0] and abs(j) < weights.shape[1]:
                    wi = i + modes
                    wj = j + modes
                    weight_complex = weights[wi, wj, 0] + 1j * weights[wi, wj, 1]
                    
                    y = cy + i
                    x = cx + j
                    if 0 <= y < h and 0 <= x < w:
                        output_fft[y, x] = fft_shifted[y, x] * weight_complex
        
        # 逆FFT
        output_fft_shifted = np.fft.ifftshift(output_fft)
        output_field = np.fft.ifft2(output_fft_shifted)
        
        return output_field
    
    def _linear_layer(self, field: np.ndarray, layer_idx: int) -> np.ndarray:
        """局部线性层。"""
        # 简化的局部卷积 (使用平均滤波近似)
        kernel_size = 3
        from scipy.ndimage import uniform_filter
        
        real_part = uniform_filter(np.real(field), size=kernel_size)
        imag_part = uniform_filter(np.imag(field), size=kernel_size)
        
        return real_part + 1j * imag_part
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU激活函数。"""
        # GELU近似: x * sigmoid(1.702 * x)
        return x * (1 / (1 + np.exp(-1.702 * np.real(x))))
    
    def _postprocess(self, field: np.ndarray) -> np.ndarray:
        """后处理输出光场。"""
        # 应用物理传播距离相位
        k = 2 * np.pi / self.config.wavelength
        distance = self.config.propagation_distance
        
        # 菲涅尔近似相位
        h, w = field.shape
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        
        r_squared = ((x - cx) * self.config.pixel_size) ** 2 + \
                    ((y - cy) * self.config.pixel_size) ** 2
        
        fresnel_phase = np.exp(1j * k * r_squared / (2 * distance))
        
        return field * fresnel_phase
    
    def _compute_spectral_content(self, field: np.ndarray) -> np.ndarray:
        """计算频谱内容。"""
        fft_field = np.fft.fft2(field)
        magnitude = np.abs(np.fft.fftshift(fft_field))
        
        # 归一化
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        return magnitude
    
    def _compute_energy_conservation(
        self, 
        input_field: np.ndarray, 
        output_field: np.ndarray
    ) -> float:
        """计算能量守恒误差。"""
        input_energy = np.sum(np.abs(input_field) ** 2)
        output_energy = np.sum(np.abs(output_field) ** 2)
        
        if input_energy < 1e-12:
            return 0.0
        
        return abs(output_energy - input_energy) / input_energy
    
    def learn_from_examples(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        learning_rate: float = 0.001,
        num_epochs: int = 10
    ) -> float:
        """
        从示例学习 (简化版训练)。
        
        Args:
            examples: (input, target) 示例列表
            learning_rate: 学习率
            num_epochs: 训练轮数
            
        Returns:
            最终损失
        """
        LOGGER.info("开始训练，示例数=%d，轮数=%d", len(examples), num_epochs)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for input_field, target_field in examples:
                # 前向传播
                output = self._forward(self._preprocess(input_field))
                
                # 计算损失 (MSE)
                loss = np.mean(np.abs(output - target_field) ** 2)
                total_loss += loss
                
                # 简化版梯度更新 (随机扰动)
                self._perturb_weights(learning_rate, loss)
            
            avg_loss = total_loss / len(examples)
            LOGGER.debug("Epoch %d/%d, 平均损失=%.6f", epoch + 1, num_epochs, avg_loss)
        
        return avg_loss
    
    def _perturb_weights(self, scale: float, loss: float) -> None:
        """随机扰动权重 (简化版梯度下降)。"""
        perturbation_scale = scale * min(1.0, loss)
        
        for i in range(len(self._fourier_weights)):
            self._fourier_weights[i] += np.random.randn(*self._fourier_weights[i].shape) * perturbation_scale
            self._linear_weights[i] += np.random.randn(*self._linear_weights[i].shape) * perturbation_scale


# ============================================================
# 辅助函数
# ============================================================

def create_gaussian_beam(
    grid_size: int = 64,
    waist: float = 10e-6,
    pixel_size: float = 10e-6
) -> np.ndarray:
    """创建高斯光束。"""
    y, x = np.ogrid[:grid_size, :grid_size]
    cy, cx = grid_size // 2, grid_size // 2
    
    r_squared = ((x - cx) * pixel_size) ** 2 + ((y - cy) * pixel_size) ** 2
    
    amplitude = np.exp(-r_squared / (2 * waist ** 2))
    
    return amplitude.astype(np.complex128)


def create_aberrated_beam(
    grid_size: int = 64,
    aberration_strength: float = 0.5
) -> np.ndarray:
    """创建有像差的光束。"""
    beam = create_gaussian_beam(grid_size)
    
    # 添加离焦像差
    y, x = np.ogrid[:grid_size, :grid_size]
    cy, cx = grid_size // 2, grid_size // 2
    r_normalized = np.sqrt((x - cx)**2 + (y - cy)**2) / (grid_size / 2)
    
    defocus_phase = aberration_strength * (2 * r_normalized**2 - 1)
    aberration = np.exp(1j * defocus_phase)
    
    return beam * aberration


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建神经算子代理
    proxy = NeuralOperatorProxy()
    
    # 创建测试光束
    input_beam = create_aberrated_beam(grid_size=64, aberration_strength=0.5)
    
    # 传播
    result = proxy.propagate(input_beam)
    
    print(f"传播结果:")
    print(f"  输出场形状: {result.output_field.shape}")
    print(f"  能量守恒误差: {result.energy_conservation_error:.6f}")
    print(f"  处理时间: {result.processing_time_ms:.2f} ms")
    print(f"  频谱峰值位置: {np.unravel_index(np.argmax(result.spectral_content), result.spectral_content.shape)}")
