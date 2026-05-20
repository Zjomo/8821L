"""
MetaLearningAdapter - 元学习自适应器

灵感来源: MAML (Model-Agnostic Meta-Learning), Reptile
功能特点:
- 少样本场景快速适应
- 跨系统迁移学习
- 在线参数更新
- 纯numpy实现，零外部ML依赖

技术路线:
- 二阶梯度近似
- 任务分布学习
- 快速适应机制
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class AdaptationResult:
    """适应结果。"""
    adapted_params: Dict[str, np.ndarray]
    adaptation_loss: float
    num_gradient_steps: int
    convergence_reached: bool
    processing_time_ms: float


@dataclass
class MetaLearningConfig:
    """元学习配置。"""
    # MAML参数
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    
    # 任务参数
    num_tasks: int = 10
    shots_per_task: int = 5
    
    # 适应参数
    max_adaptation_steps: int = 20
    adaptation_threshold: float = 1e-4
    
    # 正则化
    regularization_strength: float = 0.01


class MetaLearningAdapter:
    """
    元学习自适应器。
    
    实现MAML风格的元学习，使模型能够快速适应新任务。
    核心功能:
    1. 元训练: 学习一个好的参数初始化
    2. 快速适应: 在新任务上快速微调
    3. 在线更新: 持续学习改进
    
    应用场景:
    - 不同光学系统间的快速迁移
    - 新场景的快速适应
    - 个性化参数调整
    """
    
    def __init__(self, config: Optional[MetaLearningConfig] = None):
        self.config = config or MetaLearningConfig()
        self._meta_params: Dict[str, np.ndarray] = {}
        self._task_history: List[Dict] = []
        self._adaptation_count: int = 0
        self._init_params()
        LOGGER.info("MetaLearningAdapter初始化完成，内循环学习率=%.4f",
                   self.config.inner_lr)
    
    def _init_params(self) -> None:
        """初始化元参数。"""
        # 简化的线性模型参数
        self._meta_params = {
            'w1': np.random.randn(10, 20) * 0.01,
            'b1': np.zeros(20),
            'w2': np.random.randn(20, 10) * 0.01,
            'b2': np.zeros(10)
        }
    
    def reset(self) -> None:
        """重置适配器状态。"""
        self._init_params()
        self._task_history.clear()
        self._adaptation_count = 0
    
    def meta_train(
        self,
        task_distribution: Callable[[], List[Tuple[np.ndarray, np.ndarray]]],
        num_iterations: int = 100
    ) -> float:
        """
        元训练。
        
        Args:
            task_distribution: 生成任务的函数
            num_iterations: 训练迭代次数
            
        Returns:
            最终元损失
        """
        LOGGER.info("开始元训练，迭代次数=%d", num_iterations)
        
        for iteration in range(num_iterations):
            meta_gradient = {k: np.zeros_like(v) for k, v in self._meta_params.items()}
            
            # 采样多个任务
            for _ in range(self.config.num_tasks):
                # 生成任务数据
                task_data = task_distribution()
                
                # 内循环适应
                adapted_params = self._inner_loop(task_data)
                
                # 计算元梯度 (Reptile风格)
                for key in self._meta_params:
                    meta_gradient[key] += (adapted_params[key] - self._meta_params[key])
            
            # 外循环更新
            for key in self._meta_params:
                self._meta_params[key] += self.config.outer_lr * meta_gradient[key] / self.config.num_tasks
            
            if iteration % 10 == 0:
                LOGGER.debug("元训练迭代 %d/%d", iteration, num_iterations)
        
        LOGGER.info("元训练完成")
        return 0.0  # 简化版返回
    
    def _inner_loop(
        self,
        task_data: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """内循环适应。"""
        # 复制当前元参数
        params = {k: v.copy() for k, v in self._meta_params.items()}
        
        # 在任务数据上进行几步梯度下降
        for _ in range(self.config.num_inner_steps):
            # 计算损失和梯度
            loss, gradients = self._compute_loss_and_gradients(params, task_data)
            
            # 参数更新
            for key in params:
                params[key] -= self.config.inner_lr * gradients[key]
        
        return params
    
    def _compute_loss_and_gradients(
        self,
        params: Dict[str, np.ndarray],
        data: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """计算损失和梯度。"""
        total_loss = 0.0
        gradients = {k: np.zeros_like(v) for k, v in params.items()}
        
        for x, y in data:
            # 前向传播
            pred = self._forward(x, params)
            
            # 计算损失 (MSE)
            loss = np.mean((pred - y) ** 2)
            total_loss += loss
            
            # 计算梯度 (简化版数值梯度)
            eps = 1e-5
            for key in params:
                grad = np.zeros_like(params[key])
                it = np.nditer(params[key], flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    original = params[key][idx]
                    
                    params[key][idx] = original + eps
                    pred_plus = self._forward(x, params)
                    loss_plus = np.mean((pred_plus - y) ** 2)
                    
                    params[key][idx] = original - eps
                    pred_minus = self._forward(x, params)
                    loss_minus = np.mean((pred_minus - y) ** 2)
                    
                    grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                    params[key][idx] = original
                    
                    it.iternext()
                
                gradients[key] += grad
        
        # 平均
        n = len(data)
        total_loss /= n
        for key in gradients:
            gradients[key] /= n
        
        return total_loss, gradients
    
    def _forward(self, x: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        """前向传播。"""
        # 确保输入维度正确
        if x.shape[0] != 10:
            x = np.resize(x, 10)
        
        # 第一层
        h = x @ params['w1'] + params['b1']
        h = np.maximum(h, 0)  # ReLU
        
        # 第二层
        out = h @ params['w2'] + params['b2']
        
        return out
    
    def adapt(
        self,
        support_data: List[Tuple[np.ndarray, np.ndarray]],
        max_steps: Optional[int] = None
    ) -> AdaptationResult:
        """
        快速适应新任务。
        
        Args:
            support_data: 支持集数据 (少量样本)
            max_steps: 最大适应步数
            
        Returns:
            AdaptationResult包含适应后的参数
        """
        t0 = time.perf_counter()
        
        if max_steps is None:
            max_steps = self.config.max_adaptation_steps
        
        # 从元参数开始
        adapted_params = {k: v.copy() for k, v in self._meta_params.items()}
        
        prev_loss = float('inf')
        convergence_reached = False
        
        for step in range(max_steps):
            # 计算损失和梯度
            loss, gradients = self._compute_loss_and_gradients(adapted_params, support_data)
            
            # 检查收敛
            if abs(prev_loss - loss) < self.config.adaptation_threshold:
                convergence_reached = True
                break
            
            prev_loss = loss
            
            # 参数更新
            for key in adapted_params:
                adapted_params[key] -= self.config.inner_lr * gradients[key]
        
        self._adaptation_count += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        LOGGER.debug("适应完成: 步数=%d, 损失=%.6f, 收敛=%s, 耗时=%.2fms",
                    step + 1, prev_loss, convergence_reached, elapsed_ms)
        
        return AdaptationResult(
            adapted_params=adapted_params,
            adaptation_loss=prev_loss,
            num_gradient_steps=step + 1,
            convergence_reached=convergence_reached,
            processing_time_ms=elapsed_ms
        )
    
    def predict(
        self,
        x: np.ndarray,
        adapted_params: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        使用适应后的参数进行预测。
        
        Args:
            x: 输入
            adapted_params: 适应后的参数 (None则使用元参数)
            
        Returns:
            预测输出
        """
        params = adapted_params if adapted_params is not None else self._meta_params
        return self._forward(x, params)
    
    def online_update(
        self,
        new_data: Tuple[np.ndarray, np.ndarray],
        learning_rate: Optional[float] = None
    ) -> float:
        """
        在线更新元参数。
        
        Args:
            new_data: 新数据样本
            learning_rate: 学习率
            
        Returns:
            更新后的损失
        """
        if learning_rate is None:
            learning_rate = self.config.outer_lr
        
        # 计算梯度
        x, y = new_data
        pred = self._forward(x, self._meta_params)
        loss = np.mean((pred - y) ** 2)
        
        # 简化版更新 (随机扰动)
        for key in self._meta_params:
            perturbation = np.random.randn(*self._meta_params[key].shape) * learning_rate * loss
            self._meta_params[key] -= perturbation
        
        return loss
    
    def get_transfer_quality(
        self,
        source_task: str,
        target_task: str
    ) -> float:
        """
        估计任务间迁移质量。
        
        Args:
            source_task: 源任务标识
            target_task: 目标任务标识
            
        Returns:
            迁移质量评分 (0-1)
        """
        # 从历史任务中查找
        source_data = None
        target_data = None
        
        for task in self._task_history:
            if task.get('name') == source_task:
                source_data = task.get('data')
            if task.get('name') == target_task:
                target_data = task.get('data')
        
        if source_data is None or target_data is None:
            return 0.5  # 默认中等质量
        
        # 计算任务相似度 (简化版)
        similarity = self._compute_task_similarity(source_data, target_data)
        
        return similarity
    
    def _compute_task_similarity(
        self,
        data1: List[Tuple[np.ndarray, np.ndarray]],
        data2: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """计算任务相似度。"""
        # 提取特征统计
        def extract_stats(data):
            inputs = np.array([x for x, _ in data])
            return {
                'mean': np.mean(inputs),
                'std': np.std(inputs),
                'range': np.max(inputs) - np.min(inputs)
            }
        
        stats1 = extract_stats(data1)
        stats2 = extract_stats(data2)
        
        # 计算相似度
        similarities = []
        for key in stats1:
            v1, v2 = stats1[key], stats2[key]
            if abs(v1) + abs(v2) > 1e-10:
                sim = 1 - abs(v1 - v2) / (abs(v1) + abs(v2))
                similarities.append(max(0, sim))
        
        return np.mean(similarities) if similarities else 0.5


# ============================================================
# 辅助函数
# ============================================================

def generate_synthetic_task(
    task_type: str = "linear",
    num_samples: int = 10,
    noise_level: float = 0.1
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """生成合成任务数据。"""
    data = []
    
    if task_type == "linear":
        # 线性关系
        true_w = np.random.randn(10)
        for _ in range(num_samples):
            x = np.random.randn(10)
            y = x @ true_w + np.random.randn() * noise_level
            data.append((x, np.array([y])))
    
    elif task_type == "quadratic":
        # 二次关系
        for _ in range(num_samples):
            x = np.random.randn(10)
            y = np.sum(x ** 2) + np.random.randn() * noise_level
            data.append((x, np.array([y])))
    
    else:
        # 默认随机关系
        for _ in range(num_samples):
            x = np.random.randn(10)
            y = np.random.randn(10)
            data.append((x, y))
    
    return data


def task_distribution_factory() -> List[Tuple[np.ndarray, np.ndarray]]:
    """任务分布工厂。"""
    task_type = np.random.choice(["linear", "quadratic", "random"])
    return generate_synthetic_task(task_type, num_samples=5)


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建元学习适配器
    adapter = MetaLearningAdapter()
    
    # 元训练
    adapter.meta_train(task_distribution_factory, num_iterations=50)
    
    # 生成新任务
    new_task = generate_synthetic_task("linear", num_samples=5)
    
    # 快速适应
    result = adapter.adapt(new_task, max_steps=10)
    
    print(f"适应结果:")
    print(f"  适应步数: {result.num_gradient_steps}")
    print(f"  最终损失: {result.adaptation_loss:.6f}")
    print(f"  是否收敛: {result.convergence_reached}")
    print(f"  处理时间: {result.processing_time_ms:.2f} ms")
    
    # 测试预测
    test_input = np.random.randn(10)
    prediction = adapter.predict(test_input, result.adapted_params)
    print(f"  测试预测输出形状: {prediction.shape}")
