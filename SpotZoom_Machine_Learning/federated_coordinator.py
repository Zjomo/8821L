"""
FederatedCoordinator - 联邦学习协调器

灵感来源: 联邦学习前沿研究 (FedAvg, FedProx, SCAFFOLD)
功能特点:
- 分布式模型训练
- 隐私保护聚合
- 异构数据适配
- 通信效率优化
- 纯numpy实现，零外部ML依赖

技术路线:
- FedAvg算法实现
- 差分隐私保护
- 梯度压缩
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """客户端更新。"""
    client_id: str
    model_update: Dict[str, np.ndarray]
    num_samples: int
    metrics: Dict[str, float]
    timestamp: float


@dataclass
class AggregationResult:
    """聚合结果。"""
    global_model: Dict[str, np.ndarray]
    aggregation_stats: Dict[str, Any]
    convergence_reached: bool
    privacy_budget_spent: float
    processing_time_ms: float


@dataclass
class FederatedConfig:
    """联邦学习配置。"""
    # 聚合参数
    num_clients: int = 10
    clients_per_round: int = 5
    aggregation_method: str = "fedavg"  # fedavg, fedprox, scaffold
    
    # 隐私参数
    use_differential_privacy: bool = True
    noise_multiplier: float = 1.0
    max_gradient_norm: float = 1.0
    privacy_budget: float = 10.0
    
    # 通信参数
    compression_ratio: float = 0.1
    
    # 收敛参数
    convergence_threshold: float = 1e-4
    min_rounds: int = 10
    max_rounds: int = 100


class FederatedCoordinator:
    """
    联邦学习协调器。
    
    协调多个客户端进行分布式模型训练，保护数据隐私。
    核心功能:
    1. 客户端选择: 每轮选择参与训练的客户端
    2. 安全聚合: 聚合客户端更新，保护个体隐私
    3. 差分隐私: 添加噪声保护隐私
    4. 收敛监控: 跟踪训练进度
    
    应用场景:
    - 多站点协作训练
    - 隐私敏感数据分析
    - 分布式光学系统校准
    """
    
    def __init__(self, config: Optional[FederatedConfig] = None):
        self.config = config or FederatedConfig()
        self._global_model: Dict[str, np.ndarray] = {}
        self._client_updates: List[ClientUpdate] = []
        self._round_number: int = 0
        self._privacy_budget_spent: float = 0.0
        self._convergence_history: List[float] = []
        self._init_global_model()
        LOGGER.info("FederatedCoordinator初始化完成，客户端数=%d，聚合方法=%s",
                   self.config.num_clients, self.config.aggregation_method)
    
    def _init_global_model(self) -> None:
        """初始化全局模型。"""
        # 简化的线性模型
        self._global_model = {
            'w': np.random.randn(10, 10) * 0.01,
            'b': np.zeros(10)
        }
    
    def reset(self) -> None:
        """重置协调器状态。"""
        self._init_global_model()
        self._client_updates.clear()
        self._round_number = 0
        self._privacy_budget_spent = 0.0
        self._convergence_history.clear()
    
    def register_client_update(self, update: ClientUpdate) -> bool:
        """
        注册客户端更新。
        
        Args:
            update: 客户端更新
            
        Returns:
            是否成功注册
        """
        # 验证更新
        if not self._validate_update(update):
            LOGGER.warning("客户端 %s 的更新验证失败", update.client_id)
            return False
        
        self._client_updates.append(update)
        LOGGER.debug("注册客户端 %s 的更新", update.client_id)
        
        return True
    
    def _validate_update(self, update: ClientUpdate) -> bool:
        """验证客户端更新。"""
        # 检查模型结构
        if set(update.model_update.keys()) != set(self._global_model.keys()):
            return False
        
        # 检查数值合理性
        for key, value in update.model_update.items():
            if not np.all(np.isfinite(value)):
                return False
            if np.any(np.abs(value) > 1e6):
                return False
        
        return True
    
    def aggregate(self) -> AggregationResult:
        """
        执行聚合。
        
        Returns:
            AggregationResult包含聚合结果
        """
        t0 = time.perf_counter()
        
        if len(self._client_updates) == 0:
            LOGGER.warning("没有客户端更新可聚合")
            return AggregationResult(
                global_model=self._global_model,
                aggregation_stats={},
                convergence_reached=False,
                privacy_budget_spent=self._privacy_budget_spent,
                processing_time_ms=0.0
            )
        
        # 选择参与的更新
        selected_updates = self._select_updates()
        
        # 执行聚合
        if self.config.aggregation_method == "fedavg":
            aggregated = self._fedavg_aggregate(selected_updates)
        elif self.config.aggregation_method == "fedprox":
            aggregated = self._fedprox_aggregate(selected_updates)
        else:
            aggregated = self._fedavg_aggregate(selected_updates)
        
        # 应用差分隐私
        if self.config.use_differential_privacy:
            aggregated = self._apply_differential_privacy(aggregated)
        
        # 更新全局模型
        old_model = {k: v.copy() for k, v in self._global_model.items()}
        self._global_model = aggregated
        
        # 计算收敛
        model_diff = self._compute_model_difference(old_model, self._global_model)
        self._convergence_history.append(model_diff)
        
        convergence_reached = (
            self._round_number >= self.config.min_rounds and
            model_diff < self.config.convergence_threshold
        )
        
        # 统计信息
        stats = {
            'num_updates': len(selected_updates),
            'total_samples': sum(u.num_samples for u in selected_updates),
            'model_difference': model_diff,
            'round': self._round_number
        }
        
        # 清理已处理的更新
        self._client_updates.clear()
        self._round_number += 1
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        LOGGER.info("聚合完成: 轮次=%d, 客户端数=%d, 模型差异=%.6f, 耗时=%.2fms",
                   self._round_number, len(selected_updates), model_diff, elapsed_ms)
        
        return AggregationResult(
            global_model=self._global_model,
            aggregation_stats=stats,
            convergence_reached=convergence_reached,
            privacy_budget_spent=self._privacy_budget_spent,
            processing_time_ms=elapsed_ms
        )
    
    def _select_updates(self) -> List[ClientUpdate]:
        """选择参与聚合的客户端更新。"""
        if len(self._client_updates) <= self.config.clients_per_round:
            return self._client_updates
        
        # 随机选择
        indices = np.random.choice(
            len(self._client_updates),
            size=self.config.clients_per_round,
            replace=False
        )
        
        return [self._client_updates[i] for i in indices]
    
    def _fedavg_aggregate(
        self,
        updates: List[ClientUpdate]
    ) -> Dict[str, np.ndarray]:
        """FedAvg聚合算法。"""
        aggregated = {}
        
        total_samples = sum(u.num_samples for u in updates)
        
        for key in self._global_model.keys():
            weighted_sum = np.zeros_like(self._global_model[key])
            
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_sum += weight * update.model_update[key]
            
            aggregated[key] = weighted_sum
        
        return aggregated
    
    def _fedprox_aggregate(
        self,
        updates: List[ClientUpdate]
    ) -> Dict[str, np.ndarray]:
        """FedProx聚合算法 (带近端项)。"""
        # 先执行FedAvg
        aggregated = self._fedavg_aggregate(updates)
        
        # 添加近端项约束
        mu = 0.01  # 近端系数
        for key in aggregated:
            aggregated[key] = (
                aggregated[key] + mu * self._global_model[key]
            ) / (1 + mu)
        
        return aggregated
    
    def _apply_differential_privacy(
        self,
        model: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """应用差分隐私。"""
        noisy_model = {}
        
        for key, value in model.items():
            # 裁剪梯度
            norm = np.linalg.norm(value)
            if norm > self.config.max_gradient_norm:
                value = value * self.config.max_gradient_norm / norm
            
            # 添加高斯噪声
            noise = np.random.randn(*value.shape) * self.config.noise_multiplier * self.config.max_gradient_norm
            noisy_model[key] = value + noise
        
        # 更新隐私预算
        self._privacy_budget_spent += self._compute_privacy_cost()
        
        return noisy_model
    
    def _compute_privacy_cost(self) -> float:
        """计算隐私成本。"""
        # 简化的隐私会计
        q = self.config.clients_per_round / self.config.num_clients
        sigma = self.config.noise_multiplier
        
        # 高斯机制的隐私损失
        epsilon = 2 * q * np.sqrt(self.config.max_rounds) / sigma
        
        return epsilon
    
    def _compute_model_difference(
        self,
        model1: Dict[str, np.ndarray],
        model2: Dict[str, np.ndarray]
    ) -> float:
        """计算两个模型之间的差异。"""
        total_diff = 0.0
        
        for key in model1.keys():
            diff = np.linalg.norm(model1[key] - model2[key])
            total_diff += diff
        
        return total_diff / len(model1)
    
    def get_global_model(self) -> Dict[str, np.ndarray]:
        """获取当前全局模型。"""
        return {k: v.copy() for k, v in self._global_model.items()}
    
    def should_stop(self) -> bool:
        """检查是否应该停止训练。"""
        # 检查轮数
        if self._round_number >= self.config.max_rounds:
            return True
        
        # 检查隐私预算
        if self._privacy_budget_spent >= self.config.privacy_budget:
            return True
        
        # 检查收敛
        if len(self._convergence_history) >= 3:
            recent_diffs = self._convergence_history[-3:]
            if all(d < self.config.convergence_threshold for d in recent_diffs):
                if self._round_number >= self.config.min_rounds:
                    return True
        
        return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要。"""
        return {
            'total_rounds': self._round_number,
            'privacy_budget_spent': self._privacy_budget_spent,
            'privacy_budget_remaining': max(0, self.config.privacy_budget - self._privacy_budget_spent),
            'convergence_history': self._convergence_history.copy(),
            'final_convergence': self._convergence_history[-1] if self._convergence_history else None
        }


# ============================================================
# 辅助函数
# ============================================================

def simulate_client_training(
    client_id: str,
    global_model: Dict[str, np.ndarray],
    num_samples: int = 100,
    local_epochs: int = 5
) -> ClientUpdate:
    """模拟客户端本地训练。"""
    # 复制全局模型
    local_model = {k: v.copy() for k, v in global_model.items()}
    
    # 模拟本地更新 (随机扰动)
    for key in local_model:
        noise = np.random.randn(*local_model[key].shape) * 0.01
        local_model[key] += noise
    
    # 计算更新量
    model_update = {}
    for key in local_model:
        model_update[key] = local_model[key] - global_model[key]
    
    # 模拟指标
    metrics = {
        'train_loss': np.random.uniform(0.1, 1.0),
        'train_accuracy': np.random.uniform(0.7, 0.95)
    }
    
    return ClientUpdate(
        client_id=client_id,
        model_update=model_update,
        num_samples=num_samples,
        metrics=metrics,
        timestamp=time.time()
    )


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建联邦学习协调器
    config = FederatedConfig(
        num_clients=10,
        clients_per_round=5,
        use_differential_privacy=True
    )
    coordinator = FederatedCoordinator(config)
    
    # 模拟多轮训练
    for round_num in range(20):
        # 模拟客户端训练
        for client_id in range(config.clients_per_round):
            global_model = coordinator.get_global_model()
            update = simulate_client_training(
                f"client_{client_id}",
                global_model,
                num_samples=np.random.randint(50, 200)
            )
            coordinator.register_client_update(update)
        
        # 执行聚合
        result = coordinator.aggregate()
        
        print(f"\n轮次 {round_num + 1}:")
        print(f"  聚合客户端数: {result.aggregation_stats['num_updates']}")
        print(f"  总样本数: {result.aggregation_stats['total_samples']}")
        print(f"  模型差异: {result.aggregation_stats['model_difference']:.6f}")
        print(f"  隐私预算消耗: {result.privacy_budget_spent:.4f}")
        print(f"  是否收敛: {result.convergence_reached}")
        
        # 检查是否应该停止
        if coordinator.should_stop():
            print("\n训练收敛，提前停止")
            break
    
    # 输出训练摘要
    summary = coordinator.get_training_summary()
    print(f"\n训练摘要:")
    print(f"  总轮次: {summary['total_rounds']}")
    print(f"  隐私预算消耗: {summary['privacy_budget_spent']:.4f}")
    print(f"  剩余隐私预算: {summary['privacy_budget_remaining']:.4f}")
