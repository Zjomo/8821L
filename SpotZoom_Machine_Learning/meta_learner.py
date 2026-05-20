"""
元学习自适应控制器 (MetaLearningController)

基于Model-Agnostic Meta-Learning (MAML)的快速自适应控制器。

算法原理:
- MAML (Finn et al., 2017) — 模型无关元学习
- Reptile (Nichol et al., 2018) — 一阶元学习近似
- FOMAML — 一阶梯度元学习

功能:
- 快速任务适应
- 少样本学习
- 跨任务知识迁移

依赖: numpy, scipy
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any
from enum import Enum, auto
import numpy as np


class MetaLearningAlgorithm(Enum):
    """元学习算法类型"""
    MAML = auto()
    FOMAML = auto()
    REPTILE = auto()


@dataclass
class Task:
    """元学习任务"""
    name: str
    support_set: np.ndarray  # 支持样本 [N_support, D]
    query_set: np.ndarray    # 查询样本 [N_query, D]
    support_labels: np.ndarray
    query_labels: np.ndarray


@dataclass
class MetaLearnerState:
    """元学习器状态"""
    inner_steps: int
    inner_lr: float
    outer_lr: float
    current_task: Optional[str] = None
    adaptation_history: List[Dict[str, float]] = field(default_factory=list)


class SimpleMLP:
    """简单多层感知机（用于元学习）"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu'
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        
        # 初始化权重
        self.weights = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            self.weights.append(w.astype(np.float32))
            
        self._activation_fn = {
            'relu': lambda x: np.maximum(0, x),
            'tanh': np.tanh,
            'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        }.get(activation, lambda x: x)
        
    def forward(self, x: np.ndarray, weights: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """前向传播
        
        Parameters:
            x: 输入 [batch, input_dim]
            weights: 可选的自定义权重
            
        Returns:
            输出 [batch, output_dim]
        """
        use_weights = weights if weights is not None else self.weights
        
        h = x.astype(np.float32)
        for i, w in enumerate(use_weights[:-1]):
            h = h @ w
            h = self._activation_fn(h)
            
        h = h @ use_weights[-1]
        return h
        
    def num_params(self) -> int:
        """返回参数总数"""
        return sum(w.size for w in self.weights)
        
    def get_flat_weights(self) -> np.ndarray:
        """展平权重向量"""
        return np.concatenate([w.flatten() for w in self.weights])
        
    def set_flat_weights(self, flat: np.ndarray) -> None:
        """从展平向量设置权重"""
        idx = 0
        new_weights = []
        for w in self.weights:
            size = w.size
            new_weights.append(flat[idx:idx+size].reshape(w.shape))
            idx += size
        self.weights = new_weights


class MetaLearningController:
    """基于元学习的自适应控制器。
    
    支持MAML、FOMAML和Reptile算法，用于快速适应新任务。
    
    Attributes:
        input_dim: 输入维度
        output_dim: 输出维度
        algorithm: 元学习算法类型
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 32],
        algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.FOMAML,
        inner_steps: int = 5,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        meta_batch_size: int = 4,
        gradient_clip: float = 1.0,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.algorithm = algorithm
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.meta_batch_size = meta_batch_size
        self.gradient_clip = gradient_clip
        
        # 创建元学习器
        self.model = SimpleMLP(input_dim, hidden_dims, output_dim)
        
        # 初始化优化器状态
        self.state = MetaLearnerState(
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            outer_lr=outer_lr
        )
        
        # 任务经验缓冲
        self._task_buffer: Dict[str, List[Task]] = {}
        self._adaptation_stats: Dict[str, List[float]] = {}
        
    def _clone_weights(self) -> List[np.ndarray]:
        """复制当前权重"""
        return [w.copy() for w in self.model.weights]
        
    def _compute_gradients(
        self,
        weights: List[np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
        loss_fn: Callable[[np.ndarray, np.ndarray], float]
    ) -> List[np.ndarray]:
        """数值梯度计算（简化版）"""
        grads = []
        flat_weights = self.model.get_flat_weights()
        eps = 1e-5
        
        loss_base = loss_fn(self.model.forward(x, weights), y)
        
        for i in range(flat_weights.size):
            perturbed = flat_weights.copy()
            perturbed[i] += eps
            self.model.set_flat_weights(perturbed)
            loss_plus = loss_fn(self.model.forward(x), y)
            grads.append((loss_plus - loss_base) / eps)
            
        self.model.set_flat_weights(flat_weights)
        return [np.array(g).reshape(w.shape) for g, w in zip(
            np.array(grads).reshape(len(weights), -1),
            weights
        )]
        
    def _inner_update(
        self,
        task: Task,
        weights: List[np.ndarray],
        loss_fn: Callable[[np.ndarray, np.ndarray], float]
    ) -> Tuple[List[np.ndarray], float]:
        """内循环更新
        
        Parameters:
            task: 当前任务
            weights: 当前权重
            loss_fn: 损失函数
            
        Returns:
            更新后的权重和最终损失
        """
        adapted_weights = [w.copy() for w in weights]
        final_loss = 0.0
        
        for step in range(self.inner_steps):
            # 前向传播
            predictions = self.model.forward(task.support_set, adapted_weights)
            loss = loss_fn(predictions, task.support_labels)
            final_loss = loss
            
            # 梯度更新（简化数值梯度）
            grads = self._compute_numerical_gradients(
                adapted_weights, task.support_set, task.support_labels, loss_fn
            )
            
            # 梯度裁剪
            grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
            if grad_norm > self.gradient_clip:
                grads = [g * self.gradient_clip / grad_norm for g in grads]
                
            # 权重更新
            adapted_weights = [
                w - self.inner_lr * g
                for w, g in zip(adapted_weights, grads)
            ]
            
        return adapted_weights, final_loss
        
    def _compute_numerical_gradients(
        self,
        weights: List[np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
        loss_fn: Callable,
        eps: float = 1e-5
    ) -> List[np.ndarray]:
        """数值梯度计算（O(n)方法）"""
        flat_weights = self.model.get_flat_weights()
        n = flat_weights.size
        grads = np.zeros(n)
        
        loss_base = loss_fn(self.model.forward(x, weights), y)
        
        for i in range(n):
            flat_plus = flat_weights.copy()
            flat_plus[i] += eps
            self.model.set_flat_weights(flat_plus)
            loss_plus = loss_fn(self.model.forward(x), y)
            grads[i] = (loss_plus - loss_base) / eps
            
        self.model.set_flat_weights(flat_weights)
        
        # 重新组织为权重形状
        result = []
        idx = 0
        for w in weights:
            size = w.size
            result.append(grads[idx:idx+size].reshape(w.shape).astype(np.float32))
            idx += size
        return result
        
    def _outer_update(
        self,
        task_batch: List[Task],
        loss_fn: Callable[[np.ndarray, np.ndarray], float]
    ) -> float:
        """外循环更新"""
        meta_grad = None
        
        for task in task_batch:
            # 内循环适应
            if self.algorithm == MetaLearningAlgorithm.REPTILE:
                adapted_weights = self._reptile_inner(task, loss_fn)
            else:
                adapted_weights, _ = self._inner_update(task, self.model.weights, loss_fn)
                
            # 在查询集上评估
            query_pred = self.model.forward(task.query_set, adapted_weights)
            query_loss = loss_fn(query_pred, task.query_labels)
            
            # 计算元梯度
            grads = self._compute_numerical_gradients(
                adapted_weights, task.query_set, task.query_labels, loss_fn
            )
            
            if meta_grad is None:
                meta_grad = grads
            else:
                meta_grad = [
                    g1 + g2 for g1, g2 in zip(meta_grad, grads)
                ]
                
        # 梯度平均
        meta_grad = [g / len(task_batch) for g in meta_grad]
        
        # 元学习器更新
        self.model.weights = [
            w - self.outer_lr * g
            for w, g in zip(self.model.weights, meta_grad)
        ]
        
        return query_loss
        
    def _reptile_inner(
        self,
        task: Task,
        loss_fn: Callable
    ) -> List[np.ndarray]:
        """Reptile内循环"""
        weights = self._clone_weights()
        
        for _ in range(self.inner_steps):
            preds = self.model.forward(task.support_set, weights)
            loss = loss_fn(preds, task.support_labels)
            
            grads = self._compute_numerical_gradients(
                weights, task.support_set, task.support_labels, loss_fn
            )
            weights = [w - self.inner_lr * g for w, g in zip(weights, grads)]
            
        return weights
        
    def mse_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """MSE损失函数"""
        return float(np.mean((pred - target) ** 2))
        
    def train(
        self,
        tasks: List[Task],
        num_epochs: int = 100,
        verbose: bool = True
    ) -> List[float]:
        """元训练
        
        Parameters:
            tasks: 任务列表
            num_epochs: 训练轮数
            verbose: 是否打印进度
            
        Returns:
            训练损失历史
        """
        history = []
        
        for epoch in range(num_epochs):
            # 随机采样任务批次
            indices = np.random.choice(
                len(tasks),
                min(self.meta_batch_size, len(tasks)),
                replace=False
            )
            task_batch = [tasks[i] for i in indices]
            
            # 外循环更新
            loss = self._outer_update(task_batch, self.mse_loss)
            history.append(loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Meta loss = {loss:.6f}")
                
        return history
        
    def adapt(
        self,
        support_set: np.ndarray,
        support_labels: np.ndarray,
        num_steps: Optional[int] = None
    ) -> float:
        """快速适应新任务
        
        Parameters:
            support_set: 支持样本
            support_labels: 支持标签
            num_steps: 适应步数（默认使用inner_steps）
            
        Returns:
            适应后损失
        """
        steps = num_steps if num_steps is not None else self.inner_steps
        
        # 创建临时任务
        task = Task(
            name="adaptation",
            support_set=support_set,
            query_set=support_set,  # 使用相同数据作为查询
            support_labels=support_labels,
            query_labels=support_labels
        )
        
        # 执行内循环
        adapted_weights, loss = self._inner_update(
            task, self.model.weights, self.mse_loss
        )
        
        # 保存适应后权重
        self._adapted_weights = adapted_weights
        self._adaptation_stats.setdefault("last_loss", []).append(loss)
        
        return loss
        
    def predict(
        self,
        x: np.ndarray,
        use_adapted: bool = False
    ) -> np.ndarray:
        """预测
        
        Parameters:
            x: 输入数据
            use_adapted: 是否使用适应后的权重
            
        Returns:
            预测结果
        """
        weights = (
            self._adapted_weights
            if use_adapted and hasattr(self, '_adapted_weights')
            else self.model.weights
        )
        return self.model.forward(x, weights)
        
    def register_task(self, task: Task) -> None:
        """注册任务到经验缓冲"""
        if task.name not in self._task_buffer:
            self._task_buffer[task.name] = []
            self._adaptation_stats[task.name] = []
        self._task_buffer[task.name].append(task)
        
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """获取适应统计摘要"""
        summary = {
            'total_tasks': len(self._task_buffer),
            'total_adaptations': sum(len(v) for v in self._adaptation_stats.values()),
            'algorithm': self.algorithm.name,
            'inner_steps': self.inner_steps,
            'inner_lr': self.inner_lr,
        }
        
        for task_name, losses in self._adaptation_stats.items():
            if losses:
                summary[f'{task_name}_avg_loss'] = float(np.mean(losses))
                summary[f'{task_name}_final_loss'] = float(losses[-1])
                
        return summary
        
    def clone(self) -> 'MetaLearningController':
        """创建控制器副本"""
        clone = MetaLearningController(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims.copy(),
            algorithm=self.algorithm,
            inner_steps=self.inner_steps,
            inner_lr=self.inner_lr,
            outer_lr=self.outer_lr,
            meta_batch_size=self.meta_batch_size,
            gradient_clip=self.gradient_clip,
        )
        clone.model.weights = self._clone_weights()
        return clone


__all__ = [
    'MetaLearningController',
    'MetaLearningAlgorithm',
    'Task',
    'MetaLearnerState',
    'SimpleMLP',
]
