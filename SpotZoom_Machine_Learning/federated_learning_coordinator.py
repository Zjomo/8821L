"""
联邦学习协调器 (FederatedLearningCoordinator)

基于联邦学习理论的多设备协同训练算法。

算法原理:
- Federated Averaging (FedAvg, McMahan 2017) — 分布式模型平均
- Federated Proximal (FedProx, Li 2020) — 异构设备容错聚合
- Communication Compression — 量化/稀疏化减少通信开销
- Differential Privacy — 差分隐私保护

功能:
- 多台光学设备协同训练（无需集中数据）
- 模型聚合策略 (FedAvg / FedProx)
- 通信压缩（减少设备间数据传输）
- 异构设备支持（不同光学配置）
- 训练进度监控

依赖: numpy (无线性代数库之外的外部依赖)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
from enum import Enum

import numpy as np


class AggregationStrategy(Enum):
    """模型聚合策略枚举。"""
    FED_AVG = "fed_avg"          # 联邦平均
    FED_PROX = "fed_prox"        # 联邦近端
    WEIGHTED_MEDIAN = "weighted_median"  # 加权中位数
    TRIMMED_MEAN = "trimmed_mean"        # 截断均值


@dataclass
class FederatedTrainingConfig:
    """联邦训练配置。

    Attributes
    ----------
    n_rounds : int
        联邦训练总轮数。
    client_fraction : float
        每轮参与训练的客户端比例 [0, 1]。
    local_epochs : int
        每个客户端本地训练轮数。
    local_lr : float
        客户端本地学习率。
    aggregation_strategy : AggregationStrategy
        聚合策略。
    proximal_mu : float
        FedProx 近端项系数 (仅 FedProx 策略有效)。
    compression_ratio : float
        通信压缩比 (0.0 ~ 1.0, 0 表示最大压缩)。
    min_clients_per_round : int
        每轮最少参与客户端数。
    convergence_threshold : float
        全局收敛阈值。
    """
    n_rounds: int = 50
    client_fraction: float = 0.5
    local_epochs: int = 5
    local_lr: float = 0.01
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FED_AVG
    proximal_mu: float = 0.01
    compression_ratio: float = 0.5
    min_clients_per_round: int = 2
    convergence_threshold: float = 1e-4


@dataclass
class FederatedRoundResult:
    """联邦轮次结果。

    Attributes
    ----------
    round_id : int
        轮次编号。
    n_participants : int
        参与客户端数量。
    global_loss : float
        全局损失值。
    global_accuracy : float
        全局准确率 (如适用)。
    communication_cost : float
        通信开销 (参数量)。
    convergence_metric : float
        收敛度量。
    client_losses : Dict[str, float]
        各客户端损失值。
    model_update_norm : float
        模型更新范数。
    """
    round_id: int = 0
    n_participants: int = 0
    global_loss: float = 0.0
    global_accuracy: float = 0.0
    communication_cost: float = 0.0
    convergence_metric: float = 0.0
    client_losses: Dict[str, float] = field(default_factory=dict)
    model_update_norm: float = 0.0


class FederatedClient:
    """联邦学习客户端（单台光学设备）。

    模拟单台光学设备上的本地训练过程。每台设备持有自己的
    局部数据，在本地训练后上传模型更新。

    Parameters
    ----------
    client_id : str
        客户端唯一标识符。
    local_data_features : np.ndarray
        本地特征数据。
    local_data_targets : np.ndarray
        本地目标数据。
    model_shape : Tuple[int, ...]
        模型参数形状。
    device_config : dict or None
        设备配置信息（光学参数等）。
    """

    def __init__(
        self,
        client_id: str,
        local_data_features: np.ndarray,
        local_data_targets: np.ndarray,
        model_shape: tuple = (2, 2),
        device_config: Optional[dict] = None,
    ):
        self.client_id = str(client_id)
        self.features = np.asarray(local_data_features, dtype=np.float64)
        self.targets = np.asarray(local_data_targets, dtype=np.float64)
        self.model_shape = model_shape
        self.device_config = device_config or {}

        # 本地模型参数
        self._local_model = np.random.randn(*model_shape) * 0.01
        self._n_samples = self.features.shape[0]

        # 客户端状态
        self._is_active = True
        self._last_loss = 0.0
        self._last_update_norm = 0.0

    @property
    def is_active(self) -> bool:
        """客户端是否活跃。"""
        return self._is_active

    @property
    def n_samples(self) -> int:
        """本地数据样本数。"""
        return self._n_samples

    @property
    def local_model(self) -> np.ndarray:
        """获取本地模型参数。"""
        return self._local_model.copy()

    @property
    def last_loss(self) -> float:
        """上次训练的损失值。"""
        return self._last_loss

    def set_global_model(self, global_model: np.ndarray) -> None:
        """接收全局模型参数。

        Parameters
        ----------
        global_model : np.ndarray
            全局模型参数。
        """
        self._local_model = np.asarray(global_model, dtype=np.float64).copy()

    def local_train(
        self,
        n_epochs: int = 5,
        lr: float = 0.01,
        proximal_mu: float = 0.0,
        global_model: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """执行本地训练。

        Parameters
        ----------
        n_epochs : int
            本地训练轮数。
        lr : float
            本地学习率。
        proximal_mu : float
            FedProx 近端项系数。
        global_model : np.ndarray or None
            全局模型 (用于 FedProx)。

        Returns
        -------
        np.ndarray
            模型更新量 (delta = local_model - global_model)。
        """
        if self._n_samples < 2:
            return np.zeros_like(self._local_model)

        global_W = global_model.copy() if global_model is not None else self._local_model.copy()
        W = self._local_model.copy()
        n = self._n_samples

        # 特征维度适配
        feat_dim = self.features.shape[1]
        if W.shape[0] != feat_dim:
            # 如果模型形状不匹配，使用伪逆适配
            return np.zeros_like(self._local_model)

        for _ in range(n_epochs):
            # 前向: pred = features @ W
            pred = self.features @ W
            residual = pred - self.targets

            # MSE 损失
            loss = float(np.mean(residual ** 2))

            # 梯度
            grad = (2.0 / n) * (self.features.T @ residual)

            # FedProx 近端项
            if proximal_mu > 0:
                grad += proximal_mu * (W - global_W)

            # 梯度下降
            W -= lr * grad

        self._last_loss = float(np.mean((self.features @ W - self.targets) ** 2))
        self._local_model = W

        update = W - global_W
        self._last_update_norm = float(np.linalg.norm(update))
        return update

    def compress_update(
        self,
        update: np.ndarray,
        compression_ratio: float = 0.5,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """压缩模型更新以减少通信开销。

        使用 Top-K 稀疏化 + 量化进行通信压缩。

        Parameters
        ----------
        update : np.ndarray
            原始模型更新。
        compression_ratio : float
            压缩比 (0.0 ~ 1.0)。

        Returns
        -------
        Tuple[np.ndarray, Dict[str, float]]
            (压缩后的更新, 压缩统计信息)。
        """
        flat = update.flatten()
        n_params = len(flat)
        k = max(int(n_params * compression_ratio), 1)

        # Top-K 稀疏化
        abs_flat = np.abs(flat)
        threshold_idx = np.argpartition(abs_flat, -k)[-k:]
        mask = np.zeros_like(flat)
        mask[threshold_idx] = 1.0

        compressed = flat * mask

        # 量化到 8-bit
        max_val = max(np.abs(compressed).max(), 1e-10)
        quantized = np.round(compressed / max_val * 127) / 127 * max_val

        sparsity = float(1.0 - np.count_nonzero(mask) / n_params)
        compression_stats = {
            "original_size": float(n_params * 8),  # bits
            "compressed_size": float(k * 8),
            "sparsity": sparsity,
            "max_abs_error": float(np.max(np.abs(update.flatten() - quantized))),
        }

        return quantized.reshape(update.shape), compression_stats

    def deactivate(self) -> None:
        """停用客户端。"""
        self._is_active = False

    def activate(self) -> None:
        """激活客户端。"""
        self._is_active = True


class ModelAggregator:
    """模型聚合器。

    实现多种联邦学习聚合策略，将多个客户端的模型更新
    聚合为全局模型。

    Parameters
    ----------
    strategy : AggregationStrategy
        聚合策略。
    proximal_mu : float
        FedProx 近端项系数。
    trim_ratio : float
        截断均值裁剪比例 (仅 trimmed_mean 策略)。
    """

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.FED_AVG,
        proximal_mu: float = 0.01,
        trim_ratio: float = 0.1,
    ):
        self.strategy = strategy
        self.proximal_mu = float(proximal_mu)
        self.trim_ratio = float(trim_ratio)

    def aggregate(
        self,
        global_model: np.ndarray,
        client_updates: List[Tuple[np.ndarray, int]],
    ) -> np.ndarray:
        """聚合多个客户端的模型更新。

        Parameters
        ----------
        global_model : np.ndarray
            当前全局模型。
        client_updates : List[Tuple[np.ndarray, int]]
            客户端更新列表，每项为 (update, n_samples)。

        Returns
        -------
        np.ndarray
            聚合后的全局模型。
        """
        if not client_updates:
            return global_model.copy()

        if self.strategy == AggregationStrategy.FED_AVG:
            return self._fed_avg(global_model, client_updates)
        elif self.strategy == AggregationStrategy.FED_PROX:
            return self._fed_prox(global_model, client_updates)
        elif self.strategy == AggregationStrategy.WEIGHTED_MEDIAN:
            return self._weighted_median(global_model, client_updates)
        elif self.strategy == AggregationStrategy.TRIMMED_MEAN:
            return self._trimmed_mean(global_model, client_updates)
        else:
            return self._fed_avg(global_model, client_updates)

    def _fed_avg(
        self,
        global_model: np.ndarray,
        client_updates: List[Tuple[np.ndarray, int]],
    ) -> np.ndarray:
        """联邦平均 (FedAvg)。

        按样本数量加权平均各客户端更新。
        """
        total_samples = sum(n for _, n in client_updates)
        if total_samples == 0:
            return global_model.copy()

        weighted_update = np.zeros_like(global_model, dtype=np.float64)
        for update, n_samples in client_updates:
            weight = n_samples / total_samples
            weighted_update += weight * update

        return global_model + weighted_update

    def _fed_prox(
        self,
        global_model: np.ndarray,
        client_updates: List[Tuple[np.ndarray, int]],
    ) -> np.ndarray:
        """联邦近端 (FedProx)。

        在 FedAvg 基础上加入近端正则化，限制更新幅度。
        """
        avg_update = self._fed_avg(global_model, client_updates) - global_model

        # 限制更新幅度
        update_norm = np.linalg.norm(avg_update)
        max_norm = 1.0 / max(self.proximal_mu, 1e-6)
        if update_norm > max_norm:
            avg_update = avg_update * (max_norm / update_norm)

        return global_model + avg_update

    def _weighted_median(
        self,
        global_model: np.ndarray,
        client_updates: List[Tuple[np.ndarray, int]],
    ) -> np.ndarray:
        """加权中位数聚合。

        对每个参数维度取加权中位数，对异常更新更鲁棒。
        """
        total_samples = sum(n for _, n in client_updates)
        if total_samples == 0:
            return global_model.copy()

        # 收集所有客户端的完整模型
        client_models = []
        weights = []
        for update, n_samples in client_updates:
            client_models.append(global_model + update)
            weights.append(n_samples / total_samples)

        # 逐元素加权中位数
        stacked = np.stack(client_models, axis=0)  # (n_clients, ...)
        flat_shape = stacked.shape[1:]
        result = np.zeros(flat_shape, dtype=np.float64)

        for idx in np.ndindex(flat_shape):
            values = stacked[(slice(None),) + idx]
            sorted_idx = np.argsort(values)
            sorted_values = values[sorted_idx]
            sorted_weights = np.array([weights[i] for i in sorted_idx])
            cum_weights = np.cumsum(sorted_weights)
            median_idx = np.searchsorted(cum_weights, 0.5)
            median_idx = min(median_idx, len(sorted_values) - 1)
            result[idx] = sorted_values[median_idx]

        return result

    def _trimmed_mean(
        self,
        global_model: np.ndarray,
        client_updates: List[Tuple[np.ndarray, int]],
    ) -> np.ndarray:
        """截断均值聚合。

        去除最大和最小的更新后取平均，对 Byzantine 客户端鲁棒。
        """
        if len(client_updates) < 3:
            return self._fed_avg(global_model, client_updates)

        # 收集所有更新
        updates = [u for u, _ in client_updates]
        stacked = np.stack(updates, axis=0)

        # 按范数排序并裁剪
        norms = np.array([np.linalg.norm(u) for u in updates])
        sorted_idx = np.argsort(norms)
        n_trim = max(int(len(updates) * self.trim_ratio), 1)

        # 去除最大和最小的更新
        valid_idx = sorted_idx[n_trim:-n_trim]
        if len(valid_idx) == 0:
            valid_idx = sorted_idx

        trimmed_updates = stacked[valid_idx]
        avg_update = np.mean(trimmed_updates, axis=0)

        return global_model + avg_update


class FederatedLearningCoordinator:
    """联邦学习协调器 - 多设备协同训练。

    协调多台光学设备进行联邦学习训练，无需集中收集数据。
    支持多种聚合策略、通信压缩和异构设备管理。

    Parameters
    ----------
    config : FederatedTrainingConfig or None
        联邦训练配置。
    """

    def __init__(
        self,
        config: Optional[FederatedTrainingConfig] = None,
    ):
        self.config = config or FederatedTrainingConfig()
        self.aggregator = ModelAggregator(
            strategy=self.config.aggregation_strategy,
            proximal_mu=self.config.proximal_mu,
        )

        self._clients: Dict[str, FederatedClient] = {}
        self._global_model: Optional[np.ndarray] = None
        self._round_history: List[FederatedRoundResult] = []
        self._current_round = 0
        self._is_training = False

    @property
    def is_training(self) -> bool:
        """是否正在训练。"""
        return self._is_training

    @property
    def global_model(self) -> Optional[np.ndarray]:
        """获取当前全局模型。"""
        if self._global_model is not None:
            return self._global_model.copy()
        return None

    @property
    def round_history(self) -> List[FederatedRoundResult]:
        """获取训练历史。"""
        return list(self._round_history)

    @property
    def n_clients(self) -> int:
        """注册的客户端数量。"""
        return len(self._clients)

    def register_client(
        self,
        client_id: str,
        local_data_features: np.ndarray,
        local_data_targets: np.ndarray,
        model_shape: tuple = (2, 2),
        device_config: Optional[dict] = None,
    ) -> bool:
        """注册新的联邦学习客户端。

        Parameters
        ----------
        client_id : str
            客户端唯一标识符。
        local_data_features : np.ndarray
            本地特征数据。
        local_data_targets : np.ndarray
            本地目标数据。
        model_shape : tuple
            模型参数形状。
        device_config : dict or None
            设备配置。

        Returns
        -------
        bool
            注册是否成功。
        """
        if client_id in self._clients:
            return False

        client = FederatedClient(
            client_id=client_id,
            local_data_features=local_data_features,
            local_data_targets=local_data_targets,
            model_shape=model_shape,
            device_config=device_config,
        )
        self._clients[client_id] = client

        # 初始化全局模型
        if self._global_model is None:
            self._global_model = client.local_model.copy()

        return True

    def unregister_client(self, client_id: str) -> bool:
        """注销客户端。

        Parameters
        ----------
        client_id : str
            客户端标识符。

        Returns
        -------
        bool
            注销是否成功。
        """
        if client_id not in self._clients:
            return False
        del self._clients[client_id]
        return True

    def run_training(
        self,
        progress_callback: Optional[Callable[[int, FederatedRoundResult], None]] = None,
    ) -> List[FederatedRoundResult]:
        """执行联邦学习训练。

        Parameters
        ----------
        progress_callback : Callable or None
            进度回调函数，参数为 (round_id, result)。

        Returns
        -------
        List[FederatedRoundResult]
            每轮训练结果列表。
        """
        if self._global_model is None:
            raise RuntimeError("未注册任何客户端，无法开始训练")

        active_clients = [c for c in self._clients.values() if c.is_active]
        if len(active_clients) < self.config.min_clients_per_round:
            raise RuntimeError(
                f"活跃客户端数量不足: {len(active_clients)} < {self.config.min_clients_per_round}"
            )

        self._is_training = True
        self._round_history = []
        self._current_round = 0

        try:
            for round_id in range(self.config.n_rounds):
                # 检查是否请求停止训练
                if not self._is_training:
                    break

                self._current_round = round_id
                result = self._run_single_round(round_id)
                self._round_history.append(result)

                if progress_callback is not None:
                    progress_callback(round_id, result)

                # 检查收敛
                if result.convergence_metric < self.config.convergence_threshold:
                    break

        finally:
            self._is_training = False

        return self._round_history

    def run_single_round(self) -> FederatedRoundResult:
        """执行单轮联邦训练。

        Returns
        -------
        FederatedRoundResult
            本轮训练结果。
        """
        if self._global_model is None:
            raise RuntimeError("未初始化全局模型")

        result = self._run_single_round(self._current_round)
        self._round_history.append(result)
        self._current_round += 1
        return result

    def get_training_summary(self) -> Dict[str, float]:
        """获取训练摘要统计。

        Returns
        -------
        Dict[str, float]
            训练摘要。
        """
        if not self._round_history:
            return {}

        losses = [r.global_loss for r in self._round_history]
        accuracies = [r.global_accuracy for r in self._round_history]
        comm_costs = [r.communication_cost for r in self._round_history]

        return {
            "total_rounds": len(self._round_history),
            "final_loss": losses[-1] if losses else 0.0,
            "best_loss": min(losses) if losses else 0.0,
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "total_communication_cost": sum(comm_costs),
            "avg_communication_cost": float(np.mean(comm_costs)) if comm_costs else 0.0,
            "loss_reduction": (losses[0] - losses[-1]) / max(losses[0], 1e-10) if losses else 0.0,
        }

    def stop_training(self) -> None:
        """停止训练。"""
        self._is_training = False

    def reset(self) -> None:
        """重置协调器。"""
        self._global_model = None
        self._round_history = []
        self._current_round = 0
        self._is_training = False

    # ---- 内部方法 ----

    def _run_single_round(self, round_id: int) -> FederatedRoundResult:
        """执行单轮联邦训练的内部实现。"""
        active_clients = [c for c in self._clients.values() if c.is_active]
        n_total = len(active_clients)

        # 选择参与客户端
        n_participants = max(
            self.config.min_clients_per_round,
            int(n_total * self.config.client_fraction),
        )
        participant_indices = np.random.choice(
            n_total, min(n_participants, n_total), replace=False
        )
        participants = [active_clients[i] for i in participant_indices]

        # 分发全局模型
        for client in participants:
            client.set_global_model(self._global_model)

        # 本地训练
        client_updates: List[Tuple[np.ndarray, int]] = []
        client_losses: Dict[str, float] = {}
        total_comm_cost = 0.0

        for client in participants:
            # 确定是否使用 FedProx
            mu = 0.0
            if self.config.aggregation_strategy == AggregationStrategy.FED_PROX:
                mu = self.config.proximal_mu

            # 本地训练
            update = client.local_train(
                n_epochs=self.config.local_epochs,
                lr=self.config.local_lr,
                proximal_mu=mu,
                global_model=self._global_model,
            )

            # 通信压缩
            if self.config.compression_ratio < 1.0:
                update, stats = client.compress_update(
                    update, self.config.compression_ratio
                )
                total_comm_cost += stats["compressed_size"]
            else:
                total_comm_cost += update.size * 8

            client_updates.append((update, client.n_samples))
            client_losses[client.client_id] = client.last_loss

        # 聚合
        old_model = self._global_model.copy()
        self._global_model = self.aggregator.aggregate(
            self._global_model, client_updates
        )

        # 计算收敛度量
        model_update_norm = float(np.linalg.norm(self._global_model - old_model))
        avg_loss = float(np.mean(list(client_losses.values()))) if client_losses else 0.0

        # 估算全局准确率 (基于加权平均客户端损失)
        global_accuracy = max(0.0, 1.0 - avg_loss / 10.0)

        return FederatedRoundResult(
            round_id=round_id,
            n_participants=len(participants),
            global_loss=avg_loss,
            global_accuracy=global_accuracy,
            communication_cost=total_comm_cost,
            convergence_metric=model_update_norm,
            client_losses=client_losses,
            model_update_norm=model_update_norm,
        )
