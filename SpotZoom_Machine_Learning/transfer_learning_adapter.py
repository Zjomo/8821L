"""
迁移学习适配器 (TransferLearningAdapter)

基于经典统计学习与域适应理论的策略迁移算法。

算法原理:
- Domain Adaptation Theory (Ben-David 2010) — 领域适应误差界
- Linear Feature Mapping — 线性特征空间变换
- RBF Kernel Mapping — 非线性特征空间映射
- Few-shot Learning — 小样本快速适配
- Policy Distillation — 策略知识蒸馏 (Hinton 2015)

功能:
- 将预训练的对准策略适配到新光学系统
- 支持线性/非线性特征空间映射
- 小样本快速微调 (few-shot adaptation)
- 领域偏移度量 (domain shift measurement)
- 策略蒸馏 (policy distillation)

依赖: numpy (无线性代数库之外的外部依赖)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np


@dataclass
class AdaptationResult:
    """适配结果数据类。

    Attributes
    ----------
    success : bool
        适配是否成功。
    mapping_matrix : np.ndarray
        学习到的特征映射矩阵。
    source_loss : float
        源域上的损失值。
    target_loss : float
        目标域上的损失值。
    domain_shift_score : float
        领域偏移分数 (越小越相似)。
    adaptation_iterations : int
        实际使用的适配迭代次数。
    convergence_history : List[float]
        收敛历史记录。
    """
    success: bool = False
    mapping_matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    source_loss: float = 0.0
    target_loss: float = 0.0
    domain_shift_score: float = 0.0
    adaptation_iterations: int = 0
    convergence_history: List[float] = field(default_factory=list)


@dataclass
class DomainShiftMetric:
    """领域偏移度量结果。

    Attributes
    ----------
    mmd_distance : float
        最大均值差异 (Maximum Mean Discrepancy)。
    wasserstein_distance : float
        Wasserstein 距离 (Earth Mover's Distance)。
    kl_divergence : float
        KL 散度。
    correlation_distance : float
        相关距离 (1 - 相关系数)。
    feature_covariance_diff : float
        特征协方差差异 (Frobenius 范数)。
    is_similar : bool
        两个领域是否足够相似。
    """
    mmd_distance: float = 0.0
    wasserstein_distance: float = 0.0
    kl_divergence: float = 0.0
    correlation_distance: float = 0.0
    feature_covariance_diff: float = 0.0
    is_similar: bool = True


class TransferLearningAdapter:
    """迁移学习适配器 - 将预训练对准策略适配到新光学系统。

    通过学习源域 (预训练光学系统) 到目标域 (新光学系统) 的特征映射，
    实现对准策略的快速迁移。支持线性映射、RBF 核非线性映射、
    小样本微调和策略蒸馏。

    Parameters
    ----------
    feature_dim : int
        特征维度 (默认 2，对应 x/y 像素坐标)。
    mapping_type : str
        映射类型，可选 'linear' 或 'rbf'。
    rbf_gamma : float
        RBF 核带宽参数 (仅非线性映射时有效)。
    adaptation_lr : float
        适配学习率。
    max_iterations : int
        最大适配迭代次数。
    convergence_threshold : float
        收敛阈值。
    regularization_lambda : float
        正则化系数 (防止过拟合)。
    distillation_temperature : float
        蒸馏温度参数。
    """

    def __init__(
        self,
        feature_dim: int = 2,
        mapping_type: str = "linear",
        rbf_gamma: float = 1.0,
        adaptation_lr: float = 0.01,
        max_iterations: int = 500,
        convergence_threshold: float = 1e-6,
        regularization_lambda: float = 0.01,
        distillation_temperature: float = 2.0,
    ):
        self.feature_dim = int(feature_dim)
        self.mapping_type = str(mapping_type).lower()
        self.rbf_gamma = float(rbf_gamma)
        self.adaptation_lr = float(adaptation_lr)
        self.max_iterations = int(max_iterations)
        self.convergence_threshold = float(convergence_threshold)
        self.regularization_lambda = float(regularization_lambda)
        self.distillation_temperature = float(distillation_temperature)

        if self.mapping_type not in ("linear", "rbf"):
            raise ValueError(f"不支持的映射类型: {self.mapping_type}，可选 'linear' 或 'rbf'")

        # 线性映射矩阵 W: target ≈ W @ source
        self._W = np.eye(self.feature_dim, dtype=np.float64)

        # RBF 核中心 (非线性映射)
        self._kernel_centers: Optional[np.ndarray] = None
        self._kernel_weights: Optional[np.ndarray] = None

        # 适配状态
        self._is_adapted = False
        self._adaptation_history: List[float] = []

    @property
    def is_adapted(self) -> bool:
        """适配器是否已完成适配。"""
        return self._is_adapted

    @property
    def mapping_matrix(self) -> np.ndarray:
        """获取当前映射矩阵。"""
        return self._W.copy()

    def compute_domain_shift(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
    ) -> DomainShiftMetric:
        """计算源域与目标域之间的领域偏移度量。

        Parameters
        ----------
        source_features : np.ndarray
            源域特征矩阵，形状 (n_samples, feature_dim)。
        target_features : np.ndarray
            目标域特征矩阵，形状 (m_samples, feature_dim)。

        Returns
        -------
        DomainShiftMetric
            领域偏移度量结果。
        """
        src = np.asarray(source_features, dtype=np.float64)
        tgt = np.asarray(target_features, dtype=np.float64)

        if src.ndim == 1:
            src = src.reshape(-1, 1)
        if tgt.ndim == 1:
            tgt = tgt.reshape(-1, 1)

        # 1. 最大均值差异 (MMD) — 使用 RBF 核
        mmd = self._compute_mmd(src, tgt)

        # 2. Wasserstein 距离 (1D 排序近似)
        w_dist = self._compute_wasserstein(src, tgt)

        # 3. KL 散度 (基于高斯假设)
        kl = self._compute_kl_divergence(src, tgt)

        # 4. 相关距离
        corr_dist = self._compute_correlation_distance(src, tgt)

        # 5. 协方差差异
        cov_diff = self._compute_covariance_diff(src, tgt)

        # 综合判断是否相似
        is_similar = (
            mmd < 0.5
            and w_dist < 1.0
            and corr_dist < 0.3
        )

        return DomainShiftMetric(
            mmd_distance=float(mmd),
            wasserstein_distance=float(w_dist),
            kl_divergence=float(kl),
            correlation_distance=float(corr_dist),
            feature_covariance_diff=float(cov_diff),
            is_similar=is_similar,
        )

    def adapt(
        self,
        source_features: np.ndarray,
        source_actions: np.ndarray,
        target_features: np.ndarray,
        target_actions: Optional[np.ndarray] = None,
    ) -> AdaptationResult:
        """执行迁移学习适配。

        通过最小化源域策略在目标域上的映射误差来学习特征映射。
        如果提供了目标域动作标签，则进行监督微调。

        Parameters
        ----------
        source_features : np.ndarray
            源域特征，形状 (n_samples, feature_dim)。
        source_actions : np.ndarray
            源域动作/控制信号，形状 (n_samples, action_dim)。
        target_features : np.ndarray
            目标域特征，形状 (m_samples, feature_dim)。
        target_actions : np.ndarray or None
            目标域动作标签 (可选，用于监督微调)。

        Returns
        -------
        AdaptationResult
            适配结果。
        """
        src_feat = np.asarray(source_features, dtype=np.float64)
        src_act = np.asarray(source_actions, dtype=np.float64)
        tgt_feat = np.asarray(target_features, dtype=np.float64)

        if src_feat.ndim == 1:
            src_feat = src_feat.reshape(-1, 1)
        if src_act.ndim == 1:
            src_act = src_act.reshape(-1, 1)
        if tgt_feat.ndim == 1:
            tgt_feat = tgt_feat.reshape(-1, 1)

        # 计算领域偏移
        shift_metric = self.compute_domain_shift(src_feat, tgt_feat)

        if self.mapping_type == "linear":
            result = self._adapt_linear(src_feat, src_act, tgt_feat, target_actions)
        else:
            result = self._adapt_rbf(src_feat, src_act, tgt_feat, target_actions)

        result.domain_shift_score = (
            shift_metric.mmd_distance
            + shift_metric.correlation_distance
            + shift_metric.feature_covariance_diff
        ) / 3.0

        self._is_adapted = result.success
        return result

    def transform(self, features: np.ndarray) -> np.ndarray:
        """将源域特征映射到目标域特征空间。

        Parameters
        ----------
        features : np.ndarray
            待映射的特征，形状 (n_samples, feature_dim) 或 (feature_dim,)。

        Returns
        -------
        np.ndarray
            映射后的特征。
        """
        feat = np.asarray(features, dtype=np.float64)
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)

        if self.mapping_type == "linear":
            return (self._W @ feat.T).T
        else:
            return self._transform_rbf(feat)

    def distill_policy(
        self,
        teacher_actions: np.ndarray,
        student_features: np.ndarray,
        n_iterations: int = 200,
    ) -> Tuple[np.ndarray, float]:
        """策略蒸馏 - 将教师策略知识蒸馏到学生策略。

        通过软化教师动作分布并让学生模仿，实现知识迁移。

        Parameters
        ----------
        teacher_actions : np.ndarray
            教师策略的动作输出，形状 (n_samples, action_dim)。
        student_features : np.ndarray
            学生策略的输入特征，形状 (n_samples, feature_dim)。
        n_iterations : int
            蒸馏迭代次数。

        Returns
        -------
        Tuple[np.ndarray, float]
            (蒸馏后的学生策略矩阵, 蒸馏损失)。
        """
        teacher = np.asarray(teacher_actions, dtype=np.float64)
        features = np.asarray(student_features, dtype=np.float64)

        if teacher.ndim == 1:
            teacher = teacher.reshape(-1, 1)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        n_samples, action_dim = teacher.shape
        feat_dim = features.shape[1]

        # 软化教师输出
        T = self.distillation_temperature
        soft_teacher = teacher / T

        # 初始化学生策略矩阵
        student_W = np.random.randn(feat_dim, action_dim) * 0.01

        lr = self.adaptation_lr
        best_loss = float("inf")
        best_W = student_W.copy()

        for _ in range(n_iterations):
            # 前向: student_output = features @ W
            student_output = features @ student_W

            # 软化学生输出
            soft_student = student_output / T

            # KL 散度损失
            loss = float(np.mean((soft_student - soft_teacher) ** 2))

            # 正则化
            reg = self.regularization_lambda * np.sum(student_W ** 2)
            total_loss = loss + reg

            if total_loss < best_loss:
                best_loss = total_loss
                best_W = student_W.copy()

            # 梯度下降
            grad = (2.0 / (n_samples * T ** 2)) * (features.T @ (soft_student - soft_teacher))
            grad += 2.0 * self.regularization_lambda * student_W
            student_W -= lr * grad

        return best_W, best_loss

    def few_shot_adapt(
        self,
        source_W: np.ndarray,
        few_shot_features: np.ndarray,
        few_shot_actions: np.ndarray,
        n_shots: int = 5,
    ) -> AdaptationResult:
        """小样本快速适配。

        使用极少量的目标域样本对预训练映射进行快速微调。

        Parameters
        ----------
        source_W : np.ndarray
            源域预训练映射矩阵。
        few_shot_features : np.ndarray
            少量目标域特征样本。
        few_shot_actions : np.ndarray
            对应的动作标签。
        n_shots : int
            使用的样本数量。

        Returns
        -------
        AdaptationResult
            适配结果。
        """
        self._W = np.asarray(source_W, dtype=np.float64).copy()

        feat = np.asarray(few_shot_features, dtype=np.float64)
        act = np.asarray(few_shot_actions, dtype=np.float64)

        if feat.ndim == 1:
            feat = feat.reshape(-1, 1)
        if act.ndim == 1:
            act = act.reshape(-1, 1)

        # 只使用前 n_shots 个样本
        feat = feat[:n_shots]
        act = act[:n_shots]

        n = feat.shape[0]
        if n < 2:
            return AdaptationResult(
                success=False,
                mapping_matrix=self._W.copy(),
                adaptation_iterations=0,
            )

        # 使用岭回归进行快速微调
        lambda_reg = self.regularization_lambda * 10  # 更强的正则化防止过拟合
        A = feat.T @ feat + lambda_reg * np.eye(feat.shape[1])
        b = feat.T @ act
        try:
            self._W = np.linalg.solve(A, b).T
            # 计算损失
            pred = feat @ self._W.T
            target_loss = float(np.mean((pred - act) ** 2))
            self._is_adapted = True

            return AdaptationResult(
                success=True,
                mapping_matrix=self._W.copy(),
                target_loss=target_loss,
                adaptation_iterations=1,
                convergence_history=[target_loss],
            )
        except np.linalg.LinAlgError:
            return AdaptationResult(
                success=False,
                mapping_matrix=self._W.copy(),
                adaptation_iterations=0,
            )

    def reset(self) -> None:
        """重置适配器到初始状态。"""
        self._W = np.eye(self.feature_dim, dtype=np.float64)
        self._kernel_centers = None
        self._kernel_weights = None
        self._is_adapted = False
        self._adaptation_history = []

    # ---- 内部方法 ----

    def _adapt_linear(
        self,
        source_features: np.ndarray,
        source_actions: np.ndarray,
        target_features: np.ndarray,
        target_actions: Optional[np.ndarray],
    ) -> AdaptationResult:
        """线性映射适配。"""
        n_src = source_features.shape[0]
        n_tgt = target_features.shape[0]

        # 学习映射: target_feat ≈ W @ source_feat
        # 使用最小二乘 + 正则化
        combined_src = source_features
        combined_tgt = target_features

        # 如果有目标域标签，加入监督信号
        if target_actions is not None:
            tgt_act = np.asarray(target_actions, dtype=np.float64)
            if tgt_act.ndim == 1:
                tgt_act = tgt_act.reshape(-1, 1)

            # 同时学习特征映射和动作映射
            # 策略: target_action ≈ source_action @ W_action
            # 特征: target_feat ≈ source_feat @ W_feat
            # 联合优化
            action_dim = tgt_act.shape[1]
            feat_dim = source_features.shape[1]

            # 特征映射
            lambda_reg = self.regularization_lambda
            A_feat = combined_src.T @ combined_src + lambda_reg * np.eye(feat_dim)
            b_feat = combined_src.T @ combined_tgt
            try:
                W_feat = np.linalg.solve(A_feat, b_feat)
            except np.linalg.LinAlgError:
                W_feat = np.eye(feat_dim)

            # 动作映射
            src_act = source_actions[:min(n_src, n_tgt)]
            tgt_act_trimmed = tgt_act[:min(n_src, n_tgt)]
            A_act = src_act.T @ src_act + lambda_reg * np.eye(src_act.shape[1])
            b_act = src_act.T @ tgt_act_trimmed
            try:
                W_act = np.linalg.solve(A_act, b_act)
            except np.linalg.LinAlgError:
                W_act = np.eye(src_act.shape[1])

            self._W = W_feat

            # 计算损失
            pred_tgt = combined_src @ W_feat
            target_loss = float(np.mean((pred_tgt - combined_tgt) ** 2))
            source_loss = float(np.mean((source_actions[:min(n_src, n_tgt)] @ W_act - tgt_act_trimmed) ** 2))
        else:
            # 纯无监督特征映射
            lambda_reg = self.regularization_lambda
            feat_dim = source_features.shape[1]
            A = combined_src.T @ combined_src + lambda_reg * np.eye(feat_dim)
            b = combined_src.T @ combined_tgt
            try:
                self._W = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                self._W = np.eye(feat_dim)

            pred_tgt = combined_src @ self._W
            target_loss = float(np.mean((pred_tgt - combined_tgt) ** 2))
            source_loss = 0.0

        # 迭代优化 (梯度下降微调)
        history = [target_loss]
        lr = self.adaptation_lr
        W = self._W.copy()

        for i in range(self.max_iterations):
            pred = combined_src @ W
            residual = pred - combined_tgt
            loss = float(np.mean(residual ** 2))

            if abs(history[-1] - loss) < self.convergence_threshold:
                break

            history.append(loss)
            grad = (2.0 / n_src) * (combined_src.T @ residual)
            grad += 2.0 * self.regularization_lambda * W
            W -= lr * grad

        self._W = W
        final_loss = history[-1] if history else target_loss

        return AdaptationResult(
            success=final_loss < 10.0,
            mapping_matrix=self._W.copy(),
            source_loss=source_loss,
            target_loss=final_loss,
            adaptation_iterations=len(history),
            convergence_history=history,
        )

    def _adapt_rbf(
        self,
        source_features: np.ndarray,
        source_actions: np.ndarray,
        target_features: np.ndarray,
        target_actions: Optional[np.ndarray],
    ) -> AdaptationResult:
        """RBF 核非线性映射适配。"""
        # 使用源域特征作为核中心
        n_centers = min(source_features.shape[0], 100)
        indices = np.random.choice(source_features.shape[0], n_centers, replace=False)
        self._kernel_centers = source_features[indices].copy()

        # 计算核矩阵
        K_src = self._rbf_kernel(source_features, self._kernel_centers)
        K_tgt = self._rbf_kernel(target_features, self._kernel_centers)

        # 学习核权重: target ≈ K_src @ weights
        lambda_reg = self.regularization_lambda
        A = K_src.T @ K_src + lambda_reg * np.eye(K_src.shape[1])
        b = K_src.T @ target_features
        try:
            self._kernel_weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self._kernel_weights = np.zeros((K_src.shape[1], target_features.shape[1]))

        # 计算损失
        pred = K_src @ self._kernel_weights
        target_loss = float(np.mean((pred - target_features) ** 2))
        source_loss = 0.0

        # 迭代优化
        history = [target_loss]
        lr = self.adaptation_lr
        weights = self._kernel_weights.copy()
        n = K_src.shape[0]

        for i in range(self.max_iterations):
            pred = K_src @ weights
            residual = pred - target_features
            loss = float(np.mean(residual ** 2))

            if abs(history[-1] - loss) < self.convergence_threshold:
                break

            history.append(loss)
            grad = (2.0 / n) * (K_src.T @ residual)
            grad += 2.0 * self.regularization_lambda * weights
            weights -= lr * grad

        self._kernel_weights = weights
        final_loss = history[-1] if history else target_loss

        # 同时更新线性映射作为近似
        self._W = np.eye(self.feature_dim)

        return AdaptationResult(
            success=final_loss < 10.0,
            mapping_matrix=self._W.copy(),
            source_loss=source_loss,
            target_loss=final_loss,
            adaptation_iterations=len(history),
            convergence_history=history,
        )

    def _transform_rbf(self, features: np.ndarray) -> np.ndarray:
        """使用 RBF 核进行非线性变换。"""
        if self._kernel_centers is None or self._kernel_weights is None:
            return features.copy()

        K = self._rbf_kernel(features, self._kernel_centers)
        return K @ self._kernel_weights

    @staticmethod
    def _rbf_kernel(X: np.ndarray, centers: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """计算 RBF 核矩阵。"""
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x @ c^T
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        C_sq = np.sum(centers ** 2, axis=1, keepdims=True).T
        dist_sq = X_sq + C_sq - 2.0 * X @ centers.T
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-gamma * dist_sq)

    @staticmethod
    def _compute_mmd(source: np.ndarray, target: np.ndarray, gamma: float = 1.0) -> float:
        """计算最大均值差异 (MMD)。"""
        def _kernel_mean(X: np.ndarray, Y: np.ndarray) -> float:
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)
            Y_sq = np.sum(Y ** 2, axis=1, keepdims=True).T
            dist_sq = X_sq + Y_sq - 2.0 * X @ Y.T
            dist_sq = np.maximum(dist_sq, 0.0)
            K = np.exp(-gamma * dist_sq)
            return float(np.mean(K))

        k_ss = _kernel_mean(source, source)
        k_tt = _kernel_mean(target, target)
        k_st = _kernel_mean(source, target)
        mmd = k_ss + k_tt - 2.0 * k_st
        return float(max(mmd, 0.0))

    @staticmethod
    def _compute_wasserstein(source: np.ndarray, target: np.ndarray) -> float:
        """计算 Wasserstein 距离 (基于排序的 1D 近似)。"""
        total_dist = 0.0
        for d in range(source.shape[1]):
            s_sorted = np.sort(source[:, d])
            t_sorted = np.sort(target[:, d])
            # 对齐到相同长度
            min_len = min(len(s_sorted), len(t_sorted))
            s_sorted = s_sorted[:min_len]
            t_sorted = t_sorted[:min_len]
            total_dist += float(np.mean(np.abs(s_sorted - t_sorted)))
        return total_dist / max(source.shape[1], 1)

    @staticmethod
    def _compute_kl_divergence(source: np.ndarray, target: np.ndarray) -> float:
        """计算 KL 散度 (基于高斯假设)。"""
        mu_s = np.mean(source, axis=0)
        mu_t = np.mean(target, axis=0)
        cov_s = np.cov(source.T) + 1e-6 * np.eye(source.shape[1])
        cov_t = np.cov(target.T) + 1e-6 * np.eye(target.shape[1])

        d = source.shape[1]
        cov_t_inv = np.linalg.inv(cov_t)
        trace_term = np.trace(cov_t_inv @ cov_s)
        diff = mu_t - mu_s
        mean_term = diff @ cov_t_inv @ diff
        log_det_term = np.log(np.linalg.det(cov_t) / max(np.linalg.det(cov_s), 1e-10))

        kl = 0.5 * (trace_term + mean_term - d + log_det_term)
        return float(max(kl, 0.0))

    @staticmethod
    def _compute_correlation_distance(source: np.ndarray, target: np.ndarray) -> float:
        """计算相关距离 (1 - Pearson 相关系数)。"""
        total_dist = 0.0
        for d in range(source.shape[1]):
            s = source[:, d]
            t = target[:, d]
            min_len = min(len(s), len(t))
            s = s[:min_len]
            t = t[:min_len]
            s_centered = s - np.mean(s)
            t_centered = t - np.mean(t)
            s_std = np.std(s)
            t_std = np.std(t)
            if s_std < 1e-10 or t_std < 1e-10:
                total_dist += 1.0
                continue
            corr = float(np.mean(s_centered * t_centered) / (s_std * t_std))
            total_dist += 1.0 - abs(corr)
        return total_dist / max(source.shape[1], 1)

    @staticmethod
    def _compute_covariance_diff(source: np.ndarray, target: np.ndarray) -> float:
        """计算协方差矩阵差异 (Frobenius 范数)。"""
        cov_s = np.cov(source.T)
        cov_t = np.cov(target.T)
        if cov_s.ndim == 0:
            cov_s = np.array([[cov_s]])
        if cov_t.ndim == 0:
            cov_t = np.array([[cov_t]])
        return float(np.linalg.norm(cov_s - cov_t, 'fro'))
