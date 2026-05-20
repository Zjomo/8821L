"""
XAIDiagnostic - 可解释性诊断模块

灵感来源: SHAP, LIME, Integrated Gradients
功能特点:
- 模型决策可视化
- 特征重要性分析
- 异常根因定位
- 决策过程追踪
- 纯numpy/cv2实现，零外部ML依赖

技术路线:
- 梯度归因方法
- 注意力机制可视化
- 对比解释生成
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """解释结果。"""
    feature_importance: Dict[str, float]
    attribution_map: np.ndarray
    decision_path: List[str]
    confidence_breakdown: Dict[str, float]
    processing_time_ms: float


@dataclass
class AnomalyExplanation:
    """异常解释。"""
    anomaly_type: str
    root_causes: List[Tuple[str, float]]  # (原因, 置信度)
    affected_regions: List[Tuple[int, int, int, int]]  # (x, y, w, h)
    recommended_actions: List[str]
    severity_score: float


@dataclass
class XAIConfig:
    """XAI配置。"""
    # 归因方法参数
    num_perturbations: int = 100
    perturbation_size: float = 0.1
    
    # 可视化参数
    colormap: int = cv2.COLORMAP_JET
    alpha_blend: float = 0.5
    
    # 特征选择参数
    top_k_features: int = 10
    importance_threshold: float = 0.05
    
    # 异常检测参数
    anomaly_threshold: float = 2.0
    min_anomaly_region_size: int = 10


class XAIDiagnostic:
    """
    可解释性诊断模块。
    
    提供模型决策的可解释性分析，包括:
    1. 特征重要性: 识别对决策影响最大的特征
    2. 归因图: 可视化输入中各区域的重要性
    3. 决策路径: 追踪模型的决策过程
    4. 异常诊断: 定位异常的根因
    
    应用场景:
    - 调试模型行为
    - 理解失败案例
    - 提供用户可理解的解释
    - 合规性审计
    """
    
    def __init__(self, config: Optional[XAIConfig] = None):
        self.config = config or XAIConfig()
        self._decision_history: List[Dict] = []
        self._feature_statistics: Dict[str, Dict] = {}
        LOGGER.info("XAIDiagnostic初始化完成")
    
    def reset(self) -> None:
        """重置诊断模块。"""
        self._decision_history.clear()
        self._feature_statistics.clear()
    
    def explain_prediction(
        self,
        input_data: np.ndarray,
        prediction: Any,
        model_forward: Callable[[np.ndarray], Any]
    ) -> ExplanationResult:
        """
        解释模型预测。
        
        Args:
            input_data: 输入数据
            prediction: 模型预测结果
            model_forward: 模型前向传播函数
            
        Returns:
            ExplanationResult包含解释结果
        """
        t0 = time.perf_counter()
        
        # 计算特征重要性
        importance = self._compute_feature_importance(
            input_data, prediction, model_forward
        )
        
        # 生成归因图
        attribution = self._generate_attribution_map(
            input_data, prediction, model_forward
        )
        
        # 追踪决策路径
        decision_path = self._trace_decision_path(
            input_data, prediction
        )
        
        # 分解置信度
        confidence = self._breakdown_confidence(
            input_data, prediction, importance
        )
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        # 记录历史
        self._decision_history.append({
            'input_shape': input_data.shape,
            'prediction': prediction,
            'importance': importance,
            'timestamp': time.time()
        })
        
        LOGGER.debug("解释生成完成: 特征数=%d, 耗时=%.2fms",
                    len(importance), elapsed_ms)
        
        return ExplanationResult(
            feature_importance=importance,
            attribution_map=attribution,
            decision_path=decision_path,
            confidence_breakdown=confidence,
            processing_time_ms=elapsed_ms
        )
    
    def _compute_feature_importance(
        self,
        input_data: np.ndarray,
        prediction: Any,
        model_forward: Callable
    ) -> Dict[str, float]:
        """计算特征重要性 (基于扰动的方法)。"""
        importance = {}
        
        # 获取基准预测
        baseline_pred = model_forward(input_data)
        baseline_score = self._prediction_score(baseline_pred)
        
        # 对输入进行扰动并观察变化
        for i in range(min(self.config.num_perturbations, input_data.size)):
            # 创建扰动输入
            perturbed = input_data.copy()
            
            # 随机扰动一个区域
            if input_data.ndim >= 2:
                h, w = input_data.shape[:2]
                ph, pw = max(1, h // 10), max(1, w // 10)
                y = np.random.randint(0, max(1, h - ph))
                x = np.random.randint(0, max(1, w - pw))
                
                noise = np.random.randn(ph, pw) * self.config.perturbation_size
                if input_data.ndim == 3:
                    perturbed[y:y+ph, x:x+pw, :] += noise[:, :, None]
                else:
                    perturbed[y:y+ph, x:x+pw] += noise
            else:
                idx = np.random.randint(0, input_data.size)
                perturbed.flat[idx] += np.random.randn() * self.config.perturbation_size
            
            # 计算预测变化
            perturbed_pred = model_forward(perturbed)
            perturbed_score = self._prediction_score(perturbed_pred)
            
            # 重要性 = 预测变化幅度
            score_diff = abs(baseline_score - perturbed_score)
            
            # 记录特征重要性
            feature_name = f"region_{i}"
            importance[feature_name] = importance.get(feature_name, 0) + score_diff
        
        # 归一化
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        # 筛选重要特征
        importance = {
            k: v for k, v in importance.items() 
            if v > self.config.importance_threshold
        }
        
        # 保留top-k
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        importance = dict(sorted_imp[:self.config.top_k_features])
        
        return importance
    
    def _generate_attribution_map(
        self,
        input_data: np.ndarray,
        prediction: Any,
        model_forward: Callable
    ) -> np.ndarray:
        """生成归因图 (基于梯度的方法)。"""
        # 简化版：使用局部敏感度
        if input_data.ndim < 2:
            # 1D输入
            attribution = np.abs(np.gradient(input_data.astype(np.float64)))
        else:
            # 2D/3D输入
            gray = input_data.astype(np.float64)
            if input_data.ndim == 3:
                gray = cv2.cvtColor(input_data.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
            
            # 计算局部梯度
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            attribution = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 归一化到0-255
        if attribution.max() > 0:
            attribution = (attribution / attribution.max() * 255).astype(np.uint8)
        
        return attribution
    
    def _trace_decision_path(
        self,
        input_data: np.ndarray,
        prediction: Any
    ) -> List[str]:
        """追踪决策路径。"""
        path = []
        
        # 简化的决策路径
        path.append("输入预处理完成")
        
        # 基于输入特征添加决策节点
        if input_data.ndim >= 2:
            mean_val = np.mean(input_data)
            std_val = np.std(input_data)
            
            path.append(f"输入统计: 均值={mean_val:.3f}, 标准差={std_val:.3f}")
            
            if std_val > 50:
                path.append("检测到高对比度区域")
            
            if mean_val < 100:
                path.append("输入偏暗，增强处理")
        
        # 基于预测添加决策节点
        if isinstance(prediction, (int, float)):
            path.append(f"数值预测: {prediction:.3f}")
        elif isinstance(prediction, np.ndarray):
            path.append(f"数组预测: 形状{prediction.shape}")
        else:
            path.append(f"预测类型: {type(prediction).__name__}")
        
        path.append("决策完成")
        
        return path
    
    def _breakdown_confidence(
        self,
        input_data: np.ndarray,
        prediction: Any,
        importance: Dict[str, float]
    ) -> Dict[str, float]:
        """分解置信度组成。"""
        breakdown = {}
        
        # 数据质量分数
        if input_data.ndim >= 2:
            snr = np.mean(input_data) / (np.std(input_data) + 1e-10)
            breakdown['data_quality'] = min(1.0, snr / 10)
        else:
            breakdown['data_quality'] = 0.8
        
        # 特征清晰度分数
        if importance:
            top_importance = max(importance.values())
            breakdown['feature_clarity'] = min(1.0, top_importance * 5)
        else:
            breakdown['feature_clarity'] = 0.5
        
        # 预测稳定性分数 (基于历史)
        if len(self._decision_history) > 5:
            recent_preds = [d['prediction'] for d in self._decision_history[-5:]]
            # 简化：假设数值预测
            if all(isinstance(p, (int, float)) for p in recent_preds):
                pred_std = np.std(recent_preds)
                breakdown['prediction_stability'] = max(0, 1 - pred_std / 10)
            else:
                breakdown['prediction_stability'] = 0.7
        else:
            breakdown['prediction_stability'] = 0.7
        
        # 综合置信度
        breakdown['overall'] = np.mean(list(breakdown.values()))
        
        return breakdown
    
    def explain_anomaly(
        self,
        input_data: np.ndarray,
        normal_baseline: np.ndarray,
        anomaly_score: float
    ) -> AnomalyExplanation:
        """
        解释异常。
        
        Args:
            input_data: 异常输入
            normal_baseline: 正常基线
            anomaly_score: 异常分数
            
        Returns:
            AnomalyExplanation包含异常解释
        """
        # 计算差异图
        diff = np.abs(input_data.astype(np.float64) - normal_baseline.astype(np.float64))
        
        # 识别异常类型
        anomaly_type = self._classify_anomaly(diff, anomaly_score)
        
        # 定位根因
        root_causes = self._identify_root_causes(diff, input_data)
        
        # 检测受影响区域
        affected_regions = self._detect_anomaly_regions(diff)
        
        # 生成建议
        recommendations = self._generate_recommendations(anomaly_type, root_causes)
        
        # 计算严重程度
        severity = min(1.0, anomaly_score / self.config.anomaly_threshold)
        
        return AnomalyExplanation(
            anomaly_type=anomaly_type,
            root_causes=root_causes,
            affected_regions=affected_regions,
            recommended_actions=recommendations,
            severity_score=severity
        )
    
    def _classify_anomaly(
        self,
        diff: np.ndarray,
        anomaly_score: float
    ) -> str:
        """分类异常类型。"""
        # 基于差异统计特征分类
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        if max_diff > mean_diff * 5:
            return "点异常 (Point Anomaly)"
        elif np.std(diff) > mean_diff * 2:
            return "分布异常 (Distribution Anomaly)"
        elif anomaly_score > self.config.anomaly_threshold * 2:
            return "严重异常 (Severe Anomaly)"
        else:
            return "轻微异常 (Minor Anomaly)"
    
    def _identify_root_causes(
        self,
        diff: np.ndarray,
        input_data: np.ndarray
    ) -> List[Tuple[str, float]]:
        """识别异常根因。"""
        causes = []
        
        # 检查亮度异常
        mean_input = np.mean(input_data)
        if mean_input < 50:
            causes.append(("光照不足", 0.8))
        elif mean_input > 200:
            causes.append(("过曝", 0.7))
        
        # 检查噪声
        noise_level = np.std(diff)
        if noise_level > 30:
            causes.append(("高噪声", 0.6))
        
        # 检查对比度
        if np.max(input_data) - np.min(input_data) < 20:
            causes.append(("低对比度", 0.5))
        
        # 如果没有明显原因
        if not causes:
            causes.append(("未知原因", 0.3))
        
        return causes
    
    def _detect_anomaly_regions(
        self,
        diff: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """检测异常区域。"""
        regions = []
        
        if diff.ndim < 2:
            return regions
        
        # 二值化差异图
        threshold = np.percentile(diff, 95)
        binary = (diff > threshold).astype(np.uint8) * 255
        
        # 查找轮廓
        if binary.ndim == 3:
            binary = cv2.cvtColor(binary.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        contours, _ = cv2.findContours(
            binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= self.config.min_anomaly_region_size:
                regions.append((x, y, w, h))
        
        return regions
    
    def _generate_recommendations(
        self,
        anomaly_type: str,
        root_causes: List[Tuple[str, float]]
    ) -> List[str]:
        """生成建议。"""
        recommendations = []
        
        # 基于根因生成建议
        for cause, confidence in root_causes:
            if "光照" in cause:
                recommendations.append("调整光源强度或曝光时间")
            elif "噪声" in cause:
                recommendations.append("启用降噪处理或增加积分时间")
            elif "对比度" in cause:
                recommendations.append("调整图像增强参数")
        
        # 通用建议
        recommendations.append("检查系统校准状态")
        recommendations.append("记录异常用于后续分析")
        
        return recommendations
    
    def visualize_explanation(
        self,
        input_data: np.ndarray,
        explanation: ExplanationResult,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        可视化解释结果。
        
        Args:
            input_data: 输入数据
            explanation: 解释结果
            save_path: 保存路径
            
        Returns:
            可视化图像
        """
        # 准备输入图像
        if input_data.ndim == 2:
            vis_input = cv2.cvtColor(input_data.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif input_data.ndim == 3:
            vis_input = input_data.astype(np.uint8)
        else:
            vis_input = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # 调整大小
        h, w = 256, 256
        vis_input = cv2.resize(vis_input, (w, h))
        
        # 归因图着色
        attribution = explanation.attribution_map
        if attribution.size > 0:
            attribution_resized = cv2.resize(attribution, (w, h))
            attribution_colored = cv2.applyColorMap(attribution_resized, self.config.colormap)
            
            # 混合
            vis_output = cv2.addWeighted(
                vis_input, 1 - self.config.alpha_blend,
                attribution_colored, self.config.alpha_blend, 0
            )
        else:
            vis_output = vis_input
        
        # 添加文字信息
        y_offset = 20
        for feature, importance in list(explanation.feature_importance.items())[:5]:
            text = f"{feature}: {importance:.3f}"
            cv2.putText(vis_output, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
        
        if save_path:
            cv2.imwrite(save_path, vis_output)
        
        return vis_output


# ============================================================
# 辅助函数
# ============================================================

def create_dummy_model() -> Callable:
    """创建测试用的虚拟模型。"""
    def model_forward(x: np.ndarray) -> float:
        # 简化的模型：基于输入统计的预测
        return float(np.mean(x) + np.std(x) * 0.5)
    return model_forward


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建XAI诊断模块
    xai = XAIDiagnostic()
    
    # 创建测试数据
    test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # 创建虚拟模型
    model = create_dummy_model()
    
    # 生成解释
    prediction = model(test_image)
    explanation = xai.explain_prediction(test_image, prediction, model)
    
    print(f"解释结果:")
    print(f"  特征重要性 (Top 5):")
    for feature, importance in list(explanation.feature_importance.items())[:5]:
        print(f"    {feature}: {importance:.4f}")
    print(f"  决策路径:")
    for step in explanation.decision_path:
        print(f"    - {step}")
    print(f"  置信度分解:")
    for component, score in explanation.confidence_breakdown.items():
        print(f"    {component}: {score:.3f}")
    print(f"  处理时间: {explanation.processing_time_ms:.2f} ms")
    
    # 测试异常解释
    normal_baseline = np.ones_like(test_image) * 128
    anomaly_exp = xai.explain_anomaly(test_image, normal_baseline, anomaly_score=3.5)
    
    print(f"\n异常解释:")
    print(f"  异常类型: {anomaly_exp.anomaly_type}")
    print(f"  根因:")
    for cause, confidence in anomaly_exp.root_causes:
        print(f"    - {cause} (置信度: {confidence:.2f})")
    print(f"  建议措施:")
    for action in anomaly_exp.recommended_actions:
        print(f"    - {action}")
    print(f"  严重程度: {anomaly_exp.severity_score:.2f}")
