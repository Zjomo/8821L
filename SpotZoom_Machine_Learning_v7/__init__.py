"""
SpotZoom Machine Learning Package v7 - 前沿开源项目创新模块

基于 2025-2026 年科研前沿开源项目的创新模块集合，为 SpotZoom 光斑对准系统
提供自监督降噪、最优传输对齐、因果状态预测、傅里叶相位相关、注意力跟踪、
神经 ODE 控制、拓扑优化等高级功能。

参考开源项目:
- Noise2Void (javiribera): 自监督图像降噪 (https://github.com/javiribera/Noise2Void)
- POT / PythonOT: Python 最优传输库 (https://github.com/PythonOT/POT)
- Mamba SSM (state-spaces): 状态空间模型 (https://github.com/state-spaces/mamba)
- sub-pixel phase correlation: 亚像素相位相关配准
- TrackFormer (mikel-brostrom): Transformer 目标跟踪 (https://github.com/mikel-brostrom/TrackFormer)
- torchdiffeq (rtqichen): 神经常微分方程 (https://github.com/rtqichen/torchdiffeq)
- CMA-ES (CMA-ES/pycma): 协方差矩阵自适应进化策略 (https://github.com/CMA-ES/pycma)

模块分类:
1. 图像增强模块 (Image Enhancement)
2. 对齐与配准模块 (Alignment & Registration)
3. 状态预测模块 (State Prediction)
4. 跟踪模块 (Tracking)
5. 控制模块 (Control)
6. 优化模块 (Optimization)
"""

__version__ = "7.0.0"
__author__ = "SpotZoom Team"

__all__ = [
    # 图像增强
    "SelfSupervisedDenoiser",
    "DenoiserConfig",
    "DenoiseReport",
    # 对齐与配准
    "OptimalTransportAligner",
    "OTAlignerConfig",
    "OTAlignmentResult",
    "FourierPhaseCorrelator",
    "PhaseCorrelatorConfig",
    "PhaseCorrelationResult",
    # 状态预测
    "CausalStatePredictor",
    "CausalPredictorConfig",
    "CausalPredictionResult",
    # 跟踪
    "AttentionSpotTracker",
    "AttentionTrackerConfig",
    "AttentionTrackResult",
    # 控制
    "NeuralODEController",
    "NeuralODEConfig",
    "NeuralODEResult",
    # 优化
    "TopologyAwareOptimizer",
    "TopologyOptimizerConfig",
    "TopologyOptResult",
]


def _try_import(module_name: str):
    """尝试导入模块，失败返回 None。排除 SyntaxError 等编程错误。"""
    try:
        import importlib
        return importlib.import_module(f".{module_name}", __package__)
    except (ImportError, ModuleNotFoundError, AttributeError, OSError, ValueError):
        return None


# ============ 图像增强模块 ============
_denoiser = _try_import("self_supervised_denoiser")
if _denoiser:
    SelfSupervisedDenoiser = _denoiser.SelfSupervisedDenoiser
    DenoiserConfig = _denoiser.DenoiserConfig
    DenoiseReport = _denoiser.DenoiseReport

# ============ 对齐与配准模块 ============
_ot_aligner = _try_import("optimal_transport_aligner")
if _ot_aligner:
    OptimalTransportAligner = _ot_aligner.OptimalTransportAligner
    OTAlignerConfig = _ot_aligner.OTAlignerConfig
    OTAlignmentResult = _ot_aligner.OTAlignmentResult

_fourier_corr = _try_import("fourier_phase_correlator")
if _fourier_corr:
    FourierPhaseCorrelator = _fourier_corr.FourierPhaseCorrelator
    PhaseCorrelatorConfig = _fourier_corr.PhaseCorrelatorConfig
    PhaseCorrelationResult = _fourier_corr.PhaseCorrelationResult

# ============ 状态预测模块 ============
_causal_pred = _try_import("causal_state_predictor")
if _causal_pred:
    CausalStatePredictor = _causal_pred.CausalStatePredictor
    CausalPredictorConfig = _causal_pred.CausalPredictorConfig
    CausalPredictionResult = _causal_pred.CausalPredictionResult

# ============ 跟踪模块 ============
_attn_tracker = _try_import("attention_spot_tracker")
if _attn_tracker:
    AttentionSpotTracker = _attn_tracker.AttentionSpotTracker
    AttentionTrackerConfig = _attn_tracker.AttentionTrackerConfig
    AttentionTrackResult = _attn_tracker.AttentionTrackResult

# ============ 控制模块 ============
_neural_ode = _try_import("neural_ode_controller")
if _neural_ode:
    NeuralODEController = _neural_ode.NeuralODEController
    NeuralODEConfig = _neural_ode.NeuralODEConfig
    NeuralODEResult = _neural_ode.NeuralODEResult

# ============ 优化模块 ============
_topology_opt = _try_import("topology_aware_optimizer")
if _topology_opt:
    TopologyAwareOptimizer = _topology_opt.TopologyAwareOptimizer
    TopologyOptimizerConfig = _topology_opt.TopologyOptimizerConfig
    TopologyOptResult = _topology_opt.TopologyOptResult


# 清理临时变量
del _try_import
del _denoiser, _ot_aligner, _fourier_corr
del _causal_pred, _attn_tracker, _neural_ode, _topology_opt
