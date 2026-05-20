"""
SpotZoom Machine Learning Package v6 - 前沿开源项目创新模块

基于 2025-2026 年科研前沿开源项目的创新模块集合。

参考开源项目:
- MicroSAM (EMBL): 显微镜分割基础模型 (https://github.com/computational-cell-analytics/micro-sam)
- DeepTrack2 (DeepTrackAI): 模块化粒子追踪库 (https://github.com/DeepTrackAI/DeepTrack2)
- slmsuite (kieranjol): SLM 控制与全息工具包 (https://github.com/kieranjol/slmsuite)
- prysm (brandondube): 光学衍射仿真 (https://github.com/brandondube/prysm)
- CAREamics (CAREamics): 自监督去噪框架 (https://github.com/CAREamics/CAREamics)
- Picasso (jungmannlab): GPU 亚像素定位 (https://github.com/jungmannlab/picasso)
- DECODE (TuragaLab): 深度上下文依赖定位 (https://github.com/TuragaLab/DECODE)
- REALM (MSiemons): Zernike 模式 AO 校正 (https://github.com/MSiemons/REALM)
- BoTorch (pytorch): 贝叶斯优化 (https://github.com/pytorch/botorch)
- Optuna (optuna): 超参数自动优化 (https://github.com/optuna/optuna)

模块分类:
1. 智能检测模块 (Intelligent Detection)
2. 自适应控制模块 (Adaptive Control)
3. 光学仿真模块 (Optical Simulation)
4. 系统诊断模块 (System Diagnostics)
5. 数据增强模块 (Data Enhancement)
"""

__version__ = "6.0.0"
__author__ = "SpotZoom Team"

def _try_import(module_name: str):
    """尝试导入模块，失败返回 None。"""
    try:
        import importlib
        return importlib.import_module(f".{module_name}", __package__)
    except (ImportError, ModuleNotFoundError, AttributeError, OSError, ValueError):
        return None

# ============ 智能检测模块 ============
_multi_scale_detector = _try_import("multi_scale_spot_detector")
if _multi_scale_detector:
    MultiScaleSpotDetector = _multi_scale_detector.MultiScaleSpotDetector
    MultiScaleDetectorConfig = _multi_scale_detector.MultiScaleDetectorConfig
    MultiScaleDetectionResult = _multi_scale_detector.MultiScaleDetectionResult

_diffusion_enhancer = _try_import("diffusion_spot_enhancer")
if _diffusion_enhancer:
    DiffusionSpotEnhancer = _diffusion_enhancer.DiffusionSpotEnhancer
    DiffusionEnhancerConfig = _diffusion_enhancer.DiffusionEnhancerConfig

# ============ 自适应控制模块 ============
_bayesian_pid = _try_import("bayesian_pid_optimizer")
if _bayesian_pid:
    BayesianPIDOptimizer = _bayesian_pid.BayesianPIDOptimizer
    BayesianPIDConfig = _bayesian_pid.BayesianPIDConfig
    BayesianPIDResult = _bayesian_pid.BayesianPIDResult

_adaptive_feedforward = _try_import("adaptive_feedforward_controller")
if _adaptive_feedforward:
    AdaptiveFeedforwardController = _adaptive_feedforward.AdaptiveFeedforwardController
    FeedforwardConfig = _adaptive_feedforward.FeedforwardConfig

# ============ 光学仿真模块 ============
_psf_fingerprint = _try_import("psf_fingerprint_analyzer")
if _psf_fingerprint:
    PSFFingerprintAnalyzer = _psf_fingerprint.PSFFingerprintAnalyzer
    PSFFingerprintConfig = _psf_fingerprint.PSFFingerprintConfig
    PSFFingerprintReport = _psf_fingerprint.PSFFingerprintReport

_coherent_sensitivity = _try_import("coherent_sensitivity_analyzer")
if _coherent_sensitivity:
    CoherentSensitivityAnalyzer = _coherent_sensitivity.CoherentSensitivityAnalyzer
    SensitivityConfig = _coherent_sensitivity.SensitivityConfig
    SensitivityReport = _coherent_sensitivity.SensitivityReport

# ============ 系统诊断模块 ============
_runtime_profiler = _try_import("runtime_profiler")
if _runtime_profiler:
    RuntimeProfiler = _runtime_profiler.RuntimeProfiler
    ProfilerConfig = _runtime_profiler.ProfilerConfig
    ProfilerReport = _runtime_profiler.ProfilerReport

_convergence_guard = _try_import("convergence_guard")
if _convergence_guard:
    ConvergenceGuard = _convergence_guard.ConvergenceGuard
    ConvergenceGuardConfig = _convergence_guard.ConvergenceGuardConfig
    ConvergenceReport = _convergence_guard.ConvergenceReport

# ============ 数据增强模块 =#
_synth_diffraction = _try_import("synthetic_diffraction_generator")
if _synth_diffraction:
    SyntheticDiffractionGenerator = _synth_diffraction.SyntheticDiffractionGenerator
    DiffractionGeneratorConfig = _synth_diffraction.DiffractionGeneratorConfig

_temporal_ensemble = _try_import("temporal_ensemble_tracker")
if _temporal_ensemble:
    TemporalEnsembleTracker = _temporal_ensemble.TemporalEnsembleTracker
    TemporalEnsembleConfig = _temporal_ensemble.TemporalEnsembleConfig
    TemporalEnsembleResult = _temporal_ensemble.TemporalEnsembleResult
