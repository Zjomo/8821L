"""
SpotZoom Machine Learning Package v5 — 前沿开源项目创新模块

基于 2024-2026 年科研前沿开源项目调研，为 SpotZoom 光斑对准系统
提供新一代创新模块。

参考开源项目:
- CAREamics: 统一PyTorch去噪库 (https://github.com/CAREamics/careamics)
- slmsuite: SLM控制与GPU相位恢复 (https://github.com/holodyne/slmsuite)
- DECODE: 深度上下文依赖定位 (https://github.com/TuragaLab/DECODE)
- pyRTC: Python实时AO控制器 (https://github.com/jacotay7/pyRTC)
- dmlib: 变形镜校准库 (https://github.com/jacopoantonello/dmlib)
- Picasso: SMLM分析与漂移校正 (https://github.com/jungmannlab/picasso)
- OOPAO: 面向对象自适应光学仿真 (https://github.com/cheritier/OOPAO)
- PINNs-Torch: JIT加速PINN (https://github.com/rezaakb/pinns-torch)
- PINA: 物理信息神经网络 (https://github.com/mathLab/PINA)
- REALM: 鲁棒有效AO定位显微 (https://github.com/MSiemons/REALM)
- python-microscope: 显微镜设备控制 (https://github.com/python-microscope/microscope)

模块分类:
1. 图像增强模块 (Image Enhancement)
2. 精密定位模块 (Precision Localization)
3. 自适应光学模块 (Adaptive Optics)
4. 实时控制模块 (Real-time Control)
5. 系统诊断模块 (System Diagnostics)
"""

__version__ = "5.0.0"
__author__ = "SpotZoom Team"


def _try_import(module_name: str):
    """尝试导入模块，失败返回 None。排除 SyntaxError 等编程错误。"""
    try:
        import importlib
        return importlib.import_module(f".{module_name}", __package__)
    except (ImportError, ModuleNotFoundError, AttributeError, OSError, ValueError):
        return None


# ============ 图像增强模块 ============
_careamics_adapter = _try_import("careamics_adapter")
if _careamics_adapter:
    CAREamicsAdapter = _careamics_adapter.CAREamicsAdapter
    CAREamicsConfig = _careamics_adapter.CAREamicsConfig
    CAREamicsReport = _careamics_adapter.CAREamicsReport

_self_supervised_denoiser = _try_import("self_supervised_denoiser_v2")
if _self_supervised_denoiser:
    SelfSupervisedDenoiserV2 = _self_supervised_denoiser.SelfSupervisedDenoiserV2
    DenoiserV2Config = _self_supervised_denoiser.DenoiserV2Config
    DenoiserV2Report = _self_supervised_denoiser.DenoiserV2Report

# ============ 精密定位模块 ============
_uncertainty_localizer = _try_import("uncertainty_aware_localizer_v2")
if _uncertainty_localizer:
    UncertaintyAwareLocalizerV2 = _uncertainty_localizer.UncertaintyAwareLocalizerV2
    LocalizerV2Config = _uncertainty_localizer.LocalizerV2Config
    LocalizerV2Result = _uncertainty_localizer.LocalizerV2Result

_nena_precision = _try_import("nena_precision_assessor")
if _nena_precision:
    NeNAPrecisionAssessor = _nena_precision.NeNAPrecisionAssessor
    NeNAConfig = _nena_precision.NeNAConfig
    NeNAReport = _nena_precision.NeNAReport

_context_aware_detector = _try_import("context_aware_detector")
if _context_aware_detector:
    ContextAwareDetector = _context_aware_detector.ContextAwareDetector
    ContextDetectorConfig = _context_aware_detector.ContextDetectorConfig
    ContextDetectorResult = _context_aware_detector.ContextDetectorResult

# ============ 自适应光学模块 ============
_gpu_phase_retrieval = _try_import("gpu_phase_retrieval")
if _gpu_phase_retrieval:
    GPUPhaseRetrieval = _gpu_phase_retrieval.GPUPhaseRetrieval
    PhaseRetrievalConfig = _gpu_phase_retrieval.PhaseRetrievalConfig
    PhaseRetrievalResult = _gpu_phase_retrieval.PhaseRetrievalResult

_dm_calibrator_v2 = _try_import("dm_calibrator_v2")
if _dm_calibrator_v2:
    DMCalibratorV2 = _dm_calibrator_v2.DMCalibratorV2
    DMCalibratorV2Config = _dm_calibrator_v2.DMCalibratorV2Config
    DMCalibratorV2Report = _dm_calibrator_v2.DMCalibratorV2Report

_sensorless_ao_v2 = _try_import("sensorless_ao_v2")
if _sensorless_ao_v2:
    SensorlessAOV2 = _sensorless_ao_v2.SensorlessAOV2
    SensorlessAOV2Config = _sensorless_ao_v2.SensorlessAOV2Config
    SensorlessAOV2Report = _sensorless_ao_v2.SensorlessAOV2Report

# ============ 实时控制模块 ============
_realtime_pipeline_v2 = _try_import("realtime_pipeline_v2")
if _realtime_pipeline_v2:
    RealtimePipelineV2 = _realtime_pipeline_v2.RealtimePipelineV2
    PipelineV2Config = _realtime_pipeline_v2.PipelineV2Config
    PipelineV2Report = _realtime_pipeline_v2.PipelineV2Report

_subspace_identifier = _try_import("subspace_system_identifier")
if _subspace_identifier:
    SubspaceSystemIdentifier = _subspace_identifier.SubspaceSystemIdentifier
    SubspaceConfig = _subspace_identifier.SubspaceConfig
    SubspaceReport = _subspace_identifier.SubspaceReport

# ============ 系统诊断模块 ============
_predictive_health = _try_import("predictive_health_monitor")
if _predictive_health:
    PredictiveHealthMonitor = _predictive_health.PredictiveHealthMonitor
    PredictiveHealthConfig = _predictive_health.PredictiveHealthConfig
    PredictiveHealthReport = _predictive_health.PredictiveHealthReport

_regression_guard = _try_import("regression_guard")
if _regression_guard:
    RegressionGuard = _regression_guard.RegressionGuard
    RegressionGuardConfig = _regression_guard.RegressionGuardConfig
    RegressionGuardReport = _regression_guard.RegressionGuardReport


__all__ = [
    # 图像增强
    "CAREamicsAdapter", "CAREamicsConfig", "CAREamicsReport",
    "SelfSupervisedDenoiserV2", "DenoiserV2Config", "DenoiserV2Report",
    # 精密定位
    "UncertaintyAwareLocalizerV2", "LocalizerV2Config", "LocalizerV2Result",
    "NeNAPrecisionAssessor", "NeNAConfig", "NeNAReport",
    "ContextAwareDetector", "ContextDetectorConfig", "ContextDetectorResult",
    # 自适应光学
    "GPUPhaseRetrieval", "PhaseRetrievalConfig", "PhaseRetrievalResult",
    "DMCalibratorV2", "DMCalibratorV2Config", "DMCalibratorV2Report",
    "SensorlessAOV2", "SensorlessAOV2Config", "SensorlessAOV2Report",
    # 实时控制
    "RealtimePipelineV2", "PipelineV2Config", "PipelineV2Report",
    "SubspaceSystemIdentifier", "SubspaceConfig", "SubspaceReport",
    # 系统诊断
    "PredictiveHealthMonitor", "PredictiveHealthConfig", "PredictiveHealthReport",
    "RegressionGuard", "RegressionGuardConfig", "RegressionGuardReport",
]
