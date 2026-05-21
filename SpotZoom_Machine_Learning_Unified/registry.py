from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class OptimizationType(str, Enum):
    IMAGE_ENHANCEMENT = "image_enhancement"
    DETECTION_LOCALIZATION = "detection_localization"
    TRACKING_PREDICTION = "tracking_prediction"
    ALIGNMENT_REGISTRATION = "alignment_registration"
    ADAPTIVE_OPTICS_CONTROL = "adaptive_optics_control"
    BEAM_OPTICAL_SIMULATION = "beam_optical_simulation"
    SYSTEM_ANALYSIS = "system_analysis"
    SYSTEM_DIAGNOSTICS = "system_diagnostics"
    SYSTEM_IDENTIFICATION = "system_identification"
    HARDWARE_ABSTRACTION = "hardware_abstraction"
    GLOBAL_OPTIMIZATION = "global_optimization"


TYPE_LABELS: Dict[OptimizationType, str] = {
    OptimizationType.IMAGE_ENHANCEMENT: "图像增强",
    OptimizationType.DETECTION_LOCALIZATION: "检测与定位",
    OptimizationType.TRACKING_PREDICTION: "跟踪与预测",
    OptimizationType.ALIGNMENT_REGISTRATION: "配准与对齐",
    OptimizationType.ADAPTIVE_OPTICS_CONTROL: "自适应光学与控制",
    OptimizationType.BEAM_OPTICAL_SIMULATION: "光束与光学仿真",
    OptimizationType.SYSTEM_ANALYSIS: "系统分析",
    OptimizationType.SYSTEM_DIAGNOSTICS: "系统诊断",
    OptimizationType.SYSTEM_IDENTIFICATION: "系统辨识",
    OptimizationType.HARDWARE_ABSTRACTION: "硬件抽象",
    OptimizationType.GLOBAL_OPTIMIZATION: "全局优化",
}


MODULE_TYPE_OVERRIDES: Dict[str, OptimizationType] = {
    "adaptive_beam_propagator": OptimizationType.BEAM_OPTICAL_SIMULATION,
    "closed_loop_ao_controller": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "data_driven_mpc": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "deep_image_prior_enhancer": OptimizationType.IMAGE_ENHANCEMENT,
    "fourier_psf_analyzer": OptimizationType.SYSTEM_ANALYSIS,
    "lqg_robust_controller": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "deformable_mirror_calibrator": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "hardware_abstraction_layer": OptimizationType.HARDWARE_ABSTRACTION,
    "laplacian_autofocus": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "realtime_ao_pipeline": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "slm_holographic_spot_generator": OptimizationType.BEAM_OPTICAL_SIMULATION,
    "strehl_quality_assessor": OptimizationType.SYSTEM_ANALYSIS,
    "synthetic_data_generator": OptimizationType.BEAM_OPTICAL_SIMULATION,
    "system_identifier": OptimizationType.SYSTEM_IDENTIFICATION,
    "adaptive_scheduler": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "beam_propagation_engine": OptimizationType.BEAM_OPTICAL_SIMULATION,
    "digital_twin_enhancer": OptimizationType.BEAM_OPTICAL_SIMULATION,
    "intelligent_anomaly_healer": OptimizationType.SYSTEM_DIAGNOSTICS,
    "multi_objective_bayesian_opt": OptimizationType.GLOBAL_OPTIMIZATION,
    "multi_sensor_fusion": OptimizationType.DETECTION_LOCALIZATION,
    "online_ao_corrector": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "careamics_adapter": OptimizationType.IMAGE_ENHANCEMENT,
    "context_aware_detector": OptimizationType.DETECTION_LOCALIZATION,
    "dm_calibrator_v2": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "gpu_phase_retrieval": OptimizationType.SYSTEM_ANALYSIS,
    "nena_precision_assessor": OptimizationType.SYSTEM_ANALYSIS,
    "predictive_health_monitor": OptimizationType.SYSTEM_DIAGNOSTICS,
    "realtime_pipeline_v2": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "regression_guard": OptimizationType.SYSTEM_DIAGNOSTICS,
    "self_supervised_denoiser_v2": OptimizationType.IMAGE_ENHANCEMENT,
    "sensorless_ao_v2": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "subspace_system_identifier": OptimizationType.SYSTEM_IDENTIFICATION,
    "uncertainty_aware_localizer_v2": OptimizationType.DETECTION_LOCALIZATION,
    "adaptive_feedforward_controller": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "bayesian_pid_optimizer": OptimizationType.GLOBAL_OPTIMIZATION,
    "coherent_sensitivity_analyzer": OptimizationType.SYSTEM_ANALYSIS,
    "convergence_guard": OptimizationType.SYSTEM_DIAGNOSTICS,
    "diffusion_spot_enhancer": OptimizationType.IMAGE_ENHANCEMENT,
    "multi_scale_spot_detector": OptimizationType.DETECTION_LOCALIZATION,
    "psf_fingerprint_analyzer": OptimizationType.SYSTEM_ANALYSIS,
    "runtime_profiler": OptimizationType.SYSTEM_DIAGNOSTICS,
    "synthetic_diffraction_generator": OptimizationType.BEAM_OPTICAL_SIMULATION,
    "temporal_ensemble_tracker": OptimizationType.TRACKING_PREDICTION,
    "attention_spot_tracker": OptimizationType.TRACKING_PREDICTION,
    "causal_state_predictor": OptimizationType.TRACKING_PREDICTION,
    "fourier_phase_correlator": OptimizationType.ALIGNMENT_REGISTRATION,
    "neural_ode_controller": OptimizationType.ADAPTIVE_OPTICS_CONTROL,
    "optimal_transport_aligner": OptimizationType.ALIGNMENT_REGISTRATION,
    "self_supervised_denoiser": OptimizationType.IMAGE_ENHANCEMENT,
    "topology_aware_optimizer": OptimizationType.GLOBAL_OPTIMIZATION,
}


TYPE_DESCRIPTION_TEMPLATES: Dict[OptimizationType, str] = {
    OptimizationType.IMAGE_ENHANCEMENT: "该模块用于提升图像质量、抑制噪声或增强光斑可分辨性。",
    OptimizationType.DETECTION_LOCALIZATION: "该模块用于光斑检测、候选筛选或定位精度提升。",
    OptimizationType.TRACKING_PREDICTION: "该模块用于时序跟踪、状态估计或运动趋势预测。",
    OptimizationType.ALIGNMENT_REGISTRATION: "该模块用于图像配准、相位相关或坐标对齐。",
    OptimizationType.ADAPTIVE_OPTICS_CONTROL: "该模块用于控制回路、自适应光学补偿或实时调节。",
    OptimizationType.BEAM_OPTICAL_SIMULATION: "该模块用于光束传播、全息生成或数字孪生仿真。",
    OptimizationType.SYSTEM_ANALYSIS: "该模块用于性能评估、像差分析或频域/统计分析。",
    OptimizationType.SYSTEM_DIAGNOSTICS: "该模块用于运行诊断、健康监控或稳定性保护。",
    OptimizationType.SYSTEM_IDENTIFICATION: "该模块用于系统辨识、模型拟合或参数识别。",
    OptimizationType.HARDWARE_ABSTRACTION: "该模块用于统一设备接口或隔离底层硬件差异。",
    OptimizationType.GLOBAL_OPTIMIZATION: "该模块用于参数搜索、全局优化或调参策略求解。",
}


SUMMARY_FALLBACK_NAMES: Dict[str, str] = {
    "adaptive_beam_propagator": "自适应光束传播模拟器",
    "closed_loop_ao_controller": "闭环自适应光学控制器",
    "data_driven_mpc": "数据驱动 MPC 控制器",
    "deep_image_prior_enhancer": "Deep Image Prior 光斑增强器",
    "fourier_psf_analyzer": "傅里叶 PSF 分析器",
    "lqg_robust_controller": "LQG 鲁棒控制器",
    "deformable_mirror_calibrator": "变形镜校准器",
    "hardware_abstraction_layer": "硬件抽象层",
    "laplacian_autofocus": "拉普拉斯自动对焦器",
    "realtime_ao_pipeline": "实时 AO 流水线",
    "slm_holographic_spot_generator": "SLM 全息光斑生成器",
    "strehl_quality_assessor": "Strehl 质量评估器",
    "synthetic_data_generator": "合成数据生成器",
    "system_identifier": "系统辨识器",
    "adaptive_scheduler": "自适应调度器",
    "beam_propagation_engine": "光束传播引擎",
    "digital_twin_enhancer": "数字孪生增强器",
    "intelligent_anomaly_healer": "智能异常修复器",
    "multi_objective_bayesian_opt": "多目标贝叶斯优化器",
    "multi_sensor_fusion": "多传感器融合器",
    "online_ao_corrector": "在线 AO 校正器",
    "careamics_adapter": "CAREamics 适配器",
    "context_aware_detector": "上下文感知检测器",
    "dm_calibrator_v2": "变形镜校准器 V2",
    "gpu_phase_retrieval": "GPU 相位恢复器",
    "nena_precision_assessor": "NeNA 精度评估器",
    "predictive_health_monitor": "预测式健康监控器",
    "realtime_pipeline_v2": "实时流水线 V2",
    "regression_guard": "回归保护器",
    "self_supervised_denoiser_v2": "自监督去噪器 V2",
    "sensorless_ao_v2": "无传感器 AO 控制器 V2",
    "subspace_system_identifier": "子空间系统辨识器",
    "uncertainty_aware_localizer_v2": "不确定性感知定位器 V2",
    "adaptive_feedforward_controller": "自适应前馈控制器",
    "bayesian_pid_optimizer": "贝叶斯 PID 优化器",
    "coherent_sensitivity_analyzer": "相干灵敏度分析器",
    "convergence_guard": "收敛保护器",
    "diffusion_spot_enhancer": "扩散式光斑增强器",
    "multi_scale_spot_detector": "多尺度光斑检测器",
    "psf_fingerprint_analyzer": "PSF 指纹分析器",
    "runtime_profiler": "运行时性能分析器",
    "synthetic_diffraction_generator": "合成衍射生成器",
    "temporal_ensemble_tracker": "时序集成跟踪器",
    "attention_spot_tracker": "注意力光斑跟踪器",
    "causal_state_predictor": "因果状态预测器",
    "fourier_phase_correlator": "傅里叶相位相关器",
    "neural_ode_controller": "Neural ODE 控制器",
    "optimal_transport_aligner": "最优传输对齐器",
    "self_supervised_denoiser": "自监督去噪器",
    "topology_aware_optimizer": "拓扑感知优化器",
}


SUPPORT_CLASS_SUFFIXES = (
    "Config",
    "Result",
    "Report",
    "State",
    "Sample",
    "Scene",
    "Point",
    "Action",
    "Measurement",
    "Info",
    "Strategy",
    "Method",
)

MOJIBAKE_HINTS = "鈥鍩轰簬闂幆鑷€傚簲鍏夊妯￠�锛"


@dataclass(frozen=True)
class ModuleSpec:
    version: int
    package: str
    module_name: str
    optimization_type: OptimizationType
    type_label: str
    source_path: str
    import_path: str
    title: str
    summary: str
    primary_symbol: Optional[str]
    config_symbol: Optional[str]
    result_symbols: Tuple[str, ...]
    public_classes: Tuple[str, ...]
    public_functions: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["optimization_type"] = self.optimization_type.value
        return payload


class ModuleAdapter:
    def __init__(self, spec: ModuleSpec):
        self.spec = spec
        self._module: Optional[Any] = None

    def load_module(self) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self.spec.import_path)
        return self._module

    def get_symbol(self, symbol_name: str) -> Any:
        return getattr(self.load_module(), symbol_name)

    def build_config(self, **overrides: Any) -> Any:
        if not self.spec.config_symbol:
            raise ValueError(f"{self.spec.import_path} 未声明配置类")
        config_cls = self.get_symbol(self.spec.config_symbol)
        return config_cls(**overrides)

    def build_instance(
        self,
        *,
        config: Optional[Any] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if not self.spec.primary_symbol:
            raise ValueError(f"{self.spec.import_path} 未声明主入口类")
        target_cls = self.get_symbol(self.spec.primary_symbol)
        if config is None and self.spec.config_symbol:
            config = self.build_config(**(config_overrides or {}))
        if config is not None:
            try:
                return target_cls(config=config, **kwargs)
            except TypeError:
                try:
                    return target_cls(config, **kwargs)
                except TypeError:
                    pass
        if kwargs:
            return target_cls(**kwargs)
        try:
            return target_cls()
        except TypeError as exc:
            raise TypeError(f"{self.spec.import_path}.{self.spec.primary_symbol} 无法使用统一方式实例化") from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _score_text(text: str) -> int:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    bad = sum(text.count(ch) for ch in MOJIBAKE_HINTS)
    replacement = text.count("�") + text.count("?")
    return cjk * 4 + ascii_letters - bad * 2 - replacement


def _repair_text(text: str) -> str:
    candidates = [text]
    for encoding in ("gbk", "latin1"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except Exception:
            continue
    best = max(candidates, key=_score_text)
    return best.strip()


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _extract_public_functions(tree: ast.Module) -> Tuple[str, ...]:
    functions: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            functions.append(node.name)
    return tuple(functions)


def _extract_public_classes(tree: ast.Module) -> Tuple[str, ...]:
    classes: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(node.name)
    return tuple(classes)


def _camel_tokens(name: str) -> Tuple[str, ...]:
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+", name)
    return tuple(part.lower() for part in parts if part)


def _snake_tokens(name: str) -> Tuple[str, ...]:
    return tuple(part.lower() for part in name.split("_") if part)


def _primary_symbol_score(module_name: str, class_name: str) -> Tuple[int, int]:
    module_tokens = set(_snake_tokens(module_name))
    class_tokens = set(_camel_tokens(class_name))
    overlap = len(module_tokens & class_tokens)
    length_bonus = len(class_tokens)
    return overlap, length_bonus


def _split_symbols(module_name: str, classes: Sequence[str]) -> Tuple[Optional[str], Optional[str], Tuple[str, ...]]:
    config_symbol = next((name for name in classes if name.endswith("Config")), None)
    result_symbols = tuple(
        name for name in classes if name.endswith(("Result", "Report", "State", "Sample", "Point", "Action", "Measurement"))
    )
    candidates = [
        name
        for name in classes
        if name != config_symbol and name not in result_symbols and not name.endswith(SUPPORT_CLASS_SUFFIXES)
    ]
    primary_symbol = None
    if candidates:
        primary_symbol = max(candidates, key=lambda name: _primary_symbol_score(module_name, name))
    return primary_symbol, config_symbol, result_symbols


def _module_sort_key(path: Path) -> Tuple[int, str]:
    version = int(path.parent.name.rsplit("v", 1)[1])
    return version, path.stem


def _derive_title(module_name: str, docstring: str) -> str:
    first_line = ""
    if docstring:
        repaired = _repair_text(docstring)
        for line in repaired.splitlines():
            line = line.strip().strip('"')
            if line:
                first_line = line
                break
    if first_line:
        return first_line
    return SUMMARY_FALLBACK_NAMES.get(module_name, module_name.replace("_", " ").title())


def _derive_summary(module_name: str, optimization_type: OptimizationType, docstring: str) -> str:
    repaired = _repair_text(docstring) if docstring else ""
    lines = [line.strip(" -*\t") for line in repaired.splitlines() if line.strip()]
    generic_markers = ("参考开源项目", "参考开源论文", "灵感来源", "算法原理", "模块分类", "外部依赖")
    for line in lines[1:]:
        if any(marker in line for marker in generic_markers):
            continue
        if len(line) < 8:
            continue
        return line
    return TYPE_DESCRIPTION_TEMPLATES[optimization_type]


@lru_cache(maxsize=1)
def list_module_specs() -> Tuple[ModuleSpec, ...]:
    root = _repo_root()
    specs: List[ModuleSpec] = []
    module_files = sorted(root.glob("SpotZoom_Machine_Learning_v*/*.py"), key=_module_sort_key)
    for path in module_files:
        if path.name == "__init__.py":
            continue
        version = int(path.parent.name.rsplit("v", 1)[1])
        package = path.parent.name
        module_name = path.stem
        import_path = f"{package}.{module_name}"
        tree = _parse_module(path)
        docstring = ast.get_docstring(tree) or ""
        public_classes = _extract_public_classes(tree)
        public_functions = _extract_public_functions(tree)
        primary_symbol, config_symbol, result_symbols = _split_symbols(module_name, public_classes)
        optimization_type = MODULE_TYPE_OVERRIDES[module_name]
        specs.append(
            ModuleSpec(
                version=version,
                package=package,
                module_name=module_name,
                optimization_type=optimization_type,
                type_label=TYPE_LABELS[optimization_type],
                source_path=str(path.relative_to(root)).replace("\\", "/"),
                import_path=import_path,
                title=_derive_title(module_name, docstring),
                summary=_derive_summary(module_name, optimization_type, docstring),
                primary_symbol=primary_symbol,
                config_symbol=config_symbol,
                result_symbols=result_symbols,
                public_classes=public_classes,
                public_functions=public_functions,
            )
        )
    return tuple(specs)


def get_modules_by_version(version: int) -> Tuple[ModuleSpec, ...]:
    return tuple(spec for spec in list_module_specs() if spec.version == version)


def get_modules_by_type(optimization_type: OptimizationType | str) -> Tuple[ModuleSpec, ...]:
    if isinstance(optimization_type, str):
        optimization_type = OptimizationType(optimization_type)
    return tuple(spec for spec in list_module_specs() if spec.optimization_type == optimization_type)


def get_module_spec(version: int, module_name: str) -> ModuleSpec:
    for spec in list_module_specs():
        if spec.version == version and spec.module_name == module_name:
            return spec
    raise KeyError(f"未找到 v{version}/{module_name}")


def build_module_adapter(version: int, module_name: str) -> ModuleAdapter:
    return ModuleAdapter(get_module_spec(version, module_name))


def grouped_specs() -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {opt.value: [] for opt in OptimizationType}
    for spec in list_module_specs():
        groups[spec.optimization_type.value].append(spec.to_dict())
    return groups


def export_catalog_json(target_path: Path) -> Path:
    payload = {
        "module_count": len(list_module_specs()),
        "groups": grouped_specs(),
    }
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_path


def constructor_signature(version: int, module_name: str) -> str:
    adapter = build_module_adapter(version, module_name)
    if not adapter.spec.primary_symbol:
        return ""
    target = adapter.get_symbol(adapter.spec.primary_symbol)
    return str(inspect.signature(target))
