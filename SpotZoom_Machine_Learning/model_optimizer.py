"""
YOLO 模型推理优化器 (ModelInferenceOptimizer)

灵感来源:
- Ultralytics YOLO (https://github.com/ultralytics/ultralytics) — 一键导出 TensorRT/ONNX
- BoxMOT (https://github.com/mikel-brostrom/boxmot) — 多跟踪器统一接口
- TensorRT (NVIDIA) — GPU 推理加速 3-5x
- ONNX Runtime — 跨平台推理引擎

算法原理:
- Model Export — 模型格式转换 (PyTorch -> ONNX -> TensorRT)
- Half-Precision (FP16) — 半精度推理减少延迟
- Batch Inference — 批量推理提升吞吐
- Warm-up Cache — 预热推理缓存减少首次延迟
- Inference Profiling — 推理性能分析

功能:
- 自动检测最优推理后端 (TensorRT > ONNX > PyTorch)
- 推理性能分析与基准测试
- 模型格式转换辅助工具
- 推理缓存预热
- 内存与延迟监控

依赖: numpy, time (可选: onnxruntime, tensorrt)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("SpotZoom.ModelOptimizer")


@dataclass
class InferenceProfile:
    """推理性能分析报告。"""
    backend: str  # 推理后端名称
    model_format: str  # 模型格式
    avg_latency_ms: float  # 平均推理延迟 (毫秒)
    min_latency_ms: float  # 最小延迟
    max_latency_ms: float  # 最大延迟
    p50_latency_ms: float  # P50 延迟
    p95_latency_ms: float  # P95 延迟
    p99_latency_ms: float  # P99 延迟
    throughput_fps: float  # 吞吐量 (FPS)
    warmup_latency_ms: float  # 预热延迟
    total_samples: int  # 总推理次数
    memory_estimate_mb: float  # 估计内存占用 (MB)
    is_optimal: bool  # 是否为最优后端


@dataclass
class BackendInfo:
    """推理后端信息。"""
    name: str
    available: bool
    priority: int  # 优先级 (越小越优先)
    format_hint: str  # 推荐模型格式


class ModelInferenceOptimizer:
    """YOLO 模型推理优化器。

    提供推理后端自动选择、性能分析和优化建议。

    Parameters
    ----------
    warmup_iterations : int
        预热迭代次数。
    profile_iterations : int
        性能分析迭代次数。
    target_latency_ms : float
        目标推理延迟 (毫秒)。
    """

    def __init__(
        self,
        warmup_iterations: int = 3,
        profile_iterations: int = 20,
        target_latency_ms: float = 50.0,
    ):
        self.warmup_iterations = int(warmup_iterations)
        self.profile_iterations = int(profile_iterations)
        self.target_latency_ms = float(target_latency_ms)

        self._available_backends: List[BackendInfo] = []
        self._detect_backends()

        self._last_profile: Optional[InferenceProfile] = None
        self._inference_count: int = 0
        self._total_inference_time_s: float = 0.0

    def _detect_backends(self) -> None:
        """检测可用的推理后端。"""
        self._available_backends = []

        # 1. TensorRT (最高优先级)
        trt_available = self._check_import("tensorrt")
        self._available_backends.append(BackendInfo(
            name="TensorRT",
            available=trt_available,
            priority=1,
            format_hint="engine",
        ))

        # 2. ONNX Runtime
        onnx_available = self._check_import("onnxruntime")
        self._available_backends.append(BackendInfo(
            name="ONNX Runtime",
            available=onnx_available,
            priority=2,
            format_hint="onnx",
        ))

        # 3. OpenVINO
        ov_available = self._check_import("openvino")
        self._available_backends.append(BackendInfo(
            name="OpenVINO",
            available=ov_available,
            priority=3,
            format_hint="openvino_model",
        ))

        # 4. PyTorch (Ultralytics 默认)
        self._available_backends.append(BackendInfo(
            name="PyTorch (Ultralytics)",
            available=True,  # 总是可用
            priority=4,
            format_hint="pt",
        ))

        available = [b for b in self._available_backends if b.available]
        LOGGER.info(
            "Inference backends detected: %s",
            ", ".join(f"{b.name}({'✓' if b.available else '✗'})" for b in self._available_backends),
        )
        if available:
            LOGGER.info("Recommended backend: %s", available[0].name)

    @staticmethod
    def _check_import(module_name: str) -> bool:
        """检查模块是否可导入。"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def get_best_backend(self) -> BackendInfo:
        """获取最优可用推理后端。"""
        available = [b for b in self._available_backends if b.available]
        if not available:
            return BackendInfo(name="None", available=False, priority=99, format_hint="pt")
        return available[0]

    def get_optimization_suggestions(self) -> List[str]:
        """获取模型推理优化建议。

        Returns
        -------
        List[str]
            优化建议列表。
        """
        suggestions = []
        available_names = {b.name for b in self._available_backends if b.available}

        if "TensorRT" not in available_names:
            suggestions.append(
                "[高优先级] 安装 TensorRT 可获得 3-5x 推理加速。"
                "执行: pip install tensorrt, 然后使用 model.export(format='engine')"
            )

        if "ONNX Runtime" not in available_names:
            suggestions.append(
                "[中优先级] 安装 ONNX Runtime 可获得 2-3x 推理加速。"
                "执行: pip install onnxruntime-gpu"
            )

        if self._last_profile is not None:
            if self._last_profile.avg_latency_ms > self.target_latency_ms:
                suggestions.append(
                    f"[性能] 当前平均延迟 {self._last_profile.avg_latency_ms:.1f}ms "
                    f"超过目标 {self.target_latency_ms:.1f}ms，建议切换到更快的推理后端"
                )
            if self._last_profile.p99_latency_ms > 3 * self._last_profile.avg_latency_ms:
                suggestions.append(
                    "[稳定性] P99 延迟显著高于平均值，可能存在 GC 或资源竞争"
                )

        if not suggestions:
            suggestions.append("当前推理配置良好，无需额外优化")

        return suggestions

    def profile_inference(
        self,
        detector: Any,
        frame: np.ndarray,
    ) -> Optional[InferenceProfile]:
        """对检测器进行推理性能分析。

        Parameters
        ----------
        detector : Any
            具有 detect(frame) 方法的检测器对象。
        frame : np.ndarray
            测试用图像帧。

        Returns
        -------
        Optional[InferenceProfile]
            推理性能报告。如果分析失败返回 None。
        """
        # 确定后端名称
        backend_name = "PyTorch (Ultralytics)"
        model_format = "pt"
        if hasattr(detector, "model"):
            model = detector.model
            if hasattr(model, "predictor") and hasattr(model.predictor, "backend"):
                backend_name = str(getattr(model.predictor, 'backend', backend_name))
            task = getattr(model, 'task', None)
            if task:
                model_format = "pt"

        # 预热
        warmup_times = []
        for i in range(self.warmup_iterations):
            t0 = time.perf_counter()
            try:
                detector.detect(frame)
            except Exception:
                pass
            warmup_times.append((time.perf_counter() - t0) * 1000)

        warmup_latency = float(np.mean(warmup_times)) if warmup_times else 0.0

        # 性能分析
        latencies = []
        for _ in range(self.profile_iterations):
            t0 = time.perf_counter()
            try:
                detector.detect(frame)
            except Exception:
                continue
            latencies.append((time.perf_counter() - t0) * 1000)

        if not latencies:
            LOGGER.warning("Inference profiling failed: no successful detections")
            return None

        latencies_arr = np.array(latencies)
        total_time = float(np.sum(latencies_arr))

        profile = InferenceProfile(
            backend=backend_name,
            model_format=model_format,
            avg_latency_ms=round(float(np.mean(latencies_arr)), 3),
            min_latency_ms=round(float(np.min(latencies_arr)), 3),
            max_latency_ms=round(float(np.max(latencies_arr)), 3),
            p50_latency_ms=round(float(np.percentile(latencies_arr, 50)), 3),
            p95_latency_ms=round(float(np.percentile(latencies_arr, 95)), 3),
            p99_latency_ms=round(float(np.percentile(latencies_arr, 99)), 3),
            throughput_fps=round(len(latencies) / max(total_time / 1000.0, 0.001), 1),
            warmup_latency_ms=round(warmup_latency, 3),
            total_samples=len(latencies),
            memory_estimate_mb=0.0,  # 需要特定后端 API
            is_optimal=float(np.mean(latencies_arr)) <= self.target_latency_ms,
        )

        self._last_profile = profile
        LOGGER.info(
            "Inference profile: backend=%s, avg=%.1fms, P95=%.1fms, throughput=%.0f FPS",
            profile.backend, profile.avg_latency_ms, profile.p95_latency_ms,
            profile.throughput_fps,
        )

        return profile

    def record_inference(self, latency_s: float) -> None:
        """记录一次推理延迟。"""
        self._inference_count += 1
        self._total_inference_time_s += latency_s

    def get_runtime_stats(self) -> Dict[str, Any]:
        """获取运行时推理统计。"""
        avg_ms = (self._total_inference_time_s / max(self._inference_count, 1)) * 1000
        return {
            "total_inferences": self._inference_count,
            "total_time_s": round(self._total_inference_time_s, 3),
            "avg_latency_ms": round(avg_ms, 3),
            "current_backend": self.get_best_backend().name,
            "last_profile": {
                "avg_ms": self._last_profile.avg_latency_ms if self._last_profile else None,
                "p95_ms": self._last_profile.p95_latency_ms if self._last_profile else None,
                "fps": self._last_profile.throughput_fps if self._last_profile else None,
            } if self._last_profile else None,
        }

    def reset(self) -> None:
        """重置统计。"""
        self._inference_count = 0
        self._total_inference_time_s = 0.0
        self._last_profile = None
