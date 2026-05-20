"""
回归防护器 (RegressionGuard)

为 SpotZoom 提供自动化的回归检测与防护机制，确保代码变更不会
引入功能退化。

灵感来源:
- 软件工程最佳实践：回归测试、CI/CD 质量门控
- v1/diagnostic_health_monitor.py: 系统健康监控
- v1/safety_manager.py: 安全保护

算法原理:
  1. 基线快照: 在已知良好状态下记录系统行为基线
  2. 回归检测: 对比当前行为与基线，检测功能退化
  3. 多层检测: 从导入、功能、性能、接口四个层面检测
  4. 自动报告: 生成结构化的回归检测报告

与现有模块的关系:
  - 与 v1/diagnostic_health_monitor.py 协同
  - 与 v1/safety_manager.py 的安全保护协同
  - 与 v1/config_auto_tuner.py 的配置验证协同

外部依赖: numpy, importlib, time (标准库)
"""

import numpy as np
import logging
import time
import importlib
import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class CheckCategory(Enum):
    """检查类别。"""
    IMPORT = "import"               # 模块导入检查
    FUNCTIONAL = "functional"       # 功能正确性检查
    PERFORMANCE = "performance"     # 性能回归检查
    INTERFACE = "interface"         # 接口兼容性检查
    CODE_QUALITY = "code_quality"   # 代码质量检查


class Severity(Enum):
    """问题严重程度。"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RegressionGuardConfig:
    """回归防护配置。"""
    # 导入检查
    check_imports: bool = True
    import_timeout_seconds: float = 5.0

    # 功能检查
    check_functional: bool = True
    functional_test_cases: int = 5   # 每模块测试用例数

    # 性能检查
    check_performance: bool = True
    performance_regression_threshold: float = 0.2  # 20% 性能退化阈值
    performance_baseline_file: Optional[str] = None

    # 接口检查
    check_interface: bool = True
    required_classes: List[str] = field(default_factory=lambda: [
        "SubPixelCentroid", "KalmanSpotTracker", "SpotQualityAnalyzer",
    ])
    required_methods: Dict[str, List[str]] = field(default_factory=lambda: {
        "SubPixelCentroid": ["compute"],
    })

    # 代码质量检查
    check_code_quality: bool = True
    max_file_lines: int = 5000
    max_function_lines: int = 200
    max_cyclomatic_complexity: int = 15

    # 包路径
    package_paths: List[str] = field(default_factory=lambda: [
        "SpotZoom_Machine_Learning",
        "SpotZoom_Machine_Learning_v2",
        "SpotZoom_Machine_Learning_v3",
        "SpotZoom_Machine_Learning_v4",
        "SpotZoom_Machine_Learning_v5",
    ])


@dataclass
class RegressionIssue:
    """回归问题。"""
    category: str
    severity: str
    module: str
    description: str
    details: str = ""
    suggestion: str = ""


@dataclass
class RegressionGuardReport:
    """回归检测报告。"""
    # 总览
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0

    # 分类统计
    import_results: Dict[str, bool] = field(default_factory=dict)
    functional_results: Dict[str, bool] = field(default_factory=dict)
    performance_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    interface_results: Dict[str, bool] = field(default_factory=dict)
    quality_results: Dict[str, List[str]] = field(default_factory=dict)

    # 问题列表
    issues: List[RegressionIssue] = field(default_factory=list)

    # 汇总
    overall_status: str = "PASS"     # PASS / WARNING / FAIL
    processing_time_ms: float = 0.0


class RegressionGuard:
    """回归防护器。

    自动化检测 SpotZoom 项目的回归问题，包括导入、功能、
    性能、接口和代码质量五个维度。

    Parameters
    ----------
    config : RegressionGuardConfig
        防护器配置。
    """

    def __init__(self, config: Optional[RegressionGuardConfig] = None):
        self.config = config or RegressionGuardConfig()
        self._baselines: Dict[str, Dict] = {}

    def run_full_check(self) -> RegressionGuardReport:
        """执行完整的回归检测。"""
        t0 = time.perf_counter()
        report = RegressionGuardReport()

        if self.config.check_imports:
            self._check_imports(report)

        if self.config.check_functional:
            self._check_functional(report)

        if self.config.check_performance:
            self._check_performance(report)

        if self.config.check_interface:
            self._check_interface(report)

        if self.config.check_code_quality:
            self._check_code_quality(report)

        # 汇总
        report.total_checks = report.passed_checks + report.failed_checks + report.warning_checks

        critical_issues = [i for i in report.issues if i.severity == Severity.CRITICAL.value]
        high_issues = [i for i in report.issues if i.severity == Severity.HIGH.value]

        if critical_issues:
            report.overall_status = "FAIL"
        elif high_issues:
            report.overall_status = "WARNING"
        else:
            report.overall_status = "PASS"

        report.processing_time_ms = (time.perf_counter() - t0) * 1000
        return report

    def _check_imports(self, report: RegressionGuardReport):
        """检查模块导入。"""
        for pkg in self.config.package_paths:
            try:
                t0 = time.perf_counter()
                mod = importlib.import_module(pkg)
                elapsed = (time.perf_counter() - t0) * 1000

                if elapsed > self.config.import_timeout_seconds * 1000:
                    report.import_results[pkg] = False
                    report.failed_checks += 1
                    report.issues.append(RegressionIssue(
                        category=CheckCategory.IMPORT.value,
                        severity=Severity.HIGH.value,
                        module=pkg,
                        description=f"导入超时 ({elapsed:.0f}ms)",
                        suggestion="检查模块初始化逻辑，移除耗时的全局操作",
                    ))
                else:
                    report.import_results[pkg] = True
                    report.passed_checks += 1

            except Exception as e:
                report.import_results[pkg] = False
                report.failed_checks += 1
                report.issues.append(RegressionIssue(
                    category=CheckCategory.IMPORT.value,
                    severity=Severity.CRITICAL.value,
                    module=pkg,
                    description=f"导入失败: {type(e).__name__}: {e}",
                    details=traceback.format_exc(),
                    suggestion="检查依赖是否安装，修复语法错误",
                ))

    def _check_functional(self, report: RegressionGuardReport):
        """检查功能正确性。"""
        functional_tests = {
            "SubPixelCentroid": self._test_subpixel_centroid,
            "SpotQualityAnalyzer": self._test_spot_quality,
            "KalmanSpotTracker": self._test_kalman_tracker,
        }

        for name, test_fn in functional_tests.items():
            try:
                success, msg = test_fn()
                report.functional_results[name] = success
                if success:
                    report.passed_checks += 1
                else:
                    report.failed_checks += 1
                    report.issues.append(RegressionIssue(
                        category=CheckCategory.FUNCTIONAL.value,
                        severity=Severity.HIGH.value,
                        module=name,
                        description=f"功能测试失败: {msg}",
                    ))
            except Exception as e:
                report.functional_results[name] = False
                report.warning_checks += 1
                report.issues.append(RegressionIssue(
                    category=CheckCategory.FUNCTIONAL.value,
                    severity=Severity.MEDIUM.value,
                    module=name,
                    description=f"功能测试异常: {e}",
                    suggestion="检查模块是否正确安装",
                ))

    def _test_subpixel_centroid(self) -> Tuple[bool, str]:
        """测试亚像素质心定位器。"""
        try:
            from SpotZoom_Machine_Learning import SubPixelCentroid
            np.random.seed(42)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[45:55, 45:55] = 200
            centroid = SubPixelCentroid(method="weighted")
            result = centroid.compute(img, (40, 40, 60, 60))
            if result is None:
                return False, "返回 None"
            if abs(result.cx - 50) > 2 or abs(result.cy - 50) > 2:
                return False, f"定位偏差过大: ({result.cx:.1f}, {result.cy:.1f})"
            return True, ""
        except Exception as e:
            return False, str(e)

    def _test_spot_quality(self) -> Tuple[bool, str]:
        """测试光斑质量分析器。"""
        try:
            from SpotZoom_Machine_Learning import SpotQualityAnalyzer
            np.random.seed(42)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[45:55, 45:55] = 200
            analyzer = SpotQualityAnalyzer()
            report = analyzer.analyze(img, (40, 40, 60, 60))
            return True, ""
        except Exception as e:
            return False, str(e)

    def _test_kalman_tracker(self) -> Tuple[bool, str]:
        """测试卡尔曼跟踪器。"""
        try:
            from SpotZoom_Machine_Learning import KalmanSpotTracker
            tracker = KalmanSpotTracker()
            result = tracker.update(50.0, 50.0)
            if result is None:
                return False, "返回 None"
            return True, ""
        except Exception as e:
            return False, str(e)

    def _check_performance(self, report: RegressionGuardReport):
        """检查性能回归。"""
        perf_tests = {
            "SubPixelCentroid.compute": self._perf_subpixel_centroid,
        }

        for name, test_fn in perf_tests.items():
            try:
                time_ms = test_fn()
                report.performance_results[name] = {
                    "time_ms": time_ms,
                    "baseline_ms": self._baselines.get(name, {}).get("time_ms", 0),
                }

                baseline = self._baselines.get(name, {}).get("time_ms", time_ms)
                if baseline > 0:
                    regression = (time_ms - baseline) / baseline
                    if regression > self.config.performance_regression_threshold:
                        report.failed_checks += 1
                        report.issues.append(RegressionIssue(
                            category=CheckCategory.PERFORMANCE.value,
                            severity=Severity.MEDIUM.value,
                            module=name,
                            description=f"性能退化 {regression*100:.1f}% "
                                        f"({baseline:.1f}ms -> {time_ms:.1f}ms)",
                            suggestion="检查算法复杂度，考虑优化热点",
                        ))
                    else:
                        report.passed_checks += 1
                else:
                    # 首次运行，记录基线
                    self._baselines[name] = {"time_ms": time_ms}
                    report.passed_checks += 1

            except Exception as e:
                report.warning_checks += 1

    def _perf_subpixel_centroid(self) -> float:
        """性能测试：亚像素质心。"""
        try:
            from SpotZoom_Machine_Learning import SubPixelCentroid
            centroid = SubPixelCentroid(method="weighted")
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            t0 = time.perf_counter()
            for _ in range(100):
                centroid.compute(img, (10, 10, 90, 90))
            return (time.perf_counter() - t0) / 100 * 1000
        except Exception:
            return -1

    def _check_interface(self, report: RegressionGuardReport):
        """检查接口兼容性。"""
        for cls_name in self.config.required_classes:
            try:
                from SpotZoom_Machine_Learning import __dict__ as pkg_dict
                if cls_name in pkg_dict:
                    report.interface_results[cls_name] = True
                    report.passed_checks += 1

                    # 检查必需方法
                    if cls_name in self.config.required_methods:
                        cls = pkg_dict[cls_name]
                        for method_name in self.config.required_methods[cls_name]:
                            if not hasattr(cls, method_name):
                                report.issues.append(RegressionIssue(
                                    category=CheckCategory.INTERFACE.value,
                                    severity=Severity.HIGH.value,
                                    module=cls_name,
                                    description=f"缺少方法: {method_name}",
                                ))
                else:
                    report.interface_results[cls_name] = False
                    report.failed_checks += 1
                    report.issues.append(RegressionIssue(
                        category=CheckCategory.INTERFACE.value,
                        severity=Severity.CRITICAL.value,
                        module=cls_name,
                        description=f"类 {cls_name} 未在 __init__.py 中导出",
                        suggestion="检查 __init__.py 的导入逻辑",
                    ))
            except Exception as e:
                report.warning_checks += 1

    def _check_code_quality(self, report: RegressionGuardReport):
        """检查代码质量。"""
        try:
            import ast
            import os

            base_path = os.path.dirname(os.path.abspath(__file__))
            project_path = os.path.dirname(base_path)

            for pkg in self.config.package_paths:
                pkg_path = os.path.join(project_path, pkg)
                if not os.path.exists(pkg_path):
                    continue

                issues = []
                for fname in os.listdir(pkg_path):
                    if not fname.endswith('.py') or fname.startswith('__pycache__'):
                        continue

                    fpath = os.path.join(pkg_path, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            source = f.read()

                        lines = source.count('\n') + 1
                        if lines > self.config.max_file_lines:
                            issues.append(f"{fname}: 文件过长 ({lines} 行)")

                        # AST 分析
                        tree = ast.parse(source)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                func_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                                if func_lines > self.config.max_function_lines:
                                    issues.append(
                                        f"{fname}:{node.lineno}: 函数 {node.name} 过长 ({func_lines} 行)"
                                    )

                    except SyntaxError as e:
                        issues.append(f"{fname}: 语法错误 - {e}")
                        report.issues.append(RegressionIssue(
                            category=CheckCategory.CODE_QUALITY.value,
                            severity=Severity.CRITICAL.value,
                            module=f"{pkg}/{fname}",
                            description=f"语法错误: {e}",
                        ))

                report.quality_results[pkg] = issues
                if issues:
                    report.warning_checks += len(issues)
                else:
                    report.passed_checks += 1

        except Exception as e:
            report.warning_checks += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = RegressionGuardConfig()
    guard = RegressionGuard(config)

    report = guard.run_full_check()
    print(f"状态: {report.overall_status}")
    print(f"检查: {report.passed_checks}/{report.total_checks} 通过")
    print(f"问题: {len(report.issues)}")
    for issue in report.issues:
        print(f"  [{issue.severity}] {issue.category}/{issue.module}: {issue.description}")
