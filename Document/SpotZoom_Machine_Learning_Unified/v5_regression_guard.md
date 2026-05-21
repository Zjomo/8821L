# v5 / regression_guard

- 标题：回归防护器 (RegressionGuard)
- 优化类型：系统诊断
- 版本目录：`SpotZoom_Machine_Learning_v5`
- 导入路径：`SpotZoom_Machine_Learning_v5.regression_guard`
- 源文件：`SpotZoom_Machine_Learning_v5/regression_guard.py`
- 统一主入口：`RegressionGuard`
- 配置类：`RegressionGuardConfig`
- 结果类：RegressionGuardReport
- 公共类：CheckCategory、Severity、RegressionGuardConfig、RegressionIssue、RegressionGuardReport、RegressionGuard
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v5.regression_guard.RegressionGuardConfig | None = None)`

## 模块说明

为 SpotZoom 提供自动化的回归检测与防护机制，确保代码变更不会

该模块用于运行诊断、健康监控或稳定性保护。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(5, "regression_guard")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
