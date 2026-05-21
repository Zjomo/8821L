# v6 / convergence_guard

- 标题：Convergence Guard - 收敛保护器
- 优化类型：系统诊断
- 版本目录：`SpotZoom_Machine_Learning_v6`
- 导入路径：`SpotZoom_Machine_Learning_v6.convergence_guard`
- 源文件：`SpotZoom_Machine_Learning_v6/convergence_guard.py`
- 统一主入口：`ConvergenceGuard`
- 配置类：`ConvergenceGuardConfig`
- 结果类：ConvergenceReport
- 公共类：ConvergenceGuardConfig、ConvergenceReport、ConvergenceGuard
- 公共函数：无
- 构造签名：`(config: 'Optional[ConvergenceGuardConfig]' = None)`

## 模块说明

Inspired by:

该模块用于运行诊断、健康监控或稳定性保护。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(6, "convergence_guard")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
