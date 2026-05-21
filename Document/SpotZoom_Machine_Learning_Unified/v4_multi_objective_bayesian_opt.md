# v4 / multi_objective_bayesian_opt

- 标题：多目标贝叶斯优化器 (Multi-Objective Bayesian Optimizer)
- 优化类型：全局优化
- 版本目录：`SpotZoom_Machine_Learning_v4`
- 导入路径：`SpotZoom_Machine_Learning_v4.multi_objective_bayesian_opt`
- 源文件：`SpotZoom_Machine_Learning_v4/multi_objective_bayesian_opt.py`
- 统一主入口：`MultiObjectiveBayesianOpt`
- 配置类：`BayesianOptConfig`
- 结果类：OptimizationResult
- 公共类：AcquisitionFunction、KernelType、BayesianOptConfig、OptimizationResult、ParetoFront、MultiObjectiveBayesianOpt
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v4.multi_objective_bayesian_opt.BayesianOptConfig | None = None)`

## 模块说明

Optuna (https://github.com/optuna/optuna)

该模块用于参数搜索、全局优化或调参策略求解。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(4, "multi_objective_bayesian_opt")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
