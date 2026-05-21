# v6 / bayesian_pid_optimizer

- 标题：Bayesian PID Optimizer - 贝叶斯 PID 参数优化器
- 优化类型：全局优化
- 版本目录：`SpotZoom_Machine_Learning_v6`
- 导入路径：`SpotZoom_Machine_Learning_v6.bayesian_pid_optimizer`
- 源文件：`SpotZoom_Machine_Learning_v6/bayesian_pid_optimizer.py`
- 统一主入口：`BayesianPIDOptimizer`
- 配置类：`BayesianPIDConfig`
- 结果类：BayesianPIDResult
- 公共类：BayesianPIDConfig、BayesianPIDResult、BayesianPIDOptimizer
- 公共函数：无
- 构造签名：`(config: 'Optional[BayesianPIDConfig]' = None)`

## 模块说明

Inspired by:

该模块用于参数搜索、全局优化或调参策略求解。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(6, "bayesian_pid_optimizer")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
