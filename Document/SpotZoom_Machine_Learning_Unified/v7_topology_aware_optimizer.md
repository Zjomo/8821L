# v7 / topology_aware_optimizer

- 标题：拓扑感知优化器 (Topology Aware Optimizer)
- 优化类型：全局优化
- 版本目录：`SpotZoom_Machine_Learning_v7`
- 导入路径：`SpotZoom_Machine_Learning_v7.topology_aware_optimizer`
- 源文件：`SpotZoom_Machine_Learning_v7/topology_aware_optimizer.py`
- 统一主入口：`TopologyAwareOptimizer`
- 配置类：`TopologyOptimizerConfig`
- 结果类：ParetoPoint、TopologyOptResult
- 公共类：RestartStrategy、ConstraintMethod、ParetoPoint、TopologyOptResult、TopologyOptimizerConfig、TopologyAwareOptimizer
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v7.topology_aware_optimizer.TopologyOptimizerConfig | None = None)`

## 模块说明

基于 CMA-ES (协方差矩阵自适应进化策略) 的全局优化模块。

该模块用于参数搜索、全局优化或调参策略求解。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(7, "topology_aware_optimizer")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
