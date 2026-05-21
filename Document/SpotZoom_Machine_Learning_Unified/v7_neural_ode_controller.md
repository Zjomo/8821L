# v7 / neural_ode_controller

- 标题：神经 ODE 控制器 (Neural ODE Controller)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v7`
- 导入路径：`SpotZoom_Machine_Learning_v7.neural_ode_controller`
- 源文件：`SpotZoom_Machine_Learning_v7/neural_ode_controller.py`
- 统一主入口：`NeuralODEController`
- 配置类：`NeuralODEConfig`
- 结果类：NeuralODEResult
- 公共类：ODESolver、NeuralODEResult、NeuralODEConfig、NeuralODEController
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v7.neural_ode_controller.NeuralODEConfig | None = None)`

## 模块说明

基于神经常微分方程的连续时间控制信号生成模块。

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(7, "neural_ode_controller")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
