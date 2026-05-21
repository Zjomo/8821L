# v2 / data_driven_mpc

- 标题：数据驱动模型预测控制器 (Data-Driven Model Predictive Controller)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v2`
- 导入路径：`SpotZoom_Machine_Learning_v2.data_driven_mpc`
- 源文件：`SpotZoom_Machine_Learning_v2/data_driven_mpc.py`
- 统一主入口：`DataDrivenMPC`
- 配置类：`DataDrivenMPCConfig`
- 结果类：DataDrivenMPCResult
- 公共类：DataDrivenMPCConfig、DataDrivenMPCResult、DataDrivenMPC
- 公共函数：无
- 构造签名：`(config: 'Optional[DataDrivenMPCConfig]' = None)`

## 模块说明

leap-c (https://github.com/leap-c/leap-c): 学习型预测控制

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(2, "data_driven_mpc")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
