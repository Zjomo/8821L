# v4 / digital_twin_enhancer

- 标题：光学系统数字孪生增强器 (Optical System Digital Twin Enhancer)
- 优化类型：光束与光学仿真
- 版本目录：`SpotZoom_Machine_Learning_v4`
- 导入路径：`SpotZoom_Machine_Learning_v4.digital_twin_enhancer`
- 源文件：`SpotZoom_Machine_Learning_v4/digital_twin_enhancer.py`
- 统一主入口：`DigitalTwinEnhancer`
- 配置类：`TwinConfig`
- 结果类：TwinState、PredictionResult
- 公共类：TwinComponent、PredictionHorizon、TwinConfig、TwinState、PredictionResult、DigitalTwinEnhancer
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v4.digital_twin_enhancer.TwinConfig | None = None)`

## 模块说明

Azure Digital Twins

该模块用于光束传播、全息生成或数字孪生仿真。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(4, "digital_twin_enhancer")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
