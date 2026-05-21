# v5 / sensorless_ao_v2

- 标题：无传感器自适应光学 v2 (SensorlessAOV2)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v5`
- 导入路径：`SpotZoom_Machine_Learning_v5.sensorless_ao_v2`
- 源文件：`SpotZoom_Machine_Learning_v5/sensorless_ao_v2.py`
- 统一主入口：`SensorlessAOV2`
- 配置类：`SensorlessAOV2Config`
- 结果类：SensorlessAOV2Report
- 公共类：QualityMetric、SearchStrategy、SensorlessAOV2Config、SensorlessAOV2Report、SensorlessAOV2
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v5.sensorless_ao_v2.SensorlessAOV2Config | None = None)`

## 模块说明

基于 REALM (https://github.com/MSiemons/REALM) 的无波前传感器 Zernike 模态校正，

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(5, "sensorless_ao_v2")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
