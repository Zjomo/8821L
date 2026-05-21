# v5 / realtime_pipeline_v2

- 标题：实时控制管线 v2 (RealtimePipelineV2)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v5`
- 导入路径：`SpotZoom_Machine_Learning_v5.realtime_pipeline_v2`
- 源文件：`SpotZoom_Machine_Learning_v5/realtime_pipeline_v2.py`
- 统一主入口：`RealtimePipelineV2`
- 配置类：`PipelineV2Config`
- 结果类：PipelineState、TelemetryPoint、PipelineV2Report
- 公共类：PipelineState、StageType、PipelineV2Config、TelemetryPoint、PipelineV2Report、PipelineStage、SensorStage、ProcessorStage、ControllerStage、CorrectorStage、RealtimePipelineV2
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v5.realtime_pipeline_v2.PipelineV2Config | None = None)`

## 模块说明

基于 pyRTC (https://github.com/jacotay7/pyRTC) 的模块化实时 AO 控制架构，

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(5, "realtime_pipeline_v2")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
