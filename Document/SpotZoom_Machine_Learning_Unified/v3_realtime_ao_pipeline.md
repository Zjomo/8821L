# v3 / realtime_ao_pipeline

- 标题：实时 AO 流水线 (Realtime Adaptive Optics Pipeline)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v3`
- 导入路径：`SpotZoom_Machine_Learning_v3.realtime_ao_pipeline`
- 源文件：`SpotZoom_Machine_Learning_v3/realtime_ao_pipeline.py`
- 统一主入口：`RealtimeAOPipeline`
- 配置类：`AOPipelineConfig`
- 结果类：AOPipelineState
- 公共类：AOPipelineConfig、AOPipelineState、RealtimeAOPipeline
- 公共函数：无
- 构造签名：`(config: 'Optional[AOPipelineConfig]' = None) -> 'None'`

## 模块说明

pyRTC (https://github.com/aodtech/pyRTC): Python 实时自适应光学控制框架

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(3, "realtime_ao_pipeline")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
