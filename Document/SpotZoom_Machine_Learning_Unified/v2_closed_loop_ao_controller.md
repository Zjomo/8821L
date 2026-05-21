# v2 / closed_loop_ao_controller

- 标题：闭环自适应光学控制器 (Closed-Loop Adaptive Optics Controller)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v2`
- 导入路径：`SpotZoom_Machine_Learning_v2.closed_loop_ao_controller`
- 源文件：`SpotZoom_Machine_Learning_v2/closed_loop_ao_controller.py`
- 统一主入口：`ClosedLoopAOController`
- 配置类：`ClosedLoopAOConfig`
- 结果类：WFSMeasurement、ClosedLoopAOState
- 公共类：ClosedLoopAOConfig、WFSMeasurement、ClosedLoopAOState、ClosedLoopAOController
- 公共函数：无
- 构造签名：`(config: 'Optional[ClosedLoopAOConfig]' = None)`

## 模块说明

HCIPy (https://github.com/ehpor/hcipy): 自适应光学仿真框架

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(2, "closed_loop_ao_controller")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
