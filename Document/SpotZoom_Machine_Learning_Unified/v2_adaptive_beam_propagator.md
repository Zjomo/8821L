# v2 / adaptive_beam_propagator

- 标题：自适应光束传播模拟器 (Adaptive Beam Propagator)
- 优化类型：光束与光学仿真
- 版本目录：`SpotZoom_Machine_Learning_v2`
- 导入路径：`SpotZoom_Machine_Learning_v2.adaptive_beam_propagator`
- 源文件：`SpotZoom_Machine_Learning_v2/adaptive_beam_propagator.py`
- 统一主入口：`AdaptiveBeamPropagator`
- 配置类：`PropagationConfig`
- 结果类：BeamPropagationResult
- 公共类：PropagationConfig、BeamPropagationResult、AdaptiveBeamPropagator
- 公共函数：无
- 构造签名：`(config: 'Optional[PropagationConfig]' = None)`

## 模块说明

OpenCLAW (https://github.com/openclaw): 自适应波模拟计算库

该模块用于光束传播、全息生成或数字孪生仿真。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(2, "adaptive_beam_propagator")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
