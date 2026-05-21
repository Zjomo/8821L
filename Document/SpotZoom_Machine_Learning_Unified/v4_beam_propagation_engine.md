# v4 / beam_propagation_engine

- 标题：光束传播物理仿真引擎 (Beam Propagation Physics Engine)
- 优化类型：光束与光学仿真
- 版本目录：`SpotZoom_Machine_Learning_v4`
- 导入路径：`SpotZoom_Machine_Learning_v4.beam_propagation_engine`
- 源文件：`SpotZoom_Machine_Learning_v4/beam_propagation_engine.py`
- 统一主入口：`BeamPropagationEngine`
- 配置类：`无`
- 结果类：PropagationResult
- 公共类：PropagationMethod、BeamType、OpticalElement、ThinLens、CircularAperture、ZernikeAberration、PropagationScene、PropagationResult、BeamPropagationEngine
- 公共函数：无
- 构造签名：`()`

## 模块说明

HCIPy (https://github.com/ehpor/hcipy)

该模块用于光束传播、全息生成或数字孪生仿真。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(4, "beam_propagation_engine")
instance = adapter.build_instance()
```
