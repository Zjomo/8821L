# v3 / slm_holographic_spot_generator

- 标题：SLM 全息光斑生成器 (SLM Holographic Spot Generator)
- 优化类型：光束与光学仿真
- 版本目录：`SpotZoom_Machine_Learning_v3`
- 导入路径：`SpotZoom_Machine_Learning_v3.slm_holographic_spot_generator`
- 源文件：`SpotZoom_Machine_Learning_v3/slm_holographic_spot_generator.py`
- 统一主入口：`SLMHolographicGenerator`
- 配置类：`SLMConfig`
- 结果类：HologramResult
- 公共类：SLMConfig、HologramResult、SLMHolographicGenerator
- 公共函数：无
- 构造签名：`(config: 'Optional[SLMConfig]' = None) -> 'None'`

## 模块说明

slmsuite (https://github.com/wavefrontshaping/slm_suite): SLM 控制与全息图生成框架

该模块用于光束传播、全息生成或数字孪生仿真。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(3, "slm_holographic_spot_generator")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
