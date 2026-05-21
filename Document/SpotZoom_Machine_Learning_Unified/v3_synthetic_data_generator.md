# v3 / synthetic_data_generator

- 标题：合成数据生成器 (Synthetic Data Generator)
- 优化类型：光束与光学仿真
- 版本目录：`SpotZoom_Machine_Learning_v3`
- 导入路径：`SpotZoom_Machine_Learning_v3.synthetic_data_generator`
- 源文件：`SpotZoom_Machine_Learning_v3/synthetic_data_generator.py`
- 统一主入口：`SyntheticDataGenerator`
- 配置类：`SyntheticDataConfig`
- 结果类：SyntheticSample
- 公共类：SyntheticDataConfig、SyntheticSample、SyntheticDataGenerator
- 公共函数：无
- 构造签名：`(config: 'Optional[SyntheticDataConfig]' = None) -> 'None'`

## 模块说明

DeepTrack2 (https://github.com/DeepTrack/DeepTrack2): 深度学习光学粒子追踪

该模块用于光束传播、全息生成或数字孪生仿真。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(3, "synthetic_data_generator")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
