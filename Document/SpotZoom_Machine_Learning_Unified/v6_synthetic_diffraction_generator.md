# v6 / synthetic_diffraction_generator

- 标题：Synthetic Diffraction Generator - 合成衍射数据生成器
- 优化类型：光束与光学仿真
- 版本目录：`SpotZoom_Machine_Learning_v6`
- 导入路径：`SpotZoom_Machine_Learning_v6.synthetic_diffraction_generator`
- 源文件：`SpotZoom_Machine_Learning_v6/synthetic_diffraction_generator.py`
- 统一主入口：`SyntheticDiffractionGenerator`
- 配置类：`DiffractionGeneratorConfig`
- 结果类：DiffractionSample
- 公共类：DiffractionGeneratorConfig、DiffractionSample、SyntheticDiffractionGenerator
- 公共函数：cv2_rotation_matrix、cv2_scale_matrix
- 构造签名：`(config: 'Optional[DiffractionGeneratorConfig]' = None)`

## 模块说明

Inspired by:

该模块用于光束传播、全息生成或数字孪生仿真。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(6, "synthetic_diffraction_generator")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
