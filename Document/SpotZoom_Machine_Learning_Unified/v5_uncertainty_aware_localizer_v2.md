# v5 / uncertainty_aware_localizer_v2

- 标题：不确定性感知定位器 v2 (UncertaintyAwareLocalizerV2)
- 优化类型：检测与定位
- 版本目录：`SpotZoom_Machine_Learning_v5`
- 导入路径：`SpotZoom_Machine_Learning_v5.uncertainty_aware_localizer_v2`
- 源文件：`SpotZoom_Machine_Learning_v5/uncertainty_aware_localizer_v2.py`
- 统一主入口：`UncertaintyAwareLocalizerV2`
- 配置类：`LocalizerV2Config`
- 结果类：LocalizerV2Result
- 公共类：LocalizationMode、LocalizerV2Config、LocalizerV2Result、UncertaintyAwareLocalizerV2
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v5.uncertainty_aware_localizer_v2.LocalizerV2Config | None = None)`

## 模块说明

基于 DECODE (https://github.com/TuragaLab/DECODE) 的深度上下文依赖定位框架，

该模块用于光斑检测、候选筛选或定位精度提升。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(5, "uncertainty_aware_localizer_v2")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
