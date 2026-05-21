# v2 / deep_image_prior_enhancer

- 标题：零样本光斑图像增强器 (Deep Image Prior-inspired Spot Enhancer)
- 优化类型：图像增强
- 版本目录：`SpotZoom_Machine_Learning_v2`
- 导入路径：`SpotZoom_Machine_Learning_v2.deep_image_prior_enhancer`
- 源文件：`SpotZoom_Machine_Learning_v2/deep_image_prior_enhancer.py`
- 统一主入口：`DeepImagePriorEnhancer`
- 配置类：`无`
- 结果类：DIPEnhancementResult
- 公共类：DIPEnhancementResult、DeepImagePriorEnhancer
- 公共函数：无
- 构造签名：`(max_iterations: 'int' = 80, learning_rate: 'float' = 0.01, num_filters: 'int' = 5, filter_size: 'int' = 5, noise_std: 'float' = 0.05, early_stop_patience: 'int' = 10, energy_conservation_weight: 'float' = 0.1, smoothness_weight: 'float' = 0.05, multi_scale: 'bool' = True)`

## 模块说明

Deep Image Prior (https://github.com/DmitryUlyanov/deep-image-prior)

该模块用于提升图像质量、抑制噪声或增强光斑可分辨性。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(2, "deep_image_prior_enhancer")
instance = adapter.build_instance()
```
