# v6 / diffusion_spot_enhancer

- 标题：Diffusion Spot Enhancer - 扩散模型光斑增强器
- 优化类型：图像增强
- 版本目录：`SpotZoom_Machine_Learning_v6`
- 导入路径：`SpotZoom_Machine_Learning_v6.diffusion_spot_enhancer`
- 源文件：`SpotZoom_Machine_Learning_v6/diffusion_spot_enhancer.py`
- 统一主入口：`DiffusionSpotEnhancer`
- 配置类：`DiffusionEnhancerConfig`
- 结果类：DiffusionEnhanceResult
- 公共类：DiffusionEnhancerConfig、DiffusionEnhanceResult、DiffusionSpotEnhancer
- 公共函数：无
- 构造签名：`(config: 'Optional[DiffusionEnhancerConfig]' = None)`

## 模块说明

Inspired by:

该模块用于提升图像质量、抑制噪声或增强光斑可分辨性。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(6, "diffusion_spot_enhancer")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
