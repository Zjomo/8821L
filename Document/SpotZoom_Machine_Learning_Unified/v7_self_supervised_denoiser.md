# v7 / self_supervised_denoiser

- 标题：自监督图像降噪器 (Self-Supervised Denoiser)
- 优化类型：图像增强
- 版本目录：`SpotZoom_Machine_Learning_v7`
- 导入路径：`SpotZoom_Machine_Learning_v7.self_supervised_denoiser`
- 源文件：`SpotZoom_Machine_Learning_v7/self_supervised_denoiser.py`
- 统一主入口：`SelfSupervisedDenoiser`
- 配置类：`DenoiserConfig`
- 结果类：DenoiseReport
- 公共类：NoiseEstimationMethod、DenoiseReport、DenoiserConfig、SelfSupervisedDenoiser
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v7.self_supervised_denoiser.DenoiserConfig | None = None)`

## 模块说明

无需干净参考数据的自监督图像降噪模块。基于盲点网络 (Blind-Spot Network)

该模块用于提升图像质量、抑制噪声或增强光斑可分辨性。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(7, "self_supervised_denoiser")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
