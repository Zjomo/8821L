# v5 / self_supervised_denoiser_v2

- 标题：自监督去噪器 v2 (SelfSupervisedDenoiserV2)
- 优化类型：图像增强
- 版本目录：`SpotZoom_Machine_Learning_v5`
- 导入路径：`SpotZoom_Machine_Learning_v5.self_supervised_denoiser_v2`
- 源文件：`SpotZoom_Machine_Learning_v5/self_supervised_denoiser_v2.py`
- 统一主入口：`SelfSupervisedDenoiserV2`
- 配置类：`DenoiserV2Config`
- 结果类：DenoiserV2Report
- 公共类：DenoiseMode、DenoiserV2Config、DenoiserV2Report、SelfSupervisedDenoiserV2
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v5.self_supervised_denoiser_v2.DenoiserV2Config | None = None)`

## 模块说明

增强版自监督去噪器，整合 CAREamics 的 N2V2 算法与

该模块用于提升图像质量、抑制噪声或增强光斑可分辨性。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(5, "self_supervised_denoiser_v2")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
