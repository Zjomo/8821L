# v5 / careamics_adapter

- 标题：CAREamics 自监督去噪适配器 (v5.0)
- 优化类型：图像增强
- 版本目录：`SpotZoom_Machine_Learning_v5`
- 导入路径：`SpotZoom_Machine_Learning_v5.careamics_adapter`
- 源文件：`SpotZoom_Machine_Learning_v5/careamics_adapter.py`
- 统一主入口：`CAREamicsAdapter`
- 配置类：`CAREamicsConfig`
- 结果类：CAREamicsReport
- 公共类：DenoiseAlgorithm、BackendType、CAREamicsConfig、CAREamicsReport、CAREamicsAdapter
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v5.careamics_adapter.CAREamicsConfig | None = None)`

## 模块说明

基于 CAREamics (https://github.com/CAREamics/careamics) 的统一 PyTorch 去噪管线，

该模块用于提升图像质量、抑制噪声或增强光斑可分辨性。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(5, "careamics_adapter")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
