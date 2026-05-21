# v5 / subspace_system_identifier

- 标题：子空间系统辨识器 (SubspaceSystemIdentifier)
- 优化类型：系统辨识
- 版本目录：`SpotZoom_Machine_Learning_v5`
- 导入路径：`SpotZoom_Machine_Learning_v5.subspace_system_identifier`
- 源文件：`SpotZoom_Machine_Learning_v5/subspace_system_identifier.py`
- 统一主入口：`SubspaceSystemIdentifier`
- 配置类：`SubspaceConfig`
- 结果类：SubspaceReport
- 公共类：SubspaceConfig、SubspaceReport、SubspaceSystemIdentifier
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v5.subspace_system_identifier.SubspaceConfig | None = None)`

## 模块说明

基于 ML for AO (https://github.com/AleksandarHaber/Machine-Learning-and-System-Identification-for-Adaptive-Optics)

该模块用于系统辨识、模型拟合或参数识别。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(5, "subspace_system_identifier")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
