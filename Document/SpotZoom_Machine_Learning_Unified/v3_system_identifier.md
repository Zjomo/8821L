# v3 / system_identifier

- 标题：系统辨识器 (System Identifier)
- 优化类型：系统辨识
- 版本目录：`SpotZoom_Machine_Learning_v3`
- 导入路径：`SpotZoom_Machine_Learning_v3.system_identifier`
- 源文件：`SpotZoom_Machine_Learning_v3/system_identifier.py`
- 统一主入口：`SystemIdentifier`
- 配置类：`SysIdConfig`
- 结果类：SysIdResult
- 公共类：SysIdConfig、SysIdResult、SystemIdentifier
- 公共函数：无
- 构造签名：`(config: 'Optional[SysIdConfig]' = None) -> 'None'`

## 模块说明

ML for AO System Identification: 机器学习驱动的自适应光学系统辨识

该模块用于系统辨识、模型拟合或参数识别。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(3, "system_identifier")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
