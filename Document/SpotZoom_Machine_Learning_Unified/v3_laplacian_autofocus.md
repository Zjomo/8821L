# v3 / laplacian_autofocus

- 标题：拉普拉斯自动对焦 (Laplacian Autofocus)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v3`
- 导入路径：`SpotZoom_Machine_Learning_v3.laplacian_autofocus`
- 源文件：`SpotZoom_Machine_Learning_v3/laplacian_autofocus.py`
- 统一主入口：`LaplacianAutofocus`
- 配置类：`AutofocusConfig`
- 结果类：FocusResult
- 公共类：AutofocusConfig、FocusResult、LaplacianAutofocus
- 公共函数：无
- 构造签名：`(config: 'Optional[AutofocusConfig]' = None) -> 'None'`

## 模块说明

IRIS Autofocus by bunnie studios (https://github.com/bunnie/iris-audio):

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(3, "laplacian_autofocus")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
