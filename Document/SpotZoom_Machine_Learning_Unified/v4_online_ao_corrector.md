# v4 / online_ao_corrector

- 标题：在线自适应光学校正器 (Online Adaptive Optics Corrector)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v4`
- 导入路径：`SpotZoom_Machine_Learning_v4.online_ao_corrector`
- 源文件：`SpotZoom_Machine_Learning_v4/online_ao_corrector.py`
- 统一主入口：`OnlineAOCorrector`
- 配置类：`AOCorrectorConfig`
- 结果类：AOCorrectionResult
- 公共类：CorrectionMode、ZernikeMode、AOCorrectorConfig、AOCorrectionResult、OnlineAOCorrector
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v4.online_ao_corrector.AOCorrectorConfig | None = None)`

## 模块说明

AOtools (https://github.com/AOtools/aotools)

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(4, "online_ao_corrector")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
