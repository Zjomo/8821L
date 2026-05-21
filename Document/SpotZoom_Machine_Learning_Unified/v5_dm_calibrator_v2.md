# v5 / dm_calibrator_v2

- 标题：变形镜校准器 v2 (DMCalibratorV2)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v5`
- 导入路径：`SpotZoom_Machine_Learning_v5.dm_calibrator_v2`
- 源文件：`SpotZoom_Machine_Learning_v5/dm_calibrator_v2.py`
- 统一主入口：`DMCalibratorV2`
- 配置类：`DMCalibratorV2Config`
- 结果类：CalibrationState、DMCalibratorV2Report
- 公共类：CalibrationState、DMCalibratorV2Config、DMCalibratorV2Report、DMCalibratorV2
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v5.dm_calibrator_v2.DMCalibratorV2Config | None = None)`

## 模块说明

基于 dmlib (https://github.com/jacopoantonello/dmlib) 的变形镜校准方法，

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(5, "dm_calibrator_v2")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
