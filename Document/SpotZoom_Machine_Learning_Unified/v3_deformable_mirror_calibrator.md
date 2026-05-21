# v3 / deformable_mirror_calibrator

- 标题：变形镜校准器 (Deformable Mirror Calibrator)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v3`
- 导入路径：`SpotZoom_Machine_Learning_v3.deformable_mirror_calibrator`
- 源文件：`SpotZoom_Machine_Learning_v3/deformable_mirror_calibrator.py`
- 统一主入口：`DMCalibrator`
- 配置类：`DMCalibConfig`
- 结果类：DMCalibResult
- 公共类：DMCalibConfig、DMCalibResult、DMCalibrator
- 公共函数：无
- 构造签名：`(config: 'Optional[DMCalibConfig]' = None) -> 'None'`

## 模块说明

dmlib (https://github.com/spacetelescope/dmlib): 变形镜校准库

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(3, "deformable_mirror_calibrator")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
