# v4 / multi_sensor_fusion

- 标题：多模态传感器融合定位器 (Multi-Sensor Fusion Localizer)
- 优化类型：检测与定位
- 版本目录：`SpotZoom_Machine_Learning_v4`
- 导入路径：`SpotZoom_Machine_Learning_v4.multi_sensor_fusion`
- 源文件：`SpotZoom_Machine_Learning_v4/multi_sensor_fusion.py`
- 统一主入口：`MultiSensorFusion`
- 配置类：`FusionConfig`
- 结果类：FusionResult
- 公共类：SensorType、SensorReading、FusionConfig、FusionResult、MultiSensorFusion
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v4.multi_sensor_fusion.FusionConfig | None = None)`

## 模块说明

扩展卡尔曼滤波 (EKF) 传感器融合框架

该模块用于光斑检测、候选筛选或定位精度提升。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(4, "multi_sensor_fusion")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
