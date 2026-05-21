# v3 / hardware_abstraction_layer

- 标题：硬件抽象层 (Hardware Abstraction Layer)
- 优化类型：硬件抽象
- 版本目录：`SpotZoom_Machine_Learning_v3`
- 导入路径：`SpotZoom_Machine_Learning_v3.hardware_abstraction_layer`
- 源文件：`SpotZoom_Machine_Learning_v3/hardware_abstraction_layer.py`
- 统一主入口：`HardwareAbstractionLayer`
- 配置类：`HALConfig`
- 结果类：无
- 公共类：DeviceType、DeviceStatus、DeviceInfo、HALConfig、HardwareAbstractionLayer
- 公共函数：无
- 构造签名：`(config: 'Optional[HALConfig]' = None) -> 'None'`

## 模块说明

python-microscope (https://github.com/python-microscope/microscope): 显微镜硬件抽象层

该模块用于统一设备接口或隔离底层硬件差异。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(3, "hardware_abstraction_layer")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
