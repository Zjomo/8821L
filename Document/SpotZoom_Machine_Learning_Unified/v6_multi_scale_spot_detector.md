# v6 / multi_scale_spot_detector

- 标题：Multi-Scale Spot Detector - 多尺度光斑检测器
- 优化类型：检测与定位
- 版本目录：`SpotZoom_Machine_Learning_v6`
- 导入路径：`SpotZoom_Machine_Learning_v6.multi_scale_spot_detector`
- 源文件：`SpotZoom_Machine_Learning_v6/multi_scale_spot_detector.py`
- 统一主入口：`MultiScaleSpotDetector`
- 配置类：`MultiScaleDetectorConfig`
- 结果类：MultiScaleDetectionResult
- 公共类：MultiScaleDetectorConfig、MultiScaleDetectionResult、MultiScaleSpotDetector
- 公共函数：无
- 构造签名：`(config: 'Optional[MultiScaleDetectorConfig]' = None)`

## 模块说明

Inspired by:

该模块用于光斑检测、候选筛选或定位精度提升。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(6, "multi_scale_spot_detector")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
