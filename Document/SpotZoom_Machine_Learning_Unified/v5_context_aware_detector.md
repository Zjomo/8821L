# v5 / context_aware_detector

- 标题：上下文感知光斑检测器 (ContextAwareDetector)
- 优化类型：检测与定位
- 版本目录：`SpotZoom_Machine_Learning_v5`
- 导入路径：`SpotZoom_Machine_Learning_v5.context_aware_detector`
- 源文件：`SpotZoom_Machine_Learning_v5/context_aware_detector.py`
- 统一主入口：`ContextAwareDetector`
- 配置类：`ContextDetectorConfig`
- 结果类：ContextDetectorResult
- 公共类：ContextDetectorConfig、ContextDetectorResult、ContextAwareDetector
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v5.context_aware_detector.ContextDetectorConfig | None = None)`

## 模块说明

基于 DECODE (https://github.com/TuragaLab/DECODE) 的深度上下文依赖检测框架，

该模块用于光斑检测、候选筛选或定位精度提升。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(5, "context_aware_detector")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
