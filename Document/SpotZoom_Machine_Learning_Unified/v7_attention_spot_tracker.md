# v7 / attention_spot_tracker

- 标题：注意力光斑跟踪器 (Attention Spot Tracker)
- 优化类型：跟踪与预测
- 版本目录：`SpotZoom_Machine_Learning_v7`
- 导入路径：`SpotZoom_Machine_Learning_v7.attention_spot_tracker`
- 源文件：`SpotZoom_Machine_Learning_v7/attention_spot_tracker.py`
- 统一主入口：`AttentionSpotTracker`
- 配置类：`AttentionTrackerConfig`
- 结果类：AttentionTrackResult
- 公共类：PositionalEncodingType、Track、AttentionTrackResult、AttentionTrackerConfig、AttentionSpotTracker
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v7.attention_spot_tracker.AttentionTrackerConfig | None = None)`

## 模块说明

基于 Transformer 注意力机制的多帧光斑跟踪模块。

该模块用于时序跟踪、状态估计或运动趋势预测。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(7, "attention_spot_tracker")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
