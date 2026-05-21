# v4 / adaptive_scheduler

- 标题：实时性能自适应调度器 (Real-time Performance Adaptive Scheduler)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v4`
- 导入路径：`SpotZoom_Machine_Learning_v4.adaptive_scheduler`
- 源文件：`SpotZoom_Machine_Learning_v4/adaptive_scheduler.py`
- 统一主入口：`AdaptiveScheduler`
- 配置类：`SchedulerConfig`
- 结果类：TaskState
- 公共类：TaskPriority、TaskState、DegradationLevel、ScheduledTask、SchedulerConfig、ScheduleDecision、AdaptiveScheduler
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v4.adaptive_scheduler.SchedulerConfig | None = None)`

## 模块说明

实时系统的反馈调度 (Feedback Scheduling)

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(4, "adaptive_scheduler")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
