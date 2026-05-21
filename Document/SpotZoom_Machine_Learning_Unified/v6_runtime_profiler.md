# v6 / runtime_profiler

- 标题：Runtime Profiler - 运行时性能分析器
- 优化类型：系统诊断
- 版本目录：`SpotZoom_Machine_Learning_v6`
- 导入路径：`SpotZoom_Machine_Learning_v6.runtime_profiler`
- 源文件：`SpotZoom_Machine_Learning_v6/runtime_profiler.py`
- 统一主入口：`RuntimeProfiler`
- 配置类：`ProfilerConfig`
- 结果类：ProfilerReport
- 公共类：ProfilerConfig、StageTiming、ProfilerReport、RuntimeProfiler
- 公共函数：无
- 构造签名：`(config: 'Optional[ProfilerConfig]' = None)`

## 模块说明

Inspired by:

该模块用于运行诊断、健康监控或稳定性保护。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(6, "runtime_profiler")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
