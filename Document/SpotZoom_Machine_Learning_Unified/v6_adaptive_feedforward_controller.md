# v6 / adaptive_feedforward_controller

- 标题：Adaptive Feedforward Controller - 自适应前馈控制器
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v6`
- 导入路径：`SpotZoom_Machine_Learning_v6.adaptive_feedforward_controller`
- 源文件：`SpotZoom_Machine_Learning_v6/adaptive_feedforward_controller.py`
- 统一主入口：`AdaptiveFeedforwardController`
- 配置类：`FeedforwardConfig`
- 结果类：无
- 公共类：FeedforwardConfig、FeedforwardOutput、AdaptiveFeedforwardController
- 公共函数：无
- 构造签名：`(config: 'Optional[FeedforwardConfig]' = None)`

## 模块说明

Inspired by:

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(6, "adaptive_feedforward_controller")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
