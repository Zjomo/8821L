# v2 / lqg_robust_controller

- 标题：LQG 鲁棒控制器 (Linear-Quadratic-Gaussian Robust Controller)
- 优化类型：自适应光学与控制
- 版本目录：`SpotZoom_Machine_Learning_v2`
- 导入路径：`SpotZoom_Machine_Learning_v2.lqg_robust_controller`
- 源文件：`SpotZoom_Machine_Learning_v2/lqg_robust_controller.py`
- 统一主入口：`LQGRobustController`
- 配置类：`LQGConfig`
- 结果类：LQGState
- 公共类：LQGConfig、LQGState、LQGRobustController
- 公共函数：无
- 构造签名：`(config: 'Optional[LQGConfig]' = None)`

## 模块说明

python-control (https://github.com/python-control/python-control)

该模块用于控制回路、自适应光学补偿或实时调节。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(2, "lqg_robust_controller")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
