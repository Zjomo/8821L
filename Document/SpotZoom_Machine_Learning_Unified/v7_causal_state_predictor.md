# v7 / causal_state_predictor

- 标题：因果状态预测器 (Causal State Predictor)
- 优化类型：跟踪与预测
- 版本目录：`SpotZoom_Machine_Learning_v7`
- 导入路径：`SpotZoom_Machine_Learning_v7.causal_state_predictor`
- 源文件：`SpotZoom_Machine_Learning_v7/causal_state_predictor.py`
- 统一主入口：`CausalStatePredictor`
- 配置类：`CausalPredictorConfig`
- 结果类：CausalPredictionResult
- 公共类：ODESolverType、CausalPredictionResult、CausalPredictorConfig、CausalStatePredictor
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v7.causal_state_predictor.CausalPredictorConfig | None = None)`

## 模块说明

基于因果 (单向) 建模的实时光斑状态预测模块。使用因果 1D 卷积和

该模块用于时序跟踪、状态估计或运动趋势预测。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(7, "causal_state_predictor")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
