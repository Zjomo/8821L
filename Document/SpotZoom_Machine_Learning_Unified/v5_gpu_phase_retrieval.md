# v5 / gpu_phase_retrieval

- 标题：GPU 加速相位恢复引擎 (GPUPhaseRetrieval)
- 优化类型：系统分析
- 版本目录：`SpotZoom_Machine_Learning_v5`
- 导入路径：`SpotZoom_Machine_Learning_v5.gpu_phase_retrieval`
- 源文件：`SpotZoom_Machine_Learning_v5/gpu_phase_retrieval.py`
- 统一主入口：`GPUPhaseRetrieval`
- 配置类：`PhaseRetrievalConfig`
- 结果类：PhaseRetrievalResult
- 公共类：PhaseAlgorithm、DeviceType、PhaseRetrievalConfig、PhaseRetrievalResult、GPUPhaseRetrieval
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v5.gpu_phase_retrieval.PhaseRetrievalConfig | None = None)`

## 模块说明

基于 slmsuite (https://github.com/holodyne/slmsuite) 的 GPU 加速迭代相位恢复算法，

该模块用于性能评估、像差分析或频域/统计分析。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(5, "gpu_phase_retrieval")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
