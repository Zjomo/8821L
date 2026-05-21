# v7 / optimal_transport_aligner

- 标题：最优传输对齐器 (Optimal Transport Aligner)
- 优化类型：配准与对齐
- 版本目录：`SpotZoom_Machine_Learning_v7`
- 导入路径：`SpotZoom_Machine_Learning_v7.optimal_transport_aligner`
- 源文件：`SpotZoom_Machine_Learning_v7/optimal_transport_aligner.py`
- 统一主入口：`OptimalTransportAligner`
- 配置类：`OTAlignerConfig`
- 结果类：OTAlignmentResult
- 公共类：TransportCostMetric、OTAlignmentResult、OTAlignerConfig、OptimalTransportAligner
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v7.optimal_transport_aligner.OTAlignerConfig | None = None)`

## 模块说明

基于最优传输理论的光斑分布对齐质量评估与最优步长计算模块。

该模块用于图像配准、相位相关或坐标对齐。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(7, "optimal_transport_aligner")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
