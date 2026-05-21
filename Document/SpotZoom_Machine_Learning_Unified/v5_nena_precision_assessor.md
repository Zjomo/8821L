# v5 / nena_precision_assessor

- 标题：NeNA 精度评估器 (NeNAPrecisionAssessor)
- 优化类型：系统分析
- 版本目录：`SpotZoom_Machine_Learning_v5`
- 导入路径：`SpotZoom_Machine_Learning_v5.nena_precision_assessor`
- 源文件：`SpotZoom_Machine_Learning_v5/nena_precision_assessor.py`
- 统一主入口：`NeNAPrecisionAssessor`
- 配置类：`NeNAConfig`
- 结果类：NeNAReport
- 公共类：PrecisionMethod、NeNAConfig、NeNAReport、NeNAPrecisionAssessor
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v5.nena_precision_assessor.NeNAConfig | None = None)`

## 模块说明

基于 Picasso (https://github.com/jungmannlab/picasso) 的 NeNA 方法，

该模块用于性能评估、像差分析或频域/统计分析。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(5, "nena_precision_assessor")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
