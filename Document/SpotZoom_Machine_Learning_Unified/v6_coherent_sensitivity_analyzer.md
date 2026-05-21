# v6 / coherent_sensitivity_analyzer

- 标题：Coherent Sensitivity Analyzer - 相干灵敏度分析器
- 优化类型：系统分析
- 版本目录：`SpotZoom_Machine_Learning_v6`
- 导入路径：`SpotZoom_Machine_Learning_v6.coherent_sensitivity_analyzer`
- 源文件：`SpotZoom_Machine_Learning_v6/coherent_sensitivity_analyzer.py`
- 统一主入口：`CoherentSensitivityAnalyzer`
- 配置类：`SensitivityConfig`
- 结果类：SensitivityReport
- 公共类：SensitivityConfig、SensitivityReport、CoherentSensitivityAnalyzer
- 公共函数：无
- 构造签名：`(config: 'Optional[SensitivityConfig]' = None)`

## 模块说明

Inspired by:

该模块用于性能评估、像差分析或频域/统计分析。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(6, "coherent_sensitivity_analyzer")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
