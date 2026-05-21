# v6 / psf_fingerprint_analyzer

- 标题：PSF Fingerprint Analyzer - PSF 指纹分析器
- 优化类型：系统分析
- 版本目录：`SpotZoom_Machine_Learning_v6`
- 导入路径：`SpotZoom_Machine_Learning_v6.psf_fingerprint_analyzer`
- 源文件：`SpotZoom_Machine_Learning_v6/psf_fingerprint_analyzer.py`
- 统一主入口：`PSFFingerprintAnalyzer`
- 配置类：`PSFFingerprintConfig`
- 结果类：PSFFingerprintReport
- 公共类：PSFFingerprintConfig、PSFFingerprintReport、PSFFingerprintAnalyzer
- 公共函数：无
- 构造签名：`(config: 'Optional[PSFFingerprintConfig]' = None)`

## 模块说明

Inspired by:

该模块用于性能评估、像差分析或频域/统计分析。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(6, "psf_fingerprint_analyzer")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
