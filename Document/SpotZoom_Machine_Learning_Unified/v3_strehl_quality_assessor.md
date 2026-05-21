# v3 / strehl_quality_assessor

- 标题：Strehl 质量评估器 (Strehl Quality Assessor)
- 优化类型：系统分析
- 版本目录：`SpotZoom_Machine_Learning_v3`
- 导入路径：`SpotZoom_Machine_Learning_v3.strehl_quality_assessor`
- 源文件：`SpotZoom_Machine_Learning_v3/strehl_quality_assessor.py`
- 统一主入口：`StrehlQualityAssessor`
- 配置类：`StrehlConfig`
- 结果类：无
- 公共类：StrehlConfig、StrehlAssessment、StrehlQualityAssessor
- 公共函数：无
- 构造签名：`(config: 'Optional[StrehlConfig]' = None) -> 'None'`

## 模块说明

prysm (https://github.com/brandondube/prysm): 光学衍射与像差分析库

该模块用于性能评估、像差分析或频域/统计分析。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(3, "strehl_quality_assessor")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
