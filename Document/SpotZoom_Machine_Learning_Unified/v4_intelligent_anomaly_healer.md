# v4 / intelligent_anomaly_healer

- 标题：智能异常诊断与自愈系统 (Intelligent Anomaly Diagnosis & Self-Healing System)
- 优化类型：系统诊断
- 版本目录：`SpotZoom_Machine_Learning_v4`
- 导入路径：`SpotZoom_Machine_Learning_v4.intelligent_anomaly_healer`
- 源文件：`SpotZoom_Machine_Learning_v4/intelligent_anomaly_healer.py`
- 统一主入口：`IntelligentAnomalyHealer`
- 配置类：`AnomalyHealerConfig`
- 结果类：HealingAction、AnomalyReport
- 公共类：AnomalyType、SeverityLevel、HealingActionType、HealingAction、AnomalyReport、AnomalyHealerConfig、IntelligentAnomalyHealer
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v4.intelligent_anomaly_healer.AnomalyHealerConfig | None = None)`

## 模块说明

sktime (https://github.com/sktime/sktime)

该模块用于运行诊断、健康监控或稳定性保护。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(4, "intelligent_anomaly_healer")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
