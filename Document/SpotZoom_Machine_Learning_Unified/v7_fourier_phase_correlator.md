# v7 / fourier_phase_correlator

- 标题：傅里叶相位相关器 (Fourier Phase Correlator)
- 优化类型：配准与对齐
- 版本目录：`SpotZoom_Machine_Learning_v7`
- 导入路径：`SpotZoom_Machine_Learning_v7.fourier_phase_correlator`
- 源文件：`SpotZoom_Machine_Learning_v7/fourier_phase_correlator.py`
- 统一主入口：`FourierPhaseCorrelator`
- 配置类：`PhaseCorrelatorConfig`
- 结果类：PhaseCorrelationResult
- 公共类：WindowType、PhaseCorrelationResult、PhaseCorrelatorConfig、FourierPhaseCorrelator
- 公共函数：无
- 构造签名：`(config: SpotZoom_Machine_Learning_v7.fourier_phase_correlator.PhaseCorrelatorConfig | None = None)`

## 模块说明

基于傅里叶相位相关的高精度亚像素图像配准模块。

该模块用于图像配准、相位相关或坐标对齐。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(7, "fourier_phase_correlator")
config = adapter.build_config()
instance = adapter.build_instance(config=config)
```
