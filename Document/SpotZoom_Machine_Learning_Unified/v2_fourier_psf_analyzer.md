# v2 / fourier_psf_analyzer

- 标题：傅里叶 PSF 分析与波前重建器 (Fourier PSF Analyzer & Wavefront Reconstructor)
- 优化类型：系统分析
- 版本目录：`SpotZoom_Machine_Learning_v2`
- 导入路径：`SpotZoom_Machine_Learning_v2.fourier_psf_analyzer`
- 源文件：`SpotZoom_Machine_Learning_v2/fourier_psf_analyzer.py`
- 统一主入口：`FourierPSFAnalyzer`
- 配置类：`无`
- 结果类：PSFAnalysisResult
- 公共类：PSFAnalysisResult、WavefrontReconstruction、FourierPSFAnalyzer
- 公共函数：无
- 构造签名：`(max_zernike_order: 'int' = 4, pixel_scale_um: 'float' = 1.0, wavelength_nm: 'float' = 632.8)`

## 模块说明

HCIPy (https://github.com/ehpor/hcipy): 高对比度成像仿真框架

该模块用于性能评估、像差分析或频域/统计分析。

## 统一封装访问方式

```python
from SpotZoom_Machine_Learning_Unified import build_module_adapter

adapter = build_module_adapter(2, "fourier_psf_analyzer")
instance = adapter.build_instance()
```
