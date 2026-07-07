"""
autofocus_qt_ui - 基于 PyQt 的 AutoZoom/Focus 自动聚焦 UI 界面。

提供：
  - 实时图像预览与 ROI 框选
  - 14 项聚焦指标实时显示
  - FocusScore_ratio 曲线
  - 参考基线建立
  - 自动补焦闭环控制（真实 Z 轴 / 模拟模式）
  - 运行日志与结果导出

用法：
    cd Utils/AutoZoom
    python -m autofocus_qt_ui
"""

from .app import run_autofocus_ui

__all__ = ["run_autofocus_ui"]
