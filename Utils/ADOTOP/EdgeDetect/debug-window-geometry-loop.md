# [OPEN] debug-window-geometry-loop

## 症状
导入视频后，主窗口尺寸在不断扩大，终端持续输出 `QWindowsWindow::setGeometry` 警告。

## 初始假设
1. 原图/分割图的 QLabel 在不断刷新 `QPixmap` 后，尺寸策略或最小尺寸被 pixmap 反向影响，导致主窗口最小尺寸持续增大。
2. 某个 dock/表格控件在视频帧刷新时被频繁触发布局重算，叠加导致窗口尺寸抖动或持续增长。
3. ROI 选择或视频首帧加载后，窗口对内容做了自适应缩放，但未限制最大/最小尺寸，造成 setGeometry 反复失败。
4. 每次帧刷新时对表格或状态控件写入长文本，引发布局重新计算，但实际问题更可能是图片控件的尺寸提示。

## 调试计划
- 先确认窗口/label 的 sizePolicy、minimumSize、scaledContents、layout 约束。
- 必要时仅添加最小限度的运行时日志，不改业务逻辑。
- 找到根因后做最小修复，再回放视频验证窗口尺寸稳定。
