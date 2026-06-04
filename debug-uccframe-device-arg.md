# debug-uccframe-device-arg

Status: [OPEN]

## Symptom
- 准直工作台启动 UCC 预览时报错：`UCCFrameSource.__init__() got an unexpected keyword argument 'device'`

## Hypotheses
1. 准直工作台的调用参数名写成了 `device`，但 `UCCFrameSource` 构造函数只接受 `device_index`。
2. UCC 预览流程里还有别的参数名不匹配，修正后会暴露下一层初始化错误。
3. 预览失败并非相机设备问题，而是纯粹的调用签名错误。
4. 调试帧图片可能保存到配置的临时目录而不是 UI 当前页面目录。
5. 当前日志里的 DSHOW 警告只是相机打不开的次要现象，不是这次异常的根因。

## Evidence to collect
- `spotzoom_qt_ui/app.py` 中 UCC 预览启动代码的实际参数名。
- `SpotZoom.py` 中 `UCCFrameSource.__init__` 的真实签名。
- 调试帧图片的实际保存目录配置。

## Plan
1. 只做证据确认和最小修复。
2. 修复参数名不匹配。
3. 回答调试帧图片保存路径。
4. 重新启动验证。
