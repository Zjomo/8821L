# debug-uccframe-read-frame

Status: [OPEN]

## Symptom
- 准直工作台日志报错：`[准直工作台] UCC 帧读取失败: 'UCCFrameSource' object has no attribute 'read_frame'`
- 目标：让实时图像区能稳定读取并显示 UCC 帧

## Hypotheses
1. `UCCFrameSource` 没有 `read_frame()`，真实接口是 `grab_frame()`。
2. 运行模式的 UCC 预览使用了不同的数据源或封装，因此没有暴露这个错误。
3. `_update_alignment_ucc_preview()` 调用了错误的方法名，导致实时图像区无法刷新。
4. 修复方法名后，可能继续暴露图像格式/显示链路问题。
5. 实时图像区与准直控制区的状态同步可能还有边界问题。

## Evidence to collect
- `UCCFrameSource` 的实际帧读取方法名
- 准直工作台 UCC 更新函数的调用链
- 修复后是否能继续显示帧并更新状态文本

## Plan
1. 修复帧读取方法名。
2. 编译验证。
3. 继续边测边记，检查 UI 是否还有后续问题。
