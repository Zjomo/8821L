# debug-uccframe-none-resolution

Status: [OPEN]

## Symptom
- 准直工作台 UCC 预览失败：`'NoneType' object has no attribute 'upper'`
- 目标：继续修复该错误，并整理一份后续功能排查清单

## Hypotheses
1. `UCCFrameSource.__init__()` 对 `resolution` 直接调用了 `.upper()`，但 UI 传入了 `None`。
2. `AUTO` 分辨率在 UI 层被映射成了 `None`，但底层构造函数没有兼容。
3. 该问题与相机硬件无关，是参数归一化缺失。
4. 修复后 UCC 预览会继续进入相机打开阶段，若仍失败则暴露设备索引/后端问题。
5. 当前 UI 还需要一轮完整功能巡检，可能会暴露其它布局/状态同步问题。

## Evidence to collect
- `UCCFrameSource.__init__` 的实际参数处理逻辑
- 准直工作台启动 UCC 预览时传入的参数值
- 修复后是否还能继续进入相机打开阶段

## Plan
1. 修复 `resolution` 的 None 兼容。
2. 复测 UCC 预览启动。
3. 记录窗口内其它可见功能错误。
4. 输出一份逐项解决计划清单。
