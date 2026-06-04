# debug-mirror-axes-layout-bug

Status: [OPEN]

## Symptom
- 准直工作台点击“模拟抖动”后反复报错：`'SpotZoomQtMainWindow' object has no attribute '_get_alignment_mirror_axes'`
- 点击“设定目标点”后 UI 开始不稳定，部分按钮会超出屏幕

## Hypotheses
1. `_get_alignment_mirror_axes()` 在当前 `app.py` 中被误删或未同步。
2. 模拟抖动/闭环逻辑依赖的镜面轴映射没有被正确初始化，导致电机控制层无法工作。
3. 准直工作台右侧面板没有滚动容器，内容过高时会把按钮挤出屏幕。
4. “设定目标点”后状态刷新文本过长或布局未约束，触发页面重排不稳定。
5. 修复后，如果仍有问题，下一层会是轴映射配置或页面布局细节，而不是按钮回调本身。

## Evidence to collect
- `_get_alignment_mirror_axes()` 是否存在。
- 准直页面是否被 `QScrollArea` 或等效容器包裹。
- 目标点设定后的状态刷新是否会持续撑大布局。

## Plan
1. 先确认方法缺失与布局结构。
2. 补回最小方法并约束页面滚动。
3. 编译验证。
4. 请用户复测并反馈。
