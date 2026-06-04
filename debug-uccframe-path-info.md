# debug-uccframe-path-info

Status: [OPEN]

## Symptom
- 启动 UCC 预览时报错：`'SpotZoomQtMainWindow' object has no attribute '_alignment_ucc_path_info'`

## Hypotheses
1. `_alignment_ucc_path_info` 没有在 `_build_alignment_page()` 中创建。
2. 该属性是在别的页面或条件分支里创建，但主流程没有走到。
3. 启动按钮回调比 UI 组件创建更早触发，导致属性未初始化。
4. 该标签应与 UCC 控制行同层初始化，放在按钮下方更稳妥。

## Evidence to collect
- `_build_alignment_page()` 是否定义了 `_alignment_ucc_path_info`
- `_start_alignment_ucc_preview()` 是否在创建前就引用了它
- 修复后是否能正常更新状态文本

## Plan
1. 补上 `_alignment_ucc_path_info` 初始化。
2. 编译验证。
3. 如需，再做一次主流程冒烟测试记录其他错误。
