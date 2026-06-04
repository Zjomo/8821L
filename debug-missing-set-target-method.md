# debug-missing-set-target-method

Status: [OPEN]

## Symptom
- 启动 `python -m spotzoom_qt_ui` 时，`_build_alignment_page()` 在绑定 `btn_set_target` 回调时抛出：`AttributeError: 'SpotZoomQtMainWindow' object has no attribute '_set_alignment_target_point'`

## Hypotheses
1. `_set_alignment_target_point()` 在当前 `app.py` 中被误删或被错误拼接到别的方法里。
2. `btn_set_target.clicked.connect(self._set_alignment_target_point)` 绑定到了不存在的方法。
3. 修复后 UI 会继续构建到后续控件阶段；若还有错误，会暴露其他独立问题。
4. 当前错误只影响“准直工作台”页面初始化，不影响其它页面本身的定义。

## Evidence to collect
- `_enter_alignment_4axis_mode()` 后的代码块是否存在方法拼接错误。
- `_set_alignment_target_point()` 是否缺失。
- 修复后是否能继续进入主窗口构建。

## Plan
1. 先修复方法定义顺序与缺失方法。
2. 编译验证。
3. 若还有错误，再继续按 UI 主流程排查。
