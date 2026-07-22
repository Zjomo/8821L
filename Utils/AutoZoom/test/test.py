from focus_integration import FocusConfig, FocusIntegration

focus = FocusIntegration(
    config=FocusConfig(
        saf_capture_area=(0, 0, 400, 300),
        saf_focus_roi=(50, 50, 200, 150),
        focus_trigger_ratio=0.95,
    ),
    output_root=Path("output"),
    is_virtual_mode=lambda: False,
    capture_frame_callback=workflow._capture_current_rule_ab_frame,
    log_callback=workflow.log,
)

# 建立参考
ok = focus.capture_focus_reference(cycle_index=1)

# 循环中判断补焦
result = focus.run_autofocus_if_needed(cycle_index=1)