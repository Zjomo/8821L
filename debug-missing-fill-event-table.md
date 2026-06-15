# [OPEN] debug-missing-fill-event-table

## Symptom
`python -m spotzoom_qt_ui` crashes during window show with:
`AttributeError: 'SpotZoomQtMainWindow' object has no attribute '_fill_event_table'`

## Hypotheses
1. `_fill_event_table` method was accidentally removed while editing `app.py`.
2. A refactor renamed `_fill_event_table` but the call site in `_refresh_dashboard` was not updated.
3. The method exists but is outside the class due to indentation / placement issues, so it is not bound to `SpotZoomQtMainWindow`.
4. The current import / class definition is stale because of a partial edit that broke method ordering.

## Status
[OPEN]
