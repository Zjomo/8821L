# ObstacleAvoid optimization plan

## Scope

- Keep dry-run as the default and require explicit confirmation for real motors.
- Make pause and emergency-stop control the active controller and stage.
- Validate every ball, obstacle, goal point, and aggregation range against exactly one substrate.
- Keep one substrate-to-goal mapping for avoidance and one substrate-to-range mapping for aggregation.
- Use the same task validation and dispatch rules for simulation and motor/video mode.
- Preserve JSONL auditability and add regression tests for the new contracts.

## Execution order

1. Add shared layout validation and controller/stage control references.
2. Fix sim coordinate conversion and per-ground safety boundaries.
3. Add motor-mode task dispatch and aggregation support.
4. Run core tests, UI offscreen tests, and CLI scenarios.

## Acceptance

- Invalid objects are rejected before any stage command.
- Pause stops new commands and resume revalidates before moving.
- E-stop leaves zero commands after the request.
- OA and AG complete for valid multi-ground dry-run layouts.
- Motor mode remains confirmation-gated and uses YOLO for balls plus manual ROI zones.
