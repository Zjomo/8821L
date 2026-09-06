# XYZ Motion Extension Plan

## Scope

Extend the current obstacle-avoidance UI and simulator with microscope-style X/Y/Z stage controls and hardware-aligned motion telemetry. The virtual mode remains the default and must not open or command real hardware.

## Execution plan

1. **Contract and configuration**
   - Add a shared XYZ motion configuration with units, travel limits, steps-per-unit, max speed, acceleration, deceleration, settle time, backlash, soft-limit and collision interlock settings.
   - Validate values at the boundary and expose a serializable snapshot for UI/reporting.

2. **Stage model and telemetry**
   - Add a deterministic virtual XYZ stage that mirrors `microscope-master` axis semantics: absolute/relative moves, clamped limits, enable/disable, and persisted position when available.
   - Record per-move command, requested/applied distance, step counts, duration, peak/average speed, acceleration, position and limit status.
   - Keep the existing XY obstacle-avoidance protocol compatible through an adapter.

3. **UI controls**
   - Add X/Y/Z position displays, relative jog inputs, per-axis step size, home/zero, stop and enable controls.
   - Add motion configuration controls and a live telemetry panel; use the same controls in virtual mode with an explicit simulated indicator.

4. **Microscope integration**
   - Bind the virtual XYZ stage to `SimMicroscopeWorld.micro_stage`, so X/Y alter the camera field of view and Z alters focus, matching `microscope-master` behavior.
   - Surface stage limits and camera focus/exposure state in the UI.

5. **Diagnostics and tests**
   - Add unit tests for config validation, conversion, limits, acceleration/velocity telemetry, Z focus response and stop behavior.
   - Add offscreen UI smoke tests for XYZ controls and telemetry updates.
   - Run the full existing suite plus the new focused tests; document failures and fixes in the final handoff.

## Acceptance criteria

- A virtual X/Y/Z jog changes the simulated stage and reports position, steps, speed, acceleration and limit state.
- X/Y movement changes the simulated camera view; Z movement changes blur/focus.
- Invalid settings and out-of-range moves are rejected without partial state changes.
- Stop/disable prevents subsequent motion until explicitly enabled.
- Existing obstacle-avoidance tests and real-hardware safety gates remain green.

## Risk and rollback points

- Keep the existing `XYStageProtocol` and driver constructors backward compatible.
- Introduce the new model behind adapters so the obstacle-avoidance controller does not need a broad rewrite.
- If Qt widgets cannot be exercised in the environment, retain deterministic model tests and record UI tests as environment-limited.
