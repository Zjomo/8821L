# SpotZoom v74 Validation Report (2026-05-18)

## Scope
- Internet frontier benchmark + module mapping
- SpotZoom.py v74 supplementation verification
- Build/startup/page/API/DB/console/network/static/code-quality/module-structure checks

## v74 Integration
- `frontier_v74_adaptive_retrack_guard` is wired into defaults, bundles, config dataclass, runtime metrics, controller init, v57 pipeline, CLI args, and validations.
- Key implementation: `SpotZoom.py` around lines 16548+, with pipeline hook at 16936+.

## Build/Static Checks
- `python -m py_compile SpotZoom.py`: pass
- `python -m compileall SpotZoom.py SpotZoom_Machine_Learning`: pass
- `python -m py_compile XPS/xps_admin_console/server.py`: pass
- `node --check XPS/xps_admin_console/static/app.js`: pass

## SpotZoom Case Matrix (v74)

- smoke: status=success, elapsed_s=4.24, detect_success=5, v73_refines=0, v74_triggered=1, v74_refines=0
- offcenter_noisy: status=success, elapsed_s=4.561, detect_success=5, v73_refines=2, v74_triggered=1, v74_refines=0
- probe: status=success, elapsed_s=4.642, detect_success=5, v73_refines=3, v74_triggered=1, v74_refines=0
- reject_probe: status=failed, elapsed_s=13.309, detect_success=12, v73_refines=9, v74_triggered=4, v74_refines=0
- missing_image: status=failed, elapsed_s=0.009, detect_success=0, v73_refines=0, v74_triggered=0, v74_refines=0
- xps_fail: status=failed, elapsed_s=2.053, detect_success=0, v73_refines=0, v74_triggered=0, v74_refines=0

Observed vs v73 baseline:
- smoke/offcenter_noisy/probe remain success and faster in this run.
- missing_image/xps_fail keep expected-fail behavior.
- reject_probe failed once in the main matrix (max alignment rounds reached).
- AB replay (reject_probe, 3 runs each): baseline_v73 success=3/3, enabled_v74 success=3/3; this points to non-deterministic high-noise tail risk.

## XPS Admin Console Checks
- Server started on `127.0.0.1:8905` and probed automatically.
- HTTP status profile: {'200': 7, '404': 2, '400': 3, '413': 1}
- Frontend runtime signals: console=0, pageErrors=0, requestFailed=0
- Static assets (`/`, `/static/app.js`, `/static/styles.css`) load as 200.
- API negative-path behavior is correct (400/404/413 as expected).
- DB read/write paths remain in not-connected branch in this environment (no real controller session).

## Code Quality / Structure
- SpotZoom.py size: 23993 LOC, 22 classes, 266 functions.
- Long functions (>=200 LOC): 16 (parse_args=3591 LOC, __init__=2341 LOC).
- Coupling proxies: self.cfg refs=1477, metric increment calls=442.
- Conclusion: large monolithic controller with many stacked frontier gates; high regression and maintenance cost.

## Issues and Priority
- P1: reject_probe one-shot failure in matrix run (regression risk in high-noise conditions).
- P2: SpotZoom.py monolith and high coupling increase change risk and review cost.
- P2: XPS DB true read/write path not validated without real hardware connection.
- P3: v74 refine count stayed 0 in this matrix; benefit is not yet stable under current thresholds.

## Next Optimization Plan
1. Add `--sim-seed` and multi-run AB harness to make noisy scenarios reproducible.
2. Add a v74 gain guard: auto-bypass when v74 triggers repeatedly without net alignment gain.
3. Split SpotZoom.py into modules (CLI/config, runtime pipeline, frontier guards, reporters).
4. Add hardware-in-the-loop DB write/readback validation for xps_admin_console.

## Frontier References
- ReTracker (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/html/Tan_ReTracker_Exploring_Image_Matching_for_Robust_Online_Any_Point_Tracking_ICCV_2025_paper.html
- ReTracker PDF: https://openaccess.thecvf.com/content/ICCV2025/papers/Tan_ReTracker_Exploring_Image_Matching_for_Robust_Online_Any_Point_Tracking_ICCV_2025_paper.pdf
- Track-On: https://github.com/gorkaydemir/track_on
- CoTracker3: https://cotracker3.github.io/
- AllTracker: https://github.com/aharley/alltracker
- Trackpy adaptive search: https://soft-matter.github.io/trackpy/v0.3.0/tutorial/adaptive-search.html
- OpenCV template matching: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
- SAM2: https://github.com/facebookresearch/sam2
