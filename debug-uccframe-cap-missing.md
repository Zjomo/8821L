# debug-uccframe-cap-missing

Status: [OPEN]

## Symptom
- `UCCFrameSource.__del__` triggers `AttributeError: 'UCCFrameSource' object has no attribute 'cap'` after `VideoCapture.open()` fails.
- Console also shows DSHOW backend warning when opening camera by index.

## Hypotheses
1. `UCCFrameSource.__init__()` does not initialize `self.cap` before `open()` can fail.
2. `release()` and `__del__()` assume `self.cap` exists even on partial construction.
3. `open()` failure path leaves object in an inconsistent state.
4. Camera index/backend mismatch only exposes the cleanup bug.
5. Cleanup order during shutdown triggers a second release path.

## Evidence to collect
- Pre-fix runtime logs around `UCCFrameSource.open()/release()/__del__()`.
- State of `self.cap` on failed open.
- Whether `release()` is idempotent.

## Plan
1. Add instrumentation only.
2. Reproduce and collect logs.
3. Apply minimal fix.
4. Verify with post-fix logs.
