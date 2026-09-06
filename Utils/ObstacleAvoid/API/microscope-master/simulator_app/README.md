# Microscope Simulator Module

This directory is the reusable simulation layer built on top of the vendored
`microscope-master` package. It provides an importable, headless API as well as
the existing PyQt GUI.

## Quick start

```python
from simulator_app import SimulatorConfig, create_simulator

with create_simulator(SimulatorConfig(db_path="artifacts/demo.db")) as sim:
    sim.move_by({"x": 25.0, "z": 5.0})  # micrometres
    frame = sim.capture()                 # uint8 grayscale NumPy array
    print(sim.snapshot())
```

For a disposable simulator, omit the configuration and use the in-memory
database:

```python
from simulator_app import create_simulator

with create_simulator() as sim:
    frame = sim.capture()
```

The public facade owns the `Database`, `SQLiteStage`, and `SampleAwareCamera`
instances and closes them deterministically. Advanced integrations can access
`sim.db`, `sim.stage`, and `sim.camera` directly.

## Units and behavior

- Stage coordinates and limits are micrometres (`um`).
- The default limits are X `0..1500`, Y `0..1000`, Z `-25..25`.
- Stage moves are clipped to microscope-master's axis limits.
- Camera frames are grayscale `numpy.uint8`; Z changes simulated focus.
- Persistent state is opt-in through `SimulatorConfig(db_path=...)`.
- `:memory:` is the default, so importing the module never mutates the source tree.

## GUI entry points

Both forms are supported:

```powershell
python -m simulator_app.main --db artifacts/microscope.db
python API/microscope-master/simulator_app/main.py --db artifacts/microscope.db
```

Use `--slow-preview` to include simulated exposure delay. The GUI accepts the
same stage and camera objects as the facade, so application-specific controls
can be layered on top without copying device construction code.

## Maintenance rules

1. Keep imports package-relative; retain the direct-script fallback only in
   `main.py` for existing users.
2. Add schema changes through an explicit SQLite `PRAGMA user_version` step.
3. Keep device behavior compatible with `microscope.abc.Stage` and
   `SimulatedCamera`; put project-specific orchestration in `api.py`.
4. Add headless tests for every public API change before changing GUI code.
5. Never use a repository-local database as a default for library imports.
