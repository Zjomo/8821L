"""Public API regression tests for the reusable microscope simulator."""
import pytest


@pytest.fixture(scope="module")
def simulator_api():
    from obstacle_avoidance import sim_microscope

    sim_microscope._ensure_paths()
    import simulator_app

    return simulator_app


def test_headless_facade_capture_and_lifecycle(simulator_api):
    sim = simulator_api.create_simulator()
    try:
        frame = sim.capture()
        assert frame.shape == (600, 800)
        assert frame.dtype.name == "uint8"
        start = sim.position
        assert sim.move_by({"x": 25.0, "z": 5.0})["x"] == pytest.approx(start["x"] + 25)
        assert sim.position["z"] == pytest.approx(start["z"] + 5)
        snapshot = sim.snapshot()
        assert snapshot["stage"]["axes"]["z"]["position"] == pytest.approx(sim.position["z"])
    finally:
        sim.close()
        sim.close()  # idempotent cleanup
    assert sim.closed
    with pytest.raises(RuntimeError, match="closed"):
        sim.capture()


def test_persistent_position_and_config_validation(simulator_api, tmp_path):
    db_path = str(tmp_path / "nested" / "sim.db")
    config = simulator_api.SimulatorConfig(db_path=db_path)
    with simulator_api.MicroscopeSimulator(config) as sim:
        sim.move_to({"x": 123.0, "y": 456.0, "z": 7.0})

    with simulator_api.create_simulator(db_path=db_path) as reopened:
        assert reopened.position == pytest.approx({"x": 123.0, "y": 456.0, "z": 7.0})

    with pytest.raises(ValueError, match="exactly x, y and z"):
        simulator_api.SimulatorConfig(stage_limits={"x": (0, 1)})
