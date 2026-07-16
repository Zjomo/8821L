"""实验序列（Recipe）模块测试。"""

import pytest

from pi_spectrometer.core.recipe import (
    AcquisitionRecipe,
    RecipeAction,
    RecipeRunner,
    RecipeStep,
)
from pi_spectrometer.core.types import ROI
from pi_spectrometer.picam.demo import MockSpectrometerBackend


def test_recipe_step_validation():
    step = RecipeStep(action=RecipeAction.ACQUIRE, params={"num_frames": 1})
    step.validate()


def test_recipe_step_invalid_repeats():
    step = RecipeStep(action=RecipeAction.ACQUIRE, repeats=0)
    with pytest.raises(ValueError):
        step.validate()


def test_recipe_step_missing_param():
    step = RecipeStep(action=RecipeAction.SET_EXPOSURE)
    with pytest.raises(ValueError):
        step.validate()


def test_recipe_serialization():
    recipe = AcquisitionRecipe(name="test")
    recipe.add_step(RecipeStep(action=RecipeAction.ACQUIRE, repeats=2, delay_s=0.1))
    recipe.add_step(RecipeStep(action=RecipeAction.WAIT, params={"seconds": 0.5}))

    data = recipe.to_dict()
    restored = AcquisitionRecipe.from_dict(data)

    assert restored.name == "test"
    assert len(restored.steps) == 2
    assert restored.steps[0].repeats == 2
    assert restored.steps[1].action == RecipeAction.WAIT


def test_recipe_runner_acquire(mock_backend):
    mock_backend.connect()
    step = RecipeStep(action=RecipeAction.ACQUIRE, params={"num_frames": 1})
    runner = RecipeRunner(backend=mock_backend, steps=[step])
    results = runner.run()
    assert len(results) == 1
    assert results[0].ok


def test_recipe_runner_repeat(mock_backend):
    mock_backend.connect()
    step = RecipeStep(action=RecipeAction.ACQUIRE, repeats=3)
    runner = RecipeRunner(backend=mock_backend, steps=[step])
    results = runner.run()
    assert len(results) == 3


def test_recipe_runner_set_exposure(mock_backend):
    mock_backend.connect()
    steps = [
        RecipeStep(action=RecipeAction.SET_EXPOSURE, params={"seconds": 0.5}),
        RecipeStep(action=RecipeAction.ACQUIRE),
    ]
    runner = RecipeRunner(backend=mock_backend, steps=steps)
    runner.run()
    assert mock_backend.get_exposure() == pytest.approx(0.5)


def test_recipe_runner_set_roi(mock_backend):
    mock_backend.connect()
    steps = [
        RecipeStep(
            action=RecipeAction.SET_ROI,
            params={"x": 10, "y": 20, "width": 512, "height": 128},
        ),
    ]
    runner = RecipeRunner(backend=mock_backend, steps=steps)
    runner.run()
    roi = mock_backend.get_roi()
    assert roi.x == 10
    assert roi.y == 20
    assert roi.width == 512
    assert roi.height == 128


def test_recipe_runner_stop(mock_backend):
    mock_backend.connect()
    steps = [
        RecipeStep(action=RecipeAction.ACQUIRE, repeats=100),
    ]
    runner = RecipeRunner(backend=mock_backend, steps=steps)

    def stop_after_first(*args, **kwargs):
        runner.stop()

    runner.on_step_done = stop_after_first
    results = runner.run()
    assert len(results) <= 2
    assert not runner.is_running()


def test_recipe_runner_validation_empty():
    backend = MockSpectrometerBackend()
    runner = RecipeRunner(backend=backend, steps=[])
    with pytest.raises(ValueError):
        runner.run()


def test_recipe_runner_wait(mock_backend):
    mock_backend.connect()
    steps = [RecipeStep(action=RecipeAction.WAIT, params={"seconds": 0.05})]
    runner = RecipeRunner(backend=mock_backend, steps=steps)
    runner.run()
    assert not runner.is_running()
