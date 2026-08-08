"""Black-box tests for hill-climb direction trend selection."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pytest

AUTOZOOM_ROOT = Path(__file__).resolve().parent
if str(AUTOZOOM_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOZOOM_ROOT))

from Focus.config import AutofocusConfig
from Focus.search import HillClimbSearch


def _make_search(
    score_fn: Callable[[int], float],
) -> Tuple[HillClimbSearch, Dict[str, int], List[str]]:
    position = {"z": 0}
    logs: List[str] = []

    def move_fn(delta: int) -> None:
        position["z"] += int(delta)

    def measure_fn(_phase: str, _iteration: int):
        return float(score_fn(position["z"])), None

    cfg = AutofocusConfig(
        z_settle_time_s=0.0,
        z_direction_probe_samples=1,
        z_direction_probe_points_per_step=4,
    )
    search = HillClimbSearch(
        cfg=cfg,
        move_fn=move_fn,
        measure_fn=measure_fn,
        log_fn=logs.append,
    )
    return search, position, logs


def _determine(
    search: HillClimbSearch,
    probe_steps: List[int],
):
    return search._determine_direction_with_dynamic_sampling(
        probe_steps=probe_steps,
        min_improve=0.005,
        sample_count=1,
        target=0.95,
        upper_target=1.05,
        points_per_step=4,
    )


def test_positive_direction_is_selected_from_overall_rising_trend() -> None:
    search, position, logs = _make_search(lambda z: 0.90 + 0.003 * z)

    direction, probe_step, score, best_pos = _determine(search, [5, 10])

    assert direction == 1
    assert probe_step == 5
    assert score == pytest.approx(0.96)
    assert best_pos == 20
    assert position["z"] == 0
    assert any("direction decided by trend" in line and "dir=+1" in line for line in logs)


def test_negative_direction_is_selected_when_positive_side_declines() -> None:
    search, position, logs = _make_search(lambda z: 0.90 - 0.003 * z)

    direction, probe_step, score, best_pos = _determine(search, [5, 10])

    assert direction == -1
    assert probe_step == 5
    assert score == pytest.approx(0.96)
    assert best_pos == -20
    assert position["z"] == 0
    assert any("direction decided by trend" in line and "dir=-1" in line for line in logs)


def test_peak_inside_tolerance_returns_best_position_for_local_refine() -> None:
    scores = {
        0: 0.90,
        5: 0.96,
        10: 1.00,
        15: 0.98,
        20: 0.94,
        -5: 0.88,
        -10: 0.86,
        -15: 0.84,
        -20: 0.82,
    }
    search, position, logs = _make_search(lambda z: scores.get(z, 0.80))

    direction, probe_step, score, best_pos = _determine(search, [5, 10])

    assert direction == 1
    assert probe_step == 5
    assert score == pytest.approx(1.00)
    assert best_pos == 10
    assert position["z"] == 0
    assert any("peak is inside tolerance" in line for line in logs)


def test_peak_outside_tolerance_expands_probe_step_before_deciding() -> None:
    scores = {
        0: 0.90,
        5: 0.93,
        10: 0.92,
        15: 0.91,
        20: 0.90,
        25: 0.91,
        50: 0.93,
        75: 0.95,
        100: 0.96,
        -5: 0.87,
        -10: 0.86,
        -15: 0.85,
        -20: 0.84,
        -25: 0.83,
        -50: 0.82,
        -75: 0.81,
        -100: 0.80,
    }
    search, position, logs = _make_search(lambda z: scores.get(z, 0.80))

    direction, probe_step, score, best_pos = _determine(search, [5, 25])

    assert direction == 1
    assert probe_step == 25
    assert score == pytest.approx(0.96)
    assert best_pos == 100
    assert position["z"] == 0
    assert any("first-rise-then-fall outside tolerance" in line for line in logs)
    assert any("stage=2" in line and "dir=+1" in line for line in logs)
