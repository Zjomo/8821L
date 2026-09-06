"""Stable, headless API for the microscope-master simulator.

This module owns construction and lifecycle.  It deliberately keeps the
underlying python-microscope objects available as ``stage``, ``camera`` and
``db`` so an application can progressively adopt the simulator without a
large rewrite.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple

from .db import Database
from .devices import SampleAwareCamera, SQLiteStage

DEFAULT_STAGE_LIMITS = {
    "x": (0.0, 1500.0),
    "y": (0.0, 1000.0),
    "z": (-25.0, 25.0),
}


@dataclass(frozen=True)
class SimulatorConfig:
    """Construction settings shared by projects using the simulator."""

    db_path: str = ":memory:"
    stage_limits: Mapping[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_STAGE_LIMITS))
    fast_preview: bool = True
    sample_spec: Optional[Mapping[str, Mapping]] = None

    def __post_init__(self) -> None:
        limits = {str(name).lower(): (float(pair[0]), float(pair[1]))
                  for name, pair in self.stage_limits.items()}
        if set(limits) != {"x", "y", "z"}:
            raise ValueError("stage_limits must contain exactly x, y and z")
        for name, (lower, upper) in limits.items():
            if lower >= upper:
                raise ValueError(f"invalid {name} limits: {lower}, {upper}")
        object.__setattr__(self, "stage_limits", limits)
        object.__setattr__(self, "db_path", os.fspath(self.db_path))


class MicroscopeSimulator:
    """Headless simulator facade with deterministic resource cleanup.

    Positions are in micrometres, matching ``microscope.abc.Stage`` and the
    original simulator.  ``capture()`` returns a grayscale NumPy frame.
    """

    def __init__(self, config: SimulatorConfig | None = None):
        self.config = config or SimulatorConfig()
        path = self.config.db_path
        if path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True,
                                                             exist_ok=True)
        self.db = Database(path)
        self.stage = SQLiteStage(self.db, self.config.stage_limits)
        self.camera = SampleAwareCamera(self.stage, self.db)
        if self.config.sample_spec:
            self.camera.set_sample_spec(self.config.sample_spec)
        self.camera.set_fast_preview(self.config.fast_preview)
        self.camera.enable()
        self.stage.enable()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def position(self) -> dict:
        return dict(self.stage.position)

    def move_by(self, delta: Mapping[str, float], *, persist: bool = True) -> dict:
        self._ensure_open()
        self.stage.move_by(delta, persist=persist)
        return self.position

    def move_to(self, position: Mapping[str, float], *, persist: bool = True) -> dict:
        self._ensure_open()
        self.stage.move_to(position, persist=persist)
        return self.position

    def capture(self, timeout_s: float = 5.0):
        """Trigger and return one frame, raising TimeoutError instead of None."""
        self._ensure_open()
        return self.camera.capture(timeout_s=timeout_s)

    def snapshot(self) -> dict:
        self._ensure_open()
        return {
            "position": self.position,
            "stage": self.stage.describe(),
            "camera": {
                "exposure_s": self.camera.get_exposure_time(),
                "gain": self.camera.get_gain(),
                "pixel_size_um": self.camera._pixel_size,
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.camera.disable()
        finally:
            self.db.close()
            self._closed = True

    def __enter__(self) -> "MicroscopeSimulator":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("microscope simulator is closed")


def create_simulator(config: SimulatorConfig | None = None, **kwargs) -> MicroscopeSimulator:
    """Create a simulator from a config or keyword overrides."""
    if config is not None and kwargs:
        raise TypeError("pass either config or keyword settings, not both")
    return MicroscopeSimulator(config or SimulatorConfig(**kwargs))
