"""Alg2: fixed-beam microscope motion adapter.

Alg1 moves the selected particle directly in the synthetic world.  Alg2 models
the physical arrangement used by a microscope: the illumination/beam is fixed
and the XYZ stage is moved.  The stage motion is applied to the microscope
simulator, while the non-selected sample objects are compensated in sample
coordinates so that only the optically selected particle moves relative to the
fixed beam.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .simulator import StageError, XYStageProtocol


@dataclass(frozen=True)
class Alg2Config:
    """XYZ setup used by Alg2 (micrometres)."""

    z_safe_um: float = 5.0
    z_focus_um: float = 0.0
    focus_settle_s: float = 0.0


class Alg2Stage(XYStageProtocol):
    """XYStageProtocol facade backed by a real/simulated XYZ stage.

    The controller continues to issue small XY moves through the existing
    protocol.  ``prepare_focus`` explicitly exercises the Z axis before a task;
    the underlying microscope stage remains available for UI XYZ jogs and
    telemetry.
    """

    def __init__(self, world, track_id: Optional[int] = None,
                 config: Optional[Alg2Config] = None) -> None:
        self.world = world
        self.track_id = track_id
        self.config = config or Alg2Config()
        maker = getattr(world, "make_alg2_stage", None)
        if not callable(maker):
            raise StageError("world does not provide fixed-beam Alg2 stage")
        # In the virtual microscope the fixed beam selects one particle.  The
        # selected particle is therefore the only object updated by XY motion;
        # the rest of the sample remains a static obstacle field.
        self._xy = maker(track_id)

    def prepare_focus(self) -> None:
        stage = getattr(self.world, "motion_stage", None)
        if stage is None:
            return
        # Safe excursion followed by focus position makes Z motion explicit and
        # deterministic even when the stage starts already at focus.
        stage.enable()
        current = float(stage.position.get("z", 0.0))
        if abs(current - self.config.z_safe_um) > 1e-9:
            stage.move_to({"z": self.config.z_safe_um}, source="alg2-z-safe")
        if abs(self.config.z_focus_um - self.config.z_safe_um) > 1e-9:
            stage.move_to({"z": self.config.z_focus_um}, source="alg2-z-focus")

    def move_by(self, dx_mm: float, dy_mm: float, **kwargs) -> bool:
        return self._xy.move_by(dx_mm, dy_mm, **kwargs)

    def stop_all(self) -> None:
        stop = getattr(self._xy, "stop_all", None)
        if callable(stop):
            stop()
        stage = getattr(self.world, "motion_stage", None)
        if stage is not None:
            stage.stop()

    @property
    def last_command(self):
        return self._xy.last_command
