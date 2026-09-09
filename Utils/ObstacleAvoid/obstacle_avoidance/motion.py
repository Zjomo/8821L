"""Hardware-aligned XYZ motion model used by the simulator and Qt controls.

The model is deliberately deterministic: a move computes the same profile and
telemetry every time, while the callback is responsible for applying the new
position to a concrete device (the microscope simulator or a hardware adapter).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional


class MotionConfigError(ValueError):
    """Invalid motion settings or an unknown axis."""


class MotionLimitError(MotionConfigError):
    """A requested move exceeds a configured soft limit."""


class MotionStateError(RuntimeError):
    """Motion was requested while the stage is disabled or stopped."""


class Simulated874xController:
    """Deterministic 8742/8743-style step accounting for virtual runs.

    The real controllers receive signed step counts per motor channel.  This
    adapter does not drive hardware; it converts every virtual displacement to
    the equivalent channel, direction and step count so reports can be audited
    exactly like a motor experiment.
    """

    def __init__(self, steps_per_mm: float = 33333.0,
                 channels: Optional[Mapping[str, int]] = None) -> None:
        if steps_per_mm <= 0:
            raise MotionConfigError("steps_per_mm must be > 0")
        self.steps_per_mm = float(steps_per_mm)
        self.channels = {"x": 1, "y": 2, "z": 3}
        if channels:
            self.channels.update({str(k).lower(): int(v)
                                  for k, v in channels.items()})
        self.sequence = 0
        self.cumulative_steps = {axis: 0 for axis in self.channels}
        self._residual_steps = {axis: 0.0 for axis in self.channels}

    def command(self, delta_mm: Mapping[str, float], source: str = "virtual",
                task_id: str = "", track_id: int = -1) -> list[dict]:
        self.sequence += 1
        out = []
        for axis in ("x", "y", "z"):
            value = float(delta_mm.get(axis, 0.0) or 0.0)
            if abs(value) <= 1e-15:
                continue
            exact = value * self.steps_per_mm + self._residual_steps[axis]
            signed_steps = int(round(exact))
            self._residual_steps[axis] = exact - signed_steps
            if signed_steps == 0:
                continue
            self.cumulative_steps[axis] += signed_steps
            out.append({
                "controller": "8742/8743",
                "sequence": self.sequence,
                "channel": self.channels[axis],
                "axis": axis.upper(),
                "direction": "+" if signed_steps > 0 else "-",
                "signed_steps": signed_steps,
                "steps": abs(signed_steps),
                "delta_mm": value,
                "delta_um": value * 1000.0,
                "cumulative_steps": self.cumulative_steps[axis],
                "steps_per_mm": self.steps_per_mm,
                "source": source,
                "task_id": task_id,
                "track_id": track_id,
            })
        return out

    def snapshot(self) -> dict:
        return {"steps_per_mm": self.steps_per_mm,
                "cumulative_steps": dict(self.cumulative_steps),
                "channels": dict(self.channels)}


@dataclass(frozen=True)
class AxisMotionConfig:
    name: str
    minimum: float
    maximum: float
    steps_per_unit: float = 1.0
    max_speed: float = 100.0
    acceleration: float = 200.0
    deceleration: Optional[float] = None
    settle_s: float = 0.0

    def validate(self) -> "AxisMotionConfig":
        if not self.name or self.minimum >= self.maximum:
            raise MotionConfigError(f"invalid limits for axis {self.name!r}")
        if self.steps_per_unit <= 0:
            raise MotionConfigError(f"{self.name}.steps_per_unit must be > 0")
        if self.max_speed <= 0:
            raise MotionConfigError(f"{self.name}.max_speed must be > 0")
        if self.acceleration <= 0:
            raise MotionConfigError(f"{self.name}.acceleration must be > 0")
        if self.deceleration is not None and self.deceleration <= 0:
            raise MotionConfigError(f"{self.name}.deceleration must be > 0")
        if self.settle_s < 0:
            raise MotionConfigError(f"{self.name}.settle_s must be >= 0")
        return self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "steps_per_unit": self.steps_per_unit,
            "max_speed": self.max_speed,
            "acceleration": self.acceleration,
            "deceleration": self.deceleration,
            "settle_s": self.settle_s,
        }


def default_motion_config(x_limits=(-1000.0, 1000.0),
                          y_limits=(-1000.0, 1000.0),
                          z_limits=(-50.0, 50.0)) -> "MotionConfig":
    return MotionConfig(
        axes={
            "x": AxisMotionConfig("x", *x_limits, steps_per_unit=1.0),
            "y": AxisMotionConfig("y", *y_limits, steps_per_unit=1.0),
            "z": AxisMotionConfig("z", *z_limits, steps_per_unit=1.0,
                                  max_speed=20.0, acceleration=40.0),
        }
    )


@dataclass(frozen=True)
class MotionConfig:
    axes: Mapping[str, AxisMotionConfig] = field(default_factory=lambda: {
        "x": AxisMotionConfig("x", -1000.0, 1000.0),
        "y": AxisMotionConfig("y", -1000.0, 1000.0),
        "z": AxisMotionConfig("z", -50.0, 50.0, max_speed=20.0,
                              acceleration=40.0),
    })
    soft_limits: bool = True
    collision_interlock: bool = True

    def __post_init__(self) -> None:
        axes = dict(self.axes)
        if set(axes) != {"x", "y", "z"}:
            raise MotionConfigError("XYZ configuration must contain x, y and z")
        for name, axis in axes.items():
            if name != axis.name:
                raise MotionConfigError(f"axis key/name mismatch: {name}/{axis.name}")
            axis.validate()
        object.__setattr__(self, "axes", axes)

    def axis(self, name: str) -> AxisMotionConfig:
        try:
            return self.axes[name.lower()]
        except KeyError as exc:
            raise MotionConfigError(f"unknown axis {name!r}") from exc

    def to_dict(self) -> dict:
        return {
            "axes": {name: axis.to_dict() for name, axis in self.axes.items()},
            "soft_limits": self.soft_limits,
            "collision_interlock": self.collision_interlock,
        }


@dataclass(frozen=True)
class MotionTelemetry:
    sequence: int
    timestamp: float
    source: str
    requested_delta: Dict[str, float]
    applied_delta: Dict[str, float]
    position_before: Dict[str, float]
    position_after: Dict[str, float]
    steps: Dict[str, int]
    duration_s: float
    average_speed: Dict[str, float]
    peak_speed: Dict[str, float]
    acceleration: Dict[str, float]
    deceleration: Dict[str, float]
    limit_hit: Dict[str, bool]
    state: str = "complete"

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "source": self.source,
            "requested_delta": dict(self.requested_delta),
            "applied_delta": dict(self.applied_delta),
            "position_before": dict(self.position_before),
            "position_after": dict(self.position_after),
            "steps": dict(self.steps),
            "duration_s": self.duration_s,
            "average_speed": dict(self.average_speed),
            "peak_speed": dict(self.peak_speed),
            "acceleration": dict(self.acceleration),
            "deceleration": dict(self.deceleration),
            "limit_hit": dict(self.limit_hit),
            "state": self.state,
        }


def _profile(distance: float, axis: AxisMotionConfig) -> tuple[float, float]:
    """Return (duration, peak_speed) for a trapezoid/triangle profile."""
    d = abs(float(distance))
    if d <= 1e-12:
        return 0.0, 0.0
    acc = axis.acceleration
    dec = axis.deceleration or acc
    vmax = axis.max_speed
    distance_to_vmax = 0.5 * vmax * vmax * (1.0 / acc + 1.0 / dec)
    if d < distance_to_vmax:
        peak = math.sqrt(2.0 * d / (1.0 / acc + 1.0 / dec))
        return peak / acc + peak / dec, peak
    return vmax / acc + (d - distance_to_vmax) / vmax + vmax / dec, vmax


class VirtualXYZStage:
    """Deterministic XYZ stage with atomic limits and hardware-like telemetry."""

    def __init__(self, config: MotionConfig | None = None,
                 initial_position: Optional[Mapping[str, float]] = None,
                 on_move: Optional[Callable[[Mapping[str, float]], None]] = None,
                 home_position: Optional[Mapping[str, float]] = None):
        self.config = config or MotionConfig()
        initial = {name: 0.0 for name in self.config.axes}
        initial.update({str(k).lower(): float(v)
                       for k, v in (initial_position or {}).items()})
        for name, value in initial.items():
            axis = self.config.axis(name)
            if not axis.minimum <= value <= axis.maximum:
                raise MotionLimitError(f"initial {name}={value} outside limits")
        self._position = initial
        # home 目标位置（未指定的轴回 0）
        self._home_position = {str(k).lower(): float(v)
                               for k, v in (home_position or {}).items()}
        self._on_move = on_move
        self._enabled = True
        self._sequence = 0
        self._history: list[MotionTelemetry] = []
        self._last: Optional[MotionTelemetry] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def position(self) -> Dict[str, float]:
        return dict(self._position)

    @property
    def last_telemetry(self) -> Optional[MotionTelemetry]:
        return self._last

    @property
    def history(self) -> tuple[MotionTelemetry, ...]:
        return tuple(self._history)

    def enable(self) -> None:
        self._enabled = True

    def configure_axis(self, name: str, **changes) -> None:
        """Update a validated axis profile without changing its position."""
        name = str(name).lower()
        current = self.config.axis(name)
        data = current.to_dict()
        data.update(changes)
        data["name"] = name
        axis = AxisMotionConfig(**data).validate()
        axes = dict(self.config.axes)
        axes[name] = axis
        self.config = MotionConfig(axes=axes,
                                   soft_limits=self.config.soft_limits,
                                   collision_interlock=self.config.collision_interlock)

    def disable(self) -> None:
        self._enabled = False

    def stop(self) -> None:
        """Model an emergency stop; a later explicit enable is required."""
        self._enabled = False

    estop = stop

    def _check_target(self, target: Mapping[str, float]) -> Dict[str, bool]:
        limit_hit = {}
        for name, value in target.items():
            axis = self.config.axis(name)
            hit = value < axis.minimum or value > axis.maximum
            limit_hit[name] = hit
            if hit and self.config.soft_limits:
                raise MotionLimitError(
                    f"{name} target {value:.6g} outside "
                    f"[{axis.minimum:.6g}, {axis.maximum:.6g}]")
        return limit_hit

    def move_by(self, delta: Mapping[str, float], source: str = "ui") -> MotionTelemetry:
        if not self._enabled:
            raise MotionStateError("stage is disabled or stopped")
        requested = {str(k).lower(): float(v) for k, v in delta.items()}
        if not requested:
            raise MotionConfigError("move_by requires at least one axis")
        unknown = set(requested) - set(self.config.axes)
        if unknown:
            raise MotionConfigError(f"unknown axis {sorted(unknown)}")
        before = self.position
        target = dict(before)
        for name, amount in requested.items():
            if not math.isfinite(amount):
                raise MotionConfigError(f"non-finite delta for axis {name}")
            target[name] += amount
        limit_hit = self._check_target(target)

        profiles = {name: _profile(requested.get(name, 0.0), self.config.axis(name))
                    for name in self.config.axes}
        duration = max((p[0] for p in profiles.values()), default=0.0)
        steps = {name: int(round(requested.get(name, 0.0) *
                                 self.config.axis(name).steps_per_unit))
                 for name in self.config.axes}
        average = {name: (abs(requested.get(name, 0.0)) / duration
                          if duration else 0.0)
                   for name in self.config.axes}
        peak = {name: profiles[name][1] for name in self.config.axes}
        accel = {name: (self.config.axis(name).acceleration
                        if abs(requested.get(name, 0.0)) > 0 else 0.0)
                 for name in self.config.axes}
        decel = {name: (self.config.axis(name).deceleration or
                        self.config.axis(name).acceleration
                        if abs(requested.get(name, 0.0)) > 0 else 0.0)
                 for name in self.config.axes}
        self._sequence += 1
        telemetry = MotionTelemetry(
            sequence=self._sequence, timestamp=time.time(), source=str(source),
            requested_delta=dict(requested), applied_delta={
                name: target[name] - before[name] for name in self.config.axes},
            position_before=before, position_after=target, steps=steps,
            duration_s=duration, average_speed=average, peak_speed=peak,
            acceleration=accel, deceleration=decel, limit_hit=limit_hit)
        # Apply the device callback only after all validation/profile work; a
        # callback failure leaves the stage position and history untouched.
        if self._on_move is not None:
            try:
                self._on_move(dict(target))
            except Exception:
                raise
        self._position = target
        self._last = telemetry
        self._history.append(telemetry)
        return telemetry

    def move_to(self, position: Mapping[str, float], source: str = "ui") -> MotionTelemetry:
        target = {str(k).lower(): float(v) for k, v in position.items()}
        unknown = set(target) - set(self.config.axes)
        if unknown:
            raise MotionConfigError(f"unknown axis {sorted(unknown)}")
        return self.move_by({name: value - self._position[name]
                             for name, value in target.items()}, source=source)

    def home(self, axes=None, source: str = "home") -> MotionTelemetry:
        names = [str(a).lower() for a in (axes or self.config.axes)]
        return self.move_to({name: self._home_position.get(name, 0.0)
                             for name in names}, source=source)

    def zero(self, axes=None) -> None:
        names = [str(a).lower() for a in (axes or self.config.axes)]
        for name in names:
            self.config.axis(name)
            self._position[name] = 0.0

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "position": self.position,
            "config": self.config.to_dict(),
            "last_telemetry": (self._last.to_dict() if self._last else None),
        }
