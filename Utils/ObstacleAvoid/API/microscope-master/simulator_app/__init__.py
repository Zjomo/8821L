"""Reusable microscope-master simulation package.

The public API is intentionally small.  Most projects should import
``MicroscopeSimulator`` or ``create_simulator`` instead of manipulating the
SQLite database and simulator classes directly.  The legacy ``main.py`` GUI
continues to work unchanged.
"""

from .api import (DEFAULT_STAGE_LIMITS, MicroscopeSimulator, SimulatorConfig,
                  create_simulator)
from .db import Database
from .devices import SampleAwareCamera, SQLiteStage, SQLiteStageAxis

__all__ = [
    "DEFAULT_STAGE_LIMITS",
    "Database",
    "MicroscopeSimulator",
    "SampleAwareCamera",
    "SQLiteStage",
    "SQLiteStageAxis",
    "SimulatorConfig",
    "create_simulator",
]

__version__ = "0.1.0"
