# NEEDS YOUR INPUT: nothing. Exposes the main pieces so
# `from genmonod import default_config, fit, Dataset` works directly.

__version__ = "0.1.0"

from .config import SystemConfig, MatrixSpec, default_config
from .physics import simulate, system_rhs
from .fitting import fit, FitResult
from .data_io import Dataset

__all__ = [
    "SystemConfig", "MatrixSpec", "default_config",
    "simulate", "system_rhs",
    "fit", "FitResult",
    "Dataset",
]
