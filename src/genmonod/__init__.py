# NEEDS YOUR INPUT: nothing. Exposes the main pieces so
# `from genmonod import default_config, fit, Dataset` works directly.

__version__ = "0.1.0"

from .config import SystemConfig, MatrixSpec, default_config, DosingEvent, PassageEvent, add_dosing_event, add_passage_event, add_conjugative_transfer, set_subsets, add_production
from .physics import simulate, system_rhs
from .fitting import fit, FitResult
from .data_io import Dataset
from .strain_library import record_strain_params, apply_library
from .model_selection import compare_structures, build_structure_grid, summarize
from .sweep import set_param, ParamAxis, run_sweep, heatmap, trajectory_grid

__all__ = [
    "SystemConfig", "MatrixSpec", "default_config",
    "DosingEvent", "PassageEvent", "add_dosing_event", "add_passage_event", "add_conjugative_transfer",
    "set_subsets", "add_production",
    "simulate", "system_rhs",
    "fit", "FitResult",
    "Dataset",
    "record_strain_params", "apply_library",
    "compare_structures", "build_structure_grid", "summarize",
    "set_param", "ParamAxis", "run_sweep", "heatmap", "trajectory_grid",
]
