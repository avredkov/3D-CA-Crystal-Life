"""
Main entry point for the 3D Cellular Automaton simulation.

This module provides a clean command-line interface for running simulations
with configuration validation and runtime statistics.
"""

import CA3D_functions as CA
from pathlib import Path
from config import load_config
import sys
import time
from utils.logging_utils import append_log
from utils.data_utils import format_duration
from utils.system_utils import resolve_config_path, collect_runtime_stats

# Load simulation configuration
cfg = load_config(resolve_config_path())

# Validate system requirements and prepare for simulation
log_path = Path(cfg.output_dir) / "system_info.log"
CA.validate_and_prepare_simulation(cfg, log_path)


params = cfg.to_params()

#prepare initial state of the cells
initial_atoms = CA.prepare_initial_atoms(cfg)

config_metadata = {
    "parameters": cfg.model_dump(),
    "ruleset": cfg.ruleset_name,
    "recipe_values": dict(getattr(cfg, "recipe_values", {}) or {}),
}


#run the simulation
_t0 = time.time()
CA.CA_3D_experiment(
    params,
    initial_atoms=initial_atoms,
    ruleset=cfg.ruleset_name,
    config_metadata=config_metadata,
)
elapsed = time.time() - _t0
msg = f"Total time is: {format_duration(elapsed)}"
append_log(log_path, msg)
print('Calculations finished')
print(msg)

# Derive simple runtime statistics and append to log
collect_runtime_stats(log_path, elapsed, cfg.num_iterations)





