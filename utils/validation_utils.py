"""
Validation utilities for the 3D Cellular Automaton project.

This module provides functions for validating system requirements,
configuration parameters, and other pre-simulation checks.
"""

import sys
from pathlib import Path
from typing import List, Tuple
from .system_utils import check_cuda_available
from .data_utils import is_valid_pow2_size
from .logging_utils import append_log


def validate_cuda_availability(cfg, log_path: Path) -> None:
    """
    Validate CUDA availability and exit if not available.
    
    Args:
        cfg: Configuration object.
        log_path (Path): Path to the log file.
        
    Raises:
        SystemExit: If CUDA is not available.
    """
    if not check_cuda_available():
        append_log(log_path, "CUDA-enabled graphic card is not found. Please ensure it is present and all drivers are installed correctly")
        print("CUDA-enabled graphic card is not found. Please ensure it is present and all drivers are installed correctly")
        sys.exit(1)


def validate_lattice_sizes(cfg, log_path: Path) -> None:
    """
    Validate lattice sizes are valid powers of 2.
    
    Args:
        cfg: Configuration object.
        log_path (Path): Path to the log file.
        
    Raises:
        SystemExit: If lattice sizes are invalid.
    """
    invalid: List[Tuple[str, int]] = []
    for label, val in (
        ("lattice_size_x", int(cfg.lattice_size_x)),
        ("lattice_size_y", int(cfg.lattice_size_y)),
        ("lattice_size_z", int(cfg.lattice_size_z)),
    ):
        if not is_valid_pow2_size(val):
            invalid.append((label, val))
    
    if invalid:
        allowed = "4, 8, 16, 32, 64, 128, 256, 512"
        details = ", ".join([f"{k}={v}" for k, v in invalid])
        append_log(log_path, f"Invalid lattice sizes: {details}. Allowed powers of two in [4..512]. Aborting run.")
        print(f"Error: Invalid lattice sizes ({details}). Allowed values: {allowed}")
        sys.exit(1)


def validate_system_requirements(cfg, log_path: Path) -> None:
    """
    Perform all system requirement validations.
    
    Args:
        cfg: Configuration object.
        log_path (Path): Path to the log file.
        
    Raises:
        SystemExit: If any validation fails.
    """
    validate_cuda_availability(cfg, log_path)
    validate_lattice_sizes(cfg, log_path)
