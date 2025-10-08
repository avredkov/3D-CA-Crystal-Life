"""
Utilities package for the 3D Cellular Automaton project.

This package contains various utility modules for logging, data processing,
mathematical operations, configuration handling, and system validation.
"""

# Import commonly used utilities for easy access
from .data_utils import convert, format_duration, format_bytes, write_json, is_valid_pow2_size
from .logging_utils import append_log, append_boxed_log
from .bounds_utils import centered_bounds, simple_bounds, calculate_bounds
from .system_utils import resolve_config_path, check_cuda_available, collect_runtime_stats
from .validation_utils import validate_system_requirements, validate_cuda_availability, validate_lattice_sizes
from .math_utils import generate_log_points
from .config_utils import serialize_config_metadata

__all__ = [
    # Data utilities
    'convert', 'format_duration', 'format_bytes', 'write_json', 'is_valid_pow2_size',
    # Logging utilities
    'append_log', 'append_boxed_log',
    # Bounds utilities
    'centered_bounds', 'simple_bounds', 'calculate_bounds',
    # System utilities
    'resolve_config_path', 'check_cuda_available', 'collect_runtime_stats',
    # Validation utilities
    'validate_system_requirements', 'validate_cuda_availability', 'validate_lattice_sizes',
    # Math utilities
    'generate_log_points',
    # Config utilities
    'serialize_config_metadata',
]
