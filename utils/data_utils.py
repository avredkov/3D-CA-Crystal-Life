"""
Data utilities for the 3D Cellular Automaton project.

This module provides common data conversion and formatting functions.
"""

import numpy as np
from typing import Union, Any
from pathlib import Path


def convert(o):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        o: Object to convert.
        
    Returns:
        Converted object suitable for JSON serialization.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.ndarray, np.generic)):
        return o.tolist()
    return o


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds (float): Duration in seconds.
        
    Returns:
        str: Formatted duration string (e.g., "1d 02:30:45" or "02:30:45").
    """
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    
    if d > 0:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_bytes(n: Union[int, float]) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        n (Union[int, float]): Number of bytes.
        
    Returns:
        str: Formatted byte string (e.g., "1.5 MB").
    """
    n = float(max(0, n or 0))
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def is_valid_pow2_size(v: int) -> bool:
    """
    Check if a value is a valid power of 2 size for the lattice.
    
    Args:
        v (int): Value to check.
        
    Returns:
        bool: True if valid power of 2 in range [4, 512].
    """
    return v in (4, 8, 16, 32, 64, 128, 256, 512)


def write_json(path: Path, data: Any) -> None:
    """
    Write JSON data to a file using UTF-8 encoding.

    Args:
        path (Path): Destination file path.
        data (Any): Serializable content.
    """
    import json
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, default=convert, ensure_ascii=False, indent=2)
