"""
Bounds calculation utilities for the 3D Cellular Automaton project.

This module provides functions for calculating boundary positions and dimensions
for various initialization modes.
"""

from typing import Tuple


def calculate_bounds(center: int, size: int, dim: int, mode: str = "centered") -> Tuple[int, int]:
    """
    Calculate start and end bounds for a given center, size, and dimension.
    
    Args:
        center (int): Center position.
        size (int): Size of the region.
        dim (int): Dimension size (max valid index + 1).
        mode (str): Calculation mode - "centered" or "simple".
        
    Returns:
        Tuple[int, int]: (start, end) bounds.
    """
    left = size // 2
    right = size - left  # ensures length == size, handles even sizes
    start = center - left
    end = center + right
    
    if mode == "centered":
        # More complex clipping logic for centered mode
        if start < 0:
            start = 0
            end = size
        if end > dim:
            end = dim
            start = dim - size
    else:  # simple mode
        # Simpler clipping logic
        start = max(0, start)
        end = min(dim, end)
        
        # Adjust if clipped
        if end - start < size:
            if start == 0:
                end = min(dim, size)
            else:
                start = max(0, end - size)
    
    return start, end


def centered_bounds(center: int, size: int, dim: int) -> Tuple[int, int]:
    """
    Calculate centered bounds with complex clipping logic.
    
    Args:
        center (int): Center position.
        size (int): Size of the region.
        dim (int): Dimension size (max valid index + 1).
        
    Returns:
        Tuple[int, int]: (start, end) bounds.
    """
    return calculate_bounds(center, size, dim, mode="centered")


def simple_bounds(center: int, size: int, dim: int) -> Tuple[int, int]:
    """
    Calculate simple bounds with basic clipping logic.
    
    Args:
        center (int): Center position.
        size (int): Size of the region.
        dim (int): Dimension size (max valid index + 1).
        
    Returns:
        Tuple[int, int]: (start, end) bounds.
    """
    return calculate_bounds(center, size, dim, mode="simple")
