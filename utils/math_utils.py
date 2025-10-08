"""
Mathematical utilities for the 3D Cellular Automaton project.

This module provides mathematical functions and algorithms used throughout
the simulation system.
"""

import numpy as np
from typing import Union


def generate_log_points(start: Union[int, float] = 1, end: Union[int, float] = 2000000, 
                       N: int = 100, skew: float = 0.5) -> np.ndarray:
    """
    Generate N timesteps spaced geometrically between start and end.

    Args:
        start (Union[int, float]): Inclusive start (> 0).
        end (Union[int, float]): Inclusive end (> start).
        N (int): Number of points to generate (>= 1).
        skew (float): Monotonic skew of density across the range. 1.0 produces
            standard geometric spacing (independent of log base). Values > 1
            bias density toward the end (fewer points near start, more near end).
            Values in (0, 1) bias density toward the start. Must be > 0.

    Returns:
        np.ndarray: int32 array of length N with monotonically increasing points.

    Notes:
        Standard "log spacing" is geometric and does not depend on the choice of
        logarithm base. The optional `skew` allows emphasizing density toward one
        end while remaining geometrically increasing overall.
    """
    start = float(max(1, start))
    end = float(max(start, end))
    N = int(max(1, N))
    skew = float(max(1e-6, skew))

    if N == 1 or start == end:
        return np.int32([int(round(start))])

    # Map uniform parameter t in [0, 1] through a power curve to skew density,
    # then exponentiate over the natural log interval. This yields geometric
    # spacing when skew == 1 and biases toward `end` when skew > 1.
    t = np.linspace(0.0, 1.0, N)
    t_skewed = t ** skew
    log_start = np.log(start)
    log_end = np.log(end)
    exponents = log_start + (log_end - log_start) * t_skewed
    values = np.exp(exponents)

    return np.int32(values)
