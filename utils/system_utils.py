"""
System and configuration utilities for the 3D Cellular Automaton project.

This module provides functions for system checks, configuration resolution,
and runtime statistics collection.
"""

import os
import argparse
import re
import psutil
from pathlib import Path
from typing import Optional


def resolve_config_path() -> str:
    """
    Resolve config path from CLI --config, env, or default.
    
    Returns:
        str: Path to the configuration file.
    """
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", "-c")
        ns, _ = parser.parse_known_args()
        if getattr(ns, "config", None):
            return str(ns.config)
    except Exception:
        pass
    env = os.environ.get("CA3D_CONFIG") or os.environ.get("CONFIG_PATH")
    return env or "config.json"


def check_cuda_available() -> bool:
    """
    Check if CUDA is available on the system.
    
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    try:
        import pycuda.driver as _drv  # type: ignore
        _drv.init()
        return _drv.Device.count() > 0
    except Exception:
        return False


def collect_runtime_stats(log_path: Path, elapsed_time: float, planned_iterations: int) -> None:
    """
    Collect and log runtime statistics.
    
    Args:
        log_path (Path): Path to the log file.
        elapsed_time (float): Total elapsed time in seconds.
        planned_iterations (int): Number of planned iterations.
    """
    try:
        last_step: Optional[int] = None
        try:
            txt = Path(log_path).read_text(encoding="utf-8", errors="ignore")
            m_iter = None
            for m_iter in re.finditer(r"Snapshot\s+(\d+)\s+saved", txt):
                pass
            if m_iter:
                last_step = int(m_iter.group(1))
        except Exception:
            last_step = None
        
        # Fallback to planned iterations if we could not parse a last step
        executed_steps = int(planned_iterations if last_step is None else last_step)
        it_per_sec = (executed_steps / elapsed_time) if elapsed_time > 0 else 0.0
        
        # Process memory info (current and peak if available)
        p = psutil.Process()
        mi = p.memory_info()
        rss = getattr(mi, 'rss', 0)
        peak = getattr(mi, 'peak_wset', None)
        if peak is None:
            peak = getattr(mi, 'peak_rss', None)
        
        stats_lines = [
            f"Executed steps (approx): {executed_steps}",
            f"Mean iterations/second: {it_per_sec:.3f}",
            f"Process RSS now: {format_bytes(rss)}",
        ]
        if peak:
            stats_lines.append(f"Peak working set: {format_bytes(peak)}")
        
        from .logging_utils import append_log
        for line in stats_lines:
            append_log(log_path, line)
    except Exception:
        pass


def format_bytes(n: int | float) -> str:
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
