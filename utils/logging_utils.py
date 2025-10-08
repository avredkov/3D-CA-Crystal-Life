"""
Logging utilities for the 3D Cellular Automaton project.

This module provides centralized logging functionality to avoid duplication
across different parts of the codebase.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def append_log(log_path: Path, message: str, banner: Optional[str] = None) -> None:
    """
    Append a timestamped message to a log file.
    
    Args:
        log_path (Path): Path to the log file.
        message (str): Message to log.
        banner (Optional[str]): Banner to write if file doesn't exist.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_exists = log_path.exists()
        
        with open(log_path, "a", encoding="utf-8") as log_file:
            if not log_exists and banner:
                log_file.write(banner + "\n")
            
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] {message}\n")
    except OSError:
        pass


def append_boxed_log(log_path: Path, title: str, payload: Any, banner: Optional[str] = None) -> None:
    """
    Append a boxed section to a log file with JSON-formatted data.
    
    Args:
        log_path (Path): Path to the log file.
        title (str): Section title.
        payload (Any): Data to serialize and log.
        banner (Optional[str]): Banner to write if file doesn't exist.
    """
    try:
        import json
        from .data_utils import convert
        
        serialized = json.dumps(payload, indent=2, default=convert)
        data_lines = serialized.splitlines()
        section_lines = [f"{title}"] + data_lines
        
        log_exists = log_path.exists()
        with open(log_path, "a", encoding="utf-8") as log_file:
            if not log_exists and banner:
                log_file.write(banner + "\n")
            
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] {title}\n")
            
            # Format as boxed content
            content_width = len(banner.splitlines()[0]) - 4 if banner else 80
            boxed = _format_box(section_lines, content_width)
            log_file.write(boxed + "\n")
    except OSError:
        pass


def _format_box(lines: list[str], content_width: int) -> str:
    """Format lines as a boxed section."""
    boxed = []
    spacer = f"* {' ' * content_width} *"
    boxed.append(spacer)
    
    for line in lines:
        if not line.strip():
            boxed.append(spacer)
            continue
        
        import textwrap
        wrapped = textwrap.wrap(
            line,
            width=content_width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [" "]
        
        for segment in wrapped:
            boxed.append(f"* {segment.ljust(content_width)} *")
    
    boxed.append(spacer)
    return "\n".join(boxed)
