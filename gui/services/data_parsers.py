from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Iterable


@dataclass
class Series:
    name: str
    x: List[float]
    y: List[float]
    units: str | None = None


def _read_rows(path: Path) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip headers/comments used by save_helpers; be tolerant of separators after 'Timestep'
            if (not line or line.startswith("Timestep") or line.startswith("Dummy") or line.startswith("Atoms ")):
                # skip headers/comments used by save_helpers
                continue
            yield line


def parse_system_evolution_stats(path: str) -> Dict[str, Series]:
    p = Path(path)
    x: List[float] = []
    mobile: List[float] = []
    cryst: List[float] = []
    alpha: List[float] = []
    size: List[float] = []
    for row in _read_rows(p):
        parts = row.split()
        if len(parts) < 5:
            continue
        x.append(float(parts[0])); mobile.append(float(parts[1])); cryst.append(float(parts[2]))
        alpha.append(float(parts[3])); size.append(float(parts[4]))
    return {
        "Mobile atoms": Series("Mobile atoms", x, mobile),
        "Crystalline atoms": Series("Crystalline atoms", x, cryst),
        "Alpha": Series("Alpha", x, alpha),
        "Crystal size": Series("Crystal size", x, size),
    }


def parse_statistics_on_available_states(path: str) -> Dict[str, Series]:
    p = Path(path)
    x: List[float] = []
    terrace: List[float] = []
    step: List[float] = []
    kink: List[float] = []
    for row in _read_rows(p):
        parts = row.split()
        if len(parts) < 4:
            continue
        x.append(float(parts[0])); terrace.append(float(parts[1])); step.append(float(parts[2])); kink.append(float(parts[3]))
    return {
        "Terrace": Series("Terrace", x, terrace),
        "Step": Series("Step", x, step),
        "Kink": Series("Kink", x, kink),
    }


def parse_statistics_on_coordination(path: str) -> Dict[str, Series]:
    p = Path(path)
    x: List[float] = []
    cols: Dict[str, List[float]] = {f"Coord {i}": [] for i in range(7)}
    # Header has 7 counts and 7 fractions; we focus on counts (first 7 after timestep)
    for row in _read_rows(p):
        parts = row.split()
        if len(parts) < 8:
            continue
        x.append(float(parts[0]))
        for i in range(7):
            cols[f"Coord {i}"].append(float(parts[1 + i]))
    return {k: Series(k, x, v) for k, v in cols.items()}


def parse_macro_events(path: str) -> Dict[str, Series]:
    p = Path(path)
    # Read first non-empty line as header and split by any whitespace or tabs
    import re
    header_tokens: List[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            header_tokens = re.split(r"\s+|\t+", line)
            break
    # Remove common time column labels
    time_labels = {"timestep", "step", "time", "t"}
    categories = [c for c in header_tokens if c and c.lower() not in time_labels]
    x: List[float] = []
    data: Dict[str, List[float]] = {c: [] for c in categories}
    for row in _read_rows(p):
        parts = row.split()
        if len(parts) < (1 + len(categories)):
            continue
        x.append(float(parts[0]))
        for i, c in enumerate(categories):
            data[c].append(float(parts[1 + i]))
    return {c: Series(c, x, ys) for c, ys in data.items()}



def parse_generic_tabular(path: str) -> Dict[str, Series]:
    """
    Generic, dynamic parser for plugin .dat-like files with a header.
    Expects first line to contain column names (first should be Timestep or time).
    Remaining lines are whitespace- or tab-separated numeric data.
    """
    p = Path(path)
    # Read header (first non-empty line)
    header = ""
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                header = line
                break
    if not header:
        return {}
    cols = [c for c in header.replace("\t", " ").split() if c]
    # Identify timestep column name (fallback to first column)
    time_col_name = None
    for cand in ("Timestep", "Step", "Time", "t", "frame"):
        if cand in cols:
            time_col_name = cand
            break
    if time_col_name is None:
        time_col_name = cols[0]
    series_names = [c for c in cols if c != time_col_name]
    x: List[float] = []
    data: Dict[str, List[float]] = {c: [] for c in series_names}
    # Parse remaining lines using _read_rows (skips typical headers)
    for row in _read_rows(p):
        parts = row.split()
        if len(parts) < len(cols):
            continue
        try:
            x.append(float(parts[cols.index(time_col_name)]))
            for c in series_names:
                idx = cols.index(c)
                data[c].append(float(parts[idx]))
        except Exception:
            # Skip malformed rows
            continue
    return {c: Series(c, x, ys) for c, ys in data.items()}


def load_series(path: str) -> Dict[str, Series]:
    """
    Unified API returning a mapping of series name -> Series for any supported output file.
    Routes to specific parsers by filename; falls back to generic tabular parser.
    """
    lp = Path(path).name.lower()
    try:
        if "system_evolution_stats" in lp:
            return parse_system_evolution_stats(path)
        if "statistics_on_available_states" in lp:
            return parse_statistics_on_available_states(path)
        if "statistics_on_coordination" in lp:
            return parse_statistics_on_coordination(path)
        if "macro_events" in lp:
            return parse_macro_events(path)
        # Fallback
        return parse_generic_tabular(path)
    except Exception:
        # Last-resort fallback to empty dict on parse errors
        return {}


