#%%
import CA3D_functions as CA
from pathlib import Path
from config import load_config
import sys
import os
import argparse
import time
import re
import psutil

def _resolve_config_path() -> str:
    """Resolve config path from CLI --config, env, or default."""
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

# Load simulation configuration
cfg = load_config(_resolve_config_path())

# Verify CUDA availability early and exit with a log message if missing
def _check_cuda_available() -> bool:
    try:
        import pycuda.driver as _drv  # type: ignore
        _drv.init()
        return _drv.Device.count() > 0
    except Exception:
        return False

if not _check_cuda_available():
    _append_log(Path(cfg.output_dir) / "system_info.log", "CUDA-enabled graphic card is not found. Please ensure it is present and all drivers are installed correctly")
    print("CUDA-enabled graphic card is not found. Please ensure it is present and all drivers are installed correctly")
    sys.exit(1)

def _is_valid_pow2_size(v: int) -> bool:
    return v in (4, 8, 16, 32, 64, 128, 256, 512)

def _append_log(log_path: Path, message: str) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            from datetime import datetime as _dt
            f.write(f"[{_dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except OSError:
        pass

# Ensure output directory exists (and validate sizes before running)
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
log_path = Path(cfg.output_dir) / "system_info.log"

invalid = []
for label, val in (
    ("lattice_size_x", int(cfg.lattice_size_x)),
    ("lattice_size_y", int(cfg.lattice_size_y)),
    ("lattice_size_z", int(cfg.lattice_size_z)),
):
    if not _is_valid_pow2_size(val):
        invalid.append((label, val))

if invalid:
    allowed = "4, 8, 16, 32, 64, 128, 256, 512"
    details = ", ".join([f"{k}={v}" for k, v in invalid])
    _append_log(log_path, f"Invalid lattice sizes: {details}. Allowed powers of two in [4..512]. Aborting run.")
    print(f"Error: Invalid lattice sizes ({details}). Allowed values: {allowed}")
    sys.exit(1)


params = cfg.to_params()

#prepare initial state of the cells
initial_atoms = CA.prepare_initial_atoms(cfg)

config_metadata = {
    "parameters": cfg.model_dump(),
    "ruleset": cfg.ruleset_name,
    "recipe_values": dict(getattr(cfg, "recipe_values", {}) or {}),
}

def _format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"

#run the simulation
_t0 = time.time()
CA.CA_3D_experiment(
    params,
    initial_atoms=initial_atoms,
    ruleset=cfg.ruleset_name,
    config_metadata=config_metadata,
)
elapsed = time.time() - _t0
msg = f"Total time is: {_format_duration(elapsed)}"
_append_log(log_path, msg)
print('Calculations finished')
print(msg)

# Derive simple runtime statistics and append to log
try:
    last_step: int | None = None
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
    executed_steps = int(cfg.num_iterations if last_step is None else last_step)
    it_per_sec = (executed_steps / elapsed) if elapsed > 0 else 0.0
    # Process memory info (current and peak if available)
    p = psutil.Process()
    mi = p.memory_info()
    rss = getattr(mi, 'rss', 0)
    peak = getattr(mi, 'peak_wset', None)
    if peak is None:
        peak = getattr(mi, 'peak_rss', None)
    def _fmt_bytes(n: int | float) -> str:
        n = float(max(0, n or 0))
        for unit in ('B','KB','MB','GB','TB'):
            if n < 1024.0:
                return f"{n:.2f} {unit}"
            n /= 1024.0
        return f"{n:.2f} PB"
    stats_lines = [
        f"Executed steps (approx): {executed_steps}",
        f"Mean iterations/second: {it_per_sec:.3f}",
        f"Process RSS now: {_fmt_bytes(rss)}",
    ]
    if peak:
        stats_lines.append(f"Peak working set: {_fmt_bytes(peak)}")
    for line in stats_lines:
        _append_log(log_path, line)
except Exception:
    pass





