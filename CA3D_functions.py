import numpy as np
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.curandom as rnd
import numpy as np
import sys
import os
import platform
import psutil
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks, savgol_filter
from matplotlib.widgets import Button
import math as mt
from rules.loader import load_assign_rules
import kernels_3D as kernels
import itertools
from tqdm import tqdm
import cProfile
import json
import random
from scipy.ndimage import generic_filter
import json
import textwrap
from typing import Any, Dict, Optional, Union

from save_helpers import (
    save_coordinations,
    save_data,
    save_events,
    save_states,
    save_states_3d,
    save_system_stats,
    save_xyz_only,
    save_crystalline_age_means,
    save_site_age_means,
)
from utils.logging_utils import append_log, append_boxed_log
from utils.data_utils import convert, format_duration, format_bytes, write_json
from utils.bounds_utils import centered_bounds, simple_bounds
from utils.validation_utils import validate_system_requirements
from utils.math_utils import generate_log_points
from utils.config_utils import serialize_config_metadata




def _evaluate_drift_expression(
    expr: str,
    t: int,
    sizeX: int,
    sizeY: int,
    sizeZ: int,
    total_time: Optional[int] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """
    Evaluate a drift expression safely with restricted namespace.
    
    Args:
        expr: Expression string using variables t, x, y, z, np, sizeX, sizeY, sizeZ, total_time
        t: Current timestep
        sizeX, sizeY, sizeZ: Lattice dimensions
        total_time: Total number of iterations (optional)
        x, y, z: Optional coordinate arrays (for spatial evaluation)
        
    Returns:
        float for time-only expressions, np.ndarray for spatial expressions
    """
    # Safe namespace with only allowed functions
    safe_dict = {
        "np": np,
        "t": t,
        "sizeX": float(sizeX),
        "sizeY": float(sizeY),
        "sizeZ": float(sizeZ),
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "max": np.maximum,
        "min": np.minimum,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "atan2": np.arctan2,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "pi": np.pi,
        "e": np.e,
    }
    
    # Add total_time if provided
    if total_time is not None:
        safe_dict["total_time"] = float(total_time)
    
    # Add coordinate arrays if provided (for spatial evaluation)
    if x is not None:
        safe_dict["x"] = x
    if y is not None:
        safe_dict["y"] = y
    if z is not None:
        safe_dict["z"] = z
    
    try:
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        # Convert to numpy array/float and clamp to [-1.0, 1.0]
        if isinstance(result, np.ndarray):
            result = np.clip(result.astype(np.float32), -1.0, 1.0)
        else:
            # If spatial coordinates were provided but result is scalar, broadcast to match coordinate shape
            if x is not None:
                # Broadcast scalar to match coordinate array shape
                result = np.full_like(x, float(np.clip(float(result), -1.0, 1.0)), dtype=np.float32)
            else:
                result = float(np.clip(float(result), -1.0, 1.0))
        return result
    except Exception as e:
        # Fallback to zero on evaluation error
        if x is not None:
            return np.zeros_like(x, dtype=np.float32)
        return 0.0


def validate_and_prepare_simulation(cfg, log_path: Path) -> None:
    """
    Validate system requirements and prepare for simulation.
    
    Args:
        cfg: Configuration object.
        log_path (Path): Path to the log file.
        
    Raises:
        SystemExit: If validation fails.
    """
    # Ensure output directory exists
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Perform all validations
    validate_system_requirements(cfg, log_path)





def prepare_initial_atoms(cfg):
    if getattr(cfg, "rng_seed", None) is not None:
        np.random.seed(int(cfg.rng_seed))
    sizeX, sizeY, sizeZ = cfg.lattice_size_x, cfg.lattice_size_y, cfg.lattice_size_z
    atoms = np.int32((
        np.random.random(sizeX * sizeY * sizeZ) < cfg.initial_occupancy_fraction
    ).reshape(sizeX, sizeY, sizeZ))

    mode = cfg.init.mode
    if mode == 'single':
        s = max(1, int(cfg.init.seed_edge_length))
        # Ensure seed fits in all dimensions
        s = min(s, sizeX, sizeY, sizeZ)
        cx, cy, cz = sizeX//2, sizeY//2, sizeZ//2


        x0, x1 = centered_bounds(cx, s, sizeX)
        y0, y1 = centered_bounds(cy, s, sizeY)
        z0, z1 = centered_bounds(cz, s, sizeZ)
        atoms[x0:x1, y0:y1, z0:z1] = 2
        # Guarantee center voxel is set (covers size == 1 edge case explicitly)
        atoms[cx, cy, cz] = 2
    elif mode == 'multiple':
        s = max(1, int(cfg.init.seed_edge_length))
        n = max(1, int(cfg.init.num_seeds))
        for _ in range(n):
            x = np.random.randint(0, max(1, sizeX - s + 1))
            y = np.random.randint(0, max(1, sizeY - s + 1))
            z = np.random.randint(0, max(1, sizeZ - s + 1))
            atoms[x:x+s, y:y+s, z:z+s] = 2
    elif mode == 'flat_bottom':
        layers = max(0, int(cfg.init.num_flat_layers))
        if layers > 0:
            atoms[:, :, 0:layers] = 2
    elif mode == 'python_code':
        array = atoms
        code = cfg.init.init_python_code or ""
        exec(code, {"np": np}, {"array": array})
        atoms = array
    elif mode == 'none':
        pass
    elif mode == 'stepped_bottom':
        N = max(1, int(cfg.init.num_terraces))
        terrace_height = 1
        # Evenly distribute step starts across full Y without a large last terrace
        y_starts = np.floor(np.linspace(0, sizeY, N, endpoint=False)).astype(int)
        for k, y_start in enumerate(y_starts):
            z_top = min(sizeZ, (k + 1) * terrace_height)
            atoms[:, y_start:, :z_top] = 2
    elif mode == 'pair':
        # Two seeds centered across y-axis with given distance and sizes
        dist = max(0, int(cfg.init.center_separation))
        s1 = max(1, int(cfg.init.seed1_edge_length))
        s2 = max(1, int(cfg.init.seed2_edge_length))
        # Clamp to domain
        s1 = min(s1, sizeX, sizeY, sizeZ)
        s2 = min(s2, sizeX, sizeY, sizeZ)

        cx, cy, cz = sizeX//2, sizeY//2, sizeZ//2
        # Place centers symmetrically along y with half-distance offset
        half = dist // 2
        c1y = max(0, min(sizeY-1, cy - half))
        c2y = max(0, min(sizeY-1, cy + (dist - half)))


        x0,x1 = simple_bounds(cx, s1, sizeX)
        y0,y1 = simple_bounds(c1y, s1, sizeY)
        z0,z1 = simple_bounds(cz, s1, sizeZ)
        atoms[x0:x1, y0:y1, z0:z1] = 2

        x0,x1 = simple_bounds(cx, s2, sizeX)
        y0,y1 = simple_bounds(c2y, s2, sizeY)
        z0,z1 = simple_bounds(cz, s2, sizeZ)
        atoms[x0:x1, y0:y1, z0:z1] = 2
    elif mode == 'mounds':
        # Fill below height field h(x,y) = amplitude * (1 + sin(w * x * y))
        A = max(0, int(getattr(cfg.init, 'amplitude', 0)))
        w = float(getattr(cfg.init, 'w', 0.05))
        if A > 0:
            xs = np.arange(sizeX, dtype=float)
            ys = np.arange(sizeY, dtype=float)
            X, Y = np.meshgrid(xs, ys, indexing='ij')
            # h(x,y) = 1 + A*(1 + sin(w*x)*sin(w*y))
            h = 1.0 + A * (1.0 + np.sin(w * X) * np.sin(w * Y))
            # Clamp heights to domain bounds
            h = np.clip(h, 0, sizeZ - 1)
            z_idx = np.arange(sizeZ, dtype=int)
            for x in range(sizeX):
                # For each (x,y), fill z < h(x,y)
                z_heights = h[x].astype(int)
                for y in range(sizeY):
                    ztop = int(z_heights[y])
                    if ztop > 0:
                        atoms[x, y, :ztop] = 2
        # else A==0: nothing to fill (falls back to occupancy only)
    else:
        pass

    return atoms

def CA_3D_experiment(params, initial_atoms=None, ruleset: str = "Default", config_metadata: Optional[Dict[str, Any]] = None):

    # Snapshot scheduling is driven by data/xyz intervals (see below)
    # Snapshot intervals from configuration
    try:
        cfg_params = (config_metadata or {}).get("parameters", {}) if isinstance(config_metadata, dict) else {}
        data_every = int(cfg_params.get("data_snapshot_interval_steps", params[2]))
        xyz_every = int(cfg_params.get("xyz_snapshot_interval_steps", params[2]))
        save_last_xyz_only = bool(cfg_params.get("xyz_save_last_only", False))
        snapshot_mode = str(cfg_params.get("snapshot_schedule_mode", "linear")).lower()
        log_count = int(cfg_params.get("log_snapshot_count", 100))
        enable_age = int(bool(cfg_params.get("enable_age_calculation", False)))
    except Exception:
        data_every = params[2]
        xyz_every = params[2]
        save_last_xyz_only = False
        snapshot_mode = "linear"
        log_count = 100
        enable_age = 0
    
    # Diffusion drift mode and parameters
    try:
        cfg_params = (config_metadata or {}).get("parameters", {}) if isinstance(config_metadata, dict) else {}
        diffusion_drift_mode = str(cfg_params.get("diffusion_drift_mode", "constant")).lower()
        if diffusion_drift_mode not in ("constant", "time_dependent", "spatial_dependent", "both"):
            diffusion_drift_mode = "constant"
        diffusion_bias_x = float(cfg_params.get("diffusion_bias_x", 0.0))
        diffusion_bias_y = float(cfg_params.get("diffusion_bias_y", 0.0))
        diffusion_bias_z = float(cfg_params.get("diffusion_bias_z", 0.0))
        diffusion_bias_x_expr = cfg_params.get("diffusion_bias_x_expr") or "0"
        diffusion_bias_y_expr = cfg_params.get("diffusion_bias_y_expr") or "0"
        diffusion_bias_z_expr = cfg_params.get("diffusion_bias_z_expr") or "0"
    except Exception:
        diffusion_drift_mode = "constant"
        diffusion_bias_x = 0.0
        diffusion_bias_y = 0.0
        diffusion_bias_z = 0.0
        diffusion_bias_x_expr = "0"
        diffusion_bias_y_expr = "0"
        diffusion_bias_z_expr = "0"
 
                      
    # Walls configuration (non-transparent borders)
    wx0 = np.int32(0)
    wx1 = np.int32(0)
    wy0 = np.int32(0)
    wy1 = np.int32(0)
    wz0 = np.int32(0)
    wz1 = np.int32(0)
    try:
        if isinstance(config_metadata, dict):
            params_meta = config_metadata.get("parameters", {})
            walls_cfg = params_meta.get("walls") or {}
            wx0 = np.int32(1 if walls_cfg.get("wall_x_min") else 0)
            wx1 = np.int32(1 if walls_cfg.get("wall_x_max") else 0)
            wy0 = np.int32(1 if walls_cfg.get("wall_y_min") else 0)
            wy1 = np.int32(1 if walls_cfg.get("wall_y_max") else 0)
            wz0 = np.int32(1 if walls_cfg.get("wall_z_min") else 0)
            wz1 = np.int32(1 if walls_cfg.get("wall_z_max") else 0)
    except Exception:
        # Reason: If metadata is malformed, keep defaults (all periodic)
        wx0 = wx1 = wy0 = wy1 = wz0 = wz1 = np.int32(0)

    def calculateCA_evolution(
        atoms_gpu,
        atoms_next_gpu,
        directions_gpu,
        Ru_gpu,
        depositions_gpu,
        incorporations_gpu,
        coordinations_gpu,
        coordinations_count_gpu,
        sizeX,
        sizeY,
        sizeZ,
        iterations,
        subiterations,
        flux,
        path,
        autobreak,
        events_gpu,
        event_types,
        states_count_gpu,
        states_gpu,
        calculate_event_statistics,
        calculate_states_statistics,
        calculate_coordination_statistics,
        drift_mode,
        bias_x_const,
        bias_y_const,
        bias_z_const,
        bias_x_expr,
        bias_y_expr,
        bias_z_expr,
        biasX_gpu,
        biasY_gpu,
        biasZ_gpu,
        diffuse_ker_spatial,
    ):

        
        atoms_evolution=[]
        coordinations_evolution=[]
        x=np.arange(4)
        y=np.arange(4)
        z=np.arange(4)
        coords=np.array(list(itertools.product(x,y,z)))
        snapshot_base = Path(path)
        
        # Diffusion bias values - mode-dependent handling
        # For constant mode, use scalar values
        # For time-dependent, evaluate each timestep
        # For spatial/both, pre-compute coordinate grids if needed
        bias_x_scalar = np.float32(bias_x_const)
        bias_y_scalar = np.float32(bias_y_const)
        bias_z_scalar = np.float32(bias_z_const)
        
        # Prepare coordinate grids for spatial evaluation if needed
        x_grid = None
        y_grid = None
        z_grid = None
        if drift_mode in ("spatial_dependent", "both"):
            # Create coordinate arrays matching diffusion grid size
            # Note: The kernel accesses _DIF_INDEX(x,y,z) where _DIF_SIZE_X = 4*blockDim.x*gridDim.x = sizeX
            # So for a 64×64×64 lattice, the diffusion grid accessed is 64×64×64 (same as lattice size)
            # The kernel iterates 64 times (4×4×4 cycles) but each covers a different subgrid of the same 64×64×64 space
            dif_sizeX = sizeX
            dif_sizeY = sizeY
            dif_sizeZ = sizeZ
            x_coords = np.arange(dif_sizeX, dtype=np.float32).reshape(dif_sizeX, 1, 1)
            y_coords = np.arange(dif_sizeY, dtype=np.float32).reshape(1, dif_sizeY, 1)
            z_coords = np.arange(dif_sizeZ, dtype=np.float32).reshape(1, 1, dif_sizeZ)
            # Broadcast to full grid
            x_grid = np.broadcast_to(x_coords, (dif_sizeX, dif_sizeY, dif_sizeZ))
            y_grid = np.broadcast_to(y_coords, (dif_sizeX, dif_sizeY, dif_sizeZ))
            z_grid = np.broadcast_to(z_coords, (dif_sizeX, dif_sizeY, dif_sizeZ))
        
        # Pre-compute spatial bias arrays if mode is spatial_dependent (not both)
        if drift_mode == "spatial_dependent":
            bias_x_arr = _evaluate_drift_expression(bias_x_expr, 0, sizeX, sizeY, sizeZ, iterations, x_grid, y_grid, z_grid)
            bias_y_arr = _evaluate_drift_expression(bias_y_expr, 0, sizeX, sizeY, sizeZ, iterations, x_grid, y_grid, z_grid)
            bias_z_arr = _evaluate_drift_expression(bias_z_expr, 0, sizeX, sizeY, sizeZ, iterations, x_grid, y_grid, z_grid)
            # Ensure results are numpy arrays (handle scalar returns)
            if not isinstance(bias_x_arr, np.ndarray):
                bias_x_arr = np.full_like(x_grid, float(bias_x_arr), dtype=np.float32)
            if not isinstance(bias_y_arr, np.ndarray):
                bias_y_arr = np.full_like(y_grid, float(bias_y_arr), dtype=np.float32)
            if not isinstance(bias_z_arr, np.ndarray):
                bias_z_arr = np.full_like(z_grid, float(bias_z_arr), dtype=np.float32)
            # Ensure float32 and correct shape
            bias_x_arr = np.asarray(bias_x_arr, dtype=np.float32)
            bias_y_arr = np.asarray(bias_y_arr, dtype=np.float32)
            bias_z_arr = np.asarray(bias_z_arr, dtype=np.float32)
            biasX_gpu.set(bias_x_arr)
            biasY_gpu.set(bias_y_arr)
            biasZ_gpu.set(bias_z_arr)

        # Plateau-based early termination parameters (read from config_metadata if present)
        plateau_window = 5  # number of saved data points to consider
        plateau_tol = 0.03
        try:
            if isinstance(config_metadata, dict):
                params_meta = config_metadata.get("parameters", {})
                plateau_window = int(params_meta.get("early_termination_window", plateau_window))
                plateau_tol = float(params_meta.get("early_termination_tolerance", plateau_tol))
        except Exception:
            pass

        # Rolling buffers for mobile and crystalline counts, evaluated on data cadence only
        from collections import deque
        mobile_hist_snap = deque(maxlen=max(1, int(plateau_window)))
        cryst_hist_snap = deque(maxlen=max(1, int(plateau_window)))
        early_stop_triggered = False

        # Precompute log-distributed snapshot steps if requested
        if str(snapshot_mode).lower() == "log":
            try:
                # start=1, end=iterations per requirement
                log_steps = set(int(x) for x in generate_log_points(1, int(iterations), max(1, int(log_count))).tolist())
            except Exception:
                log_steps = set()
        else:
            log_steps = set()

        def _is_snapshot_step_linear(ts: int) -> bool:
            # Linear schedule: respect separate cadences for data and xyz
            return (ts % max(1, data_every) == 0) or ((not save_last_xyz_only) and (ts % max(1, xyz_every) == 0))

        def _is_snapshot_step_log(ts: int) -> bool:
            # Log schedule: snapshot only on generated points; always include 0 and last
            if ts == 0 or ts == (iterations - 1):
                return True
            return ts in log_steps

        def persist_snapshot(timestep: int) -> None:
            if calculate_coordination_statistics:
                coordinations_count_gpu.fill(0)
                coordinations_ker(
                    atoms_gpu,
                    coordinations_count_gpu,
                    coordinations_gpu,
                    wx0, wx1, wy0, wy1, wz0, wz1,
                    grid=(int(sizeX), int(sizeY), int(sizeZ)),
                    block=(1, 1, 1),
                )

            atoms = atoms_gpu.get_async()
            coordinations = coordinations_gpu.get_async()
            coordinations_count = None
            if calculate_coordination_statistics:
                coordinations_count = coordinations_count_gpu.get_async()
            states = None
            prev_states = None
            site_age_host = None
            cryst_age_host = None
            states_count = None
            if calculate_states_statistics:
                states = states_gpu.get_async()
                states_count = np.int32(states_count_gpu.get_async())
            if enable_age:
                cryst_age_host = cryst_age_gpu.get_async()
                if calculate_states_statistics:
                    prev_states = prev_states_gpu.get_async()
                    site_age_host = site_age_gpu.get_async()
            events_count = None
            if calculate_event_statistics:
                events_count = np.int32(events_gpu.get_async())

            # Save XYZ according to schedule or last-only policy
            do_save_xyz = False
            if save_last_xyz_only:
                do_save_xyz = (timestep == 0) or (timestep == iterations - 1)
            else:
                if str(snapshot_mode).lower() == "log":
                    do_save_xyz = _is_snapshot_step_log(timestep)
                else:
                    do_save_xyz = (timestep % max(1, xyz_every) == 0)
            if do_save_xyz:
                # Write XYZ only; stats saved on data cadence below
                save_xyz_only(
                    atoms,
                    str(snapshot_base / "system_evolution.xyz"),
                    timestep,
                    coordinations,
                    cryst_age=cryst_age_host if enable_age else None,
                )
                # Save states 3D only when XYZ is being saved (same cadence)
                if calculate_states_statistics and states is not None:
                    save_states_3d(
                        states,
                        str(snapshot_base / "free_incropropation_sites_evolution.xyz"),
                        timestep,
                        site_age=site_age_host if enable_age else None,
                    )
            # Save .dat statistics according to schedule (linear: data_every; log: on log steps)
            do_save_data = False
            if str(snapshot_mode).lower() == "log":
                do_save_data = _is_snapshot_step_log(timestep)
            else:
                do_save_data = (timestep % max(1, data_every) == 0)

            if do_save_data and calculate_coordination_statistics and coordinations_count is not None:
                save_coordinations(
                    coordinations_count,
                    str(snapshot_base / "statistics_on_coordination.dat"),
                    timestep,
                )
            if do_save_data and calculate_states_statistics and states_count is not None:
                save_states(
                    states_count,
                    str(snapshot_base / "statistics_on_available_states.dat"),
                    timestep,
                )
            if do_save_data and calculate_event_statistics and events_count is not None:
                save_events(
                    events_count,
                    str(snapshot_base / "statisitics_on"),
                    event_types,
                    timestep,
                )
            # Age statistics when enabled: crystalline by coordination and sites by type
            if do_save_data and enable_age:
                try:
                    # Mean crystalline age by coordination (1..6)
                    if coordinations is not None:
                        means_by_coord: list[float] = []
                        for k in range(1, 7):
                            mask = (atoms == 2) & (coordinations == k)
                            denom = float(mask.sum())
                            if denom > 0:
                                means_by_coord.append(float((cryst_age_host[mask].mean())))
                            else:
                                means_by_coord.append(0.0)
                        save_crystalline_age_means(
                            means_by_coord,
                            str(snapshot_base / "statistics_on_crystalline_ages.dat"),
                            timestep,
                        )
                    # Mean site ages by site type states==1,2,3
                    if states is not None and site_age_host is not None:
                        means_by_type: list[float] = []
                        for tval in (1, 2, 3):
                            mask = (states == tval)
                            denom = float(mask.sum())
                            if denom > 0:
                                means_by_type.append(float((site_age_host[mask].mean())))
                            else:
                                means_by_type.append(0.0)
                        save_site_age_means(
                            means_by_type,
                            str(snapshot_base / "statistics_on_site_ages.dat"),
                            timestep,
                        )
                except Exception:
                    pass
            # Always write system-level stats (.dat) on chosen data cadence
            if do_save_data:
                save_system_stats(
                    atoms,
                    str(snapshot_base / "system_evolution.xyz"),
                    timestep,
                )
                # Plateau-based early termination check on data cadence only
                nonlocal early_stop_triggered
                try:
                    mobile_cnt = int((atoms == 1).sum())
                    cryst_cnt = int((atoms == 2).sum())
                    mobile_hist_snap.append(mobile_cnt)
                    cryst_hist_snap.append(cryst_cnt)
                    if (autobreak == 1) and (len(mobile_hist_snap) >= mobile_hist_snap.maxlen):
                        m0, mN = mobile_hist_snap[0], mobile_hist_snap[-1]
                        c0, cN = cryst_hist_snap[0], cryst_hist_snap[-1]
                        rel_mobile = abs(mN - m0) / max(1.0, float(max(m0, mN)))
                        rel_cryst = abs(cN - c0) / max(1.0, float(max(c0, cN)))
                        if (rel_mobile <= plateau_tol) and (rel_cryst <= plateau_tol):
                            append_log(log_path, f"Autobreak triggered (plateau snapshots={mobile_hist_snap.maxlen}, tol={plateau_tol:.4f}, dM={rel_mobile:.4f}, dC={rel_cryst:.4f})", banner)
                            early_stop_triggered = True
                except Exception:
                    pass

            append_log(log_path, f"Snapshot {timestep} saved", banner)

            # Reason: Free host memory promptly after snapshot serialization.
            del atoms
            del coordinations
            if states is not None:
                del states
            if prev_states is not None:
                del prev_states
            if cryst_age_host is not None:
                del cryst_age_host
            if site_age_host is not None:
                del site_age_host
            if events_count is not None:
                del events_count
            if coordinations_count is not None:
                del coordinations_count
        for i in tqdm(range(iterations)):
            # Initial snapshot BEFORE any evolution so seeds are visible
            if i == 0:
                if calculate_states_statistics:
                    states_count_gpu.fill(0)
                    states_gpu.fill(0)
                    if enable_age:
                        analyze_neighbors(
                            atoms_gpu,
                            states_count_gpu if ((i % max(1, data_every)) == 0) else dummy_results_gpu,
                            states_gpu,
                            wx0, wx1, wy0, wy1, wz0, wz1,
                            prev_states_gpu,
                            site_age_gpu,
                            np.int32(1),
                            np.int32(1 if ((i % max(1, data_every)) == 0) else 0),
                            grid=(int(sizeX), int(sizeY), int(sizeZ)),
                            block=(1, 1, 1),
                        )
                        drv.memcpy_dtod(prev_states_gpu.gpudata, states_gpu.gpudata, states_gpu.nbytes)
                    else:
                        analyze_neighbors(
                            atoms_gpu,
                            states_count_gpu if ((i % max(1, data_every)) == 0) else dummy_results_gpu,
                            states_gpu,
                            wx0, wx1, wy0, wy1, wz0, wz1,
                            prev_states_gpu,
                            site_age_gpu,
                            np.int32(0),
                            np.int32(1 if ((i % max(1, data_every)) == 0) else 0),
                            grid=(int(sizeX), int(sizeY), int(sizeZ)),
                            block=(1, 1, 1),
                        )
                persist_snapshot(i)
                # Skip evolution for i==0 to avoid altering initial state
                continue
            gen.fill_uniform(directions_gpu)
            gen.fill_uniform(incorporations_gpu)
            if calculate_event_statistics:
                conway_ker_event_calc(
                    atoms_gpu,
                    atoms_next_gpu,
                    incorporations_gpu,
                    Ru_gpu,
                    events_gpu,
                    wx0, wx1, wy0, wy1, wz0, wz1,
                    cryst_age_gpu,
                    np.int32(1 if enable_age else 0),
                    grid=(int(sizeX), int(sizeY), int(sizeZ)),
                    block=(1, 1, 1),
                )
            else:
                conway_ker(
                    atoms_gpu,
                    atoms_next_gpu,
                    incorporations_gpu,
                    Ru_gpu,
                    wx0, wx1, wy0, wy1, wz0, wz1,
                    cryst_age_gpu,
                    np.int32(1 if enable_age else 0),
                    grid=(int(sizeX), int(sizeY), int(sizeZ)),
                    block=(1, 1, 1),
                )
            atoms_gpu, atoms_next_gpu = atoms_next_gpu, atoms_gpu

            # Update bias values for time-dependent and both modes
            if drift_mode == "time_dependent":
                bias_x_scalar = np.float32(_evaluate_drift_expression(bias_x_expr, i, sizeX, sizeY, sizeZ, iterations))
                bias_y_scalar = np.float32(_evaluate_drift_expression(bias_y_expr, i, sizeX, sizeY, sizeZ, iterations))
                bias_z_scalar = np.float32(_evaluate_drift_expression(bias_z_expr, i, sizeX, sizeY, sizeZ, iterations))
            elif drift_mode == "both":
                # Re-evaluate expressions with current timestep and upload to GPU
                bias_x_arr = _evaluate_drift_expression(bias_x_expr, i, sizeX, sizeY, sizeZ, iterations, x_grid, y_grid, z_grid)
                bias_y_arr = _evaluate_drift_expression(bias_y_expr, i, sizeX, sizeY, sizeZ, iterations, x_grid, y_grid, z_grid)
                bias_z_arr = _evaluate_drift_expression(bias_z_expr, i, sizeX, sizeY, sizeZ, iterations, x_grid, y_grid, z_grid)
                # Ensure results are numpy arrays (handle scalar returns)
                if not isinstance(bias_x_arr, np.ndarray):
                    bias_x_arr = np.full_like(x_grid, float(bias_x_arr), dtype=np.float32)
                if not isinstance(bias_y_arr, np.ndarray):
                    bias_y_arr = np.full_like(y_grid, float(bias_y_arr), dtype=np.float32)
                if not isinstance(bias_z_arr, np.ndarray):
                    bias_z_arr = np.full_like(z_grid, float(bias_z_arr), dtype=np.float32)
                # Ensure float32 and correct shape
                bias_x_arr = np.asarray(bias_x_arr, dtype=np.float32)
                bias_y_arr = np.asarray(bias_y_arr, dtype=np.float32)
                bias_z_arr = np.asarray(bias_z_arr, dtype=np.float32)
                biasX_gpu.set(bias_x_arr)
                biasY_gpu.set(bias_y_arr)
                biasZ_gpu.set(bias_z_arr)
            
            np.random.shuffle(coords)
            for k in range(64):
                # Use spatial kernel for spatial_dependent and both modes, scalar kernel otherwise
                if drift_mode in ("spatial_dependent", "both"):
                    diffuse_ker_spatial(
                        atoms_gpu,
                        directions_gpu,
                        np.int32(coords[k][0]),
                        np.int32(coords[k][1]),
                        np.int32(coords[k][2]),
                        biasX_gpu,
                        biasY_gpu,
                        biasZ_gpu,
                        wx0, wx1, wy0, wy1, wz0, wz1,
                        grid=(int(sizeX//4),int(sizeY//4),int(sizeZ//4)),
                        block=(1,1,1)
                    )
                else:
                    # Constant or time_dependent: use scalar kernel
                    diffuse_ker(
                        atoms_gpu,
                        directions_gpu,
                        np.int32(coords[k][0]),
                        np.int32(coords[k][1]),
                        np.int32(coords[k][2]),
                        bias_x_scalar,
                        bias_y_scalar,
                        bias_z_scalar,
                        wx0, wx1, wy0, wy1, wz0, wz1,
                        grid=(int(sizeX//4),int(sizeY//4),int(sizeZ//4)),
                        block=(1,1,1)
                    )

            drv.Context.synchronize()
            # Update states every timestep to keep site age accurate; only count on data cadence
            if calculate_states_statistics:
                states_gpu.fill(0)
                # Zero the count array before counting (only when we're actually counting)
                if (i % max(1, data_every)) == 0:
                    states_count_gpu.fill(0)
                if enable_age:
                    analyze_neighbors(
                        atoms_gpu,
                        states_count_gpu if ((i % max(1, data_every)) == 0) else dummy_results_gpu,
                        states_gpu,
                        wx0, wx1, wy0, wy1, wz0, wz1,
                        prev_states_gpu,
                        site_age_gpu,
                        np.int32(1),
                        np.int32(1 if ((i % max(1, data_every)) == 0) else 0),
                        grid=(int(sizeX), int(sizeY), int(sizeZ)),
                        block=(1, 1, 1),
                    )
                    drv.memcpy_dtod(prev_states_gpu.gpudata, states_gpu.gpudata, states_gpu.nbytes)
                else:
                    analyze_neighbors(
                        atoms_gpu,
                        states_count_gpu if ((i % max(1, data_every)) == 0) else dummy_results_gpu,
                        states_gpu,
                        wx0, wx1, wy0, wy1, wz0, wz1,
                        prev_states_gpu,
                        site_age_gpu,
                        np.int32(0),
                        np.int32(1 if ((i % max(1, data_every)) == 0) else 0),
                        grid=(int(sizeX), int(sizeY), int(sizeZ)),
                        block=(1, 1, 1),
                    )
            # Persist .dat and/or XYZ only when scheduled
            if str(snapshot_mode).lower() == "log":
                do_persist = _is_snapshot_step_log(i) or (save_last_xyz_only and (i in (0, iterations - 1)))
            else:
                do_persist = False
                # Data cadence controls .dat outputs and plateau detection
                if (i % max(1, data_every) == 0):
                    do_persist = True
                # XYZ cadence controls visualization frames (unless saving last-only)
                if not do_persist:
                    if save_last_xyz_only:
                        # Only 0 and final frame; 0 already handled above, final handled by condition below
                        if i == (iterations - 1):
                            do_persist = True
                    else:
                        if (i % max(1, xyz_every) == 0):
                            do_persist = True

            if do_persist:
                persist_snapshot(i)
            # Final frame is handled inside persist_snapshot via last-only policy
            atom_count = np.float32(gpuarray.sum(atoms_gpu).get())
            # If plateau detected during snapshot persistence, stop now
            if early_stop_triggered:
                break


        return atoms_evolution,coordinations_evolution

    
     
    sizeX=int(params[0][0])
    sizeY=int(params[0][1])
    sizeZ=int(params[0][2])
    iterations=params[1]
    subiterations=params[2]
    coverage=params[3]
    path=params[4]
    snapshot_base = Path(path)
    snapshot_base.mkdir(parents=True, exist_ok=True)
    device=0 #params[5]
    probabilities=params[6]
    autobreak=params[7]
    calculate_event_statistics = bool(params[8])
    calculate_states_statistics = True
    if len(params) > 9:
        calculate_states_statistics = bool(params[9])
    calculate_coordination_statistics = True
    if len(params) > 10:
        calculate_coordination_statistics = bool(params[10])

    # Initialize CUDA and select device explicitly
    ctx = None
    drv.init()
    device_count = drv.Device.count()
    try:
        device_idx = int(device)
    except Exception:
        device_idx = 0
    if device_count > 0 and (device_idx < 0 or device_idx >= device_count):
        device_idx = max(0, min(device_idx, device_count - 1))
    ctx = drv.Device(device_idx).make_context()

    ker=SourceModule(kernels.ker_code,no_extern_c=True,keep=True) #,options=['-v']
    gen=rnd.XORWOWRandomNumberGenerator(seed_getter=None, offset=0)
    Ru = np.zeros(2187, dtype=np.float32) 
    events = np.zeros(2187, dtype=np.int32) 

    event_list_path = Path(__file__).resolve().parent / "event_lists.json"
    with open(event_list_path, "r", encoding="utf-8") as file:
        event_types = json.load(file)

    banner = (
        "\n".join(
            [
                "***************************************************************************",
                "*                                                                         *",
                "*    _____ ____         ____    _       ____                _        _    *",
                "*   |___ /|  _ \       / ___|  / \     / ___|_ __ _   _ ___| |_ __ _| |   *",
                "*     |_ \| | | |_____| |     / _ \   | |   | '__| | | / __| __/ _` | |   *",
                "*    ___) | |_| |_____| |___ / ___ \  | |___| |  | |_| \__ \ || (_| | |   *",
                "*   |____/|____/       \____/_/   \_\  \____|_|   \__, |___/\__\__,_|_|   *",
                "*                                                 |___/                   *",
                "*                        version 1.0.0, 2025                              *",
                "*                             by A.V. Redkov                              *",
                "*                report the bugs to : avredkov@gmail.com                  *",
                "***************************************************************************",
            ]
        )
    )

    log_path = snapshot_base / "system_info.log"
    used_params_path = snapshot_base / "used_params.json"
    content_width = len(banner.splitlines()[0]) - 4



    gpu_name: Optional[str]
    try:
        gpu_name = drv.Device(device_idx).name()
    except drv.Error:
        gpu_name = "Unavailable"

    system_info: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "os": platform.platform(),
        "python_version": sys.version,
        "cpu": platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "memory_total_bytes": psutil.virtual_memory().total,
        "gpu": gpu_name,
    }

    append_boxed_log(log_path, "System Configuration", system_info, banner)

    params_payload = serialize_config_metadata(config_metadata)
    if not params_payload:
        params_payload = {
            "parameters": {
                "lattice_size_x": sizeX,
                "lattice_size_y": sizeY,
                "lattice_size_z": sizeZ,
                "num_iterations": iterations,
                "snapshot_interval_steps": subiterations,
                "initial_occupancy_fraction": coverage,
                "output_dir": path,
                "cuda_device_index": device,
                "recipe_values": {},
                "enable_early_termination": bool(autobreak),
                "ruleset_name": ruleset,
            },
            "init": {
                "mode": None,
                "details": None,
            },
        }

    try:
        write_json(used_params_path, params_payload)
        append_log(log_path, "Recipe parameters saved to used_params.json", banner)
        append_boxed_log(log_path, "Simulation Parameters", params_payload, banner)
    except OSError:
        append_log(log_path, "Failed to write used_params.json", banner)

    # Load and apply selected ruleset
    assign_rules = load_assign_rules(ruleset)
    param_map = dict(config_metadata.get("recipe_values") or {})
    # Evaluate rules with params into Ru
    assign_rules(Ru, param_map)
    append_log(log_path, f"Ruleset '{ruleset}' loaded with parameters {param_map}", banner)
    # Persist fully-evaluated rules to JSON for reproducibility
    try:
        rules_used_path = snapshot_base / "rules_used.json"
        write_json(rules_used_path, {
            "ruleset": ruleset,
            "parameters": param_map,
            "Ru": [float(x) for x in Ru.tolist()],
        })
        append_log(log_path, "rules_used.json written", banner)
    except OSError:
        append_log(log_path, "Failed to write rules_used.json", banner)

 
    conway_ker = ker.get_function("conway_ker")
    conway_ker_event_calc = ker.get_function("conway_ker_event_calc")
    diffuse_ker=ker.get_function("diffuse_adatoms")
    diffuse_ker_spatial=ker.get_function("diffuse_adatoms_spatial")
    coordinations_ker=ker.get_function("calculate_coordinations")
    analyze_neighbors=ker.get_function("analyze_neighbors")
    
 
    p0=np.float32(sizeX*sizeY*sizeZ*coverage)


    directions_gpu = gpuarray.zeros((sizeX, sizeY, sizeZ), dtype=np.float32)
    incorporations_gpu = gpuarray.zeros((sizeX, sizeY, sizeZ), dtype=np.float32)
    coordinations_gpu = gpuarray.zeros((sizeX, sizeY, sizeZ), dtype=np.int32)
    states_gpu = gpuarray.zeros((sizeX, sizeY, sizeZ), dtype=np.int32)
    # Always allocate age-related buffers to simplify kernel calls (unused when disabled)
    prev_states_gpu = gpuarray.zeros((sizeX, sizeY, sizeZ), dtype=np.int32)
    cryst_age_gpu = gpuarray.zeros((sizeX, sizeY, sizeZ), dtype=np.uint32)
    site_age_gpu = gpuarray.zeros((sizeX, sizeY, sizeZ), dtype=np.uint32)
    dummy_results_gpu = gpuarray.zeros((3,), dtype=np.int32)
    depositions_gpu = gpuarray.zeros((sizeX, sizeY), dtype=np.float32)
    if initial_atoms is not None:
        atoms = np.int32(initial_atoms)
    else:
        atoms = np.int32((np.random.random(sizeX * sizeY * sizeZ) < coverage).reshape(sizeX, sizeY, sizeZ))

    append_log(log_path, "Initial atoms array prepared", banner)

    coordinations_count=np.int32(np.zeros((7)))
    states_count=np.int32(np.zeros((3)))

    flux=sizeX*sizeY*sizeZ*coverage;

    # Default seed if no initial array provided
    if initial_atoms is None and (not hasattr(config_metadata, "parameters")):
        atoms[sizeX//2-1:sizeX//2+2,sizeY//2,sizeZ//2]=2
        atoms[sizeX//2,sizeY//2-1:sizeY//2+2,sizeZ//2]=2
        atoms[sizeX//2,sizeY//2,sizeZ//2-1:sizeZ//2+2]=2

    Ru_gpu=gpuarray.to_gpu(Ru)
    events_gpu=gpuarray.to_gpu(events)
    states_count_gpu=gpuarray.to_gpu(states_count)
    coordinations_count_gpu=gpuarray.to_gpu(coordinations_count)
    atoms_gpu = gpuarray.to_gpu(atoms)
    atoms_next_gpu = gpuarray.empty_like(atoms_gpu)
    del(atoms)
    
    # Allocate bias arrays for spatial/both modes (size matches diffusion grid, which equals lattice size)
    biasX_gpu = None
    biasY_gpu = None
    biasZ_gpu = None
    if diffusion_drift_mode in ("spatial_dependent", "both"):
        # The diffusion kernel uses _DIF_SIZE_X = sizeX, so bias arrays must match lattice size
        biasX_gpu = gpuarray.zeros((sizeX, sizeY, sizeZ), dtype=np.float32)
        biasY_gpu = gpuarray.zeros((sizeX, sizeY, sizeZ), dtype=np.float32)
        biasZ_gpu = gpuarray.zeros((sizeX, sizeY, sizeZ), dtype=np.float32)
        append_log(log_path, f"Allocated spatial bias arrays ({sizeX}x{sizeY}x{sizeZ})", banner)
    
    append_log(log_path, "CA experiment started", banner)
    try:
        atoms_evolution,coordinations_evolution=calculateCA_evolution(
            atoms_gpu,
            atoms_next_gpu,
            directions_gpu,
            Ru_gpu,
            depositions_gpu,
            incorporations_gpu,
            coordinations_gpu,
            coordinations_count_gpu,
            sizeX,
            sizeY,
            sizeZ,
            iterations,
            subiterations,
            flux,
            path,
            autobreak,
            events_gpu,
            event_types,
            states_count_gpu,
            states_gpu,
            calculate_event_statistics,
            calculate_states_statistics,
            calculate_coordination_statistics,
            diffusion_drift_mode,
            diffusion_bias_x,
            diffusion_bias_y,
            diffusion_bias_z,
            diffusion_bias_x_expr,
            diffusion_bias_y_expr,
            diffusion_bias_z_expr,
            biasX_gpu,
            biasY_gpu,
            biasZ_gpu,
            diffuse_ker_spatial,
        )
    except Exception as exc:  # pylint: disable=broad-except
        error_details = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        append_boxed_log(log_path, "Error", error_details, banner)
        append_log(log_path, "CA experiment aborted due to error", banner)
        raise
    else:
        append_log(log_path, "CA experiment completed successfully", banner)
        return atoms_evolution,params,coordinations_evolution
    finally:
        try:
            if ctx is not None:
                ctx.pop()
        except Exception:
            pass
        append_log(log_path, "CA experiment finished", banner)

