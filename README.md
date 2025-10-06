## 3D Cellular Automaton (CA) + Monte Carlo Crystal Growth Simulator <i> "Crystal Life" </i>

A GPU-accelerated 3D cellular automaton for crystal growth simulation described in the paper by <b>A.V. Redkov, V. Ivanov, A. Pimpinelli, V. Tonchev. <i>"Under the Kink's rule: The Way of the Crystal"</i></b>.
The simulation evolves a lattice of atoms with CUDA kernels while exporting time-series snapshots and quantitative description of crystal growth and dissoultion processes for analysis. All behavior is configured via `config.json` and modular “rulesets” in `rules/`.

### Quick start

1) Update `config.json` (output `output_dir`, lattice sizes, iterations, ruleset, etc.)
2) Ensure required Python packages and CUDA are installed (see below)
3) Run in terminal:

```bash
python CA3D.py
```

Outputs will be written under the configured `output_dir` directory.

### Installation & requirements

- Python 3.8+
- NVIDIA GPU with CUDA toolkit installed
- Python packages:
  - numpy, pydantic, typing_extensions
  - pycuda, psutil, tqdm
  - scipy, matplotlib
  - PySide6, pyqtgraph (for the optional GUI)

Example install (conda or venv recommended):

```bash
python -m venv .venv && .venv\\Scripts\\activate
pip install -r requirements.txt
```

If PyCUDA fails to build, ensure the NVIDIA CUDA Toolkit is installed and matches your driver. On Windows, verify that `curand_kernel.h` is available via the CUDA include path. We rely on the standard include `<curand_kernel.h>`; if NVCC cannot find it, add CUDA’s `include` directory to your environment (e.g., set `CUDA_PATH` or adjust your compiler include paths).

---

## Graphical User Interface (GUI)

An optional desktop GUI is included under `gui/`. It provides a modern interface for configuring simulations, previewing initial states in 3D, running jobs with live progress, analyzing outputs, editing rules, and opening XYZ results in OVITO.

### Launching the GUI

- From project root (recommended):

```bash
python -m gui.main
```

You can also launch from inside the `gui/` folder (advanced users): `python main.py`.

Requirements: `PySide6`, `pyqtgraph`, and standard scientific stack (see requirements.txt). On first run, the GUI tries to load `../config.json` automatically.

### Tabs overview

- Simulation
  - Configure domain size, iterations, snapshot interval, initialization mode and its parameters, walls, ruleset, probabilities, and diagnostics.
  - Hover over labels for tooltips showing what the parameter does and the exact config key it maps to.
  - Preview initial state in 3D (OpenGL) with crystalline spheres and mobile points; simulation cell edges are drawn with color coding by wall transparency.
  - Start/Stop the run; live progress is derived from tqdm output, logs, and stats with ETA.
  - Monitoring and output checkboxes (2×2 group): Record event statistics; Record surface state statistics; Record coordination statistics; Record cell and site ages.

- Analysis
  - Select an output directory and load time series. Choose series to plot; supports linear/log axes, autoscale, and saving plots to `plots/`.
  - **Cluster Analysis**: Analyze crystalline clusters in XYZ files with real-time progress display. Generates time series data and detailed cluster statistics.
  - "Open in OVITO" locates `system_evolution.xyz` and opens it. If OVITO is not found, you will be prompted to select the `ovito` executable; the path is remembered in `~/.3d_ca_gui/ovito_path.txt`.

- Rules
  - Edit a 2187-rule recipe using a table, load/save (with built-in protection), toggle nucleation, and visualize neighborhoods.

- About
  - Credits and project links (email and GitHub are clickable).

### Using the GUI effectively

- Keep `config.json` under the project root; use the GUI “Load”/“Save”/“Save As” buttons to manage it.
- Initialization mode section shows context-specific parameters for each `init.mode`.
- Use tooltips to discover the exact config keys and meanings before saving.
- For frequent snapshots, prefer higher `snapshot_interval_steps`.
- Use the Analysis tab to quickly verify output trends and export figures. Note that all the data saved in output folder can be also studied and visualized via other tools, i.e. Pandas, Origin, Excel, OVITO, VMD, etc.

---

## How it works (high level)

- `CA3D.py` loads and validates `config.json` via `config.load_config`, prepares the initial lattice with `CA3D_functions.prepare_initial_atoms`, and launches `CA3D_functions.CA_3D_experiment`.
- CUDA kernels in `kernels_3D.py` perform evolution (incorporation/sublimation) and diffusion and compute coordination/state diagnostics.
- At configurable time steps, the simulator persists 3D snapshots and statistics to files in the output `output_dir`.
- Rules are provided by a ruleset module in `rules/` that fills a probability lookup table `Ru` of size 2187 (3^7 local neighborhood encoding for center + 6 face neighbors).

---

## Examples (loadable from `examples/`)

You can quickly try several pre-configured scenarios by loading the JSON recipes from the `examples/` folder.

- In the GUI: click "Config file → Load" and pick any file under `examples/`.
- Headless: run `python CA3D.py --config examples/<Recipe Name>.json`.

Brief overview and measured runtimes (on an RTX 3090, 128×128×128 lattice):

- Dendrite growth (`examples/Dendrite growth.json`)
  - Description: single-seed dendritic growth in a periodic box.
  - Output folder: `output/dendrite`
  - Runtime (log): Total time - a couple of minutes.
  - Output Size: ~450 MB

- Well-shaped crystal (`examples/Well-shaped crystal.json`)
  - Description: long run forming a faceted crystal (slow growth probabilities).
  - Output folder: `output/wellshaped`
  - Runtime (log): Total time ~20-25 minutes
  - Output Size: ~3 GB

- Unstable crystal growth (`examples/Unstable crystal.json`)
  - Description: a growth of a unstable shape between the well-shaped and dendritic.
  - Output folder: `output/wellshaped`
  - Runtime (log): Total time 00:22:08 for 99800 steps; mean 75.15 it/s
  - Size: medium to large (can reach hundreds of MB with dense snapshots)

- Crystal dissolution (`examples/Crystal_dissolution.json`)
  - Description: large single crystal dissolving (growth off, sublimation on).
  - Output folder: `output/crystal_dissolution`
  - Runtime (log): Total time ~ 2-5 minutes
  - Output Size: ~800 MB

- Nucleation (`examples/Nucleation.json`)
  - Description: 3D nucleation enabled in bulk; many small clusters.
  - Output folder: `output/nucleation`
  - Runtime (log): Total time ~ 5-10 minutes~
  - Output Size: ~1.45 GB

- Particle coagulation (`examples/Particle_coagulation.json`)
  - Description: two seeds coalesce; useful for neck formation/coarsening.
  - Output folder: `output/particle_coagulation`
  - Runtime (log): Total time ~10-15 minutes
  - Output Size: ~2.8 GB

- Particle ripening (`examples/Particle_ripening.json`)
  - Description: two-particle ripening with diffusion-driven mass transfer.
  - Output folder: `output/particle_ripening`
  - Runtime (log): Total time ~10 minutes
  - Output Size: ~1.1 GB

- Full lifecycle (nucleation → growth → ripening) — long (`examples/Full_lifecycle (nucleation_growth_ripening)_long.json`)
  - Description: extended run across regimes using log-distributed snapshots.
  - Output folder: `output/full_lifecycle`
  - Runtime: varies (long run, make take a few hours)
  - Output Size: ~2.8 GB

- Dendrite vicinal growth (`examples/Dendrite vicinal growth.json`)
  - Description: dendritic growth over a stepped (vicinal) surface.
  - Output folder: `output/dendrite vicinal growth`
  - Runtime (log): Total time ~ 5-10 minutes
  - Output Size: ~2.8 GB

- Step-flow vicinal growth (`examples/Step-flow vicinal growth.json`)
  - Description: slow step-flow regime over many iterations.
  - Output folder: `output/step-flow growth`
  - Runtime (log): Total time ~ 15-20 minutes
  - Output Size: ~ 3.5 GB

- Surface growth via 2D nucleation (`examples/Surface growth via 2D nucleation.json`)
  - Description: flat surface with rare 2D nucleation and layer-by-layer growth.
  - Output folder: `output/surface growth via 2D nucleation`
  - Runtime (log): Total time ~ 10-15 minutes
  - Output Size: ~3.2 GB

- Surface smoothing (`examples/Surface_smoothing.json`)
  - Description: smoothing of mounded/rough surface under growth conditions.
  - Output folder: `output/surface_smoothing`
  - Runtime (log): Total time ~10-15 minutes
  - Output Size: ~2.7 GB

Notes on space/time:
- Disk usage depends strongly on `xyz_snapshot_interval_steps` and total steps; expect from a few MB up to tens of GB for longer runs with dense snapshots.
- The reported runtimes are from the NVIDIA RTX 3090; your hardware and settings may differ.


## Running simulations in different regimes

The initial crystalline lattice configuration is controlled by `init.mode` in `config.json` (see full reference below). Set the `output_dir` to an existing or new directory to collect outputs.

Initialization modes overview:

- `single`
  - Places one centered cubic crystalline seed (state=2) of edge length `seed_edge_length`.
  - The cube is clamped to fit entirely within the domain; the center voxel is guaranteed to be crystalline even for size 1.
  - All other sites are initialized randomly as mobile (state=1) with probability `initial_occupancy_fraction` (else empty 0).

- `multiple`
  - Places `num_seeds` randomly positioned cubic crystalline seeds, each with edge length `seed_edge_length`.
  - Seeds are independently placed and may overlap; placement is clamped to keep each cube within bounds.
  - Background mobile atoms are initialized per `initial_occupancy_fraction`.

- `flat_bottom`
  - Fills the bottom `num_flat_layers` planes along Z with crystalline atoms (state=2); if `num_flat_layers` is 0, no planes are filled.
  - Remaining sites follow random mobile initialization per `initial_occupancy_fraction`.

- `stepped_bottom`
  - Builds a staircase of `num_terraces` steps along the Y direction with step height of 1 in Z.
  - Step start positions are evenly distributed across Y to avoid a large final terrace; lower terraces extend farther in Z.
  - Useful for vicinal surface growth scenarios; background mobile atoms follow `initial_occupancy_fraction`.

- `pair`
  - Places two centered crystalline seeds (state=2) separated along Y by `center_separation`.
  - Seed sizes are `seed1_edge_length` and `seed2_edge_length`; centers are symmetrically offset around the domain center.
  - All sizes and positions are clamped to the domain; seeds may overlap for small separations; background mobile atoms per `initial_occupancy_fraction`.

- `mounds`
  - Fills the lower part of the lattice with “mounds” defined by a height field \( h(x,y) = 1 + A \cdot (1 + \sin(w x)\,\sin(w y)) \).
  - For each \((x,y)\), all sites with \( z < h(x,y) \) are set to crystalline (2). Parameters:
    - `w` (float): spatial frequency (radians per cell) in the sinusoid.
    - `amplitude` (int): mound height in cells.
  - Background mobile atoms are initialized per `initial_occupancy_fraction`.

- `python_code`
  - Executes arbitrary Python on the variable `array` (NumPy int32 of shape `[X,Y,Z]`) after the random mobile initialization step.
  - You can set lattice values directly: 0 (empty), 1 (mobile), 2 (crystalline). Example: `array[:,:,0:3] = 2` to prefill three bottom layers.
  - Use with care; invalid operations may raise exceptions at startup.

- `none`
  - No explicit seeding; the lattice starts with random mobile atoms only, per `initial_occupancy_fraction`. May be used with recipes which allow 3D nucleation.

Examples:

```json
{
  "output_dir": "D:/simulations/run1/",
  "init": { "mode": "single", "seed_edge_length": 4 }
}
```

```json
{
  "output_dir": "D:/simulations/run2/",
  "init": { "mode": "multiple", "seed_edge_length": 3, "num_seeds": 16 }
}
```

```json
{
  "output_dir": "D:/simulations/run3/",
  "init": { "mode": "flat_bottom", "num_flat_layers": 2 }
}
```

```json
{
  "output_dir": "D:/simulations/run4/",
  "init": { "mode": "stepped_bottom", "num_terraces": 8 }
}
```

```json
{
  "output_dir": "D:/simulations/run5/",
  "init": {
    "mode": "pair", "center_separation": 50, "seed1_edge_length": 16, "seed2_edge_length": 28
  }
}
```

```json
{
  "output_dir": "D:/simulations/run6/",
  "init": { "mode": "python_code", "init_python_code": "array[:,:,0:3]=2" }
}
```

```json
{
  "output_dir": "D:/simulations/run7/",
  "init": { "mode": "single", "seed_edge_length": 8 },
  "enable_age_calculation": true,
  "recipe_values": {
    "Pk": 1.0, "Ps": 0.5, "Pn": 0.1,
    "Pke": 0.0, "Pse": 0.0, "Pne": 0.0
  }
}
```

Set `num_iterations` and `snapshot_interval_steps` to control runtime and snapshot cadence. Snapshots are saved every `snapshot_interval_steps`, including the initial state at step 0.
For a starting point, copy `config.example.json` to `config.json` and tweak as needed.

---

---

## Configuration reference (`config.json`)

Schema is defined in `config.py` with Pydantic. All keys have safe defaults unless marked required.

- Domain size
  - `lattice_size_x`, `lattice_size_y`, `lattice_size_z` (int): Lattice dimensions. Must be powers of two between 4 and 512 inclusive: 4, 8, 16, 32, 64, 128, 256, 512. Values may differ across axes. The GUI provides discrete sliders for these values.
- Time control
  - `num_iterations` (int): Total evolution steps (> 0).
  - `snapshot_schedule_mode` ("linear"|"log"): Snapshot scheduling mode. `linear` preserves the existing interval-based behavior; `log` saves snapshots at logarithmically distributed timesteps.
  - If `snapshot_schedule_mode="linear"`:
    - `data_snapshot_interval_steps` (int): Interval for saving .dat statistics (events, states, coordination).
    - `xyz_snapshot_interval_steps` (int): Interval for saving frames to `system_evolution.xyz`.
    - `xyz_save_last_only` (bool): If true, save only the first (0) and last frame to XYZ; .dat statistics are unaffected and follow `data_snapshot_interval_steps`.
  - If `snapshot_schedule_mode="log"`:
    - `log_snapshot_count` (int): Number of log-distributed timesteps(default 100). The range is fixed to start at 1 and end at `num_iterations`. Useful for studying long processes like Ostwald ripening.
- Population
  - `initial_occupancy_fraction` (float): Initial probability of a site being occupied (by mobile atom, state=1) before seeding crystalline atoms (state=2).
- I/O
  - `output_dir` (string, required): Output directory. Will be created if missing.
- Early stop
  - `enable_early_termination` (bool): If true, stop early when population dynamics plateau.
  - `early_termination_window` (int): Number of last saved data points (on `data_snapshot_interval_steps` cadence) to consider for plateau detection (default 5).
  - `early_termination_tolerance` (float): Relative-change threshold in [0,1] across the window for both mobile and crystalline counts (default 0.03 = 3%). If both remain within tolerance across the last `early_termination_window` saved points, the simulation stops early.
- Rules & probabilities
  - `ruleset_name` (string): Name of rules module under `rules/` (default `"Default"`).
  - `recipe_values` (object): Parameter values for the selected ruleset:
    - `Pk` (float): Growth probability at kink sites
    - `Ps` (float): Growth probability at step sites  
    - `Pn` (float): Growth probability at nucleation sites
    - `Pke` (float): Sublimation probability at kink sites
    - `Pse` (float): Sublimation probability at step sites
    - `Pne` (float): Sublimation probability at nucleation sites
- Diagnostics (enable/disable extra analysis and files)
  - `enable_event_statistics` (bool): Per-neighborhood event counts.
  - `enable_surface_state_statistics` (bool): Surface state counts and 3D markers.
  - `enable_coordination_statistics` (bool): Neighbor coordination histogram for crystalline atoms.
  - `enable_age_calculation` (bool): Record crystalline cell ages and surface-site ages at every timestep. When enabled:
    - `system_evolution.xyz` includes an extra last column with crystalline age (timesteps).
    - `free_incropropation_sites_evolution.xyz` includes an extra last column with site age (timesteps).
    - Additional .dat files are written on the data snapshot cadence (see Outputs).
- Boundaries
  - `walls.{wall_x_min,wall_x_max,wall_y_min,wall_y_max,wall_z_min,wall_z_max}` (bool): If true, that face is a non-periodic wall (no wrap). If false, periodic in that direction.
  - `single`: `{ "seed_edge_length": int }` — centered cubic seed.
  - `multiple`: `{ "seed_edge_length": int, "num_seeds": int }` — `num_seeds` random cubic seeds.
  - `flat_bottom`: `{ "num_flat_layers": int }` — fill bottom planes.
  - `stepped_bottom`: `{ "num_terraces": int }` — staircase terraces across Y (1 layer per step).
  - `pair`: `{ "center_separation": int, "seed1_edge_length": int, "seed2_edge_length": int }` — two centered seeds separated along Y.
  - `mounds`: `{ "w": float, "amplitude": int }` — fill below sinusoidal mound height.
  - `python_code`: `{ "init_python_code": string }` — execute Python against variable `array`.
  - `none`: `{}` — random occupancy only (per `initial_occupancy_fraction`).

Validation rules enforce positive sizes, valid probability vector length (6), and sane per-mode parameters.

---

## Outputs (written under `output_dir`)

- `system_info.log`
  - Hardware banner and timestamped log entries; includes boxed sections for system configuration and parameters.
- `used_params.json`
  - Normalized snapshot of the effective parameters, probabilities, ruleset, and init block used for the run.
- `system_evolution.xyz`
  - Appended per snapshot. Each frame:
    - Line 1: number of atoms in frame
    - Line 2: comment
    - Lines 3..: `<state> <x> <y> <z> <coordination> [<crystal_age>]` for each occupied site; state 1=mobile, 2=crystalline. The optional `<crystal_age>` is present only if `enable_age_calculation` is true.
- `system_evolution_stats.dat`
  - Time series with columns: `Timestep, Mobile atoms, Crystalline atoms, Alpha, Crystal size (if single crystal)`.
- `free_incropropation_sites_evolution.xyz` (if `enable_surface_state_statistics`)
  - 3D dummy atoms marking surface states at each snapshot; values: 1=terrace, 2=step, 3=kink.
  - If `enable_age_calculation` is true: `<state> <x> <y> <z> <site_age>` (extra last column).
- `statistics_on_available_states.dat` (if `enable_surface_state_statistics`)
  - Per snapshot counts of terrace/step/kink sites.
- `statistics_on_coordination.dat` (if `enable_coordination_statistics`)
  - Per snapshot histogram of crystalline-atom coordinations (0..6) and corresponding fractions.
- `statistics_on_macro_events.dat`, `statistics_on_all_events.dat` (if `enable_event_statistics`)
  - Macro file aggregates event counts by category; detailed file lists counts for each of the 2187 neighborhood rules. Event categories are defined in `event_lists.json` (e.g., `a2a`, `a2s`, `a2k`, `sfa`, `sf2`, `sfk`, `subl_zero_neighbors`, `other`).
- `statistics_on_crystalline_ages.dat` (if `enable_age_calculation`)
  - Mean crystalline age by coordination class (1..6). Columns: `Timestep`, `coord_1_mean_age`, `coord_2_mean_age`, …, `coord_6_mean_age`.
- `statistics_on_site_ages.dat` (if `enable_age_calculation`)
  - Mean site age by surface-state type. Columns: `Timestep`, `Site_on_terrace_mean_age`, `Site_at_step_mean_age`, `Site_at_kink_mean_age`.

Notes:
- XYZ and .dat streams have independent intervals. The GUI progress bar and logging now follow .dat saves to provide smoother feedback even when XYZ frames are sparse.
- Ensure `output_dir` is unique per run to avoid mixing data between runs.

### Cluster Analysis

The simulator includes built-in cluster analysis functionality accessible through the GUI Analysis tab. This feature analyzes crystalline clusters in the simulation output and provides detailed statistics about cluster evolution over time.

#### Running Cluster Analysis

1. **Complete a simulation** and ensure `system_evolution.xyz` is generated
2. **Open the GUI** and navigate to the Analysis tab
3. **Browse to your output directory** using the "Browse" button
4. **Click "Clustering"** button to start the analysis
5. **Monitor progress** in the popup dialog showing real-time log output
6. **View results** in the generated files (see Output Structure below)

#### Cluster Analysis Features

- **Real-time progress**: Live log display in popup window with progress bar
- **Automatic parameter detection**: Reads lattice size and snapshot intervals from `used_params.json`
- **3D cluster detection**: Uses NetworkX for connected component analysis with periodic boundary conditions
- **Time series output**: Generates `.dat` files with timestep-based cluster statistics
- **Detailed JSON results**: Individual snapshot analysis and comprehensive summary files

#### Cluster Analysis Output Files

When cluster analysis is performed, the following files are generated:

- `cluster_analysis.dat` (in main output directory)
  - Time series data with columns: `Timestep`, `Number_of_Clusters`, `Mean_Cluster_Size`
  - Timesteps calculated as `snapshot_index × xyz_snapshot_interval_steps`
  - Ready for plotting and further analysis

- `cluster_analysis/` (subdirectory)
  - `comprehensive_cluster_analysis.json`: Complete analysis results with parameters and all snapshot data
  - `snapshot_N_results.json`: Individual snapshot analysis files for detailed inspection

#### Cluster Analysis Parameters

- **Target atom type**: 2 (crystalline atoms)
- **Minimum cluster size**: 1 (configurable)
- **Lattice size**: Auto-detected from simulation parameters
- **Timestep calculation**: Based on `xyz_snapshot_interval_steps` from simulation config
- **Periodic boundaries**: Properly handled in 3D cluster detection

#### Example Cluster Analysis Output

```
# Timestep	Number_of_Clusters	Mean_Cluster_Size
0	5	12.4
10	4	15.2
20	3	18.7
30	2	22.1
40	1	25.8
```

This data can be imported into plotting software (Origin, Excel, Python matplotlib) for visualization of cluster evolution over time.

### Running and performance tips

- `snapshot_interval_steps` impacts I/O volume. Larger intervals reduce disk writes and speed up runs; too small intervals create very large XYZ files and slowdowns.
- Lattice size `lattice_size_*` has a strong memory impact. E.g., 256×256×256 requires substantially more VRAM and disk than 128³.
- Diffusion cost scales with volume and the number of diffusion sub-steps in the kernel; increasing the domain increases runtime nonlinearly.
- Typical performance: on a single NVIDIA RTX 3090, a 128×128×128 system runs around ~250 timesteps/second with `snapshot_interval_steps ≈ 500`.
- Multi-GPU: for multi-GPU support please contact the authors.

Windows vs Linux
- Windows: ensure CUDA Toolkit is installed and available in PATH/INCLUDE/LIB (or `CUDA_PATH` is set). File permissions can restrict creating `output_dir` at root paths; pick a user-writable directory.
- Linux: install CUDA from your distro/NVIDIA repos; ensure your user has write permissions to `output_dir`.

### File size expectations and cleanup

- XYZ files can be very large for big domains (256³ and higher). Multi-GB output is common depending on `snapshot_interval_steps` and run length.
- Use a dedicated per-run `output_dir` and consider archiving or deleting intermediate frames when not needed.
- Consider increasing `snapshot_interval_steps` to reduce size, or postprocess to decimate frames.

Example run directory structure:

```
output_dir/
  system_info.log
  used_params.json
  system_evolution.xyz
  system_evolution_stats.dat
  cluster_analysis.dat                     # if cluster analysis performed
  statistics_on_coordination.dat           # if enabled
  statistics_on_available_states.dat       # if enabled
  statistics_on_macro_events.dat           # if enabled
  statistics_on_all_events.dat             # if enabled
  statistics_on_crystalline_ages.dat       # if enable_age_calculation
  statistics_on_site_ages.dat              # if enable_age_calculation
  free_incropropation_sites_evolution.xyz  # if enabled
  cluster_analysis/                        # if cluster analysis performed
    comprehensive_cluster_analysis.json
    snapshot_0_results.json
    snapshot_1_results.json
    ...
```

### Event categories (event_lists.json)

`event_lists.json` maps the 2187 microscopic neighborhood rules into human-readable macroscopic categories (e.g., `a2a` - attachment to terrace (to any neighboring atom), `a2s` - attachment to step, `a2k` - attachment to kink, `sfa` - sublimation from terrace (from a single neighboring atom), `sf2` - sublimation from step , `sfk` - sublimation from kink, etc.). This grouping controls how macro vs. detailed event outputs are aggregated:

- Macro: `statisitics_on_macro_events.dat` sums counts across all rules belonging to the same category.
- Detailed: `statisitics_on_all_events.dat` lists counts for each of the 2187 rules separately.

This mapping determines, for example, that many different local 3-neighbor cases are all counted as “incorporation into a kink.” If you need a different grouping scheme, edit `event_lists.json` to redefine which rule indices belong to which macro categories.

---

## Rulesets and custom recipes

The simulator uses a 2187-length probability lookup table `Ru` (base-3 neighborhood encoding: center + 6 face neighbors). A ruleset is a Python module under `rules/` that defines how to fill `Ru` from a set of high-level “recipe parameters”.

### The recipe format (symbolic + per-index assignments)

Rules are authored symbolically so that GUI and headless workflows stay in sync. A typical recipe module looks like:

```python
import numpy as np

# 1) Parameter schema (probabilities in [0,1]) — actual keys/labels from built-in recipes
RECIPE_PARAMETERS = [
    {
        "key": 'Pk', "label": 'Incorporation at kink',   "description": 'Probability of incorporation at kink sites and other sites with 3 neighbours',"default": 1,
    },
    {
        "key": 'Ps', "label": 'Incorporation at step', "description": 'Probability of incorporation at step sites and other sites with 2 neighbours',"default": 1,
    },
    {
        "key": 'Pn', "label": 'Incorporation at nucleation', "description": 'Probability of nucleation on terrace (or attachment to single crystalline atoms)', "default": 1,
    },
    {
        "key": 'Pke', "label": 'Sublimation at kink', "description": 'Probability of detachment at kink sites and other sites with 3 neighbours',
        "default": 0,
    },
    {
        "key": 'Pse', "label": 'Sublimation at step', "description": 'Probability of detachment at step sites and other sites with 2 neighbours',
        "default": 0,
    },
    {
        "key": 'Pne', "label": 'Sublimation at nucleation', "description": 'Probability of detachment from single-neighbour configurations',
        "default": 0,
    },
]

# 2) Symbolic expressions and numeric literals
# RULE_EXPR[i] is "0", "1", "" (numeric), or a parameter key.
# RULE_VAL[i] is used only when RULE_EXPR[i] == "".
RULE_EXPR = [
    # ... length 2187 list of strings ... for example:
    # "Pse", "0", "1", "Pn", ""
]
RULE_VAL = [
    # ... length 2187 list of floats in [0,1] used when RULE_EXPR is empty
]

# 3) assign_rules translates symbols into numbers at runtime
def assign_rules(Ru: np.ndarray, params: dict) -> None:
    _defaults = {p["key"]: p.get("default", 0.0) for p in RECIPE_PARAMETERS}
    Ru[:] = 0.0
    for i, sym in enumerate(RULE_EXPR):
        if sym == '0':
            Ru[i] = 0.0
        elif sym == '1':
            Ru[i] = 1.0
        elif sym:
            v = float(params.get(sym, _defaults.get(sym, 0.0)))
            Ru[i] = max(0.0, min(1.0, v))
        else:
            v = float(RULE_VAL[i])
            Ru[i] = max(0.0, min(1.0, v))
```

Key points:
- The GUI (Rules tab) shows `RULE_EXPR` entries as symbols (parameter keys), 0, or 1. When the entry is empty, it shows the numeric literal from `RULE_VAL`.
- Saving a recipe in the GUI writes `RECIPE_PARAMETERS`, `RULE_EXPR`, `RULE_VAL`, and per-index `assign_rules`—so headless users can tweak rules easily.
- Parameter edits (add/rename/delete) are propagated into `RULE_EXPR` on save. Deleting a parameter converts all its usages to 0.

### Selecting a ruleset and setting parameters

- In the GUI (Simulation → Ruleset and probabilities): choose a ruleset from the dropdown; the parameter table updates automatically. Edit “Value” (0..1) for each parameter and Save the config.
- In headless mode: set `"ruleset_name"` and `"recipe_values"` in `config.json`, e.g.:

```json
{
  "ruleset_name": "Default",
  "recipe_values": {
    "Pk": 1.0,
    "Ps": 1.0,
    "Pn": 1.0,
    "Pke": 0.0,
    "Pse": 0.0,
    "Pne": 0.0
  }
}
```

At simulation start, the engine resolves all symbols with the parameter map, evaluates `Ru`, and writes `rules_used.json` (contains `ruleset`, `parameters`, and the full numeric `Ru`).

### Built-in default recipes

- `Default` (no 3D nucleation)
  - Incorporation and sublimation probabilities depend only on the count of crystalline neighbors (face-sharing) around the site.
  - For a mobile atom (center=1) with `n` crystalline neighbors among the six faces:
    - `n = 0` → incorporation probability = 0
    - `n = 1` → incorporation probability = `Pn` (nucleation)
    - `n = 2` → incorporation probability = `Ps` (step)
    - `n = 3` → incorporation probability = `Pk` (kink)
    - `n > 3` → incorporation probability = 1
  - For a crystalline atom (center=2), sublimation mirrors the same mapping using `Pke`, `Pse`, `Pne` probabilities. But if n=0, the sublimation probability is 1, if n>3 it equals to 0.
  - Consequence: no crystalline atom can appear without at least one neighboring crystalline atom; bulk 3D nucleation is not allowed in this ruleset.

- `Default_with_3D_nucleation`
  - Identical to `Default` except for a single additional rule enabling bulk 3D nucleation:
  - If the central atom is mobile (1) and all six face neighbors are mobile (1), the central atom becomes crystalline with probability 1.0.
  - Use by setting `"ruleset_name": "Default_with_3D_nucleation"` in `config.json`.

Both recipes use the neighborhood encoding described earlier (center + 6 faces, base-3 index). See `rules/Default.py` and `rules/Default_with_3D_nucleation.py` for complete symbolic sources.

### Running with custom config paths (headless and GUI)

- Headless (recommended):

```bash
# Use an explicit config path anywhere on your system
python CA3D.py --config "D:/path/to/configs/run1.json"

# Alternatively, set an env var (picked up automatically)
# Windows (PowerShell)
$env:CA3D_CONFIG = "D:/path/to/configs/run1.json"; python CA3D.py
# Linux/macOS (bash)
CA3D_CONFIG="/home/user/configs/run1.json" python CA3D.py
```

- GUI:
  - In the Simulation tab, use the "Config file" controls to Load any `.json` (it can live outside the project).
  - When you press Start, the GUI runs `CA3D.py` with that exact path, so your external config is used as-is.
  - Relative output directories like `./output/Dendrite` are resolved from the app working directory and created if missing.

## Visualizing XYZ outputs

The simulator writes `system_evolution.xyz` and optionally surface-state and event diagnostics. You can visualize XYZ files with:

- [OVITO](https://www.ovito.org/)
  - Import `system_evolution.xyz`. OVITO auto-detects frames in a multi-frame XYZ.
  - Map particle property `state` (first column) to color: 1 = mobile, 2 = crystalline.
  - Optionally use the coordination column to color-code local environments.
  - For clearer visuals showing only crystalline atoms: add the "Select type" modifier, choose value 1 (mobile), then add the "Delete selected" modifier. This removes mobile atoms and leaves only crystalline atoms in view.

- [VMD](https://www.ks.uiuc.edu/Research/vmd/)
  - Load the XYZ file; use coloring by atom type or a custom script to map the `state` column.

- Other tools
  - Any viewer supporting multi-frame XYZ can be used. The file format per frame is:
    - line 1: number of atoms
    - line 2: comment
    - lines 3..: `state x y z coordination`

Tip: For large systems, decimate frames or increase `snapshot_interval_steps` to keep files manageable.

---

## Boundary conditions

Set `walls.wall_x_min`/`wall_x_max`/`wall_y_min`/`wall_y_max`/`wall_z_min`/`wall_z_max` to `true` to make that boundary a non-periodic wall. When a wall is present:
- Diffusing atoms cannot hop across that boundary.
- CUDA neighbor reads across that boundary return 0.
Set to `false` to keep periodic wrapping in that direction.

---


## Troubleshooting

- PyCUDA import/build issues: verify CUDA toolkit installation and that your Python environment can find CUDA headers and libraries.
- Empty or missing outputs: confirm `output_dir` exists or is creatable; ensure `num_iterations` and `snapshot_interval_steps` are positive; check `system_info.log` for errors.
\- Early termination: disable by setting `enable_early_termination` to `false`. When enabled, the engine stops if the ratio of mobile/crystalline atoms does not change beyond `early_termination_tolerance` across the last `early_termination_window` saved points. This works for both crystallization and dissolution regimes.

---

## Citation / Contact

Please refer to the paper <b>A.V. Redkov, V. Ivanov, A. Pimpinelli, V. Tonchev. <i>"Under the Kink's rule: The Way of the Crystal"</i></b> (<i>in press</i>).

For issues, please contact the authors.

