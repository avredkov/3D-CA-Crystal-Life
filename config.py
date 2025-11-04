from __future__ import annotations

from typing import List, Union, Optional, Dict
from typing_extensions import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


class SingleInitConfig(BaseModel):
    mode: Literal["single"]
    seed_edge_length: int = 2


class MultipleInitConfig(BaseModel):
    mode: Literal["multiple"]
    seed_edge_length: int = 2
    num_seeds: int = 1


class FlatBottomInitConfig(BaseModel):
    mode: Literal["flat_bottom"]
    num_flat_layers: int = 1


class PythonCodeInitConfig(BaseModel):
    mode: Literal["python_code"]
    init_python_code: str = ""


class NoneInitConfig(BaseModel):
    mode: Literal["none"]


class SteppedBottomInitConfig(BaseModel):
    mode: Literal["stepped_bottom"]
    num_terraces: int = 16


class PairInitConfig(BaseModel):
    mode: Literal["pair"]
    center_separation: int = 12
    seed1_edge_length: int = 2
    seed2_edge_length: int = 2


class MoundsInitConfig(BaseModel):
    mode: Literal["mounds"]
    w: float = 0.174  # spatial frequency (radians per cell)
    amplitude: int = 12  # mound height in cells


InitConfig = Union[
    SingleInitConfig,
    MultipleInitConfig,
    FlatBottomInitConfig,
    PythonCodeInitConfig,
    NoneInitConfig,
    SteppedBottomInitConfig,
    PairInitConfig,
    MoundsInitConfig,
]


class RuleProbabilities(BaseModel):
    def to_list(self) -> List[float]:
        return [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


class SimulationConfig(BaseModel):
    """
    Simulation configuration parameters (renamed and simplified).

    Use load_config to load from JSON and to_params to convert into
    the ordered list required by CA3D_functions.CA_3D_experiment.
    """

    # Domain sizes
    lattice_size_x: int = 256
    lattice_size_y: int = 256
    lattice_size_z: int = 256

    # Time control
    num_iterations: int = 10000
    data_snapshot_interval_steps: int = 100
    xyz_snapshot_interval_steps: int = 100
    xyz_save_last_only: bool = False

    # Snapshot scheduling mode and parameters
    # linear: use data/xyz snapshot intervals as before
    # log: use logarithmically distributed timesteps via generate_log_points
    snapshot_schedule_mode: Literal["linear", "log"] = "linear"
    log_snapshot_count: int = 100

    # Surface/physics
    initial_occupancy_fraction: float = 0.05
    
    # Diffusion drift mode: "constant", "time_dependent", "spatial_dependent", or "both"
    diffusion_drift_mode: Literal["constant", "time_dependent", "spatial_dependent", "both"] = "constant"
    
    # Diffusion bias parameters (range: -1.0 to 1.0) - used for constant mode
    # 0 = equal probability in both directions
    # 1 = only positive direction (P=1/3)
    # -1 = only negative direction (P=1/3)
    diffusion_bias_x: float = 0.0
    diffusion_bias_y: float = 0.0
    diffusion_bias_z: float = 0.0
    
    # Diffusion bias expressions - used for time_dependent, spatial_dependent, or both modes
    # Expression strings that can use variables: t (timestep), x, y, z (coordinates), np (numpy)
    # Examples: "0.5*np.sin(t/100)", "0.5*np.sin(x/100)", "0.5*np.sin(x/100 + t/1000)"
    diffusion_bias_x_expr: Optional[str] = None
    diffusion_bias_y_expr: Optional[str] = None
    diffusion_bias_z_expr: Optional[str] = None

    # Paths
    output_dir: str

    # Analysis/experiment
    enable_early_termination: bool = True
    # Early termination (plateau detection) parameters
    early_termination_window: int = 5  # number of saved data points to consider for plateau
    early_termination_tolerance: float = 0.03  # relative change threshold (e.g., 0.03 = 3%)
    cuda_device_index: int = 0

    # Reproducibility (optional)
    rng_seed: Optional[int] = None

    # Recipe parameter values: key -> float
    recipe_values: Dict[str, float] = Field(default_factory=dict)

    # Ruleset plugin name (module under package `rules`)
    ruleset_name: str = "Default"

    # Diagnostics
    enable_event_statistics: bool = True
    enable_surface_state_statistics: bool = True
    enable_coordination_statistics: bool = True
    # Monitoring
    enable_age_calculation: bool = False

    # Initialization regime (grouped, discriminated union)
    init: InitConfig = SingleInitConfig(mode="single", seed_edge_length=2)

    # Boundary conditions (non-transparent walls)
    class WallConfig(BaseModel):
        wall_x_min: bool = False
        wall_x_max: bool = False
        wall_y_min: bool = False
        wall_y_max: bool = False
        wall_z_min: bool = False
        wall_z_max: bool = False

    walls: WallConfig = WallConfig()

    @model_validator(mode="after")
    def validate_values(self) -> "SimulationConfig":
        # Sizes and time
        if self.lattice_size_x <= 0 or self.lattice_size_y <= 0 or self.lattice_size_z <= 0:
            raise ValueError("All lattice dimensions must be positive integers")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive integer")
        # normalize intervals
        if self.data_snapshot_interval_steps <= 0:
            self.data_snapshot_interval_steps = 1
        if self.xyz_snapshot_interval_steps <= 0:
            self.xyz_snapshot_interval_steps = 1
        # snapshot schedule sanity
        if self.snapshot_schedule_mode not in ("linear", "log"):
            self.snapshot_schedule_mode = "linear"
        # Clamp log params to sensible ranges
        if self.log_snapshot_count < 1:
            self.log_snapshot_count = 1
        # Ensure output directory is a string path
        _ = str(self.output_dir)
        # Per-mode constraints
        if isinstance(self.init, SingleInitConfig):
            if self.init.seed_edge_length < 2:
                raise ValueError("seed_edge_length must be >= 2")
        elif isinstance(self.init, MultipleInitConfig):
            if self.init.seed_edge_length < 2:
                raise ValueError("seed_edge_length must be >= 2")
            if self.init.num_seeds < 1:
                raise ValueError("num_seeds must be >= 1")
        elif isinstance(self.init, FlatBottomInitConfig):
            if self.init.num_flat_layers < 1:
                raise ValueError("num_flat_layers must be >= 1")
        elif isinstance(self.init, PythonCodeInitConfig):
            pass
        elif isinstance(self.init, PairInitConfig):
            if self.init.seed1_edge_length < 2 or self.init.seed2_edge_length < 2:
                raise ValueError("seed1_edge_length and seed2_edge_length must be >= 2 for pair mode")
            if self.init.center_separation < 0:
                raise ValueError("center_separation must be >= 0")
        elif isinstance(self.init, SteppedBottomInitConfig):
            if self.init.num_terraces < 1:
                raise ValueError("num_terraces must be >= 1 for stepped_bottom mode")
        elif isinstance(self.init, MoundsInitConfig):
            if self.init.amplitude < 0:
                raise ValueError("amplitude must be >= 0 for mounds mode")
            # frequency can be any real number; no further validation here
        # RNG seed
        if self.rng_seed is not None and self.rng_seed < 0:
            raise ValueError("rng_seed must be a non-negative integer or null")
        # Early termination params
        if self.early_termination_window < 1:
            self.early_termination_window = 1
        if not (0.0 <= float(self.early_termination_tolerance) <= 1.0):
            # Clamp into [0,1] to keep semantics of relative tolerance
            self.early_termination_tolerance = min(1.0, max(0.0, float(self.early_termination_tolerance)))
        # Clamp diffusion bias parameters to [-1.0, 1.0] for constant mode
        if not (-1.0 <= float(self.diffusion_bias_x) <= 1.0):
            self.diffusion_bias_x = max(-1.0, min(1.0, float(self.diffusion_bias_x)))
        if not (-1.0 <= float(self.diffusion_bias_y) <= 1.0):
            self.diffusion_bias_y = max(-1.0, min(1.0, float(self.diffusion_bias_y)))
        if not (-1.0 <= float(self.diffusion_bias_z) <= 1.0):
            self.diffusion_bias_z = max(-1.0, min(1.0, float(self.diffusion_bias_z)))
        
        # Validate drift mode
        if self.diffusion_drift_mode not in ("constant", "time_dependent", "spatial_dependent", "both"):
            self.diffusion_drift_mode = "constant"
        
        # Validate expressions for non-constant modes
        if self.diffusion_drift_mode in ("time_dependent", "spatial_dependent", "both"):
            for axis, expr in [("x", self.diffusion_bias_x_expr), ("y", self.diffusion_bias_y_expr), ("z", self.diffusion_bias_z_expr)]:
                if expr is None or (isinstance(expr, str) and len(expr.strip()) == 0):
                    # Set default zero expression if missing
                    setattr(self, f"diffusion_bias_{axis}_expr", "0")
        
        return self

    def to_params(self) -> list:
        """
        Convert config model into ordered params list expected by
        CA3D_functions.CA_3D_experiment.

        Returns:
            list: Ordered parameters list (length 11) matching CA3D_functions usage.
        """
        return [
            [int(self.lattice_size_x), int(self.lattice_size_y), int(self.lattice_size_z)],
            int(self.num_iterations),
            int(self.data_snapshot_interval_steps),
            float(self.initial_occupancy_fraction),
            str(self.output_dir),
            int(self.cuda_device_index),
            list(),  # reserved position to match CA3D_functions signature
            int(1 if self.enable_early_termination else 0),
            bool(self.enable_event_statistics),
            bool(self.enable_surface_state_statistics),
            bool(self.enable_coordination_statistics),
        ]


def load_config(config_path: str = "config.json") -> SimulationConfig:
    """
    Load simulation configuration from a JSON file.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        SimulationConfig: Validated configuration object.
    """
    import json

    def strip_json_comments(text: str) -> str:
        """Remove // line comments and /* */ block comments from JSON text."""
        import re
        # Remove /* block comments */
        text = re.sub(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", "", text, flags=re.DOTALL)
        # Remove // line comments
        text = re.sub(r"(^|\s)//.*$", "", text, flags=re.MULTILINE)
        return text

        
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
        cleaned = strip_json_comments(raw)
        data = json.loads(cleaned)
    
    # Handle config files that have a "parameters" wrapper
    if "parameters" in data and isinstance(data["parameters"], dict):
        data = data["parameters"]
    
    # Ensure required fields have default values if missing
    if "output_dir" not in data or not data["output_dir"]:
        data["output_dir"] = "./output/"
    
    return SimulationConfig(**data)


__all__ = ["SimulationConfig", "RuleProbabilities", "load_config"]


