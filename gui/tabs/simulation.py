from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import QTimer
from collections import deque
import time
from pathlib import Path
from typing import Optional

from gui.services.config_service import ConfigService
from gui.services.sim_runner import SimRunner, clear_output_directory_contents
from config import SimulationConfig
from gui.components.GLPreview import GLPreview
from gui.services.rules_service import list_rulesets, get_recipe_parameters


class SimulationTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # Ensure tooltips are shown reliably across platforms
        try:
            self.setAttribute(QtCore.Qt.WA_AlwaysShowToolTips, True)
        except Exception:
            pass

        # Split left controls and right preview
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()

        # Config file controls
        cfg_group = QtWidgets.QGroupBox("Config file")
        cfg_layout = QtWidgets.QHBoxLayout(cfg_group)
        self.cfg_path = QtWidgets.QLineEdit()
        btn_load = QtWidgets.QPushButton("Load")
        btn_save = QtWidgets.QPushButton("Save")
        btn_save_as = QtWidgets.QPushButton("Save As")
        btn_reset = QtWidgets.QPushButton("Reset")
        for w in (self.cfg_path, btn_load, btn_save, btn_save_as, btn_reset):
            cfg_layout.addWidget(w)
        left.addWidget(cfg_group)

        # Domain & time (widgets only; added to tabs later)
        dom_group = QtWidgets.QGroupBox("Domain & time")
        dom = QtWidgets.QGridLayout(dom_group)
        # Allowed lattice sizes: powers of two in [4..512]
        self._pow2_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
        # Use sliders mapped to these discrete values
        self.size_x = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.size_x.setRange(0, len(self._pow2_sizes) - 1); self.size_x.setSingleStep(1); self.size_x.setPageStep(1); self.size_x.setTickInterval(1); self.size_x.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.size_y = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.size_y.setRange(0, len(self._pow2_sizes) - 1); self.size_y.setSingleStep(1); self.size_y.setPageStep(1); self.size_y.setTickInterval(1); self.size_y.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.size_z = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.size_z.setRange(0, len(self._pow2_sizes) - 1); self.size_z.setSingleStep(1); self.size_z.setPageStep(1); self.size_z.setTickInterval(1); self.size_z.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.num_iter = QtWidgets.QSpinBox(); self.num_iter.setRange(1, 10_000_000)
        dom.addWidget(QtWidgets.QLabel("lattice_size_x"), 0, 0); dom.addWidget(self.size_x, 0, 1)
        dom.addWidget(QtWidgets.QLabel("lattice_size_y"), 0, 2); dom.addWidget(self.size_y, 0, 3)
        dom.addWidget(QtWidgets.QLabel("lattice_size_z"), 0, 4); dom.addWidget(self.size_z, 0, 5)
        dom.addWidget(QtWidgets.QLabel("num_iterations"), 1, 0); dom.addWidget(self.num_iter, 1, 1)
        # Snapshot control handled via data/xyz intervals
        # Tabs container
        self.tabs = QtWidgets.QTabWidget()
        left.addWidget(self.tabs)
        # Tab 1: Simulation box and time
        tab_size = QtWidgets.QWidget(); tab_size_l = QtWidgets.QVBoxLayout(tab_size)
        size_box = QtWidgets.QGroupBox(""); size_box.setToolTip("Domain sizes and total iterations (config: lattice_size_*, num_iterations)")
        size_l = QtWidgets.QGridLayout(size_box)
        try:
            size_l.setVerticalSpacing(4)
            size_l.setHorizontalSpacing(8)
            size_l.setContentsMargins(6, 6, 6, 6)
            tab_size_l.setSpacing(6)
        except Exception:
            pass
        lbl_lx = QtWidgets.QLabel("Lattice size X"); lbl_lx.setToolTip("Number of cells along X (config: lattice_size_x)")
        lbl_ly = QtWidgets.QLabel("Lattice size Y"); lbl_ly.setToolTip("Number of cells along Y (config: lattice_size_y)")
        lbl_lz = QtWidgets.QLabel("Lattice size Z"); lbl_lz.setToolTip("Number of cells along Z (config: lattice_size_z)")
        lbl_iter = QtWidgets.QLabel("Number of iterations"); lbl_iter.setToolTip("Total simulation steps (config: num_iterations)")
        self.size_x.setToolTip("config: lattice_size_x (4..512, powers of two)")
        self.size_y.setToolTip("config: lattice_size_y (4..512, powers of two)")
        self.size_z.setToolTip("config: lattice_size_z (4..512, powers of two)")
        self.num_iter.setToolTip("config: num_iterations")
        # Layout as a single column: title row then slider row per axis with numeric tick labels
        size_l.addWidget(lbl_lx, 0, 0)
        size_l.addWidget(self.size_x, 1, 0)
        size_l.addWidget(self._build_tick_labels_row(self._pow2_sizes), 2, 0)
        size_l.addWidget(lbl_ly, 3, 0)
        size_l.addWidget(self.size_y, 4, 0)
        size_l.addWidget(self._build_tick_labels_row(self._pow2_sizes), 5, 0)
        size_l.addWidget(lbl_lz, 6, 0)
        size_l.addWidget(self.size_z, 7, 0)
        size_l.addWidget(self._build_tick_labels_row(self._pow2_sizes), 8, 0)
        # leave two empty rows for spacing before iterations
        size_l.addItem(QtWidgets.QSpacerItem(0, 8, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum), 9, 0)
        size_l.addItem(QtWidgets.QSpacerItem(0, 8, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum), 10, 0)
        size_l.addWidget(lbl_iter, 11, 0)
        size_l.addWidget(self.num_iter, 12, 0)
        tab_size_l.addWidget(size_box)
        # keep content stuck to the top of the section
        tab_size_l.addStretch(1)
        self.tabs.addTab(tab_size, "Simulation box and time")

        # Initialization
        init_group = QtWidgets.QGroupBox("Initialization"); init_group.setToolTip("Initialization mode parameters and random seed (config: init.*, rng_seed)")
        init = QtWidgets.QGridLayout(init_group)
        self.init_mode = QtWidgets.QComboBox()
        self.init_mode.addItems([
            "single", "multiple", "flat_bottom", "stepped_bottom", "pair", "mounds", "python_code", "none",
        ])
        self.rng_seed = QtWidgets.QSpinBox(); self.rng_seed.setRange(0, 2_000_000_000); self.rng_seed.setSpecialValueText("(none)")
        self.rng_seed.setValue(0)
        lbl_mode = QtWidgets.QLabel("Initialization mode"); lbl_mode.setToolTip("Select how to generate the initial lattice (config: init.mode)")
        self.init_mode.setToolTip("config: init.mode")
        lbl_seed = QtWidgets.QLabel("Random seed"); lbl_seed.setToolTip("Seed for RNG to reproduce initial lattice (config: rng_seed). 0 = none")
        self.rng_seed.setToolTip("config: rng_seed")
        init.addWidget(lbl_mode, 0, 0); init.addWidget(self.init_mode, 0, 1)
        init.addWidget(lbl_seed, 0, 2); init.addWidget(self.rng_seed, 0, 3)
        # Init mode specific subsettings (stacked pages)
        self.init_mode_stack = QtWidgets.QStackedWidget()
        self._mode_to_index: dict[str, int] = {}
        self._build_init_mode_pages()
        # Pre-create shared widgets used across tabs
        # Physics/probabilities widgets
        self.initial_occ = QtWidgets.QDoubleSpinBox(); self.initial_occ.setRange(0.0, 1.0); self.initial_occ.setSingleStep(0.0001); self.initial_occ.setDecimals(4)
        
        # Walls and diagnostics widgets
        self.wall_x_min = QtWidgets.QCheckBox("wall_x_min")
        self.wall_x_max = QtWidgets.QCheckBox("wall_x_max")
        self.wall_y_min = QtWidgets.QCheckBox("wall_y_min")
        self.wall_y_max = QtWidgets.QCheckBox("wall_y_max")
        self.wall_z_min = QtWidgets.QCheckBox("wall_z_min")
        self.wall_z_max = QtWidgets.QCheckBox("wall_z_max")
        self.en_events = QtWidgets.QCheckBox("enable_event_statistics")
        self.en_states = QtWidgets.QCheckBox("enable_surface_state_statistics")
        self.en_coord = QtWidgets.QCheckBox("enable_coordination_statistics")
        # Ruleset
        self.ruleset_name = QtWidgets.QComboBox(); self.ruleset_name.setEditable(False)
        # Output / compute widgets
        self.output_dir = QtWidgets.QLineEdit()
        btn_browse_out = QtWidgets.QPushButton("Browse…")
        self.cuda_index = QtWidgets.QSpinBox(); self.cuda_index.setRange(0, 16)
        # Early termination controls
        self.en_early_stop = QtWidgets.QCheckBox("Early stop (plateau detection)")
        self.en_early_stop.setToolTip("Stop early when growth/dissolution plateaus (config: enable_early_termination)")
        # Advanced early-stop parameters (shown only when enabled)
        self.early_win = QtWidgets.QSpinBox(); self.early_win.setRange(1, 10_000_000)
        self.early_tol = QtWidgets.QDoubleSpinBox(); self.early_tol.setRange(0.0, 1.0); self.early_tol.setSingleStep(0.001); self.early_tol.setDecimals(4)
        self.early_win_lbl = QtWidgets.QLabel("Plateau window (saved data points)")
        self.early_tol_lbl = QtWidgets.QLabel("Plateau tolerance (relative change, 0..1)")
        # Friendlier hints explaining impact
        self.early_win_lbl.setToolTip("How many recent saved data points are used to detect a plateau. Larger values require a longer flat trend before stopping.")
        self.early_tol_lbl.setToolTip("How flat the trend must be to stop early. Smaller values are stricter and will stop sooner when the curve is flat.")
        self.early_win.setToolTip("Number of last saved data points (on data snapshot cadence) to check for plateau (config: early_termination_window)")
        self.early_tol.setToolTip("Relative-change threshold across the window for both mobile and crystalline populations (config: early_termination_tolerance)")

        # Tab 2: Initial state
        tab_init = QtWidgets.QWidget(); tab_init_l = QtWidgets.QVBoxLayout(tab_init)
        init_box = QtWidgets.QGroupBox("Initialization")
        init_box.setLayout(init)
        tab_init_l.addWidget(init_box)
        params_box = QtWidgets.QGroupBox("Mode parameters"); params_box.setToolTip("Context-specific parameters for the selected initialization mode (config: init.*)")
        vb_params = QtWidgets.QVBoxLayout(params_box)
        vb_params.addWidget(self.init_mode_stack)
        tab_init_l.addWidget(params_box)
        # Initial occupancy
        occ_box = QtWidgets.QGroupBox("Initial occupancy"); occ_box.setToolTip("Initial mobile occupancy (config: initial_occupancy_fraction)")
        occ_l = QtWidgets.QGridLayout(occ_box)
        lbl_occ = QtWidgets.QLabel("Initial occupancy fraction of mobile atoms"); lbl_occ.setToolTip("Fraction of mobile atoms at start (config: initial_occupancy_fraction)")
        self.initial_occ.setToolTip("config: initial_occupancy_fraction")
        occ_l.addWidget(lbl_occ, 0, 0); occ_l.addWidget(self.initial_occ, 0, 1)
        tab_init_l.addWidget(occ_box)
        # Transparency of walls
        walls_box = QtWidgets.QGroupBox("Transparency of walls")
        wl = QtWidgets.QGridLayout(walls_box)
        hint = "Unchecked = transparent (periodic); checked = non-transparent wall (config: walls.*)"
        lbl_walls = QtWidgets.QLabel("Walls (unchecked = transparent)"); lbl_walls.setToolTip(hint)
        wl.addWidget(lbl_walls, 0, 0, 1, 2)
        self.wall_x_min.setText("X- wall"); self.wall_x_min.setToolTip("config: walls.wall_x_min. " + hint)
        self.wall_x_max.setText("X+ wall"); self.wall_x_max.setToolTip("config: walls.wall_x_max. " + hint)
        self.wall_y_min.setText("Y- wall"); self.wall_y_min.setToolTip("config: walls.wall_y_min. " + hint)
        self.wall_y_max.setText("Y+ wall"); self.wall_y_max.setToolTip("config: walls.wall_y_max. " + hint)
        self.wall_z_min.setText("Z- wall"); self.wall_z_min.setToolTip("config: walls.wall_z_min. " + hint)
        self.wall_z_max.setText("Z+ wall"); self.wall_z_max.setToolTip("config: walls.wall_z_max. " + hint)
        wl.addWidget(self.wall_x_min, 1, 0); wl.addWidget(self.wall_x_max, 1, 1)
        wl.addWidget(self.wall_y_min, 2, 0); wl.addWidget(self.wall_y_max, 2, 1)
        wl.addWidget(self.wall_z_min, 3, 0); wl.addWidget(self.wall_z_max, 3, 1)
        tab_init_l.addWidget(walls_box)
        self.tabs.addTab(tab_init, "Initial state")

        
        phys = QtWidgets.QVBoxLayout()
        self.prob_table = QtWidgets.QTableWidget(0, 3)
        self.prob_table.setHorizontalHeaderLabels(["Label", "Description", "Value"])
        try:
            # Hide row header (keys will be stored internally per row)
            self.prob_table.verticalHeader().setVisible(False)
            # Enable wrapping and auto row height
            self.prob_table.setWordWrap(True)
            hh = self.prob_table.horizontalHeader()
            # Label = contents, Description = stretch, Value = contents
            hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
            hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
            hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        except Exception:
            pass
        phys.addWidget(self.prob_table)
        # Tab 3: Ruleset and probabilities
        tab_rules = QtWidgets.QWidget(); tab_rules_l = QtWidgets.QVBoxLayout(tab_rules)
        rules_box = QtWidgets.QGroupBox("Select Recipe"); rules_box.setToolTip("Select recipe (config: ruleset_name)")
        rb = QtWidgets.QHBoxLayout(rules_box)
        lbl_ruleset = QtWidgets.QLabel("Recipe name"); lbl_ruleset.setToolTip("Python module in rules/ to import (config: ruleset_name)")
        self.ruleset_name.setToolTip("config: ruleset_name")
        rb.addWidget(lbl_ruleset); rb.addWidget(self.ruleset_name)
        # Update probabilities and persist selection when ruleset changes
        def _on_ruleset_changed(_: str) -> None:
            try:
                self._populate_probabilities_table()
                # also reflect in current config in memory so Save uses it
                if self._cfg_svc.current:
                    c = self._cfg_svc.current
                    c.ruleset_name = self.ruleset_name.currentText() or "Default"
            except Exception:
                pass
        try:
            self.ruleset_name.currentTextChanged.connect(_on_ruleset_changed)
        except Exception:
            pass
        tab_rules_l.addWidget(rules_box)
        probs_box = QtWidgets.QGroupBox("Probabilities"); probs_box.setToolTip("Recipe parameters defined by selected ruleset")
        probs_container = QtWidgets.QWidget(); probs_container.setLayout(phys)
        layout_for_box = QtWidgets.QVBoxLayout(); layout_for_box.addWidget(probs_container)
        probs_box.setLayout(layout_for_box)
        tab_rules_l.addWidget(probs_box)
        self.tabs.addTab(tab_rules, "Rule recipe and probabilities")
        try:
            self._rules_tab_index = self.tabs.indexOf(tab_rules)
        except Exception:
            self._rules_tab_index = -1
        # Tab 4: Monitoring and output
        tab_out = QtWidgets.QWidget(); tab_out_l = QtWidgets.QVBoxLayout(tab_out)
        out_box = QtWidgets.QGroupBox("Monitoring and output"); out_box.setToolTip("Output directory, snapshot intervals, diagnostics, CUDA device (config: output_dir, data_snapshot_interval_steps, xyz_snapshot_interval_steps, xyz_save_last_only, cuda_device_index)")
        out_l = QtWidgets.QGridLayout(out_box)
        try:
            out_l.setHorizontalSpacing(8)
            out_l.setVerticalSpacing(4)
            out_l.setContentsMargins(6, 6, 6, 6)
            # Make input columns flexible
            out_l.setColumnStretch(0, 0)  # labels
            out_l.setColumnStretch(1, 1)  # inputs
            out_l.setColumnStretch(2, 1)  # extra inputs / checkbox
            out_l.setColumnStretch(3, 1)
            out_l.setColumnStretch(4, 0)  # browse button
        except Exception:
            pass
        lbl_outdir = QtWidgets.QLabel("Output directory"); lbl_outdir.setToolTip("Where outputs (.dat, .xyz, logs) are saved (config: output_dir)")
        self.output_dir.setToolTip("config: output_dir")
        # Snapshot controls with mode toggle
        self.snap_mode = QtWidgets.QComboBox(); self.snap_mode.addItems(["linear", "log"]) ; self.snap_mode.setToolTip("Choose snapshot schedule: linear (intervals) or log (logarithmic points). config: snapshot_schedule_mode")
        self.data_snap = QtWidgets.QSpinBox(); self.data_snap.setRange(1, 10_000_000)
        self.xyz_snap = QtWidgets.QSpinBox(); self.xyz_snap.setRange(1, 10_000_000)
        self.xyz_last_only = QtWidgets.QCheckBox("Save last XYZ only")
        lbl_mode = QtWidgets.QLabel("Snapshot schedule mode"); lbl_mode.setToolTip("linear: use intervals; log: use logarithmically distributed timesteps. config: snapshot_schedule_mode")
        lbl_data = QtWidgets.QLabel("Data snapshot interval (steps)"); lbl_data.setToolTip("Write .dat statistics every N steps (config: data_snapshot_interval_steps)")
        lbl_xyz = QtWidgets.QLabel("XYZ snapshot interval (steps)"); lbl_xyz.setToolTip("Write system_evolution.xyz every N steps (config: xyz_snapshot_interval_steps)")
        self.data_snap.setToolTip("config: data_snapshot_interval_steps")
        self.xyz_snap.setToolTip("config: xyz_snapshot_interval_steps")
        self.xyz_last_only.setToolTip("If checked, save only the first and the last XYZ frames (config: xyz_save_last_only)")

        # Log params
        self.log_count = QtWidgets.QSpinBox(); self.log_count.setRange(1, 1_000_000); self.log_count.setToolTip("Number of log-distributed points (start=1, end=num_iterations). config: log_snapshot_count")
        lbl_log_count = QtWidgets.QLabel("Number of snapshots")

        out_l.addWidget(lbl_outdir, 0, 0); out_l.addWidget(self.output_dir, 0, 1, 1, 3); out_l.addWidget(btn_browse_out, 0, 4)
        out_l.addWidget(lbl_mode, 1, 0); out_l.addWidget(self.snap_mode, 1, 1)
        # log row directly below mode
        out_l.addWidget(lbl_log_count, 2, 0); out_l.addWidget(self.log_count, 2, 1)
        # linear rows come after
        out_l.addWidget(lbl_data, 3, 0); out_l.addWidget(self.data_snap, 3, 1)
        out_l.addWidget(lbl_xyz, 4, 0); out_l.addWidget(self.xyz_snap, 4, 1)
        # place checkbox below XYZ controls with left indent
        out_l.addWidget(self.xyz_last_only, 5, 1, 1, 2)
        # Statistics and monitoring checkboxes grouped as 2x2 grid
        self.en_events.setText("Record event statistics"); self.en_events.setToolTip("Enable writing macro/detailed events (config: enable_event_statistics)")
        self.en_states.setText("Record surface state statistics"); self.en_states.setToolTip("Enable writing terrace/step/kink stats (config: enable_surface_state_statistics)")
        self.en_coord.setText("Record coordination statistics"); self.en_coord.setToolTip("Enable writing coordination distributions (config: enable_coordination_statistics)")
        self.en_age = QtWidgets.QCheckBox("Record cell and site ages")
        self.en_age.setToolTip("Track crystalline cell age and surface-site age each timestep; add ages to XYZ outputs (config: enable_age_calculation)")
        grid_stats = QtWidgets.QGridLayout()
        try:
            grid_stats.setHorizontalSpacing(16)
            grid_stats.setVerticalSpacing(4)
            grid_stats.setContentsMargins(0, 0, 0, 0)
        except Exception:
            pass
        grid_stats.addWidget(self.en_events, 0, 0)
        grid_stats.addWidget(self.en_states, 0, 1)
        grid_stats.addWidget(self.en_coord, 1, 0)
        grid_stats.addWidget(self.en_age, 1, 1)
        stats_widget = QtWidgets.QWidget(); stats_widget.setLayout(grid_stats)
        out_l.addWidget(stats_widget, 6, 0, 1, 4)
        # CUDA and early stop on next row
        lbl_cuda = QtWidgets.QLabel("CUDA device index"); lbl_cuda.setToolTip("GPU device index to use (config: cuda_device_index)")
        self.cuda_index.setToolTip("config: cuda_device_index")
        # Place Early stop just below the Record... checkboxes row
        out_l.addWidget(self.en_early_stop, 7, 0, 1, 2)
        # Early-stop parameters row (hidden unless checkbox is checked)
        out_l.addWidget(self.early_win_lbl, 8, 0); out_l.addWidget(self.early_win, 8, 1)
        out_l.addWidget(self.early_tol_lbl, 9, 0); out_l.addWidget(self.early_tol, 9, 1)
        # Move CUDA row below early-stop controls
        out_l.addWidget(lbl_cuda, 10, 0); out_l.addWidget(self.cuda_index, 10, 1)

        # Toggle UI sections based on snapshot mode
        def _toggle_snapshot_mode(mode: str) -> None:
            is_log = (mode == "log")
            for w in (lbl_data, self.data_snap, lbl_xyz, self.xyz_snap, self.xyz_last_only):
                w.setVisible(not is_log)
            for w in (lbl_log_count, self.log_count):
                w.setVisible(is_log)
        _toggle_snapshot_mode("linear")
        try:
            self.snap_mode.currentTextChanged.connect(_toggle_snapshot_mode)
        except Exception:
            pass
        # Visibility toggle
        def _toggle_early_params(checked: bool) -> None:
            for w in (self.early_win_lbl, self.early_win, self.early_tol_lbl, self.early_tol):
                w.setVisible(bool(checked))
        _toggle_early_params(False)
        try:
            self.en_early_stop.toggled.connect(_toggle_early_params)
        except Exception:
            pass
        tab_out_l.addWidget(out_box)
        self.tabs.addTab(tab_out, "Monitoring and output")

        # Run controls (below tabs)
        run_group = QtWidgets.QGroupBox("Run")
        run = QtWidgets.QHBoxLayout(run_group)
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100)
        self.run_status = QtWidgets.QLabel("Idle")
        for w in (self.btn_start, self.btn_stop, self.progress, self.run_status):
            run.addWidget(w)
        left.addWidget(run_group)

        # Preview panel
        # Right side preview (square)
        self.preview = GLPreview(self)
        # Increase preview size by ~15%
        self.preview.setMinimumSize(int(480*1.15), int(480*1.15))
        self.preview.setMaximumWidth(int(640*1.15))
        # Preview and Reset View buttons above GL view
        btn_preview = QtWidgets.QPushButton("Preview initial state")
        btn_reset_view = QtWidgets.QPushButton("Reset view")
        self.chk_include_mobile = QtWidgets.QCheckBox("Show mobile atoms (state 1)")
        # Place GL preview, then the checkbox and points status below it
        row_btns = QtWidgets.QHBoxLayout()
        row_btns.addWidget(btn_preview)
        row_btns.addWidget(btn_reset_view)
        row_btns.addStretch(1)
        right.addLayout(row_btns)
        right.addWidget(self.preview)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.chk_include_mobile)
        # forward status label from GLPreview
        hl.addWidget(self.preview.status)
        hl.addStretch(1)
        right.addLayout(hl)

        # Assemble
        left_container = QtWidgets.QWidget(); left_container.setLayout(left)
        right_container = QtWidgets.QWidget(); right_container.setLayout(right)
        root.addWidget(left_container)
        root.addWidget(right_container)
        # Favor left controls width to avoid scrolling; keep preview narrower
        try:
            root.setStretch(0, 2)
            root.setStretch(1, 2)
        except Exception:
            pass

        # Services and wiring
        self._cfg_svc = ConfigService()
        self._runner = SimRunner(cwd=Path(__file__).resolve().parents[2])
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll_progress)
        self._hist = deque(maxlen=30)
        self._t0: float | None = None
        self._last_atoms = None

        btn_load.clicked.connect(self._on_load)
        btn_save.clicked.connect(self._on_save)
        btn_save_as.clicked.connect(self._on_save_as)
        btn_reset.clicked.connect(self._on_reset)
        btn_preview.clicked.connect(self._on_preview)
        btn_reset_view.clicked.connect(lambda: self.preview.view.setCameraPosition(distance=self.preview.view.opts['distance']))
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_browse_out.clicked.connect(self._on_browse_output)
        self.chk_include_mobile.toggled.connect(self._on_toggle_include_mobile)
        self.init_mode.currentTextChanged.connect(self._on_init_mode_changed)
        # populate ruleset dropdown on startup
        self._refresh_rulesets()
        # refresh when switching into Ruleset tab
        try:
            self.tabs.currentChanged.connect(self._on_tab_changed)
        except Exception:
            pass

        # Live validation: connect basic fields
        for sp in (self.size_x, self.size_y, self.size_z, self.num_iter, self.data_snap, self.xyz_snap, self.log_count):
            sp.valueChanged.connect(self._on_field_changed)
        try:
            self.snap_mode.currentTextChanged.connect(lambda _: self._on_field_changed())
        except Exception:
            pass

        # Try to preload ../config.json by default
        try:
            default_cfg = Path(__file__).resolve().parents[2] / "config.json"
            if default_cfg.exists():
                cfg = self._cfg_svc.load(str(default_cfg))
                self.cfg_path.setText(str(default_cfg))
                self._apply_cfg_to_ui(cfg)
        except Exception:
            pass

    # --- Binding helpers ---
    def _resolve_output_dir(self, path_text: str) -> Path:
        """Resolve output directory relative to runner cwd when not absolute."""
        p = Path(path_text or "")
        try:
            if not p.is_absolute():
                return (self._runner.cwd / p).resolve()
            return p.resolve()
        except Exception:
            return p
    def _apply_cfg_to_ui(self, cfg: SimulationConfig) -> None:
        # Map actual sizes to slider indices (powers of two 4..512)
        try:
            def to_index(val: int) -> int:
                try:
                    return max(0, self._pow2_sizes.index(int(val)))
                except ValueError:
                    # Fallback: choose nearest allowed
                    diffs = [abs(int(val) - v) for v in self._pow2_sizes]
                    return int(diffs.index(min(diffs)))
            self.size_x.setValue(to_index(int(cfg.lattice_size_x)))
            self.size_y.setValue(to_index(int(cfg.lattice_size_y)))
            self.size_z.setValue(to_index(int(cfg.lattice_size_z)))
        except Exception:
            pass
        self.num_iter.setValue(int(cfg.num_iterations))
        # Snapshot intervals shown below
        try:
            self.data_snap.setValue(int(cfg.data_snapshot_interval_steps))
            self.xyz_snap.setValue(int(cfg.xyz_snapshot_interval_steps))
            self.xyz_last_only.setChecked(bool(cfg.xyz_save_last_only))
            # mode & log params
            self.snap_mode.setCurrentText(str(getattr(cfg, "snapshot_schedule_mode", "linear")))
            self.log_count.setValue(int(getattr(cfg, "log_snapshot_count", 100)))
        except Exception:
            pass
        # monitoring
        try:
            self.en_age.setChecked(bool(getattr(cfg, "enable_age_calculation", False)))
        except Exception:
            pass
        # init & seed
        try:
            self._write_init_to_ui(cfg)
        except Exception:
            pass
        self.rng_seed.setValue(int(cfg.rng_seed or 0))
        # physics
        self.initial_occ.setValue(float(cfg.initial_occupancy_fraction))
        # Populate dynamic probabilities table (new schema)
        try:
            self._populate_probabilities_table()
        except Exception:
            pass
        # rules & diagnostics
        # set ruleset dropdown selection
        self._refresh_rulesets()
        name = str(cfg.ruleset_name)
        idx = self.ruleset_name.findText(name)
        if idx >= 0:
            self.ruleset_name.setCurrentIndex(idx)
        elif name:
            self.ruleset_name.addItem(name)
            self.ruleset_name.setCurrentText(name)
        self.en_events.setChecked(bool(cfg.enable_event_statistics))
        self.en_states.setChecked(bool(cfg.enable_surface_state_statistics))
        self.en_coord.setChecked(bool(cfg.enable_coordination_statistics))
        self.wall_x_min.setChecked(bool(cfg.walls.wall_x_min))
        self.wall_x_max.setChecked(bool(cfg.walls.wall_x_max))
        self.wall_y_min.setChecked(bool(cfg.walls.wall_y_min))
        self.wall_y_max.setChecked(bool(cfg.walls.wall_y_max))
        self.wall_z_min.setChecked(bool(cfg.walls.wall_z_min))
        self.wall_z_max.setChecked(bool(cfg.walls.wall_z_max))
        # compute & output
        self.cuda_index.setValue(int(cfg.cuda_device_index))
        self.en_early_stop.setChecked(bool(cfg.enable_early_termination))
        try:
            self.early_win.setValue(int(getattr(cfg, "early_termination_window", 100)))
            self.early_tol.setValue(float(getattr(cfg, "early_termination_tolerance", 0.03)))
        except Exception:
            pass
        self.output_dir.setText(str(cfg.output_dir))

    def _collect_ui_to_cfg(self) -> SimulationConfig:
        # Create or update current config from UI fields we have in this skeleton
        base: Optional[SimulationConfig] = self._cfg_svc.current
        data = base.model_dump() if base else {"output_dir": "./output/"}
        # Translate slider indices back to actual sizes
        def from_index(idx: int) -> int:
            idx = max(0, min(int(idx), len(self._pow2_sizes) - 1))
            return int(self._pow2_sizes[idx])
        data.update({
            "lattice_size_x": from_index(int(self.size_x.value())),
            "lattice_size_y": from_index(int(self.size_y.value())),
            "lattice_size_z": from_index(int(self.size_z.value())),
            "num_iterations": int(self.num_iter.value()),
            # Snapshot intervals persisted
            "data_snapshot_interval_steps": int(self.data_snap.value()),
            "xyz_snapshot_interval_steps": int(self.xyz_snap.value()),
            "xyz_save_last_only": bool(self.xyz_last_only.isChecked()),
            "snapshot_schedule_mode": self.snap_mode.currentText() or "linear",
            "log_snapshot_count": int(self.log_count.value()),
            "initial_occupancy_fraction": float(self.initial_occ.value()),
            "ruleset_name": self.ruleset_name.currentText() or "Default",
            "cuda_device_index": int(self.cuda_index.value()),
            "enable_early_termination": bool(self.en_early_stop.isChecked()),
            "early_termination_window": int(self.early_win.value()),
            "early_termination_tolerance": float(self.early_tol.value()),
            "enable_event_statistics": bool(self.en_events.isChecked()),
            "enable_surface_state_statistics": bool(self.en_states.isChecked()),
            "enable_coordination_statistics": bool(self.en_coord.isChecked()),
            "enable_age_calculation": bool(self.en_age.isChecked()),
            "output_dir": self.output_dir.text() or data.get("output_dir", "./output/"),
            "rng_seed": int(self.rng_seed.value()) or None,
        })
        # walls
        data["walls"] = {
            "wall_x_min": self.wall_x_min.isChecked(),
            "wall_x_max": self.wall_x_max.isChecked(),
            "wall_y_min": self.wall_y_min.isChecked(),
            "wall_y_max": self.wall_y_max.isChecked(),
            "wall_z_min": self.wall_z_min.isChecked(),
            "wall_z_max": self.wall_z_max.isChecked(),
        }
        # recipe values
        data["recipe_values"] = self._collect_recipe_values()
        # init mode with per-mode fields
        data["init"] = self._read_init_from_ui()
        return SimulationConfig(**data)

    def _refresh_rulesets(self) -> None:
        names = list_rulesets() or []
        if not names:
            names = ["Default"]
        prev = self.ruleset_name.currentText()
        self.ruleset_name.blockSignals(True)
        self.ruleset_name.clear()
        self.ruleset_name.addItems(names)
        # Always select the first item if previous isn't present, so UI updates
        if prev:
            idx = self.ruleset_name.findText(prev)
            if idx >= 0:
                self.ruleset_name.setCurrentIndex(idx)
            else:
                self.ruleset_name.setCurrentIndex(0)
        else:
            self.ruleset_name.setCurrentIndex(0)
        self.ruleset_name.blockSignals(False)
        # Trigger table refresh explicitly
        try:
            self._populate_probabilities_table()
        except Exception:
            pass

    def eventFilter(self, obj, event):
        return super().eventFilter(obj, event)

    def _populate_probabilities_table(self) -> None:
        try:
            name = self.ruleset_name.currentText() or "Default"
            params = get_recipe_parameters(name) or []
        except Exception:
            params = []
        self.prob_table.setRowCount(0)
        # Load current config values
        try:
            from config import load_config
            cfg = load_config(self.cfg_path.text() or "config.json")
            values_map = dict(getattr(cfg, "recipe_values", {}) or {})
        except Exception:
            values_map = {}
        for p in params:
            row = self.prob_table.rowCount()
            self.prob_table.insertRow(row)
            label = QtWidgets.QTableWidgetItem(str(p.get("label", "")))
            label.setFlags(label.flags() & ~QtCore.Qt.ItemIsEditable)
            desc = QtWidgets.QTableWidgetItem(str(p.get("description", "")))
            desc.setFlags(desc.flags() & ~QtCore.Qt.ItemIsEditable)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            try:
                spin.setDecimals(5)
            except Exception:
                pass
            spin.setSingleStep(0.00001)
            key = str(p.get("key"))
            val = values_map.get(key, p.get("default", 0.0))
            try:
                spin.setValue(float(val))
            except Exception:
                spin.setValue(float(p.get("default", 0.0)))
            # Store the key on the label item (hidden from view)
            try:
                label.setData(QtCore.Qt.UserRole, key)
            except Exception:
                pass
            # Improve readability
            try:
                label.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                desc.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                # Tooltips for full text
                label.setToolTip(label.text())
                desc.setToolTip(desc.text())
            except Exception:
                pass
            self.prob_table.setItem(row, 0, label)
            self.prob_table.setItem(row, 1, desc)
            self.prob_table.setCellWidget(row, 2, spin)
        try:
            # Adjust heights after population to show wrapped text
            self.prob_table.resizeRowsToContents()
        except Exception:
            pass

    def _collect_recipe_values(self) -> dict:
        out = {}
        for row in range(self.prob_table.rowCount()):
            # Retrieve key stored on label item
            label_item = self.prob_table.item(row, 0)
            key = str(label_item.data(QtCore.Qt.UserRole)) if label_item else ""
            spin = self.prob_table.cellWidget(row, 2)
            if key and isinstance(spin, QtWidgets.QDoubleSpinBox):
                out[key] = float(spin.value())
        return out

    def _build_tick_labels_row(self, values: list[int]) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        try:
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(0)
        except Exception:
            pass
        for i, v in enumerate(values):
            lbl = QtWidgets.QLabel(str(v))
            try:
                lbl.setAlignment(QtCore.Qt.AlignCenter)
            except Exception:
                pass
            grid.addWidget(lbl, 0, i)
            try:
                grid.setColumnStretch(i, 1)
            except Exception:
                pass
        return w

    def _on_tab_changed(self, index: int) -> None:
        try:
            if hasattr(self, "_rules_tab_index") and index == self._rules_tab_index:
                prev = self.ruleset_name.currentText()
                self._refresh_rulesets()
                if prev:
                    idx = self.ruleset_name.findText(prev)
                    if idx >= 0:
                        self.ruleset_name.setCurrentIndex(idx)
        except Exception:
            pass

    # --- Actions ---
    def _on_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load config.json", str(Path.cwd()), "JSON (*.json)")
        if not path:
            return
        cfg = self._cfg_svc.load(path)
        self.cfg_path.setText(path)
        self._apply_cfg_to_ui(cfg)

    def _on_save(self) -> None:
        cfg = self._collect_ui_to_cfg()
        self._cfg_svc.current = cfg
        try:
            self._cfg_svc.save()
        except PermissionError as e:
            QtWidgets.QMessageBox.warning(self, "Protected config", str(e))

    def _on_save_as(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save config as", self.cfg_path.text() or "config.json", "JSON (*.json)")
        if not path:
            return
        cfg = self._collect_ui_to_cfg()
        self._cfg_svc.current = cfg
        try:
            self._cfg_svc.save_as(path)
            self.cfg_path.setText(path)
        except PermissionError as e:
            QtWidgets.QMessageBox.warning(self, "Protected config", str(e))

    def _on_reset(self) -> None:
        cfg = self._cfg_svc.reset()
        self.cfg_path.clear()
        self._apply_cfg_to_ui(cfg)

    def _on_browse_output(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", self.output_dir.text() or str(Path.cwd()))
        if d:
            self.output_dir.setText(d)

    def _on_field_changed(self) -> None:
        # Minimal live validation example
        try:
            cfg = self._collect_ui_to_cfg()
            self._cfg_svc.validate(cfg)
            self._set_status_ok()
        except Exception as e:  # show simple error
            self._set_status_error(str(e))

    def _set_status_ok(self) -> None:
        self.progress.setFormat("")
        self.progress.setStyleSheet("")

    def _set_status_error(self, msg: str) -> None:
        self.progress.setFormat(msg[:80])
        self.progress.setStyleSheet("QProgressBar { color: red; }")

    def _on_preview(self) -> None:
        try:
            cfg = self._collect_ui_to_cfg()
            self._cfg_svc.validate(cfg)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid configuration", str(e))
            return
        # Generate initial atoms and show in the GL preview
        try:
            from CA3D_functions import prepare_initial_atoms
            atoms = prepare_initial_atoms(cfg)
            self._last_atoms = atoms
            walls_dict = {
                "wall_x_min": bool(getattr(cfg.walls, "wall_x_min", False)),
                "wall_x_max": bool(getattr(cfg.walls, "wall_x_max", False)),
                "wall_y_min": bool(getattr(cfg.walls, "wall_y_min", False)),
                "wall_y_max": bool(getattr(cfg.walls, "wall_y_max", False)),
                "wall_z_min": bool(getattr(cfg.walls, "wall_z_min", False)),
                "wall_z_max": bool(getattr(cfg.walls, "wall_z_max", False)),
            }
            self.preview.show_atoms(
                atoms,
                include_mobile=bool(self.chk_include_mobile.isChecked()),
                walls=walls_dict,
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Preview failed", str(e))

    # --- Init mode UI helpers ---
    def _build_init_mode_pages(self) -> None:
        # single
        w_single = QtWidgets.QWidget(); gl = QtWidgets.QGridLayout(w_single)
        self.single_seed_edge = QtWidgets.QSpinBox(); self.single_seed_edge.setRange(1, 1024)
        self.single_seed_edge.setToolTip("Edge length of the single cubic seed (config: init.seed_edge_length)")
        lbl_s_se = QtWidgets.QLabel("Seed edge length (cells)"); lbl_s_se.setToolTip("config: init.seed_edge_length")
        gl.addWidget(lbl_s_se, 0, 0); gl.addWidget(self.single_seed_edge, 0, 1)
        self._mode_to_index["single"] = self.init_mode_stack.addWidget(w_single)
        # multiple
        w_multiple = QtWidgets.QWidget(); gm = QtWidgets.QGridLayout(w_multiple)
        self.mult_seed_edge = QtWidgets.QSpinBox(); self.mult_seed_edge.setRange(1, 1024)
        self.mult_seed_edge.setToolTip("Edge length of each seed (config: init.seed_edge_length)")
        self.mult_num_seeds = QtWidgets.QSpinBox(); self.mult_num_seeds.setRange(1, 1_000_000)
        self.mult_num_seeds.setToolTip("Number of seeds placed randomly (config: init.num_seeds)")
        lbl_m_se = QtWidgets.QLabel("Seed edge length (cells)"); lbl_m_se.setToolTip("config: init.seed_edge_length")
        lbl_m_ns = QtWidgets.QLabel("Number of seeds"); lbl_m_ns.setToolTip("config: init.num_seeds")
        gm.addWidget(lbl_m_se, 0, 0); gm.addWidget(self.mult_seed_edge, 0, 1)
        gm.addWidget(lbl_m_ns, 1, 0); gm.addWidget(self.mult_num_seeds, 1, 1)
        self._mode_to_index["multiple"] = self.init_mode_stack.addWidget(w_multiple)
        # flat_bottom
        w_flat = QtWidgets.QWidget(); gf = QtWidgets.QGridLayout(w_flat)
        self.flat_num_layers = QtWidgets.QSpinBox(); self.flat_num_layers.setRange(0, 1_000_000)
        self.flat_num_layers.setToolTip("Initial number of flat layers at bottom (config: init.num_flat_layers)")
        lbl_f_nl = QtWidgets.QLabel("Number of flat layers"); lbl_f_nl.setToolTip("config: init.num_flat_layers")
        gf.addWidget(lbl_f_nl, 0, 0); gf.addWidget(self.flat_num_layers, 0, 1)
        self._mode_to_index["flat_bottom"] = self.init_mode_stack.addWidget(w_flat)
        # stepped_bottom
        w_step = QtWidgets.QWidget(); gs = QtWidgets.QGridLayout(w_step)
        self.step_num_terraces = QtWidgets.QSpinBox(); self.step_num_terraces.setRange(1, 1_000_000)
        self.step_num_terraces.setToolTip("Number of terraces in staircase bottom (config: init.num_terraces)")
        lbl_st_nt = QtWidgets.QLabel("Number of terraces"); lbl_st_nt.setToolTip("config: init.num_terraces")
        gs.addWidget(lbl_st_nt, 0, 0); gs.addWidget(self.step_num_terraces, 0, 1)
        self._mode_to_index["stepped_bottom"] = self.init_mode_stack.addWidget(w_step)
        # pair
        w_pair = QtWidgets.QWidget(); gp = QtWidgets.QGridLayout(w_pair)
        self.pair_center_sep = QtWidgets.QSpinBox(); self.pair_center_sep.setRange(0, 1_000_000)
        self.pair_center_sep.setToolTip("Distance between seed centers (config: init.center_separation)")
        self.pair_seed1_edge = QtWidgets.QSpinBox(); self.pair_seed1_edge.setRange(1, 1024)
        self.pair_seed1_edge.setToolTip("Edge length of seed 1 (config: init.seed1_edge_length)")
        self.pair_seed2_edge = QtWidgets.QSpinBox(); self.pair_seed2_edge.setRange(1, 1024)
        self.pair_seed2_edge.setToolTip("Edge length of seed 2 (config: init.seed2_edge_length)")
        lbl_p_sep = QtWidgets.QLabel("Center separation (cells)"); lbl_p_sep.setToolTip("config: init.center_separation")
        lbl_p_s1 = QtWidgets.QLabel("Seed 1 edge length (cells)"); lbl_p_s1.setToolTip("config: init.seed1_edge_length")
        lbl_p_s2 = QtWidgets.QLabel("Seed 2 edge length (cells)"); lbl_p_s2.setToolTip("config: init.seed2_edge_length")
        gp.addWidget(lbl_p_sep, 0, 0); gp.addWidget(self.pair_center_sep, 0, 1)
        gp.addWidget(lbl_p_s1, 1, 0); gp.addWidget(self.pair_seed1_edge, 1, 1)
        gp.addWidget(lbl_p_s2, 2, 0); gp.addWidget(self.pair_seed2_edge, 2, 1)
        self._mode_to_index["pair"] = self.init_mode_stack.addWidget(w_pair)
        # mounds
        w_mounds = QtWidgets.QWidget(); gmnd = QtWidgets.QGridLayout(w_mounds)
        self.mounds_w = QtWidgets.QDoubleSpinBox(); self.mounds_w.setRange(0.0, 100.0); self.mounds_w.setDecimals(6); self.mounds_w.setSingleStep(0.01)
        self.mounds_amp = QtWidgets.QSpinBox(); self.mounds_amp.setRange(0, 1_000_000)
        lbl_m_w = QtWidgets.QLabel("Spatial frequency w"); lbl_m_w.setToolTip("Frequency in h(x,y)=1+A*(1+sin(w*x)*sin(w*y)) (config: init.w)")
        lbl_m_a = QtWidgets.QLabel("Amplitude A (cells)"); lbl_m_a.setToolTip("Mound height in cells (config: init.amplitude)")
        self.mounds_w.setToolTip("config: init.w"); self.mounds_amp.setToolTip("config: init.amplitude")
        gmnd.addWidget(lbl_m_w, 0, 0); gmnd.addWidget(self.mounds_w, 0, 1)
        gmnd.addWidget(lbl_m_a, 1, 0); gmnd.addWidget(self.mounds_amp, 1, 1)
        self._mode_to_index["mounds"] = self.init_mode_stack.addWidget(w_mounds)
        # python_code
        w_py = QtWidgets.QWidget(); vpy = QtWidgets.QVBoxLayout(w_py)
        self.py_code = QtWidgets.QPlainTextEdit(); self.py_code.setPlaceholderText("""# Python code to initialize lattice\n# array is a NumPy int32 of shape [X, Y, Z]\n# Example: fill three bottom layers crystalline\n# array[:, :, 0:3] = 2\n"""); self.py_code.setToolTip("Python snippet to set initial lattice (config: init.init_python_code)")
        vpy.addWidget(self.py_code)
        self._mode_to_index["python_code"] = self.init_mode_stack.addWidget(w_py)
        # none
        w_none = QtWidgets.QWidget(); vnone = QtWidgets.QVBoxLayout(w_none)
        vnone.addWidget(QtWidgets.QLabel("No additional parameters"))
        self._mode_to_index["none"] = self.init_mode_stack.addWidget(w_none)

    def _on_init_mode_changed(self, mode: str) -> None:
        try:
            idx = self._mode_to_index.get(mode, 0)
            self.init_mode_stack.setCurrentIndex(idx)
            self._on_field_changed()
        except Exception:
            pass

    def _write_init_to_ui(self, cfg: SimulationConfig) -> None:
        # Set combobox and stacked page; fill fields
        try:
            self.init_mode.setCurrentText(str(cfg.init.mode))  # type: ignore[attr-defined]
        except Exception:
            pass
        mode = self.init_mode.currentText()
        self._on_init_mode_changed(mode)
        from config import SingleInitConfig, MultipleInitConfig, FlatBottomInitConfig, PythonCodeInitConfig, NoneInitConfig, SteppedBottomInitConfig, PairInitConfig, MoundsInitConfig
        if isinstance(cfg.init, SingleInitConfig):
            self.single_seed_edge.setValue(int(cfg.init.seed_edge_length))
        elif isinstance(cfg.init, MultipleInitConfig):
            self.mult_seed_edge.setValue(int(cfg.init.seed_edge_length))
            self.mult_num_seeds.setValue(int(cfg.init.num_seeds))
        elif isinstance(cfg.init, FlatBottomInitConfig):
            self.flat_num_layers.setValue(int(cfg.init.num_flat_layers))
        elif isinstance(cfg.init, SteppedBottomInitConfig):
            self.step_num_terraces.setValue(int(cfg.init.num_terraces))
        elif isinstance(cfg.init, PairInitConfig):
            self.pair_center_sep.setValue(int(cfg.init.center_separation))
            self.pair_seed1_edge.setValue(int(cfg.init.seed1_edge_length))
            self.pair_seed2_edge.setValue(int(cfg.init.seed2_edge_length))
        elif isinstance(cfg.init, MoundsInitConfig):
            self.mounds_w.setValue(float(cfg.init.w))
            self.mounds_amp.setValue(int(cfg.init.amplitude))
        elif isinstance(cfg.init, PythonCodeInitConfig):
            self.py_code.setPlainText(str(cfg.init.init_python_code or ""))
        elif isinstance(cfg.init, NoneInitConfig):
            pass

    def _read_init_from_ui(self) -> dict:
        mode = self.init_mode.currentText()
        if mode == "single":
            return {"mode": mode, "seed_edge_length": int(self.single_seed_edge.value())}
        if mode == "multiple":
            return {"mode": mode, "seed_edge_length": int(self.mult_seed_edge.value()), "num_seeds": int(self.mult_num_seeds.value())}
        if mode == "flat_bottom":
            return {"mode": mode, "num_flat_layers": int(self.flat_num_layers.value())}
        if mode == "stepped_bottom":
            return {"mode": mode, "num_terraces": int(self.step_num_terraces.value())}
        if mode == "pair":
            return {"mode": mode, "center_separation": int(self.pair_center_sep.value()), "seed1_edge_length": int(self.pair_seed1_edge.value()), "seed2_edge_length": int(self.pair_seed2_edge.value())}
        if mode == "mounds":
            return {"mode": mode, "w": float(self.mounds_w.value()), "amplitude": int(self.mounds_amp.value())}
        if mode == "python_code":
            return {"mode": mode, "init_python_code": self.py_code.toPlainText()}
        # none
        return {"mode": "none"}

    def _on_start(self) -> None:
        try:
            # Check CUDA availability before proceeding
            try:
                import pycuda.driver as _drv  # type: ignore
                _drv.init()
                _cnt = _drv.Device.count()
                if _cnt <= 0:
                    raise RuntimeError("No CUDA devices")
            except Exception:
                QtWidgets.QMessageBox.warning(
                    self,
                    "CUDA not available",
                    "CUDA-enabled graphic card is not found. Please ensure it is present and all drivers are installed correctly.",
                )
                return
            # Check output directory cleanliness (resolve relative to runner cwd)
            out_dir = self._resolve_output_dir(self.output_dir.text() or "")
            if out_dir.exists() and any(out_dir.iterdir()):
                resp = QtWidgets.QMessageBox.question(
                    self,
                    "Non-empty output directory",
                    f"The output directory '{out_dir}' is not empty.\nDelete existing files before starting?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                if resp == QtWidgets.QMessageBox.Yes:
                    # Best-effort clean of entire directory contents (files and subdirectories)
                    clear_output_directory_contents(out_dir)
                else:
                    return
            # Save current config to ensure CA3D.py picks it up
            cfg = self._collect_ui_to_cfg()
            self._cfg_svc.current = cfg
            self._cfg_svc.save()
            # Pass explicit config path if known (supports configs outside project root)
            cfg_path = self._cfg_svc.current_path if getattr(self._cfg_svc, "current_path", None) else None
            self._runner.start(str(cfg_path) if cfg_path else None)
            self.progress.setValue(0)
            self.progress.setFormat("%p%")
            self._hist.clear()
            self._t0 = time.time()
            self.run_status.setText("Running")
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self._poll_timer.start()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Start failed", str(e))

    def _on_stop(self) -> None:
        try:
            self._runner.stop()
            self._poll_timer.stop()
            self.run_status.setText("Stopped")
            self.progress.setFormat("Stopped %p%")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Stop failed", str(e))

    def _on_toggle_include_mobile(self) -> None:
        try:
            if self._last_atoms is not None:
                self.preview.show_atoms(self._last_atoms, include_mobile=bool(self.chk_include_mobile.isChecked()))
        except Exception:
            pass

    def _poll_progress(self) -> None:
        # Estimate progress from stats file
        try:
            cfg = self._cfg_svc.current or self._collect_ui_to_cfg()
            # Resolve cfg.output_dir relative to runner cwd if not absolute
            out_dir = self._resolve_output_dir(str(cfg.output_dir))
            # If process already ended, consider run completed for UI purposes
            if not self._runner.is_running:
                self._poll_timer.stop()
                self.progress.setValue(100)
                self.run_status.setText("Completed")
                self.progress.setFormat("Completed 100%")
                self.btn_start.setEnabled(True)
                self.btn_stop.setEnabled(False)
                return
            # Determine total iterations from used_params.json if available
            total = int(cfg.num_iterations)
            used_params = out_dir / "used_params.json"
            if used_params.exists():
                try:
                    import json as _json
                    data = _json.loads(used_params.read_text(encoding="utf-8"))
                    params = data.get("parameters") or {}
                    if isinstance(params, dict) and "num_iterations" in params:
                        total = int(params.get("num_iterations", total))
                except Exception:
                    pass

            # Prefer progress from system_info.log snapshots
            log_path = out_dir / "system_info.log"
            pct_from_log: int | None = None
            step_from_log: int | None = None
            if log_path.exists():
                try:
                    with open(log_path, "rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        tail = 16384
                        f.seek(max(0, size - tail))
                        chunk = f.read().decode(errors="ignore")
                    if "CA experiment completed successfully" in chunk:
                        self.progress.setValue(100)
                        self.run_status.setText("Completed")
                        self.progress.setFormat("Completed 100%")
                        self.btn_start.setEnabled(True)
                        self.btn_stop.setEnabled(False)
                        return
                    import re as _re
                    m = None
                    for m in _re.finditer(r"Snapshot\s+(\d+)\s+saved", chunk):
                        pass
                    if m:
                        step_from_log = int(m.group(1))
                        pct_from_log = max(0, min(100, int(step_from_log * 100 / max(1, total))))
                except Exception:
                    pass

            stats_path = out_dir / "system_evolution_stats.dat"
            if not stats_path.exists():
                # try stem-based name just in case
                stats_candidates = list(out_dir.glob("*_stats.dat"))
                if stats_candidates:
                    stats_path = stats_candidates[0]
                else:
                    # Use log-derived progress if present
                    if pct_from_log is not None:
                        self.progress.setValue(pct_from_log)
                        if pct_from_log >= 100:
                            self.run_status.setText("Completed")
                            self.progress.setFormat("Completed 100%")
                            self.btn_start.setEnabled(True)
                            self.btn_stop.setEnabled(False)
                        return
                    return
            # Read last non-empty, non-header line
            last_line = ""
            with open(stats_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("Timestep"):
                        continue
                    last_line = line
            # Build candidates from tqdm, log snapshot, and stats
            pct_tqdm: int | None = None
            pct_log: int | None = pct_from_log
            pct_stats: int | None = None
            step_val: int | None = None
            if getattr(self._runner, 'last_percent', None) is not None:
                pct_tqdm = int(self._runner.last_percent)
            if last_line:
                parts = last_line.split()
                step_val = int(float(parts[0]))
                pct_stats = max(0, min(100, int(step_val * 100 / max(1, total))))
            # choose the highest available as most advanced signal
            candidates = [p for p in (pct_tqdm, pct_log, pct_stats) if p is not None]
            if not candidates:
                return
            pct = max(candidates)
            self.progress.setValue(int(pct))
            # Update status with snapshot info if available
            if step_from_log is not None:
                self.run_status.setText(f"Running (snapshot {step_from_log}/{total})")
            # Update ETA based on moving average of step rate
            if step_val is not None and total > 0:
                now = time.time()
                self._hist.append((now, step_val))
                if len(self._hist) >= 2:
                    t0, s0 = self._hist[0]
                    t1, s1 = self._hist[-1]
                    dt = max(0.001, t1 - t0)
                    ds = max(0, s1 - s0)
                    rate = ds / dt  # steps per second
                    remaining = max(0, total - s1)
                    eta_s = remaining / rate if rate > 0 else None
                    if eta_s is not None:
                        h = int(eta_s // 3600)
                        m = int((eta_s % 3600) // 60)
                        s = int(eta_s % 60)
                        self.progress.setFormat(f"%p% - ETA {h:02d}:{m:02d}:{s:02d}")
            if not self._runner.is_running:
                self._poll_timer.stop()
                # If process ended, force completion for clarity
                self.progress.setValue(100)
                self.run_status.setText("Completed")
                self.progress.setFormat("Completed 100%")
                self.btn_start.setEnabled(True)
                self.btn_stop.setEnabled(False)
        except Exception:
            # Keep UI resilient; ignore polling errors
            pass


