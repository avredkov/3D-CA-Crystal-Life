from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys
import os
import subprocess
from typing import Optional, Dict
from html import escape

from PySide6 import QtWidgets, QtCore, QtGui
from gui.services.config_service import ConfigService
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from gui.services.data_parsers import load_series
import sys
from pathlib import Path
import io
import json
from contextlib import redirect_stdout
# Add utils directory to path for cluster analysis
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils"))
from cluster_analysis_gui import analyze_xyz_file
import importlib.util
import sys

# Import the plotting functions from the file with spaces in the name
plot_path = Path(__file__).parent.parent.parent / "utils" / "Plot cluster paths.py"
spec = importlib.util.spec_from_file_location("plot_cluster_paths", plot_path)
plot_module = importlib.util.module_from_spec(spec)
sys.modules["plot_cluster_paths"] = plot_module
spec.loader.exec_module(plot_module)

plot_histogram_data_v4 = plot_module.plot_histogram_data_v4
read_histogram_data = plot_module.read_histogram_data


class PlotViewerDialog(QtWidgets.QDialog):
    """Custom dialog for displaying plot images."""
    
    def __init__(self, parent=None, title="Plot Viewer", image_path=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(1200, 800)
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create scroll area for the image
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Create label for the image
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        scroll_area.setWidget(self.image_label)
        
        # Add buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.open_folder_button = QtWidgets.QPushButton("Open Folder")
        self.close_button = QtWidgets.QPushButton("Close")
        button_layout.addWidget(self.open_folder_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)
        
        # Connect signals
        self.close_button.clicked.connect(self.accept)
        self.open_folder_button.clicked.connect(self.open_folder)
        
        self.image_path = image_path
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)


class InteractiveClusterDistributionDialog(QtWidgets.QDialog):
    """Qt dialog that embeds an interactive Matplotlib figure for cluster distribution.

    The dialog provides a toolbar for pan/zoom and saves the figure to the given path
    when closing, ensuring the interactive view is also persisted to disk.
    """
    def __init__(self, parent=None, title="Cluster Distribution Analysis", save_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(1400, 900)
        self._save_path = save_path

        layout = QtWidgets.QVBoxLayout(self)

        # Matplotlib canvas and toolbar (higher DPI for crisper text)
        self.figure = Figure(figsize=(12, 7), dpi=300, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Buttons row
        row = QtWidgets.QHBoxLayout()
        row.addStretch()
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_close = QtWidgets.QPushButton("Close")
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_close)
        layout.addLayout(row)

        self.btn_close.clicked.connect(self.accept)
        self.btn_save.clicked.connect(self._on_save)

    def plot_cluster_distribution(self, timesteps, size_bins, hist_data, alpha_times, alpha_values):
        """Render the interactive plot using Matplotlib primitives (no pyplot).

        Args:
            timesteps: 1D array-like of timestep values
            size_bins: 1D array-like of cluster size bins
            hist_data: 2D numpy array of shape (len(timesteps), len(size_bins))
            alpha_times: 1D array-like for alpha timeline
            alpha_values: 1D array-like for alpha values
        """
        import numpy as np
        from matplotlib.colors import LogNorm

        fig = self.figure
        fig.clear()

        # Gridspec to place colorbar on the right
        gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.1)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])

        ax1 = ax_main
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()

        # Move secondary axes to the left for multi-axis layout
        ax2.yaxis.set_label_position('left'); ax2.yaxis.tick_left()
        ax3.yaxis.set_label_position('left'); ax3.yaxis.tick_left()
        ax2.spines['right'].set_visible(False); ax2.spines['left'].set_visible(True)
        ax3.spines['right'].set_visible(False); ax3.spines['left'].set_visible(True)

        # Offset the left spines outward to avoid overlap but keep all visible
        ax1.spines['left'].set_position(('outward', 0))
        ax2.spines['left'].set_position(('outward', 30))   # halved from 50
        ax3.spines['left'].set_position(('outward', 60))   # halved from 100

        # Compute aggregates
        timesteps = np.asarray(timesteps)
        size_bins = np.asarray(size_bins)
        hist_data = np.asarray(hist_data)
        total_clusters = hist_data.sum(axis=1)
        mean_size = np.zeros_like(total_clusters, dtype=float)
        for i in range(len(timesteps)):
            if total_clusters[i] > 0:
                mean_size[i] = np.sum(hist_data[i] * size_bins) / total_clusters[i]

        # Build scatter data
        T, S = np.meshgrid(timesteps, size_bins)
        t_flat = T.flatten(); s_flat = S.flatten(); counts_flat = hist_data.T.flatten()
        mask = counts_flat > 0
        t_plot = t_flat[mask]; s_plot = s_flat[mask]; counts_plot = counts_flat[mask]

        # Symbol sizes and norm (reduced sizes for a lighter visual)
        sizes = np.interp(counts_plot, (counts_plot.min(), counts_plot.max()), (1, 10))
        norm = LogNorm(vmin=max(counts_plot.min(), 1e-9), vmax=counts_plot.max())

        scatter = ax1.scatter(t_plot, s_plot, c=counts_plot, s=sizes, alpha=0.7, norm=norm, cmap='viridis', label='Cluster Distribution')

        # Mean size (ax1) and totals (ax2) with thinner lines and high-contrast colors
        mean_color = '#ff7f0e'        # rich orange
        total_color = '#1f77b4'       # vivid blue
        alpha_color = '#2ca02c'       # vibrant green
        ax1.plot(timesteps, mean_size, color=mean_color, linewidth=1, label='Mean Size', antialiased=True)
        ax2.plot(timesteps, total_clusters, color=total_color, linewidth=1, label='Total Number of Clusters', antialiased=True)

        # Alpha (ax3)
        ax3.plot(alpha_times, alpha_values, color=alpha_color, linewidth=1, label='α')

        # Scales and labels
        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.set_xlim(left=max(10, float(timesteps.min() if len(timesteps) else 10)))
        # Smaller, sharper fonts and axis title colors matching lines
        ax1.set_xlabel('Timestep', fontsize=4)
        ax1.set_ylabel('Cluster Size / Mean size', fontsize=4, color=mean_color, labelpad=2)
        ax2.set_ylabel('Total Number of Clusters', color=total_color, fontsize=4, labelpad=2)
        ax3.set_ylabel('α', color=alpha_color, rotation=0, labelpad=2, fontsize=4)
        ax1.tick_params(axis='both', labelsize=4, pad=1)
        ax1.tick_params(axis='y', colors=mean_color, pad=1)
        ax2.tick_params(axis='y', labelsize=4, colors=total_color, pad=1)
        ax3.tick_params(axis='y', labelsize=4, colors=alpha_color, pad=1)

        # Colorbar
        cbar = fig.colorbar(scatter, cax=ax_cbar, label='Number of Clusters')
        cbar.ax.tick_params(labelsize=4)
        cbar.ax.yaxis.label.set_size(4)

        # Grid and legend
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', prop={ 'size': 6 })

        self.canvas.draw_idle()

    def _on_save(self):
        if not self._save_path:
            dlg = QtWidgets.QFileDialog(self, "Save figure as", str(Path.cwd()), "PNG Files (*.png);;All Files (*)")
            if dlg.exec():
                files = dlg.selectedFiles()
                if files:
                    self._save_path = files[0]
        if self._save_path:
            try:
                self.figure.savefig(self._save_path, dpi=300, bbox_inches='tight')
                QtWidgets.QMessageBox.information(self, "Saved", f"Figure saved to: {self._save_path}")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Save failed", str(e))
                
    def accept(self) -> None:
        # Auto-save on close if a path is provided
        if self._save_path:
            try:
                self.figure.savefig(self._save_path, dpi=300, bbox_inches='tight')
            except Exception:
                pass
        return super().accept()
    
    def load_image(self, image_path):
        """Load and display an image."""
        try:
            pixmap = QtGui.QPixmap(image_path)
            if not pixmap.isNull():
                # Scale the image to fit the dialog while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.size() * 0.9, 
                    QtCore.Qt.KeepAspectRatio, 
                    QtCore.Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.image_path = image_path
            else:
                self.image_label.setText("Failed to load image")
        except Exception as e:
            self.image_label.setText(f"Error loading image: {str(e)}")
    
    def open_folder(self):
        """Open the folder containing the plot."""
        if self.image_path and os.path.exists(self.image_path):
            folder_path = os.path.dirname(self.image_path)
            if os.name == 'nt':  # Windows
                os.startfile(folder_path)
            elif os.name == 'posix':  # macOS and Linux
                subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', folder_path])


class ClusterAnalysisProgressDialog(QtWidgets.QDialog):
    """Custom progress dialog for cluster analysis with log display."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Analysis Progress")
        self.setModal(True)
        self.resize(600, 400)
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)
        
        # Log display
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QtGui.QFont("Consolas", 9))
        layout.addWidget(self.log_text)
        
        # Cancel button
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        
        # Store original stdout
        self.original_stdout = sys.stdout
        self.log_buffer = io.StringIO()
        
    def start_logging(self):
        """Start redirecting stdout to the log display."""
        sys.stdout = self.log_buffer
        
    def stop_logging(self):
        """Stop redirecting stdout and restore original."""
        sys.stdout = self.original_stdout
        
    def update_log(self):
        """Update the log display with new content."""
        content = self.log_buffer.getvalue()
        if content:
            self.log_text.append(content)
            self.log_buffer.seek(0)
            self.log_buffer.truncate(0)
            # Auto-scroll to bottom
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
            QtWidgets.QApplication.processEvents()
            
    def add_message(self, message: str):
        """Add a message directly to the log display."""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        QtWidgets.QApplication.processEvents()
        
    def closeEvent(self, event):
        """Ensure stdout is restored when dialog is closed."""
        self.stop_logging()
        super().closeEvent(event)


class AnalysisTab(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        self.series_map: Dict[str, object] = {}

        # Select folder with the simulation results
        run_group = QtWidgets.QGroupBox("Select folder with the simulation results")
        run = QtWidgets.QHBoxLayout(run_group)
        self.out_dir = QtWidgets.QLineEdit()
        btn_browse = QtWidgets.QPushButton("Browse…")
        for w in (self.out_dir, btn_browse):
            run.addWidget(w)
        root.addWidget(run_group)

        # Summary pane
        self.summary = QtWidgets.QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(140)
        root.addWidget(self.summary)

        # Plot area
        plot_group = QtWidgets.QGroupBox("Plots")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        self.plot = pg.PlotWidget()
        self.plot.addLegend()
        # White background for plots
        try:
            self.plot.setBackground('w')
            # Black foreground (axes/labels) and antialiasing
            try:
                pg.setConfigOptions(foreground='k', antialias=True)
            except Exception:
                pass
            try:
                axl = self.plot.getAxis('left'); axb = self.plot.getAxis('bottom')
                axl.setTextPen('k'); axl.setPen('k')
                axb.setTextPen('k'); axb.setPen('k')
            except Exception:
                pass
        except Exception:
            pass
        try:
            self.plot.setLabel('bottom', 'Timestep')
        except Exception:
            pass
        self._color_index = 0
        # Distinct color palette and line styles for better separation
        self._palette = [
            '#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e',
            '#17becf', '#e377c2', '#7f7f7f', '#bcbd22', '#8c564b',
        ]
        self._styles = [
            QtCore.Qt.SolidLine,
            QtCore.Qt.DashLine,
            QtCore.Qt.DotLine,
            QtCore.Qt.DashDotLine,
            QtCore.Qt.DashDotDotLine,
        ]
        plot_layout.addWidget(self.plot)
        # Log-scale toggles
        xy_row = QtWidgets.QHBoxLayout()
        # Removed X/Y selector and Plot Y vs X per request
        self.chk_log_x = QtWidgets.QCheckBox("Log X")
        self.chk_log_y = QtWidgets.QCheckBox("Log Y")
        xy_row.addWidget(self.chk_log_x)
        xy_row.addWidget(self.chk_log_y)
        plot_layout.addLayout(xy_row)

        # Series selector for core/events plotting
        sel_group = QtWidgets.QGroupBox("Series selection")
        sel = QtWidgets.QVBoxLayout(sel_group)
        self.series_list = QtWidgets.QListWidget()
        # Allow per-item toggle without modifiers
        self.series_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        sel_btns = QtWidgets.QHBoxLayout()
        self.btn_clear = QtWidgets.QPushButton("Clear plot")
        self.btn_save_plot = QtWidgets.QPushButton("Save plot")
        sel_btns.addWidget(self.btn_clear)
        sel_btns.addWidget(self.btn_save_plot)
        sel.addWidget(self.series_list)
        sel.addLayout(sel_btns)
        plot_layout.addWidget(sel_group)


        root.addWidget(plot_group)

        # Reserved actions
        act_group = QtWidgets.QGroupBox("Reserved analysis actions")
        act = QtWidgets.QHBoxLayout(act_group)
        self.btn_fractal = QtWidgets.QPushButton("Fractal dimension")
        self.btn_cluster = QtWidgets.QPushButton("Clustering")
        self.btn_distribution = QtWidgets.QPushButton("Plot cluster distribution")
        self.btn_open_ovito = QtWidgets.QPushButton("Open in OVITO")
        for b in (self.btn_fractal, self.btn_distribution):
            b.setEnabled(False)
            act.addWidget(b)
        self.btn_cluster.setEnabled(True)  # Enable clustering button
        act.addWidget(self.btn_cluster)
        self.btn_open_ovito.setEnabled(True)
        act.addWidget(self.btn_open_ovito)
        root.addWidget(act_group)

        # Wiring
        btn_browse.clicked.connect(self._on_browse)
        self.chk_log_x.toggled.connect(self._on_log_toggle)
        self.chk_log_y.toggled.connect(self._on_log_toggle)
        self.btn_clear.clicked.connect(self._on_clear_plot)
        self.btn_save_plot.clicked.connect(self._on_save_plot)
        self.series_list.itemSelectionChanged.connect(self._on_series_selection_changed)
        self.btn_cluster.clicked.connect(self._on_cluster_analysis)
        self.btn_distribution.clicked.connect(self._on_plot_cluster_distribution)
        self.btn_open_ovito.clicked.connect(self._on_open_in_ovito)
        # name -> PlotDataItem mapping to avoid duplicates and support removal
        self._plotted: Dict[str, object] = {}
        # name -> QPen mapping to preserve series colors/styles across selection changes
        self._pen_for_series: Dict[str, object] = {}

        # Try autoload from current config
        try:
            cfg_svc = ConfigService()
            default_cfg = Path(__file__).resolve().parents[2] / "config.json"
            cfg = None
            if default_cfg.exists():
                cfg = cfg_svc.load(str(default_cfg))
            elif cfg_svc.current is not None:
                cfg = cfg_svc.current
            if cfg is not None and getattr(cfg, 'output_dir', None):
                outp = Path(str(cfg.output_dir))
                self.out_dir.setText(str(outp))
                if outp.exists() and any(outp.glob("*.dat")):
                    self._on_load()
                else:
                    self.summary.setPlainText("Output folder is empty or does not exist. Run a simulation first.")
                # OVITO button remains enabled; availability is checked on click
        except Exception:
            pass

    def _on_browse(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", self.out_dir.text() or str(Path.cwd()))
        if d:
            self.out_dir.setText(d)
            # Auto-load results just after selection
            self._on_load()

    def _on_load(self) -> None:
        out = Path(self.out_dir.text().strip())
        if not out.exists():
            QtWidgets.QMessageBox.warning(self, "Invalid directory", "Selected output directory does not exist.")
            return
        # Summary from used_params.json
        used = out / "used_params.json"
        if used.exists():
            try:
                data = json.loads(used.read_text(encoding="utf-8"))
                self.summary.setHtml(self._render_params_html(data, folder=str(out)))
                # Enable reserved actions based on init.mode if available
                mode = None
                if isinstance(data, dict):
                    init = data.get("init") or {}
                    mode = init.get("mode") if isinstance(init, dict) else None
                if mode == "single":
                    self.btn_fractal.setEnabled(True)
                if mode in ("multiple", "none"):
                    self.btn_cluster.setEnabled(True)
                    self.btn_distribution.setEnabled(True)
            except Exception as e:
                self.summary.setHtml(f"<b>Failed to read used_params.json:</b> {escape(str(e))}")
        else:
            self.summary.setHtml("<i>No used_params.json found</i>")

        # Generic load: scan all .dat files and load series (skip overly detailed events file)
        self.plot.clear(); self._color_index = 0
        try:
            self._plotted.clear()
        except Exception:
            pass
        self.series_map.clear()
        # Ensure X axis starts from zero
        try:
            self.plot.setLimits(xMin=0)
        except Exception:
            pass
        for dat in sorted(out.glob("*.dat")):
            # Skip extremely detailed events file(s) to avoid UI overload
            if "all_events" in dat.name.lower():
                continue
            try:
                series = load_series(str(dat))
                if not series:
                    continue
                stem = dat.stem
                # For system_evolution_stats, only update snapshot max; plotting is driven by selection
                # (Snapshot browser removed)
                # Store series with file stem prefix to avoid name collisions
                for k, v in series.items():
                    self.series_map[f"{stem}: {k}"] = v
            except Exception:
                # Skip malformed files
                pass
        # Auto-scale to shown data extents
        self._autoscale_plot()

        # Build (file, series, key) and sort by (file, series); omit all_events
        entries = []
        for key in self.series_map.keys():
            stem, sep, series_name = key.partition(":")
            entries.append((stem.strip(), series_name.strip(), key))
        entries.sort(key=lambda t: (t[0].lower(), t[1].lower()))
        names = []
        for stem, series_name, key in entries:
            if "all_events" in stem.lower():
                continue
            names.append(key)
        # Populate series list
        self.series_list.clear()
        self.series_list.addItems(names)
        # Default-select Alpha, Crystalline atoms, Mobile atoms if present
        defaults = {"Alpha", "Crystalline atoms", "Mobile atoms"}
        for i in range(self.series_list.count()):
            label = self.series_list.item(i).text()
            part = label.split(":", 1)[1].strip() if ":" in label else label
            if part in defaults:
                self.series_list.item(i).setSelected(True)
        # Sync plots to default selection
        self._sync_plots_to_selection()
        # Update reserved actions enablement (OVITO detection + XYZ presence)
        try:
            self._update_reserved_actions_enabled(out)
        except Exception:
            pass


    def _on_log_toggle(self) -> None:
        try:
            self.plot.setLogMode(x=self.chk_log_x.isChecked(), y=self.chk_log_y.isChecked())
            # Ensure limits valid for log scale (strictly > 0)
            if self.chk_log_x.isChecked():
                try:
                    self.plot.setLimits(xMin=1e-9)
                except Exception:
                    pass
            else:
                try:
                    self.plot.setLimits(xMin=0)
                except Exception:
                    pass
            # Prefer pyqtgraph's internal autorange (equivalent to pressing 'a') to avoid extreme defaults
            try:
                self.plot.enableAutoRange(x=True, y=True)
                self.plot.plotItem.vb.autoRange()
            except Exception:
                pass
        except Exception:
            pass

    def _on_plot_selected(self) -> None:
        # Sync plots to current selection
        self._sync_plots_to_selection()
        self._autoscale_plot()

    def _autoscale_plot(self) -> None:
        try:
            bounds = self._compute_bounds()
            if bounds is None:
                return
            xmin, xmax, ymin, ymax = bounds
            try:
                self.plot.enableAutoRange(x=False, y=False)
            except Exception:
                pass
            try:
                self.plot.setXRange(xmin, xmax, padding=0.0)
                self.plot.setYRange(ymin, ymax, padding=0.0)
            except Exception:
                self.plot.enableAutoRange(x=True, y=True)
        except Exception:
            pass

    def _compute_bounds(self):
        """Compute display bounds from selection with robust log handling."""
        try:
            import numpy as np, math
            selected = [it.text() for it in self.series_list.selectedItems()]
            xmins = []; xmaxs = []; ymins = []; ymaxs = []
            if selected:
                for name in selected:
                    s = self.series_map.get(name)
                    if s is None:
                        continue
                    xd = getattr(s, 'x', [])
                    yd = getattr(s, 'y', [])
                    xf, yf = self._filter_for_log_axes(xd, yd)
                    if len(xf) == 0:
                        continue
                    xmins.append(np.nanmin(xf)); xmaxs.append(np.nanmax(xf))
                    ymins.append(np.nanmin(yf)); ymaxs.append(np.nanmax(yf))
            else:
                items = getattr(self.plot, 'listDataItems', lambda: [])()
                for it in items:
                    xd, yd = it.getData()
                    if xd is None or yd is None:
                        continue
                    xf, yf = self._filter_for_log_axes(xd, yd)
                    if len(xf) == 0:
                        continue
                    xmins.append(np.nanmin(xf)); xmaxs.append(np.nanmax(xf))
                    ymins.append(np.nanmin(yf)); ymaxs.append(np.nanmax(yf))
            if not xmins:
                return None
            xmin = float(min(xmins)); xmax = float(max(xmaxs))
            ymin = float(min(ymins)); ymax = float(max(ymaxs))
            # Guard against non-finite/degenerate
            for v in (xmin, xmax, ymin, ymax):
                if not math.isfinite(v):
                    return None
            if xmax <= xmin or ymax <= ymin:
                return None
            # Build final ranges
            if self.chk_log_x.isChecked():
                xmin = max(xmin, 1e-12); xmax = max(xmax, xmin * (1.0 + 1e-9))
                ex_min = math.floor(math.log10(xmin)); ex_max = math.ceil(math.log10(xmax))
                if ex_max - ex_min > 8:
                    ex_min = ex_max - 8
                xmin, xmax = 10.0 ** ex_min, 10.0 ** ex_max
            else:
                xmin = max(xmin, 0.0)
                xr = xmax - xmin
                if xr > 0:
                    xmin -= 0.05 * xr; xmax += 0.05 * xr
            if self.chk_log_y.isChecked():
                ymin = max(ymin, 1e-12); ymax = max(ymax, ymin * (1.0 + 1e-9))
                ey_min = math.floor(math.log10(ymin)); ey_max = math.ceil(math.log10(ymax))
                if ey_max - ey_min > 8:
                    ey_min = ey_max - 8
                ymin, ymax = 10.0 ** ey_min, 10.0 ** ey_max
            else:
                yr = ymax - ymin
                if yr > 0:
                    ymin -= 0.05 * yr; ymax += 0.05 * yr
            return xmin, xmax, ymin, ymax
        except Exception:
            return None

    def _on_clear_plot(self) -> None:
        try:
            self.plot.clear()
            self._color_index = 0
            self._plotted.clear()
            self.series_list.clearSelection()
        except Exception:
            pass

    def _on_save_plot(self) -> None:
        try:
            out_base = Path(self.out_dir.text().strip() or ".")
            plots_dir = out_base / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            # Build filename from selected series names (tags only, without stems)
            selected = [it.text() for it in self.series_list.selectedItems()]
            tag_parts = []
            for label in selected:
                part = label.split(":", 1)[1].strip() if ":" in label else label
                # sanitize
                safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '-' for ch in part)
                tag_parts.append(safe)
            tags = "_".join(tag_parts)[:80] if tag_parts else "plot"
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            target = plots_dir / f"{tags}_{ts}.png"
            exp = ImageExporter(self.plot.plotItem)
            try:
                params = exp.parameters()
                params['width'] = 2000
            except Exception:
                pass
            exp.export(str(target))
            QtWidgets.QMessageBox.information(self, "Plot saved", f"Saved to: {target}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(e))

    def _config_dir(self) -> Path:
        return Path.home() / ".3d_ca_gui"

    def _ovito_path_file(self) -> Path:
        return self._config_dir() / "ovito_path.txt"

    def _read_saved_ovito(self) -> Optional[str]:
        try:
            p = self._ovito_path_file()
            if p.exists():
                val = p.read_text(encoding="utf-8").strip()
                if val and os.path.isfile(val):
                    return val
        except Exception:
            pass
        return None

    def _save_ovito_path(self, path: str) -> None:
        try:
            cfgd = self._config_dir()
            cfgd.mkdir(parents=True, exist_ok=True)
            self._ovito_path_file().write_text(path, encoding="utf-8")
        except Exception:
            pass

    def _detect_ovito(self) -> Optional[str]:
        # Prefer saved path
        saved = self._read_saved_ovito()
        if saved:
            return saved
        try:
            path = shutil.which('ovito')
            if path:
                return path
            # Windows fallback: common installation paths (best-effort)
            if sys.platform.startswith('win'):
                candidates = [
                    os.path.expandvars(r"%ProgramFiles%\OVITO\ovito.exe"),
                    os.path.expandvars(r"%ProgramFiles(x86)%\OVITO\ovito.exe"),
                ]
                for c in candidates:
                    if os.path.isfile(c):
                        return c
        except Exception:
            pass
        return None

    def _find_xyz(self, out: Path) -> Optional[Path]:
        try:
            p = out / "system_evolution.xyz"
            if p.exists():
                return p
            cands = list(out.glob("*.xyz"))
            return cands[0] if cands else None
        except Exception:
            return None

    def _update_reserved_actions_enabled(self, out: Path) -> None:
        # Check if cluster_analysis folder exists to enable distribution button
        cluster_analysis_dir = out / "cluster_analysis"
        if cluster_analysis_dir.exists():
            self.btn_distribution.setEnabled(True)
        else:
            self.btn_distribution.setEnabled(False)

    def _on_cluster_analysis(self) -> None:
        """Perform cluster analysis on the selected XYZ file."""
        try:
            out = Path(self.out_dir.text().strip())
            if not out.exists():
                QtWidgets.QMessageBox.warning(self, "Invalid directory", "Selected output directory does not exist.")
                return
            
            # Find XYZ file
            xyz_file = self._find_xyz(out)
            if not xyz_file:
                QtWidgets.QMessageBox.information(self, "Not found", "No XYZ file found in the selected output directory.")
                return
            
            # Get lattice size and xyz interval from config or use defaults
            lattice_size = 256  # Default size
            xyz_interval = 1    # Default interval
            timesteps_override = None
            try:
                # Try to get parameters from used_params.json
                used_params = out / "used_params.json"
                if used_params.exists():
                    import json
                    with open(used_params, 'r') as f:
                        params = json.load(f)
                    # Handle both direct and wrapped parameter structures
                    if "parameters" in params and isinstance(params["parameters"], dict):
                        params = params["parameters"]
                    if "lattice_size_x" in params:
                        lattice_size = params["lattice_size_x"]
                    if "snapshot_schedule_mode" in params and str(params["snapshot_schedule_mode"]).lower() == "log":
                        # Derive log timesteps consistent with generate_log_points(1, num_iterations, log_snapshot_count)
                        try:
                            import numpy as _np
                            from CA3D_functions import generate_log_points as _gen
                            n_iter = int(params.get("num_iterations", 0))
                            log_n = int(params.get("log_snapshot_count", 100))
                            if n_iter > 0 and log_n > 0:
                                timesteps_override = [int(x) for x in _gen(1, n_iter, log_n).tolist()]
                        except Exception:
                            timesteps_override = None
                    else:
                        if "xyz_snapshot_interval_steps" in params:
                            xyz_interval = params["xyz_snapshot_interval_steps"]
            except Exception as e:
                print(f"Could not determine parameters from config: {e}")
                # Use default values
            
            # Show custom progress dialog
            progress_dialog = ClusterAnalysisProgressDialog(self)
            progress_dialog.show()
            QtWidgets.QApplication.processEvents()
            
            # Start logging to the dialog
            progress_dialog.start_logging()
            
            try:
                # Perform cluster analysis with progress updates
                def update_progress():
                    progress_dialog.update_log()
                    QtWidgets.QApplication.processEvents()
                
                results = analyze_xyz_file(
                    xyz_file_path=str(xyz_file),
                    output_dir=str(out),
                    lattice_size=lattice_size,
                    target_type=2,  # Crystalline atoms
                    min_cluster_size=1,
                    xyz_interval=xyz_interval,
                    timesteps_override=timesteps_override,
                    progress_callback=update_progress
                )
                
                # Final log update
                progress_dialog.update_log()
                
            finally:
                # Stop logging and close dialog
                progress_dialog.stop_logging()
                progress_dialog.close()
            
            if "error" in results:
                QtWidgets.QMessageBox.warning(self, "Analysis Error", f"Cluster analysis failed: {results['error']}")
                return
            
            # Show success message
            cluster_analysis_dir = out / "cluster_analysis"
            dat_file = out / "cluster_analysis.dat"
            
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setWindowTitle("Cluster Analysis Complete")
            msg.setText("Cluster analysis completed successfully!")
            msg.setDetailedText(f"Results saved to:\n{cluster_analysis_dir}\n\nMain data file: {dat_file}")
            msg.exec()
            
            # Update summary to show cluster analysis results
            if "snapshot_results" in results and results["snapshot_results"]:
                total_snapshots = len(results["snapshot_results"])
                avg_clusters = sum(r["number_of_clusters"] for r in results["snapshot_results"]) / total_snapshots
                avg_mean_size = sum(r["mean_cluster_size"] for r in results["snapshot_results"]) / total_snapshots
                
                cluster_summary = f"\n\n--- Cluster Analysis Results ---\n"
                cluster_summary += f"Total snapshots analyzed: {total_snapshots}\n"
                cluster_summary += f"Average number of clusters: {avg_clusters:.2f}\n"
                cluster_summary += f"Average mean cluster size: {avg_mean_size:.2f}\n"
                cluster_summary += f"Results saved to: {cluster_analysis_dir}\n"
                
                current_text = self.summary.toPlainText()
                self.summary.setPlainText(current_text + cluster_summary)
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Analysis Error", f"Cluster analysis failed: {str(e)}")
            print(f"Cluster analysis error: {e}")
            import traceback
            traceback.print_exc()

    def _on_plot_cluster_distribution(self) -> None:
        """Plot cluster distribution using the histogram data."""
        try:
            out = Path(self.out_dir.text().strip())
            if not out.exists():
                QtWidgets.QMessageBox.warning(self, "Invalid directory", "Selected output directory does not exist.")
                return
            
            # Check if cluster_analysis folder exists
            cluster_analysis_dir = out / "cluster_analysis"
            if not cluster_analysis_dir.exists():
                QtWidgets.QMessageBox.information(self, "Not found", "Cluster analysis folder not found. Please run cluster analysis first.")
                return
            
            # Look for the histogram data file
            histogram_file = cluster_analysis_dir / "_size_distribution_data.txt"
            if not histogram_file.exists():
                # Try to generate histogram data from existing JSON files
                print("Histogram data file not found, attempting to generate from existing JSON files...")
                if not self._generate_histogram_from_json(cluster_analysis_dir, histogram_file):
                    QtWidgets.QMessageBox.information(self, "Not found", 
                        "Histogram data file not found and could not be generated from existing JSON files.\n\n"
                        "Please ensure that:\n"
                        "1. Cluster analysis has been completed\n"
                        "2. comprehensive_cluster_analysis.json exists in cluster_analysis folder\n"
                        "3. The JSON file contains valid snapshot results")
                    return
            
            # Look for the alpha data file (system_evolution_stats.dat)
            alpha_file = out / "system_evolution_stats.dat"
            if not alpha_file.exists():
                QtWidgets.QMessageBox.information(self, "Not found", "System evolution stats file not found.")
                return
            
            # Quiet: no console/log output
            
            # Show progress dialog
            progress_dialog = QtWidgets.QProgressDialog("Generating cluster distribution plot...", "Cancel", 0, 0, self)
            progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
            progress_dialog.show()
            QtWidgets.QApplication.processEvents()
            
            try:
                # Read histogram data (quiet)
                timesteps, size_bins, hist_data = read_histogram_data(str(histogram_file))
                
                # Prepare interactive dialog and save path
                interactive_png = cluster_analysis_dir / "cluster_distribution_plot_interactive.png"
                
                progress_dialog.close()
                
                # Read alpha data for overlay (quiet, skip invalid header lines)
                def _read_alpha_quiet(file_path: str):
                    try:
                        import numpy as np
                        ts = []
                        av = []
                        with open(file_path, 'r') as f:
                            for line in f:
                                s = line.strip()
                                if not s or s.startswith('#'):
                                    continue
                                parts = s.split()
                                try:
                                    if len(parts) >= 4:
                                        t = float(parts[0])
                                        a = float(parts[3])
                                        ts.append(t)
                                        av.append(a)
                                except Exception:
                                    continue
                        return np.array(ts), np.array(av)
                    except Exception:
                        try:
                            import numpy as np
                            return np.array([]), np.array([])
                        except Exception:
                            return [], []

                alpha_times, alpha_values = _read_alpha_quiet(str(alpha_file))
                
                # Display interactive plot embedded in Qt
                dlg = InteractiveClusterDistributionDialog(self, "Cluster Distribution Analysis", str(interactive_png))
                dlg.plot_cluster_distribution(timesteps, size_bins, hist_data, alpha_times, alpha_values)
                dlg.exec()
                
            except Exception as e:
                progress_dialog.close()
                QtWidgets.QMessageBox.warning(self, "Plot Error", f"Failed to generate plot: {str(e)}")
                
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Plot Error", f"Plot generation failed: {str(e)}")

    def _generate_histogram_from_json(self, cluster_analysis_dir: Path, histogram_file: Path) -> bool:
        """
        Generate histogram data file from existing JSON files in cluster_analysis directory.
        
        Args:
            cluster_analysis_dir: Path to cluster_analysis directory
            histogram_file: Path where to save the histogram file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # List all files in cluster_analysis directory for debugging
            print(f"Files in cluster_analysis directory: {list(cluster_analysis_dir.glob('*'))}")
            
            # Look for comprehensive_cluster_analysis.json first
            comprehensive_json = cluster_analysis_dir / "comprehensive_cluster_analysis.json"
            if comprehensive_json.exists():
                print("Found comprehensive_cluster_analysis.json")
                # Load the comprehensive results
                with open(comprehensive_json, 'r') as f:
                    results = json.load(f)
                
                if "snapshot_results" not in results:
                    print("No snapshot_results found in comprehensive JSON")
                    return False
            else:
                print("comprehensive_cluster_analysis.json not found, trying individual snapshot files...")
                # Try to load from individual snapshot files
                snapshot_files = sorted(cluster_analysis_dir.glob("snapshot_*_results.json"))
                if not snapshot_files:
                    print("No snapshot files found")
                    return False
                
                results = {"snapshot_results": []}
                for snapshot_file in snapshot_files:
                    with open(snapshot_file, 'r') as f:
                        snapshot_data = json.load(f)
                        results["snapshot_results"].append(snapshot_data)
                
                print(f"Loaded {len(results['snapshot_results'])} snapshots from individual files")
            
            # Collect all unique cluster sizes across all snapshots
            all_sizes = set()
            for snapshot in results["snapshot_results"]:
                if "size_distribution" in snapshot:
                    all_sizes.update(snapshot["size_distribution"].keys())
            
            if not all_sizes:
                print("No size distribution data found in JSON files")
                return False
            
            # Sort sizes for consistent ordering
            size_bins = sorted([int(size) for size in all_sizes])
            
            # Create histogram data matrix
            timesteps = []
            hist_data = []
            
            for snapshot in results["snapshot_results"]:
                timestep = snapshot.get("timestep", 0)
                timesteps.append(timestep)
                
                # Create histogram row for this timestep
                row = []
                size_dist = snapshot.get("size_distribution", {})
                
                for size in size_bins:
                    count = size_dist.get(str(size), 0)
                    row.append(count)
                
                hist_data.append(row)
            
            # Write histogram data file
            with open(histogram_file, 'w') as f:
                # Write header
                header = "Timestep, " + ", ".join([f"Size_{size}" for size in size_bins])
                f.write(header + "\n")
                
                # Write data rows
                for i, timestep in enumerate(timesteps):
                    row_data = [str(timestep)] + [str(count) for count in hist_data[i]]
                    f.write(", ".join(row_data) + "\n")
            
            print(f"Generated histogram data from JSON: {len(timesteps)} timesteps, {len(size_bins)} size bins")
            return True
            
        except Exception as e:
            print(f"Error generating histogram from JSON: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _on_open_in_ovito(self) -> None:
        try:
            out = Path(self.out_dir.text().strip() or ".")
            xyz = self._find_xyz(out)
            if not xyz:
                QtWidgets.QMessageBox.information(self, "Not found", "No XYZ file found in the selected output directory.")
                return
            exe = self._detect_ovito()
            if not exe:
                # Ask user to locate OVITO executable
                QtWidgets.QMessageBox.information(self, "OVITO not detected", "OVITO was not found. Please select OVITO executable (ovito/ovito.exe).")
                filt = "Executable (*.exe);;All Files (*)" if sys.platform.startswith('win') else "All Files (*)"
                dlg = QtWidgets.QFileDialog(self, "Select OVITO executable")
                dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
                dlg.setNameFilter(filt)
                if dlg.exec():
                    files = dlg.selectedFiles()
                    if files:
                        cand = files[0]
                        if os.path.isfile(cand):
                            self._save_ovito_path(cand)
                            exe = cand
            if exe:
                subprocess.Popen([exe, str(xyz)])
                return
            # Fallback to system default opener
            if sys.platform.startswith('win'):
                os.startfile(str(xyz))  # type: ignore[attr-defined]
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(xyz)])
            else:
                subprocess.Popen(['xdg-open', str(xyz)])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Open failed", str(e))

    def _next_pen(self, width: int = 1):
        try:
            color = self._palette[self._color_index % len(self._palette)]
            style = self._styles[(self._color_index // len(self._palette)) % len(self._styles)]
            pen = pg.mkPen(color, width=max(2, width), style=style)
        except Exception:
            pen = pg.mkPen(width=max(2, width))
        self._color_index += 1
        return pen

    def _sync_plots_to_selection(self) -> None:
        # Incremental sync to preserve colors/styles
        desired = {it.text() for it in self.series_list.selectedItems()}
        # Add new selections
        for name in desired:
            if name in self._plotted:
                continue
            s = self.series_map.get(name)
            if s is None:
                continue
            try:
                xd = getattr(s, 'x', [])
                yd = getattr(s, 'y', [])
                xf, yf = self._filter_for_log_axes(xd, yd)
                if len(xf) == 0:
                    continue
                pen = self._pen_for_series.get(name)
                if pen is None:
                    pen = self._next_pen(width=2)
                    self._pen_for_series[name] = pen
                item = self.plot.plot(xf, yf, pen=pen, name=name)
                self._plotted[name] = item
            except Exception:
                continue
        # Remove deselected
        to_remove = [n for n in list(self._plotted.keys()) if n not in desired]
        for n in to_remove:
            try:
                item = self._plotted.pop(n)
                self.plot.removeItem(item)
            except Exception:
                pass

    def _on_series_selection_changed(self) -> None:
        try:
            self._sync_plots_to_selection()
            # Use internal autorange for stability on both linear/log
            try:
                self.plot.enableAutoRange(x=True, y=True)
                self.plot.plotItem.vb.autoRange()
            except Exception:
                self._autoscale_plot()
        except Exception:
            pass

    def _filter_for_log_axes(self, x, y):
        """Filter out non-finite and non-positive (when log) data pairs."""
        try:
            import numpy as np
            x = np.asarray(x)
            y = np.asarray(y)
            m = np.isfinite(x) & np.isfinite(y)
            if self.chk_log_x.isChecked():
                m &= (x > 0)
            if self.chk_log_y.isChecked():
                m &= (y > 0)
            xf = x[m]
            yf = y[m]
            if (len(xf) == 0 or len(yf) == 0) and (self.chk_log_x.isChecked() or self.chk_log_y.isChecked()):
                # Fallback: replace non-positive values with small epsilon to allow plotting
                eps = 1e-9
                x2 = np.where(np.isfinite(x), x, np.nan)
                y2 = np.where(np.isfinite(y), y, np.nan)
                if self.chk_log_x.isChecked():
                    x2 = np.where(x2 > 0, x2, eps)
                if self.chk_log_y.isChecked():
                    y2 = np.where(y2 > 0, y2, eps)
                m2 = np.isfinite(x2) & np.isfinite(y2)
                xf = x2[m2]
                yf = y2[m2]
            return xf, yf
        except Exception:
            return x, y

    def _on_series_item_pressed(self, item) -> None:
        try:
            # Toggle selection state of this item only
            item.setSelected(not item.isSelected())
            self._sync_plots_to_selection()
            self._autoscale_plot()
        except Exception:
            pass

    def _render_params_html(self, data, folder: Optional[str] = None) -> str:
        """Build a compact, readable HTML table for used_params.json.

        Args:
            data: Parsed JSON object (dict/list/primitive).
            folder: Optional folder path to display in header.

        Returns:
            HTML string suitable for QTextEdit.setHtml.
        """
        def render_value(val) -> str:
            if isinstance(val, dict):
                return render_table(val)
            if isinstance(val, list):
                key_td_style = "padding:6px 10px;border:1px solid #ddd;background:#f7f7f9;white-space:nowrap;width:50%;font-weight:600"
                val_td_style = "padding:6px 10px;border:1px solid #ddd;width:50%;white-space:nowrap;word-break:normal;overflow-wrap:normal"
                rows = []
                for i, v in enumerate(val):
                    val_html = f"<div style='white-space:nowrap;overflow-x:auto'>{render_value(v)}</div>"
                    rows.append(f"<tr><td width='50%' style='{key_td_style}'>[{i}]</td><td width='50%' style='{val_td_style}'>{val_html}</td></tr>")
                return f"<table width='100%' style='border-collapse:collapse;margin:4px 0 8px 0;width:100%'>{''.join(rows)}</table>"
            if isinstance(val, bool):
                return "<span style='color:#005a9e;font-weight:600'>True</span>" if val else "<span style='color:#8a2d2d;font-weight:600'>False</span>"
            if val is None:
                return "<span style='color:#666'>null</span>"
            if isinstance(val, (int, float)):
                return f"<span style='color:#2b6a30'>{escape(str(val))}</span>"
            return escape(str(val))

        def render_table(d: Dict) -> str:
            rows = []
            key_td_style = "padding:6px 10px;border:1px solid #ddd;background:#f7f7f9;white-space:nowrap;width:50%;font-weight:600"
            val_td_style = "padding:6px 10px;border:1px solid #ddd;width:50%;white-space:nowrap;word-break:normal;overflow-wrap:normal"
            for k, v in d.items():
                key_html = f"<td width='50%' style='{key_td_style}'>{escape(str(k))}</td>"
                val_inner = f"<div style='white-space:nowrap;overflow-x:auto'>{render_value(v)}</div>"
                val_html = f"<td width='50%' style='{val_td_style}'>{val_inner}</td>"
                rows.append(f"<tr>{key_html}{val_html}</tr>")
            return "<table width='100%' style='border-collapse:collapse;width:100%;margin-top:6px;table-layout:fixed'>" + ''.join(rows) + "</table>"

        title = "Simulation parameters"
        head = f"<div style='font-size:14px;font-weight:700;margin-bottom:4px'>{title}</div>"

        # Top-level special handling: split into left Parameters and right sections
        if isinstance(data, dict):
            special_order = ["ruleset", "recipe_values", "init"]
            # Flatten an optional top-level 'parameters' dict into the left panel
            left_map: Dict = {}
            params = data.get("parameters") if isinstance(data.get("parameters"), dict) else None
            if params:
                for k, v in params.items():
                    left_map[k] = v
            for k, v in data.items():
                if k in special_order or k == "parameters":
                    continue
                left_map[k] = v
            right_sections = []

            def render_section(section_title: str, mapping) -> str:
                # Ensure we have a dict to tabulate; fallback to single value row
                if isinstance(mapping, dict):
                    rows_html = []
                    key_td_style = "padding:6px 10px;border:1px solid #ddd;background:#f7f7f9;white-space:nowrap;width:50%;font-weight:600"
                    val_td_style = "padding:6px 10px;border:1px solid #ddd;width:50%;white-space:nowrap;word-break:normal;overflow-wrap:normal"
                    for k2, v2 in mapping.items():
                        key_html = f"<td width='50%' style='{key_td_style}'>{escape(str(k2))}</td>"
                        val_inner = f"<div style='white-space:nowrap;overflow-x:auto'>{render_value(v2)}</div>"
                        val_html = f"<td width='50%' style='{val_td_style}'>{val_inner}</td>"
                        rows_html.append(f"<tr>{key_html}{val_html}</tr>")
                    rows = ''.join(rows_html) if rows_html else "<tr><td colspan='2' style='padding:6px 10px;border:1px solid #ddd;color:#666'>No entries</td></tr>"
                else:
                    key_td_style = "padding:6px 10px;border:1px solid #ddd;background:#f7f7f9;white-space:nowrap;width:50%;font-weight:600"
                    val_td_style = "padding:6px 10px;border:1px solid #ddd;width:50%;white-space:nowrap;word-break:normal;overflow-wrap:normal"
                    val_inner = f"<div style='white-space:nowrap;overflow-x:auto'>{render_value(mapping)}</div>"
                    rows = f"<tr><td width='50%' style='{key_td_style}'>value</td><td width='50%' style='{val_td_style}'>{val_inner}</td></tr>"
                header = f"<tr><th colspan='2' style='text-align:left;padding:8px 10px;background:#ececf1;border:1px solid #ddd;font-weight:700'>{escape(section_title)}</th></tr>"
                return f"<table width='100%' style='border-collapse:collapse;width:100%;margin:0 0 10px 0;table-layout:fixed'>{header}{rows}</table>"

            # Left panel: show only the rows (no per-row left label cell titled 'parameters')
            left_html = ""
            if left_map:
                # Header only once for left
                header = "<tr><th colspan='2' style='text-align:left;padding:8px 10px;background:#ececf1;border:1px solid #ddd;font-weight:700'>Parameters</th></tr>"
                rows = []
                key_td_style = "padding:6px 10px;border:1px solid #ddd;background:#f7f7f9;white-space:nowrap;width:50%;font-weight:600"
                val_td_style = "padding:6px 10px;border:1px solid #ddd;width:50%;white-space:nowrap;word-break:normal;overflow-wrap:normal"
                for k, v in left_map.items():
                    key_html = f"<td width='50%' style='{key_td_style}'>{escape(str(k))}</td>"
                    val_inner = f"<div style='white-space:nowrap;overflow-x:auto'>{render_value(v)}</div>"
                    val_html = f"<td width='50%' style='{val_td_style}'>{val_inner}</td>"
                    rows.append(f"<tr>{key_html}{val_html}</tr>")
                left_html = f"<table width='100%' style='border-collapse:collapse;width:100%;margin:0 0 10px 0;table-layout:fixed'>{header}{''.join(rows)}</table>"

            # Display names for right-side sections
            display_names = {
                "ruleset": "Rule recipe name",
                "recipe_values": "Used values for recipe",
                "init": "Initilization mode",
            }
            for key in special_order:
                if key in data:
                    right_sections.append(render_section(display_names.get(key, key), data.get(key)))
            
            right_html = ''.join(right_sections) if right_sections else ""

            # Two-column layout
            layout = (
                "<table width='100%' cellpadding='0' cellspacing='0' style='width:100%;border-collapse:collapse;table-layout:fixed'><tr>"
                f"<td width='50%' style='vertical-align:top;width:50%;padding-right:12px'>{left_html}</td>"
                f"<td width='50%' style='vertical-align:top;width:50%;padding-left:12px'><div style='display:block;width:100%'>{right_html}</div></td>"
                "</tr></table>"
            )
            return head + layout

        # Fallback for non-dict top-level
        body = render_value(data)
        return head + body


