from __future__ import annotations

from typing import Optional, List
from PySide6 import QtWidgets
from PySide6.QtGui import QVector3D
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class NeighborhoodPreview(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        # Controls row
        ctrl = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("Reset view")
        ctrl.addWidget(self.btn_reset)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)

        self.view = gl.GLViewWidget()
        self.view.setMinimumHeight(260)
        layout.addWidget(self.view)
        try:
            self.view.setBackgroundColor('k')
        except Exception:
            pass
        # Small grid from -1 to 1 in XY plane at Z=0
        self.grid = gl.GLGridItem()
        try:
            self.grid.setSize(2, 2, 1)
            self.grid.setSpacing(1, 1, 1)
        except Exception:
            pass
        self.view.addItem(self.grid)
        self.axis_items = []
        try:
            import numpy as _np
            # X axis (red)
            x_pos = _np.array([[0, 0, 0], [2.0, 0, 0]], dtype=float)
            x_item = gl.GLLinePlotItem(pos=x_pos, color=(1.0, 0.1, 0.1, 1.0), width=4, antialias=True)
            self.view.addItem(x_item); self.axis_items.append(x_item)
            # Y axis (green)
            y_pos = _np.array([[0, 0, 0], [0, 2.0, 0]], dtype=float)
            y_item = gl.GLLinePlotItem(pos=y_pos, color=(0.1, 0.9, 0.1, 1.0), width=4, antialias=True)
            self.view.addItem(y_item); self.axis_items.append(y_item)
            # Z axis (blue)
            z_pos = _np.array([[0, 0, 0], [0, 0, 2.0]], dtype=float)
            z_item = gl.GLLinePlotItem(pos=z_pos, color=(0.2, 0.5, 1.0, 1.0), width=4, antialias=True)
            self.view.addItem(z_item); self.axis_items.append(z_item)
            # Axis labels (3D text)
            self.axis_text = []
            try:
                GLTextItem = self._resolve_gl_text()
                if GLTextItem is not None:
                    xt = GLTextItem('X', color=(1.0, 0.1, 0.1, 1.0)); xt.translate(2.2, 0.0, 0.0)
                    yt = GLTextItem('Y', color=(0.1, 0.9, 0.1, 1.0)); yt.translate(0.0, 2.2, 0.0)
                    zt = GLTextItem('Z', color=(0.2, 0.5, 1.0, 1.0)); zt.translate(0.0, 0.0, 2.2)
                    for t in (xt, yt, zt):
                        self.view.addItem(t); self.axis_text.append(t)
            except Exception:
                pass
        except Exception:
            pass
        self.scatter0: Optional[gl.GLScatterPlotItem] = None
        self.mesh_items: list[gl.GLMeshItem] = []
        self.legend = QtWidgets.QLabel("0: empty  1: mobile (yellow)  2: crystalline (green)    Dirs: Self, +X Right, +Y Down, -X Left, -Y Up, -Z Back, +Z Front")
        layout.addWidget(self.legend)

        # Wire controls
        self.btn_reset.clicked.connect(self._on_reset)

        # Predefined neighbor positions (self, right, down, left, up, back, front)
        self.positions = np.array([
            [0, 0, 0],   # self
            [1, 0, 0],   # right (+x)
            [0, 1, 0],   # down  (+y)
            [-1, 0, 0],  # left  (-x)
            [0, -1, 0],  # up    (-y)
            [0, 0, -1],  # back  (-z)
            [0, 0, 1],   # front (+z)
        ], dtype=float)
        # Align initial camera to the same framing used after rule selection
        self._frame_view()

    def set_vector(self, vec: List[int]) -> None:
        if len(vec) != 7:
            return
        # Clear previous items
        if self.scatter0:
            self.view.removeItem(self.scatter0)
            self.scatter0 = None
        for it in self.mesh_items:
            self.view.removeItem(it)
        self.mesh_items.clear()

        # Empty neighbors: small gray dots
        pts0 = [pos for pos, v in zip(self.positions, vec) if v == 0]
        if pts0:
            pts0 = np.array(pts0, dtype=float)
            self.scatter0 = gl.GLScatterPlotItem(pos=pts0, color=(0.6, 0.6, 0.6, 0.8), size=6.0)
            self.view.addItem(self.scatter0)

        # Occupied neighbors: world-space spheres that touch (radius ~0.5 when spacing==1)
        try:
            md = gl.MeshData.sphere(rows=20, cols=40, radius=0.5)
        except Exception:
            md = None
        # Clear existing text labels
        if not hasattr(self, 'text_items'):
            self.text_items = []
        for t in getattr(self, 'text_items', []):
            try:
                self.view.removeItem(t)
            except Exception:
                pass
        self.text_items = []
        labels = [
            "Self", "Right", "Down", "Left", "Up", "Back", "Front"
        ]
        for pos, v in zip(self.positions, vec):
            if v in (1, 2) and md is not None:
                color = (254/255.0, 204/255.0, 92/255.0, 1.0) if v == 1 else (139/255.0, 198/255.0, 80/255.0, 1.0)
                item = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader='shaded')
                item.translate(float(pos[0]), float(pos[1]), float(pos[2]))
                self.view.addItem(item)
                self.mesh_items.append(item)
        # Text labels disabled per request
        # Center camera to show full spheres without cutting
        self._frame_view()

    def _frame_view(self) -> None:
        try:
            # Use a bounding box that accounts for sphere radius 0.5
            pts = self.positions
            mins = pts.min(axis=0) - 0.6
            maxs = pts.max(axis=0) + 0.6
            center = (mins + maxs) * 0.5
            span = np.maximum(maxs - mins, 1.0)
            # Reduce camera distance so objects appear larger in the previews
            dist = float(np.max(span) * 3.5)
            self.view.opts['center'] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
            self.view.opts['fov'] = 40
            self.view.setCameraPosition(distance=dist, elevation=20, azimuth=30)
            # No text overlays to update
        except Exception:
            pass

    def _on_reset(self) -> None:
        self._frame_view()

    def reset_view(self) -> None:
        """Public method to reset the camera/view framing."""
        self._frame_view()

    

    # Text rendering disabled


    def showEvent(self, ev) -> None:  # type: ignore[override]
        super().showEvent(ev)
        try:
            # Ensure initial camera matches selection framing upon first show
            self._frame_view()
        except Exception:
            pass

