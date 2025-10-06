from __future__ import annotations

from typing import Optional, Dict
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtGui import QVector3D
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from .ImpostorSpheres import ImpostorSpheres


class GLPreview(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)
        self.setMinimumHeight(420)
        self.scatter: Optional[gl.GLScatterPlotItem] = None
        self.scatter1: Optional[gl.GLScatterPlotItem] = None
        self.mesh_items: list[gl.GLMeshItem] = []
        self.edge_items: list[gl.GLLinePlotItem] = []
        # Sphere MeshData caches and reusable pool (for LOD and batching)
        self._sphere_md_low = None
        self._sphere_md_med = None
        self._sphere_md_high = None
        try:
            self._sphere_md_low = gl.MeshData.sphere(rows=6, cols=12, radius=0.5)
            self._sphere_md_med = gl.MeshData.sphere(rows=8, cols=16, radius=0.5)
            self._sphere_md_high = gl.MeshData.sphere(rows=12, cols=24, radius=0.5)
        except Exception:
            pass
        self._sphere_pool: list[gl.GLMeshItem] = []
        # Remember last walls to redraw borders if not passed
        self._last_walls: Optional[Dict[str, bool]] = None
        # GPU impostor renderer for crystalline atoms
        self._impostor_crys: Optional[ImpostorSpheres] = None

        # Connect to camera/view changes to keep impostor size in sync with zoom
        try:
            self.view.sigCameraPositionChanged.connect(self._on_camera_changed)  # type: ignore[attr-defined]
        except Exception:
            pass
        # No grid background
        try:
            self.view.setBackgroundColor('k')
        except Exception:
            pass
        self.status = QtWidgets.QLabel("")

    def show_atoms(self, atoms: np.ndarray, include_mobile: bool = False, max_points: int = 2_000_000, walls: Optional[Dict[str, bool]] = None, max_mesh_spheres: int = 5000) -> None:
        # Render state==2 (crystalline) always, optionally state==1 (mobile)
        idx1 = np.column_stack(np.nonzero(atoms == 1)) if include_mobile else np.zeros((0, 3), dtype=int)
        idx2 = np.column_stack(np.nonzero(atoms == 2))
        if len(idx1) > max_points // 2:
            choice = np.random.choice(len(idx1), size=max_points // 2, replace=False)
            idx1 = idx1[choice]
        if len(idx2) > max_points // 2:
            choice = np.random.choice(len(idx2), size=max_points // 2, replace=False)
            idx2 = idx2[choice]
        pts1 = idx1[:, [0, 1, 2]].astype(float) if len(idx1) else np.zeros((0, 3), dtype=float)
        pts2 = idx2[:, [0, 1, 2]].astype(float) if len(idx2) else np.zeros((0, 3), dtype=float)

        # Clear previous
        if self.scatter:
            try:
                self.view.removeItem(self.scatter)
            except Exception:
                pass
            self.scatter = None
        if self.scatter1:
            try:
                self.view.removeItem(self.scatter1)
            except Exception:
                pass
            self.scatter1 = None
        # Reuse pooled spheres; hide them for now
        for it in getattr(self, "_sphere_pool", []):
            try:
                it.setVisible(False)
            except Exception:
                pass
        
        for it in self.mesh_items:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.mesh_items.clear()
        for it in self.edge_items:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.edge_items.clear()

        count = 0
        # Show ALL crystalline atoms; LOD will switch to scatter automatically if too many
        pts2_all = pts2

        # Always render crystalline atoms on GPU via impostors
        if len(pts2_all):
            if self._impostor_crys is None:
                self._impostor_crys = ImpostorSpheres()
                self.view.addItem(self._impostor_crys)
            self._impostor_crys.setVisible(True)
            # Slightly larger than 0.5 so spheres almost touch visually
            self._impostor_crys.setData(pts2_all, color=(139/255.0, 198/255.0, 80/255.0, 1.0), size_px=0.0)
            self._impostor_crys.setWorldRadius(0.8)
            try:
                # approximate FOV and viewport height from widget size
                vp_h = max(1, self.view.height())
                # pyqtgraph default FOV ~60 deg
                self._impostor_crys.setViewParams(fov_deg=60.0, viewport_h=vp_h)
            except Exception:
                pass
            count += len(pts2_all)
        else:
            if self._impostor_crys is not None:
                try:
                    self._impostor_crys.setVisible(False)
                except Exception:
                    pass
        if len(pts1):
            # Always render mobile atoms as scatter points, size 5, with depth-tested opaque rendering
            self.scatter1 = gl.GLScatterPlotItem(
                pos=pts1,
                color=(254/255.0, 204/255.0, 92/255.0, 1.0),  # #fecc5c
                size=1.0,
                pxMode=True,
                glOptions='opaque',
            )
            self.view.addItem(self.scatter1)
            count += len(pts1)

        # Auto-center to include all placeholders (domain extents)
        if len(pts1) or len(pts2):
            allpts = pts2 if not len(pts1) else (pts1 if not len(pts2) else np.vstack([pts1, pts2]))
            center = allpts.mean(axis=0)
            span = np.maximum(allpts.max(axis=0) - allpts.min(axis=0), 1.0)
            dist = float(np.max(span) * 2.0)
            self.view.opts['center'] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
            self.view.setCameraPosition(distance=dist)
        else:
            self.view.setCameraPosition(distance=10.0)

        self.status.setText(f"Points shown: {count}")
        self.view.update()

        # Draw simulation cell edges with wall transparency coloring; remember last walls
        try:
            if walls is not None:
                self._last_walls = walls
            use_walls = walls if (walls is not None) else self._last_walls
            if use_walls is not None:
                sx, sy, sz = atoms.shape
                self._draw_cell_edges(sx, sy, sz, use_walls)
        except Exception:
            pass

    def _on_camera_changed(self) -> None:
        # Update impostor shader params based on current viewport
        try:
            if self._impostor_crys is not None and self._impostor_crys.isVisible():
                vp_h = max(1, self.view.height())
                self._impostor_crys.setViewParams(fov_deg=60.0, viewport_h=vp_h)
                self.view.update()
        except Exception:
            pass

    def _draw_cell_edges(self, sx: int, sy: int, sz: int, walls: Dict[str, bool]) -> None:
        # Compute box corners in index space expanded by 0.5 so radius 0.5 spheres touch faces
        x0, x1 = -0.5, sx - 0.5
        y0, y1 = -0.5, sy - 0.5
        z0, z1 = -0.5, sz - 0.5

        # Edge list: each entry is (points along edge), (associated faces)
        edges = [
            # bottom rectangle (z=z0)
            ([(x0, y0, z0), (x1, y0, z0)], ("wall_y_min", "wall_z_min")),
            ([(x1, y0, z0), (x1, y1, z0)], ("wall_x_max", "wall_z_min")),
            ([(x1, y1, z0), (x0, y1, z0)], ("wall_y_max", "wall_z_min")),
            ([(x0, y1, z0), (x0, y0, z0)], ("wall_x_min", "wall_z_min")),
            # top rectangle (z=z1)
            ([(x0, y0, z1), (x1, y0, z1)], ("wall_y_min", "wall_z_max")),
            ([(x1, y0, z1), (x1, y1, z1)], ("wall_x_max", "wall_z_max")),
            ([(x1, y1, z1), (x0, y1, z1)], ("wall_y_max", "wall_z_max")),
            ([(x0, y1, z1), (x0, y0, z1)], ("wall_x_min", "wall_z_max")),
            # vertical edges
            ([(x0, y0, z0), (x0, y0, z1)], ("wall_x_min", "wall_y_min")),
            ([(x1, y0, z0), (x1, y0, z1)], ("wall_x_max", "wall_y_min")),
            ([(x1, y1, z0), (x1, y1, z1)], ("wall_x_max", "wall_y_max")),
            ([(x0, y1, z0), (x0, y1, z1)], ("wall_x_min", "wall_y_max")),
        ]

        # Colors: austere green and red
        col_green = (0.15, 0.55, 0.25, 0.9)
        col_red = (0.65, 0.25, 0.25, 0.9)

        for pts, (face_a, face_b) in edges:
            a = bool(walls.get(face_a, False))  # True means non-transparent
            b = bool(walls.get(face_b, False))
            # Determine coloring: both transparent -> green; both non-transparent -> red; mixed -> striped
            if not a and not b:
                # all transparent
                pos = np.array(pts, dtype=float)
                item = gl.GLLinePlotItem(pos=pos, color=col_green, width=2, antialias=True, glOptions='opaque')
                self.view.addItem(item)
                self.edge_items.append(item)
            elif a and b:
                pos = np.array(pts, dtype=float)
                item = gl.GLLinePlotItem(pos=pos, color=col_red, width=2, antialias=True, glOptions='opaque')
                self.view.addItem(item)
                self.edge_items.append(item)
            else:
                # striped: split edge into segments alternating red/green
                p0 = np.array(pts[0], dtype=float)
                p1 = np.array(pts[1], dtype=float)
                segs = 20
                for i in range(segs):
                    t0 = i / segs
                    t1 = (i + 1) / segs
                    q0 = p0 * (1 - t0) + p1 * t0
                    q1 = p0 * (1 - t1) + p1 * t1
                    color = col_red if i % 2 == 0 else col_green
                    pos = np.vstack([q0, q1])
                    item = gl.GLLinePlotItem(pos=pos, color=color, width=2, antialias=True, glOptions='opaque')
                    self.view.addItem(item)
                    self.edge_items.append(item)
        # Adjust camera to see entire cell
        cx = (sx - 1) / 2.0
        cy = (sy - 1) / 2.0
        cz = (sz - 1) / 2.0
        span = float(max(sx, sy, sz))
        self.view.opts['center'] = QVector3D(cx, cy, cz)
        self.view.setCameraPosition(distance=span * 2.0)


