from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from OpenGL import GL
from pyqtgraph.opengl import GLGraphicsItem


class ImpostorSpheres(GLGraphicsItem.GLGraphicsItem):
    """
    GPU impostor spheres using point sprites in a single draw call.
    - Positions uploaded to a VBO; rendered as GL_POINTS
    - Fragment shader discards outside circle and shades a sphere
    - Opaque rendering to avoid white blending at high densities
    """

    def __init__(self) -> None:
        super().__init__()
        self._pos: Optional[np.ndarray] = None
        self._color: Tuple[float, float, float, float] = (0.5, 0.8, 0.5, 1.0)
        self._size_px: float = 8.0
        self._prog: Optional[int] = None
        self._vbo: Optional[int] = None
        self._u_color_loc: Optional[int] = None
        self._u_world_radius_loc: Optional[int] = None
        self._u_vp_h_loc: Optional[int] = None
        self._u_tan_half_fov_loc: Optional[int] = None
        self._world_radius: float = 0.5
        self._vp_h: float = 800.0
        self._tan_half_fov: float = 1.0

    def setData(self, pos: np.ndarray, color: Tuple[float, float, float, float] = (0.5, 0.8, 0.5, 1.0), size_px: float = 8.0) -> None:
        self._pos = np.ascontiguousarray(pos.astype(np.float32)) if pos is not None and len(pos) else None
        self._color = color
        self._size_px = float(size_px)
        self.update()

    def setWorldRadius(self, radius: float) -> None:
        self._world_radius = float(radius)
        self.update()

    def setViewParams(self, fov_deg: float, viewport_h: int) -> None:
        # gl_PointSize = (radius * vp_h) / (zEye * tan(fov/2))
        import math
        self._vp_h = float(max(1, viewport_h))
        self._tan_half_fov = float(math.tan(math.radians(max(1e-3, fov_deg)) * 0.5))
        self.update()

    def initializeGL(self) -> None:
        # Compile simple compatibility-profile shaders (GLSL 120)
        vs_src = """
        #version 120
        attribute vec3 a_pos;
        uniform float u_world_radius;
        uniform float u_viewport_h;
        uniform float u_tan_half_fov;
        void main() {
            vec4 eye = gl_ModelViewMatrix * vec4(a_pos, 1.0);
            float zEye = -eye.z;
            gl_Position = gl_ModelViewProjectionMatrix * vec4(a_pos, 1.0);
            gl_PointSize = (u_world_radius * u_viewport_h) / max(1e-3, zEye * u_tan_half_fov);
        }
        """
        fs_src = """
        #version 120
        uniform vec4 u_color;
        void main() {
            // gl_PointCoord in [0,1]; make circle
            vec2 p = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(p, p);
            if (r2 > 1.0) discard;
            // simple lambert-like shading using pseudo normal
            vec3 n = normalize(vec3(p, sqrt(1.0 - r2)));
            // flipped light to the opposite direction
            vec3 L = normalize(vec3(-0.5, -0.4, -0.7));
            float diff = max(dot(n, L), 0.0);
            // add simple Phong specular using view dir ~ +Z in eye space
            vec3 R = reflect(-L, n);
            vec3 V = vec3(0.0, 0.0, 1.0);
            float spec = pow(max(dot(R, V), 0.0), 32.0);
            // brighter: higher ambient, boosted diffuse, visible specular
            float ambient = 0.55;
            float intensity = clamp(ambient + 1.25 * diff + 0.6 * spec, 0.0, 1.0);
            gl_FragColor = vec4(u_color.rgb * intensity, u_color.a);
        }
        """
        self._prog = self._link_program(vs_src, fs_src)
        self._u_color_loc = GL.glGetUniformLocation(self._prog, b"u_color")
        self._u_world_radius_loc = GL.glGetUniformLocation(self._prog, b"u_world_radius")
        self._u_vp_h_loc = GL.glGetUniformLocation(self._prog, b"u_viewport_h")
        self._u_tan_half_fov_loc = GL.glGetUniformLocation(self._prog, b"u_tan_half_fov")
        # Create VBO
        self._vbo = GL.glGenBuffers(1)

    def _link_program(self, vs_src: str, fs_src: str) -> int:
        def compile_shader(src: str, stype: int) -> int:
            sh = GL.glCreateShader(stype)
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            ok = GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS)
            if not ok:
                log = GL.glGetShaderInfoLog(sh)
                raise RuntimeError(f"Shader compile error: {log}")
            return sh
        vs = compile_shader(vs_src, GL.GL_VERTEX_SHADER)
        fs = compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glBindAttribLocation(prog, 0, b"a_pos")
        GL.glLinkProgram(prog)
        ok = GL.glGetProgramiv(prog, GL.GL_LINK_STATUS)
        if not ok:
            log = GL.glGetProgramInfoLog(prog)
            raise RuntimeError(f"Program link error: {log}")
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        return prog

    def paint(self) -> None:
        if self._pos is None or len(self._pos) == 0:
            return
        if self._prog is None:
            self.initializeGL()

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        
        try:
            GL.glEnable(GL.GL_POINT_SPRITE)
            GL.glEnable(GL.GL_VERTEX_PROGRAM_POINT_SIZE)
        except Exception:
            pass

        GL.glUseProgram(self._prog)
        # uniforms
        if self._u_color_loc is not None:
            GL.glUniform4f(self._u_color_loc, *self._color)
        if self._u_world_radius_loc is not None:
            GL.glUniform1f(self._u_world_radius_loc, float(self._world_radius))
        if self._u_vp_h_loc is not None:
            GL.glUniform1f(self._u_vp_h_loc, float(self._vp_h))
        if self._u_tan_half_fov_loc is not None:
            GL.glUniform1f(self._u_tan_half_fov_loc, float(self._tan_half_fov))

        # upload positions
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._pos.nbytes, self._pos, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        # draw
        GL.glDrawArrays(GL.GL_POINTS, 0, int(self._pos.shape[0]))

        # cleanup
        GL.glDisableVertexAttribArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glUseProgram(0)


