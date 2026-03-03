"""
gabor_renderer.py
=================
Self-contained OpenGL rendering module for the Quest 2AFC experiment.
Handles window creation, shader compilation, Gabor stimulus animation,
text overlay, and response collection.

Standalone import example::

    import glfw
    from gabor_renderer import ExperimentRenderer

    glfw.init()
    renderer = ExperimentRenderer(diagonal_inch=27, visual_radius_deg=2.0)
    renderer.init_window()
    renderer.set_condition("ach", mean_luminance=50.0)
    renderer.show_gabor(contrast=1.0, distance_m=1.0,
                        spatial_frequency_cpp=0.1, speed=240, duration=2.0)
    renderer.destroy()
    glfw.terminate()
"""

import json
import time
import numpy as np
import freetype as _ft

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from scipy.interpolate import PchipInterpolator


# ==========================================================
# Physical / display helpers  (re-exported for convenience)
# ==========================================================

def compute_display_width(diagonal_inch: float, R_x: int, R_y: int) -> float:
    D_m = diagonal_inch * 0.0254
    return D_m * (R_x / np.sqrt(R_x**2 + R_y**2))


def compute_spatiotemporal_frequency(R_x, W, d, f_p, v_p):
    """Return (rho_cpd, omega_hz, theta_rad)."""
    theta   = 2 * np.arctan(W / (2 * d))
    rho_rad = (R_x * f_p) / theta
    rho_cpd = rho_rad * (np.pi / 180)
    omega   = f_p * v_p
    return rho_cpd, omega, theta


def visual_radius_deg_to_px(visual_radius_deg, d, W, R_x):
    phi_rad = np.deg2rad(visual_radius_deg)
    R_phys  = d * np.tan(phi_rad)
    return (R_phys / W) * R_x


# ==========================================================
# Luminance LUT
# ==========================================================

def load_pchip_model(json_path="calibrate_display/Measure_specbos/pixel_luminance_model_pchip_B100_C100_s800.json"):
    with open(json_path) as f:
        model = json.load(f)
    pchip = PchipInterpolator(model["luminance_samples"], model["pixel_samples"])
    lum   = np.array(model["luminance_samples"])
    return pchip, float(lum.min()), float(lum.max())


def build_lut(pchip_model, L_min, L_max, n_samples=1024):
    L_vals = np.linspace(L_min, L_max, n_samples)
    return L_vals, np.clip(pchip_model(L_vals), 0.0, 1.0)


# ==========================================================
# DKL colour matrices
# ==========================================================

def get_color_matrices(mean_luminance: float, color_direction: str):
    """Return (M_combined, dkl_bg, col_dir). color_direction: 'ach'|'rg'|'yv'."""
    lms_gray = np.array([0.739876529525622, 0.320136241543338, 0.020793708751515])
    mc1 = lms_gray[0] / lms_gray[1]
    mc2 = (lms_gray[0] + lms_gray[1]) / lms_gray[2]

    M_lms_dkl = np.array([[1,  1,    0],
                           [1, -mc1,  0],
                           [-1, -1,  mc2]])
    M_lms_xyz = np.array([
        [ 2.629129278399650, -3.780202391780134, 10.294956387893450],
        [ 0.865649062438827,  1.215555811642301, -0.984175688105352],
        [-0.008886561474676,  0.081612628990755,  51.371024830897888]
    ])
    # M_xyz_rgb = np.array([
    #     [ 3.2406, -1.5372, -0.4986],
    #     [-0.9689,  1.8758,  0.0415],
    #     [ 0.0557, -0.2040,  1.0570]
    # ])
    M_xyz_rgb = np.array([
        [3.5323692968899123, -1.030316300209483, 0.042765311058415995],
        [-1.7514489461570455, 1.9868426359424745, -0.15752633607896482],
        [-0.5428350001133714, -0.003937472019811095, 1.0596334228448179]
    ]).T # 拟合反了，别忘了转置
    M_combined = M_xyz_rgb @ M_lms_xyz @ np.linalg.inv(M_lms_dkl)

    white_point_d65 = np.array([0.9505, 1.0000, 1.0888])
    M_xyz_lms = np.array([
        [ 0.187596268556126,  0.585168649077728, -0.026384263306304],
        [-0.133397430663221,  0.405505777260049,  0.034502127690364],
        [ 0.000244379021663, -0.000542995890619,  0.019406849066323]
    ])
    dkl_bg  = mean_luminance * (white_point_d65 @ M_xyz_lms.T) @ M_lms_dkl.T

    col_map = {"ach": [1,0,0], "rg": [0,1,0], "yv": [0,0,1]}
    if color_direction not in col_map:
        raise ValueError("color_direction must be 'ach', 'rg', or 'yv'")
    return M_combined, dkl_bg, np.array(col_map[color_direction], float)


# ==========================================================
# FreeType text rendering
# ==========================================================

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
_ft_face   = None


def _get_face(size_px=48):
    global _ft_face
    if _ft_face is None:
        _ft_face = _ft.Face(_FONT_PATH)
    _ft_face.set_pixel_sizes(0, size_px)
    return _ft_face


def render_text_to_pixels(text: str, size_px: int = 36):
    face, PAD = _get_face(size_px), 4
    pen_x, glyphs = 0, []
    for ch in text:
        face.load_char(ch, _ft.FT_LOAD_RENDER)
        g = face.glyph
        bmp = (np.array(g.bitmap.buffer, dtype=np.uint8)
               .reshape(g.bitmap.rows, g.bitmap.width)
               if g.bitmap.rows and g.bitmap.width
               else np.zeros((0, 0), np.uint8))
        glyphs.append(dict(bitmap=bmp, rows=g.bitmap.rows, width=g.bitmap.width,
                           bearing_x=g.bitmap_left, bearing_y=g.bitmap_top,
                           advance=g.advance.x >> 6, pen_x=pen_x))
        pen_x += g.advance.x >> 6

    if not glyphs:
        return np.zeros((1,1,4), np.uint8), 1, 1

    max_above = max((i['bearing_y']             for i in glyphs), default=1)
    max_below = max(max((i['rows']-i['bearing_y'] for i in glyphs), default=0), 0)
    total_h, total_w = max_above + max_below + PAD*2, pen_x + PAD
    canvas     = np.zeros((total_h, total_w, 4), dtype=np.uint8)
    baseline_y = PAD + max_above

    for info in glyphs:
        x0, y0 = info['pen_x']+info['bearing_x'], baseline_y-info['bearing_y']
        bmp = info['bitmap']
        h, w = bmp.shape
        if not (h and w): continue
        cx0=max(x0,0); cx1=min(x0+w,total_w)
        cy0=max(y0,0); cy1=min(y0+h,total_h)
        if cx1>cx0 and cy1>cy0:
            bx0=cx0-x0; by0=cy0-y0
            alpha = bmp[by0:by0+(cy1-cy0), bx0:bx0+(cx1-cx0)]
            canvas[cy0:cy1, cx0:cx1, :3] = 255
            canvas[cy0:cy1, cx0:cx1,  3] = alpha

    return canvas, total_w, total_h


def upload_text_texture(text: str, size_px: int = 36):
    img, w, h = render_text_to_pixels(text, size_px=size_px)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    for p, v in [(GL_TEXTURE_MIN_FILTER, GL_LINEAR),
                 (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                 (GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE),
                 (GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE)]:
        glTexParameteri(GL_TEXTURE_2D, p, v)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img.flatten())
    return tex, w, h


# ==========================================================
# GLSL shaders
# ==========================================================

_VS = """
#version 330
layout(location=0) in vec2 position;
out vec2 fragCoord;
void main(){ fragCoord=position; gl_Position=vec4(position,0,1); }
"""

_FS_GABOR = """
#version 330
in vec2 fragCoord; out vec4 FragColor;
uniform float contrast, spatial_freq, phase, screen_width, screen_height, radius, mean_lum;
uniform vec3 dkl_bg, col_dir;
uniform mat3 M_dkl2rgb;
uniform float L_min, L_max;
uniform int   lut_size;
uniform samplerBuffer lut_tex;
const float PI = 3.141592653589793;

float lut_lookup(float L){
    float t = clamp((L - L_min)/(L_max - L_min), 0.0, 1.0);
    int   i = int(t * float(lut_size - 1));
    return texelFetch(lut_tex, i).r;
}
void main(){
    float x=(fragCoord.x*.5+.5)*screen_width, y=(fragCoord.y*.5+.5)*screen_height;
    float dx=x-screen_width*.5, dy=y-screen_height*.5;
    float g=exp(-(dx*dx+dy*dy)/(2.*radius*radius));
    float mod_val=g*cos(2.*PI*(spatial_freq*x-phase))*contrast*mean_lum;
    vec3 lin=M_dkl2rgb*(dkl_bg+mod_val*col_dir);
    if(min(lin.r,min(lin.g,lin.b))<0.){ FragColor=vec4(1,0,0,1); return; }
    FragColor=vec4(lut_lookup(lin.r),lut_lookup(lin.g),lut_lookup(lin.b),1.);
}
"""

_FS_FLAT = """
#version 330
in vec2 fragCoord; out vec4 FragColor;
uniform vec3 bg_color;
void main(){ FragColor=vec4(bg_color,1.); }
"""

_FS_DIGIT = """
#version 330
in vec2 fragCoord; out vec4 FragColor;
uniform vec3 bg_color, fg_color;
uniform int digit;
uniform float cx_ndc, cy_ndc, char_size_ndc;
float box(vec2 p,vec2 b){ vec2 d=abs(p)-b; return length(max(d,0.))+min(max(d.x,d.y),0.); }
void main(){
    vec2 l=(fragCoord-vec2(cx_ndc,cy_ndc))/char_size_ndc; float m=0.,t=.18;
    if(digit==1){
        if(box(l-vec2(0.,0.),vec2(t*.5,.75))<0.)m=1.;
        if(box(l-vec2(-.15,.62),vec2(.18,t*.4))<0.)m=1.;
    }else{
        if(box(l-vec2(0.,.72),vec2(.38,t))<0.)m=1.;
        if(box(l-vec2(.32,.42),vec2(t,.28))<0.)m=1.;
        if(box(l-vec2(0.,.10),vec2(.38,t))<0.)m=1.;
        if(box(l-vec2(-.32,-.22),vec2(t,.28))<0.)m=1.;
        if(box(l-vec2(0.,-.55),vec2(.38,t))<0.)m=1.;
    }
    FragColor=m>.5?vec4(fg_color,1.):vec4(bg_color,1.);
}
"""

_FS_TEX = """
#version 330
in vec2 fragCoord; out vec4 FragColor;
uniform sampler2D tex;
uniform float tex_x, tex_y, tex_w, tex_h;
void main(){
    vec2 uv=fragCoord*.5+.5;
    float u=(uv.x-(tex_x*.5+.5))/tex_w, v=(uv.y-(tex_y*.5+.5))/tex_h;
    if(u>=0.&&u<=1.&&v>=0.&&v<=1.) FragColor=texture(tex,vec2(u,1.-v));
    else discard;
}
"""


# ==========================================================
# VAO helper
# ==========================================================

def _make_fullscreen_vao():
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    verts = np.array([-1,-1, 1,-1, 1,1, -1,1], dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    idx = np.array([0,1,2,2,3,0], dtype=np.uint32)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    return vao


# def _quad(vao):
#     glBindVertexArray(vao)
#     glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
# def _quad(vao):
#     # 只渲染中心 800x800 区域
#     win_w = glGetIntegerv(GL_VIEWPORT)[2]
#     win_h = glGetIntegerv(GL_VIEWPORT)[3]
#
#     box_size = 800
#     x = int((win_w - box_size) / 2)
#     y = int((win_h - box_size) / 2)
#
#     glEnable(GL_SCISSOR_TEST)
#     glScissor(x, y, box_size, box_size)
#
#     glBindVertexArray(vao)
#     glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
#
#     glDisable(GL_SCISSOR_TEST)
def _quad(vao, box_size_px=None):
    if box_size_px is None:
        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        return
    win_w = glGetIntegerv(GL_VIEWPORT)[2]
    win_h = glGetIntegerv(GL_VIEWPORT)[3]
    x = int((win_w - box_size_px) / 2)
    y = int((win_h - box_size_px) / 2)
    glEnable(GL_SCISSOR_TEST)
    glScissor(x, y, box_size_px, box_size_px)
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    glDisable(GL_SCISSOR_TEST)
# ==========================================================
# Public renderer class
# ==========================================================

class ExperimentRenderer:
    """
    OpenGL rendering context for the Quest 2AFC Gabor experiment.

    Lifecycle
    ---------
    1. ``glfw.init()``
    2. ``renderer = ExperimentRenderer(diagonal_inch, visual_radius_deg)``
    3. ``renderer.init_window()``
    4. ``renderer.set_condition(color, luminance)``   ← once per condition
    5. Rendering calls: show_flat / show_interval_cue / show_gabor /
       wait_for_response / show_progress_screen
    6. ``renderer.destroy()`` → ``glfw.terminate()``

    Parameters
    ----------
    diagonal_inch : float      Physical screen diagonal (inches).
    visual_radius_deg : float  Gabor envelope radius (visual degrees).
    monitor_index : int        0-based GLFW monitor index (default 1).
    lut_json_path : str        Path to PCHIP calibration JSON.
    """

    def __init__(self, diagonal_inch: float, visual_radius_deg: float,
                 background_size_deg=5.0,
                 monitor_index: int = 1,
                 lut_json_path: str = "calibrate_display/Measure_specbos/pixel_luminance_model_pchip_B100_C100_s800.json"):
        self.diagonal_inch     = diagonal_inch
        self.visual_radius_deg = visual_radius_deg
        self.monitor_index     = monitor_index
        self._lut_json_path    = lut_json_path
        self.window = self.width = self.height = self.refresh = self.W = None
        self._M_combined = self._dkl_bg = self._col_dir = self._mean_luminance = None
        self.background_size_deg = background_size_deg

    def _bg_size_px(self, distance_m):
        half_rad = np.deg2rad(self.background_size_deg / 2.0)
        size_m = 2.0 * distance_m * np.tan(half_rad)  # 物理边长（米）
        return int((size_m / self.W) * self.width)

    # --- lifecycle ---

    def init_window(self):
        mon  = glfw.get_monitors()[self.monitor_index]
        mode = glfw.get_video_mode(mon)
        self.width, self.height, self.refresh = mode.size.width, mode.size.height, mode.refresh_rate
        self.window = glfw.create_window(self.width, self.height, "Quest 2AFC", mon, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.W    = compute_display_width(self.diagonal_inch, self.width, self.height)
        self._vao = _make_fullscreen_vao()
        glClearColor(0.0, 0.0, 0.0, 1.0)  # 背景设为纯黑
        pchip, L_min, L_max = load_pchip_model(self._lut_json_path)
        self._L_min, self._L_max = L_min, L_max
        self._lut_L, self._lut_p = build_lut(pchip, L_min, L_max)
        # 上传 LUT 到 Texture Buffer Object（只需一次）
        self._lut_tbo = glGenBuffers(1)
        glBindBuffer(GL_TEXTURE_BUFFER, self._lut_tbo)
        glBufferData(GL_TEXTURE_BUFFER,
                     self._lut_p.astype(np.float32).nbytes,
                     self._lut_p.astype(np.float32),
                     GL_STATIC_DRAW)

        self._lut_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_BUFFER, self._lut_tex)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, self._lut_tbo)
        mk = lambda fs: compileProgram(compileShader(_VS, GL_VERTEX_SHADER),
                                       compileShader(fs,  GL_FRAGMENT_SHADER), validate=False)
        self._sh_gabor = mk(_FS_GABOR)
        self._sh_flat  = mk(_FS_FLAT)
        self._sh_digit = mk(_FS_DIGIT)
        self._sh_tex   = mk(_FS_TEX)

    def set_condition(self, color_direction: str, mean_luminance: float):
        """Set colour direction ('ach'|'rg'|'yv') and mean luminance (cd/m²)."""
        self._M_combined, self._dkl_bg, self._col_dir = get_color_matrices(
            mean_luminance, color_direction)
        self._mean_luminance = mean_luminance

    def destroy(self):
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None

    # --- internal ---

    def _loc(self, sh, name): return glGetUniformLocation(sh, name)

    def poll_and_check_escape(self) -> bool:
        glfw.poll_events()
        return glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS

    # --- rendering API ---

    def show_flat(self, distance_m: float, rgb=(0.2, 0.2, 0.2)):
        """Single uniform-colour frame."""
        glUseProgram(self._sh_flat)
        glUniform3f(self._loc(self._sh_flat, "bg_color"), *rgb)
        glClear(GL_COLOR_BUFFER_BIT)
        _quad(self._vao, self._bg_size_px(distance_m))
        glfw.swap_buffers(self.window)

    def show_interval_cue(self, interval_number: int,  distance_m: float, duration: float = 0.5) -> bool:
        """Show digit '1' or '2' for *duration* seconds. Returns True on ESC."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                return True
            sh = self._sh_digit
            glUseProgram(sh)
            glUniform3f(self._loc(sh,"bg_color"), 0.15, 0.15, 0.15)
            glUniform3f(self._loc(sh,"fg_color"), 0., 0., 0.)
            glUniform1i(self._loc(sh,"digit"),         interval_number)
            glUniform1f(self._loc(sh,"cx_ndc"),        0.)
            glUniform1f(self._loc(sh,"cy_ndc"),        0.)
            glUniform1f(self._loc(sh,"char_size_ndc"), 0.18)
            glClear(GL_COLOR_BUFFER_BIT)
            _quad(self._vao, self._bg_size_px(distance_m))
            glfw.swap_buffers(self.window)
        return False

    def show_gabor(self, contrast: float, distance_m: float,
                   spatial_frequency_cpp: float, speed: float,
                   duration: float, ramp_duration: float = 0.2) -> bool:
        """
        Animate a drifting Gabor for *duration* seconds.

        Parameters
        ----------
        contrast              Michelson contrast (0 = blank).
        distance_m            Viewing distance in metres.
        spatial_frequency_cpp Spatial frequency in cycles/pixel.
        speed                 Drift in pixels/second.
        duration              Display time in seconds.

        Returns True if ESC pressed.
        """
        if self._M_combined is None:
            raise RuntimeError("Call set_condition() before show_gabor().")
        sh = self._sh_gabor
        glUseProgram(sh)
        # 绑定 LUT TBO 到纹理单元 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, self._lut_tex)
        glUniform1i(self._loc(sh, "lut_tex"), 0)
        glUniform1i(self._loc(sh, "lut_size"), len(self._lut_p))
        # L_min / L_max 保持不变，继续用 glUniform1f 传
        glUniform1f(self._loc(sh,"L_min"), self._L_min)
        glUniform1f(self._loc(sh,"L_max"), self._L_max)
        glUniformMatrix3fv(self._loc(sh,"M_dkl2rgb"), 1, GL_FALSE,
                           self._M_combined.T.flatten().astype(np.float32))
        glUniform3f(self._loc(sh,"dkl_bg"),        *self._dkl_bg)
        glUniform3f(self._loc(sh,"col_dir"),        *self._col_dir)
        glUniform1f(self._loc(sh,"mean_lum"),        self._mean_luminance)
        # glUniform1f(self._loc(sh,"contrast"),        contrast)
        glUniform1f(self._loc(sh,"spatial_freq"),    spatial_frequency_cpp)
        glUniform1f(self._loc(sh,"screen_width"),    self.width)
        glUniform1f(self._loc(sh,"screen_height"),   self.height)
        glUniform1f(self._loc(sh,"radius"),
                    visual_radius_deg_to_px(self.visual_radius_deg,
                                            distance_m, self.W, self.width))
        phase      = 0.0
        phase_step = spatial_frequency_cpp * speed / self.refresh
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                return True
            phase = (phase + phase_step) % 1.0
            t_elapsed = time.perf_counter() - t0  # ← 新增
            ramped_contrast = contrast * min(t_elapsed / ramp_duration, 1.0)  # ← 新增
            glUniform1f(self._loc(sh, "phase"), phase)
            glUniform1f(self._loc(sh, "contrast"), ramped_contrast)  # ← 新增，替换循环外那行
            glClear(GL_COLOR_BUFFER_BIT)
            _quad(self._vao, self._bg_size_px(distance_m))
            glfw.swap_buffers(self.window)
        return False

    def draw_text_overlay(self, text, distance_m: float, y_ndc=-0.75, size_px=36, bg=(0.25, 0.25, 0.25)):
        tex, tw, th = upload_text_texture(text, size_px=size_px)
        ndcw = tw / self.width * 2.0
        ndch = th / self.height * 2.0
        sh = self._sh_tex
        glUseProgram(sh)
        glActiveTexture(GL_TEXTURE1)  # ← 改为 TEXTURE1
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(self._loc(sh, "tex"), 1)  # ← 对应改为 1
        glUniform1f(self._loc(sh, "tex_y"), y_ndc)
        glUniform1f(self._loc(sh, "tex_w"), ndcw)
        glUniform1f(self._loc(sh, "tex_h"), ndch)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        _quad(self._vao, self._bg_size_px(distance_m))
        glDisable(GL_BLEND)
        glDeleteTextures(1, [tex])

    def wait_for_response(self, distance_m: float):
        """
        Block until ← (→ interval 1) or → (→ interval 2) pressed.
        Returns (interval_chosen: int | None, esc_pressed: bool).
        """
        L1 = "Please choose:"
        L2 = "\u25c4  first signal"
        L3 = "\u25ba  second signal"
        bg = (0.25, 0.25, 0.25)
        while True:
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                return None, True
            for key, val in [(glfw.KEY_LEFT, 1), (glfw.KEY_RIGHT, 2)]:
                if glfw.get_key(self.window, key) == glfw.PRESS:
                    while glfw.get_key(self.window, key) == glfw.PRESS:
                        glfw.poll_events()
                    return val, False
            glUseProgram(self._sh_flat)
            glUniform3f(self._loc(self._sh_flat,"bg_color"), *bg)
            glClear(GL_COLOR_BUFFER_BIT)
            _quad(self._vao, self._bg_size_px(distance_m))
            self.draw_text_overlay(L1, distance_m, y_ndc= 0.05, size_px=8, bg=bg)
            self.draw_text_overlay(L2, distance_m, y_ndc= 0.00, size_px=8, bg=bg)
            self.draw_text_overlay(L3, distance_m, y_ndc= -0.05, size_px=8, bg=bg)
            glfw.swap_buffers(self.window)
            time.sleep(0.005)

    def show_progress_screen(self, conditions_done: int, conditions_total: int,
                             elapsed_sec: float, distance_m: float, duration_auto: float = 3.0) -> bool:
        """Progress bar + ETA for *duration_auto* seconds. Returns True on ESC."""
        t0   = time.perf_counter()
        bg   = (0.1, 0.1, 0.1)
        frac = conditions_done / max(conditions_total, 1)
        eta  = (elapsed_sec / conditions_done * (conditions_total - conditions_done)
                if elapsed_sec > 0 and conditions_done > 0 else 0.0)
        l1 = f"Condition {conditions_done} of {conditions_total}  ({frac*100:.0f}%)"
        l2 = f"Elapsed: {int(elapsed_sec//60)}m {int(elapsed_sec%60):02d}s"
        l3 = f"ETA: {int(eta // 60)}m {int(eta % 60):02d}s"
        print(f"\n  {l1}\n  {l2}\n")
        while time.perf_counter() - t0 < duration_auto:
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                return True
            glClear(GL_COLOR_BUFFER_BIT)
            sh = self._sh_flat
            glUseProgram(sh)
            glUniform3f(self._loc(sh,"bg_color"), *bg)
            _quad(self._vao, self._bg_size_px(distance_m))
            glEnable(GL_SCISSOR_TEST)
            bh = max(1, int(self.height*.04)); bfw = int(self.width*.70)
            bx = int(self.width*.15);          by  = int(self.height*.44)
            glScissor(bx, by, bfw, bh)
            glUniform3f(self._loc(sh,"bg_color"), .3,.3,.3); _quad(self._vao, self._bg_size_px(distance_m))
            fw = int(bfw * frac)
            if fw > 0:
                glScissor(bx, by, fw, bh)
                glUniform3f(self._loc(sh,"bg_color"), .1,.75,.25); _quad(self._vao, self._bg_size_px(distance_m))
            glDisable(GL_SCISSOR_TEST)
            self.draw_text_overlay(l1, distance_m, y_ndc=.05, size_px=8, bg=bg)
            self.draw_text_overlay(l2, distance_m, y_ndc=0.0, size_px=8, bg=bg)
            self.draw_text_overlay(l3, distance_m, y_ndc=-0.05, size_px=8, bg=bg)
            glfw.swap_buffers(self.window)
            time.sleep(1.0 / self.refresh)
        return False