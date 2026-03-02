"""
Quest 2AFC Experiment
=====================
A two-alternative forced-choice (2AFC) experiment using PsychoPy's Quest
staircase to estimate the distance threshold at which a Gabor stimulus
becomes detectable (test = with contrast, reference = contrast 0).

Controls during a trial:
  Left arrow  → chose interval 1 as "test"
  Right arrow → chose interval 2 as "test"

Mid-experiment early stop:
  ESC during the inter-trial / progress screen → graceful exit

Author: generated for YanchengCai
"""

import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from psychopy.data import QuestHandler
import time
import csv
import threading
import os
import argparse
import random
from control_display.control_display_main import rsbg_init, rsbg_update, rsbg_cleanup

# Audio feedback (sounddevice + numpy; falls back to WAV+aplay if unavailable)
try:
    import sounddevice as _sd
    _AUDIO_BACKEND = 'sounddevice'
except Exception:
    _AUDIO_BACKEND = 'none'


# ==========================================================
# Audio feedback
# ==========================================================

def _beep(frequency_hz, duration_s=0.15, volume=0.4, sample_rate=44100):
    """
    Play a pure sine-wave beep asynchronously.
    Tries sounddevice first; falls back to writing a WAV + system player.
    frequency_hz : 1200 = correct (high), 400 = incorrect (low)
    """
    import threading, wave as _wave, tempfile, os, subprocess

    def _play():
        # ---- build PCM samples ----
        n = int(sample_rate * duration_s)
        t = np.linspace(0, duration_s, n, endpoint=False)
        sig = np.sin(2 * np.pi * frequency_hz * t).astype(np.float32)
        fade = min(int(sample_rate * 0.02), n)
        sig[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        sig *= volume

        # ---- try sounddevice ----
        try:
            import sounddevice as sd
            sd.play(sig, samplerate=sample_rate, blocking=True)
            return
        except Exception:
            pass

        # ---- fallback: write WAV, play with system tool ----
        pcm = (sig * 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            with _wave.open(tmp_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm.tobytes())
            for player in ('aplay', 'paplay', 'afplay', 'pw-play'):
                try:
                    subprocess.run([player, tmp_path],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   timeout=duration_s + 1.0)
                    return
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            print(f"[audio] no player found for beep at {frequency_hz} Hz")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    threading.Thread(target=_play, daemon=True).start()


BEEP_CORRECT   = 1200   # Hz  – high pitch = correct
BEEP_INCORRECT =  400   # Hz  – low  pitch = incorrect

# ==========================================================
# 固定物理参数（与 MOA 保持一致）
# ==========================================================

INITIAL_DISTANCE = 0.5
UNIT_PER_M = 100 / 1.6

MIN_DIST = 0.5
MAX_DIST = 2.1

L_MIN = 0.023666
L_MAX = 117.05665780294088

MOA_CSV = "MOA_results.csv"
QUEST_CSV = "Quest_results.csv"

TRIALS_PER_CONDITION = 20
QUEST_CONVERGE_THRESH = 0.005  # metres – early stop if Quest step < this

# ==========================================================
# 几何计算（与 MOA 完全相同）
# ==========================================================

def compute_display_width(diagonal_inch, R_x, R_y):
    D_m = diagonal_inch * 0.0254
    return D_m * (R_x / np.sqrt(R_x ** 2 + R_y ** 2))


def compute_spatiotemporal_frequency(R_x, W, d, f_p, v_p):
    theta = 2 * np.arctan(W / (2 * d))
    rho_rad = (R_x * f_p) / theta
    rho_cpd = rho_rad * (np.pi / 180)
    omega = f_p * v_p
    return rho_cpd, omega, theta


def visual_radius_deg_to_px(visual_radius_deg, d, W, R_x):
    phi_rad = np.deg2rad(visual_radius_deg)
    R_phys = d * np.tan(phi_rad)
    return (R_phys / W) * R_x


# ==========================================================
# DKL 颜色矩阵（与 MOA 完全相同）
# ==========================================================

def get_color_matrices(mean_luminance, color_direction):
    lms_gray = np.array([0.739876529525622,
                         0.320136241543338,
                         0.020793708751515])

    mc1 = lms_gray[0] / lms_gray[1]
    mc2 = (lms_gray[0] + lms_gray[1]) / lms_gray[2]

    M_lms_dkl = np.array([[1, 1, 0],
                          [1, -mc1, 0],
                          [-1, -1, mc2]])

    M_dkl_lms = np.linalg.inv(M_lms_dkl)

    M_lms_xyz = np.array([
        [2.629129278399650, -3.780202391780134, 10.294956387893450],
        [0.865649062438827, 1.215555811642301, -0.984175688105352],
        [-0.008886561474676, 0.081612628990755, 51.371024830897888]
    ])

    M_xyz_rgb = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    M_combined = M_xyz_rgb @ M_lms_xyz @ M_dkl_lms

    white_point_d65 = np.array([0.9505, 1.0000, 1.0888])

    M_xyz_lms = np.array([
        [0.187596268556126, 0.585168649077728, -0.026384263306304],
        [-0.133397430663221, 0.405505777260049, 0.034502127690364],
        [0.000244379021663, -0.000542995890619, 0.019406849066323]
    ])

    lms_wp = white_point_d65 @ M_xyz_lms.T
    dkl_wp = lms_wp @ M_lms_dkl.T
    dkl_bg = mean_luminance * dkl_wp

    if color_direction == "ach":
        col_dir = np.array([1, 0, 0])
    elif color_direction == "rg":
        col_dir = np.array([0, 1, 0])
    elif color_direction == "yv":
        col_dir = np.array([0, 0, 1])
    else:
        raise ValueError("color_direction must be ach/rg/yv")

    return M_combined, dkl_bg, col_dir


# ==========================================================
# 从 MOA CSV 读取起始距离
# ==========================================================

def load_moa_distance(name, color, speed, luminance):
    """
    返回 MOA 中 name 对应 color/speed/luminance 条件的平均 distance_m。
    如果找不到，返回 1.0 m。
    """
    if not os.path.exists(MOA_CSV):
        return 1.0

    distances = []
    try:
        with open(MOA_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get('name') == name and
                        row.get('color') == color and
                        abs(float(row.get('speed_px_per_sec', -1)) - speed) < 1e-3 and
                        abs(float(row.get('mean_luminance', -1)) - luminance) < 1e-3):
                    distances.append(float(row['distance_m']))
    except Exception as e:
        print(f"[WARN] Could not read MOA CSV: {e}")
        return 1.0

    if distances:
        return float(np.mean(distances))
    return 1.0


# ==========================================================
# 读取已完成条件（断点续跑）
# ==========================================================

def load_completed_conditions(name, csv_path=QUEST_CSV):
    """
    Returns a set of condition keys (color, speed, luminance) that already
    have TRIALS_PER_CONDITION trials logged for `name` in the Quest CSV.
    Keys are tuples: (color, speed_float, luminance_float)
    """
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    from collections import Counter
    counts = Counter()
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('name') != name:
                    continue
                key = (
                    row.get('color', ''),
                    float(row.get('speed_px_per_sec', 0)),
                    float(row.get('mean_luminance', 0)),
                )
                counts[key] += 1
        for key, cnt in counts.items():
            if cnt >= TRIALS_PER_CONDITION:
                completed.add(key)
    except Exception as e:
        print(f"[WARN] Could not read Quest CSV for resume: {e}")
    return completed



# ==========================================================
# freetype-py 字体渲染（LiberationSerif ≈ Times New Roman）
# ==========================================================

import freetype as _ft

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
_ft_face = None  # lazily initialised after OpenGL context is created

def _get_face(size_px=48):
    global _ft_face
    if _ft_face is None:
        _ft_face = _ft.Face(_FONT_PATH)
    _ft_face.set_pixel_sizes(0, size_px)
    return _ft_face


def render_text_to_pixels(text, size_px=36):
    """
    Render a string with freetype into an RGBA numpy array.
    Canvas height is computed from actual glyph metrics (no clipping).
    Returns (rgba [H,W,4], width, height). Text is white on transparent bg.
    """
    face = _get_face(size_px)
    PAD = 4  # pixels of padding top/bottom/right

    # ---- first pass: load glyphs and measure ----
    pen_x = 0
    glyphs = []
    for ch in text:
        face.load_char(ch, _ft.FT_LOAD_RENDER)
        g = face.glyph
        bmp_arr = (np.array(g.bitmap.buffer, dtype=np.uint8)
                   .reshape(g.bitmap.rows, g.bitmap.width)
                   if g.bitmap.rows and g.bitmap.width
                   else np.zeros((0, 0), np.uint8))
        glyphs.append({
            'bitmap':    bmp_arr,
            'rows':      g.bitmap.rows,
            'width':     g.bitmap.width,
            'bearing_x': g.bitmap_left,
            'bearing_y': g.bitmap_top,
            'advance':   g.advance.x >> 6,
            'pen_x':     pen_x,
        })
        pen_x += g.advance.x >> 6

    if not glyphs:
        return np.zeros((1, 1, 4), np.uint8), 1, 1

    # ---- compute true bounding box from actual glyph extents ----
    max_above = max((info['bearing_y'] for info in glyphs), default=1)
    max_below = max((info['rows'] - info['bearing_y'] for info in glyphs), default=0)
    max_below = max(max_below, 0)

    total_h = max_above + max_below + PAD * 2
    total_w = pen_x + PAD

    canvas = np.zeros((total_h, total_w, 4), dtype=np.uint8)

    # baseline sits at PAD + max_above from the top
    baseline_y = PAD + max_above

    for info in glyphs:
        x0 = info['pen_x'] + info['bearing_x']
        y0 = baseline_y - info['bearing_y']
        bmp = info['bitmap']
        h, w = bmp.shape
        if h == 0 or w == 0:
            continue
        x1, y1 = x0 + w, y0 + h
        # safe clipping
        cx0 = max(x0, 0); cx1 = min(x1, total_w)
        cy0 = max(y0, 0); cy1 = min(y1, total_h)
        bx0 = cx0 - x0;  bx1 = bx0 + (cx1 - cx0)
        by0 = cy0 - y0;  by1 = by0 + (cy1 - cy0)
        if cx1 > cx0 and cy1 > cy0:
            alpha = bmp[by0:by1, bx0:bx1]
            canvas[cy0:cy1, cx0:cx1, 0] = 255
            canvas[cy0:cy1, cx0:cx1, 1] = 255
            canvas[cy0:cy1, cx0:cx1, 2] = 255
            canvas[cy0:cy1, cx0:cx1, 3] = alpha

    return canvas, total_w, total_h


def upload_text_texture(text, size_px=36):
    """Upload rendered text as an OpenGL RGBA texture. Returns (tex_id, w, h)."""
    img, w, h = render_text_to_pixels(text, size_px=size_px)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img.flatten())
    return tex, w, h

# ==========================================================
# 平台控制
# ==========================================================

class PlatformController:

    def __init__(self, port):
        self.port = rsbg_init(port)
        rsbg_update(self.port, 'reset')
        rsbg_update(self.port, 'move_and_wait', 0)

    def meters_to_units(self, distance_m):
        delta = distance_m - INITIAL_DISTANCE
        return int(round(-delta * UNIT_PER_M))

    def move_to(self, distance_m):
        units = self.meters_to_units(distance_m)
        rsbg_update(self.port, 'move_and_wait', units)

    def cleanup(self):
        rsbg_update(self.port, 'move_and_wait', 0)
        rsbg_cleanup(self.port)


# ==========================================================
# GLSL shaders（共用）
# ==========================================================

VERTEX_SHADER = """
#version 330
layout(location=0) in vec2 position;
out vec2 fragCoord;
void main(){
    fragCoord = position;
    gl_Position = vec4(position,0,1);
}
"""

FRAGMENT_SHADER_GABOR = """
#version 330
in vec2 fragCoord;
out vec4 FragColor;

uniform float contrast;
uniform float spatial_freq;
uniform float phase;
uniform float screen_width;
uniform float screen_height;
uniform float radius;
uniform float mean_lum;
uniform float L_min;
uniform float L_max;
uniform vec3 dkl_bg;
uniform vec3 col_dir;
uniform mat3 M_dkl2rgb;

const float PI = 3.141592653589793;

float srgb_inverse_eotf(float x){
    if (x <= 0.0031308)
        return 12.92*x;
    else
        return 1.055*pow(x,1.0/2.4)-0.055;
}

vec3 srgb_inverse_eotf3(vec3 c){
    return vec3(
        srgb_inverse_eotf(c.r),
        srgb_inverse_eotf(c.g),
        srgb_inverse_eotf(c.b)
    );
}

void main(){
    float x = (fragCoord.x*0.5+0.5)*screen_width;
    float y = (fragCoord.y*0.5+0.5)*screen_height;

    float cx = screen_width*0.5;
    float cy = screen_height*0.5;

    float dx = x-cx;
    float dy = y-cy;

    float gaussian = exp(-(dx*dx+dy*dy)/(2.0*radius*radius));
    float carrier = cos(2.0*PI*(spatial_freq*x-phase));
    float gabor = gaussian*carrier;

    float modulation = gabor*contrast*mean_lum;
    vec3 dkl_pixel = dkl_bg + modulation*col_dir;

    vec3 lin_rgb = M_dkl2rgb * dkl_pixel;

    if(min(lin_rgb.r,min(lin_rgb.g,lin_rgb.b))<0.0){
        FragColor=vec4(1,0,0,1);
        return;
    }

    vec3 linear_norm = (lin_rgb-L_min)/(L_max-L_min);
    linear_norm = clamp(linear_norm,0.0,1.0);

    vec3 pixel = srgb_inverse_eotf3(linear_norm);
    FragColor = vec4(pixel,1);
}
"""

FRAGMENT_SHADER_FLAT = """
#version 330
in vec2 fragCoord;
out vec4 FragColor;
uniform vec3 bg_color;
void main(){
    FragColor = vec4(bg_color, 1.0);
}
"""

FRAGMENT_SHADER_TEXT = """
#version 330
in vec2 fragCoord;
out vec4 FragColor;
uniform vec3 bg_color;
uniform vec3 fg_color;
// We render digit glyphs via a simple pixel-art SDF approach using uniforms
uniform int digit;           // 1 or 2
uniform float cx_ndc;        // center x in NDC
uniform float cy_ndc;        // center y in NDC
uniform float char_size_ndc; // half-size of character in NDC

float box(vec2 p, vec2 b){
    vec2 d = abs(p)-b;
    return length(max(d,0.0))+min(max(d.x,d.y),0.0);
}

void main(){
    // Pixel position in NDC
    vec2 uv = fragCoord;
    // Character space: map to [-1,1] within the char box
    vec2 local = (uv - vec2(cx_ndc, cy_ndc)) / char_size_ndc;

    float mask = 0.0;
    float thick = 0.18; // stroke thickness in local coords

    if(digit == 1){
        // Vertical bar
        float bar = box(local - vec2(0.0, 0.0), vec2(thick*0.5, 0.75));
        // Serif / top-left diagonal hint
        float top = box(local - vec2(-0.15, 0.62), vec2(0.18, thick*0.4));
        if(bar < 0.0 || top < 0.0) mask = 1.0;
    } else {
        // "2" built from line segments
        // Top horizontal
        if(box(local - vec2(0.0, 0.72), vec2(0.38, thick)) < 0.0) mask=1.0;
        // Top-right vertical (upper)
        if(box(local - vec2(0.32, 0.42), vec2(thick, 0.28)) < 0.0) mask=1.0;
        // Middle horizontal
        if(box(local - vec2(0.0, 0.10), vec2(0.38, thick)) < 0.0) mask=1.0;
        // Bottom-left vertical (lower)
        if(box(local - vec2(-0.32, -0.22), vec2(thick, 0.28)) < 0.0) mask=1.0;
        // Bottom horizontal
        if(box(local - vec2(0.0, -0.55), vec2(0.38, thick)) < 0.0) mask=1.0;
    }

    if(mask > 0.5)
        FragColor = vec4(fg_color, 1.0);
    else
        FragColor = vec4(bg_color, 1.0);
}
"""




FRAGMENT_SHADER_TEX = """
#version 330
in vec2 fragCoord;
out vec4 FragColor;
uniform sampler2D tex;
uniform float tex_x;   // NDC left edge
uniform float tex_y;   // NDC bottom edge
uniform float tex_w;   // NDC width
uniform float tex_h;   // NDC height
void main(){
    vec2 uv = fragCoord * 0.5 + 0.5;
    float u = (uv.x - (tex_x * 0.5 + 0.5)) / tex_w;
    float v = (uv.y - (tex_y * 0.5 + 0.5)) / tex_h;
    if(u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0){
        FragColor = texture(tex, vec2(u, 1.0 - v));
    } else {
        discard;
    }
}
"""

# ==========================================================
# OpenGL helpers
# ==========================================================

def make_fullscreen_vao():
    vertices = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    indices  = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    return vao


def draw_quad(vao):
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)


# ==========================================================
# Rendering context (runs on main thread after glfw.init)
# ==========================================================

class ExperimentRenderer:
    """
    Manages the OpenGL window on monitor index 1 and exposes
    high-level methods for each screen phase.
    """

    def __init__(self, diagonal_inch, visual_radius_deg, monitor_index=1):
        self.diagonal_inch = diagonal_inch
        self.visual_radius_deg = visual_radius_deg
        self.monitor_index = monitor_index
        self.window = None
        self.width = None
        self.height = None
        self.refresh = None
        self.W = None  # physical display width in meters
        self._gabor_shader = None
        self._flat_shader = None
        self._text_shader = None
        self._tex_shader = None
        self._vao = None
        self._M_combined = None
        self._dkl_bg = None
        self._col_dir = None

    # ----------------------------------------------------------
    def init_window(self):
        monitors = glfw.get_monitors()
        monitor = monitors[self.monitor_index]
        mode = glfw.get_video_mode(monitor)
        self.width  = mode.size.width
        self.height = mode.size.height
        self.refresh = mode.refresh_rate
        self.window = glfw.create_window(
            self.width, self.height, "Quest 2AFC", monitor, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.W = compute_display_width(self.diagonal_inch, self.width, self.height)
        self._vao = make_fullscreen_vao()

        self._gabor_shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER_GABOR, GL_FRAGMENT_SHADER)
        )
        self._flat_shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER_FLAT, GL_FRAGMENT_SHADER)
        )
        self._text_shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER_TEXT, GL_FRAGMENT_SHADER)
        )
        self._tex_shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER_TEX, GL_FRAGMENT_SHADER)
        )

    # ----------------------------------------------------------
    def set_condition(self, color_direction, mean_luminance):
        self._M_combined, self._dkl_bg, self._col_dir = get_color_matrices(
            mean_luminance, color_direction)
        self._mean_luminance = mean_luminance

    # ----------------------------------------------------------
    def poll_and_check_escape(self):
        glfw.poll_events()
        return glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS

    # ----------------------------------------------------------
    def show_flat(self, rgb=(0.2, 0.2, 0.2)):
        """Show a flat color frame."""
        glUseProgram(self._flat_shader)
        glUniform3f(glGetUniformLocation(self._flat_shader, "bg_color"), *rgb)
        glClear(GL_COLOR_BUFFER_BIT)
        draw_quad(self._vao)
        glfw.swap_buffers(self.window)

    # ----------------------------------------------------------
    def show_interval_cue(self, interval_number, duration=0.5):
        """
        Show dark-grey background + digit (1 or 2) for `duration` seconds.
        Returns True if ESC pressed.
        """
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                return True
            glUseProgram(self._text_shader)
            glUniform3f(glGetUniformLocation(self._text_shader, "bg_color"), 0.15, 0.15, 0.15)
            glUniform3f(glGetUniformLocation(self._text_shader, "fg_color"), 0.0, 0.0, 0.0)
            glUniform1i(glGetUniformLocation(self._text_shader, "digit"), interval_number)
            glUniform1f(glGetUniformLocation(self._text_shader, "cx_ndc"), 0.0)
            glUniform1f(glGetUniformLocation(self._text_shader, "cy_ndc"), 0.0)
            glUniform1f(glGetUniformLocation(self._text_shader, "char_size_ndc"), 0.18)
            glClear(GL_COLOR_BUFFER_BIT)
            draw_quad(self._vao)
            glfw.swap_buffers(self.window)
        return False

    # ----------------------------------------------------------
    def show_gabor(self, contrast, distance_m, spatial_frequency_cpp, speed, duration):
        """
        Animate a Gabor patch for `duration` seconds at given distance.
        Returns True if ESC pressed.
        """
        glUseProgram(self._gabor_shader)
        # Upload fixed uniforms
        glUniformMatrix3fv(
            glGetUniformLocation(self._gabor_shader, "M_dkl2rgb"),
            1, GL_FALSE,
            np.array(self._M_combined.T.flatten(), dtype=np.float32)
        )
        glUniform3f(glGetUniformLocation(self._gabor_shader, "dkl_bg"),    *self._dkl_bg)
        glUniform3f(glGetUniformLocation(self._gabor_shader, "col_dir"),   *self._col_dir)
        glUniform1f(glGetUniformLocation(self._gabor_shader, "L_min"),     L_MIN)
        glUniform1f(glGetUniformLocation(self._gabor_shader, "L_max"),     L_MAX)
        glUniform1f(glGetUniformLocation(self._gabor_shader, "mean_lum"),  self._mean_luminance)
        glUniform1f(glGetUniformLocation(self._gabor_shader, "contrast"),  contrast)
        glUniform1f(glGetUniformLocation(self._gabor_shader, "spatial_freq"), spatial_frequency_cpp)
        glUniform1f(glGetUniformLocation(self._gabor_shader, "screen_width"),  self.width)
        glUniform1f(glGetUniformLocation(self._gabor_shader, "screen_height"), self.height)

        radius_px = visual_radius_deg_to_px(
            self.visual_radius_deg, distance_m, self.W, self.width)
        glUniform1f(glGetUniformLocation(self._gabor_shader, "radius"), radius_px)

        phase = 0.0
        phase_step = spatial_frequency_cpp * speed / self.refresh

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                return True
            phase = (phase + phase_step) % 1.0
            glUniform1f(glGetUniformLocation(self._gabor_shader, "phase"), phase)
            glClear(GL_COLOR_BUFFER_BIT)
            draw_quad(self._vao)
            glfw.swap_buffers(self.window)
        return False

    # ----------------------------------------------------------
    def draw_text_overlay(self, text, y_ndc=-0.75, size_px=36, bg=(0.25, 0.25, 0.25)):
        """
        Render a text string centered horizontally at y_ndc (NDC coords).
        Uses freetype LiberationSerif. Call before swap_buffers.
        """
        tex, tw, th = upload_text_texture(text, size_px=size_px)
        # Convert pixel dims to NDC dims
        ndcw = tw / self.width  * 2.0
        ndch = th / self.height * 2.0
        glUseProgram(self._tex_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(glGetUniformLocation(self._tex_shader, "tex"), 0)
        # Center horizontally
        glUniform1f(glGetUniformLocation(self._tex_shader, "tex_x"), -ndcw / 2.0)
        glUniform1f(glGetUniformLocation(self._tex_shader, "tex_y"),  y_ndc)
        glUniform1f(glGetUniformLocation(self._tex_shader, "tex_w"),  ndcw)
        glUniform1f(glGetUniformLocation(self._tex_shader, "tex_h"),  ndch)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        draw_quad(self._vao)
        glDisable(GL_BLEND)
        glDeleteTextures(1, [tex])

    # ----------------------------------------------------------
    def wait_for_response(self):
        """
        Block until left (→ chose 1) or right (→ chose 2) arrow pressed.
        Shows instruction text on screen while waiting.
        Returns (interval_chosen, esc_pressed).
        interval_chosen: 1 or 2
        """
        LINE1 = "Please choose:"
        LINE2 = "\u25c4  first interval is the signal"
        LINE3 = "\u25ba  second interval is the signal"
        bg = (0.25, 0.25, 0.25)

        while True:
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                return None, True
            if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
                while glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
                    glfw.poll_events()
                return 1, False
            if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
                while glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
                    glfw.poll_events()
                return 2, False

            # Draw grey background
            glUseProgram(self._flat_shader)
            glUniform3f(glGetUniformLocation(self._flat_shader, "bg_color"), *bg)
            glClear(GL_COLOR_BUFFER_BIT)
            draw_quad(self._vao)

            # Draw three lines of text
            self.draw_text_overlay(LINE1, y_ndc= 0.10, size_px=36, bg=bg)
            self.draw_text_overlay(LINE2, y_ndc=-0.05, size_px=36, bg=bg)
            self.draw_text_overlay(LINE3, y_ndc=-0.20, size_px=36, bg=bg)

            glfw.swap_buffers(self.window)
            time.sleep(0.005)

    # ----------------------------------------------------------
    def show_progress_screen(self, conditions_done, conditions_total,
                             elapsed_sec, duration_auto=3.0):
        """
        Show progress bar + ETA text on screen for `duration_auto` seconds.
        Returns True if ESC pressed.
        """
        t0 = time.perf_counter()
        bg = (0.1, 0.1, 0.1)

        frac = conditions_done / max(conditions_total, 1)
        if elapsed_sec > 0 and conditions_done > 0:
            eta_sec = elapsed_sec / conditions_done * (conditions_total - conditions_done)
        else:
            eta_sec = 0

        elapsed_min = int(elapsed_sec // 60)
        elapsed_s   = int(elapsed_sec  % 60)
        eta_min     = int(eta_sec // 60)
        eta_s       = int(eta_sec  % 60)

        line1 = f"Condition {conditions_done} of {conditions_total}  ({frac*100:.0f}%)"
        line2 = f"Elapsed: {elapsed_min}m {elapsed_s:02d}s    ETA: {eta_min}m {eta_s:02d}s"

        print(f"\n  {line1}")
        print(f"  {line2}\n")

        while time.perf_counter() - t0 < duration_auto:
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                return True

            glClear(GL_COLOR_BUFFER_BIT)

            # Dark background
            glUseProgram(self._flat_shader)
            glUniform3f(glGetUniformLocation(self._flat_shader, "bg_color"), *bg)
            draw_quad(self._vao)

            # Progress bar via glScissor
            glEnable(GL_SCISSOR_TEST)
            bar_h = max(1, int(self.height * 0.04))
            bar_y = int(self.height * 0.44)
            bar_full_w = int(self.width * 0.7)
            bar_x = int(self.width * 0.15)

            glScissor(bar_x, bar_y, bar_full_w, bar_h)
            glUniform3f(glGetUniformLocation(self._flat_shader, "bg_color"), 0.3, 0.3, 0.3)
            draw_quad(self._vao)

            filled_w = int(bar_full_w * frac)
            if filled_w > 0:
                glScissor(bar_x, bar_y, filled_w, bar_h)
                glUniform3f(glGetUniformLocation(self._flat_shader, "bg_color"), 0.1, 0.75, 0.25)
                draw_quad(self._vao)
            glDisable(GL_SCISSOR_TEST)

            # Freetype text lines
            self.draw_text_overlay(line1, y_ndc= 0.20, size_px=36, bg=bg)
            self.draw_text_overlay(line2, y_ndc= 0.05, size_px=30, bg=bg)

            glfw.swap_buffers(self.window)
            time.sleep(1.0 / self.refresh)

        return False

    # ----------------------------------------------------------
    def destroy(self):
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None


# ==========================================================
# Quest wrapper
# ==========================================================

def make_quest(start_val, min_val=MIN_DIST, max_val=MAX_DIST, n_trials=TRIALS_PER_CONDITION):
    """
    QuestHandler tracking distance threshold (linear domain).
    Target: 75% correct in 2AFC (chance=0.5).

    Distance↑ = harder, but Quest assumes intensity↑ = easier.
    We work in NEGATED distance space: Quest intensity = -distance_m.
    Correct response is passed directly (1=correct, 0=incorrect).
    When answer is CORRECT at distance d, Quest pushes to harder (farther → more negative).
    When answer is WRONG,  Quest pulls back to easier (closer → less negative).
    """
    quest = QuestHandler(
        startVal=-start_val,                    # negated: Quest works in -distance
        startValSd=(max_val - min_val) / 4.0,
        pThreshold=0.75,
        nTrials=n_trials,
        beta=3.5,
        delta=0.01,
        gamma=0.5,                              # chance level for 2AFC
        grain=0.001,
        range=max_val - min_val,
        minVal=-max_val,
        maxVal=-min_val,
        method='quantile'
    )
    return quest


def quest_suggest(quest):
    """Return next suggested distance in metres (positive)."""
    return float(np.clip(-quest._questNextIntensity, MIN_DIST, MAX_DIST))


def quest_mean(quest):
    """Return Quest mean estimate in metres (positive)."""
    return float(np.clip(-quest.mean(), MIN_DIST, MAX_DIST))


def quest_sd(quest):
    return float(quest.sd())


# ==========================================================
# CSV output
# ==========================================================

def init_quest_csv(csv_path):
    exists = os.path.exists(csv_path)
    empty  = (not exists) or os.stat(csv_path).st_size == 0
    fh = open(csv_path, 'a', newline='')
    writer = csv.writer(fh)
    if empty:
        writer.writerow([
            "name",
            "color",
            "speed_px_per_sec",
            "contrast",
            "mean_luminance",
            "spatial_frequency_cpp",
            "diagonal_inch",
            "visual_radius_deg",
            "trial_index",
            "distance_m_tested",
            "test_interval",       # 1 or 2
            "observer_response",   # 1 or 2
            "correct",             # 1/0
            "quest_estimate_m",
            "retinal_spatial_freq_cpd",
            "temporal_freq_hz",
        ])
    return fh, writer


# ==========================================================
# Main experiment
# ==========================================================

def run_one_trial(renderer, platform, condition, distance_m, stimulus_duration):
    """
    Run a single 2AFC trial.

    Returns (correct: bool, esc: bool)
      correct = True  if observer correctly identified test interval
    """
    color = condition['color']
    speed = condition['speed']
    contrast = condition['contrast']
    luminance = condition['luminance']
    sf_cpp = condition['spatial_frequency']

    # Clamp to physical limits
    distance_m = float(np.clip(distance_m, MIN_DIST, MAX_DIST))

    # Move platform
    platform.move_to(distance_m)
    renderer.set_condition(color, luminance)

    # Pseudorandom interval order: balanced across calls, here just random per trial
    # (caller can track balance if needed; for Quest this is trial-wise random)
    test_interval = random.choice([1, 2])
    ref_interval  = 3 - test_interval  # the other one

    # ---- Interval 1 cue ----
    esc = renderer.show_interval_cue(1, duration=0.5)
    if esc: return False, True

    # ---- Interval 1 stimulus ----
    c1 = contrast if test_interval == 1 else 0.0
    esc = renderer.show_gabor(c1, distance_m, sf_cpp, speed, stimulus_duration)
    if esc: return False, True

    # ---- ISI (flat grey) ----
    renderer.show_flat((0.2, 0.2, 0.2))
    time.sleep(0.3)

    # ---- Interval 2 cue ----
    esc = renderer.show_interval_cue(2, duration=0.5)
    if esc: return False, True

    # ---- Interval 2 stimulus ----
    c2 = contrast if test_interval == 2 else 0.0
    esc = renderer.show_gabor(c2, distance_m, sf_cpp, speed, stimulus_duration)
    if esc: return False, True

    # ---- Response screen (wait for left/right) ----
    renderer.show_flat((0.2, 0.2, 0.2))
    chosen, esc = renderer.wait_for_response()
    if esc: return False, True

    correct = (chosen == test_interval)
    return correct, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",                 default='YanchengCai')
    parser.add_argument("--colors",     nargs="+", default=["ach","rg","yv"])
    parser.add_argument("--speeds",     nargs="+", type=float, default=[120, 240, 300])
    parser.add_argument("--luminance_list", nargs="+", type=float, default=[1, 10, 50])
    parser.add_argument("--spatial_frequency_cpp", type=float, default=0.1)
    parser.add_argument("--ach_contrast", type=float, default=1.0)
    parser.add_argument("--rg_contrast",  type=float, default=0.15)
    parser.add_argument("--yv_contrast",  type=float, default=0.92)
    parser.add_argument("--diagonal_inch", type=float, default=27)
    parser.add_argument("--visual_radius_deg", type=float, default=2.0)
    parser.add_argument("--port",       default="/dev/ttyACM0")
    parser.add_argument("--duration",   type=float, default=2.0,
                        help="Duration (seconds) to show each stimulus interval (default 2)")
    parser.add_argument("--monitor_index", type=int, default=1,
                        help="Index of the display monitor (0-based, default 1)")
    args = parser.parse_args()

    # ----------------------------------------------------------
    # Build condition list
    # ----------------------------------------------------------
    conditions = []
    for color in args.colors:
        contrast = getattr(args, f"{color}_contrast")
        for speed in args.speeds:
            for luminance in args.luminance_list:
                conditions.append({
                    "color":             color,
                    "speed":             speed,
                    "contrast":          contrast,
                    "luminance":         luminance,
                    "spatial_frequency": args.spatial_frequency_cpp,
                })
    random.shuffle(conditions)

    # ---- Resume: skip already-completed conditions ----
    completed_keys = load_completed_conditions(args.name)
    if completed_keys:
        before = len(conditions)
        conditions = [
            c for c in conditions
            if (c['color'], c['speed'], c['luminance']) not in completed_keys
        ]
        skipped = before - len(conditions)
        print(f"[RESUME] Skipping {skipped} already-completed condition(s).")

    total_conditions = len(conditions)

    # ----------------------------------------------------------
    # CSV output
    # ----------------------------------------------------------
    fh, writer = init_quest_csv(QUEST_CSV)

    # ----------------------------------------------------------
    # Hardware init
    # ----------------------------------------------------------
    platform = PlatformController(args.port)

    # ----------------------------------------------------------
    # OpenGL / window init
    # ----------------------------------------------------------
    glfw.init()
    renderer = ExperimentRenderer(
        diagonal_inch=args.diagonal_inch,
        visual_radius_deg=args.visual_radius_deg,
        monitor_index=args.monitor_index
    )
    renderer.init_window()

    experiment_start = time.perf_counter()
    aborted = False

    try:
        for cond_idx, cond in enumerate(conditions):
            color    = cond['color']
            speed    = cond['speed']
            contrast = cond['contrast']
            luminance = cond['luminance']
            sf_cpp   = cond['spatial_frequency']

            # ---- Quest start value from MOA ----
            start_dist = load_moa_distance(args.name, color, speed, luminance)
            start_dist = float(np.clip(start_dist, MIN_DIST, MAX_DIST))

            print(f"\n{'='*60}")
            print(f"Condition {cond_idx+1}/{total_conditions}")
            print(f"  Color: {color}  Speed: {speed} px/s  "
                  f"Luminance: {luminance} cd/m²")
            print(f"  Contrast: {contrast}  "
                  f"Quest start dist: {start_dist:.3f} m")
            print(f"{'='*60}")

            quest = make_quest(start_dist)

            # Pseudorandom balance tracker for test_interval order
            # We ensure across 20 trials that test appears ~10× in each slot.
            # We pre-generate a balanced sequence and use it.
            half = TRIALS_PER_CONDITION // 2
            interval_sequence = [1] * half + [2] * half
            random.shuffle(interval_sequence)
            seq_idx = 0

            trial_idx = 0
            prev_dist_suggested = None   # for convergence early-stop
            last_step_mm = 0.0              # step size shown in terminal
            converged = False
            for trial_idx in range(TRIALS_PER_CONDITION):
                # Quest-suggested distance
                dist_suggested = quest_suggest(quest)

                # ---- Convergence early-stop ----
                if prev_dist_suggested is not None:
                    step = abs(dist_suggested - prev_dist_suggested)
                    last_step_mm = step * 1000
                    if step < QUEST_CONVERGE_THRESH:
                        print(f"  [CONVERGE] Quest step {step*1000:.2f} mm < "
                              f"{QUEST_CONVERGE_THRESH*1000:.0f} mm threshold "
                              f"at trial {trial_idx+1} — stopping early.")
                        converged = True
                        break
                prev_dist_suggested = dist_suggested

                # Use balanced interval from pre-generated sequence
                test_interval = interval_sequence[seq_idx % len(interval_sequence)]
                seq_idx += 1

                # -- Move platform --
                platform.move_to(dist_suggested)
                renderer.set_condition(color, luminance)

                # ---- Interval 1 cue ----
                esc = renderer.show_interval_cue(1, duration=0.5)
                if esc:
                    aborted = True; break

                # ---- Interval 1 stimulus ----
                c1 = contrast if test_interval == 1 else 0.0
                esc = renderer.show_gabor(c1, dist_suggested, sf_cpp, speed, args.duration)
                if esc:
                    aborted = True; break

                # ---- ISI ----
                renderer.show_flat((0.2, 0.2, 0.2))
                time.sleep(0.3)

                # ---- Interval 2 cue ----
                esc = renderer.show_interval_cue(2, duration=0.5)
                if esc:
                    aborted = True; break

                # ---- Interval 2 stimulus ----
                c2 = contrast if test_interval == 2 else 0.0
                esc = renderer.show_gabor(c2, dist_suggested, sf_cpp, speed, args.duration)
                if esc:
                    aborted = True; break

                # ---- Response ----
                renderer.show_flat((0.2, 0.2, 0.2))
                chosen, esc = renderer.wait_for_response()
                if esc:
                    aborted = True; break

                correct = int(chosen == test_interval)

                # ---- Audio feedback ----
                if correct:
                    _beep(BEEP_CORRECT)
                else:
                    _beep(BEEP_INCORRECT)

                # ---- Update Quest ----
                quest.addResponse(correct, intensity=-dist_suggested)  # negated distance: Quest works in -dist space

                # ---- Compute retinal frequencies for logging ----
                W_m = renderer.W
                rho_cpd, omega, theta = compute_spatiotemporal_frequency(
                    renderer.width, W_m, dist_suggested, sf_cpp, speed)

                # ---- Log to CSV ----
                writer.writerow([
                    args.name,
                    color,
                    speed,
                    contrast,
                    luminance,
                    sf_cpp,
                    args.diagonal_inch,
                    args.visual_radius_deg,
                    trial_idx,
                    round(dist_suggested, 4),
                    test_interval,
                    chosen,
                    correct,
                    round(quest_mean(quest), 4),
                    round(rho_cpd, 4),
                    round(omega, 4),
                ])
                fh.flush()

                print(f"  Trial {trial_idx+1:2d}: dist={dist_suggested:.3f}m  "
                      f"test_int={test_interval}  chosen={chosen}  "
                      f"correct={bool(correct)}  "
                      f"quest_est={quest_mean(quest):.3f}m  "
                      f"step={last_step_mm:.1f}mm")

            if aborted:
                print("\n[ESC] Experiment aborted by user.")
                break

            if converged:
                print(f"  Quest converged after {trial_idx} trials "
                      f"(estimate={quest_mean(quest):.4f} m, SD={quest_sd(quest):.4f} m)")

            # ---- Condition complete: show progress ----
            elapsed = time.perf_counter() - experiment_start
            esc = renderer.show_progress_screen(
                conditions_done=cond_idx + 1,
                conditions_total=total_conditions,
                elapsed_sec=elapsed,
                duration_auto=3.0
            )
            if esc:
                aborted = True
                print("\n[ESC] Experiment aborted at progress screen.")
                break

            print(f"\n  Quest final estimate: {quest_mean(quest):.4f} m  "
                  f"(SD={quest_sd(quest):.4f} m)")

    finally:
        platform.cleanup()
        fh.close()
        renderer.destroy()
        glfw.terminate()
        print("\nExperiment finished safely.")
        if aborted:
            print("(Session was aborted early — partial data saved to CSV.)")


if __name__ == "__main__":
    main()