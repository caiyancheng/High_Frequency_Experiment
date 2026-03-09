import os
import sys

# ==========================================================
# Linux: Force NVIDIA GPU (Optimus / PRIME systems)
# ==========================================================
# if sys.platform.startswith("linux"):
#     os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
#     os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


# ==========================================================
# Geometry utilities
# ==========================================================

def compute_display_width(diagonal_inch: float, R_x: int, R_y: int):
    D_m = diagonal_inch * 0.0254
    W = D_m * (R_x / np.sqrt(R_x ** 2 + R_y ** 2))
    return W


def compute_spatiotemporal_frequency(R_x, W, d, f_p, v_p):
    theta = 2 * np.arctan(W / (2 * d))
    rho_rad = (R_x * f_p) / theta
    rho_cpd = rho_rad * (np.pi / 180)
    omega = f_p * v_p
    return rho_cpd, omega, theta


def visual_radius_deg_to_px(visual_radius_deg, d, W, R_x):
    phi_rad = np.deg2rad(visual_radius_deg)
    R_phys = d * np.tan(phi_rad)
    radius_px = (R_phys / W) * R_x
    return radius_px


# ==========================================================
# Color space matrices  (exact match to Color_space_Transform.py)
# ==========================================================

def get_color_matrices(mean_luminance: float, color_direction: str,
                       dkl_ratios=None):
    """
    Returns
    -------
    M_combined : ndarray (3,3)
        Single matrix: DKL -> linear sRGB  (row-vector convention,
        i.e.  rgb = dkl @ M_combined.T )
    dkl_bg : ndarray (3,)
        DKL coordinates of the achromatic background at L_b = mean_luminance.
        This is the pixel value where gabor = 0 (Gaussian envelope → 0).
    col_dir : ndarray (3,)
        DKL modulation direction (e.g. (1,0,0) for ach, (0,1,0) for rg …).
        contrast is defined along this direction, matching Code 2.
    """

    if dkl_ratios is None:
        dkl_ratios = np.array([1.0, 1.0, 1.0])

    # ---------- LMS <-> DKL (Code 3: lms2dkl_d65 / dkl2lms_d65) ----------
    lms_gray = np.array([0.739876529525622,
                         0.320136241543338,
                         0.020793708751515])
    mc1 = lms_gray[0] / lms_gray[1]
    mc2 = (lms_gray[0] + lms_gray[1]) / lms_gray[2]

    M_lms_dkl = np.array([[1, 1, 0],
                          [1, -mc1, 0],
                          [-1, -1, mc2]])
    M_dkl_lms = np.linalg.inv(M_lms_dkl)  # DKL -> LMS

    # ---------- LMS -> XYZ  (Code 3: lms2006_2xyz) ----------
    M_lms_xyz = np.array([
        [2.629129278399650, -3.780202391780134, 10.294956387893450],
        [0.865649062438827, 1.215555811642301, -0.984175688105352],
        [-0.008886561474676, 0.081612628990755, 51.371024830897888]
    ])

    # ---------- XYZ -> linear RGB  (Code 3: cm_xyz2rgb, rec709/sRGB) ----------
    M_xyz_rgb = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    # ---------- Combined: DKL -> linear RGB ----------
    M_combined = M_xyz_rgb @ M_lms_xyz @ M_dkl_lms  # shape (3, 3)

    # ---------- Background DKL at L_b = mean_luminance ----------
    # Code 2:
    #   C_dkl = lms2dkl_d65( xyz2lms2006( white_point_d65 * L_b ) )
    #         = L_b * lms2dkl_d65( xyz2lms2006( white_point_d65 ) )
    #
    # Because both xyz2lms2006 and lms2dkl_d65 are LINEAR transforms,
    # scaling XYZ by L_b scales DKL by the same factor.
    # So  dkl_bg = L_b * dkl_wp  (NOT just dkl_wp !)
    white_point_d65 = np.array([0.9505, 1.0000, 1.0888])

    M_xyz_lms = np.array([
        [0.187596268556126, 0.585168649077728, -0.026384263306304],
        [-0.133397430663221, 0.405505777260049, 0.034502127690364],
        [0.000244379021663, -0.000542995890619, 0.019406849066323]
    ])
    lms_wp = white_point_d65 @ M_xyz_lms.T
    dkl_wp = lms_wp @ M_lms_dkl.T  # normalised white point in DKL

    dkl_bg = mean_luminance * dkl_wp  # ← KEY FIX: scale by L_b

    # ---------- Modulation direction ----------
    # Code 2:
    #   col_dir = dkl_ratios[k] along one axis
    #   modulation = (T_vid_single_channel - L_b) * col_dir
    #              = gaussian * sin(...) * contrast * L_b * col_dir
    #
    # contrast is defined in the fixed colour direction → just pass col_dir as-is.
    if color_direction == 'ach':
        col_dir = np.array([dkl_ratios[0], 0.0, 0.0])
    elif color_direction == 'rg':
        col_dir = np.array([0.0, dkl_ratios[1], 0.0])
    elif color_direction == 'yv':
        col_dir = np.array([0.0, 0.0, dkl_ratios[2]])
    else:
        raise ValueError("color_direction must be 'ach', 'rg', or 'yv'")

    return M_combined, dkl_bg, col_dir


# ==========================================================
# Main Gabor Renderer
# ==========================================================

def show_moving_gabor_240hz(
        contrast=0.9,
        spatial_freq_cpp=0.1,
        speed_px_per_sec=10.0,
        visual_radius_deg=10.0,
        mean_luminance=100.0,
        peak_luminance=400.0,
        diagonal_inch=27,
        viewing_distance=1.0,
        monitor_index=0,
        color_direction='ach',  # 'ach' | 'rg' | 'yv'
        dkl_ratios=None,
):
    """
    color_direction
    ---------------
    'ach'  – achromatic / luminance Gabor  (grey on grey)
    'rg'   – red-green isoluminant Gabor   (background stays grey)
    'yv'   – yellow-violet isoluminant Gabor (background stays grey)

    contrast is defined along the chosen DKL axis, matching Code 2.
    """

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    monitors = glfw.get_monitors()
    monitor = monitors[monitor_index]
    mode = glfw.get_video_mode(monitor)

    width = mode.size.width
    height = mode.size.height
    refresh = mode.refresh_rate

    print(f"\nUsing monitor : {width}x{height} @ {refresh} Hz")
    print(f"Color direction: {color_direction}")

    # ----------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------
    W = compute_display_width(diagonal_inch, width, height)
    rho_cpd, omega, theta = compute_spatiotemporal_frequency(
        width, W, viewing_distance, spatial_freq_cpp, speed_px_per_sec)
    radius_px = visual_radius_deg_to_px(
        visual_radius_deg, viewing_distance, W, width)

    print(f"Retinal spatial frequency : {rho_cpd:.3f} cpd")
    print(f"Temporal frequency        : {omega:.3f} Hz")
    print(f"Gabor radius              : {radius_px:.1f} px\n")

    # ----------------------------------------------------------
    # Color matrices
    # ----------------------------------------------------------
    M_combined, dkl_bg, col_dir = get_color_matrices(
        mean_luminance, color_direction, dkl_ratios)

    # Verify background linear RGB (should be ~mean_luminance for all three channels)
    bg_rgb = dkl_bg @ M_combined.T
    print(f"Background linear RGB (should be ~{mean_luminance}): {bg_rgb}")

    # ----------------------------------------------------------
    # GLFW / OpenGL
    # ----------------------------------------------------------
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
    # window = glfw.create_window(width, height, "240Hz Moving Gabor",
    #                             monitor, None)
    # glfw.make_context_current(window)
    # glfw.swap_interval(1)
    # 创建普通窗口
    window = glfw.create_window(width, height,
                                "240Hz Moving Gabor",
                                monitor, None)
    glfw.make_context_current(window)
    print("GL_VENDOR  :", glGetString(GL_VENDOR).decode())
    print("GL_RENDERER:", glGetString(GL_RENDERER).decode())
    print("GL_VERSION :", glGetString(GL_VERSION).decode())
    glfw.swap_interval(1)

    # -------------------------
    # ESC key to exit
    # -------------------------
    def key_callback(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, key_callback)

    # ----------------------------------------------------------
    # Shader
    # ----------------------------------------------------------

    vertex_shader = """
    #version 330
    layout(location = 0) in vec2 position;
    out vec2 fragCoord;
    void main() {
        fragCoord = position;
        gl_Position = vec4(position, 0.0, 1.0);
    }
    """

    fragment_shader = """
    #version 330
    in  vec2 fragCoord;
    out vec4 FragColor;

    uniform float contrast;
    uniform float mean_lum;        // L_b
    uniform float peak_lum;
    uniform float spatial_freq;
    uniform float phase;
    uniform float screen_width;
    uniform float screen_height;
    uniform float radius;

    uniform vec3 dkl_bg;           // DKL of background = L_b * dkl_wp
    uniform vec3 col_dir;          // DKL modulation direction
    uniform mat3 M_dkl2rgb;        // DKL -> linear sRGB  (column-major GLSL mat3)

    const float PI = 3.141592653589793;

    // sRGB transfer function
    float srgb_encode(float x) {
        if (x <= 0.0031308)
            return 12.92 * x;
        else
            return 1.055 * pow(x, 1.0/2.4) - 0.055;
    }
    vec3 srgb_encode3(vec3 c) {
        return vec3(srgb_encode(c.r), srgb_encode(c.g), srgb_encode(c.b));
    }

    void main() {
        // ---- pixel coordinates ----
        float x  = (fragCoord.x * 0.5 + 0.5) * screen_width;
        float y  = (fragCoord.y * 0.5 + 0.5) * screen_height;
        float cx = screen_width  * 0.5;
        float cy = screen_height * 0.5;
        float dx = x - cx;
        float dy = y - cy;

        // ---- Gabor (gaussian * cosine carrier) ----
        float sigma    = radius;
        float gaussian = exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma));
        float carrier  = cos(2.0 * PI * (spatial_freq * x - phase));
        float gabor    = gaussian * carrier;      // range [-1, +1]

        // ---- DKL modulation ----
        // Matches Code 2:
        //   (T_vid_single_channel - L_b) = gaussian * sinusoid
        //                                = gabor * contrast * L_b
        //   dkl_pixel = dkl_bg + modulation * col_dir
        //
        // contrast is defined along col_dir (fixed colour direction).
        float modulation = gabor * contrast * mean_lum;
        vec3 dkl_pixel   = dkl_bg + modulation * col_dir;

        // ---- DKL -> linear RGB ----
        // M_dkl2rgb is stored column-major (standard GLSL):
        //   lin_rgb = M_dkl2rgb * dkl_pixel
        vec3 lin_rgb = M_dkl2rgb * dkl_pixel;

        // ---- Check for negative values before clamping ----
        if (min(lin_rgb.r, min(lin_rgb.g, lin_rgb.b)) < 0.0) {
            // Set output color to red (or any color you prefer) to indicate an error
            FragColor = vec4(1.0, 0.0, 0.0, 1.0);  // red color to indicate an error
            return;  // Exit the fragment shader early to prevent clamping
        }
        // ---- Normalize by peak luminance & clamp ----
        lin_rgb = clamp(lin_rgb / peak_lum, 0.0, 1.0);

        // ---- sRGB gamma ----
        FragColor = vec4(srgb_encode3(lin_rgb), 1.0);
    }
    """

    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )

    # ----------------------------------------------------------
    # Quad geometry
    # ----------------------------------------------------------
    vertices = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

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

    # ----------------------------------------------------------
    # Upload static uniforms
    # ----------------------------------------------------------
    glUseProgram(shader)

    # mat3 upload:
    #   GLSL  lin_rgb = M * v  means M is column-major.
    #   In NumPy (row-major), M_combined[i,j] = element row i, col j.
    #   glUniformMatrix3fv with transpose=GL_FALSE expects column-major flat array,
    #   which equals M_combined.T.flatten() in NumPy.
    M_col_major = np.array(M_combined.T.flatten(), dtype=np.float32)
    glUniformMatrix3fv(
        glGetUniformLocation(shader, "M_dkl2rgb"),
        1, GL_FALSE, M_col_major
    )

    glUniform3f(glGetUniformLocation(shader, "dkl_bg"),
                float(dkl_bg[0]), float(dkl_bg[1]), float(dkl_bg[2]))
    glUniform3f(glGetUniformLocation(shader, "col_dir"),
                float(col_dir[0]), float(col_dir[1]), float(col_dir[2]))

    # ----------------------------------------------------------
    # Render loop
    # ----------------------------------------------------------
    phase = 0.0
    phase_step = spatial_freq_cpp * speed_px_per_sec / refresh

    while not glfw.window_should_close(window):
        glfw.poll_events()
        phase = (phase + phase_step) % 1.0

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(shader)

        glUniform1f(glGetUniformLocation(shader, "contrast"), contrast)
        glUniform1f(glGetUniformLocation(shader, "mean_lum"), mean_luminance)
        glUniform1f(glGetUniformLocation(shader, "peak_lum"), peak_luminance)
        glUniform1f(glGetUniformLocation(shader, "spatial_freq"), spatial_freq_cpp)
        glUniform1f(glGetUniformLocation(shader, "phase"), phase)
        glUniform1f(glGetUniformLocation(shader, "screen_width"), float(width))
        glUniform1f(glGetUniformLocation(shader, "screen_height"), float(height))
        glUniform1f(glGetUniformLocation(shader, "radius"), float(radius_px))

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glfw.swap_buffers(window)

    glfw.terminate()


# ==========================================================
# MAIN
# ==========================================================

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--contrast", type=float, default=0.15)
    parser.add_argument("--spatial_freq_cpp", type=float, default=0.1)
    parser.add_argument("--speed_px_per_sec", type=float, default=100.0)
    parser.add_argument("--visual_radius_deg", type=float, default=5.0)
    parser.add_argument("--mean_luminance", type=float, default=100.0)
    parser.add_argument("--peak_luminance", type=float, default=400.0)
    parser.add_argument("--diagonal_inch", type=float, default=27)
    parser.add_argument("--viewing_distance", type=float, default=1.0)
    parser.add_argument("--monitor_index", type=int, default=1)
    parser.add_argument("--color_direction", type=str, default="rg") # 'ach' [C: 0-1] | 'rg' [C: 0-0.15] | 'yv' [C: 0-0.92]

    args = parser.parse_args()

    show_moving_gabor_240hz(
        contrast=args.contrast,
        spatial_freq_cpp=args.spatial_freq_cpp,
        speed_px_per_sec=args.speed_px_per_sec,
        visual_radius_deg=args.visual_radius_deg,
        mean_luminance=args.mean_luminance,
        peak_luminance=args.peak_luminance,
        diagonal_inch=args.diagonal_inch,
        viewing_distance=args.viewing_distance,
        monitor_index=args.monitor_index,
        color_direction=args.color_direction
    )
