import os
import sys
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import argparse


# ==========================================================
# Physical display calibration (Code2)
# ==========================================================

L_MIN = 0.023666
L_MAX = 117.05665780294088


# ==========================================================
# Geometry utilities
# ==========================================================

def compute_display_width(diagonal_inch: float, R_x: int, R_y: int):
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
# Color space matrices (与你原始代码一致)
# ==========================================================

def get_color_matrices(mean_luminance, color_direction, dkl_ratios=None):

    if dkl_ratios is None:
        dkl_ratios = np.array([1.0, 1.0, 1.0])

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

    if color_direction == 'ach':
        col_dir = np.array([dkl_ratios[0], 0, 0])
    elif color_direction == 'rg':
        col_dir = np.array([0, dkl_ratios[1], 0])
    elif color_direction == 'yv':
        col_dir = np.array([0, 0, dkl_ratios[2]])
    else:
        raise ValueError("color_direction must be ach/rg/yv")

    return M_combined, dkl_bg, col_dir


# ==========================================================
# Main Gabor
# ==========================================================

def show_moving_gabor_240hz(
        contrast,
        spatial_freq_cpp,
        speed_px_per_sec,
        visual_radius_deg,
        mean_luminance,
        diagonal_inch,
        viewing_distance,
        monitor_index,
        color_direction):

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
    W = compute_display_width(diagonal_inch, width, height)
    rho_cpd, omega, theta = compute_spatiotemporal_frequency(
        width, W, viewing_distance, spatial_freq_cpp, speed_px_per_sec)

    radius_px = visual_radius_deg_to_px(
        visual_radius_deg, viewing_distance, W, width)
    print(f"Retinal spatial frequency : {rho_cpd:.3f} cpd")
    print(f"Temporal frequency        : {omega:.3f} Hz")
    print(f"Gabor radius              : {radius_px:.1f} px\n")

    M_combined, dkl_bg, col_dir = get_color_matrices(
        mean_luminance, color_direction)
    bg_rgb = dkl_bg @ M_combined.T
    print(f"Background linear RGB (should be ~{mean_luminance}): {bg_rgb}")

    window = glfw.create_window(width, height,
                                "Physical Gabor",
                                monitor, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # ======================================================
    # Shader
    # ======================================================

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
    in vec2 fragCoord;
    out vec4 FragColor;

    uniform float contrast;
    uniform float mean_lum;
    uniform float spatial_freq;
    uniform float phase;
    uniform float screen_width;
    uniform float screen_height;
    uniform float radius;

    uniform float L_min;
    uniform float L_max;

    uniform vec3 dkl_bg;
    uniform vec3 col_dir;
    uniform mat3 M_dkl2rgb;

    const float PI = 3.141592653589793;

    float srgb_inverse_eotf(float x) {
        if (x <= 0.0031308)
            return 12.92 * x;
        else
            return 1.055 * pow(x, 1.0/2.4) - 0.055;
    }

    vec3 srgb_inverse_eotf3(vec3 c) {
        return vec3(
            srgb_inverse_eotf(c.r),
            srgb_inverse_eotf(c.g),
            srgb_inverse_eotf(c.b)
        );
    }

    void main() {

        float x = (fragCoord.x * 0.5 + 0.5) * screen_width;
        float y = (fragCoord.y * 0.5 + 0.5) * screen_height;

        float cx = screen_width * 0.5;
        float cy = screen_height * 0.5;

        float dx = x - cx;
        float dy = y - cy;

        float gaussian = exp(-(dx*dx + dy*dy) / (2.0 * radius * radius));
        float carrier = cos(2.0 * PI * (spatial_freq * x - phase));
        float gabor = gaussian * carrier;

        float modulation = gabor * contrast * mean_lum;
        vec3 dkl_pixel = dkl_bg + modulation * col_dir;

        vec3 lin_rgb = M_dkl2rgb * dkl_pixel;   // cd/m²
        
        // ---- Check for negative values before clamping ----
        if (min(lin_rgb.r, min(lin_rgb.g, lin_rgb.b)) < 0.0) {
            // Set output color to red (or any color you prefer) to indicate an error
            FragColor = vec4(1.0, 0.0, 0.0, 1.0);  // red color to indicate an error
            return;  // Exit the fragment shader early to prevent clamping
        }
        
        // -------- Physical luminance → pixel --------

        vec3 linear_norm = (lin_rgb - L_min) / (L_max - L_min);
        linear_norm = clamp(linear_norm, 0.0, 1.0);

        vec3 pixel = srgb_inverse_eotf3(linear_norm);

        FragColor = vec4(pixel, 1.0);
    }
    """

    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )

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

    glUseProgram(shader)

    glUniformMatrix3fv(
        glGetUniformLocation(shader, "M_dkl2rgb"),
        1, GL_FALSE,
        np.array(M_combined.T.flatten(), dtype=np.float32)
    )

    glUniform3f(glGetUniformLocation(shader, "dkl_bg"), *dkl_bg.astype(np.float32))
    glUniform3f(glGetUniformLocation(shader, "col_dir"), *col_dir.astype(np.float32))

    glUniform1f(glGetUniformLocation(shader, "L_min"), L_MIN)
    glUniform1f(glGetUniformLocation(shader, "L_max"), L_MAX)

    phase = 0.0
    phase_step = spatial_freq_cpp * speed_px_per_sec / refresh

    while not glfw.window_should_close(window):
        glfw.poll_events()
        phase = (phase + phase_step) % 1.0

        glClear(GL_COLOR_BUFFER_BIT)

        glUniform1f(glGetUniformLocation(shader, "contrast"), contrast)
        glUniform1f(glGetUniformLocation(shader, "mean_lum"), mean_luminance)
        glUniform1f(glGetUniformLocation(shader, "spatial_freq"), spatial_freq_cpp)
        glUniform1f(glGetUniformLocation(shader, "phase"), phase)
        glUniform1f(glGetUniformLocation(shader, "screen_width"), width)
        glUniform1f(glGetUniformLocation(shader, "screen_height"), height)
        glUniform1f(glGetUniformLocation(shader, "radius"), radius_px)

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glfw.swap_buffers(window)

    glfw.terminate()


# ==========================================================
# CLI (保留全部原始参数，除了 peak_luminance)
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--contrast", type=float, default=1)
    parser.add_argument("--spatial_freq_cpp", type=float, default=0.2)
    parser.add_argument("--speed_px_per_sec", type=float, default=200.0)
    parser.add_argument("--visual_radius_deg", type=float, default=2.0)
    parser.add_argument("--mean_luminance", type=float, default=50.0)
    parser.add_argument("--diagonal_inch", type=float, default=27)
    parser.add_argument("--viewing_distance", type=float, default=1.0)
    parser.add_argument("--monitor_index", type=int, default=1)
    parser.add_argument("--color_direction", type=str, default="ach") # 'ach' [C: 0-1] | 'rg' [C: 0-0.15] | 'yv' [C: 0-0.92]

    args = parser.parse_args()

    show_moving_gabor_240hz(
        contrast=args.contrast,
        spatial_freq_cpp=args.spatial_freq_cpp,
        speed_px_per_sec=args.speed_px_per_sec,
        visual_radius_deg=args.visual_radius_deg,
        mean_luminance=args.mean_luminance,
        diagonal_inch=args.diagonal_inch,
        viewing_distance=args.viewing_distance,
        monitor_index=args.monitor_index,
        color_direction=args.color_direction
    )