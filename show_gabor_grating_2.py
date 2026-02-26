import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


# ==========================================================
# Geometry utilities (from Code 2)
# ==========================================================

def compute_display_width(diagonal_inch: float, R_x: int, R_y: int):
    D_m = diagonal_inch * 0.0254
    W = D_m * (R_x / np.sqrt(R_x**2 + R_y**2))
    return W


def compute_spatiotemporal_frequency(
    R_x: int,
    W: float,
    d: float,
    f_p: float,
    v_p: float,
):
    """
    Exact geometric mapping (no small-angle approximation)
    """

    theta = 2 * np.arctan(W / (2 * d))  # radians

    rho_rad = (R_x * f_p) / theta
    rho_cpd = rho_rad * (np.pi / 180)

    omega = f_p * v_p

    return rho_cpd, omega, theta

def visual_radius_deg_to_px(
        visual_radius_deg: float,
        d: float,
        W: float,
        R_x: int
):
    """
    Convert visual radius (deg) to pixel radius.
    Exact geometry. No small-angle approximation.
    """

    phi_rad = np.deg2rad(visual_radius_deg)

    # physical radius (meters)
    R_phys = d * np.tan(phi_rad)

    # convert to pixels
    radius_px = (R_phys / W) * R_x

    return radius_px


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
):

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    monitors = glfw.get_monitors()
    monitor = monitors[monitor_index]
    mode = glfw.get_video_mode(monitor)

    width = mode.size.width
    height = mode.size.height
    refresh = mode.refresh_rate

    print(f"\nUsing monitor: {width}x{height} @ {refresh} Hz")

    # ==========================================================
    # Exact retinal frequency computation
    # ==========================================================

    W = compute_display_width(diagonal_inch, width, height)

    rho_cpd, omega, theta = compute_spatiotemporal_frequency(
        width,
        W,
        viewing_distance,
        spatial_freq_cpp,
        speed_px_per_sec
    )
    radius_px = visual_radius_deg_to_px(
        visual_radius_deg,
        viewing_distance,
        W,
        width
    )

    print("--------------------------------------------------")
    print(f"Gabor radius: {visual_radius_deg:.2f} deg")
    print(f"Gabor radius: {radius_px:.2f} pixels")
    print(f"Viewing distance: {viewing_distance} m")
    print(f"Display width: {W:.4f} m")
    print(f"Horizontal FOV: {np.degrees(theta):.2f} deg")
    print(f"Spatial frequency: {spatial_freq_cpp:.4f} cycles/pixel")
    print(f"Retinal spatial frequency: {rho_cpd:.3f} cpd")
    print(f"Temporal frequency: {omega:.3f} Hz")
    print("--------------------------------------------------\n")

    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

    window = glfw.create_window(
        width, height,
        "240Hz Moving Gabor",
        monitor,
        None
    )

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # ==========================================================
    # Shader
    # ==========================================================

    vertex_shader = """
    #version 330
    layout(location = 0) in vec2 position;
    out vec2 fragCoord;
    void main() {
        fragCoord = position;
        gl_Position = vec4(position, 0.0, 1.0);
    }
    """

    fragment_shader = f"""
    #version 330
    in vec2 fragCoord;
    out vec4 FragColor;

    uniform float contrast;
    uniform float spatial_freq;
    uniform float phase;
    uniform float mean_lum;
    uniform float peak_lum;
    uniform float screen_width;
    uniform float screen_height;
    uniform float radius;

    const float PI = 3.141592653589793;

    float srgb_encode(float x) {{
        if (x <= 0.0031308)
            return 12.92 * x;
        else
            return 1.055 * pow(x, 1.0/2.4) - 0.055;
    }}

    void main() {{

        float x = (fragCoord.x * 0.5 + 0.5) * screen_width;
        float y = (fragCoord.y * 0.5 + 0.5) * screen_height;

        float cx = screen_width / 2.0;
        float cy = screen_height / 2.0;

        float dx = x - cx;
        float dy = y - cy;

        float sigma = radius;

        float gaussian =
            exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma));

        float carrier =
            cos(2.0 * PI * (spatial_freq * x - phase));

        float gabor = gaussian * carrier;

        float linear_lum =
            mean_lum * (1.0 + contrast * gabor);

        linear_lum = clamp(linear_lum, 0.0, peak_lum);

        float linear01 = linear_lum / peak_lum;
        float srgb = srgb_encode(linear01);

        FragColor = vec4(vec3(srgb), 1.0);
    }}
    """

    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )

    vertices = np.array([
        -1, -1,
         1, -1,
         1,  1,
        -1,  1
    ], dtype=np.float32)

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

    # ==========================================================
    # Phase control
    # ==========================================================

    phase = 0.0
    phase_step = spatial_freq_cpp * speed_px_per_sec / refresh

    while not glfw.window_should_close(window):

        glfw.poll_events()

        # phase += phase_step
        # if phase > 1e6:
        #     phase -= 1e6
        phase = (phase + phase_step) % 1.0

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(shader)

        glUniform1f(glGetUniformLocation(shader, "contrast"), contrast)
        glUniform1f(glGetUniformLocation(shader, "spatial_freq"), spatial_freq_cpp)
        glUniform1f(glGetUniformLocation(shader, "phase"), phase)
        glUniform1f(glGetUniformLocation(shader, "mean_lum"), mean_luminance)
        glUniform1f(glGetUniformLocation(shader, "peak_lum"), peak_luminance)
        glUniform1f(glGetUniformLocation(shader, "screen_width"), float(width))
        glUniform1f(glGetUniformLocation(shader, "screen_height"), float(height))
        glUniform1f(glGetUniformLocation(shader, "radius"), float(radius_px))

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        # glFinish()
        glfw.swap_buffers(window)

    glfw.terminate()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    show_moving_gabor_240hz(
        contrast=0.5,
        spatial_freq_cpp=0.1,
        speed_px_per_sec=100.0,
        visual_radius_deg=5.0,
        mean_luminance=100.0,
        peak_luminance=400.0,
        diagonal_inch=27,     # 改这里
        viewing_distance=1  # 改这里
    )
