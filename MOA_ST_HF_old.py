import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from pynput import keyboard
import time
import csv
import threading
import os
import argparse
from control_display.control_display_main import rsbg_init, rsbg_update, rsbg_cleanup
import random

# ==========================================================
# 固定物理参数
# ==========================================================

INITIAL_DISTANCE = 0.5
UNIT_PER_M = 100 / 1.6
STEP = 0.05

MIN_DIST = 0.5
MAX_DIST = 2.1

L_MIN = 0.023666
L_MAX = 117.05665780294088


# ==========================================================
# 几何计算
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
# DKL 颜色矩阵
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
# 键盘监听（不影响渲染线程）
# ==========================================================

class KeyboardController:

    def __init__(self):
        self.up_pressed = False
        self.down_pressed = False
        self.shift_pressed = False
        self.space_pressed = False
        self.escape_pressed = False

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.up_pressed = True
            elif key == keyboard.Key.down:
                self.down_pressed = True
            elif key == keyboard.Key.shift:
                self.shift_pressed = True
            elif key == keyboard.Key.space:
                self.space_pressed = True
            elif key == keyboard.Key.esc:
                self.escape_pressed = True
        except:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.up:
                self.up_pressed = False
            elif key == keyboard.Key.down:
                self.down_pressed = False
            elif key == keyboard.Key.shift:
                self.shift_pressed = False
        except:
            pass

    def stop(self):
        self.listener.stop()
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
        # rsbg_update(self.port, 'move', units)

    def cleanup(self):
        rsbg_update(self.port, 'move_and_wait', 0)
        rsbg_cleanup(self.port)


# ==========================================================
# 渲染线程
# ==========================================================

class RealtimeRenderer(threading.Thread):

    def __init__(self, speed, contrast, color_direction, mean_luminance,
                 spatial_frequency_cpp, diagonal_inch, visual_radius_deg, monitor_index=1):
        super().__init__()
        self.speed = speed
        self.contrast = contrast
        self.color_direction = color_direction
        self.mean_luminance = mean_luminance
        self.spatial_frequency_cpp = spatial_frequency_cpp

        self.running = True
        self.viewing_distance = INITIAL_DISTANCE
        self.monitor_index = monitor_index
        self.diagonal_inch = diagonal_inch
        self.visual_radius_deg = visual_radius_deg

    def update_distance(self, d):
        self.viewing_distance = d

    def stop(self):
        self.running = False

    def run(self):
        monitors = glfw.get_monitors()
        monitor = monitors[self.monitor_index]
        mode = glfw.get_video_mode(monitor)

        width = mode.size.width
        height = mode.size.height
        refresh = mode.refresh_rate

        window = glfw.create_window(width, height,
                                    "Physical Gabor",
                                    monitor, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        W = compute_display_width(self.diagonal_inch, width, height)
        M_combined, dkl_bg, col_dir = get_color_matrices(
            self.mean_luminance,
            self.color_direction
        )

        vertex_shader = """
        #version 330
        layout(location=0) in vec2 position;
        out vec2 fragCoord;
        void main(){
            fragCoord = position;
            gl_Position = vec4(position,0,1);
        }
        """

        fragment_shader = """
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

        shader = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )

        vertices = np.array([-1,-1,1,-1,1,1,-1,1],dtype=np.float32)
        indices = np.array([0,1,2,2,3,0],dtype=np.uint32)

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER,vbo)
        glBufferData(GL_ARRAY_BUFFER,vertices.nbytes,vertices,GL_STATIC_DRAW)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices.nbytes,indices,GL_STATIC_DRAW)

        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,None)
        glEnableVertexAttribArray(0)

        glUseProgram(shader)

        glUniformMatrix3fv(
            glGetUniformLocation(shader,"M_dkl2rgb"),
            1,GL_FALSE,
            np.array(M_combined.T.flatten(),dtype=np.float32)
        )

        glUniform3f(glGetUniformLocation(shader,"dkl_bg"),*dkl_bg)
        glUniform3f(glGetUniformLocation(shader,"col_dir"),*col_dir)
        glUniform1f(glGetUniformLocation(shader,"L_min"),L_MIN)
        glUniform1f(glGetUniformLocation(shader,"L_max"),L_MAX)

        phase = 0

        while not glfw.window_should_close(window) and self.running:

            glfw.poll_events()
            # ---- ESC 强制终止 ----
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                self.running = False
                break
            radius_px = visual_radius_deg_to_px(
                self.visual_radius_deg,
                self.viewing_distance,
                W,
                width
            )

            phase_step = self.spatial_frequency_cpp * self.speed / refresh
            phase = (phase + phase_step) % 1.0

            glClear(GL_COLOR_BUFFER_BIT)

            glUniform1f(glGetUniformLocation(shader,"contrast"),self.contrast)
            glUniform1f(glGetUniformLocation(shader,"mean_lum"),self.mean_luminance)
            glUniform1f(glGetUniformLocation(shader,"spatial_freq"),self.spatial_frequency_cpp)
            glUniform1f(glGetUniformLocation(shader,"phase"),phase)
            glUniform1f(glGetUniformLocation(shader,"screen_width"),width)
            glUniform1f(glGetUniformLocation(shader,"screen_height"),height)
            glUniform1f(glGetUniformLocation(shader,"radius"),radius_px)

            glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,None)
            glfw.swap_buffers(window)


# ==========================================================
# 主实验
# ==========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='YanchengCai')
    parser.add_argument("--colors", nargs="+", default=["ach","rg","yv"])
    parser.add_argument("--speeds", nargs="+", type=float, default=[240]) #, 300, 450, 600])
    parser.add_argument("--luminance_list", nargs="+", type=float, default=[50])
    parser.add_argument("--spatial_frequency_cpp", type=float, default=0.1)
    parser.add_argument("--ach_contrast", type=float, default=1.0)
    parser.add_argument("--rg_contrast", type=float, default=0.15)
    parser.add_argument("--yv_contrast", type=float, default=0.92)
    parser.add_argument("--diagonal_inch", type=float, default=27)
    parser.add_argument("--visual_radius_deg", type=float, default=2.0)
    parser.add_argument("--port", default="/dev/ttyACM0")

    args = parser.parse_args()

    csv_file = "MOA_results.csv"
    file_exists = os.path.exists(csv_file)

    csv_handle = open(csv_file, 'a', newline='')
    writer = csv.writer(csv_handle)

    if not file_exists:
        writer.writerow([
            "name",
            "color",
            "speed_px_per_sec",
            "contrast",
            "mean_luminance",
            "spatial_frequency_cpp",
            "diagonal_inch",
            "visual_radius_deg",
            "resolution_x",
            "resolution_y",
            "refresh_rate",
            "distance_m",
            "retinal_spatial_frequency_cpd",
            "temporal_frequency_hz",
            "theta_deg",
            "gabor_radius_px"
        ])

    platform = PlatformController(args.port)

    try:
        conditions = []
        for color in args.colors:
            contrast = getattr(args, f"{color}_contrast")
            for speed in args.speeds:
                for luminance in args.luminance_list:
                    conditions.append({
                        "color": color,
                        "speed": speed,
                        "contrast": contrast,
                        "luminance": luminance,
                        "spatial_frequency": args.spatial_frequency_cpp
                    })
        random.shuffle(conditions)
        glfw.init()
        for cond in conditions:

            color = cond["color"]
            speed = cond["speed"]
            contrast = cond["contrast"]
            luminance = cond["luminance"]
            spatial_frequency_cpp = cond["spatial_frequency"]

            current_distance = INITIAL_DISTANCE
            platform.move_to(current_distance)

            renderer = RealtimeRenderer(
                speed,
                contrast,
                color,
                luminance,
                spatial_frequency_cpp,
                args.diagonal_inch,
                args.visual_radius_deg,
            )

            renderer.start()

            confirmed = False
            print("\n==============================")
            print(f"Observer              : {args.name}")
            print(f"Color direction       : {color}")
            print(f"Contrast              : {contrast}")
            print(f"Mean luminance        : {luminance} cd/m²")
            print(f"Diagonal size         : {args.diagonal_inch} inch")
            print(f"Speed                 : {speed} pixels/sec")
            print(f"Display spatial freq. : {spatial_frequency_cpp} cycles/pixel")
            last_sent_distance = None

            keyboard_ctrl = KeyboardController()
            while not confirmed:
                if keyboard_ctrl.escape_pressed:
                    renderer.stop()
                    renderer.join()
                    keyboard_ctrl.stop()
                    raise KeyboardInterrupt
                step_size = STEP * 20 if keyboard_ctrl.shift_pressed else STEP
                if keyboard_ctrl.up_pressed:
                    current_distance += step_size
                if keyboard_ctrl.down_pressed:
                    current_distance -= step_size
                current_distance = max(MIN_DIST, min(MAX_DIST, current_distance))
                if keyboard_ctrl.space_pressed:
                    confirmed = True
                if last_sent_distance is None or abs(current_distance - last_sent_distance) > 1e-4:
                    platform.move_to(current_distance)
                    renderer.update_distance(current_distance)
                    last_sent_distance = current_distance
                time.sleep(0.005)  # 200Hz 控制循环

            # ==========================================================
            # 停止渲染
            # ==========================================================

            renderer.stop()
            renderer.join()

            # ==========================================================
            # ======== 使用 Code2 的物理公式重新计算 =========
            # ==========================================================
            monitors = glfw.get_monitors()
            monitor = monitors[1]
            mode = glfw.get_video_mode(monitor)

            width = mode.size.width
            height = mode.size.height
            refresh = mode.refresh_rate

            W = compute_display_width(
                args.diagonal_inch,
                width,
                height
            )

            rho_cpd, omega, theta = compute_spatiotemporal_frequency(
                width,
                W,
                current_distance,
                spatial_frequency_cpp,
                speed
            )

            radius_px = visual_radius_deg_to_px(
                args.visual_radius_deg,
                current_distance,
                W,
                width
            )

            # ==========================================================
            # ================= 终端打印 =================
            # ==========================================================
            print(f"Refresh rate          : {refresh} Hz")
            print(f"Viewing distance      : {current_distance:.3f} m")
            print(f"Resolution            : {width} x {height}")
            print(f"Spatial freq (retina) : {rho_cpd:.3f} cpd")
            print(f"Temporal frequency    : {omega:.3f} Hz")
            print(f"Visual angle theta    : {np.rad2deg(theta):.3f} deg")
            print(f"Gabor radius          : {radius_px:.2f} px")
            print("==============================\n")

            # ==========================================================
            # ================= 写入 CSV =================
            # ==========================================================

            writer.writerow([
                args.name,
                color,
                speed,
                contrast,
                luminance,
                spatial_frequency_cpp,
                args.diagonal_inch,
                args.visual_radius_deg,
                width,
                height,
                refresh,
                round(current_distance, 4),
                round(rho_cpd, 4),
                round(omega, 4),
                round(np.rad2deg(theta), 4),
                round(radius_px, 2)
            ])

            csv_handle.flush()

    finally:
        platform.cleanup()
        csv_handle.close()
        keyboard_ctrl.stop()
        print("Experiment finished safely.")


if __name__ == "__main__":
    main()
    glfw.terminate()