"""
MOA (Method of Adjustment) Experiment
======================================
手动调节观察距离，直到观察者按空格确认。

按键说明：
  ↑ / ↓           调节距离（±0.01 m）
  Shift + ↑/↓     快速调节（±0.20 m）
  Ctrl (hold)      刺激 contrast → 0（松开恢复）
  Space            确认当前距离
  ESC              中止实验

渲染说明：
  渲染循环是 gabor_renderer.py show_gabor() 的逐行复刻，
  只增加了距离控制逻辑。_quad() 保证只渲染中心 800×800 区域。

平台控制说明：
  move_to() 在独立线程中执行，主线程渲染循环不阻塞。
  渲染期间显示的是"目标距离"，平台在后台追赶。
"""

import argparse
import csv
import os
import random
import threading
import time

import glfw
import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT, GL_FALSE,
    GL_TEXTURE0, GL_TEXTURE_BUFFER,
    glActiveTexture, glBindTexture, glClear,
    glUniform1f, glUniform1i, glUniform3f,
    glUniformMatrix3fv, glUseProgram,
)

from gabor_render import (
    ExperimentRenderer,
    compute_spatiotemporal_frequency,
    visual_radius_deg_to_px,
    _quad,
)
from control_display.control_display_main import rsbg_init, rsbg_update, rsbg_cleanup

# ==========================================================
# 常量
# ==========================================================

INITIAL_DISTANCE = 1
UNIT_PER_M       = 100 / 1.6
MIN_DIST         = 1
MAX_DIST         = 2.6
STEP_FINE        = 0.01
STEP_COARSE      = 0.20
MOA_CSV          = "MOA_results.csv"
DEFAULT_REPEATS  = 1


# ==========================================================
# 平台控制（非阻塞版）
# ==========================================================

class PlatformController:
    """
    move_to() 立即返回，实际移动在后台线程执行。
    _moving 标志表示平台正在移动中。
    """
    def __init__(self, port):
        self.port     = rsbg_init(port)
        self._lock    = threading.Lock()
        self._moving  = False
        rsbg_update(self.port, 'reset')
        rsbg_update(self.port, 'move_and_wait', 0)

    @property
    def is_moving(self):
        return self._moving

    def move_to(self, distance_m: float):
        """发起一次移动（非阻塞）。若平台正在移动，跳过本次指令避免指令堆积。"""
        if self._moving:
            return                          # 上一次还没跑完，丢弃新指令
        delta = distance_m - INITIAL_DISTANCE
        units = int(round(-delta * UNIT_PER_M))

        def _worker():
            self._moving = True
            try:
                rsbg_update(self.port, 'move_and_wait', units)
            finally:
                self._moving = False

        threading.Thread(target=_worker, daemon=True).start()

    def move_to_blocking(self, distance_m: float):
        """阻塞版，用于实验开始前的初始定位。"""
        delta = distance_m - INITIAL_DISTANCE
        units = int(round(-delta * UNIT_PER_M))
        rsbg_update(self.port, 'move_and_wait', units)

    def cleanup(self):
        rsbg_update(self.port, 'move_and_wait', 0)
        rsbg_cleanup(self.port)


# ==========================================================
# CSV
# ==========================================================

def init_moa_csv(path: str):
    empty = not os.path.exists(path) or os.stat(path).st_size == 0
    fh = open(path, 'a', newline='')
    w  = csv.writer(fh)
    if empty:
        w.writerow([
            "name", "color", "speed_px_per_sec", "contrast", "mean_luminance",
            "spatial_frequency_cpp", "diagonal_inch", "visual_radius_deg",
            "resolution_x", "resolution_y", "refresh_rate",
            "repeat_index", "distance_m",
            "retinal_spatial_frequency_cpd", "temporal_frequency_hz",
            "theta_deg", "gabor_radius_px",
            "sf_cpp_condition", "speed_px_per_sec_condition",
        ])
    return fh, w


def load_completed_conditions(path: str, name: str) -> set:
    """
    读取已有 CSV，返回已完成条件的集合。
    每条记录用 (color, speed, lum, sf_cpp, rep) 作为唯一键。
    """
    completed = set()
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return completed
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("name") != name:
                continue
            key = (
                row["color"],
                float(row["speed_px_per_sec"]),
                float(row["mean_luminance"]),
                float(row["spatial_frequency_cpp"]),
                int(row["repeat_index"]),
            )
            completed.add(key)
    return completed


# ==========================================================
# 主实验
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--name",                      default="YanchengCai")
    parser.add_argument("--name", default='YifanDing')
    # parser.add_argument("--name", default="Rafal Mantiuk")
    parser.add_argument("--colors",      nargs="+",    default=["ach", "rg", "yv"])
    # parser.add_argument("--ach_speeds", nargs="+", type=float, default=[40, 80, 120, 180])
    # 10, 15, 20, 30, 45
    parser.add_argument("--ach_speeds", nargs="+", type=float, default=[200, 300, 450])
    parser.add_argument("--rg_speeds", nargs="+", type=float, default=[100, 150, 200])
    parser.add_argument("--yv_speeds", nargs="+", type=float, default=[400, 500, 600])
    parser.add_argument("--ach_luminance_list", nargs="+", type=float, default=[1])
    parser.add_argument("--rg_luminance_list", nargs="+", type=float, default=[1])
    parser.add_argument("--yv_luminance_list", nargs="+", type=float, default=[1])
    parser.add_argument("--ach_spatial_frequency_cpp", type=float, default=0.1)
    parser.add_argument("--rg_spatial_frequency_cpp", type=float, default=0.1)
    parser.add_argument("--yv_spatial_frequency_cpp", type=float, default=0.02)
    parser.add_argument("--ach_contrast",              type=float, default=0.9)
    parser.add_argument("--rg_contrast",               type=float, default=0.14)
    parser.add_argument("--yv_contrast",               type=float, default=0.92)
    parser.add_argument("--diagonal_inch",             type=float, default=27)
    parser.add_argument("--visual_radius_deg",         type=float, default=2.0)
    parser.add_argument("--port",                      default="/dev/ttyACM0")
    parser.add_argument("--repeats",                   type=int,   default=DEFAULT_REPEATS)
    parser.add_argument("--monitor_index",             type=int,   default=1)
    args = parser.parse_args()

    SKIP_CONDITIONS = {
        # ("yv", 600, 5),  # (color, speed, lum)
        # ("yv", 200, 50),
        # ("rg", 225, 5),  # 以后想跳过的其他条件也加在这里
    }
    # ---------- 条件列表 ----------
    conditions = []
    for color in args.colors:
        contrast = getattr(args, f"{color}_contrast")
        for speed in getattr(args, f"{color}_speeds"):
            for lum in getattr(args, f"{color}_luminance_list"):
                if (color, speed, lum) in SKIP_CONDITIONS:
                    print(f"  [手动跳过] {color} speed={speed} lum={lum}")
                    continue
                for rep in range(args.repeats):
                    conditions.append(dict(
                        color=color, speed=speed,
                        contrast=contrast, lum=lum,
                        sf_cpp=getattr(args, f"{color}_spatial_frequency_cpp"), rep=rep,
                    ))
    random.shuffle(conditions)

    # ==========================================================
    # 启动时打印各通道空间频率范围
    # ==========================================================
    def _print_cpd_ranges():
        import math
        glfw.init()
        monitors = glfw.get_monitors()
        mon = monitors[args.monitor_index] if args.monitor_index < len(monitors) else monitors[0]
        mode = glfw.get_video_mode(mon)
        res_x, res_y = mode.size.width, mode.size.height
        diag_px = math.sqrt(res_x ** 2 + res_y ** 2)
        W = args.diagonal_inch * 0.0254 * (res_x / diag_px)  # 屏幕物理宽度 (m)
        ppd_per_meter = (math.pi / 180) * (res_x / W)  # PPD / 距离(m)

        print("\n" + "=" * 62)
        print(f"  空间频率范围预览  ({res_x}×{res_y}, {args.diagonal_inch}\" 屏)")
        print(f"  观察距离范围: {MIN_DIST} m ~ {MAX_DIST} m")
        print("-" * 62)
        print(f"  {'通道':<6} {'sf_cpp':>8}  {'近端(cpd)':>10}  {'远端(cpd)':>10}  {'范围':>22}")
        print("-" * 62)
        for color in args.colors:
            sf_cpp = getattr(args, f"{color}_spatial_frequency_cpp")
            cpd_near = sf_cpp * ppd_per_meter * MIN_DIST
            cpd_far = sf_cpp * ppd_per_meter * MAX_DIST
            print(f"  {color:<6} {sf_cpp:>8.3f}  {cpd_near:>10.3f}  {cpd_far:>10.3f}"
                  f"  [{cpd_near:.3f}, {cpd_far:.3f}] cpd")
        print("=" * 62 + "\n")

    _print_cpd_ranges()
    # ==========================================================

    fh, writer = init_moa_csv(MOA_CSV)
    completed  = load_completed_conditions(MOA_CSV, args.name)
    total      = len(conditions)
    skipped    = sum(
        1 for c in conditions
        if (c["color"], c["speed"], c["lum"], c["sf_cpp"], c["rep"]) in completed
    )
    print(f"共 {total} 个条件，CSV 中已完成 {skipped} 个，本次运行 {total - skipped} 个。")
    platform   = PlatformController(args.port)

    glfw.init()
    renderer = ExperimentRenderer(
        diagonal_inch=args.diagonal_inch,
        visual_radius_deg=args.visual_radius_deg,
        monitor_index=args.monitor_index,
        lut_json_path="calibrate_display/Measure_specbos/pixel_luminance_model_pchip_B100_C100_s800_dark.json"
    )
    renderer.init_window()

    # ==========================================================
    # 按键回调：注册一次，整个实验共用
    # ==========================================================
    _pending_keys = []

    def _key_cb(window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            _pending_keys.append((key, mods))

    glfw.set_key_callback(renderer.window, _key_cb)

    experiment_start = time.perf_counter()
    aborted          = False

    try:
        for cond_idx, cond in enumerate(conditions):
            color    = cond["color"]
            speed    = cond["speed"]
            contrast = cond["contrast"]
            lum      = cond["lum"]
            sf_cpp   = cond["sf_cpp"]
            rep      = cond["rep"]

            # 检查是否已有记录，跳过已完成条件
            cond_key = (color, speed, lum, sf_cpp, rep)
            if cond_key in completed:
                print(f"  [跳过] 条件 {cond_idx+1}/{total}  {color} {speed}px/s lum={lum} rep={rep}")
                continue


            # 随机起点消除锚定偏差
            current_dist       = 1.1 #float(np.random.uniform(MIN_DIST, MAX_DIST))
            last_platform_dist = None   # 强制第一帧就发送指令

            print(f"\n{'='*60}")
            print(f"条件 {cond_idx+1}/{total}  重复 {rep+1}/{args.repeats}")
            print(f"  颜色: {color}  速度: {speed} px/s  亮度: {lum} cd/m²")
            print(f"  对比度: {contrast}  起始距离: {current_dist:.3f} m")
            print(f"  ↑/↓ 调节  Shift+↑/↓ 快速  Ctrl=空白  Space 确认  ESC 中止")
            print('='*60)

            # 条件开始前移动到随机起点（阻塞，此时渲染还未开始，不影响体验）
            platform.move_to_blocking(current_dist)
            renderer.set_condition(color, lum)

            # 清除上一个条件残留的按键事件
            _pending_keys.clear()

            # ==============================================================
            # 渲染循环：逐行复刻 show_gabor()，增加距离控制
            # ==============================================================
            sh = renderer._sh_gabor
            glUseProgram(sh)

            # --- 绑定 LUT TBO（与 show_gabor 完全一致）---
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_BUFFER, renderer._lut_tex)
            glUniform1i(renderer._loc(sh, "lut_tex"),  0)
            glUniform1i(renderer._loc(sh, "lut_size"), len(renderer._lut_p))
            glUniform1f(renderer._loc(sh, "L_min"),    renderer._L_min)
            glUniform1f(renderer._loc(sh, "L_max"),    renderer._L_max)
            glUniformMatrix3fv(
                renderer._loc(sh, "M_dkl2rgb"), 1, GL_FALSE,
                renderer._M_combined.T.flatten().astype(np.float32),
            )
            glUniform3f(renderer._loc(sh, "dkl_bg"),      *renderer._dkl_bg)
            glUniform3f(renderer._loc(sh, "col_dir"),      *renderer._col_dir)
            glUniform1f(renderer._loc(sh, "mean_lum"),      lum)
            glUniform1f(renderer._loc(sh, "spatial_freq"),  sf_cpp)
            glUniform1f(renderer._loc(sh, "screen_width"),  renderer.width)
            glUniform1f(renderer._loc(sh, "screen_height"), renderer.height)

            phase      = 0.0
            phase_step = sf_cpp * speed / renderer.refresh
            confirmed  = False

            while not confirmed and not aborted:
                # --- poll_events：触发回调，填充 _pending_keys ---
                glfw.poll_events()

                # --- 消费按键队列 ---
                while _pending_keys:
                    key, mods = _pending_keys.pop(0)
                    shift_held = mods & glfw.MOD_SHIFT
                    step = STEP_COARSE if shift_held else STEP_FINE

                    if key == glfw.KEY_UP:
                        current_dist = min(current_dist + step, MAX_DIST)
                    elif key == glfw.KEY_DOWN:
                        current_dist = max(current_dist - step, MIN_DIST)
                    elif key == glfw.KEY_SPACE:
                        confirmed = True
                    elif key == glfw.KEY_ESCAPE:
                        aborted = True

                # --- Ctrl 按住 → contrast = 0（持续状态，保持轮询）---
                ctrl_held = (
                    glfw.get_key(renderer.window, glfw.KEY_LEFT_CONTROL)  == glfw.PRESS or
                    glfw.get_key(renderer.window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
                )
                display_contrast = 0.0 if ctrl_held else contrast

                # --- 非阻塞平台移动（距离变化且平台空闲时才发指令）---
                if (last_platform_dist is None or
                        abs(current_dist - last_platform_dist) > 1e-4):
                    platform.move_to(current_dist)   # 立即返回，后台线程执行
                    last_platform_dist = current_dist

                # --- 重新绑定 gabor shader（draw_text_overlay 会切走 program）---
                glUseProgram(sh)

                # --- 每帧更新的 uniform ---
                phase = (phase + phase_step) % 1.0
                glUniform1f(renderer._loc(sh, "phase"),    phase)
                glUniform1f(renderer._loc(sh, "contrast"), display_contrast)
                glUniform1f(
                    renderer._loc(sh, "radius"),
                    visual_radius_deg_to_px(
                        renderer.visual_radius_deg,
                        current_dist, renderer.W, renderer.width,
                    ),
                )

                # --- 渲染 ---
                glClear(GL_COLOR_BUFFER_BIT)
                # _quad(renderer._vao)    # scissor 中心 800×800
                _quad(renderer._vao, renderer._bg_size_px(current_dist))

                # # --- 提示文字（多行居中，适配 800×800 显示区域）---
                # moving_tag = " [moving]" if platform.is_moving else ""
                # blank_tag  = " [BLANK]"  if ctrl_held          else ""
                # renderer.draw_text_overlay(
                #     f"Distance: {current_dist:.3f} m{moving_tag}{blank_tag}",
                #     y_ndc=-0.72, size_px=22,
                # )
                # renderer.draw_text_overlay(
                #     "Up/Down: adjust    Shift+Arrow: fast",
                #     y_ndc=-0.82, size_px=20,
                # )
                # renderer.draw_text_overlay(
                #     "Space: confirm    Ctrl: blank    ESC: abort",
                #     y_ndc=-0.91, size_px=20,
                # )

                glfw.swap_buffers(renderer.window)

            # ==============================================================

            if aborted:
                print("\n[ESC] 实验已中止。")
                break

            # ---- 等待平台停稳再记录 ----
            while platform.is_moving:
                time.sleep(0.05)

            # ---- 记录结果 ----
            rho_cpd, omega, theta = compute_spatiotemporal_frequency(
                renderer.width, renderer.W, current_dist, sf_cpp, speed,
            )
            radius_px = visual_radius_deg_to_px(
                renderer.visual_radius_deg, current_dist, renderer.W, renderer.width,
            )
            writer.writerow([
                args.name, color, speed, contrast, lum, sf_cpp,
                args.diagonal_inch, args.visual_radius_deg,
                renderer.width, renderer.height, renderer.refresh,
                rep, round(current_dist, 4),
                round(rho_cpd, 4), round(omega, 4),
                round(float(np.rad2deg(theta)), 4), round(radius_px, 2),
                sf_cpp, speed,
            ])
            fh.flush()

            print(f"  ✓ 距离: {current_dist:.4f} m  "
                  f"rho={rho_cpd:.3f} cpd  omega={omega:.3f} Hz")

            # ---- 进度屏 ----
            elapsed = time.perf_counter() - experiment_start
            esc = renderer.show_progress_screen(
                conditions_done=cond_idx + 1,
                conditions_total=total,
                elapsed_sec=elapsed,
                distance_m=current_dist,
                duration_auto=2.0,
            )
            if esc:
                aborted = True
                print("\n[ESC] 在进度屏中止。")
                break

    finally:
        platform.cleanup()
        fh.close()
        renderer.destroy()
        glfw.terminate()
        print("\n实验已安全结束。")
        if aborted:
            print("（提前中止——部分数据已保存至 MOA_results.csv）")


if __name__ == "__main__":
    main()

#     function
#     cpd_inv = get_cpd_inv_from_ppd(ss, ppd)
#
#     cpd = ppd. / 2; %cpd
#     c_cpd = power(cpd, 1 / 3); %cube
#     root
#     cpd
#     cpd_inv = -1. * c_cpd;
#
# end