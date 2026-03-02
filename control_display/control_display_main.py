import serial
import time
from serial.tools import list_ports

def get_arduino_rsbg_serial_port():
    ports = list_ports.comports()
    for p in ports:
        print(f"Info: Checking port {p.device}")
        try:
            sp = serial.Serial(p.device, 9600, timeout=10)
            time.sleep(3)
            sp.write(b'i')
            aid = sp.read(11).decode(errors='ignore')

            if aid == "arduino_gan":
                sp.close()
                return p.device

            sp.close()
        except Exception:
            pass

    return None


def rsbg_init(port):
    s = None
    try:
        # 1. 尝试打开端口
        s = serial.Serial(port, 9600, timeout=2)  # 缩短 timeout 用于快速检测
        # 2. 关键：等待 Arduino 重启完成 (2-3秒)
        print(f"Connecting to {port}, waiting for reset...")
        time.sleep(2)

        # 3. 清空缓冲区，防止读取到重启时的乱码
        s.reset_input_buffer()

        # 4. 尝试握手
        s.write(b'i')
        device_id = s.read(11).decode(errors='ignore')

        if device_id == "arduino_gan":
            print("Connected successfully!")
            s.write(b'c')  # 发送 calibrate 或其他指令
            return s
        else:
            print(f"Device ID mismatch: Got '{device_id}'")
            s.close()
    except Exception as e:
        print(f"Error opening port: {e}")
        if s: s.close()

    # 如果指定端口失败，再去遍历
    print("Searching for the correct port...")
    # ... 调用 get_arduino_rsbg_serial_port ...

# def rsbg_init(port):
#     """
#     Opens a connection to serial port and verifies arduino_gan
#     """
#     is_correct_port = False
#     s = None
#
#     while not is_correct_port:
#         try:
#             s = serial.Serial(port, 9600, timeout=10)
#             time.sleep(3)
#
#             s.write(b'i')
#             id_bytes = s.read(11)
#             device_id = id_bytes.decode(errors='ignore')
#
#         except Exception:
#             device_id = ''
#
#         if not device_id or device_id != "arduino_gan":
#             if s:
#                 s.close()
#
#             print("INFO: Wrong port detected. Finding arduino_gan port.")
#
#             rsbg_port = get_arduino_rsbg_serial_port()
#
#             if rsbg_port:
#                 port = rsbg_port
#                 s = serial.Serial(port, 9600, timeout=10)
#                 time.sleep(3)
#                 s.write(b'c')
#             else:
#                 print("ERROR: Arduino did not respond, check if it is connected.")
#                 break
#         else:
#             is_correct_port = True
#             s.write(b'c')
#
#     return s

def rsbg_update(s, command, val=None):
    """
    Updates the state of gantry
    """

    if command == 'initialize':
        s.write(f"m 50".encode())

    elif command == 'reset':
        s.write(b'r')
        time.sleep(2)
        print("Reset signal sent and waited 2s.")

    elif command == 'calibrate':
        s.write(b'c')

    elif command == 'set_velocity':
        s.write(f"v {val}".encode())

    elif command == 'move':
        s.write(f"m {val}".encode())

    elif command == 'move_and_wait':
        s.write(f"m {val}".encode())
        s.timeout = 60
        flag_str = s.read(3).decode(errors='ignore')
        return flag_str

    elif command == 'goto':
        s.write(f"a {val}".encode())

    elif command == 'move_continuous_far':
        s.write(b'+')

    elif command == 'move_continuous_near':
        s.write(b'-')

    elif command == 'stop':
        s.write(b's')

def rsbg_cleanup(s):
    s.close()

if __name__ == "__main__":

    arduino_port = rsbg_init('/dev/ttyACM0')

    rsbg_update(arduino_port, 'reset')
    rsbg_update(arduino_port, 'move_and_wait', -20) # 100 -> 160 cm
    rsbg_update(arduino_port, 'move_and_wait', 0) # 注意每个都是绝对值，即相对于初始状态的运动

    rsbg_cleanup(arduino_port)