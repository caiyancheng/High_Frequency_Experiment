import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

for p in ports:
    print("Device:", p.device)
    print("Description:", p.description)
    print("HWID:", p.hwid)
    print("--------------")