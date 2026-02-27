import serial

ser = serial.Serial("/dev/ttyUSB0", 921600)
print("opened")
ser.close()