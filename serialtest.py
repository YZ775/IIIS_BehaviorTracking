import serial
import serial.tools.list_ports
print(serial.tools.list_ports.comports()[3])
ser = serial.Serial('/dev/cu.usbmodem143203', 9600)
ser.write(b"0.9")
ser.write(b'\n')

ser.close()