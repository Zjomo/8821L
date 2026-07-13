import serial
import time

ser = serial.Serial("COM20", 9600, bytesize=8, parity='N', stopbits=1, timeout=1)

print("串口已打开:", ser.is_open)

# 开灯
cmd_on = bytes.fromhex("A0 01 00 A1")
print("发送开灯:", cmd_on)
ser.write(cmd_on)
ser.flush()
time.sleep(2)
'''
# 关灯
cmd_off = bytes.fromhex("A0 01 01 A2")
print("发送关灯:", cmd_off)
ser.write(cmd_off)
ser.flush()
time.sleep(2)
'''
ser.close()
print("串口已关闭")