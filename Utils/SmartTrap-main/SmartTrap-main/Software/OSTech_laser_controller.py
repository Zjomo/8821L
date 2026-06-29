"""
Implementation of the LaserController class compatible with laser controllers from OSTech.
"""

import serial
from laser_controller import LaserController

class OSTechLaserController(LaserController):
    def __init__(self):
        self.output_on = False
        self.current = 260  # mA
        self.laser_ser = None

    def connect(self, adress):
        """
        Connect to the laser at the given adress.
        """
        if self.laser_ser is not None:
            print("Laser already connected")
            return

        print("Trying to connect laser")
        try:
            self.laser_ser = serial.Serial(
                port=adress,  # Port to connect to
                baudrate=9600,  # Baud rate
                bytesize=serial.EIGHTBITS,  # Data bits
                parity=serial.PARITY_NONE,  # Parity
                stopbits=serial.STOPBITS_ONE,  # Stop bits
                timeout=2,  # Read timeout in seconds
                xonxoff=False,  # Software flow control
                rtscts=False,  # Hardware flow control (RTS/CTS)
                write_timeout=2,
            )
            print("Connected laser at adress "+adress)
        except Exception as E:
            print("Failed to connect laser", E)
            self.laser_ser = None


    def init_laser(self):
        if self.laser_ser is None:
            return
        # Turn on temperature controller
        message = "TCR\r"
        self.laser_ser.write(message.encode('utf-8'))

    def set_current(self, current):
        """
        Sets the current in mA for laser
        """
        if self.laser_ser is None:
            return
        
        if int(current)>400 or int(current)<0:
            return
        self.current = current        
        if self.laser_ser is not None:
            message = "LCT" + str(self.current) + "\r"
            self.laser_ser.write(message.encode('utf-8'))

    def is_connected(self):
        return self.laser_ser is not None

    def is_output_on(self):
        return self.output_on

    def turn_on_output(self):
        if self.laser_ser is not None:
            self.laser_ser.write(b"LR\r")
            self.output_on = True
    
    def turn_off_output(self):
        if self.laser_ser is not None:
            self.laser_ser.write(b"LS\r")
            self.output_on = False

    def disconnect(self):
        if self.laser_ser is not None:
            self.laser_ser.close()    
            self.laser_ser = None
            print("Connection to laser closed")