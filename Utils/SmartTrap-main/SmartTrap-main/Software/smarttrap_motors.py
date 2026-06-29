"""
Implementations of the motor_controls using the default SmartTrap controller.
----------------------------------------------------
Classes:

- SmartTrapMotor: An implementation of the Motor which works with the default motors and electronics
controller of the SmartTrap
- ObjectiveMotor: An ObjectiveMotor interfacing the default arduino based objective stepper motor.
"""

from PyQt6.QtCore import QTimer
import numpy as np
from motor_controls import Motor, ObjectiveMovement
import serial

class SmartTrapMotor(Motor):
    """
    Class which handles the motor control for the minitweezers setup.
    Uses the control parameters and data channels to set the motor speeds and positions.
    Implements the MotorInterface abstract base class.    
    """
    def __init__(self, c_p, data_channels):
        self.c_p = c_p
        self.data_channels = data_channels
        self.move_to_location_timer = QTimer()
        self.move_to_location_timer.timeout.connect(self.move_to_location_check)
        self.move_to_location_timer.start(100) # Check every 100 ms if we have reached the target position.
        self.convert_speed_factor = 182 # Factor to convert microns/s to ticks/s. 1 micron = 182 ticks.

    def set_speed(self, speed):
        # Sets the move to location speed
        if np.abs(float(speed)) < 100:
            speed = float(speed)
        else:
            speed = float(100)
        if speed >= 0.5: # Very small speeds do not work well.
            # Convert to appropriate target speed for the minitweeers
            self.c_p['minitweezers_goto_speed'] = int(speed*self.convert_speed_factor)
        else:
            self.c_p['minitweezers_goto_speed'] = int(0.5*self.convert_speed_factor)
    
    def get_speed(self):
        return float(self.c_p['minitweezers_goto_speed']/self.convert_speed_factor)

    def move_to_location(self, position):
        
        if self.c_p['move_to_location']:
            # Stop if already moving.
            self.c_p['move_to_location'] = False
            self.c_p['motor_x_target_speed'] = 0
            self.c_p['motor_y_target_speed'] = 0
            self.c_p['motor_z_target_speed'] = 0
            return
        # if self.c_p['minitweezers_connected']:
        self.c_p['minitweezers_target_pos'][0] = int(position[0])
        self.c_p['minitweezers_target_pos'][1] = int(position[1])
        self.c_p['minitweezers_target_pos'][2] = int(position[2])
        self.c_p['move_to_location'] = True

    def move_to_location_check(self):
        dist_x = self.c_p['minitweezers_target_pos'][0] - self.data_channels['Motor_x_pos'].get_data(1)[0]
        dist_y = self.c_p['minitweezers_target_pos'][1] - self.data_channels['Motor_y_pos'].get_data(1)[0]
        dist_z = self.c_p['minitweezers_target_pos'][2] - self.data_channels['Motor_z_pos'].get_data(1)[0]
        if dist_x**2<35 and dist_y**2<35 and dist_z**2<8:
            self.c_p['move_to_location'] = False

    def get_x_position(self):
        return self.data_channels['Motor_x_pos'].get_data(1)[0]
    
    def get_y_position(self):
        return self.data_channels['Motor_y_pos'].get_data(1)[0]
    
    def get_z_position(self):
        return self.data_channels['Motor_z_pos'].get_data(1)[0]

    def stop_moving(self):
        # Stop if already moving.
        self.c_p['move_to_location'] = False
        self.c_p['motor_x_target_speed'] = 0
        self.c_p['motor_y_target_speed'] = 0
        self.c_p['motor_z_target_speed'] = 0

    def is_moving(self):
        # This should really be is moving to location.
        return self.c_p['move_to_location']
    
    def limit_speed(self, speed):
        if speed > 32767:
            return 32767
        if speed < -32767:
            return -32767
        return speed
        
    def move_at_speed(self, x_speed, y_speed, z_speed=0):
        self.c_p['motor_x_target_speed'] = int(self.limit_speed(x_speed*self.convert_speed_factor))
        self.c_p['motor_y_target_speed'] = int(self.limit_speed(y_speed*self.convert_speed_factor))
        self.c_p['motor_z_target_speed'] = int(self.limit_speed(z_speed*self.convert_speed_factor))


class ObjectiveMotor(ObjectiveMovement):

    """
    Class which handles the motor control for the objective stepper motor.
    Uses the control parameters and data channels to set the motor speeds and positions.
    Implements the ObjectiveMovement abstract base class.    
    """
    def __init__(self):
        self.arduino_connected = False
        self.last_write = "Q"
    def connect(self, port):
        self.ArduinoUnoSerial = None
        try:
            self.ArduinoUnoSerial = serial.Serial(port, 9600)
            print("Connected to Arduino Uno.")
            self.arduino_connected = True
        except Exception as E:
            print(E)
            print("Could not connect to Arduino Uno for objective stepper control!")

    def slow_towards_sample(self):
        self.last_write = 'Q'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)

    def slow_away_from_sample(self):
        self.last_write = 'W'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)

    def fast_towards_sample(self):
        self.last_write = 'E'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)

    def fast_away_from_sample(self):
        self.last_write = 'R'
        message = self.last_write.encode('utf-8')
        self.ArduinoUnoSerial.write(message)