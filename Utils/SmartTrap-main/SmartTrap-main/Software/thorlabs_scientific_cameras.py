"""
Wrapper functions for using Thorlabs Scientific cameras with the SmartTrap system.
------------------------------------------------------
Classes

- ThorlabsScientificCamera: Implementation of the CameraProtocol compatible with the thorlabs 
scientific camera.
"""

import numpy as np
import os
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE
from camera_controls import CameraProtocol
import thorlabs_tsi_sdk.tl_camera

from time import sleep
import os
import sys


def configure_path():
    """
    Configures the path to the Thorlab dlls to simplify imports.
    """

    is_64bits = sys.maxsize > 2**32
    relative_path_to_dlls = '.' + os.sep + 'dlls' + os.sep

    if is_64bits:
        relative_path_to_dlls += '64_lib'
    else:
        relative_path_to_dlls += '32_lib'

    # Check if __file__ is defined, else use the current working directory
    try:
        absolute_path_to_file_directory = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        absolute_path_to_file_directory = os.getcwd()

    absolute_path_to_dlls = os.path.abspath(
        absolute_path_to_file_directory + os.sep + relative_path_to_dlls)
    
    os.environ['PATH'] = absolute_path_to_dlls + os.pathsep + os.environ['PATH']

    try:
        os.add_dll_directory(absolute_path_to_dlls)
        print("Path added successfully")
    except AttributeError:
        print(f"Failed to add path{absolute_path_to_dlls}")
        pass


class ThorlabsScientificCamera(CameraProtocol):
    """
    Implementation of the CameraProtocol compatible with Thorlabs Scientific cameras.
    Requires that the Thorlabs Scientific camera software is installed.
    """
    def __init__(self):
        configure_path()
        self.capturing = False
        self.is_grabbing = False
        self.sdk = TLCameraSDK()  # Initialize the SDK
        self.camera = None
        self.exposure_time = 1000

    def connect_camera(self):
        try:
            available_cameras = self.sdk.discover_available_cameras()
            if len(available_cameras) > 0:
                self.camera = self.sdk.open_camera(available_cameras[0])
                self.camera.is_frame_rate_control_enabled = False
                # Set unlimited frame grabbing
                self.camera.frames_per_trigger_zero_for_unlimited = 0  
                self.camera.arm(2)  # Arm the camera for capture
                sleep(0.2)
                print("Thorlabs camera connected")
                return True
            else:
                print("No cameras found!")
                return False
        except Exception as ex:
            print(f"Failed to connect to camera: {ex}")
            return False

    def disconnect_camera(self):
        if self.camera is not None:
            try:
                self.camera.disarm()  # Disarm before closing
                self.camera.close()
                self.camera = None
            except Exception as ex:
                print(f"Error disconnecting camera: {ex}")

    def capture_image(self):
        if not self.is_grabbing:
            # Set default exposure time (modify as needed)
            self.camera.exposure_time_us = self.exposure_time 
            self.camera.issue_software_trigger()
            self.is_grabbing = True

        try:
            frame = self.camera.get_pending_frame_or_null() # Get the latest frame
            if frame is not None:
                # Return the captured image as a NumPy array
                return np.copy(frame.image_buffer).astype(np.uint8) 
            else:
                print("No frame available!")
                return None
        except Exception as ex:
            print(f"Error capturing image: {ex}")
            self.reconnect_camera()

    def stop_grabbing(self):
        try:
            self.is_grabbing = False
        except Exception as ex:
            print(f"Error stopping grabbing: {ex}")

    def set_frame_rate(self, frame_rate):
        try:
            self.camera.frame_rate_fps = frame_rate
            print(f"Frame rate set to {frame_rate} FPS")
        except Exception as ex:
            print(f"Frame rate not accepted by camera: {ex}")

    def set_gain(self, gain):
        try:
            self.camera.gain = int(gain)
            print(f"Gain set to {gain}")
        except Exception as ex:
            print(f"Gain not accepted by camera: {ex}")

    def set_AOI(self, AOI):
        try:
            self.stop_grabbing()
            self.camera.disarm()
            ROI = thorlabs_tsi_sdk.tl_camera.ROI(AOI[0],AOI[2],AOI[1],AOI[3])
            self.camera.roi = ROI
            self.camera.arm(2)
            print(f"AOI set to {AOI}")
            AOI[0] = self.camera.roi.upper_left_x_pixels
            AOI[1] = self.camera.roi.lower_right_x_pixels+1
            AOI[2] = self.camera.roi.upper_left_y_pixels
            AOI[3] = self.camera.roi.lower_right_y_pixels+1
            return AOI
        except Exception as ex:
            print(f"AOI not accepted by camera: {ex}")
            return (0,0,0,0)

    def set_exposure_time(self, exposure_time):
        try:
            self.camera.exposure_time_us = int(exposure_time)
            self.exposure_time = int(exposure_time)
            print(f"Exposure time set to {exposure_time} microseconds")
        except Exception as ex:
            print(f"Exposure time not accepted by camera: {ex}")

    def get_exposure_time(self):
        try:
            return self.camera.exposure_time_us
        except Exception as ex:
            print(f"Error getting exposure time: {ex}")
            return None

    def get_fps(self):
        try:
            return self.camera.get_measured_frame_rate_fps() 
        except Exception as ex:
            print(f"Error getting FPS: {ex}")
            return None

    def get_sensor_size(self):
        try:
            width = int(
                self.camera.roi_range.lower_right_x_pixels_max \
                    - self.camera.roi_range.upper_left_x_pixels_min +1)
            height = int(
                self.camera.roi_range.lower_right_y_pixels_max \
                    - self.camera.roi_range.upper_left_y_pixels_min +1)
            return width, height
        except Exception as ex:
            print(f"Error getting sensor size: {ex}")
            return None

    def reconnect_camera(self):
        print("Reconnecting to the camera...")
        self.disconnect_camera()
        sleep(0.5)
        self.connect_camera()
