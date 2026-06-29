"""
Classes:
- TimeoutException: Camera timeout exception
- BaslerCamera: CameraProtocol which controls cameras from Basler
"""

import numpy as np
from camera_controls import CameraProtocol
from pypylon import pylon 
from time import sleep

class TimeoutException(Exception):
    print("Timeout of camera!")

class BaslerCamera(CameraProtocol):
    """
    Wrapper for Basler cameras using the pypylon SDK.

    Provides a high-level interface for connecting, configuring, and capturing
    images from a Basler camera. Supports setting exposure, gain, frame rate,
    and area of interest (AOI), while handling common error cases gracefully.

    Inherits from
    -------------
    CameraProtocol
        Abstract base defining the expected camera interface.

    Attributes
    ----------
    capturing : bool
        Whether the camera is currently capturing (not actively used here).
    img : pylon.PylonImage
        Internal buffer for retrieved images.
    is_grabbing : bool
        Whether the camera has started a grabbing session.
    AOI : list[int]
        Current area of interest, formatted as [x0, x1, y0, y1].

    Methods
    -------
    connect_camera()
        Connect to the first available Basler camera and configure defaults.
    disconnect_camera()
        Safely close and release the camera.
    capture_image()
        Acquire a single image (NumPy array). Starts grabbing if needed.
    stop_grabbing()
        Stop the current grabbing session.
    set_frame_rate(frame_rate)
        Set the acquisition frame rate in Hz.
    set_gain(gain)
        Set the analog gain of the sensor.
    set_AOI(AOI)
        Configure the area of interest (AOI) with alignment to sensor constraints.
    set_exposure_time(exposure_time)
        Set the camera exposure time in microseconds.
    get_exposure_time()
        Return the current exposure time in microseconds.
    get_fps()
        Return the current resulting frame rate (Hz).
    get_sensor_size()
        Return the maximum width and height of the sensor.

    Notes
    -----
    - AOI (area of interest) dimensions and offsets must align to multiples of
      16 pixels. Invalid AOI configurations are automatically corrected.
    - On capture errors, the camera attempts to reconnect automatically.
    - The class assumes only one camera is connected (uses `CreateFirstDevice()`).
    - Timeout errors (default 3000 ms) are caught and printed as warnings.

    Example
    -------
    >>> cam = BaslerCamera()
    >>> if cam.connect_camera():
    ...     frame = cam.capture_image()
    ...     print(frame.shape)
    ...     cam.set_exposure_time(10000)   # 10 ms
    ...     cam.set_frame_rate(50)         # 50 Hz
    ...     cam.disconnect_camera()
    """

    def __init__(self):
        self.capturing = False
        self.img = pylon.PylonImage()
        self.is_grabbing = False        
        self.AOI = [0,2000,0,2000]

    def capture_image(self):
        """
        Captures a single image. Will start continous grabbing with the OneByOne grab strategy
        if this is not already toggled.
        """
        if not self.is_grabbing:
            self.cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self.is_grabbing = True
        try:
            with self.cam.RetrieveResult(3000) as result:
                self.img.AttachGrabResultBuffer(result)
                if result.GrabSucceeded():
                    image = np.uint8(self.img.GetArray())
                    self.img.Release()
                    return image
        except TimeoutException as TE:
            print(f"Warning, camera timed out {TE}")

        except Exception as ex:
            print(f"Warning, camera error!\n {ex}")
            print("Trying to reconnect camera\n Camera may be overheating! Consider lowering \
                  framerate!")
            self.disconnect_camera()
            sleep(0.5)
            self.connect_camera()

    def connect_camera(self):
        """
        Connect to the first instance of basler camera.
        Returns true if a camera was connected.

        """
        try:
            tlf = pylon.TlFactory.GetInstance()
            self.cam = pylon.InstantCamera(tlf.CreateFirstDevice())
            self.cam.Open()
            sleep(0.2)
            self.cam.AcquisitionFrameRateEnable = False # Default is max fps
            try:
               self.camera.SensorReadoutMode.SetValue("Fast")
            except Exception as ex:
                print(f"Sensor readout mode not accepted by camera, {ex}")
            width, height = self.get_sensor_size()
            self.AOI = [0, width, 0, height]
            print("Basler AOI: ", self.AOI)
            return True
        except Exception as ex:
            self.cam = None
            print(ex)
            return False
        
    def disconnect_camera(self):
        self.stop_grabbing()
        self.cam.Close()
        self.cam = None

    def stop_grabbing(self):
        try:
            self.cam.StopGrabbing()
        except Exception as ex:
            print(ex)
            pass
        self.is_grabbing = False

    def set_frame_rate(self, frame_rate):
        """
        Sets the maximum framerate of the camera.
        """
        self.cam.AcquisitionFrameRateEnable = True
        self.cam.AcquisitionFrameRateEnable.SetValue(True)
        
        try:
            self.cam.AcquisitionFrameRate.SetValue(float(frame_rate))
        except Exception as ex:
            print(f"Frame rate not accepted by camera, {ex}")

    def set_gain(self,gain):
        """
        Sets the gain on the camera (amplifies the image in software). 
        """
        try:
            print(f"Setting gain to {gain}")
            self.cam.Gain.Value = int(gain)
        except:

            pass
        

    def set_AOI(self, AOI):
        '''
        Function for setting AOI of basler camera to c_p['AOI'].
        '''
        self.stop_grabbing()
        try:
            '''
            The order in which you set the size and offset parameters matter.
            If you ever get the offset + width greater than max width the
            camera won't accept your valuse. Thereof the if-else-statements
            below. Conditions might need to be changed if the usecase of this
            funciton change.
            '''
            width = int(AOI[1] - AOI[0])
            offset_x = AOI[0]
            height = int(AOI[3] - AOI[2])
            offset_y = AOI[2]
            width -= width % 16
            height -= height % 16
            offset_x -= offset_x % 16
            offset_y -= offset_y % 16
            self.video_width = width
            self.video_height = height
            self.cam.OffsetX = 0
            self.cam.OffsetY = 0
            sleep(0.1)
            self.cam.Width = width
            self.cam.Height = height
            self.cam.OffsetX = offset_x
            self.cam.OffsetY = offset_y
            AOI[0] = offset_x
            AOI[1] = offset_x + width
            AOI[2] = offset_y
            AOI[3] = offset_y + height

        except Exception as ex:
            print(f"AOI not accepted, AOI: {AOI}, error {ex}")

    def set_exposure_time(self, exposure_time):
        self.stop_grabbing()
        try:
            self.cam.ExposureTime = exposure_time

        except Exception as ex:
            print(f"Exposure time not accepted by camera, {ex}")

    def get_exposure_time(self):

        return self.cam.ExposureTime()

    def get_fps(self):
        fps = round(float(self.cam.ResultingFrameRate.GetValue()), 1)
        return fps

    def get_sensor_size(self):
        width = int(self.cam.Width.GetMax())
        height = int(self.cam.Height.GetMax())
        return width, height
