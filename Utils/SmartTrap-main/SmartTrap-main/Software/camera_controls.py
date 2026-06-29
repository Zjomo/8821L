"""
This file defines the different classes and functions needed to operate cameras and use them with
the graphical interface.
----------------------------
Classes:

- CameraClicks: Used to set the area of interest of the camera by clicking and dragging on the 
screen
- CameraMeasurements: Used to measure distances on the screen (in microns) by left and right 
clicking.
- CameraProtocol: Abstract camera class which defines the functions a camera needs to be used by the
main interface.
- TestCamera: A camera that class used for testing which outputs white images consisting of white
noise.
- CameraThread: The thread which is used to continously capture images using the camera.
- VideoFormatError: Error sent if an unrecognised video format is selected.
- VideoWriterThread: Thread that when requested writes the recorded images to a video.
"""



from __future__ import annotations

from typing import Protocol, runtime_checkable, Tuple
from typing_extensions import TypeAlias

import os
from copy import copy, deepcopy
from threading import Thread
from time import perf_counter, sleep, time

import cv2
import numpy as np
import skvideo.io
from PyQt6.QtGui import QColor, QPen
from numpy.typing import NDArray

from mouse_tool import MouseTool


class CameraClicks(MouseTool):
    """
    CameraClicks is a mouse interaction handler for camera controls, enabling users to interactively
    select and zoom into regions of interest (AOI) on an image using mouse events.
    This class provides methods to handle mouse press, release, move, and double-click events,
    and draws selection rectangles on the image. It updates camera parameters based on user input,
    such as defining a new AOI when the user drags a rectangle with the mouse. The tool is intended
    for use in graphical applications where camera view manipulation is required.
    Attributes:
        c_p (dict): A dictionary containing camera parameters and mouse interaction state.
        x_0 (int): Initial x-coordinate for mouse interaction.
        y_0 (int): Initial y-coordinate for mouse interaction.
        red_pen (QPen): Pen used to draw the selection rectangle.
    Methods:
        draw(qp): Draws the selection rectangle if the appropriate tool is active.
        mousePress(): Handles mouse press events for different tools.
        mouseRelease(): Handles mouse release events, updating the AOI if a valid selection is made.
        mouseDoubleClick(): Handles mouse double-click events (currently not implemented).
        mouseMove(): Handles mouse move events for dragging actions.
        getToolName(): Returns the name of the tool.
        getToolTip(): Returns a tooltip describing the tool's functionality.        
    """
    def __init__(self, c_p):
        self.c_p = c_p
        self.x_0 = 0
        self.y_0 = 0
        self.red_pen = QPen()
        self.red_pen.setColor(QColor('red'))
        self.red_pen.setWidth(2)

    def draw(self, qp):
        if self.c_p['mouse_params'][0] == 1:
            qp.setPen(self.red_pen)                
            x1,y1,x2,y2 = self.c_p['mouse_params'][1:5]
            qp.drawRect(x1,y1,x2-x1,y2-y1)
            return

    def mousePress(self):

        # left click
        if self.c_p['mouse_params'][0] == 1:
            pass
        # Right click -drag
        if self.c_p['mouse_params'][0] == 2:
            pass
        
    def mouseRelease(self):
        if self.c_p['mouse_params'][0] != 1:
            return
        x0, y0, x1, y1 = self.c_p['mouse_params'][1:5]
        dx = x1 - x0
        dy = y1 - y0
        if dx**2 < 100 or dy**2 < 100:
            print(dx,dy)
            return
        left = int(x0 * self.c_p['image_scale'])
        right = int(x1 *self.c_p['image_scale'])
        if right < left:
            tmp = right
            right = left
            left = tmp
            # left, right = right, left

        up = int(y0 * self.c_p['image_scale'])
        down = int(y1 * self.c_p['image_scale'])
        if up < down:
            tmp = up
            up = down
            down = tmp
            # up, down = down, up

        # TODO This rapidly escalates if the AOI is not accepted
        self.c_p['AOI'] = [self.c_p['AOI'][0] + left,self.c_p['AOI'][0] + right,
                           self.c_p['AOI'][2] + down,self.c_p['AOI'][2] + up]
        self.c_p['new_settings_camera'] = [True, 'AOI']
        
    def mouseDoubleClick(self):
        # Not used by camera tool
        pass
    
    def mouseMove(self):
        if self.c_p['mouse_params'][0] == 2:
            pass
        
    def getToolName(self):
        return "Camera tool"

    def getToolTip(self):
        return "Use the mouse to zoom in on the screen."


class CameraMeasurements(MouseTool):
    """
    Allows the user to measure distances on the camera.
    Left click for first position and right click for second position.
    Positions are marked with small circles.
    """
    def __init__(self, c_p):
        self.c_p = c_p
        self.x_prev_A = 0
        self.y_prev_A = 0
        self.x_prev_B = 0
        self.y_prev_B = 0
        self.red_pen = QPen(QColor(255,0,0))
        self.blue_pen = QPen(QColor(0,0,255))
        self.circle_radii = 6

    def mousePress(self):
        # left click
        if self.c_p['mouse_params'][0] == 1:
            self.x_prev_A = self.c_p['mouse_params'][1]
            self.y_prev_A = self.c_p['mouse_params'][2]
        # Right click
        if self.c_p['mouse_params'][0] == 2:
            self.x_prev_B = self.c_p['mouse_params'][1]
            self.y_prev_B = self.c_p['mouse_params'][2]
        dx = self.c_p['image_scale']*(self.x_prev_B - self.x_prev_A) * self.c_p['microns_per_pix']
        dy = self.c_p['image_scale']*(self.y_prev_B - self.y_prev_A) * self.c_p['microns_per_pix']

        print(f"CLick - dx: {dx:.3f}, dy: {dy:.3f}, length {((dx**2 + dy**2)**0.5):.3f} [microns]")        
        
    def mouseRelease(self):
        if self.c_p['mouse_params'][0] == 2:
            pass
        
    def mouseDoubleClick(self):
        pass
    
    def draw(self, qp):
        qp.setPen(self.red_pen)

        qp.drawEllipse(self.x_prev_A-self.circle_radii,
                       self.y_prev_A-self.circle_radii,
                       self.circle_radii*2, self.circle_radii*2)
        
        if self.x_prev_B is not None:
            qp.setPen(self.blue_pen)
            qp.drawEllipse(self.x_prev_B-self.circle_radii,
                           self.y_prev_B-self.circle_radii,
                           self.circle_radii*2, self.circle_radii*2)

    def mouseMove(self):
        dx = self.c_p['image_scale']*(self.c_p['mouse_params'][3]
                                      - self.x_prev_A) * self.c_p['microns_per_pix']
        dy = self.c_p['image_scale']*(self.c_p['mouse_params'][4]
                                      - self.y_prev_A) * self.c_p['microns_per_pix']
        print(f"dx: {dx:.3f}, dy: {dy:.3f}, length {((dx**2 + dy**2)**0.5):.3f} [microns]")

        if self.c_p['mouse_params'][0] == 1:
            pass

        if self.c_p['mouse_params'][0] == 2:
            pass

    def getToolName(self):
        return "Camera ruler"

    def getToolTip(self):
        tooltip = """Measures distances on the camera. Left click for first position and right 
        click for second position.\n Drag and left click to measure continuously."""
        return tooltip
        

AOI: TypeAlias = list[int]


@runtime_checkable
class CameraProtocol(Protocol):
    """Structural camera interface using Protocols."""

    # Connect/disconnect
    def connect_camera(self) -> None:
        """Connect to the camera."""

    # Image acquisition
    def capture_image(self) -> NDArray[np.generic]:
        """Capture a single image and return a NumPy array."""

    # Core configuration
    def set_aoi(self, aoi: AOI) -> AOI:
        """
        Set the Area Of Interest.
        Must accept exactly four ints and return the AOI actually set
        (also exactly four ints).
        """

    def get_sensor_size(self) -> Tuple[int, int]:
        """Return the full sensor size as (width, height) in pixels."""

    def set_exposure_time(self, exposure_ms: float) -> None:
        """Set exposure time in milliseconds."""

    def set_frame_rate(self, fps: float) -> None:
        """Set frame rate in frames per second."""

    def set_gain(self, gain_db: float) -> None:
        """Set analog/digital gain in dB."""

class TestCamera(CameraProtocol):
    """
    A test camera which generates random images.
    Used for testing the GUI without a real camera.
    """

    def __init__(self):
        
        self.width = 1024
        self.height = 1024
        self.image_width = self.width
        self.image_height = self.height
        self.exposure_time = 10
        self.AOI = [0,self.width, 0, self.height]

    def connect_camera(self):
        print("Connected to test camera")

    def set_AOI(self, AOI):
        print(f"Set AOI to {AOI}")
        if AOI[0] < 0 or AOI[1] > self.width or AOI[2] < 0 or AOI[3] > self.height:
            print("AOI out of bounds, ignoring")
            AOI = [self.AOI[0],self.AOI[1],self.AOI[2],self.AOI[3]]
            # TODO fix AOI error
            print(self.AOI)
            return
        width = int(AOI[1] - AOI[0])
        height = int(AOI[3] - AOI[2])
        if width < 10 or height < 10:
            print("AOI too small, ignoring")
            return self.AOI
        self.AOI = AOI
        self.image_width = width
        self.image_height = height
        return self.AOI

    def get_sensor_size(self):
        return self.width, self.height

    def set_exposure_time(self, exposure_time):
        if exposure_time < 1 or exposure_time > 1_000:
            print("Exposure time out of bounds, ignoring")
            return
        print(f"Set exposure time to {exposure_time} microseconds")
        self.exposure_time = exposure_time

    def get_exposure_time(self):
        return self.exposure_time  # microseconds

    def set_frame_rate(self, frame_rate):
        print(f"Set frame rate to {frame_rate} fps")

    def set_gain(self, gain):
        print(f"Set gain to {gain}")

    def capture_image(self):
        # Generate a random grayscale image
        image = np.random.randint(0, 256, (self.image_height, self.image_width), dtype=np.uint8)
        sleep(self.exposure_time*1e-3)  # Simulate a short delay for capturing the image
        return image

class CameraThread(Thread):

    def __init__(self, c_p, camera):
        """
        Initiates a camera thread.

        Parameters
        ----------
        c_p : TYPE
            DESCRIPTION. Control parameters used to get commands from the GUI
            or from an automation procedure. Also transmits imformation the
            other direction.
        camera : TYPE
            DESCRIPTION. A camera object implementing the CameraInterface.

        Returns
        -------
        None.

        """
        Thread.__init__(self)
        self.camera = camera
        
        self.camera.connect_camera()
        c_p['camera_width'], c_p['camera_height'] = camera.get_sensor_size()
        self.c_p = c_p
        
        # Zoom out
        self.c_p['AOI'] = [0, self.c_p['camera_width'], 0,
                   self.c_p['camera_height']]
        self.c_p['new_settings_camera'] = [True, 'AOI']
        self.setDaemon(True)

    def update_camera_settings(self):
        if self.c_p['new_settings_camera'][1] == 'AOI':
            self.camera.set_AOI(self.c_p['AOI'])
        elif self.c_p['new_settings_camera'][1] == 'exposure_time':
            self.camera.set_exposure_time(self.c_p['exposure_time'])
        elif self.c_p['new_settings_camera'][1] == 'frame_rate':
            self.camera.set_frame_rate(self.c_p['target_frame_rate'])
        elif self.c_p['new_settings_camera'][1] == 'gain':
            self.camera.set_gain(self.c_p['image_gain'])

        # Resetting the new_settings_camera parameter
        self.c_p['new_settings_camera'] = [False, None]

    def run(self):
        self.c_p['exposure_time'] = self.camera.get_exposure_time()
        count = 0
        while self.c_p['program_running']:
            if self.c_p['new_settings_camera'][0]:
                self.update_camera_settings()
            count += 1
            if count % 110 == 5: 
                p_t = perf_counter()
            self.c_p['image'] = self.camera.capture_image()
            if self.c_p['image'] is None:
                print("None image error!")
            if self.c_p['recording']:
                img = copy(self.c_p['image'])
                name = copy(self.c_p['video_name'])
                self.c_p['frame_queue'].put([img, name,
                                             self.c_p['video_format'], time()])
                
            # Here we perform an estimation of the current fps
            if count % 110 == 105: 
                self.c_p['fps'] = 101 / (perf_counter()-p_t)


class VideoFormatError(Exception):
    """
    Raised when a video format is not supported.
    """
    pass


def create_avi_video_writer(c_p, video_name, image_width, image_height):
    """
    Function for creating a VideoWriter.
    Will also save the relevant parameters of the experiments.
    Returns
    -------
    video : VideoWriter
        A video writer for creating a video.
    experiment_info_name : String
        Name of experiment being run.
    exp_info_params : Dictionary
        Dictionary with controlparameters describing the experiment.
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    
    video_name = c_p['recording_path'] + '/' + video_name + '.avi'
    is_color = len(np.shape(c_p['image'])) > 2 and np.shape(c_p['image'])[2] == 3
    return cv2.VideoWriter(video_name, fourcc, min(500, c_p['fps']),
                           (image_height, image_width), isColor=is_color)
    

def create_mp4_video_writer(c_p, video_name=None):
    """
    Creates a high quality video writer for lossless recording.
    """

    if video_name is None:
        video_name = get_video_name(c_p=c_p)
    tmp = min(500, int(c_p['fps']))
    frame_rate = str(max(25, tmp))  
    if tmp < 25:
        print('Warning, skvideo cannot handle framerates below 25 fps so\
        reverting to 25.')
    video_name = c_p['recording_path'] + '/' + video_name + '.mp4'
    video = skvideo.io.FFmpegWriter(video_name, outputdict={
                                     '-b': c_p['bitrate'],
                                     '-r': frame_rate,  
                                    })
    return video


def npy_generator(path):
    """
    Used to read all the images in a npy image folder one at a time. Takes the
    full path as input and outputs an image. Outputs None if there are no more
    images to read.
    """

    def sorting_key(file_name):
        """Key function to sort files based on the starting number."""
        start_num = file_name.split('-')[0]
        return int(start_num)

    # Sort the files using the custom key
    directory = sorted(os.listdir(path), key=sorting_key) 
    done = False
    num = '-1'  # First frame to load

    while not done:
        done = True
        for file in directory:
            print(file)
            idx = file.find('-')
            print(str(int(file[:idx])),str(int(num)+1))
            if file[:idx] == str(int(num)+1) and file[-4:] == '.npy':
                images = np.load(os.path.join(path, file))
                num = file[idx+1:-4]
                done = False
                for image in images:
                    yield image
    while True:
        yield None


def get_video_name(c_p, base_name=''):
    """
    Returns an auto-generated name of the video. The name has the time of
    creation in the title to be easy to locate.
    """
    import datetime
    now = datetime.now()
    print(c_p['measurement_name'], base_name)
    name = 'video-' + c_p['measurement_name'] + '-' + str(now.hour)
    name += '-' + str(now.minute) + '-' + str(now.second)+'-fps-'
    name += str(c_p['fps'])
    return name


class VideoWriterThread(Thread):
    """
    A class which simply deques the latest frame and prints it to a video.
    """

    def __init__(self, thread_id, name, c_p):
        Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.c_p = c_p
        self.setDaemon(True)

        self.sleep_time = 0.1
        self.frame = None
        self.video_width = np.shape(self.c_p['image'])[0]
        self.video_height = np.shape(self.c_p['image'])[1]
        self.format = self.c_p['video_format']
        self.last_frame_format = self.format
        self.video_created = False
        self.video_name = self.c_p['video_name']
        self.frame_buffer = []
        self.frame_buffer_size = 100
        self.frame_count = 0
        self.frame_time = 0
        self.frame_timings = np.zeros(10_000_000)
        self.video_writer = None
        self.np_save_path = None

    def close_video(self):
        """
        Closes and the current video and deletes the python object.

        Returns
        -------
        None.

        """
        self.video_created = False
        try:
            if self.last_frame_format == 'mp4':
                if self.video_writer is not None:
                    self.video_writer.close()
                del self.video_writer
            elif self.last_frame_format == 'avi':
                if self.video_writer is not None:
                    self.video_writer.release()
                    print("Closed AVI writer")
                del self.video_writer
            else:
                # is npy, save what remains of buffer then clear it
                self.np_save_frames()
                self.frame_count = 0

        except Exception as err:
            # The program tries to close a
            print(f"No video to close {err}")

    def np_save_frames(self):
        """
        Saves all the frames in the buffer to a .npy file.
        """
        if self.frame_count < 3: 
            print("Not enough frames to save, only ", self.frame_count)
            return
        nbr_frames = self.frame_count % self.frame_buffer_size
        if nbr_frames == 0 and self.frame_count != 0:
            nbr_frames = self.frame_buffer_size
        lower_lim = str(max(self.frame_count-nbr_frames, 0))
        upper_lim = str(self.frame_count-1)

        filename = lower_lim + '-' + upper_lim + '.npy'
        with open(self.np_save_path+filename, 'wb') as f:
            np.save(f, self.frame_buffer[:nbr_frames])
        if not self.video_created:
            
            with open(self.np_save_path+'frame_time.npy', 'wb') as f:
                nbr_frames = min(self.frame_count, 10_000_000)
                np.save(f, self.frame_timings[:nbr_frames])
        self.frame_buffer *= 0

    def create_np_writer(self, video_name):
        """
        Creates a folder and saves path to it for saving the numpy images
        """
        if video_name is None:
            video_name = get_video_name(c_p=self.c_p)
        self.np_save_path = self.c_p['recording_path'] + '/' + video_name + '/'
        try:
            os.mkdir(self.np_save_path)
        except Exception as ex:
            print(f"Directory already exist, {ex}")

    def write_to_NPY(self):
        nbr_frames = self.frame_count % self.frame_buffer_size
        if nbr_frames == 0 and self.frame_count > 0:
            # Save the frames into target folder and with suitable name
            self.np_save_frames()
        try:
            self.frame_buffer[nbr_frames, :, :] = deepcopy(self.frame)
            if self.frame_count<10_000_000:
                self.frame_timings[self.frame_count] = self.frame_time
            self.frame_count += 1
        except Exception as ex:
            print(f"Trouble writing frame, {ex}")
            self.close_video()
            self.frame_count = 0

    def write_frame(self):
        """
        Writes a frame to the current video_writer.
        If the format is "npy" we instead put it in our np-array of images.
            If the np-array of images reaches a special threshold then it is
            automatically saved.
            # Reasonable threshold perhaps 100_000 frames?
        """
        if self.format == 'mp4':
            self.video_writer.writeFrame(self.frame)
        elif self.format == 'avi':
            self.video_writer.write(self.frame)
        else:
            self.write_to_NPY()

        # Let the caller know that a frame was successfully added to the output
        return True

    def create_video_writer(self, video_name):
        """
        Creates a video writer for the current video.
        """
        # Adjust the video shape to match the images
        image_shape = np.shape(self.frame)
        self.video_width = int(image_shape[0])
        self.video_height = int(image_shape[1])
        if self.format == 'mp4':
            self.video_writer = create_mp4_video_writer(c_p=self.c_p,
                                                       video_name=video_name)

        elif self.format == 'avi':
            self.video_writer = create_avi_video_writer(self.c_p,
                                                       video_name,
                                                       self.video_width,
                                                       self.video_height)
            print("created avi writer")

        elif self.format == 'npy':
            # Calculates an appropriate buffer size based on the size in memory
            # the frames take up.
            self.frame_buffer_size = int(501760000 / (self.video_width * self.video_height))
            if len(image_shape) < 3: 
                self.frame_buffer = np.uint8(np.zeros([self.frame_buffer_size,
                                                       self.video_width,
                                                       self.video_height,
                                                       ]))
            elif image_shape[2] == 3:
                self.frame_buffer = np.uint8(np.zeros([self.frame_buffer_size,
                                                       self.video_width,
                                                       self.video_height, 3]))
            self.frame_count = 0
            
            self.create_np_writer(video_name)
        else:
            raise VideoFormatError(f"Video format{self.c_p['video_format']}\
                                   not recognized!")

        self.video_created = True

    def write_video_info(self, filename):
        """
        Writes a small .txt file with basic info such as exposure time, and framerate
        of a video.
        """
        filepath = self.c_p['recording_path'] + '/' + filename+'_info.txt'
        with open(filepath, 'w') as f:
            f.write("Video framerate:"+str(self.c_p['fps'])+ "fps\n")
            f.write(
                "Exposure time "
                + str(self.c_p['exposure_time'])
                + "microseconds (or milliseconds depending on model of camera)\n"
                )
            f.write("Video format: " + str(self.c_p['video_format']) + "\n")
            f.write("Video resolution: " + str(self.video_width) 
                    + "x" + str(self.video_height) + "\n")

    def run(self):
        self.c_p['video_idx'] = 0
        while self.c_p['program_running']:
            sleep(self.sleep_time)
            self.c_p['saving_video'] = False

            while self.c_p['recording'] or not self.c_p['frame_queue'].empty()\
                    and self.c_p['program_running']:
                self.c_p['saving_video'] = True

                if not self.c_p['frame_queue'].empty():

                    [self.frame, source_video, self.format, self.frame_time] = (
                        self.c_p['frame_queue'].get())

                    # Check that name and size are correct, if not create a new
                    image_shape = np.shape(self.frame)
                    if image_shape[0] != self.video_width or\
                            image_shape[1] != self.video_height:
                        self.video_width = image_shape[0]
                        self.video_height = image_shape[1]
                        self.close_video()
                    # Check if name and format is ok
                    if self.video_name != source_video:
                        self.close_video()
                        self.video_name = source_video
                    if not self.last_frame_format == self.format:
                        self.close_video()

                    if not self.video_created:
                        size = '_' + str(self.video_width) + 'x'
                        size += str(self.video_height)
                        self.write_video_info(self.video_name+size)
                        self.create_video_writer(self.video_name+size)
                    self.last_frame_format = self.format
                    self.write_frame()
                else:
                    # Queue empty
                    sleep(0.001)
            if self.video_created:
                self.close_video()

