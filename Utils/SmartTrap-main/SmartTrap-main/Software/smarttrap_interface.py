"""
This file contains the graphical user interface of the SmartTrap. 
------------------------------------------------
Classes:

- ScreenWorker: A helper thread that updates the screen (camera feed) of the interface.
- MainWindow: The main graphical user interface of SmartTrap. 
"""


import math
import pickle
import sys
from functools import partial
from time import sleep

import numpy as np

import sys
if sys.platform.startswith("win"):
    # Imported for side-effects on Windows file menu behavior.
    import win32com.client as win32  # noqa: F401

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QPoint
from PyQt6.QtGui import (
    QColor,
    QDoubleValidator,
    QFont,
    QImage,
    QKeySequence,
    QPen,
    QPixmap,
    QPolygon,
    QPainter,
    QAction,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QToolBar,
)

import auto_controller_v6 as AutoController
import laser_controller
import motor_controls

from camera_controls import CameraThread, VideoWriterThread, CameraClicks, CameraMeasurements
from control_parameters import default_c_p, get_data_dicitonary_smarttrap, ControlParametersViewer, CurrentValueWindow
from laser_move_widget import LaserPiezoWidget, MinitweezersLaserMove
from live_plots import PlotWindow
from instrument_driver import InstrumentControllerThread
from laser_protocols import PullingProtocolWidget
from QWidget_dock_container import QWidgetWindowDocker
from real_time_tracking import TrackingControlWidget
from smarttrap_config import get_defualt_config, create_devices
from microfluidics_controllers import (
    MicrofluidicsControllerWidget,
    ConfigurePumpWidget,
)

from data_saver import DataSaverThread

DEFAULT_RADIUS_PIX = 250

class ScreenWorker(QThread):
    """
    Worker thread which updates th screen continously with the images of the camera.

    Inherits from QThread to handler worker thread setup, signals and wrap-up.
    """

    changePixmap = pyqtSignal(QImage)

    def __init__(self, c_p, data, *args, **kwargs):
        super(ScreenWorker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.c_p = c_p
        self.data_channels = data
        self.args = args
        self.kwargs = kwargs

        # Create different colored pens for drawing on the screen
        self.green_pen = QPen()
        self.green_pen.setColor(QColor('green'))
        self.green_pen.setWidth(3)
        self.blue_pen = QPen()
        self.blue_pen.setColor(QColor('blue'))
        self.blue_pen.setWidth(3)
        self.red_pen = QPen()
        self.red_pen.setColor(QColor('red'))
        self.red_pen.setWidth(3)


    def draw_particle_positions(self,centers, pen=None, radii=None, info=None, info_labels=None):
        """
        Draws ellipses for particle positions on a QPainter canvas, with optional radii and
        annotation. Iterates over a list of particle center coordinates and draws an ellipse for
        each. The size of each ellipse can be set via `radii`, or defaults to a preset value.
        Optionally, info (numeric values or labels) can be displayed next to each particle.
        Appearance can be customized with a QPen. Parameters:
            centers (list): (x, y) coordinates for particle centers.
            pen (QPen, optional): Pen for drawing. Defaults to self.red_pen.
            radii (list, optional): Radii for each particle. Defaults to preset value.
            info (list, optional): Info values to display next to each particle.
            info_labels (str, optional): Label prefix before each info value.
        Notes:
            - Adapts ellipse size and text position based on image scale.
            - Handles missing/mismatched info gracefully.
            - Intended for use within a class managing QPainter (`self.qp`) and config (`self.c_p`).
        """
        
        if radii is None:
            radii = [DEFAULT_RADIUS_PIX]*len(centers)
        if len(radii)>0:
            radius = radii[0]
        else:
            radius = DEFAULT_RADIUS_PIX
        try:
            rx = int(100/self.c_p['image_scale'])
        except:
            rx=20
        ry = rx
        font_size = int(rx/4)

        # Create a QFont object with the desired font size
        self.qp.setFont(QFont("Arial", font_size))
        if pen is None:
            self.qp.setPen(self.red_pen)
        else:
            self.qp.setPen(pen)
        for idx, pos in enumerate(centers):
            # Adopting what we draw to the predicted radii
            try:
                rx = int(2*radii[idx]/self.c_p['image_scale'])
                ry = rx
            except Exception as E:
                rx = int(DEFAULT_RADIUS_PIX/self.c_p['image_scale'])
                ry = rx
                pass
            try:
                x = int(pos[0] / self.c_p['image_scale'])
                y = int(pos[1] / self.c_p['image_scale'])
            except IndexError as IE:
                return

            self.qp.drawEllipse(x-int(rx/2)-1, y-int(ry/2)-1, rx, ry)
                    # Check if information display is enabled
            if info is not None:
                # You can customize this part to show whatever information you want
                try:
                    if idx > len(info) or len(info) == 0:
                        info_text = str(idx)
                        continue
                except TypeError:
                    continue
                try:
                    info_text = info_labels + str(round(info[idx],1))
                    # Position for the text: adjust the x, y as needed for text
                    # to not overlap the circle
                    text_x = int(x +1.1*rx)
                    text_y = int(y)

                    # Draw the text
                    self.qp.drawText(text_x, text_y, info_text)
                except Exception as E:
                    # There can be an index errror here which is harmless, caused by a missing
                    # detection in the deep learning thread
                    pass

    def draw_pipette(self, pen=None):
        """
        Draws the bounding box of the pipette
        """
        if pen is None:
            self.qp.setPen(self.green_pen)
        else:
            self.qp.setPen(pen)
        x = int((self.c_p['pipette_location'][0] - self.c_p['pipette_location'][2] / 2)
                / self.c_p['image_scale'])
        y = int((self.c_p['pipette_location'][1]) / self.c_p['image_scale'])
        w = int(self.c_p['pipette_location'][2] / self.c_p['image_scale'])
        h = int(self.c_p['pipette_location'][3] / self.c_p['image_scale'])
        self.qp.drawRect(x, y, w, h)

    def preprocess_image(self):

        # Check if offset and gain should be applied.
        if self.c_p['image_offset'] != 0:
            self.image += int(self.c_p['image_offset'])
            self.image = np.uint8(self.image)
            
    def draw_central_circle(self):
        """
        Draws a circle at the center of the image.
        """
        self.blue_pen.setColor(QColor('blue'))
        cx = int((self.c_p['camera_width']/2 - self.c_p['AOI'][0])/self.c_p['image_scale'])
        cy = int((self.c_p['camera_height']/2 - self.c_p['AOI'][2])/self.c_p['image_scale'])
        rx = 50
        ry = 50
        self.qp.drawEllipse(cx-int(rx/2)-1, cy-int(ry/2)-1, rx, ry)

    def draw_lasers(self):
        """
        Marks the laser positions on the image using two crosses, one per laser
        """

        length = 10  # half-length of the cross arms

        #Laser A
        self.qp.setPen(self.red_pen)
        x = int((self.c_p['laser_position_A_predicted'][0] - self.c_p['AOI'][0])
                / self.c_p['image_scale'])
        y = int((self.c_p['laser_position_A_predicted'][1] - self.c_p['AOI'][2])
                / self.c_p['image_scale'])
        # Define the lines for the cross
        self.qp.drawLine(x - length, y, x + length, y)
        # Draw the vertical line of the cross
        self.qp.drawLine(x, y - length, x, y + length)

        # Laser B        
        self.qp.setPen(self.blue_pen)
        x = int((self.c_p['laser_position_B_predicted'][0] - self.c_p['AOI'][0])
                / self.c_p['image_scale'])
        y = int((self.c_p['laser_position_B_predicted'][1] - self.c_p['AOI'][2])
                / self.c_p['image_scale'])
        # Draw the cross
        self.qp.drawLine(x - length, y, x + length, y)
        # Draw the vertical line of the cross
        self.qp.drawLine(x, y - length, x, y + length)

    def draw_force(self):
        """
        Draws a force vector as an arrow on the GUI, representing the total force acting on a 
        trapped particle. This method checks if a particle is currently trapped. If so, it
        calculates the starting position of the arrow based on the trapped particle's position and
        the image scale. The direction and length of the arrow are determined by the total force
        components (F_total_X, F_total_Y), scaled appropriately. The arrow is drawn using a blue
        pen, and an arrowhead is added to indicate direction. The method uses the QPainter object
        (`self.qp`) to render the line and arrowhead on the GUI.
        """

        if not self.data_channels['particle_trapped'].get_data(1)[0]:
            return
        x = self.c_p['Trapped_particle_position'][0] / self.c_p['image_scale']
        y = self.c_p['Trapped_particle_position'][1] / self.c_p['image_scale']
        start_point = QPoint(int(x), int(y))
        xf = 3 * self.data_channels['F_total_X'].get_data(1)[0] / self.c_p['image_scale']
        yf = -3 * self.data_channels['F_total_Y'].get_data(1)[0] / self.c_p['image_scale']
        x_end = int(x+xf)
        y_end = int(y+yf)
        end_point = QPoint(x_end, y_end)
        self.qp.setPen(self.blue_pen)
        self.qp.drawLine(start_point, end_point)

        # Coordinates for the arrowhead
        angle = math.atan2(yf, xf)
        arrowhead_length = 10/ self.c_p['image_scale']
        angle1 = angle + math.pi / 6  # Angle offset for one side of the arrowhead
        angle2 = angle - math.pi / 6  # Angle offset for the other side

        x1 = x_end - arrowhead_length * math.cos(angle1)
        y1 = y_end - arrowhead_length * math.sin(angle1)
        x2 = x_end - arrowhead_length * math.cos(angle2)
        y2 = y_end - arrowhead_length * math.sin(angle2)

        # Create the arrowhead polygon
        arrow_head = QPolygon([
            end_point,
            QPoint(int(x1), int(y1)),
            QPoint(int(x2), int(y2))
        ])

        # Draw the arrowhead
        self.qp.drawPolygon(arrow_head)      

    def get_boring_particles(self):
        """
        Extracts the particles that are not the trapped particle or the pipette particle, returns
        the indices of them.
        """

        positions = np.copy(self.c_p['predicted_particle_positions'])
        if len(positions) == 0:
            return None
        mask = np.ones(len(positions), dtype=bool)
        
        self.data_channels['particle_trapped'].get_data(1)[0]
        if self.data_channels['particle_trapped'].get_data(1)[0]:
            target_point = self.c_p['Trapped_particle_position'][0:2]
            distances = np.linalg.norm(positions - target_point, axis=1)
            index_of_closest = np.argmin(distances)
            mask[index_of_closest] = False

        if self.c_p['particle_in_pipette'] and self.c_p['pipette_located']:        
            target_point = self.c_p['pipette_particle_location'][0:2]
            distances = np.linalg.norm(positions - target_point, axis=1)
            index_of_closest = np.argmin(distances)
            mask[index_of_closest] = False

        return mask

    def run(self):
        """
        Main loop for updating and rendering the GUI image frame. Continuously processes image data,
        applies preprocessing, and updates the display with various overlays such as particles,
        pipette, lasers, and other graphical elements based on the current control parameters.
        Handles both test and live modes, manages frame scaling, and emits the updated QPixmap for
        display. Also manages drawing of additional information such as particle positions, force
        vectors, and zoom rectangles, with error handling for missing or mismatched data.This 
        function is intended to be run in a separate thread or process to keep the GUI responsive.
        """

        while self.c_p['program_running']:
            
            if self.c_p['image'] is not None:
                self.image = np.array(self.c_p['image'])
            else:
                print("Frame missed!")
                continue

            W, H = self.c_p['frame_size']
            self.c_p['image_scale'] = max(self.image.shape[1]/W, self.image.shape[0]/H)
            self.preprocess_image()            

            # It is quite sensitive to the format here, won't accept any missmatch
            if len(np.shape(self.image)) < 3:
                QT_Image = QImage(self.image, self.image.shape[1],
                                       self.image.shape[0],
                                       QImage.Format.Format_Grayscale8)
                QT_Image = QT_Image.convertToFormat(QImage.Format.Format_RGB888)
            else:                
                QT_Image = QImage(self.image, self.image.shape[1],
                                       self.image.shape[0],
                                       QImage.Format.Format_RGB888)
                
            picture = QT_Image.scaled(
                W,H,
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            
            sleep(0.04) # This practically sets the refresh rate of the video feed, 25 fps default.
            
            # Paint extra items on the screen

            self.qp = QPainter(picture)

            # Draw zoom in rectangle
            try:
                self.c_p['click_tools'][self.c_p['mouse_params'][5]].draw(self.qp)
            except Exception as E:
                print(E)
                print(len(self.c_p['click_tools']))
                
            self.qp.setPen(self.blue_pen)
            if self.c_p['central_circle_on']:
                self.draw_central_circle()

                
            if (
                self.c_p['draw_pipette']
                and self.c_p['pipette_location'][0] is not None
                and self.c_p['pipette_located']
            ):
                self.draw_pipette()

            # Draw the particles if the parameters tell us to do so
            if self.c_p['tracking_on'] and self.c_p['draw_particles']:       
                info = None
                info_labels = None

                # Check if ther are particles in the trap/pipette if that is the case then don't
                # draw them in red.                               
                indices = self.get_boring_particles()
                try:
                    info = self.c_p['z-predictions'][indices]
                    centers=self.c_p['predicted_particle_positions'][indices]
                    radii=self.c_p['predicted_particle_radii'][indices]
                except Exception as e:
                    info = self.c_p['z-predictions']
                    centers = self.c_p['predicted_particle_positions']
                    radii = self.c_p['predicted_particle_radii']
                info_labels = 'z: '
                if not self.c_p['draw_z_text']:
                    info = None
                    info_labels = None
                self.draw_particle_positions(
                    centers=centers,
                    radii=radii,
                    info=info,
                    info_labels=info_labels)
                
            if self.c_p['draw_TnP_particles']:

                if self.data_channels['particle_trapped'].get_data(1)[0]:
                    
                    info = None
                    info_labels = None
                    if self.c_p['draw_z_text']:
                        info = [self.c_p['Trapped_particle_position'][2]]
                        info_labels = 'z: '
                    self.draw_particle_positions(
                        [self.c_p['Trapped_particle_position'][0:2]],
                        radii=[self.c_p['Trapped_particle_position'][3]],
                        pen=self.blue_pen,
                        info=info,
                        info_labels=info_labels,)
                    if self.c_p['draw_force']:
                        self.draw_force()
                
                if self.c_p['particle_in_pipette'] and self.c_p['pipette_located']:
  
                    info = None
                    info_labels = None
                    if self.c_p['draw_z_text']:
                        info = [self.c_p['pipette_particle_location'][2]]
                        info_labels = 'z: '
                    self.draw_particle_positions(
                        [self.c_p['pipette_particle_location'][0:2]],
                        radii=[self.c_p['pipette_particle_location'][3]],
                        pen=self.green_pen,
                        info=info,
                        info_labels= info_labels,
                        )
                
            if self.c_p['draw_lasers']:
                self.draw_lasers()
            self.qp.end()
            self.changePixmap.emit(picture)


class MainWindow(QMainWindow):
    """
    This class implements the main graphical user interface (GUI) window for the Optical
    Tweezers (OT) control software. It is built on top of QMainWindow and provides a
    comprehensive interface for controlling, monitoring, and recording experiments involving
    optical tweezers.

    Key Features:
    - Initializes and manages hardware threads for cameras, microcontrollers, and other
        devices.
    - Provides toolbars and menus for camera control, data recording, and experiment
        configuration.
    - Handles live video display, image snapshots, and video recording.
    - Manages data acquisition, saving, and export functionalities.
    - Offers interactive controls for motors, lasers, microfluidics, and other experiment
        components via dockable widgets.
    - Supports saving and recalling experiment positions, as well as zeroing and resetting
        force/position sensors.
    - Integrates plotting windows for live data visualization and analysis.
    - Handles user interactions such as mouse events for experiment manipulation and tool
        selection.

    The MainWindow class serves as the central hub for user interaction, hardware
    communication, and experiment management in the OT software suite.
    """


    def __init__(self, testmode=False):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Optical tweezers: Main window")
        self.c_p = default_c_p()

        # Import configuration parameters and add them to the control parameters
        config = get_defualt_config()
        for key in config:
            self.c_p[key] = config[key]
        
        self.data_channels = get_data_dicitonary_smarttrap()
        self.video_idx = 0
        self.data_idx = 0 # Index of data saved
        self.saving = False
        self.data_channel_viewer = None

        # Loading the controllers. These are system specific
        #
        if testmode:
            self.create_test_controllers()
        else:
            self.camera, self.object_tracker, self.motor_controller, self.objective_motor, \
            self.laser_A, self.laser_B, self.microfluidics_controller, self.valve_controller, \
            self.pipette_pump, self.instrument_driver = create_devices(self.c_p, self.data_channels)            
        self.start_threads()
        self.plot_windows = None

        # Set up camera window
        H = int(1080/4)
        W = int(1920/4)
        sleep(0.1)
        self.c_p['frame_size'] = int(self.c_p['camera_width']/2), int(self.c_p['camera_height']/2)
        self.camera_window_label = QLabel("Camera window")
        self.camera_window_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setCentralWidget(self.camera_window_label)
        self.camera_window_label.setMinimumSize(W,H)
        self.painter = QPainter(self.camera_window_label.pixmap())
        th = ScreenWorker(c_p=self.c_p, data=self.data_channels)
        th.changePixmap.connect(self.setImage)
        th.start()

        # Create toolbar for camera
        self.create_camera_toolbar()
        self.addToolBarBreak() 

        self.create_mouse_toolbar()

        # Create menus and drop down options
        self.menu = self.menuBar()
        self.create_filemenu()
        self.drop_down_window_menu()
        self.action_menu()

        self.move_objective_toolbar= motor_controls.ObjectiveStepperControllerToolbar(
            self.objective_motor , self)
        self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.move_objective_toolbar)
        
        # Creating UI elements to be used in the main window
        self.motor_widget = motor_controls.MotorControllerWindow(self.c_p, self.motor_controller)
        self.motor_control_dock = QWidgetWindowDocker(self.motor_widget, "Motor controls")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.motor_control_dock)
        
        self.laser_mover_widget = LaserPiezoWidget(self.c_p, self.data_channels)
        self.laser_mover_dock = QWidgetWindowDocker(self.laser_mover_widget, "Wiggler controls")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.laser_mover_dock)

        self.laser_controller_widget = laser_controller.LaserControllerWidget(
            self.c_p, self,self.laser_A, self.laser_B)
        self.laser_controller_dock = QWidgetWindowDocker(self.laser_controller_widget, "Laser controls")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.laser_controller_dock)

        self.pulling_protocol_widget = PullingProtocolWidget(self.c_p, self.data_channels)
        self.pulling_protocol_dock = QWidgetWindowDocker(self.pulling_protocol_widget, "Pulling protocol")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.pulling_protocol_dock)

        self.microfludics_widget = MicrofluidicsControllerWidget(self.c_p,
                                                                self.microfluidics_controller,
                                                                self.valve_controller,
                                                                self.pipette_pump                                                           
                                                                )
        
        self.microfludics_dock = QWidgetWindowDocker(self.microfludics_widget,
                                                    "Microfluidics controls")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.microfludics_dock)
        
        self.show()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.camera_window_label.setPixmap(QPixmap.fromImage(image))

    def create_test_controllers(self):
        """
        This function is called when running in testmode and provides the interface with mock 
        devices that let it operate without any extrernal hardware.
        """
        from camera_controls import TestCamera
        self.camera = TestCamera()
        self.camera.connect_camera()

        from real_time_tracking import TestTracker
        self.object_tracker = TestTracker()

        from motor_controls import TestMotorController, TestObjectiveMovement
        self.motor_controller = TestMotorController()        
        self.objective_motor = TestObjectiveMovement()
        self.objective_motor.connect("COM4")

        from laser_controller import TestLaserController
        self.laser_A = TestLaserController()
        self.laser_A.connect("COM5")
        self.laser_B = TestLaserController()
        self.laser_B.connect("COM6")

        from microfluidics_controllers import (TestMicrofluidicsController,
                                                 TestValveController, TestPipettePump)
        self.microfluidics_controller = TestMicrofluidicsController()
        self.microfluidics_controller.connect("COM7")
        self.valve_controller = TestValveController()
        self.valve_controller.connect("COM8")
        self.pipette_pump = TestPipettePump()
        self.pipette_pump.connect("COM9")


    def start_threads(self):
        """
        This function starts the various communications threads 
            - Camera: Captures images continously
            - Instrument controller: Reads the various photosensors and communicatates with 
                the actuators on the instrument
            - Video writer: Writes images to videos when recording
            - data saver: Saves data in the background.
            - Auto-controller: Controls the various autonomous protocols and handles tracking
        """

        # Start camera thread
        self.camera_thread = None
        self.instrument_controller_thread = None

        self.camera
        if self.camera is not None:
            self.camera_thread = CameraThread(self.c_p, self.camera)
            self.camera_thread.start()


        # Start instrument controller thread
        try:            
            self.instrument_controller_thread = InstrumentControllerThread(self.c_p, self.instrument_driver)
            self.instrument_controller_thread.start()
            sleep(0.1)
            
        except Exception as E:
            print("Warning could not connect to instrument controller.")
            print(E)

        self.video_writer_thread = VideoWriterThread(2, 'video thread', self.c_p)
        self.video_writer_thread.start()

        self.data_saver_thread = DataSaverThread(self.c_p, self.data_channels)
        self.data_saver_thread.start()


        try:
            self.auto_controller_thread = AutoController.AutoControllerThread(
                self.c_p,
                self.data_channels,
                object_tracker=self.object_tracker,
                motor_controller=self.motor_controller,
                main_window=self,
                data_saver=self.data_saver_thread,
                pipette_pump=self.pipette_pump
                )
            self.auto_controller_thread.start()
            print("Auto controller started")
        except Exception as E:
            print(E)

    def create_mouse_toolbar(self):
        """
        Creates and configures the mouse toolbar with various interactive tools for the application.
        This method initializes the mouse toolbar by adding a set of tool widgets (such as camera clicks,
        motor control, laser movement, and measurement tools) to the application's toolbar. Each tool is
        associated with a QAction, which is added to the toolbar and can be triggered via mouse or keyboard
        shortcuts (Ctrl+1 to Ctrl+0). The toolbar allows users to select and activate different mouse tools
        for interacting with the application's main interface.
        The currently selected mouse tool is set based on the value in `self.c_p['mouse_params'][5]`.
        """

        self.c_p['click_tools'].append(CameraClicks(self.c_p))
        self.c_p['click_tools'].append(motor_controls.MotorMouseMove(self.c_p,
                                                                    self.data_channels,
                                                                    self.motor_controller))
        self.c_p['click_tools'].append(MinitweezersLaserMove(self.c_p))
        self.c_p['click_tools'].append(AutoController.SelectLaserPosition(self.c_p))
        self.c_p['click_tools'].append(CameraMeasurements(self.c_p))

        self.c_p['mouse_params'][5] = 0

        self.mouse_toolbar = QToolBar("Mouse tools")
        self.addToolBar(self.mouse_toolbar)
        self.mouse_actions = []
        number_keys = [Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4, Qt.Key.Key_5, 
               Qt.Key.Key_6, Qt.Key.Key_7, Qt.Key.Key_8, Qt.Key.Key_9, Qt.Key.Key_0]

        for idx, tool in enumerate(self.c_p['click_tools']):
            self.mouse_actions.append(QAction(tool.getToolName(), self))
            self.mouse_actions[-1].setToolTip(tool.getToolTip()+"\nShortcut: Ctrl+"+str(idx+1))
            command = partial(self.set_mouse_tool, idx)
            self.mouse_actions[-1].triggered.connect(command)
            self.mouse_actions[-1].setCheckable(True)
            if idx < 10:
                self.mouse_actions[-1].setShortcut(
                    QKeySequence(Qt.Modifier.CTRL | number_keys[idx]))
            self.mouse_toolbar.addAction(self.mouse_actions[-1])
        self.mouse_actions[self.c_p['mouse_params'][5]].setChecked(True)
        
    def set_mouse_tool(self, tool_no=0):
        """
        Sets the active mouse tool based on the provided tool number.
        This method updates the internal state to reflect the selected mouse tool,
        unchecks all mouse action buttons, and checks the button corresponding to
        the selected tool. If the provided tool number exceeds the available tools,
        the method returns without making changes.
        Args:
            tool_no (int, optional): The index of the mouse tool to activate. Defaults to 0.
        """

        if tool_no > len(self.c_p['click_tools']):
            return
        self.c_p['mouse_params'][5] = tool_no
        for tool in self.mouse_actions:
            tool.setChecked(False)
        self.mouse_actions[tool_no].setChecked(True)
        print("Tool set to ", tool_no)

    def set_gain(self, gain):
        """
        Sets the camera gain value based on the input from the gain_LineEdit widget.
        Retrieves the gain value entered by the user, converts it to a float, and updates the camera
        parameters dictionary (`self.c_p`) with the new gain value. Also flags that new camera
        settings are available. If the input is invalid (e.g., empty or non-numeric), the function
        silently ignores the error.
        Parameters
        ----------
        gain : float
            The gain value to set (not used directly, as the value is read from the widget).
        """
        
        gain = float(self.gain_LineEdit.text())
        try:
            self.c_p['image_gain'] = gain
            self.c_p['new_settings_camera'] = [True, 'gain']
        except ValueError:
            # Harmless, someone deleted all the numbers in the line-edit
            pass

    def create_filemenu(self):
        """
        Creates the 'File' menu and its submenus for the application's main menu bar.
        This method adds the following to the menu:
            - A 'Recording format' submenu to select the video file format (avi, mp4, npy).
            - An 'Image format' submenu to select the image file format (png, jpg, npy).
            - An action to set the save path for experiment files.
            - An action to set the default filename for saved data, video, and image files.
            - An action to save data to a .npy file.
        Each menu item is connected to its corresponding handler method.
        """
        file_menu = self.menu.addMenu("File")
        file_menu.addSeparator()

        # Create submenu for setting recording(video) format
        format_submenu = file_menu.addMenu("Recording format")
        for f in self.c_p['available_video_formats']:
            format_command= partial(self.set_video_format, f)
            format_action = QAction(f, self)
            format_action.setStatusTip(f"Set recording format to {f}")
            format_action.triggered.connect(format_command)
            format_submenu.addAction(format_action)

        # Submenu for setting the image format
        image_format_submenu = file_menu.addMenu("Image format")        
        for f in self.c_p['available_image_formats']:
            format_command= partial(self.set_image_format, f)
            format_action = QAction(f, self)
            format_action.setStatusTip(f"Set image format to {f}")
            format_action.triggered.connect(format_command)
            image_format_submenu.addAction(format_action)

        # Data recording format submenu
        data_format_submenu = file_menu.addMenu("Data format")        
        for f in self.c_p['available_data_formats']:
            format_command= partial(self.set_data_format, f)
            format_action = QAction(f, self)
            format_action.setStatusTip(f"Set data format to {f}")
            format_action.triggered.connect(format_command)
            data_format_submenu.addAction(format_action)

        # Add command to set the savepath of the experiments.
        set_save_action = QAction("Set save path", self)
        set_save_action.setStatusTip("Set save path")
        set_save_action.triggered.connect(self.set_save_path)
        file_menu.addAction(set_save_action)

        set_filename_action = QAction("Set filename", self)
        set_filename_action.setStatusTip("Set filename for saved, data, video and image files")
        set_filename_action.triggered.connect(self.set_default_filename)
        file_menu.addAction(set_filename_action)

        # Add command to save the data
        save_data_action = QAction("Save data", self)
        save_data_action.setStatusTip("Save data to file")
        save_data_action.triggered.connect(self.save_data_to_dict) # Dump data before
        file_menu.addAction(save_data_action)
    
    def record_data(self):
        """
        Toggles the data recording state.
        If data recording is currently active, this method stops the recording process,
        updates the toggle action text to "Start recording data", and unchecks the toggle.
        If data recording is not active, it starts the recording process, updates the toggle
        action text to "Stop recording data", and checks the toggle.
        This method is typically connected to a UI action for starting or stopping data recording.
        """

        if (self.data_saver_thread.is_saving()):
            self.stop_saving()
            self.toggle_data_record_action.setText("Start recording data")
            self.toggle_data_record_action.setChecked(False)
        else:
            self.start_saving()
            self.toggle_data_record_action.setText("Stop recording data")
            self.toggle_data_record_action.setChecked(True)

    def start_saving(self):
        """
        Initializes the saving process by recording the current indices of relevant data channels.
        Sets the starting indices for PSD, motor, and prediction data channels, and marks the saving
          state as active.
        """        
        self.data_saver_thread.start_saving()        

    def stop_saving(self):
        """
        Stops the data saving process, collects the recorded data from all relevant channels,
        and saves it to a file.
        This method finalizes the current data recording session by:
        - Setting the saving flag to False.
        - Determining the stop indices for each data channel.
        - Handling cases where channels may not have sampled correctly by adjusting indices.
        - Extracting the relevant data slices for each channel, accounting for channels sampled at
          different rates.
        - Saving the collected data to a file using pickle.
        - Incrementing the data file index for future recordings.
        The method ensures that data from all channels is synchronized as much as possible,
        and handles wrap-around cases where the stop index is less than the start index.
        """
        self.data_saver_thread.stop_saving()        

    def save_data_to_dict(self):
        """
        Prompts the user for a filename, collects data from enabled data channels, and saves the
        data to a file. The function displays an input dialog to the user to specify a filename
        for saving the data. If a valid name is entered, it gathers data from all data channels
        that have saving enabled, and serializes the collected data into a file using pickle.
        The file is saved in the directory specified by 'recording_path' in the configuration
        parameters.

        Returns:
            None
        """

        text, ok = QInputDialog.getText(self, 'Filename dialog', 'Set name for data to be saved:')
        if not ok:
            print("No valid name entered")
            return
        filename = self.c_p['recording_path'] + '/' + text
        self.c_p['save_idx'] = self.data_channels['PSD_A_P_X'].index
        sleep(0.1) # Make sure all channels have reached this point
        data = {}
        for channel in self.data_channels:
            if self.data_channels[channel].saving_toggled:
                data[channel] = self.data_channels[channel].get_data_spaced(1e9)
        print(f"Saving data to {filename}")
        with open(filename, 'wb') as f:
                pickle.dump(data, f)

    def action_menu(self):
        action_menu = self.menu.addMenu("Actions")

        self.save_position_action = QAction("Save position", self)
        self.save_position_action.setStatusTip("Save current position")
        self.save_position_action.triggered.connect(self.save_position)
        action_menu.addAction(self.save_position_action)

        self.action_references = {}
        self.delete_action_references = {}
        self.saved_positions_submenu = action_menu.addMenu("Go to saved positions")

        self.zero_force_action = QAction("Zero force", self)
        self.zero_force_action.setStatusTip(
            "Zero force for current value, resets it if it's already zeroed")
        self.zero_force_action.triggered.connect(self.zero_force_PSDs)
        action_menu.addAction(self.zero_force_action)

        self.reset_force_psds_action = QAction("Reset force PSDs", self)
        self.reset_force_psds_action.setStatusTip("Reset force PSDs their default values")
        self.reset_force_psds_action.triggered.connect(self.reset_force_PSDs)
        action_menu.addAction(self.reset_force_psds_action)

        self.zero_position_action = QAction("Zero position", self)
        self.zero_position_action.setStatusTip(
            "Zero position for current value, resets it if it's already zeroed")
        self.zero_position_action.triggered.connect(self.zero_position_PSDs)
        action_menu.addAction(self.zero_position_action)

        self.reset_position_psds_action = QAction("Reset position PSDs", self)
        self.reset_position_psds_action.setStatusTip("Reset position PSDs their default values")
        self.reset_position_psds_action.triggered.connect(self.reset_position_PSDs)
        action_menu.addAction(self.reset_position_psds_action)

        # Added some space between the delete and move to submenues to avoid accidental clicks.
        self.delete_position_submenu = action_menu.addMenu("Delete saved positions")

        for idx in range(len(self.c_p['saved_positions'])):
            self.add_position(idx)
        for idx in range(len(self.c_p['saved_positions'])):
            self.delete_positon_action(idx)
            
        def toggle_central_circle():
            self.c_p['central_circle_on'] = not self.c_p['central_circle_on']

        self.central_circle_button = QAction("Toggle center circle", self)
        self.central_circle_button.setStatusTip(
            "Turns on/off the central circle used for alignment")
        self.central_circle_button.setCheckable(True)
        self.central_circle_button.setChecked(self.c_p['central_circle_on'])
        self.central_circle_button.triggered.connect(toggle_central_circle)
        action_menu.addAction(self.central_circle_button)

    def zero_force_PSDs(self):
        """
        Zeroes the force readings from the PSD (Position Sensitive Detectors) channels by updating 
        the mean force offsets. This method calculates the mean values from the latest data samples
        of the PSD force channels (X and Y for both A and B), updates the corresponding force mean
        offsets, and recalculates the PSD means accordingly. It also sends a command to trigger the
        update on the connected hardware and zeroes the Z force by adjusting the null offset for the
        total Z force channel.
        Side Effects:
            - Modifies 'PSD_force_means' and 'PSD_means' in the configuration parameter dictionary
              (self.c_p).
            - Updates the 'Photodiode_sum_to_force' Z offset.
            - Sets a command flag for the hardware.
            - Prints the updated PSD mean and force mean for the first channel.
        """
        
        self.c_p['PSD_force_means'][0] += (
            np.mean(self.data_channels['PSD_A_F_X'].get_data_spaced(1000)))
        self.c_p['PSD_force_means'][1] += (
            np.mean(self.data_channels['PSD_A_F_Y'].get_data_spaced(1000)))
        self.c_p['PSD_force_means'][2] += (
            np.mean(self.data_channels['PSD_B_F_X'].get_data_spaced(1000)))
        self.c_p['PSD_force_means'][3] += (
            np.mean(self.data_channels['PSD_B_F_Y'].get_data_spaced(1000)))

        self.c_p['PSD_means'][0] = 32768 + np.uint16(self.c_p['PSD_force_means'][0])
        self.c_p['PSD_means'][1] = 32768 + np.uint16(self.c_p['PSD_force_means'][1])
        self.c_p['PSD_means'][2] = 32768 + np.uint16(self.c_p['PSD_force_means'][2])
        self.c_p['PSD_means'][3] = 32768 + np.uint16(self.c_p['PSD_force_means'][3])
        # print(self.c_p['PSD_means'][0], self.c_p['PSD_force_means'][0])
        
        self.c_p['portenta_command_1'] = 1

        # Zero the Z force by resetting the null offset
        self.c_p['Photodiode_sum_to_force'][2] -= np.mean(
            self.data_channels['F_total_Z'].get_data_spaced(1000))

    def reset_force_PSDs(self):
        self.c_p['PSD_force_means'] = [0, 0, 0, 0]
        self.c_p['portenta_command_1'] = 2
        

    def zero_position_PSDs(self):
        
        self.c_p['PSD_position_means'][0] += (
            np.mean(self.data_channels['PSD_A_P_X'].get_data_spaced(1000)))
        self.c_p['PSD_position_means'][1] += (
            np.mean(self.data_channels['PSD_A_P_Y'].get_data_spaced(1000)))
        self.c_p['PSD_position_means'][2] += (
            np.mean(self.data_channels['PSD_B_P_X'].get_data_spaced(1000)))
        self.c_p['PSD_position_means'][3] += (
            np.mean(self.data_channels['PSD_B_P_Y'].get_data_spaced(1000)))

        self.c_p['PSD_means'][0] = 32768 + np.uint16(self.c_p['PSD_position_means'][0])
        self.c_p['PSD_means'][1] = 32768 + np.uint16(self.c_p['PSD_position_means'][1])
        self.c_p['PSD_means'][2] = 32768 + np.uint16(self.c_p['PSD_position_means'][2])
        self.c_p['PSD_means'][3] = 32768 + np.uint16(self.c_p['PSD_position_means'][3])
        self.c_p['portenta_command_1'] = 4

    def reset_position_PSDs(self):
        self.c_p['PSD_position_means'] = [0, 0, 0, 0]
        self.c_p['portenta_command_1'] = 5

    def add_position(self, idx):
        position_command = partial(self.goto_position, idx)
        position_action = QAction(self.c_p['saved_positions'][idx][0], self)
        position_action.setStatusTip(
            f"Move to saved position {self.c_p['saved_positions'][idx][0]}")
        position_action.setToolTip(f"Move to saved position at: {self.c_p['saved_positions'][idx]}")
        position_action.triggered.connect(position_command)
        self.action_references[idx] = position_action
        self.saved_positions_submenu.addAction(position_action)

        self.delete_positon_action(idx)

    def remove_position(self, idx):
        print("Removing position", idx)
        if idx in self.action_references:
            position_action = self.action_references[idx]
            self.saved_positions_submenu.removeAction(position_action)
            self.delete_position_submenu.removeAction(self.delete_action_references[idx])
            del self.action_references[idx]  # Remove the reference from the dictionary
            del self.delete_action_references[idx]
            print("Action removed")

    def delete_positon_action(self, idx):
        delete_command = partial(self.remove_position, idx)
        delete_action = QAction(self.c_p['saved_positions'][idx][0], self)
        delete_action.setStatusTip(f"Delete saved position {self.c_p['saved_positions'][idx][0]}")
        delete_action.triggered.connect(delete_command)
        self.delete_position_submenu.addAction(delete_action)
        self.delete_action_references[idx] = delete_action

    def delete_position(self, idx):

        self.c_p['saved_positions'].pop(idx)

    def set_default_filename(self):
        text, ok = QInputDialog.getText(self, 'Filename dialog', 'Enter name of your files:')
        if ok:
            self.video_idx = 0
            self.data_idx = 0
            self.c_p['image_idx'] = 0
            self.c_p['filename'] = text
            self.c_p['video_name'] = text + '_video_' + str(self.video_idx)
            print(f"Filename is now {text}")

    def save_position(self):
        """
        Opens a dialog to save the current motor position with a user-defined name.
        The method retrieves the current X, Y, and Z positions from the motor controller,
        prompts the user to enter a name for the position, and if confirmed, saves the position
        along with its coordinates to the list of saved positions.
        """

        x = self.motor_controller.get_x_position()
        y = self.motor_controller.get_y_position()
        z = self.motor_controller.get_z_position()
        text, ok = QInputDialog.getText(self, 'Save position dialog', 'Enter name of position:')
        if ok:
            self.c_p['saved_positions'].append([text, x, y, z])
            print(f"Saved position {x}, {y} ,{z} as position: {text}")
            self.add_position(len(self.c_p['saved_positions'])-1)
        else:
            print("No position saved")

    def goto_position(self, idx):
        """
        Moves the system to a saved position specified by the given index.
        Parameters:
            idx (int): Index of the saved position to move to.
        """
        self.motor_controller.move_to_location(self.c_p['saved_positions'][idx][1:4])

    def drop_down_window_menu(self):
        # Create windows drop down menu
        window_menu = self.menu.addMenu("Windows")
        window_menu.addSeparator()

        self.open_plot_window = QAction("Live plotter", self)
        self.open_plot_window.setToolTip("Open live plotting window.")
        self.open_plot_window.triggered.connect(self.open_new_plot_window)
        self.open_plot_window.setCheckable(False)
        window_menu.addAction(self.open_plot_window)

        self.open_positions_window = QAction("Position PSDs", self)
        self.open_positions_window.setToolTip("""Open window for position PSDs.\n This is a
                                              specially configured version of the live plotter""")
        self.open_positions_window.triggered.connect(self.open_position_PSD_window)
        self.open_positions_window.setCheckable(False)
        window_menu.addAction(self.open_positions_window)

        self.open_force_window = QAction("Force PSDs", self)
        self.open_force_window.setToolTip("""Open window for force PSDs.\n This is a specially
                                          configured version of the live plotter""")
        self.open_force_window.triggered.connect(self.open_Force_PSD_window)
        self.open_force_window.setCheckable(False)
        window_menu.addAction(self.open_force_window)

        self.force_distance_window_action = QAction("Force distance curve - Y", self)
        self.force_distance_window_action.setToolTip("""Opens a plotting window with laser position 
                                                     reading and y-force""")
        self.force_distance_window_action.triggered.connect(self.open_force_distance_window_y)
        self.force_distance_window_action.setCheckable(False)
        window_menu.addAction(self.force_distance_window_action)

        self.force_distance_window_action_x = QAction("Force distance curve - X", self)
        self.force_distance_window_action_x.setToolTip(
            "Opens a plotting window with laser position reading and x-force")
        self.force_distance_window_action_x.triggered.connect(self.open_force_distance_window_x)
        self.force_distance_window_action_x.setCheckable(False)
        window_menu.addAction(self.force_distance_window_action_x)
        
        self.open_motor_action = QAction("Motor window", self)
        self.open_motor_action.setToolTip("Open window for manual motor control.")
        self.open_motor_action.triggered.connect(self.open_motor_control_window)
        self.open_motor_action.setCheckable(False)
        window_menu.addAction(self.open_motor_action)

        self.open_tracking_action = QAction("Tracking controls", self)
        self.open_tracking_action.setToolTip("Open window for tracking control.")
        self.open_tracking_action.triggered.connect(self.open_tracking_window)
        self.open_tracking_action.setCheckable(False)
        window_menu.addAction(self.open_tracking_action)
        
        self.open_laser_movement_action = QAction("Laser piezos window", self)
        self.open_laser_movement_action.setToolTip("Open window controlling piezos of lasers.")
        self.open_laser_movement_action.triggered.connect(self.open_laser_movement_window)
        self.open_laser_movement_action.setCheckable(False)
        window_menu.addAction(self.open_laser_movement_action)
        
        self.open_channel_viewer = QAction("Data channels", self)
        self.open_channel_viewer.setToolTip("""Opens a separate window in which the current values
                                            \n of the data channels is displayed.""")
        self.open_channel_viewer.triggered.connect(self.open_channels_winoow)
        self.open_channel_viewer.setCheckable(False)
        window_menu.addAction(self.open_channel_viewer)

        self.open_controlparameters_viewer_action = QAction("Control parameters", self)
        self.open_controlparameters_viewer_action.setToolTip("Opens a window for viewing.")
        self.open_controlparameters_viewer_action.triggered.connect(
            self.open_controlparameters_viewer)
        self.open_controlparameters_viewer_action.setCheckable(False)
        window_menu.addAction(self.open_controlparameters_viewer_action)

        self.auto_controller_action = QAction("Auto controller", self)
        self.auto_controller_action.setToolTip(
            "Opens a window for interfacing the auto controller.")
        self.auto_controller_action.triggered.connect(self.open_auto_controller_window)
        self.auto_controller_action.setCheckable(False)
        window_menu.addAction(self.auto_controller_action)

        self.open_stokes_action = QAction("Stokes test window", self)
        self.open_stokes_action.setToolTip(
            "Opens a window for the performing the stokes test autonomously.")
        self.open_stokes_action.triggered.connect(self.open_stokes_window)
        self.open_stokes_action.setCheckable(False)
        window_menu.addAction(self.open_stokes_action)

        self.open_laser_window_action = QAction("Laser controller", self)
        self.open_laser_window_action.setToolTip(
            "Opens a window for interfacing the laser controller.")
        self.open_laser_window_action.triggered.connect(self.open_laser_window)
        self.open_laser_window_action.setCheckable(False)
        window_menu.addAction(self.open_laser_window_action)

        self.open_pulling_protocoL_window  = QAction("Pulling protocol", self)
        self.open_pulling_protocoL_window.setToolTip("Opens a window for the pulling protocol.")
        self.open_pulling_protocoL_window.triggered.connect(self.open_pulling_protocol_window)
        self.open_pulling_protocoL_window.setCheckable(False)
        window_menu.addAction(self.open_pulling_protocoL_window)

        self.open_microfluidics_window = QAction("Microfluidics controller", self)
        self.open_microfluidics_window.setToolTip(
            "Opens a window for the microfluidics controller.")
        self.open_microfluidics_window.triggered.connect(self.open_microfluidics_window_func)
        self.open_microfluidics_window.setCheckable(False)
        window_menu.addAction(self.open_microfluidics_window)

        self.open_pump_config_action = QAction("Pump auto configuration", self)
        self.open_pump_config_action.setToolTip("""Opens a window for configuring the pump for
                                                autonomous use, selecting which channel is connected
                                                 to what type of particles.""")
        self.open_pump_config_action.triggered.connect(self.open_pump_config_window)
        self.open_pump_config_action.setCheckable(False)
        window_menu.addAction(self.open_pump_config_action)

    def open_channels_winoow(self):
        self.data_channel_viewer = CurrentValueWindow(self.c_p, self.data_channels)
        self.data_channel_viewer.show()

    def open_controlparameters_viewer(self):
        self.c_p_viewer = ControlParametersViewer(self.c_p)
        self.c_p_viewer.show()

    def set_video_format(self, video_format):
        self.c_p['video_format'] = video_format

    def open_motor_control_window(self):
        self.motor_control_dock.show()
        if self.motor_control_dock.isFloating():
            self.motor_control_dock.setFloating(False)

    def open_laser_window(self):

        self.laser_controller_dock.show()
        if self.laser_controller_dock.isFloating():
            self.laser_controller_dock.setFloating(False)

    def open_pulling_protocol_window(self):
        self.pulling_protocol_dock.show()
        if self.pulling_protocol_dock.isFloating():
            self.pulling_protocol_dock.setFloating(False)

    def open_microfluidics_window_func(self):
        self.microfludics_dock.show()
        if self.microfludics_dock.isFloating():
            self.microfludics_dock.setFloating(False)

    def open_laser_movement_window(self):
        self.laser_mover_dock.show()
        if self.laser_mover_dock.isFloating():
            self.laser_mover_dock.setFloating(False)


    def set_image_format(self, image_format):
        self.c_p['image_format'] = image_format
        
    def set_data_format(self, data_format):
        self.c_p['data_file_format'] = data_format

    def set_video_name(self, string):
        self.c_p['video_name'] = string

    def set_exposure_time(self):
        # Updates the exposure time of the camera to what is inside the textbox
        self.c_p['exposure_time'] = float(self.exposure_time_LineEdit.text())
        self.c_p['new_settings_camera'] = [True, 'exposure_time']

    def set_frame_rate(self):
        # Updates the frame rate of the camera to what is inside the textbox
        self.c_p['target_frame_rate'] = float(self.frame_rate_LineEdit.text())
        self.c_p['new_settings_camera'] = [True, 'frame_rate']

    def set_save_path(self):
        fname = QFileDialog.getExistingDirectory(self, "Save path")
        if len(fname) > 3:
            # If len is less than 3 then the action was cancelled and we should not update
            # the path.
            self.c_p['recording_path'] = fname

    def zoom_out(self):
        self.c_p['AOI'] = [0, self.c_p['camera_width'], 0,
                   self.c_p['camera_height']]
        self.c_p['new_settings_camera'] = [True, 'AOI']

    def toggle_recording(self):
        # Turns on/off recording
        # Need to add somehting to indicate the number of frames left to save when recording.
        self.c_p['recording'] = not self.c_p['recording']
        if self.c_p['recording']:
            self.c_p['video_name'] = self.c_p['filename'] + '_video' + str(self.video_idx)
            self.video_idx += 1
            self.record_action.setToolTip("Turn OFF recording.\n can also be toggled with CTRL+R")
        else:
            self.record_action.setToolTip("Turn ON recording.\n can also be toggled with CTRL+R")
        self.record_action.setChecked(self.c_p['recording'])

    def toggle_tracking_view(self):
        self.c_p['draw_particles'] = not self.c_p['draw_particles']
        self.c_p['draw_TnP_particles'] = not self.c_p['draw_TnP_particles']

    def toggle_draw_lasers(self):
        self.c_p['draw_lasers'] = not self.c_p['draw_lasers']

    def toggle_pipette_view(self):
        self.c_p['draw_pipette'] = not self.c_p['draw_pipette']
    
    def toggle_z_text(self):
        self.c_p['draw_z_text'] = not self.c_p['draw_z_text']
    
    def toggle_TnP_view(self):
        self.c_p['draw_TnP_particles'] = not self.c_p['draw_TnP_particles']

    def toggle_force_view(self):
        self.c_p['draw_force'] = not self.c_p['draw_force']

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_image()

    def resize_image(self):
        current_size = self.camera_window_label.size()
        width = current_size.width()
        height = current_size.height()
        self.c_p['frame_size'] = width, height


    def mouseMoveEvent(self, e):
        self.c_p['mouse_params'][3] = e.pos().x()-self.camera_window_label.pos().x()
        self.c_p['mouse_params'][4] = e.pos().y()-self.camera_window_label.pos().y()
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mouseMove()

    def mousePressEvent(self, e):
        
        self.c_p['mouse_params'][1] = e.pos().x()-self.camera_window_label.pos().x()
        self.c_p['mouse_params'][2] = e.pos().y()-self.camera_window_label.pos().y()

        if e.button() == Qt.MouseButton.LeftButton:
            self.c_p['mouse_params'][0] = 1
        if e.button() == Qt.MouseButton.RightButton:
            self.c_p['mouse_params'][0] = 2
        if e.button() == Qt.MouseButton.MiddleButton:
            self.c_p['mouse_params'][0] = 3
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mousePress()

    def mouseReleaseEvent(self, e):

        self.c_p['mouse_params'][3] = e.pos().x()-self.camera_window_label.pos().x()
        self.c_p['mouse_params'][4] = e.pos().y()-self.camera_window_label.pos().y()
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mouseRelease()
        self.c_p['mouse_params'][0] = 0

    def mouseDoubleClickEvent(self, e):        
        x = e.pos().x()-self.camera_window_label.pos().x()
        y = e.pos().y()-self.camera_window_label.pos().y()
        print(x*self.c_p['image_scale'] ,y*self.c_p['image_scale'] )
        self.c_p['click_tools'][self.c_p['mouse_params'][5]].mouseDoubleClick()

    def open_new_plot_window(self):
        if self.plot_windows is None:
            self.plot_windows = []
        self.plot_windows.append(PlotWindow(self.c_p,
                                            data=self.data_channels,
                                            x_keys=['T_time'],
                                            y_keys=['F_total_Y'],
                                            default_plot_length=80_000,
                                            ))

        self.plot_windows[-1].show()

    def open_position_PSD_window(self):
        if self.plot_windows is None:
            self.plot_windows = []
        self.plot_windows.append(PlotWindow(self.c_p, data=self.data_channels,
                                            x_keys=['PSD_A_P_X','PSD_B_P_X'],
                                            y_keys=['PSD_A_P_Y','PSD_B_P_Y'],
                                            aspect_locked=True,
                                            grid_on=True,
                                            title='Position PSDs'))
        self.plot_windows[-1].show()

    def open_Force_PSD_window(self):
        if self.plot_windows is None:
            self.plot_windows = []
        self.plot_windows.append(PlotWindow(self.c_p,
                                            data=self.data_channels,
                                            x_keys=['PSD_A_F_X','PSD_B_F_X'],
                                            y_keys=['PSD_A_F_Y','PSD_B_F_Y'],
                                            aspect_locked=True,
                                            grid_on=True,
                                            title='Force PSDs'))
        self.plot_windows[-1].show()

    def open_force_distance_window_y(self):
        """
        Opens a new plot window displaying the force-distance curve for the Y axis.
        This method creates and shows a PlotWindow instance configured to plot 'F_total_Y' versus 
        'Position_Y' using the provided data channels. The plot window is appended to the list of
        currently open plot windows.
        """
        if self.plot_windows is None:
            self.plot_windows = []
        self.plot_windows.append(PlotWindow(self.c_p,
                                            data=self.data_channels,
                                            x_keys=['Position_Y'],
                                            y_keys=['F_total_Y'],
                                            aspect_locked=False,
                                            grid_on=True,
                                            title='Force-distance curve',
                                            default_plot_length=200_000))
        self.plot_windows[-1].show()

    def open_force_distance_window_x(self):
        """
        Opens a new plot window displaying the force-distance curve for the X direction.
        This method creates and shows a PlotWindow instance that visualizes the relationship
        between the 'Position_X' and 'F_total_X' data channels. The plot window is configured
        with an unlocked aspect ratio, grid enabled, a specific title, and a default plot length.
        The new window is appended to the list of currently open plot windows.
        If the plot_windows attribute is None, it initializes it as an empty list before adding the
        new window.
        """

        if self.plot_windows is None:
            self.plot_windows = []
        self.plot_windows.append(PlotWindow(self.c_p,
                                            data=self.data_channels,
                                            x_keys=['Position_X'], y_keys=['F_total_X'],
                                            aspect_locked=False, grid_on=True,
                                            title='Force-distance curve',
                                            default_plot_length=200_000))
        self.plot_windows[-1].show()

    def open_pump_config_window(self):
        self.pump_config_window = ConfigurePumpWidget(self.c_p)
        self.pump_config_window.show()

    def open_auto_controller_window(self):
        self.auto_controller_window = AutoController.AutoControlWidget(self.c_p, self.data_channels,
                                                                       self.motor_controller)
        self.auto_controller_window.show()

    def open_stokes_window(self):
        self.stokes_test_window = AutoController.StokesTestWidget(self.c_p, self.motor_controller)
        self.stokes_test_window.show()

    def open_tracking_window(self):
        self.dep_learning_window = TrackingControlWidget(self.c_p)
        self.dep_learning_window.show()

    def closeEvent(self, event):
        if self.plot_windows is not None:
            for w in self.plot_windows:
                w.close()
        self.__del__

    def __del__(self):
        self.c_p['program_running'] = False
        if self.camera_thread is not None:
            self.camera_thread.join()
        if self.instrument_controller_thread is not None:
            self.instrument_controller_thread.join()
        if self.ArduinoUnoSerial is not None and self.ArduinoUnoSerial.is_open:
            self.ArduinoUnoSerial.close()
        self.data_saver_thread.join()
        self.video_writer_thread.join()

    def create_camera_toolbar(self):
        """
        Creates a toolbar for camera controls and settings.
        This method initializes a toolbar with various actions and input fields for controlling
        camera settings such as zoom, recording, snapshot, exposure time, frame rate, and gain.
        It also includes toggle buttons for displaying tracking, pipette, laser positions, Z text,
        and force view on the camera feed. Each action is connected to its respective method for
        handling the functionality.
        """
        
        self.camera_toolbar = QToolBar("Camera tools")
        self.addToolBar(self.camera_toolbar)
        
        self.zoom_action = QAction("Zoom out", self)
        self.zoom_action.setToolTip("Resets the field of view of the camera.\n CTRL+O")
        self.zoom_action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_O))
        self.zoom_action.triggered.connect(self.zoom_out)
        self.zoom_action.setCheckable(False)

        self.record_action = QAction("Record video", self)
        self.record_action.setToolTip("Turn ON recording.\n CTRL+R")
        self.record_action.setShortcut('Ctrl+R')
        self.record_action.triggered.connect(self.toggle_recording)
        self.record_action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_R))
        self.record_action.setCheckable(True)

        self.snapshot_action = QAction("Snapshot", self)
        self.snapshot_action.setToolTip("Take snapshot of camera view.\n CTRL+S")
        self.snapshot_action.triggered.connect(self.data_saver_thread.snapshot)
        self.snapshot_action.setCheckable(False)    
        # Create a shortcut and connect it to a custom method
        self.snapshot_action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_S))

        self.set_exp_tim = QAction("Set exposure time", self)
        self.set_exp_tim.setToolTip("Sets exposure time to the value in the textboox")
        self.set_exp_tim.triggered.connect(self.set_exposure_time)

        self.camera_toolbar.addAction(self.zoom_action)
        self.camera_toolbar.addAction(self.record_action)
        self.camera_toolbar.addAction(self.snapshot_action)

        # Add a toggle buttons for drawing
        self.toggle_tracking_view_action = QAction("Toggle tracking view", self)
        self.toggle_tracking_view_action.setToolTip(
            "Toggle whether the tracking should be drawn in real-time, on/off")
        self.toggle_tracking_view_action.setCheckable(True)
        self.toggle_tracking_view_action.triggered.connect(self.toggle_tracking_view)
        self.camera_toolbar.addAction(self.toggle_tracking_view_action)

        self.toggle_pipette_view_action = QAction("Toggle pipette view", self)
        self.toggle_pipette_view_action.setToolTip(
            "Toggle if pipette should be drawn, can be on/off")
        self.toggle_pipette_view_action.setCheckable(True)
        self.toggle_pipette_view_action.triggered.connect(self.toggle_pipette_view)
        self.camera_toolbar.addAction(self.toggle_pipette_view_action)

        self.toggle_laser_view_action = QAction("View laser position", self)
        self.toggle_laser_view_action.setToolTip(
            "Toggle if lasers should be drawn, can be on/off")
        self.toggle_laser_view_action.setCheckable(True)
        self.toggle_laser_view_action.triggered.connect(self.toggle_draw_lasers)
        self.camera_toolbar.addAction(self.toggle_laser_view_action)

        self.toggle_z_text_action = QAction("Toggle z text", self)
        self.toggle_z_text_action.setToolTip("Toggle if z text should be drawn, can be on/off")
        self.toggle_z_text_action.setCheckable(True)
        self.toggle_z_text_action.triggered.connect(self.toggle_z_text)
        self.camera_toolbar.addAction(self.toggle_z_text_action)

        self.draw_force_action = QAction("Draw force", self)
        self.draw_force_action.setToolTip(
            "Toggle if force acting on the trapped particle should be drawn, can be on/off")
        self.draw_force_action.setCheckable(True)
        self.draw_force_action.triggered.connect(self.toggle_force_view)
        self.camera_toolbar.addAction(self.draw_force_action)
        
        self.exposure_time_LineEdit = QLineEdit()
        self.exposure_time_LineEdit.setValidator(QDoubleValidator(0.99,99.99,2))
        self.exposure_time_LineEdit.setText(str(self.c_p['exposure_time']))
        self.camera_toolbar.addWidget(self.exposure_time_LineEdit)
        self.camera_toolbar.addAction(self.set_exp_tim)

        self.set_frame_rate_action = QAction("Set target fps", self)
        self.set_frame_rate_action.setToolTip("""Sets frame rate to the value in the textboox,\n
                                                    is an upper bound on the actual frame rate.""")
        self.set_frame_rate_action.triggered.connect(self.set_frame_rate)

        self.frame_rate_LineEdit = QLineEdit()
        self.frame_rate_LineEdit.setValidator(QDoubleValidator(0.1,99.99,2))
        self.frame_rate_LineEdit.setText(str(self.c_p['target_frame_rate']))
        self.camera_toolbar.addWidget(self.frame_rate_LineEdit)
        self.camera_toolbar.addAction(self.set_frame_rate_action)

        self.set_gain_action = QAction("Set gain", self)
        self.set_gain_action.setToolTip("Sets software gain to the value in the textboox")
        self.set_gain_action.triggered.connect(self.set_gain)

        self.gain_LineEdit = QLineEdit()
        self.gain_LineEdit.setToolTip("Set software gain on displayed image.")
        self.gain_LineEdit.setValidator(QDoubleValidator(0.1,3,3))
        self.gain_LineEdit.setText(str(self.c_p['image_gain']))
        self.camera_toolbar.addWidget(self.gain_LineEdit)
        self.camera_toolbar.addAction(self.set_gain_action)

        self.toggle_data_record_action = QAction("Start saving data", self)
        self.toggle_data_record_action.setToolTip(
            f"""Turn ON recodording of data.\n
            Data will be saved to fileneame set in files windows.\n
            Can save a maximum of {self.data_channels['T_time'].max_len} data points before 
            overwriting old ones.\n CTRL+D""")
        self.toggle_data_record_action.setCheckable(True)
        self.toggle_data_record_action.triggered.connect(self.record_data)
        self.toggle_data_record_action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_D))
        self.camera_toolbar.addAction(self.toggle_data_record_action)
        self.setStyleSheet("""
                QToolButton:pressed {
                background-color: lightblue; /* Temporary color on click */
                }
                QToolButton:checked {
                    background-color: lightgreen;
                }
                QToolButton {
                    background-color: lightgray;
                    border: 1px solid black;
                    border-radius: 1px;
                    padding: 1px;
                }
            """)
