"""
This file defines the classes used to control the motors of the instrument.
----------------------------------
Classes:

- Motor: Abstract class which defines an interface for the motors used to control the motors
steering the sample stage.
- ObjectiveMovement: Abstract class which is used for the motorized movement objective movement.
- TestMotorController: A mock implementation of the Motor used for testing.
- TestObjectiveMovement: A mock implementation of the ObjectiveMovement used for testing.
- MotorControllerWindow: QWidget which contains buttons for manuall control of the motors and for
setting the speed of them.
- MotorMouseMove: A MouseTool used for moving the motors with the mouse (clicking or clicking and
dragging on the camera feed screen).
- ObjectiveStepperControllerToolbar: Toolbar for controlling the objective position.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Sequence

from PyQt6.QtWidgets import (    
    QLineEdit, QCheckBox, QPushButton, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QGridLayout,
    QToolBar
)

from PyQt6.QtGui import QDoubleValidator, QKeySequence, QAction
from mouse_tool import MouseTool
from time import time
import numpy as np


@runtime_checkable
class Motor(Protocol):
    """
    Protocol for an 3-axis motorized stage.

    Implementations must expose basic kinematics (absolute move, velocity move),
    speed control, position readout, motion state, and stop.

    Conventions
    -----------
    - Positions are sequences of floats with length equal to the number of axes.
    - Velocity commands are passed as varargs, e.g. ``move_at_speed(vx, vy[, vz, ...])``.
    - Units are implementation-defined; by convention ``set_speed`` uses microns/s.

    Methods
    -------
    set_speed(mm_per_s: float) -> None
        Set the nominal stage speed (per-axis moves scale from this).
    get_speed() -> float
        Return the current nominal speed.
    move_to_location(position: Sequence[float]) -> None
        Move to an absolute multi-axis position.
    move_at_speed(*v: float) -> None
        Command a constant-velocity move with per-axis components.
    get_position() -> Sequence[float]
        Return the current multi-axis position.
    stop_moving() -> None
        Halt all motion immediately/safely.
    is_moving() -> bool
        Return True if any axis is currently in motion.
    """
    def set_speed(self, mm_per_s: float) -> None: ...
    def get_speed(self) -> float: ...

    def move_to_location(self, position: Sequence[float]) -> None: ...
    def move_at_speed(self, *v: float) -> None: ...

    def get_position(self) -> Sequence[float]: ...
    def stop_moving(self) -> None: ...
    def is_moving(self) -> bool: ...


@runtime_checkable
class ObjectiveMovement(Protocol):
    """
    Protocol for coarse/fine objective motion relative to the sample.

    Intended for microscope objective actuators that offer discrete speed
    presets in the approach/withdraw directions e.g. stepper motors.

    Methods
    -------
    connect() -> None
        Establish connection to the actuator.
    is_connected() -> bool
        Return True if the actuator is connected.
    slow_towards_sample() -> None
        Move slowly towards the sample.
    slow_away_from_sample() -> None
        Move slowly away from the sample.
    fast_towards_sample() -> None
        Move quickly towards the sample.
    fast_away_from_sample() -> None
        Move quickly away from the sample.
    """
    def connect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def slow_towards_sample(self) -> None: ...
    def slow_away_from_sample(self) -> None: ...
    def fast_towards_sample(self) -> None: ...
    def fast_away_from_sample(self) -> None: ...



class TestMotorController(Motor):
    """
    Minimal motor controller for testing.

    Simulates a 3-axis stage with position state and a nominal speed. All
    motions are instantaneous except for ``move_at_speed``, which integrates a
    one-second step for demonstration.

    Attributes
    ----------
    speed : float
        Nominal speed (default 10.0; capped at 100.0 in ``set_speed``).
    x_pos, y_pos, z_pos : float
        Current simulated positions.

    Methods
    -------
    set_speed(speed)
        Set the nominal speed; values > 100 are clipped to 100.
    get_speed()
        Return the current nominal speed.
    move_to_location(position)
        Instantly set (x, y, z) to the provided absolute position.
    move_at_speed(x_speed, y_speed, z_speed=0)
        Advance the position by 1 s worth of the given per-axis speeds.
    get_x_position(), get_y_position(), get_z_position()
        Convenience accessors for per-axis positions.
    stop_moving()
        Print a stop message (no internal state change).
    is_moving()
        Always returns False (no background motion simulation).

    Example
    -------
    >>> m = TestMotorController()
    >>> m.set_speed(50)
    >>> m.move_to_location([0, 0, 0])
    >>> m.move_at_speed(1, -2, 0.5)  # integrates ~1 s step
    >>> m.get_speed(), (m.get_x_position(), m.get_y_position(), m.get_z_position())
    (50.0, (1.0, -2.0, 0.5))
    """

    def __init__(self):
        self.speed = 10.0
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0

    def set_speed(self, speed):
        if np.abs(float(speed)) < 100:
            self.speed = float(speed)
        else:
            self.speed = float(100)

    def get_speed(self):
        return self.speed

    def move_to_location(self, position):
        print(f"Moving to location {position}")
        self.x_pos = position[0]
        self.y_pos = position[1]
        self.z_pos = position[2]

    def get_x_position(self):
        return self.x_pos
    
    def get_y_position(self):
        return self.y_pos
    
    def get_z_position(self):
        return self.z_pos

    def stop_moving(self):
        print("Stopping movement")

    def is_moving(self):
        return False

    def move_at_speed(self, x_speed, y_speed, z_speed=0):
        print(f"Moving at speed x: {x_speed}, y: {y_speed}, z: {z_speed}")
        self.x_pos += x_speed * 1
        self.y_pos += y_speed * 1
        self.z_pos += z_speed * 1

class TestObjectiveMovement(ObjectiveMovement):
    """
    Minimal objective actuator for testing.

    Provides connect status and prints actions for slow/fast moves in the
    towards/away directions. No physical motion is performed.

    Attributes
    ----------
    connected : bool
        Connection flag toggled by ``connect``.

    Methods
    -------
    connect(adress)   # sic: parameter name kept for compatibility
        Mark the actuator as connected and print the address.
    is_connected()
        Return the connection flag.
    slow_towards_sample(), slow_away_from_sample()
        Print slow movement direction.
    fast_towards_sample(), fast_away_from_sample()
        Print fast movement direction.

    Example
    -------
    >>> obj = TestObjectiveMovement()
    >>> obj.connect("COM3")
    Connected to objective at COM3
    >>> obj.fast_towards_sample()
    Moving fast towards sample
    >>> obj.is_connected()
    True
    """
    def __init__(self):
        self.connected = False

    def connect(self, adress):
        self.connected = True
        print(f"Connected to objective at {adress}")

    def is_connected(self):
        return self.connected

    def slow_towards_sample(self):
        print("Moving slow towards sample")

    def slow_away_from_sample(self):
        print("Moving slow away from sample")

    def fast_towards_sample(self):
        print("Moving fast towards sample")

    def fast_away_from_sample(self):
        print("Moving fast away from sample")

class MotorControllerWindow(QWidget):
    """
    Qt widget for manual control of a motorized sample stage.

    The widget provides interactive controls to set motor speed and to move
    the stage along the x, y, and z axes using either buttons or keyboard controls. 

    Parameters
    ----------
    c_p :
        Shared configuration/state dictionary. Expected keys (minimum):
            - 'blue_led' : int
                State of the sample illumination LED.
                Convention: 0 = ON, 1 = OFF.
    motor_controller : Motor
        Motor controller interface. Must implement:
            - ``get_speed() -> float``
            - ``set_speed(value: float)``
            - ``move_at_speed(x: float, y: float, z: float)``
            - ``stop_moving()``

    UI Elements
    -----------
    • Speed control:
        - QLabel showing speed units (µm/s).
        - QLineEdit for setting motor speed with validation (0–100 µm/s).
        - Preset QPushButtons for 1, 10, and 100 µm/s.
    • Movement arrows (QGridLayout):
        - ↑ Up, ← Left, ↓ Down, → Right.
          Each button supports both press-and-hold and keyboard shortcuts.
    • Sample z-axis:
        - QPushButtons for "Sample forward" (PgUp) and "Sample backward" (PgDn).
    • LED toggle:
        - QCheckBox to switch the sample illumination LED on/off.

    Notes
    -----
    - Keyboard shortcuts mirror the arrow and page up/down keys for ergonomic
      control without the mouse.
    - Button press events initiate stage motion at the current set speed.
      Release events stop motion (for continuous movement).
    - LED state is mapped to ``c_p['blue_led']`` with convention
      0 = ON, 1 = OFF.

    Methods
    -------
    toggle_led(state)
        Toggle LED state and update ``c_p['blue_led']`` accordingly.
    move_up()
        Move stage in the positive y direction at current speed.
    stop_y()
        Stop stage motion along the y axis.
    move_down()
        Move stage in the negative y direction at current speed.
    move_right()
        Move stage in the negative x direction at current speed.
    stop_x()
        Stop stage motion along the x axis.
    move_left()
        Move stage in the positive x direction at current speed.
    objective_forward()
        Move stage in the positive z direction at current speed.
    objective_stop()
        Stop stage motion along the z axis.
    objective_backward()
        Move stage in the negative z direction at current speed.

    Example
    -------
    >>> c_p = {"blue_led": 1}
    >>> motor = MyMotorController()
    >>> window = MotorControllerWindow(c_p, motor)
    >>> window.show()
    """
        
    def __init__(self, c_p, motor_controller: "Motor"):
        super().__init__()
        self.c_p = c_p
        self.motor_controller = motor_controller
        self.setWindowTitle("Motor controller")

        main = QVBoxLayout()

        # --- Speed label + edit ---
        self.label = QLabel("Set motor speed (0–100) microns/s")
        main.addWidget(self.label)

        self.SpeedLineEdit = QLineEdit()
        self.SpeedLineEdit.setValidator(QDoubleValidator(0.0, 100.0, 1, self))
        self.SpeedLineEdit.setText(str(self.motor_controller.get_speed()))
        self.SpeedLineEdit.setToolTip("""Sets the motor speed. Units are in microns/second.\n
                                      The mose movement is independent of the speed set here.""")
        self.SpeedLineEdit.textChanged.connect(self.motor_controller.set_speed)
        main.addWidget(self.SpeedLineEdit)

        # --- Presets row ---
        presets = QHBoxLayout()
        self.slow_speed_button = QPushButton('1 micron/s')
        self.slow_speed_button.clicked.connect(lambda: self.SpeedLineEdit.setText("1"))
        presets.addWidget(self.slow_speed_button)
        self.medium_speed_button = QPushButton('10 microns/s')
        self.medium_speed_button.clicked.connect(lambda: self.SpeedLineEdit.setText("10"))
        presets.addWidget(self.medium_speed_button)
        self.fast_speed_button = QPushButton('100 microns/s')
        self.fast_speed_button.clicked.connect(lambda: self.SpeedLineEdit.setText("100"))
        presets.addWidget(self.fast_speed_button)
        main.addLayout(presets)

        # --- Arrows (grid) ---
        arrows = QGridLayout()
        self.up_button = QPushButton("↑ Up")
        self.up_button.pressed.connect(self.move_up)
        self.up_button.released.connect(self.stop_y)
        self.up_button.setShortcut(QKeySequence("Up"))
        self.up_button.setToolTip("Moves left in the sample. Also accessible via the left arrow key.")
        arrows.addWidget(self.up_button, 0, 1)

        self.left_button = QPushButton("← Left")
        self.left_button.setShortcut(QKeySequence("Left"))
        self.left_button.pressed.connect(self.move_left)
        self.left_button.released.connect(self.stop_x)
        self.left_button.setToolTip("Moves down in the sample. Also accessible via the down arrow key.")
        arrows.addWidget(self.left_button, 1, 0)

        self.down_button = QPushButton("↓ Down")
        self.down_button.setShortcut(QKeySequence("Down"))
        self.down_button.pressed.connect(self.move_down)
        self.down_button.released.connect(self.stop_y)
        self.down_button.setToolTip("Moves right in the sample. Also accessible via the right arrow key.")
        arrows.addWidget(self.down_button, 1, 1)

        self.right_button = QPushButton("→ Right")
        self.right_button.setShortcut(QKeySequence("Right"))
        self.right_button.pressed.connect(self.move_right)
        self.right_button.setToolTip("Moves up in the sample. Also accessible via the up arrow key.")
        self.right_button.released.connect(self.stop_x)
        arrows.addWidget(self.right_button, 1, 2)

        main.addLayout(arrows)

        # --- Sample forward/backward ---
        sample = QHBoxLayout()
        self.objective_forward_button = QPushButton("Sample forward")
        self.objective_forward_button.setShortcut(QKeySequence("PgUp"))
        self.objective_forward_button.pressed.connect(self.objective_forward)
        self.objective_forward_button.released.connect(self.objective_stop)
        self.objective_forward_button.setToolTip("Moves the sample forwards towards the imaging objective. Also accessible via the page up key.")
        sample.addWidget(self.objective_forward_button)

        self.objective_backward_button = QPushButton("Sample backward")
        self.objective_backward_button.setShortcut(QKeySequence("PgDown"))
        self.objective_backward_button.pressed.connect(self.objective_backward)
        self.objective_backward_button.released.connect(self.objective_stop)
        self.objective_backward_button.setToolTip("Moves the sample backwards towards the imaging objective. Also accessible via the page down key.")
        sample.addWidget(self.objective_backward_button)

        main.addLayout(sample)

        # --- LED toggle ---
        main.addWidget(QLabel("Sample LED ON/OFF"))
        self.led_button = QCheckBox()
        self.led_button.setChecked(self.c_p['blue_led'] == 0)
        self.led_button.setStyleSheet("""
            QCheckBox::indicator {
                width: 30px;
                height: 30px;
            }
        """)
        main.addWidget(self.led_button)

        self.setLayout(main)

    def toggle_led(self, state):
        """
        Toggle the saving_toggled property of the DataChannel when the checkbox is toggled.
        """
        self.c_p['blue_led'] = 0 if bool(state) else 1
        if self.c_p['blue_led'] == 0:
            print("LED on")
        else:
            print("LED off")

    def move_up(self):
        self.motor_controller.move_at_speed(self.motor_controller.get_speed(),0,0)

    def stop_y(self):
        self.motor_controller.stop_moving()

    def move_down(self):
        self.motor_controller.move_at_speed(-self.motor_controller.get_speed(),0,0)

    def move_right(self):
        self.motor_controller.move_at_speed(0,-self.motor_controller.get_speed(),0)

    def stop_x(self):
        self.motor_controller.stop_moving()

    def move_left(self):
        self.motor_controller.move_at_speed(0,self.motor_controller.get_speed(),0)

    def objective_forward(self):
        self.motor_controller.move_at_speed(0,0,self.motor_controller.get_speed())

    def objective_stop(self):
        self.motor_controller.stop_moving()

    def objective_backward(self):
        self.motor_controller.move_at_speed(0,0,-self.motor_controller.get_speed())

class MotorMouseMove(MouseTool):
    """
    Mouse-based motor control tool for interactive sample movement.

    This tool extends :class:`MouseTool` to translate user mouse actions into
    motor stage commands. It allows centering by click, dragging for continuous
    x/y motion, and vertical control via right-click or scroll-like drag.

    Parameters
    ----------
    c_p : dict
        Shared configuration/state dictionary. Expected keys include:
        - 'mouse_params' : list
            Encodes mouse interaction state as [button, x_click, y_click,
            x_move, y_move].
        - 'camera_width', 'camera_height' : int
            Camera dimensions in pixels.
        - 'AOI' : list[int]
            Active area of interest [x0, x1, y0, y1].
        - 'image_scale' : float
            Scale factor between pixels and world units.
        - 'ticks_per_pixel' : float
            Conversion factor from pixels to motor ticks.
    data_channels : dict
        Metadata about active data streams (used for knowing current position
        and speed).
    MotorController : Motor
        Motor controller instance implementing the :class:`Motor` protocol.

    Attributes
    ----------
    x_0, y_0, z_0 : int
        Initial reference positions for movement.
    x_prev, y_prev, z_prev : int
        Previous mouse positions for drag calculations.
    prev_t : float
        Timestamp of last update.
    speed_factor : float
        Scaling factor applied to mouse movement → motor velocity.

    Mouse interactions
    ------------------
    - Left click (button=1):  
      Re-center stage to the clicked point, adjusted for AOI and scale.
    - Right click (button=2):  
      Drag to move the stage continuously in x/y. Stops if mouse does not move.
    - Middle click (button=3):  
      Drag vertically to move the stage in z.

    Methods
    -------
    mousePress()
        Handle click or drag start: recenter on left click, prepare drag state
        on right/middle click.
    mouseRelease()
        Stop stage motion when mouse button is released.
    mouseDoubleClick()
        No-op (placeholder for future).
    draw(qp)
        No-op (placeholder for visualization).
    check_speed(speed) -> int
        Clamp speed to the range [-32767, 32767], enforcing a minimum ±2.
    mouseMove()
        Handle continuous drag movement:  
        - Right drag → call ``move_at_speed`` in x/y.  
        - Middle drag → call ``move_at_speed`` in z.
    getToolName() -> str
        Return tool name "Minitweezers motor".
    getToolTip() -> str
        Return short description of tool functionality.
    """
    
    def __init__(self, c_p, data_channels, MotorController):
        self.c_p = c_p
        self.data_channels = data_channels # Needed to know the position and speed
        self.x_0 = 0
        self.y_0 = 0
        self.z_0 = 0
        self.x_prev = 0
        self.y_prev = 0
        self.z_prev = 0
        self.prev_t = time()
        self.speed_factor = 4
        self.MotorController = MotorController

    def mousePress(self):
        # left click
        if self.c_p['mouse_params'][0] == 1:
            center_x = int((self.c_p['camera_width']/2 - self.c_p['AOI'][0])/self.c_p['image_scale'])
            center_y = int((self.c_p['camera_height']/2 - self.c_p['AOI'][2])/self.c_p['image_scale'])

            # Checking that the click was made in an ok position
            width = (self.c_p['AOI'][1] - self.c_p['AOI'][0])/self.c_p['image_scale'] / 2
            height = (self.c_p['AOI'][3] - self.c_p['AOI'][2])/self.c_p['image_scale'] / 2
            if self.c_p['mouse_params'][1] < center_x - width or self.c_p['mouse_params'][1] > center_x + width:
                return
            if self.c_p['mouse_params'][2] < center_y - height or self.c_p['mouse_params'][2] > center_y + height:
                return

            dx_ticks = (self.c_p['mouse_params'][1] - center_x) * self.c_p['image_scale']*self.c_p['ticks_per_pixel']
            dy_ticks = (self.c_p['mouse_params'][2] - center_y) * self.c_p['image_scale']*self.c_p['ticks_per_pixel']

            x = int(self.MotorController.get_x_position() + dx_ticks)
            y = int(self.MotorController.get_y_position() - dy_ticks)

            self.MotorController.move_to_location((x,y,self.MotorController.get_z_position()))
            
  
        # Right click -drag
        if self.c_p['mouse_params'][0] == 2:
            if self.x_prev == self.c_p['mouse_params'][1] and self.y_prev == self.c_p['mouse_params'][2]:
                self.MotorController.stop_moving()
                return
            self.x_prev = self.c_p['mouse_params'][1]
            self.y_prev = self.c_p['mouse_params'][2]

        if self.c_p['mouse_params'][0] == 3:
            self.z_0 = self.c_p['mouse_params'][2]
            self.z_prev = self.c_p['mouse_params'][2]

    def mouseRelease(self):
        self.MotorController.stop_moving()

    def mouseDoubleClick(self):
        pass
    
    def draw(self, qp):
        pass
    
    def check_speed(self, speed):
        if 0 < speed < 2:
            return 2
        if 0 > speed > -2:
            return -2
        if speed > 32767:
            return 32767
        if speed < -32767:
            return  -32767
        return speed

    def mouseMove(self):
        """
        This function is called when the mouse is moved. It is used to move the motors
        in real time.
        """
        if self.c_p['mouse_params'][0] == 2:

            dx = (self.c_p['mouse_params'][3] - self.x_prev)
            dy = (self.c_p['mouse_params'][4] - self.y_prev)
            x_speed = self.check_speed(dx * self.speed_factor)
            y_speed = self.check_speed(dy * self.speed_factor)
            
            self.MotorController.move_at_speed(int(y_speed), int(x_speed), 0) # x and y are mirrored
            self.x_prev = self.c_p['mouse_params'][3]
            self.y_prev = self.c_p['mouse_params'][4]
            
        elif self.c_p['mouse_params'][0] == 3:
            dz = (self.c_p['mouse_params'][4] - self.z_prev)
            z_speed = self.check_speed(dz * self.speed_factor/5)            
            self.MotorController.move_at_speed(0,0,int(z_speed))
            self.z_prev = self.c_p['mouse_params'][4]

    def getToolName(self):
        return "Minitweezers motor"

    def getToolTip(self):
        return "Move the motors by clicking or dragging on the screen"


class ObjectiveStepperControllerToolbar(QToolBar):
    """
    Simple toolbar to control the objective stepper motor. Can move the stepper motor
    either towards or away from the sample. Can move slowly or fast (big or small steps).

    Parameters
    -------------
    objective_mover: An ObjectiveMovement object, assumed to be connected alredy.

    parent: A QT main window into which the toolbar can be placed.

    Methods
    -------------
    
    UI elements
    -------------
    Has 4 buttons for movement fast and slow towards and away from the sample.
    """
    def __init__(self, objective_mover, parent):
        super().__init__("Objective Controller", parent)
        self.objective_mover = objective_mover
        
        # Add move towards action
        self.slow_towards_action = QAction('Slow towards sample')
        self.slow_towards_action.triggered.connect(self.objective_mover.slow_towards_sample)
        self.addAction(self.slow_towards_action)

        # Add move away action
        self.slow_away_action = QAction('Slow away from sample')
        self.slow_away_action.triggered.connect(self.objective_mover.slow_away_from_sample)   
        self.addAction(self.slow_away_action)

        # Add move fast towards action
        self.fast_towards_action = QAction('Fast towards sample')
        self.fast_towards_action.triggered.connect(self.objective_mover.fast_towards_sample)
        self.addAction(self.fast_towards_action)

        # Add move fast away action
        self.fast_away_sample_action = QAction('Fast away from sample')
        self.fast_away_sample_action.triggered.connect(self.objective_mover.fast_away_from_sample)
        self.addAction(self.fast_away_sample_action)