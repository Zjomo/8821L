"""
Laser Control: Protocols, Mock Backend, and PyQt6 Widget

This module defines a protocol interface for laser control, a test/mock
implementation, and a PyQt6 widget that provides live control and an
automated power protocol (useful when changing currents during experiments).

Contents
--------
Protocols
    LaserController
        Hardware abstraction for a current-controlled laser source with
        connect/disconnect, output enable, and current set/read behavior.

Mock backend
    TestLaserController
        In-memory simulation of a laser controller (no hardware needed).

Qt widget
    LaserControllerWidget
        GUI for two lasers (A/B): connect, enable/disable output, set current,
        select COM ports, and run a timed power protocol that coordinates with
        an external acquisition GUI.

Shared state (c_p)
------------------
The widget integrates with a shared configuration/state dict `c_p`.
Typical keys (extend as needed):
    - 'program_running' : bool
    - 'recording_path' : str
    - 'recording' : bool
    - 'laser_A_port', 'laser_B_port' : str (e.g. "COM3")
    - 'laser_A_current', 'laser_B_current' : int (mA)
    - 'laser_A_current_current', 'laser_B_current_current' : int (mA, display)
    - 'laser_A_on', 'laser_B_on' : bool
    - 'laser_power_protocol_running' : bool
    - 'power_protocol_currents' : list[[mA_A, mA_B, duration_s], ...]
    - 'filename' : str (output naming set by the protocol)

Notes
-----
- The power protocol runs on a 100 ms QTimer and cycles through
  `c_p['power_protocol_currents']`. For each step, currents are set, an optional
  mid-interval snapshot is taken via `OT_GUI.snapshot()`, and recording is
  toggled via `OT_GUI.start_saving()/toggle_recording()/stop_saving()`.
- Two laser backends are supported simultaneously and addressed as A/B.
"""


from PyQt6.QtWidgets import (
 QVBoxLayout, QWidget, QLabel, QPushButton, QFormLayout, QHBoxLayout, QSpinBox,
)

from PyQt6.QtCore import QTimer
from typing import Protocol, runtime_checkable
from time import sleep, time
from functools import partial
import os



@runtime_checkable
class LaserController(Protocol):
    """
    Protocol for classes that control a laser source.

    Implementations must provide connection management, output gating, and a
    method to set the drive current (mA). Designed for interchangeable hardware
    backends.

    Methods
    -------
    connect() -> None
        Establish connection to the device (e.g., open serial port).
    disconnect() -> None
        Close the device connection.
    is_connected() -> bool
        Return True if the device is connected.
    set_current(current: float) -> None
        Set the laser drive current in mA (device-specific range/quantization).
    turn_on_output() -> None
        Enable laser output (interlock permitting).
    turn_off_output() -> None
        Disable laser output.
    is_output_on() -> bool
        Return True if the laser output is enabled.
    """
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...

    def set_current(self, current: float) -> None: ...

    def turn_on_output(self) -> None: ...
    def turn_off_output(self) -> None: ...
    def is_output_on(self) -> bool: ...


class TestLaserController(LaserController):
    """
    Mock laser controller for development and testing.

    Simulates a current-controlled laser with on/off output and a simple
    "connected" flag. No hardware interaction occurs.

    Attributes
    ----------
    current : float
        Simulated laser current (mA), clamped to [0, 400] in `set_current`.
    output_on : bool
        Output enable state.
    laser_ser : Any
        Truthy when "connected"; None when disconnected.

    Methods
    -------
    connect(adress)
        Mark as connected and print the target address.
    disconnect()
        Mark as disconnected and clear state.
    is_connected() -> bool
        Return connection status.
    set_current(current)
        Set simulated current; ignore values outside [0, 400] mA.
    turn_on_output(), turn_off_output()
        Toggle the simulated output enable.
    is_output_on() -> bool
        Return output enable state.

    Example
    -------
    >>> laser = TestLaserController()
    >>> laser.connect("COM5")
    Mock connect to laser at COM5
    >>> laser.set_current(120)
    Mock set current to 120 mA
    >>> laser.turn_on_output()
    Mock turn on laser output
    >>> laser.is_output_on()
    True
    """

    def __init__(self):
        self.current = 0
        self.output_on = False
        self.laser_ser = None

    def connect(self, adress):
        print(f"Mock connect to laser at {adress}")
        self.laser_ser = True

    def set_current(self, current):
        if int(current)>400 or int(current)<0:
            return
        self.current = current        
        print(f"Mock set current to {self.current} mA")

    def is_connected(self):
        return self.laser_ser is not None

    def is_output_on(self):
        return self.output_on

    def turn_on_output(self):
        print("Mock turn on laser output")
        self.output_on = True
    
    def turn_off_output(self):
        print("Mock turn off laser output")
        self.output_on = False

    def disconnect(self):
        print("Mock disconnect laser")
        self.laser_ser = None

class LaserControllerWidget(QWidget):
    """
    PyQt6 widget for controlling two laser sources (A and B) and running a timed
    power protocol.

    The widget connects to two `LaserController`-compatible backends, exposes
    on/off toggles, current setpoints (0–400 mA), COM port selectors, and a
    button to start/stop an automated power protocol that steps through
    `(mA_A, mA_B, duration_s)` tuples while coordinating file naming and data
    acquisition with an external GUI (`OT_GUI`).

    Parameters
    ----------
    c_p : dict
        Shared configuration/state dictionary. Expected keys include:
            - 'laser_A_port', 'laser_B_port' : str (e.g. "COM3")
            - 'laser_A_current', 'laser_B_current' : int (mA)
            - 'laser_A_current_current', 'laser_B_current_current' : int (mA)
            - 'laser_A_on', 'laser_B_on' : bool
            - 'laser_power_protocol_running' : bool
            - 'power_protocol_currents' : list[[mA_A, mA_B, duration_s], ...]
            - 'recording_path' : str
            - 'recording' : bool
            - 'program_running' : bool
            - 'filename' : str (set by the protocol)
    OT_GUI : object
        Acquisition/recording controller exposing:
            - `start_saving()`, `stop_saving()`, `toggle_recording()`, `snapshot()`.
    laser_A, laser_B : LaserController
        Laser backends implementing the `LaserController` protocol.

    UI Elements
    -----------
    • Per-laser (A/B):
        - Connect button (uses `c_p['laser_*_port']`)
        - Turn ON/OFF button (output enable)
        - SpinBox current setpoint (0–400 mA) + "Set current" button
        - COM port SpinBox (numeric suffix applied to "COM")
        - Live label for current readout (from `c_p`)
    • Common:
        - "Set both currents" button
        - "Toggle automatic power changing experiment" (checkable)

    Notes
    -----
    - A QTimer (100 ms) runs `laser_power_protocol()`:
        * Initializes at first step by setting currents, naming output files,
          and starting recording.
        * At mid-step (`duration_s/2`), triggers `OT_GUI.snapshot()`.
        * On step end, toggles recording, increments step, and repeats until
          all protocol entries are consumed.
    - The widget attempts to connect both lasers on initialization if not
      already connected.

    Methods
    -------
    initiate_interface()
        Build and layout the controls for lasers A and B and common actions.
    laser_power_protocol()
        Execute the timed stepping of currents and recording/snapshot logic.
    set_laser_A_current(), set_laser_B_current()
        Apply current setpoints to the respective laser.
    toggle_laser_A(), toggle_laser_B()
        Enable/disable output for the respective laser.
    set_laser_A_port(), set_laser_B_port()
        Update `c_p` with the selected COM port.
    set_both_currents()
        Convenience to set A and B currents together.
    connect_laser(laser)
        Connect either 'A' or 'B' device using `c_p`-configured port.
    closeEvent(event)
        Disconnect both lasers and accept the close event.
    """
    def __init__(self, c_p, OT_GUI, laser_A, laser_B):
        super().__init__()
        self.c_p = c_p
        self.OT_GUI = OT_GUI
        self.setWindowTitle("Laser controller")
        
        self.current_A_edit_val = self.c_p['laser_A_current']
        self.current_B_edit_val = self.c_p['laser_B_current']
        self.laser_A = laser_A
        self.laser_B = laser_B

        if not self.laser_A.is_connected():
            self.laser_A.connect(self.c_p['laser_A_port'])
        if not self.laser_B.is_connected():
            self.laser_B.connect(self.c_p['laser_B_port'])

        # create a timer for timed updates of lasers, used primarily when doing red blood cells experiments
        self.timer = QTimer(self)

        # set timer timeout callback function
        self.timer.timeout.connect(self.laser_power_protocol)

        # set the timer to fire every 100 milliseconds (10 times per second)
        self.timer.start(100)
        
        self.experiment_start_time = 0
        self.current_power_start_time = 0
        self.experiment_idx = 0
        self.experiment_started = False
        self.snapshot_taken = False
        self.time_interval = 10 # Decrease?
        self.particle_no = 0

        self.initiate_interface()        


    def get_name(self, idx):
        folder = f'\particle_no-{self.particle_no}'
        if not os.path.exists(self.c_p['recording_path']+folder):
            os.mkdir(self.c_p['recording_path']+folder)
        self.c_p['filename'] = (folder + '\particle_experiment_no-' + str(self.experiment_idx) +'_A' +
                                str(self.c_p['power_protocol_currents'][idx][0]) + '-B' + 
                                str(self.c_p['power_protocol_currents'][idx][1]))
        return

    def start_data_recording(self):
        self.OT_GUI.start_saving()
        if self.c_p['recording']:
            self.OT_GUI.toggle_recording()
            print("Warning recording was already on, turning off")
            time.sleep(0.05)
        self.OT_GUI.toggle_recording()
        self.snapshot_taken = False

    def stop_data_recording(self):
        self.OT_GUI.stop_saving()
        if not self.c_p['recording']:
            print("Warning recording was turned off")
            return
        self.OT_GUI.toggle_recording()

    def laser_power_currents(self):
        self.current_A_edit_val = int(self.c_p['power_protocol_currents'][self.experiment_idx][0])
        self.current_B_edit_val = int(self.c_p['power_protocol_currents'][self.experiment_idx][1])
        self.time_interval = int(self.c_p['power_protocol_currents'][self.experiment_idx][2])

        self.laser_A_current_box.setValue(int(self.current_A_edit_val))
        self.laser_B_current_box.setValue(int(self.current_B_edit_val))
        self.set_laser_A_current()
        self.set_laser_B_current()
        sleep(0.05)

    def laser_power_protocol(self):
        self.toggle_experiment_button.setChecked(self.c_p['laser_power_protocol_running'])
        if not self.c_p['laser_power_protocol_running']:
            if self.experiment_started:
                self.stop_data_recording()
                self.toggle_experiment_button.setChecked(False)
            self.experiment_start_time = 0
            self.current_power_start_time = 0
            self.experiment_idx = 0
            self.experiment_started = False
            return

        if self.experiment_idx == 0 and not self.experiment_started:
            self.experiment_started = True
            self.experiment_start_time = time()
            self.current_power_start_time = time()
            self.laser_power_currents()
            self.get_name(self.experiment_idx)
            self.particle_no += 1
            self.start_data_recording()
            
        if self.c_p['program_running'] and self.c_p['laser_power_protocol_running']:
            dt = time()-self.current_power_start_time

            # In the middle of the experiment, take a snapshot
            # TODO change snapshorts
            if dt > self.time_interval/2 and not self.snapshot_taken: 
                self.OT_GUI.snapshot()
                self.snapshot_taken = True

            if dt > self.time_interval:
                print(f"Stopped recording data{dt}\n {self.experiment_idx}")
                self.stop_data_recording()
                self.current_power_start_time = time()
                self.experiment_idx += 1
                if self.experiment_idx >= len(self.c_p['power_protocol_currents']):
                    print("Experiment done")
                    self.c_p['laser_power_protocol_running'] = False
                    self.experiment_started = False
                    self.toggle_experiment_button.setChecked(False)
                    return
                self.laser_power_currents()
                self.get_name(self.experiment_idx)
                self.start_data_recording()

    def initiate_interface(self):
        layout = QVBoxLayout()

        # Laser A
        self.laser_A_label = QLabel("Laser A", self)
        layout.addWidget(self.laser_A_label)
        self.laser_A_layout = QFormLayout()
        self.connectA_btn = QPushButton("Connect", self)
        connect_laser_A = partial(self.connect_laser, 'A')
        self.connectA_btn.clicked.connect(connect_laser_A)
        self.laser_A_layout.addRow("Connect", self.connectA_btn)
        self.toggle_A_btn = QPushButton("Turn ON", self)        
        self.toggle_A_btn.clicked.connect(self.toggle_laser_A)
        self.laser_A_layout.addRow("Output", self.toggle_A_btn)

        # Craete a layout for the current edit box and the set current button
        self.currentA_layout = QHBoxLayout()

        # Create spinbox for current
        self.laser_A_current_box = QSpinBox()
        self.laser_A_current_box.setRange(0, 400)
        self.laser_A_current_box.setValue(self.c_p['laser_A_current'])
        self.laser_A_current_box.valueChanged.connect(self.set_current_A)
        self.currentA_layout.addWidget(self.laser_A_current_box)

        self.set_current_button = QPushButton("Set current", self)
        self.set_current_button.clicked.connect(self.set_laser_A_current)
        self.currentA_layout.addWidget(self.set_current_button)

        self.current_now_A_label = QLabel(
            f"Current (mA): {self.c_p['laser_A_current_current']} ma", self)
        self.laser_A_layout.addRow(self.current_now_A_label)
        

        self.laser_A_layout.addRow("Set current", self.currentA_layout)
        
        self.laser_A_port_select = QSpinBox()
        self.laser_A_port_select.setRange(0, 20)  # Assuming you have 10 COM ports; adjust accordingly
        self.laser_A_port_select.setValue(int(self.c_p['laser_A_port'][3:])) 
        self.laser_A_port_select.valueChanged.connect(self.set_laser_A_port)
        self.laser_A_layout.addRow("COM Port", self.laser_A_port_select)

        layout.addLayout(self.laser_A_layout)


        # Laser B
        self.laserB_label = QLabel("Laser B", self)
        layout.addWidget(self.laserB_label)
        self.laserB_layout = QFormLayout()
        self.connectB_btn = QPushButton("Connect", self)
        connect_laserB = partial(self.connect_laser, 'B')

        self.connectB_btn.clicked.connect(connect_laserB)
        self.laserB_layout.addRow("Connect", self.connectB_btn)
        self.toggle_B_btn = QPushButton("Turn ON", self)        
        self.toggle_B_btn.clicked.connect(self.toggle_laser_B)
        self.laserB_layout.addRow("Power", self.toggle_B_btn)

        # Craete a layout for the current edit box and the set current button
        self.currentB_layout = QHBoxLayout()

        self.laser_B_current_box = QSpinBox()
        self.laser_B_current_box.setRange(0, 400)
        self.laser_B_current_box.setValue(self.c_p['laser_B_current'])
        self.laser_B_current_box.valueChanged.connect(self.set_current_B)
        self.currentB_layout.addWidget(self.laser_B_current_box)
        
        self.set_current_button = QPushButton("Set current", self)
        self.set_current_button.clicked.connect(self.set_laser_B_current)
        self.currentB_layout.addWidget(self.set_current_button)
        self.laserB_layout.addRow("Set current", self.currentB_layout)

        self.current_now_B_label = QLabel(
            f"Current (mA): {self.c_p['laser_B_current_current']} ma", self)
        self.laserB_layout.addRow(self.current_now_B_label)

        self.laser_B_port_select = QSpinBox()
        self.laser_B_port_select.setRange(0, 20)
        self.laser_B_port_select.setValue(int(self.c_p['laser_B_port'][3:])) 
        self.laser_B_port_select.valueChanged.connect(self.set_laser_B_port)
        self.laserB_layout.addRow("COM Port", self.laser_B_port_select)

        # Common stuff
        self.set_both_currents_button = QPushButton("Set currents", self)
        self.set_both_currents_button.clicked.connect(self.set_both_currents)
        self.laserB_layout.addRow("Set both currents", self.set_both_currents_button)

        self.toggle_experiment_button = QPushButton("Toggle automatic power changing experiment", self)
        self.toggle_experiment_button.clicked.connect(self.toggle_laser_power_experiment)
        self.toggle_experiment_button.setCheckable(True)
        self.laserB_layout.addRow("Toggle experiment", self.toggle_experiment_button)

       
        layout.addLayout(self.laserB_layout)

        self.setLayout(layout)

    def set_laser_A_port(self):
        value = self.laser_A_port_select.value()
        self.c_p['laser_A_port'] = f"COM{value}"

    def set_laser_B_port(self, value):
        value = self.laser_B_port_select.value()
        self.c_p['laser_B_port'] = f"COM{value}"

    def connect_laser(self, laser):
        if laser == 'A':
            self.laser_A.connect(self.c_p['laser_A_port'])

        elif laser == 'B':
            self.laser_B.connect(self.c_p['laser_B_port'])

    def set_current_A(self, current):
        self.current_A_edit_val = int(current)
    
    def set_current_B(self, current):
        self.current_B_edit_val = int(current)

    def toggle_laser_power_experiment(self):
        self.c_p['laser_power_protocol_running'] = not self.c_p['laser_power_protocol_running']

    def set_both_currents(self):
        self.set_laser_A_current()
        self.set_laser_B_current()

    def set_laser_A_current(self):
        current = self.laser_A_current_box.value()
        self.laser_A.set_current(current)

    def set_laser_B_current(self):
        current = self.laser_B_current_box.value()
        self.laser_B.set_current(current)

    def toggle_laser_A(self):        
        if self.laser_A.is_output_on():
            self.laser_A.turn_off_output()
            self.toggle_A_btn.setText("Turn ON")
            self.c_p['laser_A_on'] = False
        else:
            self.laser_A.turn_on_output()
            self.toggle_A_btn.setText("Turn OFF")
            self.c_p['laser_A_on'] = True

    def toggle_laser_B(self):
        if self.laser_B.is_output_on():
            self.laser_B.turn_off_output()
            self.toggle_B_btn.setText("Turn ON")
            self.c_p['laser_B_on'] = False
        else:
            self.laser_B.turn_on_output()
            self.toggle_B_btn.setText("Turn OFF")
            self.c_p['laser_B_on'] = True

    def disconnect_laser(self, laser):
        if laser == 'A' and self.laser_A.is_connected():
            self.laser_A.disconnect()
        elif laser == 'B' and self.laser_B.is_connected():
            self.laser_B.disconnect()

    def closeEvent(self, event):
        self.disconnect_laser('A')
        self.disconnect_laser('B')
        event.accept()

