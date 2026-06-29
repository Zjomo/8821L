
"""
Laser Experiment Protocols & UI
================================

High-level protocols for scripted laser/particle control and a Qt widget
to configure and run them. Protocols share a uniform interface so new
behaviors can be added with minimal boilerplate.

Overview
--------
- `ExperimentLaserProtocol` defines the protocol API (start/stop/run/state,
  parameter getters/setters with descriptions/tooltips/limits).
- Concrete protocols translate user parameters into compact `protocol_data`
  messages and device commands via `c_p`.
- `PullingProtocolWidget` exposes a small UI to select a protocol, edit
  parameters, and toggle execution; it calls `run_protocol()` at ~25 Hz.
  Allowing for dynamic control also by the host computer.

Shared State
------------
All protocols expect a shared dictionary `c_p` with (at minimum):
- `protocol_data`: `List[int]` of length ≥ 7 for device-side commands.
- `portenta_command_2`: `int` indicating which trap is auto-aligned (1=A, 2=B).
- `PSD_to_force`: `Tuple[float, float]` for converting PSD units → pN.

Notes
-----
- Protocols must be non-blocking: `run_protocol()` advances one step and
  returns immediately; the widget’s timer drives progression.
- Parameter arrays are lists of floats; use `get_parameter_*` helpers to
  populate labels, tooltips, and value limits in the UI.
"""

from __future__ import annotations
from typing import Protocol, List, Any, Dict

from PyQt6.QtWidgets import (QVBoxLayout,QFormLayout, QLabel,  QWidget, QPushButton, QComboBox,
                             QGridLayout, QDoubleSpinBox)

from PyQt6.QtCore import QTimer
import numpy as np
from time import sleep, time


class ExperimentLaserProtocol(Protocol):
    """
    Protocol interface for scripted laser control.

    Methods
    -------
    start_protocol() -> None
        Arm/start the protocol; should not block.
    run_protocol() -> None
        Advance one non-blocking step; safe to call from a timer.
    stop_protocol() -> None
        Disarm/stop the protocol; should not block.
    is_running() -> bool
        Return True while the protocol is active.

    Text metadata
    -------------
    get_protocol_name() -> str
        Short, UI-friendly name.
    get_protocol_description() -> str
        One–two line description for the widget.

    Parameters
    ----------
    get_parameters() -> List[float]
        Current parameter values.
    set_parameters(params: List[float]) -> None
        Update parameters (use None to keep a value unchanged).
    get_parameter_descriptions() -> List[str]
        Human-readable labels for parameters.
    get_parameter_tooltips() -> List[str]
        Tooltips for parameter editors.
    get_parameter_limits() -> List[List[float]]
        Per-parameter [min, max] bounds.
    """
    # Control
    def start_protocol(self) -> None: ...
    def run_protocol(self) -> None: ...
    def stop_protocol(self) -> None: ...
    def is_running(self) -> bool: ...

    # Text
    def get_protocol_description(self) -> str: ...
    def get_protocol_name(self) -> str: ...

    def get_parameter_descriptions(self) -> List[str]: ...
    def get_parameter_tooltips(self) -> List[str]: ...
    def get_parameter_limits(self) -> List[List[float]]: ...

    # Params (list-of-floats)
    def get_parameters(self) -> List[float]: ...
    def set_parameters(self, params: List[float]) -> None: ...


class ConstantSpeedProtocol(ExperimentLaserProtocol):
    """
    Moves the laser at constant speed between two positions along the chosen axis (x or y).
    Automatically toggles the autoalign function.

    """

    _AXIS_NAMES = {1: "X", 2: "Y"}
    def __init__(self, c_p, axis=1):
            if axis not in self._AXIS_NAMES:
                raise ValueError(f"Axis {axis!r} not supported. Choose one of {list(self._AXIS_NAMES)}.")

            self.c_p = c_p
            self.axis = int(axis)
            self.axis_name = self._AXIS_NAMES[axis]
            self._running = False

            # Defaults (floats)
            self._defaults: List[float] = [5000.0, 50000.0, 10.0]
            self._params: List[float] = self._defaults.copy()

    def get_protocol_name(self) -> str:
        return "Constant speed " + self.axis_name + " axis"

    def set_parameters(self, parameters) -> None:
        if parameters[0] is not None:
            lower_pos_limit = parameters[0]
        else:
            lower_pos_limit = self._params[0]
        if parameters[1] is not None:
            upper_pos_limit = parameters[1]
        else:
            upper_pos_limit = self._params[1]
        if parameters[2] is not None:
            self._params[2] = parameters[2]

        if 0 <= lower_pos_limit < upper_pos_limit:
            self._params[0] = lower_pos_limit
            self._params[1] = upper_pos_limit
        else:
            print("Parameters not accepted")


    def get_parameters(self) -> None:
        return [self._params[0], self._params[1], self._params[2]]

    def get_protocol_description(self) -> str: 
        description = ""
        match self.axis:                
                case 1:
                    description = "Moves laser A at constant speed along x-axis. \n" \
            "Moves between the two positions specified below"               
                case 2:
                    description = "Moves laser A at constant speed along y-axis. \n" \
            "Moves between the two positions specified below"
                case 3:
                    description = "Moves laser B at constant speed along x-axis. \n" \
            "Moves between the two positions specified below"
                case 4:
                    description = "Moves laser B at constant speed along y-axis. \n" \
            "Moves between the two positions specified below"
        return description

    def get_parameter_tooltips(self) -> List[str]:
        lower_lim_tooltip = ("Lower limit of the protocol, in nm. \n NOTE: The lower limit must"
                             "be smaller than the upper limit!")
        upper_lim_tooltip = ("Upper limit of the protocol, in nm. \n NOTE: The upper limit must be "
                             "larger than the lower limit!")
        stepsize_tip = " Movement speed in nm/s"
        return [lower_lim_tooltip, upper_lim_tooltip, stepsize_tip]
    
    def get_parameter_descriptions(self) -> List[str]:
        lower_lim_tooltip = "Lower Limit " 
        upper_lim_tooltip = "Upper limit"
        stepsize_tip = " Movement speed in nm/s"
        return [lower_lim_tooltip, upper_lim_tooltip, stepsize_tip]

    def get_parameter_limits(self) -> List[float]:
        return [[0,65535], [0,65535], [-100_000, 100_000]]

    def run_protocol(self) -> None:
        if not self._running:
            print("Protocol not started yet")
            return
        self.c_p['portenta_command_2'] = 2 if self.axis < 3 else 1

        self.c_p['protocol_data'][0] = self.axis
        
        lower_limit = int(self._params[0])
        upper_limit = int(self._params[1])
        step_size = int(self._params[2])
        
        # Function to split a 16-bit number into two 8-bit numbers
        split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF]
        
        self.c_p['protocol_data'][1:3] = split_16_bit(upper_limit) # upper_limit
        self.c_p['protocol_data'][3:5] = split_16_bit(lower_limit) # lower_limit
        self.c_p['protocol_data'][5:7] = split_16_bit(step_size)

    def stop_protocol(self) -> None:
        self.c_p['protocol_data'][0] = 0
        self._running = False


    def start_protocol(self) -> None:
        self.c_p['protocol_data'][0] = self.axis
        self._running = True

    def is_running(self) -> bool:
        return self._running


class Push2ForceProtocol(ExperimentLaserProtocol):
    """
    Push a trapped particle in one direction until a target force is reached.
    """

    _AXIS_NAMES = {1: "Left", 2: "Right", 3: "Up", 4: "Down"}
    _PROTOCOL_IDS = {1:5, 2:7, 3:9, 4:11}
    def __init__(self, c_p, axis=1):
            if axis not in self._AXIS_NAMES:
                raise ValueError(f"Axis {axis!r} not supported. Choose one of {list(self._AXIS_NAMES)}.")
            self.c_p = c_p
            self.axis = int(axis)
            self.axis_name = self._AXIS_NAMES[axis]
            self.axis_dir = 0 if self.axis <3 else 1
            self._running = False
            self._protocol_id = self._PROTOCOL_IDS[axis]

            # Defaults (floats)
            self._defaults: List[float] = [5.0, 0.0, 10.0]
            self._params: List[float] = self._defaults.copy()

    def get_protocol_name(self) -> str:
        return "Push to force " + self.axis_name

    def set_parameters(self, parameters) -> None:

        if parameters[0] is not None:
            lower_pos_limit = parameters[0]
        else:
            lower_pos_limit = self._params[0]
        if parameters[1] is not None:
            upper_pos_limit = parameters[1]
        else:
            upper_pos_limit = self._params[1]

        if parameters[2] is not None:
            self._params[2] = parameters[2]

        self._params[0] = lower_pos_limit
        self._params[1] = upper_pos_limit


    def get_parameters(self) -> None:
        return [self._params[0], self._params[1], self._params[2]]

    def get_protocol_description(self) -> str:
        description = f"Moves trapped particle {self.axis_name} until the maximum force is reached, it then stops"               
        return description

    def get_parameter_tooltips(self) -> List[str]:
        lower_lim_tooltip = ("The maximum force that the system will be allowed to push with")
        upper_lim_tooltip = ("")
        stepsize_tip = " Movement speed in nm/s"
        return [lower_lim_tooltip, upper_lim_tooltip, stepsize_tip]
    
    def get_parameter_descriptions(self) -> List[str]:
        lower_lim_tooltip = "Maximum pushing force" 
        upper_lim_tooltip = "-"
        stepsize_tip = " Movement speed in nm/s"
        return [lower_lim_tooltip, upper_lim_tooltip, stepsize_tip]

    def get_parameter_limits(self) -> List[float]:
        return [[0,120], [0,0], [0, 10_000]]

    def _calc_force_bits(self, value, axis=0):
        split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF]
        val = int(value /(self.c_p['PSD_to_force'][axis]*2))
        return split_16_bit(val + 32768)

    def run_protocol(self) -> None:
        if not self._running:
            print("Protocol not started yet")
            return
        self.c_p['portenta_command_2'] = 2

        self.c_p['protocol_data'][0] = self.self._protocol_id
        
        lower_limit = int(self._params[0])
        upper_limit = int(self._params[1])
        step_size = int(self._params[2])
        
        # Function to split a 16-bit number into two 8-bit numbers
        split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF]
        
        self.c_p['protocol_data'][1:3] = self._calc_force_bits(upper_limit, self.axis_dir) # upper_limit
        self.c_p['protocol_data'][3:5] = self._calc_force_bits(lower_limit) # lower_limit
        self.c_p['protocol_data'][5:7] = split_16_bit(step_size)

    def stop_protocol(self) -> None:
        self.c_p['protocol_data'][0] = 0
        self._running = False

    def start_protocol(self) -> None:
        self.c_p['protocol_data'][0] = self.axis
        self._running = True

    def is_running(self) -> bool:
        return self._running


class ConstantForceProtocol(ExperimentLaserProtocol):
    """
    Maintains a constant force along X and Y by moving one trap and auto-aligning the second trap.
    Set the target force values for both x and y (can be negative).
    """
    
    def __init__(self, c_p):
            self.c_p = c_p
            self._running = False
            self.laser = 'A'
            self.x_force_split = np.array([128,0], dtype=np.uint8)
            self.y_force_split = np.array([128,0], dtype=np.uint8)

            # Defaults (floats)
            self._params: List[float] = [0.0, 0.0, 0.0]

    def get_protocol_name(self) -> str:
        return "Constant force"

    def set_parameters(self, parameters) -> None:
        
        # Checks that neither parameter exceeds the accepted values and sets them
        # By splitting the value into two
        if np.abs(parameters[0] /(self.c_p['PSD_to_force'][0]*2)) < 32768:
            self._params[0] = parameters[0]
            self.x_force_split = self._calc_force_bits(self._params[0],0)

        if np.abs(parameters[1] /(self.c_p['PSD_to_force'][1]*2)) < 32768:
            self._params[1] = parameters[1]
            self.y_force_split = self._calc_force_bits(self._params[1], 1)

    def _calc_force_bits(self, value, axis=0):
        split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF]
        val = int(value /(self.c_p['PSD_to_force'][axis]*2))
        return split_16_bit(val + 32768)

    def get_parameters(self) -> None:
        return [self._params[0], self._params[1], self._params[2]]

    def get_protocol_description(self) -> str: 
        description = f"Constant force protocol. B autoalignign and following A to keep a constant force"
        return description

    def get_parameter_tooltips(self) -> List[str]:
        lower_lim_tooltip = ("The target force which the trap will try to maintain along X-axis")
        upper_lim_tooltip = ("The target force which the trap will try to maintain along Y-axis")
        stepsize_tip = "Not used in this protocol"
        return [lower_lim_tooltip, upper_lim_tooltip, stepsize_tip]
    
    def get_parameter_descriptions(self) -> List[str]:
        lower_lim_tooltip = "Constant force X (pN)" 
        upper_lim_tooltip = "Constant force Y (pN)"
        stepsize_tip = "-"
        return [lower_lim_tooltip, upper_lim_tooltip, stepsize_tip]

    def get_parameter_limits(self) -> List[float]:
        return [[-120,120], [-120,120], [-100_000, 100_000]]

    def run_protocol(self) -> None:
        if not self._running:
            print("Protocol not started yet")
            return

        self.c_p['portenta_command_2'] = 2
        self.c_p['protocol_data'][0] = 21 # B autoaligning        
        self.c_p['protocol_data'][1:3] = self.x_force_split 
        self.c_p['protocol_data'][3:5] = self.y_force_split 

    def stop_protocol(self) -> None:
        # Disengage protocol (turn it off)
        self.c_p['protocol_data'][0] = 0
        self._running = False

    def start_protocol(self) -> None:
        self._running = True

    def is_running(self) -> bool:
        return self._running


class ForceLimitProtocol(ExperimentLaserProtocol):
    """
    Sweep between lower/upper force limits along one axis at a fixed speed.
    """
    def __init__(self, c_p, data_channels, axis='Y'):
            self.c_p = c_p
            self.axis = axis
            self._running = False
            self.data_channels = data_channels

            self.current_force = 0
            self.previous_force = 0
            self.current_position = 0
            self.previous_position = 0
            self.force_move_direction = 1 # 1 for increasing, -1 for decreasing
            self._running = False
            self._defaults: List[float] = [0, 1.0, 10.0]
            self._params: List[float] = self._defaults.copy()

    def get_protocol_name(self) -> str:
        return "Force limit: "+self.axis

    def set_parameters(self, parameters) -> None:
        if parameters[0] is not None:
            lower_pos_limit = parameters[0]
        else:
            lower_pos_limit = self._params[0]
        if parameters[1] is not None:
            upper_pos_limit = parameters[1]
        else:
            upper_pos_limit = self._params[1]
        if parameters[2] is not None and parameters[2] > 0:
            self._params[2] = parameters[2]

        if lower_pos_limit < upper_pos_limit:
            self._params[0] = lower_pos_limit
            self._params[1] = upper_pos_limit
        else:
            print("Parameters not accepted")

    def get_parameters(self) -> None:
        return [self._params[0], self._params[1], self._params[2]]

    def get_parameter_limits(self) -> List[List[float]]:
        return [[-120, 120], [-120, 120], [0, 10_000]]

    def get_parameter_tooltips(self) -> List[str]:
        lower_lim_tooltip = ("Lower limit of the protocol, in pN. "
        "\n NOTE: The lower limit must be smaller than the upper limit!")
        upper_lim_tooltip = ("Upper limit of the protocol, in pN. \n "\
        "NOTE: The upper limit must be larger than the lower limit!")
        stepsize_tip = "Movement speed in nm/s"
        return [lower_lim_tooltip, upper_lim_tooltip, stepsize_tip]
    
    def get_parameter_descriptions(self) -> List[str]:
        lower_lim_tooltip = "Lower Limit (pN)" 
        upper_lim_tooltip = "Upper limit (pN)"
        stepsize_tip = " Movement speed in nm/s"
        return [lower_lim_tooltip, upper_lim_tooltip, stepsize_tip]

    def get_protocol_description(self) -> str: 
        description = f"Move at constant speed between two force levels along {self.axis} axis."
        return description

    def is_running(self) -> bool:
        return self._running

    def run_protocol(self):
        if not self._running:
            print("Protocol not started")
            return
        split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF]
        
        # Set the parameters for a good stepsize and maximum force
        self.c_p['protocol_data'][1:3] = [18,128]
        self.c_p['protocol_data'][3:5] = [128,128]
        self.c_p['protocol_data'][5:7] = split_16_bit(int(self._params[2]))
        if self.axis == 'Y':
             # Calculates the current force
            self.current_force = np.mean(self.data_channels['F_total_Y'].get_data(100))
            if self.c_p['portenta_command_2'] == 1: # A is being autoaligned
                
                self.current_position = self.data_channels['dac_by'].get_data(1)[0]
                
                if self.force_move_direction == 1:
                    self.c_p['protocol_data'][0] = 10
                else:
                    self.c_p['protocol_data'][0] = 12             

            elif self.c_p['portenta_command_2'] == 2: # B is autoaligned
                self.current_position = self.data_channels['dac_ay'].get_data(1)[0]
                if self.force_move_direction == 1:
                    self.c_p['protocol_data'][0] = 9
                else:
                    self.c_p['protocol_data'][0] = 11
        else:
            self.c_p['portenta_command_2'] == 2
            self.current_force = np.mean(self.data_channels['F_total_X'].get_data(100))
            if self.c_p['portenta_command_2'] == 1: # A is being autoaligned
                self.current_position = self.data_channels['dac_bx'].get_data(1)[0]
            elif self.c_p['portenta_command_2'] == 2: # B is autoaligned
                self.current_position = self.data_channels['dac_ax'].get_data(1)[0]

        if self.current_force > self._params[1]: # Force increasing
            # Switch direction
            self.force_move_direction = -1 # Moving up

        if self.current_force < self._params[0]:
            # Switch direction
            self.force_move_direction = 1

        if self.current_position > 62_000:
            self.force_move_direction = -1
        elif self.current_position < 2_000:
            self.force_move_direction = 1

        self.previous_force = self.current_force

    def start_protocol(self) -> None:
        self._running = True

    def stop_protocol(self) -> None:
        self._running = False
        self.c_p['protocol_data'][0] = 0


class PushAndWaitProtocol(ExperimentLaserProtocol):
    """
    Push toward a target force, wait for a specified duration, then pull back.
    """
    _DIRECTIONS = ["Left", "Right","Up","Down"]
    def __init__(self, c_p, data_channels, direction="Left"):
        self.c_p = c_p
        self._running = False
        self.data_channels = data_channels
        if direction not in self._DIRECTIONS:
            raise ValueError(f"Axis {direction!r} not supported. Choose one of {list(self._DIRECTIONS)}.")
        self.direction = direction
        self.current_force = 0
        self.previous_force = 0
        self.current_position = 0
        self.previous_position = 0
        self.force_move_direction = 1 # 1 for increasing, -1 for decreasing
        self._running = False
        self._defaults: List[float] = [0, 1.0, 10.0,10]
        self._params: List[float] = self._defaults.copy()


    def get_protocol_name(self) -> str:
        return f"Push {self.direction} & wait"

    def set_parameters(self, parameters) -> None:
        if parameters[0] is not None:
            lower_pos_limit = parameters[0]
        else:
            lower_pos_limit = self._params[0]
        if parameters[1] is not None:
            upper_pos_limit = parameters[1]
        else:
            upper_pos_limit = self._params[1]
        if parameters[2] is not None:
            self._params[2] = parameters[2]
        
        if parameters[3]>0:
            self._params[3] = parameters[3]

        self._params[0] = lower_pos_limit
        self._params[1] = upper_pos_limit

    def get_parameters(self) -> None:
        return [self._params[0], self._params[1], self._params[2], self._params[3]]

    def get_parameter_limits(self) -> List[List[float]]:
        return [[-120, 120], [-120, 120], [-100_000, 100_000], [0, 10_000]]

    def get_parameter_tooltips(self) -> List[str]:
        tooltip_0 = ("The target pushing force for the protocol, in pN. "
        "Force it will aim to reach when pushing particles together ")
        tooltip_1 = ("The target pulling force for the protocol, in pN. "
        "Force it will aim to reach when pulling particles apart, will stop at this force ")
        tooltip_2 = ("Speed of movement in nm/s")
        tooltip_3 = ("The duration during which the particles will wait while pushing (after pushing force was reached)")

        return [tooltip_0, tooltip_1, tooltip_2, tooltip_3]
    
    def get_parameter_descriptions(self) -> List[str]:
        description_0 = "Pushing force limit" 
        description_1 = "Pulling force limit"
        description_2 = "Movement speed (nm/s)"
        description_3 = "Waiting time (s)"
        return [description_0, description_1, description_2 ,description_3]

    def get_protocol_description(self) -> str: 
        description = f"Move at constant speed, pushing the particle {self.direction}."
        return description


    def start_protocol(self) -> None:
        print("Starting protocol")
        self._running = True

    def stop_protocol(self) -> None:
        print("Stopping protocol")
        self._running = False
        self.c_p['protocol_data'][0] = 0

    def is_running(self) -> bool:
        return self._running

    def run_protocol(self, axis='X', direction="right"):
        """
        This function controls the force limit protocol. It tells the portenta to move along the selected axis (x or y)
        between the force limits set by the user in the forcelimitspinboxes. The movement is performed with the 
        Force A_Y positive (or equivalent) protocol with the limit set very high (e.g 20_000). Once the force limit
        is exceeded (either positive or negative) the direction is switched.
        
        """
        
        
        if self.entanglement_step == 'waiting':
            if time() - self.entanglement_start_time > self.entanglement_wait_time:
                self.entanglement_step = 'pulling'
            else:
                return

        if axis == 'Y':
            self.current_force = np.mean(self.data_channels['F_total_Y'].get_data(100)) # Calculates the current force
            if self.c_p['portenta_command_2'] == 1: # A is being autoaligned
                
                self.current_position = self.data_channels['dac_by'].get_data(1)[0]
                
                if self.force_move_direction == 1:
                    # set the force limit appropriately first
                    self.c_p['protocol_data'][0] = 10
                else:
                    self.c_p['protocol_data'][0] = 12             

            elif self.c_p['portenta_command_2'] == 2: # B is autoaligned
                self.current_position = self.data_channels['dac_ay'].get_data(1)[0]
                if self.force_move_direction == 1:
                    self.c_p['protocol_data'][0] = 9
                else:
                    self.c_p['protocol_data'][0] = 11
        else:
            self.current_force = np.mean(self.data_channels['F_total_X'].get_data(100)) # Calculates the current force
            if direction == "right": # AX+ and BX -
                if self.c_p['portenta_command_2'] == 1: # A is being autoaligned
                    self.current_position = self.data_channels['dac_bx'].get_data(1)[0]

                    if self.force_move_direction == -1:
                        # Pulling
                        self.c_p['protocol_data'][0] = 6
                    else:
                        # Pushing
                        self.c_p['protocol_data'][0] = 8 # B-X - protocol

                elif self.c_p['portenta_command_2'] == 2: # B is autoaligned
                    self.current_position = self.data_channels['dac_ax'].get_data(1)[0]
                    if self.force_move_direction == 1:
                        # Pushing
                        self.c_p['protocol_data'][0] = 5 # A-X + protocol
                    else:
                        # Pulling
                        self.c_p['protocol_data'][0] = 7

        # TODO have one be the pushing force (in pN) and the other being the pulling force (in pN)
        self.push_force_limit = self.parameter_0_box.value()
        self.pull_force_limit = self.parameter_1_box.value()


        if self.entanglement_step == 'pushing':
            # Force negative when pushing, assuming direction=right
            if self.current_force < self.push_force_limit: 
                # Switch direction
                self.force_move_direction = -1 # Moving up                
                self.entanglement_step = 'waiting'
                self.entanglement_start_time = time()
                print("Switching to waiting")

        elif self.entanglement_step == 'pulling':
            if self.current_force >= self.pull_force_limit:
                # The protocol automatically stop and wait, just let it do so.
                sleep(0.1)
                return

        # Pulling is only terminated if the movement limit is reached, 
        # otherwise it will just continue pulling.
        if self.current_position > 62_000:
            self.force_move_direction = -1
            if self.entanglement_step == 'pushing':
                self.entanglement_step = 'pulling'
                print("Switching to pulling")
            elif self.entanglement_step == 'pulling':
                self.entanglement_step = 'pushing'
                print("Switching to pushing")

        elif self.current_position < 2_000:
            self.force_move_direction = 1
            if self.entanglement_step == 'pushing':
                self.entanglement_step = 'pulling'
                print("Switching to pulling")
            elif self.entanglement_step == 'pulling':
                self.entanglement_step = 'pushing'
                print("Switching to pushing")

        self.previous_force = self.current_force

class PullingProtocolWidget(QWidget):
    """
    Qt widget to select, parameterize, and run laser protocols.

    Parameters
    ----------
    c_p : dict
        Shared control/state dictionary consumed by protocols.
    data_channels : Mapping[str, Any]
        Live data feeds used by some protocols (force/position).

    UI
    --
    - Protocol dropdown populated from concrete implementations.
    - Per-parameter editors with labels, tooltips, limits, and live values.
    - Toggle button to start/stop the selected protocol.
    - Timer (~40 ms) that calls `run_protocol()` while active.

    Notes
    -----
    The widget is protocol-agnostic: it queries names, descriptions,
    parameter metadata, and delegates execution to the active protocol.
    """

    def __init__(self, c_p, data_channels):

        super().__init__()
        self.c_p = c_p
        self.data_channels = data_channels
        self.setWindowTitle("Pulling Protocol")
        self.protocol_index = 0
        self.protocols = [
            ConstantSpeedProtocol(c_p, axis=1), # A-X
            ConstantSpeedProtocol(c_p, axis=2), # A-Y
            ForceLimitProtocol(c_p, data_channels, axis='X'),
            ForceLimitProtocol(c_p, data_channels, axis='Y'),
            ConstantForceProtocol(c_p),
            Push2ForceProtocol(c_p, 1), # Left
            Push2ForceProtocol(c_p, 2), # Right
            Push2ForceProtocol(c_p, 3), # Up
            Push2ForceProtocol(c_p, 4), # Down
            PushAndWaitProtocol(c_p, data_channels),
        ]
        self.initiate_interface()
        self.timer = QTimer()
        self.timer.setInterval(40) # sets the delay of the timer and thereby how often it should update.
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def initiate_interface(self):
        layout = QVBoxLayout()

        self.protocol_selection_box = QComboBox()
        for protocol in self.protocols:
            self.protocol_selection_box.addItem(protocol.get_protocol_name())


        self.protocol_selection_box.setCurrentIndex(self.protocol_index)
        self.protocol_selection_box.currentIndexChanged.connect(self.select_protocol)
        layout.addWidget(QLabel("Select protocol:"))
        layout.addWidget(self.protocol_selection_box)

        self.protocol_description_label = QLabel(
            self.protocols[self.protocol_index].get_protocol_description())
        layout.addWidget(self.protocol_description_label)

        grid = QGridLayout()
        self.l1 = QLabel("Paremeter")
        self.l2 = QLabel("New value")
        self.l3 = QLabel("Current value")
        grid.addWidget(self.l1, 0, 0)
        grid.addWidget(self.l2, 0, 1)
        grid.addWidget(self.l3, 0, 2)

        self.parameter_0_box = QDoubleSpinBox() # Change to a double spinbox if we want to allow for decimal values.
        self.parameter_0_box.setDecimals(1)
        self.parameter_0_box.setRange(0, 65535)
        self.parameter_0_box.setValue(0)
        self.parameter_0_box.setToolTip("Lower limit of the protocol, in nm. \n NOTE: The lower \
                                        limit must be smaller than the upper limit!\n also acts to \
                                        set the threshold force value in the approach to surface \
                                        experiments.")
        self.parameter_0_label = QLabel("")
        self.parameter_0_value = QLabel("")
        grid.addWidget(self.parameter_0_label, 1, 0)
        grid.addWidget(self.parameter_0_box, 1, 1)
        grid.addWidget(self.parameter_0_value, 1, 2)

        self.parameter_1_box = QDoubleSpinBox()
        self.parameter_1_box.setRange(0, 65535)
        self.parameter_1_box.setDecimals(1)
        self.parameter_1_box.setToolTip("")    
        self.parameter_1_label = QLabel("")
        self.parameter_1_value = QLabel("")
        grid.addWidget(self.parameter_1_label, 2,0)
        grid.addWidget(self.parameter_1_box, 2,1)
        grid.addWidget(self.parameter_1_value, 2,2)

        self.parameter_2_box = QDoubleSpinBox()
        self.parameter_2_box.setRange(0, 65332)
        self.parameter_2_box.setDecimals(1) 
        self.parameter_2_box.setValue(int(self.c_p['protocol_data'][5])*256 + int(self.c_p['protocol_data'][6]))
        self.parameter_2_label = QLabel("")
        self.parameter_2_value = QLabel("")
        grid.addWidget(self.parameter_2_label, 3,0)
        grid.addWidget(self.parameter_2_box, 3,1)
        grid.addWidget(self.parameter_2_value, 3,2)

        self.parameter_3_box = QDoubleSpinBox()
        self.parameter_3_box.setDecimals(1)
        self.parameter_3_label = QLabel("")
        self.parameter_3_value = QLabel(str(0))
        grid.addWidget(self.parameter_3_label, 4,0)
        grid.addWidget(self.parameter_3_box, 4,1)
        grid.addWidget(self.parameter_3_value, 4,2)

        self.set_parameters_button = QPushButton("Set parameters")
        self.set_parameters_button.clicked.connect(self.set_parameters)
        self.set_parameters_button.setCheckable(False)
        self.set_parameters_button.setToolTip("Sets the parameters to the current values in the" \
        "spinboxes.")
        self.boxes = [self.parameter_0_box, self.parameter_1_box, self.parameter_2_box, self.parameter_3_box]
        self.parameter_labels = [self.parameter_0_label, self.parameter_1_label,
                                 self.parameter_2_label, self.parameter_3_label]
        self.parameter_values = [self.parameter_0_value, self.parameter_1_value, 
                                 self.parameter_2_value, self.parameter_3_value]


        # Add toggle protocol button
        self.toggle_protocol_button = QPushButton("Toggle protocol")
        self.toggle_protocol_button.clicked.connect(self.toggle_protocol)
        self.toggle_protocol_button.setCheckable(True)
        self.toggle_protocol_button.setChecked(self.c_p['protocol_data'][0])
        self.toggle_protocol_button.setToolTip("Toggles the selected protocol on/off. \
                                             \n NOTE: You cannot control either piezo manually when\
                                              a protocol is running!")

        layout.addLayout(grid)
        layout.addWidget(self.set_parameters_button)
        layout.addWidget(self.toggle_protocol_button)

        self.setLayout(layout)
        self.configure_widget()

    def toggle_entanglement_protocol(self):
        self.entanglement_protocol_running = self.advancedProtocoltoggle.isChecked()
        if self.entanglement_protocol_running:
            self.entanglement_step = 'pushing'
            self.force_limit_protocol_running = False

        if not self.entanglement_protocol_running and self.toggle_protocol_button.isChecked():
            print("De-toggling protocol mover")
            self.toggle_protocol_button.setChecked(False)
            self.c_p['protocol_data'][0] = 0

    def set_wait_time(self, value):
        if value > 0:
            self.entanglement_wait_time = float(value)

    def set_widget_values(self):
        pass

    def configure_widget(self):
        protocol = self.protocols[self.protocol_index]
        self.protocol_description_label.setText(protocol.get_protocol_description())
        
        parameters = protocol.get_parameters()
        limits = protocol.get_parameter_limits()
        descriptions = protocol.get_parameter_descriptions()
        tooltips = protocol.get_parameter_tooltips()

        for idx, limit in enumerate(limits):
            print(limit)
            self.boxes[idx].setRange(limit[0], limit[1])
            self.boxes[idx].setToolTip(tooltips[idx])
            self.parameter_labels[idx].setText(descriptions[idx])
            self.parameter_values[idx].setText(str(parameters[idx]))

    def set_parameters(self):
        parameters = [self.parameter_0_box.value(), self.parameter_1_box.value(),
                      self.parameter_2_box.value(), self.parameter_3_box.value()]
        self.protocols[self.protocol_index].set_parameters(parameters)
        self.configure_widget()

    def refresh(self):
        if self.protocols[self.protocol_index].is_running():
            self.protocols[self.protocol_index].run_protocol()

    def select_protocol(self, index):
        """
        Selects the protocol to be run and configures the interface to accomodate the different
        protocols (changing units and descriptions).
        """
        if not self.protocol_index == index:
            self.protocols[self.protocol_index].stop_protocol()
            self.toggle_protocol_button.setChecked(False)

        self.protocol_index = index
        self.configure_widget()
        
    def toggle_protocol(self):
        protocol = self.protocols[self.protocol_index]
        if self.toggle_protocol_button.isChecked():
            protocol.start_protocol()
        else:
            protocol.stop_protocol()
