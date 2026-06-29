"""
Microfluidics Hardware Abstraction & UI

This module defines protocol interfaces (hardware abstraction layer) for a
microfluidics system—pressure controller, valve controller, and pipette pump—
along with lightweight test (mock) implementations, a real-time monitor thread,
and Qt widgets for configuration and live control.

Contents
--------
Protocols
    MicrofluidicsController
        Interface for multi-channel pressure controllers
        (connect/disconnect, set/get pressure, channel count).
    ValveController
        Interface for valve arrays (connect, connection status, toggle valve,
        read states).
    PipettePump
        Interface for pipette pump PSUs (connect/disconnect, power control,
        suction on/off, status).

Test (mock) implementations
    TestMicrofluidicsController
        3-channel in-memory pressure controller; 1-based channel indexing,
        pressure range 0–2000 mbar.
    TestValveController
        In-memory valve state map with connect/toggle/query methods.
    TestPipettePump
        In-memory PSU with % power setting and suction state.

Realtime monitor
    MicrofluidicsMonitorThread
        QThread that synchronizes target ↔ measured pressures, valve states,
        and pipette pump power/suction with a shared config dict `c_p`.
        Emits `progress(list)` after each 500 ms cycle.

Qt widgets
    ConfigurePumpWidget
        Form for assigning fluidics channels/valves and per-channel max
        pressures for autonomous procedures (writes to `c_p`).
    MicrofluidicsControllerWidget
        Live control panel: per-channel pressure set/read, valve toggles, and
        pipette pump PSU controls. Spawns `MicrofluidicsMonitorThread` and
        auto-refreshes every 500 ms.

Shared state (c_p)
------------------
The monitoring thread and widgets cooperate via a shared dictionary `c_p`.
Expected keys (minimum; extend as needed):
    - 'program_running' : bool
    - 'current_pressures' : Sequence[float]      # per channel, mbar
    - 'target_pressures'  : Sequence[float]      # per channel, mbar
    - 'valves_used'       : Sequence[int]
    - 'valves_open'       : Mapping[int, bool]
    - 'pipette_pump_on'           : bool
    - 'pipette_pump_target_power' : float        # PSU setpoint (e.g., V or %)
    - 'pipette_pump_current_power': float
    - For ConfigurePumpWidget:
        * 'capillary_1_fluidics_channel' : [channel_idx, max_mbar, valve_idx]
        * 'capillary_2_fluidics_channel' : [channel_idx, max_mbar, valve_idx]
        * 'central_fluidics_channel'     : [channel_idx, max_mbar, valve_idx]

Notes
-----
- Pressure channels use 1-based indexing on the controller (0 and 1 may map to
  the same channel on some devices).
- The monitor loop is fixed at 500 ms; adjust in `MicrofluidicsMonitorThread.run`.
- Test classes are drop-in for UI and logic development without hardware.

Quick start
-----------
>>> c_p = {
...     'program_running': True,
...     'current_pressures': [0.0, 0.0, 0.0],
...     'target_pressures':  [0.0, 0.0, 0.0],
...     'valves_used': [0, 1],
...     'valves_open': {0: False, 1: True},
...     'pipette_pump_on': False,
...     'pipette_pump_target_power': 0.0,
...     'pipette_pump_current_power': 0.0,
...     'capillary_1_fluidics_channel': [0, 800.0, 0],
...     'capillary_2_fluidics_channel': [1, 800.0, 1],
...     'central_fluidics_channel':     [2, 800.0, 2],
... }
>>> pump   = TestMicrofluidicsController()
>>> valves = TestValveController()
>>> psu    = TestPipettePump()
>>> pump.connect(); valves.connect(); psu.connect()
>>> monitor_widget = MicrofluidicsControllerWidget(c_p, pump, valves, psu)  # Qt context required
"""

from typing import Protocol, runtime_checkable

from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import QThread, pyqtSignal, QTimer

@runtime_checkable
class MicrofluidicsController(Protocol):
    """
    Protocol defining the interface for microfluidics pressure controllers.

    Any implementation must provide connection handling, pressure control
    per channel, and a way to query the number of available channels.

    Methods
    -------
    connect() -> None
        Establish connection to the controller.
    disconnect() -> None
        Close connection to the controller.
    set_pressure(channel: str, value_kpa: float) -> None
        Set the pressure of a given channel in kilopascals.
    get_pressure(channel: str) -> float
        Return the current measured pressure for a given channel.
    get_number_channels() -> int
        Return the number of pressure channels supported by the controller.
    """
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def set_pressure(self, channel: str, value_kpa: float) -> None: ...
    def get_pressure(self, channel: str) -> float: ...
    def get_number_channels(self) -> int: ...


@runtime_checkable
class ValveController(Protocol):
    """
    Protocol defining the interface for valve array controllers.

    Implementations must handle connection state, switching individual valves,
    and querying the state of all valves.

    Methods
    -------
    connect(address: str | None = None) -> None
        Establish connection to the valve controller, optionally at a given address.
    is_connected() -> bool
        Return True if the controller is connected.
    toggle_valve(valve_id: str, state: int) -> None
        Set a valve to a given state (typically 0 = closed, 1 = open).
    get_valve_states() -> list[int]
        Return the current states of all valves as a list of integers.
    """
    def connect(self, address: str | None = None) -> None: ...
    def is_connected(self) -> bool: ...
    def toggle_valve(self, valve_id: str, state: int) -> None: ...
    def get_valve_states(self) -> list[int]: ...


@runtime_checkable
class PipettePump(Protocol):
    """
    Protocol defining the interface for pipette pump power supplies.

    Provides methods for connection management, power control, and
    activation/deactivation of suction.

    Methods
    -------
    connect(address: str | None = None) -> None
        Establish connection to the pipette pump, optionally at a given address.
    disconnect() -> None
        Close connection to the pump.
    is_connected() -> bool
        Return True if the pump is connected.
    set_power(power: float) -> None
        Set the pump PSU output voltage.
    get_power() -> float
        Return the current PSU output voltage.
    activate_suction() -> float
        Enable suction mode and return the resulting power level.
    deactivate_suction() -> float
        Disable suction mode and return the resulting power level.
    suction_active() -> bool
        Return True if suction is currently active.
    """
    def connect(self, address: str | None = None) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def set_power(self, power: float) -> None: ...
    def get_power(self) -> float: ...
    def activate_suction(self) -> float: ...
    def deactivate_suction(self) -> float: ...
    def suction_active(self) -> bool: ...

class TestMicrofluidicsController(MicrofluidicsController):

    """
    Minimal mock implementation of a microfluidics pump controller.

    This class simulates a multi-channel microfluidics pressure controller
    without requiring hardware. It exposes the same interface as a real
    controller, enabling testing and development of UI or logic that depends
    on a pump backend.

    Parameters
    ----------
    None

    Attributes
    ----------
    connected : bool
        Whether the controller is currently marked as connected.
    num_channels : int
        Number of simulated pressure channels. Defaults to 3.
    pressures : list[float]
        Current simulated pressures for each channel, in mbar.
    verbose : bool
        If True, pressure changes are printed to stdout.

    Methods
    -------
    connect(address=None)
        Mark the controller as connected. Prints a message to stdout.
    disconnect()
        Mark the controller as disconnected. Prints a message to stdout.
    set_pressure(channel, pressure)
        Set the pressure (mbar) of a specific channel. Channel indices
        are 1-based. Valid range is [0, 2000] mbar.
    get_pressure(channel)
        Get the pressure (mbar) of a specific channel. Channel indices
        are 1-based.
    get_number_channels()
        Return the number of available channels.

    Example
    -------
    >>> ctrl = TestMicrofluidicsController()
    >>> ctrl.connect()
    Connected to test pump controller
    >>> ctrl.set_pressure(1, 500)
    >>> ctrl.get_pressure(1)
    500
    >>> ctrl.disconnect()
    Disconnected from test pump controller
    """
    def __init__(self):
        self.connected = False
        self.num_channels = 3
        self.pressures = [0]*self.num_channels
        self.verbose = False

    def connect(self, adress=None):
        self.connected = True
        print("Connected to test pump controller")

    def disconnect(self):
        self.connected = False
        print("Disconnected from test pump controller")

    def set_pressure(self, channel, pressure):
        if not self.connected:
            raise Exception("Not connected to pump controller")
        if channel < 1 or channel > self.num_channels:
            raise Exception("Channel out of range")
        if pressure < 0 or pressure > 2000:
            raise Exception("Pressure out of range")
        self.pressures[channel-1] = pressure

        if self.verbose:
            print(f"Set pressure of channel {channel} to {pressure} mbar")

    def get_pressure(self, channel):
        if not self.connected:
            raise Exception("Not connected to pump controller")
        if channel < 1 or channel > self.num_channels:
            raise Exception("Channel out of range")
        return self.pressures[channel-1]

    def get_number_channels(self):
        return self.num_channels


class TestValveController(ValveController):
    """
    Minimal mock implementation of a valve controller for testing and development.

    This class simulates a valve controller without requiring hardware. It
    maintains internal valve states in a dictionary and provides the same
    interface as a real controller, making it useful for unit tests or
    development on systems without connected hardware.

    Parameters
    ----------
    None

    Attributes
    ----------
    connected : bool
        Whether the controller is currently marked as connected.
    valve_states : dict[int, bool]
        Mapping of valve indices to their states (True = open, False = closed).
    verbose : bool
        If True, state changes are printed to stdout.

    Methods
    -------
    connect(address=None)
        Mark the controller as connected. Prints a message to stdout.
    is_connected()
        Return True if the controller is marked as connected.
    toggle_valve(valve_id, state)
        Set the state of a valve. Requires the controller to be connected.
        If verbose, prints the updated state.
    get_valve_states()
        Return a dictionary of the current valve states. Requires the
        controller to be connected.

    Example
    -------
    >>> vc = TestValveController()
    >>> vc.connect()
    Connected to test valve controller
    >>> vc.toggle_valve(0, True)
    >>> vc.get_valve_states()
    {0: True}
    """


    def __init__(self):
        self.connected = False
        self.valve_states = {} # TODO extend this a little
        self.verbose = False

    def connect(self, address=None):
        self.connected = True
        print("Connected to test valve controller")

    def is_connected(self):
        return self.connected

    def toggle_valve(self, valve_id, state):
        if not self.connected:
            raise Exception("Not connected to valve controller")
        self.valve_states[valve_id] = state
        if self.verbose:
            print(f"Set valve {valve_id} to state {state}")

    def get_valve_states(self):
        if not self.connected:
            raise Exception("Not connected to valve controller")
        return self.valve_states
    

class TestPipettePump(PipettePump):

    def __init__(self):
        self.connected = False
        self.power = 0
        self.suction_active_state = False
        self.verbose = False
    
    def connect(self, address=None):
        self.connected = True
        print("Connected to test pipette pump")

    def disconnect(self):
        self.connected = False
        print("Disconnected from test pipette pump")

    def is_connected(self):
        return self.connected

    def set_power(self, power:float):
        if not self.connected:
            raise Exception("Not connected to pipette pump")
        if power < 0 or power > 100:
            raise Exception("Power out of range")
        self.power = power
        if self.verbose:
            print(f"Set power to {power}%")

    def get_power(self) -> float:
        if not self.connected:
            raise Exception("Not connected to pipette pump")
        return self.power
    
    def activate_suction(self):

        if not self.connected:
            raise Exception("Not connected to pipette pump")
        self.suction_active_state = True
        print(f"Activated suction")
    
    def suction_active(self):
        return self.suction_active_state

    def deactivate_suction(self):
        if not self.connected:
            raise Exception("Not connected to pipette pump")
        self.suction_active_state = False
        if self.verbose:
            print(f"Deactivated suction")

class MicrofluidicsMonitorThread(QThread):
    """
    Background worker thread for continuous synchronization of the
    microfluidics system state.

    The monitor loop periodically sets channel pressures to their target
    values, reads back measured pressures, updates valve states, and manages
    the pipette pump PSU if present. Updates are pushed to the UI/main thread
    via Qt signals.

    Signals
    -------
    finished : pyqtSignal()
        Emitted once the monitoring loop exits cleanly.
    progress : pyqtSignal(list)
        Emitted after each update cycle with the list of current pressures.

    Parameters
    ----------
    microfluidicsController : object
        Controller for the pressure channels. Must implement
        ``get_number_channels()``, ``set_pressure(channel, value)``,
        and ``get_pressure(channel)``.
    valve_controller : object
        Controller for the valve array. Must implement
        ``is_connected()`` and ``toggle_valve(index, state)``.
    c_p : dict
        Shared configuration/state dictionary. Expected keys (minimum):
            - 'program_running' : bool
                Controls whether the monitor loop is active.
            - 'target_pressures' : Sequence[float]
                Target pressure per channel (mbar).
            - 'current_pressures' : Sequence[float]
                Current measured pressures, updated in-place (mbar).
            - 'valves_used' : Sequence[int]
                Indices of valves to monitor/control.
            - 'valves_open' : Mapping[int, bool]
                Valve index → open (True) / closed (False).
            - 'pipette_pump_target_power' : float
                Target output voltage of the pipette pump PSU (V).
            - 'pipette_pump_current_power' : float
                Live measured PSU output voltage (V).
            - 'pipette_pump_on' : bool
                Power state of the pipette pump PSU.
    pipette_pump : object, optional
        Pipette pump PSU interface. Must implement
        ``is_connected()``, ``set_power(value)``, ``get_power()``,
        ``activate_suction()``, and ``deactivate_suction()``.

    Notes
    -----
    - The monitoring loop runs at a fixed interval of 500 ms
      (see ``QThread.msleep(500)``).
    - Pressures are applied with 1-based channel indexing
      (``channel+1``) due to controller-specific convention.
    - If a pressure set/get fails, the exception is caught and the
      corresponding channel is set to 0 mbar.
    - All updates to shared state are performed in-place on ``c_p``.
    - Emits :pyattr:`progress` after each cycle with the current pressures,
      which can be connected to UI elements for live updates.

    Methods
    -------
    set_pressures()
        Apply target pressures from ``c_p['target_pressures']`` to all channels.
    get_pressures()
        Read current pressures from the controller into
        ``c_p['current_pressures']``.
    check_pipette_pump()
        Update pipette pump PSU state (power, on/off) based on ``c_p``.
    run()
        Main loop: update pressures, valves, and pipette pump every 500 ms.
    """

    # Define signals to communicate with the main thread
    finished = pyqtSignal()
    progress = pyqtSignal(list)

    def __init__(self, microfluidicsController, valve_controller, c_p, pipette_pump=None):
        super().__init__()
        self.microfluidicsController = microfluidicsController
        self.c_p = c_p
        self.pipette_pump = pipette_pump
        self.valve_controller = valve_controller

    def set_pressures(self):
        """
        Sets the pressures of all the channels to the target values in target_pressures.
        """

        for channel in range(self.microfluidicsController.get_number_channels()):
            # Indexing starts at 1 in the controller. Also 0 and 1 map to the same channel.
            try:
                self.microfluidicsController.set_pressure(
                    channel+1, self.c_p['target_pressures'][channel])
            except RuntimeError as E:
                print("Could not set pressure")

    def get_pressures(self):
        for channel in range(self.microfluidicsController.get_number_channels()):
            try:
                self.c_p['current_pressures'][channel] = self.microfluidicsController.get_pressure(channel+1)
            except RuntimeError as E:
                self.c_p['current_pressures'][channel] = 0
                print(E)
    
    def check_pipette_pump(self):
        self.pipette_pump.set_power(self.c_p['pipette_pump_target_power'])
        if self.c_p['pipette_pump_on']:
            self.pipette_pump.activate_suction()
        else:
            self.pipette_pump.deactivate_suction()
        self.c_p['pipette_pump_current_power'] = self.pipette_pump.get_power()

    def run(self):
        # Place your background task here
        while self.c_p['program_running']:
            self.set_pressures()
            self.get_pressures()

            # Set the valves to the correct state
            if self.valve_controller.is_connected():
                for index in self.c_p['valves_used']:                    
                    self.valve_controller.toggle_valve(index, self.c_p['valves_open'][index])
            
            if self.pipette_pump is not None and self.pipette_pump.is_connected():
                self.check_pipette_pump()
            self.progress.emit(self.c_p['current_pressures'])
            QThread.msleep(500) # Sleep for specified number of milliseconds
        self.finished.emit()


class ConfigurePumpWidget(QWidget): # TODO move this to the autocontroller 
    """
    Widget used to change the settings of the fluidics channels. Specifically to
    configure for the autonomous system which channels contain which particles and what
    is a reasonable pressure to use when flowing particles trough the capillaries.
    """

    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(225, 225, 250))
        self.setPalette(pal)

        self.initUI()

    def initUI(self):
        """
        Initiates the various user interface components
        """
        self.layout = QVBoxLayout()
        self.setWindowTitle("Configure Pump")
        self.capillary_1_label = QLabel("Capillary 1")
        self.layout.addWidget(self.capillary_1_label)
        self.capillary_1_channel_spinbox = QSpinBox()
        self.capillary_1_channel_spinbox.setRange(1, 3)
        self.capillary_1_channel_spinbox.setValue(self.c_p['capillary_1_fluidics_channel'][0])
        self.capillary_1_channel_spinbox.valueChanged.connect(
            lambda value: self.set_capillary_1_channel(value))
        self.layout.addWidget(self.capillary_1_channel_spinbox)

        self.capillary_1_max_pressure_input = QDoubleSpinBox()
        self.capillary_1_max_pressure_input.setRange(0, 2000)
        self.capillary_1_max_pressure_input.setValue(self.c_p['capillary_1_fluidics_channel'][1])
        self.capillary_1_max_pressure_input.valueChanged.connect(
            lambda value: self.set_capillary_1_flow_pressure(value))
        self.layout.addWidget(self.capillary_1_max_pressure_input)

        self.capillary_1_valve_label = QLabel("Capillary 1 Valve")
        self.layout.addWidget(self.capillary_1_valve_label)
        self.capillary_1_valve_input = QSpinBox()
        self.capillary_1_valve_input.setRange(0, 7)
        self.capillary_1_valve_input.setValue(self.c_p['capillary_1_fluidics_channel'][2])
        self.capillary_1_valve_input.valueChanged.connect(
            lambda value: self.set_capillary_1_valve(value))
        self.layout.addWidget(self.capillary_1_valve_input)

        self.capillary_2_label = QLabel("Capillary 2")
        self.layout.addWidget(self.capillary_2_label)
        self.capillary_2_channel_spinbox = QSpinBox()
        self.capillary_2_channel_spinbox.setRange(1, 3)
        self.capillary_2_channel_spinbox.setValue(self.c_p['capillary_2_fluidics_channel'][0])
        self.capillary_2_channel_spinbox.valueChanged.connect(
            lambda value: self.set_capillary_2_channel(value))
        self.layout.addWidget(self.capillary_2_channel_spinbox)

        self.capillary_2_max_pressure_input = QDoubleSpinBox()
        self.capillary_2_max_pressure_input.setRange(0, 2000)
        self.capillary_2_max_pressure_input.setValue(self.c_p['capillary_2_fluidics_channel'][1])
        self.capillary_2_max_pressure_input.valueChanged.connect(
            lambda value: self.set_capillary_2_flow_pressure(value))
        self.layout.addWidget(self.capillary_2_max_pressure_input)

        self.capillary_2_valve_label = QLabel("Capillary 2 Valve")
        self.layout.addWidget(self.capillary_2_valve_label)
        self.capillary_2_valve_input = QSpinBox()
        self.capillary_2_valve_input.setRange(0, 7)
        self.capillary_2_valve_input.setValue(self.c_p['capillary_2_fluidics_channel'][2])
        self.capillary_2_valve_input.valueChanged.connect(
            lambda value: self.set_capillary_2_valve(value))
        self.layout.addWidget(self.capillary_2_valve_input)

        self.main_label = QLabel("Central Channel")
        self.layout.addWidget(self.main_label)
        self.main_channel_spinbox = QSpinBox()
        self.main_channel_spinbox.setRange(1, 3)
        self.main_channel_spinbox.setValue(self.c_p['central_fluidics_channel'][0])
        self.main_channel_spinbox.valueChanged.connect(
            lambda value: self.set_main_channel(value))
        self.layout.addWidget(self.main_channel_spinbox)

        self.main_max_pressure_input = QDoubleSpinBox()
        self.main_max_pressure_input.setRange(0, 2000)
        self.main_max_pressure_input.setValue(self.c_p['central_fluidics_channel'][1])
        self.main_max_pressure_input.valueChanged.connect(
            lambda value: self.set_main_flow_pressure(value))
        self.layout.addWidget(self.main_max_pressure_input)

        self.central_valve_label = QLabel("Central Valve")
        self.layout.addWidget(self.central_valve_label)
        self.main_valve_input = QSpinBox()
        self.main_valve_input.setRange(0, 7)
        self.main_valve_input.setValue(self.c_p['central_fluidics_channel'][2])
        self.main_valve_input.valueChanged.connect(
            lambda value: self.set_main_valve(value))
        self.layout.addWidget(self.main_valve_input)
                
        self.setLayout(self.layout)
        self.show()

    def set_capillary_1_channel(self, channel):
        self.c_p['capillary_1_fluidics_channel'][0] = int(channel-1)

    def set_capillary_2_channel(self, channel):
        self.c_p['capillary_2_fluidics_channel'][0] = int(channel-1)

    def set_main_channel(self, channel):
        self.c_p['central_fluidics_channel'][0] = int(channel-1)

    def set_capillary_1_flow_pressure(self, pressure):
        self.c_p['capillary_1_fluidics_channel'][1] = float(pressure)

    def set_capillary_2_flow_pressure(self, pressure):
        self.c_p['capillary_2_fluidics_channel'][1] = float(pressure)

    def set_main_flow_pressure(self, pressure):
        self.c_p['central_fluidics_channel'][1] = float(pressure)

    def set_capillary_1_valve(self, valve):
        self.c_p['capillary_1_fluidics_channel'][2] = int(valve)

    def set_capillary_2_valve(self, valve):
        self.c_p['capillary_2_fluidics_channel'][2] = int(valve)

    def set_main_valve(self, valve):
        self.c_p['central_fluidics_channel'][2] = int(valve)
    

class MicrofluidicsControllerWidget(QWidget):
    """
    Qt widget for interactive control and monitoring of a microfluidics setup.

    The widget auto-generates channel controls from the connected controller,
    provides live readout of measured pressures, and exposes toggles for
    external peripherals (valves and a pipette pump PSU). A background monitor
    thread keeps the UI in sync with hardware state, while a timer periodically
    refreshes the view.

    Parameters
    ----------
    c_p : dict
        Shared configuration/state dictionary used for both UI and back-end control.
        Expected keys (minimum):
            - 'current_pressures' : Sequence[float]
                Latest measured pressure per channel (mbar).
            - 'target_pressures' : Sequence[float]
                Target pressure per channel (mbar) set by the UI.
            - 'valves_used' : Sequence[int]
                Indices of valves that should be displayed/controlled.
            - 'valves_open' : Mapping[int, bool]
                Valve index → open (True) / closed (False).
            - 'pipette_pump_current_power' : float
                Currently measured output voltage of the pipette pump PSU (V).
            - 'pipette_pump_target_power' : float
                Target output voltage for the pipette pump PSU (V).
            - 'pipette_pump_on' : bool
                Power state of the pipette pump PSU.
    microfluidicsController : object, optional
        Controller providing access to the pressure channels. Must implement
        ``get_number_channels() -> int`` and be compatible with
        ``MicrofluidicsMonitorThread``.
    valve_controller : object, optional
        Low-level controller for valves, used by the monitor thread to read/update state.
    pipette_pump : object, optional
        Pipette pump PSU interface. Expected to support
        ``disconnect_from_psu()`` and be compatible with the monitor thread.

    UI Elements
    -----------
    • Per-channel controls:
        - QDoubleSpinBox (0–2000 mbar, step 0.1) to set target pressure.
        - QLabel displaying live measured pressure: ``"Pressure {x} mbar"``.
    • Pipette pump PSU:
        - QDoubleSpinBox (0–12 V, step 0.1) for target voltage.
        - Checkable QPushButton to toggle PSU on/off.
    • Valves:
        - One checkable QPushButton per valve in ``valves_used``.
          Red = closed, Green = open.

    Notes
    -----
    - Starts a ``MicrofluidicsMonitorThread`` that emits ``progress`` updates to
      synchronize ``c_p['current_pressures']``, ``c_p['target_pressures']``,
      valve states, and pump power.
    - A QTimer (500 ms) calls :meth:`refresh` to mirror external state changes
      in the UI without user interaction.
    - Background color is set to a light lavender for quick visual identification.
    - The widget calls ``show()`` during initialization.

    Methods
    -------
    create_channel_controls()
        Build per-channel labels, spin boxes, and readouts from the controller.
    update_pressures(values)
        Update pressure labels and sync spin boxes with ``c_p['target_pressures']``.
    set_pressure(channel, pressure)
        Set target pressure for a channel in ``c_p``.
    toggle_valve(valve_index)
        Toggle the state of a valve in ``c_p['valves_open']``.
    set_pipette_pump_power(value)
        Set target PSU voltage in ``c_p['pipette_pump_target_power']``.
    toggle_pipette_pump()
        Update ``c_p['pipette_pump_on']`` from the toggle button state.
    refresh()
        Pull external state from ``c_p`` to the UI (pump power, on/off, valves).
    closeEvent(event)
        Accept close and disconnect ``pipette_pump`` from PSU.

    Example
    -------
    >>> c_p = {
    ...     'current_pressures': [0.0, 0.0, 0.0, 0.0],
    ...     'target_pressures':  [0.0, 0.0, 0.0, 0.0],
    ...     'valves_used': [0, 2],
    ...     'valves_open': {0: False, 2: True},
    ...     'pipette_pump_current_power': 0.0,
    ...     'pipette_pump_target_power':  0.0,
    ...     'pipette_pump_on': False,
    ... }
    >>> widget = MicrofluidicsControllerWidget(
    ...     c_p,
    ...     microfluidicsController=my_controller,
    ...     valve_controller=my_valves,
    ...     pipette_pump=my_psu
    ... )
    >>> widget.show()
    """


    def __init__(self, c_p, microfluidicsController=None, valve_controller=None, pipette_pump=None):
        super().__init__()
        self.c_p = c_p
        self.microfluidicsController = microfluidicsController
        self.valve_controller = valve_controller        

        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(225, 225, 250))
        self.setPalette(pal)

        self.initUI()
        self.pipette_pump = pipette_pump 
        self.pumpMonitorThread = MicrofluidicsMonitorThread(
            self.microfluidicsController, self.valve_controller, self.c_p, self.pipette_pump)
        self.pumpMonitorThread.progress.connect(self.update_pressures)
        self.pumpMonitorThread.start()

        self.update_timer = QTimer()
        self.update_timer.setInterval(500)
        self.update_timer.timeout.connect(self.refresh)
        self.update_timer.start()
        print("Pump monitor started")

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setWindowTitle("Microfluidics Controller")
        self.create_channel_controls()

        # Create button for calibrating the pump
        # Also create button for connecting the pump and potentially also for disconnecting it
        # Eventually we will also need to add the valves here.

        # buttons for controlling the pump PSU
        self.pipette_pump_max_power_spinbox = QDoubleSpinBox()
        self.pipette_pump_max_power_spinbox.setRange(0, 12)
        self.pipette_pump_max_power_spinbox.setSingleStep(0.1)
        self.pipette_pump_max_power_spinbox.setSuffix(" V")
        self.pipette_pump_max_power_spinbox.setValue(self.c_p['pipette_pump_current_power'])
        self.pipette_pump_max_power_spinbox.valueChanged.connect(
            lambda value: self.set_pipette_pump_power(value))
        self.layout.addWidget(self.pipette_pump_max_power_spinbox)

         # TODO have this react to external events
        self.toggle_pipette_pump_button = QPushButton("Toggle Pump PSU")
        self.toggle_pipette_pump_button.clicked.connect(lambda: self.toggle_pipette_pump())
        self.toggle_pipette_pump_button.setCheckable(True)
        self.toggle_pipette_pump_button.setChecked(self.c_p['pipette_pump_on'])
        self.layout.addWidget(self.toggle_pipette_pump_button)

        self.layout.addWidget(QLabel("Valves"))
        self.valve_buttons = []
        for valve_index in self.c_p['valves_used']:
            self.valve_buttons.append(QPushButton(f"Valve {valve_index}"))
            self.valve_buttons[-1].setCheckable(True)
            self.valve_buttons[-1].setChecked(self.c_p['valves_open'][valve_index])
            self.valve_buttons[-1].clicked.connect(
                lambda checked, valve_index=valve_index: self.toggle_valve(valve_index))
            self.valve_buttons[-1].setStyleSheet("""
                QPushButton {
                    background-color: red;
                }
                QPushButton:checked {
                    background-color: green;
                    color: white;
                }
                """)
            self.layout.addWidget(self.valve_buttons[-1])

        self.setLayout(self.layout)
        self.show()
    
    def toggle_valve(self, valve_index):
        self.c_p['valves_open'][valve_index] = not self.c_p['valves_open'][valve_index]

    def create_channel_controls(self):
        """
        Crates the UI elments needed to control the channels of the pump
        """
        self.pressure_spinboxes = []
        self.pressure_monitor_labels = []

        for channel in range(self.microfluidicsController.get_number_channels()):
            # Create a label for the channel
            label = QLabel("Channel " + str(channel+1))
            self.layout.addWidget(label)

            # Create a spinbox for setting the pressure
            self.pressure_spinboxes.append(QDoubleSpinBox())
            self.pressure_spinboxes[-1].setRange(0, 2000)
            self.pressure_spinboxes[-1].setSingleStep(0.1)
            self.pressure_spinboxes[-1].setSuffix(" mbar")
            self.pressure_spinboxes[-1].valueChanged.connect(
                lambda value, channel=channel: self.set_pressure(channel, value))
            self.layout.addWidget(self.pressure_spinboxes[-1])

            # Create a label for monitoring the pressure
            self.pressure_monitor_labels.append(
                QLabel(f"Pressure {self.c_p['current_pressures'][channel]} mbar"))
            self.layout.addWidget(self.pressure_monitor_labels[-1])

    def update_pressures(self, values):
        for channel in range(self.microfluidicsController.get_number_channels()):
            self.pressure_monitor_labels[channel].setText(
                f"Pressure {self.c_p['current_pressures'][channel]} mbar")
            self.pressure_spinboxes[channel].setValue(self.c_p['target_pressures'][channel])

    def set_pressure(self, channel, pressure):
        self.c_p['target_pressures'][channel] = float(pressure)

    def set_pipette_pump_power(self, value):
        self.c_p['pipette_pump_target_power'] = value

    def toggle_pipette_pump(self):
        self.c_p['pipette_pump_on'] = self.toggle_pipette_pump_button.isChecked()

    def refresh(self):
        
        self.pipette_pump_max_power_spinbox.setValue(self.c_p['pipette_pump_target_power'])
        self.toggle_pipette_pump_button.setChecked(self.c_p['pipette_pump_on'])

        for button, index in zip(self.valve_buttons, self.c_p['valves_used']):
            button.setChecked(self.c_p['valves_open'][index])

    def closeEvent(self, event):
        event.accept()
        self.pipette_pump.disconnect_from_psu()
            
