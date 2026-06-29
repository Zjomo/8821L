"""
The interface and functions needed to control the pipette puller. Run this from the terminal to
start the interface used for pipette pulling.
-------------------------------------------------------
Classes

- PowerSupply: Protocol for power supply used to power the pipette puller. Implement this protocol
if you are looking to use a different power supply from the standard (TENMA) with your puller.
- TenmaPullerPSU: Implementation of the PowerSupply class compatible with TENMA power supplies.
- PSUControlPanel: Widget used for defining and controlling pipette pulling protocols.
- CurrentProtocolThread: Thread used to gradually increase (ramp up) the current going trough the
pipette puller and control the heating.
- main: Run this to start the program.
"""

import serial
import time
import sys

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QDoubleSpinBox, QGridLayout
from PyQt6.QtCore import QEvent, QThread, pyqtSignal, Qt
from time import sleep
import numpy as np

from typing import Protocol, runtime_checkable, Optional


@runtime_checkable
class PowerSupply(Protocol):
    """
    Minimal interface for a serial-controlled bench PSU used by the puller.

    This protocol is intentionally aligned to the current TenmaPullerPSU class so
    that it can be replaced with other models with zero changes to calling code.
    """

    # --- Required public attributes ---
    com_port: str
    baud_rate: int

    # --- Lifecycle & connection ---
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def set_com_port(self, com_port: str) -> None: ...

    # --- Output control ---
    def output_on(self) -> None: ...
    def output_off(self) -> None: ...

    # --- Setpoints ---
    # Accept str or float because TenmaPullerPSU formats strings for the SCPI-ish commands.
    def set_voltage(self, voltage: float | str) -> None: ...
    def set_current(self, current: float | str) -> None: ...

    # --- Measurements / queries ---
    # TenmaPullerPSU currently returns strings read from the serial buffer.
    def read_voltage(self) -> Optional[str]: ...
    def read_current(self) -> Optional[str]: ...
    def get_status(self) -> Optional[str]: ...


class TenmaPullerPSU(PowerSupply):
    def __init__(self, com_port="COM6"):
        self.com_port = com_port
        self.baud_rate = 9600   # The baud rate of the PSU
        self._is_connected = False
        self.is_output_on = False
        self.ser = None
        self.connect()

    def connect(self):
        if self._is_connected:
            return
        try:
            # Establish a serial connection to the PSU
            ser = serial.Serial(self.com_port, self.baud_rate, timeout=1)
            print(f"Connected to PSU on {self.com_port} at {self.baud_rate} baud.")

            # Example command to send (modify as per your PSU's protocol)
            ser.write(b'*IDN?\n')

            # Wait for response and read it
            time.sleep(0.2)  # Adjust as necessary
            response = ser.readline().decode().strip()
            print(f"Response from PSU: {response}")

            # Close the serial connection
            self._is_connected = True
            self.ser = ser

        except Exception as e:
            self._is_connected = False
            self.ser = None
            print(f"Error: {e}")

    def disconnect(self):
        if not self._is_connected:
            print("PSU not connected")
            return
        self.ser.close()
        self.ser = None
        self._is_connected = False

    def is_connected(self):
        return self._is_connected

    # TODO add possiblity to set com-port after starting

    # Function to set voltage
    def set_voltage(self, voltage):
        if not self._is_connected:
            print("PSU not connected")
            return
        command = f"VSET1:{voltage}\n".encode()
        self.ser.write(command)

    # Function to set current
    def set_current(self, current):
        if not self._is_connected:
            print("PSU not connected")
            return
        command = f"ISET1:{current}\n".encode()
        self.ser.write(command)

    def output_on(self):
        if not self._is_connected:
            print("PSU not connected")
            return
        command = f"OUT1:\n".encode()
        self.is_output_on = True
        self.ser.write(command)

    def output_off(self):
        if not self._is_connected:
            print("PSU not connected")
            return
        command = f"OUT0:\n".encode()  
        self.ser.write(command)
        self.is_output_on = False


    def read_voltage(self):
        if not self._is_connected:
            print("PSU not connected")
            return
        command = f"VOUT1?\n".encode()
        self.ser.write(command)
        voltage = self.ser.readline().decode().strip()
        return voltage

    def read_current(self):
        if not self._is_connected:
            print("PSU not connected")
            return
        command = f"IOUT1?\n".encode()
        self.ser.write(command)
        current = self.ser.readline().decode().strip()
        return current

    def get_status(self):
        if not self._is_connected:
            print("PSU not connected")
            return
        command = f"STATUS?\n".encode()
        self.ser.write(command)
        sleep(0.01)
        status = self.ser.readline().decode().strip()
        if len(status)>0:
            print("status is ", ord(status))
            sixth_bit =  ord(status) & 64
            print("sixth bit is ", sixth_bit)
        return status


class PSUControlPanel(QWidget):
    """
    Widget used for defining and controlling pipette pulling protocols.
    Will automatically connect to the default power supply.
    """
    def __init__(self,
                 COM_PORT="COM5",
                 ramp_duration=8.5,
                 max_time=11,
                 max_current=3.27,
                 ):
        super().__init__()

        # Change the PSU here if you are using a different power supply.
        self.PSU = TenmaPullerPSU(COM_PORT)
        self.output_on = False
        self.protocol = [
            [0.1, 0.5],
            [0.2, 0.5],
            [0.3, 0.5],
            [0.4, 0.5],
            [0.5, 0.5],
            [0.6, 0.5],
            [0.7, 0.5],
            [0.8, 0.5],
            [0.9, 0.5],
            [1.0, 0.5],
        ]
        self.ramp_duration = ramp_duration
        self.max_time = max_time
        self.max_current = max_current
        self.ramp_frequency = 20
        self.voltage = 5
        self.initUI()
        self.create_protocol()
        self.PSU.set_voltage(self.voltage)        

    def initUI(self):
        # Layout
        layout = QGridLayout()

          # Voltage control
        self.voltage_input = QDoubleSpinBox(self)
        self.voltage_input.setRange(0, 10)  # Set voltage limits (0V to 30V, adjust as needed)
        self.voltage_input.setSingleStep(0.1)
        self.voltage_input.setValue(self.voltage)
        self.set_voltage_button = QPushButton('Set Voltage', self)
        self.set_voltage_button.clicked.connect(self.PSU.set_voltage)
        layout.addWidget(QLabel('Voltage (V):'), 0, 0)
        layout.addWidget(self.voltage_input, 0, 1)
        layout.addWidget(self.set_voltage_button, 0, 2)

        # Current control
        self.current_input = QDoubleSpinBox(self)
        self.current_input.setRange(0, 3.5)  # Set current limits (0A to 5A, adjust as needed)
        self.current_input.setSingleStep(0.1)
        self.set_current_button = QPushButton('Set Current', self)
        self.set_current_button.clicked.connect(self.set_current)
        layout.addWidget(QLabel('Current (A):'), 1, 0)
        layout.addWidget(self.current_input, 1, 1)
        layout.addWidget(self.set_current_button, 1, 2)

        # Output control
        self.output_button = QPushButton('Toggle Output', self)
        self.output_button.clicked.connect(self.toggle_output)
        self.output_button.setCheckable(True)
        self.output_button.setChecked(self.PSU.is_output_on)
        layout.addWidget(self.output_button, 2, 0)

        # Disconnect
        self.disconnect_button = QPushButton('Disconnect', self)
        self.disconnect_button.clicked.connect(self.disconnect)
        layout.addWidget(self.disconnect_button, 2, 1)

        self.create_protocol_button = QPushButton('Create Protocol', self)
        self.create_protocol_button.clicked.connect(self.create_protocol)
        layout.addWidget(self.create_protocol_button, 3, 0)

        # Button to start the protocol
        self.start_protocol_button = QPushButton('Start Protocol', self)
        self.start_protocol_button.clicked.connect(self.start_protocol)
        layout.addWidget(self.start_protocol_button, 3, 1)

        # Button to stop the protocol
        self.stop_protocol_button = QPushButton('Stop Protocol', self)
        self.stop_protocol_button.clicked.connect(self.stop_protocol)
        layout.addWidget(self.stop_protocol_button, 3, 2)


        layout.addWidget(QLabel('Ramp duration (S):'), 4, 0)
        self.ramp_duration_input = QDoubleSpinBox(self)
        self.ramp_duration_input.setRange(0, 100)
        self.ramp_duration_input.setSingleStep(0.1)
        self.ramp_duration_input.setValue(self.ramp_duration)
        layout.addWidget(self.ramp_duration_input, 4, 1)
        self.set_ramp_duration_button = QPushButton('Set Ramp Duration', self)
        self.set_ramp_duration_button.clicked.connect(self.set_ramp_duration)
        layout.addWidget(self.set_ramp_duration_button, 4, 2)        

        layout.addWidget(QLabel('Max time (S):'), 5, 0)
        self.max_time_input = QDoubleSpinBox(self)
        self.max_time_input.setRange(0, 100)
        self.max_time_input.setSingleStep(0.1)
        self.max_time_input.setValue(self.max_time)
        layout.addWidget(self.max_time_input, 5, 1)
        self.set_max_time_button = QPushButton('Set Max Time', self)
        self.set_max_time_button.clicked.connect(self.set_max_time)
        layout.addWidget(self.set_max_time_button, 5, 2)

        self.max_current_input = QDoubleSpinBox(self)
        self.max_current_input.setRange(0, 3.5)
        self.max_current_input.setSingleStep(0.1)
        self.max_current_input.setValue(self.max_current)
        layout.addWidget(QLabel('Max current (A):'), 6, 0)
        layout.addWidget(self.max_current_input, 6, 1)
        self.set_max_current_button = QPushButton('Set Max Current', self)
        self.set_max_current_button.clicked.connect(self.set_max_current)
        layout.addWidget(self.set_max_current_button, 6, 2)

        self.protocol_label = QLabel("Current protocol")
        # layout.addWidget(self.protocol_label)
        self.protocol_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Place it on row 7, col 0, spanning 1 row and 3 columns
        layout.addWidget(
            self.protocol_label,
            7, 0,
            1, 3,
            Qt.AlignmentFlag.AlignCenter
        )
        # Set the layout
        self.setLayout(layout)
        self.setWindowTitle('Pipette puller Control Panel')

    def set_protocol_label(self):
        """
        Update the protocol label so it reflects the current protocol parameters.
        Prefers the model's attributes if present; otherwise falls back to UI inputs.
        """
        print("Setting protocol")
        # --- Pull parameters (prefer attributes created by your logic; else read the widgets) ---
        voltage = getattr(self, "voltage", self.voltage_input.value())
        # Current setpoint (if you keep one); else show what's in the input field
        current_setpoint = getattr(self, "current", self.current_input.value())
        ramp_duration = getattr(self, "ramp_duration", self.ramp_duration_input.value())
        max_time = getattr(self, "max_time", self.max_time_input.value())
        max_current = getattr(self, "max_current", self.max_current_input.value())

        # PSU info (best-effort)
        psu_name = type(self.PSU).__name__ if hasattr(self, "PSU") else "—"
        psu_port = getattr(self.PSU, "com_port", "—") if hasattr(self, "PSU") else "—"
        psu_output_on = getattr(self.PSU, "is_output_on", False)

        # --- Derived values ---
        if ramp_duration is None or ramp_duration == 0:
            ramp_str = "instant"
            ramp_rate_str = "∞ V/s"
        else:
            ramp_rate = float(voltage) / float(ramp_duration)
            ramp_str = f"{ramp_duration:.2f} s"
            ramp_rate_str = f"{ramp_rate:.3f} V/s"

        # --- Compose a compact, readable label (rich text) ---
        text = (
            "<b>Current protocol</b><br>"
            f"• Voltage setpoint: <b>{float(voltage):.3f} V</b><br>"
            f"• Current value: <b>{float(current_setpoint):.3f} A</b> "
            f"(limit: <b>{float(max_current):.3f} A</b>)<br>"
            f"• Ramp: <b>{ramp_str}</b> @ <b>{ramp_rate_str}</b><br>"
            f"• Max time: <b>{float(max_time):.2f} s</b><br>"
            f"• PSU: <b>{psu_name}</b> on <b>{psu_port}</b> — Output: "
            f"<b>{'ON' if psu_output_on else 'OFF'}</b><br>"
            f"• Connection: <b>{'CONNECTED' if self.PSU.is_connected() else 'DISCONNECTED'}</b><br>"
        )

        self.protocol_label.setText(text)

    def set_voltage(self):
        voltage = self.voltage_input.value()
        self.PSU.set_voltage(voltage)  # Assuming set_voltage function and PSU object

    def set_current(self):
        current = self.current_input.value()
        self.PSU.set_current(current)  # Assuming set_current function and PSU object

    def set_ramp_duration(self):
        ramp_duration = self.ramp_duration_input.value()
        self.ramp_duration = ramp_duration

    def set_max_time(self):
        max_time = self.max_time_input.value()
        self.max_time = max(max_time, self.ramp_duration)
    
    def set_max_current(self):
        max_current = self.max_current_input.value()
        self.max_current = max_current

    def toggle_output(self):
        # This function should toggle the PSU output on or off
        # You need to keep track of the state of the PSU output
        if self.output_on == False:
            self.PSU.output_on()
            self.output_on = True
        else:
            self.PSU.output_off()
            self.output_on = False

    def create_protocol(self):
        self.protocol = np.ones((int(self.max_time*self.ramp_frequency),2))
        self.protocol[:,1] = np.linspace(0,self.max_time, int(self.max_time*self.ramp_frequency))
        self.protocol[:,0] *= self.max_current
        ramp_voltages = np.linspace(0,self.max_current, int(self.ramp_duration*self.ramp_frequency))
        self.protocol[:len(ramp_voltages),0] = ramp_voltages
        self.set_protocol_label()

    def start_protocol(self):
        if not self.output_on:
            self.toggle_output()
        self.PSU.output_on()
        self.output_on = True
        self.protocol_thread = CurrentProtocolThread(self.PSU, self.protocol)
        self.protocol_thread.update_signal.connect(self.handle_protocol_update)
        self.protocol_thread.start()

    def stop_protocol(self):
        if self.protocol_thread:
            self.PSU.output_off()
            self.output_on = False
            if self.output_on:
                self.toggle_output()

    def handle_protocol_update(self, message):
        # Handle updates from the protocol thread (e.g., update a status label)
        pass


    def closeEvent(self, event: QEvent):
        """
        Reimplemented close event to handle PSU disconnection
        when the GUI window is closed.
        """
        # Add your PSU disconnect logic here
        self.PSU.disconnect()
        event.accept()

class CurrentProtocolThread(QThread):
    """
    Thread used to gradually increase (ramp up) the current going trough the pipette puller and
    control the heating. Will automatically also turn on and off the output (power).
    """

    # Signal to update the GUI or status
    update_signal = pyqtSignal(str)

    def __init__(self, PSU: PowerSupply, protocol):
        super().__init__()
        self.PSU = PSU
        self.protocol = protocol
        self.running = False

    def run(self):
        self.running = True
        start = time.time()
        for current, timing in self.protocol:
            while time.time() - start < timing:
                sleep(0.005)
            print(time.time() - start, current)
            if not self.running:
                break
            self.PSU.set_current(current)  # Assuming set_voltage function
            self.update_signal.emit(f"Current set to {current} A")
        self.update_signal.emit("Protocol completed")
        self.PSU.output_off()

    def stop(self):
        self.running = False
        self.update_signal.emit("Protocol stopped")
        self.PSU.output_off()


def main():
    app = QApplication(sys.argv)
    ex = PSUControlPanel(
        COM_PORT="COM6",
                 ramp_duration=8.5,
                 max_time=10,
                 max_current=3.25, # CHeck this
                 
    )
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
