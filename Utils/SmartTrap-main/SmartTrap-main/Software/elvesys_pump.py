"""
Defines the classes needed to use the microfluidics from elvesys.
----------------------------------------------------
Classes:

- MUXWireValveController: A ValveController which is used to control the MUXWire controller.
- ElvesysMicrofluidicsController: A MicrofluidicsController which is used with the OB1 microfluidics
pump.
- PipettePump: A Pipette pump used powered by a TENMA programmable power supply.
"""

import sys
from email.header import UTF8

sys.path.append("ElveysisPump/SDK_V3_08_02/DLL/DLL64")#add the path of the library here
sys.path.append("ElveysisPump/SDK_V3_08_02/DLL/Python/Python_64")

"""
Note there is a line which needs to be changed in the Elveflow64.py file to get the correct import:
Replaced: ElveflowDLL=CDLL('D:/dev/SDK/DLL64/DLL64/Elveflow64.dll')
With: ElveflowDLL=CDLL("path2DLL/Elveflow64.dll")
where path2DLL is the path to the DLL on your system relative to this script.
Similar corrections may need to be made when installing on a different system.
"""
from ctypes import *
from Elveflow64 import *
import serial
import time
import numpy as np
from microfluidics_controllers import MicrofluidicsController, ValveController, PipettePump

from ctypes import c_int32, byref, c_double
import numpy as np

class MUXWireValveController(ValveController):
    def __init__(self):
        self._connected = False
        self.Instr_ID = c_int32()
        # keep hardware buffer of 16; logical user space is 8 valves
        self._states = np.zeros(16, dtype=int)

    def connect(self, address: str | None = "COM3") -> None:
        if address is None:
            address = "COM3"
        error = MUX_Initialization(address.encode("ascii"), byref(self.Instr_ID))
        if error != 0:
            raise RuntimeError(f"MUX init failed (addr={address}, error={error})")
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    def toggle_valve(self, valve_id: str, state: int) -> None:
        """
        valve_id: "0".."7" (string per ); we coerce to int for the device.
        """
        if not self._connected:
            raise RuntimeError("MUX not connected")
        idx = int(valve_id)
        if not (0 <= idx <= 7):
            raise ValueError(f"Valve index out of range: {idx} (expected 0..7)")
        self._states[idx] = state
        self._push_states_to_device()

    def get_valve_states(self) -> list[int]:
        # return a mapping as specified by the 
        return self._states

    # --- helpers ---
    def _push_states_to_device(self) -> None:
        valve_state = (c_int32 * 16)(0)
        for i in range(16):
            valve_state[i] = c_int32(int(self._states[i]))
        error = MUX_Set_all_valves(self.Instr_ID.value, valve_state, 16)
        if error != 0:
            raise RuntimeError(f"Failed to set valve states (error={error})")



class ElvesysMicrofluidicsController(MicrofluidicsController):
    """
    Controller class to be used with the OB1 controlelr from Elvesys.
    Allows for easily setting and checking the pressure on the different channels through the 
    set_pressure and get_pressure functions.
    """
    def __init__(self):
        self.Instr_ID = c_int32()
        self.nbr_channels = 3
        self.Calib = (c_double * 1000)()
        self._connected = False

    def connect(self, address: str | None = None) -> None:
        """
        Conencts to a controller at the specified adress (e.g. "COM1").

        """
        if not address:
            raise ValueError("address is required for Elvesys OB1")
        error = OB1_Initialization(address.encode("ascii"), 0, 0, 0, 0, byref(self.Instr_ID))
        print(f"OB1 init error: {error}, ID: {self.Instr_ID.value}")
        if self.Instr_ID.value < 0 or error != 0:
            raise RuntimeError(
                f"Failed to connect OB1 at {address} (error={error}, id={self.Instr_ID.value})")
        self._connected = True

    def disconnect(self) -> None:
        """
        Disconnects the controller
        """
        if not self._connected:
            return
        error = OB1_Destructor(self.Instr_ID.value)
        if error != 0:
            raise RuntimeError(f"OB1_Close failed (error={error})")
        self._connected = False

    def set_pressure(self, channel: str, value_mpa: float) -> None:
        if not self._connected:
            raise RuntimeError("OB1 not connected")
        ch = c_int32(int(channel))
        if value_mpa < 0.1:
            value_mpa = 0.1
        target = c_double(float(value_mpa))
        error = OB1_Set_Press(self.Instr_ID.value, ch, target, byref(self.Calib), 1000)
        if error != 0:
            raise RuntimeError(
                f"OB1_Set_Press failed (ch={channel}, mPa={value_mpa}, error={error})")

    def get_pressure(self, channel: str) -> float:
        if not self._connected:
            raise RuntimeError("OB1 not connected")
        ch = c_int32(int(channel))
        out = c_double()
        error = OB1_Get_Press(self.Instr_ID.value, ch, 1, byref(self.Calib), byref(out), 1000)
        if error != 0:
            raise RuntimeError(f"OB1_Get_Press failed (ch={channel}, error={error})")
        return float(out.value)

    # Optional extra API (not in )
    def get_number_channels(self) -> int:
        return self.nbr_channels


class PipettePump(PipettePump):
    """
    This class is used to control the Tenma 72-2540 PSU that powers the pump which is attached to
    the pipette. The same PSU can also be used for the pipette puller.    
    """
    def __init__(self, baud_rate=9600):
        self._connected = False
        self._output_on = False

    def connect(self, com_port, baud_rate=9600):
        try:
            # Establish a serial connection to the PSU
            ser = serial.Serial(com_port, baud_rate, timeout=1)
            print(f"Connected to PSU on {com_port} at {baud_rate} baud.")

            # Example command to send (modify as per your PSU's protocol)
            ser.write(b'*IDN?\n') # Example command, replace with your PSU's specific command

            # Wait for response and read it
            time.sleep(1)  # Adjust as necessary
            self.ser = ser

            response = ser.readline().decode().strip()

            print(f"Response from PSU: {response}")
            self._connected = True
            # Close the serial connection
        except Exception as e:
            print(f"Error: {e}")

    def disconnect(self):
        self.ser.close()

    def is_connected(self):
        return self._connected

    # Function to set voltage
    def set_voltage(self, voltage):
        if not self._connected:
            return
        command = f"VSET1:{voltage}\n".encode()
        self.ser.write(command)

    # Function to set current
    def set_current(self,current):
        command = f"ISET1:{current}\n".encode()
        self.ser.write(command)

    def activate_suction(self):
        if not self._connected:
            return  
        command = f"OUT1:\n".encode()
        self.ser.write(command)
        self._output_on = True

    def set_power(self, power):
        self.set_voltage(power)  # Assuming power is proportional to the voltage

    def deactivate_suction(self):
        command = f"OUT0:\n".encode()
        self.ser.write(command)
        self._output_on = False
    
    def suction_active(self):
        return self._output_on

     # Function to read voltage

    def get_power(self):
        return self.read_voltage()

    def read_voltage(self):
        command = f"VOUT1?\n".encode()
        self.ser.write(command)
        voltage = self.ser.readline().decode().strip()
        return float(voltage)

    def read_current(self):
        command = f"IOUT1?\n".encode()
        self.ser.write(command)
        current = self.ser.readline().decode().strip()
        return float(current)

    def get_status(self):
        from time import sleep
        command = f"STATUS?\n".encode()
        self.ser.write(command)
        sleep(0.01)
        status = self.ser.readline().decode().strip()
        if len(status)>0:
            print("status is ", ord(status))
            sixth_bit =  ord(status) & 64
            print("sixth bit is ", sixth_bit)
        return status
