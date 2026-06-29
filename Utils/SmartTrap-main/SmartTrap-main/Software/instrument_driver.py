"""
This file defines the classes needed to communicate with the instrument.
---------------------------------------------
Classes

- InstrumentDriver: An abstract class which defines an interface for drivers for optical tweezers
instruments.
- InstrumentControllerThread: A thread which runs in the background and uses an InstrumentDriver
to sample data from the opticalt tweezers instrument.
"""

from typing import Protocol
from threading import Thread


class InstrumentDriver(Protocol):
    """
    Driver class for the instrument. The instrument drivers' run function is called repeatedly to 
    send commands to the instrument and read data from it into the data_channels(it is the 
    instrument driver that should sample the various detectors of the instrument).
    """
    def run(self) -> None: ...
    def is_connected(self) -> bool: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...


class InstrumentControllerThread(Thread):
    """
    This thread handles communications with the instrument. Continously calls the run function
    of the instrument driver to read data into the data_channels.
    """
    def __init__(self, c_p, instrument_driver):
        Thread.__init__(self)
        self.c_p = c_p
        self.instrument_driver = instrument_driver

    def run(self):
        self.instrument_driver.connect()
        while self.c_p['program_running']:
            self.instrument_driver.run()
        self.instrument_driver.disconnect()