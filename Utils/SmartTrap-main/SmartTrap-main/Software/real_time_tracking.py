"""
In this file the various classes used for tracking are defined.
-----------------------------------
Classes:

- ObjectTracker: Abstract class for the tracking of particles and objects.
- TestTracker: Mock implementation of ObjectTracker used for testing.
- TrackingControlWidget: A QWidget used to configure the tracking in the system.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, Tuple, List

import numpy as np

from PyQt6.QtWidgets import (
    QPushButton, QVBoxLayout, QWidget, QFileDialog
)

@runtime_checkable
class ObjectTracker(Protocol):
    """
    ObjectTracker parent class. Assumes that analyze_frame is called first and tehn the particle
    particle positions and pipette are extracted from the predict position functions. The z-position
    is also made in similar manner.
    The load functions are used to load the correct network(s).
    """

    # Pipeline
    def analyze_frame(self, frame: Any) -> None: ...
    def predict_particle_positions(self) -> Tuple[np.ndarray, np.ndarray]: ...
    def predict_pipette_position(self) -> Tuple[List[float], List[float], bool]: ...
    def predict_z_positions(self, frame: Any, positions: Any) -> np.ndarray: ...
    def pipette_tilt_detection(self, dict) -> None:...

    # GUI loaders
    def load_2D_model(self) -> None: ...
    def load_z_model(self) -> None: ...


class TestTracker(ObjectTracker):
    """
    A mock implementation of the ObjecTracker used for testing.
    """
    def __init__(self):
        self.pipett_location = [0, 0, 0, 0]  # x, y, width, height
        self.pipette_tip_location = [0, 0]  # x, y
        self.results = None

    def analyze_frame(self, frame):
        
        pass

    def predict_particle_positions(self):
        return np.array([[np.random.randint(0,100), np.random.randint(0,100)]]), np.array([2])

    def predict_pipette_position(self):
        pipette_located = True
        self.pipett_location[0] = 200
        self.pipett_location[1] = 200
        self.pipett_location[2] = 50
        self.pipett_location[3] = 200
        self.pipette_tip_location[0] = 200
        self.pipette_tip_location[1] = 100
        return self.pipett_location, self.pipette_tip_location, pipette_located

    def predict_z_positions(self, frame, positions):
        z_vals = []
        for pos in positions:
            x = (pos[0]-256)/2
            z_vals.append(x)
        return np.array(z_vals)

    def pipette_tilt_detection(self,_):

        return


class TrackingControlWidget(QWidget):
    """
    Widget used to control the tracking in the system. Importantly allows users to turn off the
    tracking which can be important for saving compute resources in less powerful computers.
    """
    def __init__(self, c_p, object_tracker=None):
        super().__init__()
        self.c_p = c_p
        self.object_tracker = object_tracker
        layout = QVBoxLayout()
        
        self.setWindowTitle("Tracking controller")

        self.toggle_tracking_button = QPushButton('Tracking on')
        self.toggle_tracking_button.pressed.connect(self.toggle_tracking)
        self.toggle_tracking_button.setCheckable(True)
        self.toggle_tracking_button.setChecked(self.c_p['tracking_on'])
        layout.addWidget(self.toggle_tracking_button)


        # The various toggles tells the autocontroller which features to use.
        self.toggle_z_tracking_button = QPushButton('Z-tracking on')
        self.toggle_z_tracking_button.pressed.connect(self.toggle_z_tracking)
        self.toggle_z_tracking_button.setCheckable(True)
        self.toggle_z_tracking_button.setChecked(self.c_p['z-tracking'])
        layout.addWidget(self.toggle_z_tracking_button)

        self.set_Z_zero_button = QPushButton('Set Z offset')
        self.set_Z_zero_button.pressed.connect(self.set_Z_zero)
        self.set_Z_zero_button.setCheckable(False)
        layout.addWidget(self.set_Z_zero_button)

        self.load_z_model_button = QPushButton('Load z-model')
        self.load_z_model_button.pressed.connect(self.load_z_model)
        self.load_z_model_button.setCheckable(False)
        layout.addWidget(self.load_z_model_button)

        self.load_2D_model_button = QPushButton('Load 2D model')
        self.load_2D_model_button.pressed.connect(self.load_2D_model)
        self.load_2D_model_button.setCheckable(False)
        layout.addWidget(self.load_2D_model_button)
                
        self.setLayout(layout)

    def load_z_model(self):
        network_name = QFileDialog.getOpenFileName(self, 'Load network')
        print(f"Opening network {network_name[0]}")        
        self.object_tracker.load_z_model(network_name[0])
        
    def toggle_tracking(self):
        self.c_p['tracking_on'] = not self.c_p['tracking_on']

    def toggle_z_tracking(self):
        self.c_p['z-tracking'] = not self.c_p['z-tracking']

    def set_Z_zero(self):
        if self.c_p['particle_trapped']:
            self.c_p['z-offset'] += self.c_p['Trapped_particle_position'][2]

    def load_2D_model(self):
        
        network_name = QFileDialog.getOpenFileName(self, 'Load network')
        print(f"Opening network {network_name[0]}")
        self.object_tracker.load_2D_model(network_name[0])

