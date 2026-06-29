"""
This file contains tools used to manually move the lasers in the SmartTrap system.
--------------------------------------------------------
Classes:

- LaserPiezoWidget: A Qwwidget used to move the lasers of the two standard lasers in the SmartTrap.
- MinitweezersLaserMove: A MouseTool used for manually moving the lasers in the sample by clicking
and dragging.
"""

from PyQt6.QtWidgets import (QDoubleSpinBox, QSlider, QHBoxLayout,
    QPushButton, QVBoxLayout, QWidget, QLabel
)

from PyQt6.QtCore import Qt, QTimer

import numpy as np

from mouse_tool import MouseTool


class LaserPiezoWidget(QWidget):
    """
    Defines a simple widget for manually moving the two lasers of the system by using sliders.
    Can also toggle the automatic alignment (autoalign) of the lasers.
    The widget updates the laser sliders if there is a protcol separate running, indicating 
    the relative laser position.
    """

    def __init__(self, c_p, data_channels):
        super().__init__()
        self.c_p = c_p
        self.data_channels = data_channels
        self.setWindowTitle("Piezo wiggler Controller")
        layout = QVBoxLayout()
        self.label = QLabel("Piezo A x")
        layout.addWidget(self.label)

        # Create the slider for the piezo A x-axis
        self.piezo_Ax_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.piezo_Ax_slider.setGeometry(50,50, 200, 50)
        self.piezo_Ax_slider.setMinimum(0)
        self.piezo_Ax_slider.setMaximum(65535)
        self.piezo_Ax_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.piezo_Ax_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.piezo_Ax_slider.setTickInterval(1)
        self.piezo_Ax_slider.valueChanged[int].connect(self.set_piezo_Ax_value) 

        layout.addWidget(self.piezo_Ax_slider)
        
        self.label_AY = QLabel("Piezo A y")
        layout.addWidget(self.label_AY)
        # Create the slider for the piezo A y-axis
        self.piezo_Ay_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.piezo_Ay_slider.setGeometry(50,50, 200, 50)
        self.piezo_Ay_slider.setMinimum(0)
        self.piezo_Ay_slider.setMaximum(65535)
        self.piezo_Ay_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.piezo_Ay_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.piezo_Ay_slider.setTickInterval(1)
        self.piezo_Ay_slider.valueChanged[int].connect(self.set_piezo_Ay_value) 

        layout.addWidget(self.piezo_Ay_slider)

        self.label_BX = QLabel("Piezo B x")
        layout.addWidget(self.label_BX)
        # Create the slider for the piezo B x-axis
        self.piezo_Bx_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.piezo_Bx_slider.setGeometry(50,50, 200, 50)
        self.piezo_Bx_slider.setMinimum(0)
        self.piezo_Bx_slider.setMaximum(65535)
        self.piezo_Bx_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.piezo_Bx_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.piezo_Bx_slider.setTickInterval(1)
        self.piezo_Bx_slider.valueChanged[int].connect(self.set_piezo_Bx_value) 

        layout.addWidget(self.piezo_Bx_slider)
        
        self.label_BY = QLabel("Piezo B y")
        layout.addWidget(self.label_BY)
        # Create the slider for the piezo B y-axis
        self.piezo_By_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.piezo_By_slider.setGeometry(50,50, 200, 50)
        self.piezo_By_slider.setMinimum(0)
        self.piezo_By_slider.setMaximum(65535)
        self.piezo_By_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.piezo_By_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.piezo_By_slider.setTickInterval(1)
        self.piezo_By_slider.valueChanged[int].connect(self.set_piezo_By_value) 

        layout.addWidget(self.piezo_By_slider)
        self.autoalign_A_button = QPushButton("Autoalign A")
        self.autoalign_A_button.clicked.connect(self.toggle_autoalign_A)
        self.autoalign_A_button.setCheckable(True)
        self.autoalign_A_button.setToolTip("Makes laser A automatically align to laser B "
                                           "(autoalign) so that they both lasers apply the same"\
                                            " force.")
        self.autoalign_A_button.setChecked(self.c_p['portenta_command_2']==1)
        layout.addWidget(self.autoalign_A_button)

        self.autoalign_B_button = QPushButton("Autoalign B")
        self.autoalign_B_button.clicked.connect(self.toggle_autoalign_B)
        self.autoalign_B_button.setCheckable(True)
        self.autoalign_B_button.setToolTip("Makes laser B automatically align to laser B "
                                           "(autoalign) so that they both lasers apply the same"\
                                            " force.")
        self.autoalign_B_button.setChecked(self.c_p['portenta_command_2']==2)
        layout.addWidget(self.autoalign_B_button)

        self.center_piezos_button = QPushButton("Center Piezos")
        self.center_piezos_button.clicked.connect(self.center_piezos)
        self.center_piezos_button.setCheckable(False)
        self.center_piezos_button.setToolTip("Moves the piezo actuators to the center of their" \
        " movement range, positioning the lasers in the middle of their movement range.")
        layout.addWidget(self.center_piezos_button)

        # Add the calibration spinboxes
        self.create_calibration_spinboxes()
        layout.addLayout(self.calibration_layout_a)
        layout.addLayout(self.calibration_layout_b)

        self.set_calibration_factors_button = QPushButton("Set calibration factors")
        self.set_calibration_factors_button.clicked.connect(self.set_calibration_factors)
        self.set_calibration_factors_button.setCheckable(False)
        self.set_calibration_factors_button.setToolTip("Set the calibration factors for " \
        "PSD-to-force conversion on the controller.\n Needed for accurate force measurements.")
        layout.addWidget(self.set_calibration_factors_button)
        

        self.set_slider_values()
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.setInterval(100) # sets the fps of the timer
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def create_calibration_spinboxes(self):
        self.calibration_layout_a = QHBoxLayout()

        # Add a QLabel with text "AX factor"
        ax_label = QLabel("AX factor")
        self.calibration_layout_a.addWidget(ax_label)

        self.AX_calibration_spinbox = QDoubleSpinBox()
        self.AX_calibration_spinbox.setRange(0, 0.65000)
        self.AX_calibration_spinbox.setSingleStep(0.00010)
        self.AX_calibration_spinbox.setDecimals(5)
        self.AX_calibration_spinbox.setValue(self.c_p['PSD_to_force'][0])
        self.AX_calibration_spinbox.valueChanged.connect(self.set_AX_calibration)

        # Add the QDoubleSpinBox to the QHBoxLayout
        self.calibration_layout_a.addWidget(self.AX_calibration_spinbox)

        # Add a QLabel with text "AY factor"
        ay_label = QLabel("AY factor")
        self.calibration_layout_a.addWidget(ay_label)

        self.AY_calibration_spinbox = QDoubleSpinBox()
        self.AY_calibration_spinbox.setRange(0, 0.65000)
        self.AY_calibration_spinbox.setSingleStep(0.00010)
        self.AY_calibration_spinbox.setDecimals(5)
        self.AY_calibration_spinbox.setValue(self.c_p['PSD_to_force'][1])
        self.AY_calibration_spinbox.valueChanged.connect(self.set_AY_calibration)

        # Add the QDoubleSpinBox to the QHBoxLayout
        self.calibration_layout_a.addWidget(self.AY_calibration_spinbox)
        
        
        self.calibration_layout_b = QHBoxLayout()

        # Add a QLabel with text "BX factor"
        bx_label = QLabel("BX factor")
        self.calibration_layout_b.addWidget(bx_label)

        self.BX_calibration_spinbox = QDoubleSpinBox()
        self.BX_calibration_spinbox.setRange(0, 0.65000)
        self.BX_calibration_spinbox.setSingleStep(0.00010)
        self.BX_calibration_spinbox.setDecimals(5)
        self.BX_calibration_spinbox.setValue(self.c_p['PSD_to_force'][2])
        self.BX_calibration_spinbox.valueChanged.connect(self.set_BX_calibration)

        # Add the QDoubleSpinBox to the QHBoxLayout
        self.calibration_layout_b.addWidget(self.BX_calibration_spinbox)

        # Add a QLabel with text "BY factor"
        by_label = QLabel("BY factor")
        self.calibration_layout_b.addWidget(by_label)

        self.BY_calibration_spinbox = QDoubleSpinBox()
        self.BY_calibration_spinbox.setRange(0, 0.65000)
        self.BY_calibration_spinbox.setSingleStep(0.00010)
        self.BY_calibration_spinbox.setDecimals(5)
        self.BY_calibration_spinbox.setValue(self.c_p['PSD_to_force'][3])
        self.BY_calibration_spinbox.valueChanged.connect(self.set_BY_calibration)

        # Add the QDoubleSpinBox to the QHBoxLayout
        self.calibration_layout_b.addWidget(self.BY_calibration_spinbox)


    def refresh(self):
        self.set_slider_values()
        self.autoalign_A_button.setChecked(self.c_p['portenta_command_2']==1)
        self.autoalign_B_button.setChecked(self.c_p['portenta_command_2']==2)

    def set_AX_calibration(self):
        self.c_p['PSD_to_force'][0] = self.AX_calibration_spinbox.value()

    def set_AY_calibration(self):
        self.c_p['PSD_to_force'][1] = self.AY_calibration_spinbox.value()

    def set_BX_calibration(self):
        self.c_p['PSD_to_force'][2] = self.BX_calibration_spinbox.value()

    def set_BY_calibration(self):
        self.c_p['PSD_to_force'][3] = self.BY_calibration_spinbox.value()

    def set_calibration_factors(self):
        self.c_p['portenta_command_1'] = 3

    def set_slider_values(self):
        self.piezo_Ax_slider.setValue(self.c_p['piezo_A'][0])
        self.piezo_Ay_slider.setValue(self.c_p['piezo_A'][1])
    
        self.piezo_By_slider.setValue(self.c_p['piezo_B'][1])
        self.piezo_Bx_slider.setValue(self.c_p['piezo_B'][0])

    def center_piezos(self):
        self.c_p['piezo_A'] = [32768, 32768]
        self.c_p['piezo_B'] = [32768, 32768]
        self.set_slider_values()

    def set_piezo_Ax_value(self):
        self.c_p['piezo_A'][0] = int(self.piezo_Ax_slider.value())

    def set_piezo_Ay_value(self):
        self.c_p['piezo_A'][1] = int(self.piezo_Ay_slider.value())
    
    def set_piezo_Bx_value(self):
        self.c_p['piezo_B'][0] = int(self.piezo_Bx_slider.value())

    def set_piezo_By_value(self):
        self.c_p['piezo_B'][1] = int(self.piezo_By_slider.value())

    def toggle_autoalign_A(self):
        if self.c_p['portenta_command_2'] == 0 or self.c_p['portenta_command_2'] == 2:
            self.c_p['portenta_command_2'] = 1            
        else:
            self.c_p['piezo_A'] = np.int32([self.data_channels['dac_ax'].get_data_spaced(1)[0],
                                            self.data_channels['dac_ay'].get_data_spaced(1)[0]])
            self.c_p['piezo_B'] = np.int32([self.data_channels['dac_bx'].get_data_spaced(1)[0],
                                            self.data_channels['dac_by'].get_data_spaced(1)[0]])
            self.c_p['portenta_command_2'] = 0

            self.set_slider_values()

    def toggle_autoalign_B(self):
        if self.c_p['portenta_command_2'] == 0 or self.c_p['portenta_command_2'] == 1:
            self.c_p['portenta_command_2'] = 2
        else:
            self.c_p['piezo_A'] = np.int32([self.data_channels['dac_ax'].get_data_spaced(1)[0],
                                            self.data_channels['dac_ay'].get_data_spaced(1)[0]])
            self.c_p['piezo_B'] = np.int32([self.data_channels['dac_bx'].get_data_spaced(1)[0],
                                            self.data_channels['dac_by'].get_data_spaced(1)[0]])
            self.c_p['portenta_command_2'] = 0

            self.set_slider_values()

    
class MinitweezersLaserMove(MouseTool):
    """
    A MouseTool used to  manually moving the lasers in the sample by clicking and dragging.
    Left click for laser A and Right click for laser B.
    """
    def __init__(self, c_p):
        self.c_p = c_p
        self.speed_factor = 4_400 * self.c_p['microns_per_pix']
        self.x_prev_A = 0
        self.y_prev_A = 0
        self.x_prev_B = 0
        self.y_prev_B = 0

    def mousePress(self):
        # left click
        if self.c_p['mouse_params'][0] == 1:
            self.x_prev_A = self.c_p['mouse_params'][1]
            self.y_prev_A = self.c_p['mouse_params'][2]
        # Right click -drag
        if self.c_p['mouse_params'][0] == 2:
            self.x_prev_B = self.c_p['mouse_params'][1]
            self.y_prev_B = self.c_p['mouse_params'][2]
        # Scroll wheel
        elif self.c_p['mouse_params'][0] == 3:
            pass
        
    def mouseRelease(self):
        if self.c_p['mouse_params'][0] == 2:
            pass
        
    def mouseDoubleClick(self):
        pass
    
    def draw(self, qp):
        pass
    def check_limit(self, number):
        if number < 0:
            return 0
        if number>65535:
            return 65535
        return number

    def mouseMove(self):

        if self.c_p['mouse_params'][0] == 1: # A
            dx = int(self.c_p['image_scale']*(self.c_p['mouse_params'][3] - self.x_prev_A)*self.speed_factor)
            dy = int(self.c_p['image_scale']*(self.c_p['mouse_params'][4] - self.y_prev_A)*self.speed_factor)

            self.c_p['piezo_A'][0] = self.check_limit(dx+self.c_p['piezo_A'][0])
            self.c_p['piezo_A'][1] = self.check_limit(dy+self.c_p['piezo_A'][1])

            self.x_prev_A = self.c_p['mouse_params'][3]
            self.y_prev_A = self.c_p['mouse_params'][4] 

        if self.c_p['mouse_params'][0] == 2: # B
            dx = int(self.c_p['image_scale']*(self.c_p['mouse_params'][3] - self.x_prev_B)*self.speed_factor)
            dy = int(self.c_p['image_scale']*(self.c_p['mouse_params'][4] - self.y_prev_B)*self.speed_factor)

            self.c_p['piezo_B'][0] = self.check_limit(-dx+self.c_p['piezo_B'][0])
            self.c_p['piezo_B'][1] = self.check_limit(dy+self.c_p['piezo_B'][1])
            self.x_prev_B = self.c_p['mouse_params'][3]
            self.y_prev_B = self.c_p['mouse_params'][4] 

    def getToolName(self):
        return "Piezo manual move tool"

    def getToolTip(self):
        return "Move the laser by dragging on the screen.\n Left click for laser A, right click for laser B."
        

