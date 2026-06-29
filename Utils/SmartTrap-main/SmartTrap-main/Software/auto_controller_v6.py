"""
This file defines the autonomous protocols and functions.

------------------------------
Classes:

- StokesTestWidget: A small widget which is used to configure and run stokes drag tests.
- AutonomousProtocol: An autonomous protocol class which defines the functions the autonomous
protocols need to implement.
- DNAPulling: An AutonomousProtocol which runs the DNA pulling experiment
- ElectrostaticRepulsion: An AutonomousProtocol which runs the electrostatic particle-particle
interaction experiment
- ParticleCharacterization: An AutonomousProtocol which runs the particle characterization 
experiment (selecting particles in a specific size range and measuring their size)
- RBCStretching: An AutonomousProtocol which runs the red blood cell stretching experiment.
- SelectLaserPosition: A MouseTool which is used to indicate where in the camera feed the lasers
are.
- AutoControllerThread: A thread which runs the various autonomous protocols and subroutines as 
well as the real-time image analysis.
"""


import numpy as np
from PyQt6.QtWidgets import (QComboBox,
    QPushButton, QVBoxLayout, QWidget, QFileDialog,
    QHBoxLayout, QSpinBox,
)
from PyQt6 import  QtGui
from PyQt6.QtCore import QTimer
from typing import Protocol

from mouse_tool import MouseTool
from threading import Thread
from time import sleep, time

import cv2

import warnings

warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    message=".*torch.cuda.amp.autocast.*"
)

class StokesTestWidget(QWidget):
    """
    A widget to control the stokes test. Allows to set the left, right, center, up and down
    positions and to start and stop the test.
    The test moves the motor between the left and right positions and measures the force on
    the trapped particle.
    The test can be configured to move a maximum number of times before stopping.    
    """

    def __init__(self, c_p, motor_controller):
        super().__init__()
        self.c_p = c_p
        self.motor_controller = motor_controller
        self.stokes_test_step = "startup"

        self.init_ui()
                
        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
    
    def init_ui(self):
        """
        Initializes the user interface with buttons for setting positions and 
        toggling the stokes test.        
        """

        layout = QVBoxLayout()
        self.setWindowTitle("Stokes Test")

        self.toggle_test_button = QPushButton('Toggle Stokes test')
        self.toggle_test_button.pressed.connect(self.toggle_test)
        self.toggle_test_button.setCheckable(True)
        layout.addWidget(self.toggle_test_button)

        self.max_move_count_box = QSpinBox()
        self.max_move_count_box.setRange(1, 100)
        self.max_move_count_box.setValue(self.c_p['stokes_max_move_count'])
        self.max_move_count_box.valueChanged.connect(self.set_max_move_count)
        layout.addWidget(self.max_move_count_box)

        self.set_left_pos_button = QPushButton('Set left position')
        self.set_left_pos_button.pressed.connect(self.set_left_pos)
        layout.addWidget(self.set_left_pos_button)

        self.set_right_pos_button = QPushButton('Set right position')
        self.set_right_pos_button.pressed.connect(self.set_right_pos)
        layout.addWidget(self.set_right_pos_button)

        self.set_center_pos_button = QPushButton('Set center position')
        self.set_center_pos_button.pressed.connect(self.set_center_pos)
        layout.addWidget(self.set_center_pos_button)

        self.set_up_pos_button = QPushButton('Set up position')
        self.set_up_pos_button.pressed.connect(self.set_up_pos)
        layout.addWidget(self.set_up_pos_button)

        self.set_down_pos_button = QPushButton('Set down position')
        self.set_down_pos_button.pressed.connect(self.set_down_pos)
        layout.addWidget(self.set_down_pos_button)

        self.setLayout(layout)

    def toggle_test(self):
        if self.c_p['stokes_test_running']:
            print("Toggling stokes test")
            self.c_p['stokes_test_running'] = False
        else:
            self.c_p['stokes_test_running'] = True

    def set_max_move_count(self, count):
        self.c_p['stokes_max_move_count'] = int(count)

    def get_position(self):
        x = int(self.motor_controller.get_x_position())
        y = int(self.motor_controller.get_y_position())
        z = int(self.motor_controller.get_z_position())
        return [x, y, z]
    
    def calc_distance(self, pos1, pos2):
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)

    def refresh(self):
        self.toggle_test_button.setChecked(self.c_p['stokes_test_running'])
        if not self.c_p['stokes_test_running']:    
            return
        self.toggle_test_button.setChecked(self.c_p['stokes_test_running'])

    def set_left_pos(self):
        self.c_p['stokes_left_pos'] = self.get_position()
    
    def set_right_pos(self):
        self.c_p['stokes_right_pos'] = self.get_position()
    
    def set_center_pos(self):
        self.c_p['stokes_center_pos'] = self.get_position()

    def set_up_pos(self):
        self.c_p['stokes_up_pos'] = self.get_position()
    
    def set_down_pos(self):
        self.c_p['stokes_down_pos'] = self.get_position()


def convert_to_uint16(num):
    if num < 0:
        return np.uint16(0)
    if num > 65535:
        return np.uint16(65535)
    return np.uint16(num)


class AutonomousProtocol(Protocol):
    """
    Interface class for the autonomous protocols.
    The protocols are run using the run method which is called repeatedly by the AutoController.
    The protocol should maintain its own state (e.g. "aligning particles") and return the current 
    state as a string when run is called. The protocol can be started and stopped using the start
    and stop methods.    
    """
    def is_running(self) -> bool: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def run(self) -> str: ... # Should return the part of the autonomous protocol is being executed
    def reset_experiment(self) -> None:... # This should bring back the protocol to the starting 
    # state and reset parameters accordingly (e.g. pulling limits)

class DNAPulling(AutonomousProtocol):
    """
    AutonomousProtocol which is used to perform autonomous DNA pulling expriments.
    The protocol first collects two particles, one is trapped and the other is placed in the
    pipette. Then it tries to fish for a DNA molecule by touching the particles together and moving
    them apart repeatedly until a molecule is detected. Once a molecule is detected it aligns the
    particles in x and z and then finds the pulling limits. Finally it starts the pulling protocol.
    """

    def __init__(self, c_p, auto_controller,
                 stretching_distance=6,
                 min_stretch_distance=4,
                 stretch_force=69,
                 stretch_speed=100,):
        self.c_p = c_p
        self.auto_controller = auto_controller # So it can all on functions of the autocontroller 
        self.running = False
        self.stage = "collecting_particles"
        self.max_fishing_time = 200 # try for at most ca 3 minutes with a single particle
        self.fishing_start_time = time()
        self.limits_found = [False,False]
        self.stretching_distance = stretching_distance
        self.min_stretch_distance = min_stretch_distance
        self.stretch_force = stretch_force
        self.experiment_start_time = time()
        self.stretch_speed = stretch_speed
 

    def start(self):
        self.running = True
        self.stage= 'collecting_particles' # Starting step and also default
    
    def stop(self):
        self.running = False
        self.stage = 'collecting_particles' # Starting step
    
    def is_running(self):
        return self.running

    def reset_experiment(self):
        self.stage = "collecting_particles"
        self.terminate_pulling_protocol()
         # Call autocontroller to restart experiment by flushing chamber
        self.auto_controller.restart_experiment()


    def check_DNA_present(self, force_limit=50, min_dist=2, max_dist=6):
        # Simple check to see if there is a molecule being stretched.
        # this will always be positive since the pipette is below the trapped particle
        dy = self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1] 
        radii_sum = (self.c_p['pipette_particle_location'][3]
                     + self.c_p['Trapped_particle_position'][3])

        # Check DNA prescence
        particle_separation = (dy-radii_sum) * self.c_p['microns_per_pix']
        force = np.abs(np.mean(self.auto_controller.data_channels['F_total_Y'].get_data(30)) )
        if (particle_separation > min_dist) and \
            (particle_separation < max_dist) and (force > force_limit):
            print(f"DNA detected {particle_separation} microns surface to surface, \
                  force of {force} pN")
            return True
        return False


    def check_molecule_broken(self, molecule_length=5, force_limit=20):
        """
        Checks if a molecule used for stretching has been broken.
        Does so by checking if the force is small at a large particle-particle separation.
        Returns True if the molecule is broken, otherwise false.
        """
        force = np.abs(np.mean(self.auto_controller.data_channels['F_total_Y'].get_data(30)))
        dy = self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1] 
        radii_sum = (self.c_p['pipette_particle_location'][3] 
                     + self.c_p['Trapped_particle_position'][3])
        separation = (dy - radii_sum) * self.c_p['microns_per_pix']
        return force < force_limit and separation > molecule_length

    def compensate_DNA_x_position(self, x_force):
        """
        Compensates for any x-misalignment between the particles when a DNA molecule is attached.
        """

        x_dist = int(x_force*400)
        if self.c_p['portenta_command_2'] == 1: # A autoaligning
            target_x = int(self.c_p['piezo_B'][0] - x_dist)
            if target_x < 2000 or target_x > 63000:
                # Something went wrong, we cannot compensate
                print("Cannot compensate x-position, target out of range")
                return
            self.c_p['piezo_B'][0] = target_x
        else:
            target_x = int(self.c_p['piezo_A'][0] + x_dist)
            if target_x < 2000 or target_x > 63000:
                # Something went wrong, we cannot compensate
                print("Cannot compensate x-position, target out of range")
                return
            self.c_p['piezo_A'][0] = target_x
        return False

    def compensate_DNA_z_position(self,dz):
        """
        Compensates for any z-misalignment between the particles when a DNA molecule is attached.
        """
        if dz > 0:
            self.c_p['minitweezers_target_pos'][2] = int(
                self.auto_controller.motor_controller.get_z_position() + 4)
        if dz < 0:
            self.c_p['minitweezers_target_pos'][2] = int(
                self.auto_controller.motor_controller.get_z_position() - 4)        
        self.c_p['minitweezers_target_pos'][0] = int(
            self.auto_controller.motor_controller.get_x_position())
        self.c_p['minitweezers_target_pos'][1] = int(
            self.auto_controller.motor_controller.get_y_position())

        self.c_p['move_to_location'] = True
        print("Compensating z-position")

    def compensate_x_z_DNA_molecule(self,  force_limit=5, position_limit=0.85, max_dist=6):
        """
        Compensats for deviaitons in x or z position when pulling a DNA molecule and this has
        not attached right at the top of the SA bead.
        
        """
        # Move in y until we get a positive force or the particles are touching.  
        z_dev = np.mean(
            self.auto_controller.data_channels['trapped_particle_z_position'].get_data(5))
        print(f"z_displacement{z_dev}")
        if np.abs(z_dev) > 0.4:
            self.compensate_DNA_z_position(z_dev) # Not doing anything here for now.
            return False

        x_force = np.mean(self.auto_controller.data_channels['trapped_x_force'].get_data(2))
        if np.abs(x_force) > 1:
            self.compensate_DNA_x_position(x_force)
            return False # DNA present, but  Compensating x-position
        return True # DNA present and aligned

    def find_pulling_limits(self):
        """
        Locates appropriate pulling limits for the DNA pulling protocol. These are based on the
        stretch_distance and the stretch_force set in the initation of the protocol.
        """
        
        dy = self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1] 
        radii_sum = (self.c_p['pipette_particle_location'][3]
                     + self.c_p['Trapped_particle_position'][3])
        y_force = np.abs(np.mean(self.auto_controller.data_channels['F_total_Y'].get_data(20)))

        # Check if upper limit of protocol is found, if not move up
        if not self.limits_found[0]:
            # moving up to stretch the molecule
            self.auto_controller.move_up_and_down(max_dist=self.stretching_distance * 1.7,
                                    force_limit_lower=1000,
                                    # Max force is the maximum force the trap can excert.
                                    force_limit_upper=self.stretch_force, 
                                    position_limit=1,
                                    direction='up')

            force_distance = (y_force >= self.stretch_force
                                and dy-radii_sum > (self.min_stretch_distance
                                                /self.c_p['microns_per_pix']))

            if force_distance or self.c_p['piezo_B'][1] < 1200 or self.c_p['piezo_A'][1] < 1200:
                self.limits_found[0] = True
                if self.c_p['portenta_command_2'] == 1:
                    self.c_p['protocol_limits_dac'][0] = self.c_p['piezo_B'][1]
                else:
                    self.c_p['protocol_limits_dac'][0] = self.c_p['piezo_A'][1]

            elif y_force > self.stretch_force:
                #
                print("Particle stuck or multiple molecules")
                self.reset_experiment()
                return "Force too high too soon"

        if not self.limits_found[1] and self.limits_found[0]:

            # Pushing the particles down
            # how close in terms of radiis the particles are allowed to get to each other.
            position_limit = 1.2 
            self.auto_controller.move_up_and_down(max_dist=self.stretching_distance,
                                    force_limit_lower=1000,
                                    force_limit_upper=self.stretch_force,
                                    position_limit=position_limit,
                                    direction='down')
            
            if (dy < radii_sum*position_limit or self.c_p['piezo_B'][1]>63800 or 
                self.c_p['piezo_A'][1]>63800):
                self.limits_found[1] = True
                if self.c_p['portenta_command_2'] == 1:
                    self.c_p['protocol_limits_dac'][1] = self.c_p['piezo_B'][1]
                else:
                    self.c_p['protocol_limits_dac'][1] = self.c_p['piezo_A'][1]
            elif y_force > 100:
                    #
                    print("Particle stuck or multiple molecules")
                    self.limits_found = [False,False]
                    self.restart_experiment()
                    return "Force too high too soon"

    def start_pulling_protocol(self):
        """
        Initiates the pulling protocol to run on the electronics controller for smooth operation.
        """
        self.protocol_started = True
        # Set the protocol
        split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF] 
        self.c_p['protocol_data'][1:3] = split_16_bit(int(self.c_p['protocol_limits_dac'][1]))
        self.c_p['protocol_data'][3:5] = split_16_bit(int(self.c_p['protocol_limits_dac'][0]))
        self.c_p['protocol_data'][5:7] = split_16_bit(int(self.stretch_speed))
        if self.c_p['portenta_command_2'] == 1:
            self.c_p['protocol_data'][0] = 4 # Protocol activates Ay - B autoaligning
        else:
            self.c_p['protocol_data'][0] = 2 # Protocol activates By        
        if not self.auto_controller.data_saver.is_saving():
            self.auto_controller.main_window.record_data()

    def terminate_pulling_protocol(self):
        """
        Safely terminates the pulling protocol by turning it off on the microcontroller.
        """
        self.limits_found = [False,False]
        self.c_p['protocol_data'][0] = 0

        if self.auto_controller.data_saver.is_saving():
            # We call the main_window function to ensure that the saving is indicated in the 
            # main window too.
            self.auto_controller.main_window.record_data()

    def run(self):
        """
        Run will first collect 2 particles, placing one of them in the trap. This is a quite
        standard procedure (used by all single molecule protocols and most particle ineteraction
        protocols).
        It is therefore handled largelly by the autocontroller directly and the rest of the
        procedures is more tailored to the 
        """
        if not (self.auto_controller.particle_trapped and 
                self.auto_controller.particle_in_pipette and 
                self.auto_controller.check_particles_can_touch_with_piezo()):
            # Terminate the protocol (does nothing if it is not running)
            self.terminate_pulling_protocol()

            # Set the defualt stage
            self.stage = "collecting_particles"
            # Collect two particles and aling them
            self.auto_controller.collect_and_align_2_particles() 
            return self.stage

        # If we just collected two particles, then the next step is DNA fishing
        if self.stage == "collecting_particles":
            self.stage = 'fishing_DNA'
            # Set start time of fishing
            self.fishing_start_time = time()
 
        if self.stage == 'fishing_DNA':
            if time()-self.fishing_start_time > self.max_fishing_time:
                self.reset_experiment()

            # Check if we have finished the measurement, if so we should restart the experiment.            
            self.auto_controller.touch_trapped_2_pieptte_particle()
            
            # If there is DNA present then we move on to alignment
            if self.check_DNA_present():
                if self.compensate_x_z_DNA_molecule():
                    # Molecule attached and aligned, can go on to create the protocol
                    self.stage = 'creating_protocol'
                else:
                    # This is a small sub-step of fishing
                    return "Aligning beads with DNA"

        if self.stage == 'creating_protocol':
            if not (self.limits_found[0] and self.limits_found[0]):
                self.find_pulling_limits()
            else:
                self.stage = "running_experiment"
                self.experiment_start_time = time()
                self.start_pulling_protocol()

        if self.stage == "running_experiment":
            
            # Check if the molecule broke, if that is the case go back to fishing DNA
            if self.check_molecule_broken():
                self.terminate_pulling_protocol()
                self.stage = 'fishing_DNA'

            # If the maximum experiment time has expired, then reset the experiment
             # Same max time for all experiments can be set in the main window
            if time() - self.experiment_start_time > self.c_p['measurement_time']:                
                self.reset_experiment()
        
        # Return self.stage to let the caller know what is happening now
        return self.stage
        

class ElectrostaticRepulsion(AutonomousProtocol):
    """
    Autonomous protocol which measures repulsive forces between two charged particles.
    The protocol first collects two particles, one is trapped and the other is placed in the pipette.
    Then it aligns the particles in x and z and finds the movement limits.
    Finally it starts the repulsion protocol.
    """

    def __init__(self, c_p, auto_controller, max_force=5, movement_speed=5):
        self.c_p = c_p
        self.auto_controller = auto_controller
        self.running = False
        self.stage = "collecting_particles"
        self.limits_found = [False,False]
        self.max_force = max_force
        self.movement_speed = movement_speed
        self.experiment_start_time = time()
        self.max_separation = 0.5 # Maximum separation between the particles during experiment
        # microns, surface to surface
        

    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True
        self.stage= 'collecting_particles' # Starting step and also default

    def stop(self) -> None:
        self.running = False
        self.stage = 'collecting_particles' # Starting step

    def reset_experiment(self) -> None:
        self.stage = "collecting_particles"
        self.terminate_protocol()
        self.auto_controller.restart_experiment()
    
    def align_particles(self):
        """
        Aligns the particle in the optical trap to the one in the pipette. Does so with greater
        care than what is needed for single molecule experiments.        
        """

        # Move in y until we get a positive force or the particles are touching.  
        # this will always be positive since the pipette is below the trapped particle
        dy = self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1] 
        radii_sum = (self.c_p['pipette_particle_location'][3] 
                     + self.c_p['Trapped_particle_position'][3])
 
        # Check z-alignment, changed to an average to get a more reliable reading.
        z_bead = np.mean(
            self.auto_controller.data_channels['trapped_particle_z_position'].get_data(10))
        z_pipette = np.mean(
            self.auto_controller.data_channels['pipette_particle_z_position'].get_data(10))
        dz = (z_bead - z_pipette)
        # Fix z-alginment if the particles are not touching and are missaligned in z
        if np.abs(dz) > 0.5:
            if dy > (radii_sum*1+0.5/self.c_p['microns_per_pix']):
                self.c_p['focus_z_trap_pipette'] = True
                return False # need aligning in z

            else:
                self.auto_controller.move_up_and_down(max_dist=5,
                                                      force_limit_lower=self.max_force,
                                                      force_limit_upper=self.max_force,
                                                      position_limit=1)
                return False # move away to align in z
        self.c_p['focus_z_trap_pipette'] = False

        # Align in x
        if self.auto_controller.align_trapped_and_pipette_X():
            # Aligning in x
            return False
        return True

    def initiate_movement_protocol(self):
        split_16_bit = lambda num: [(num >> 8) & 0xFF, num & 0xFF] 
        self.c_p['protocol_data'][1:3] = split_16_bit(int(self.c_p['protocol_limits_dac'][1]))
        self.c_p['protocol_data'][3:5] = split_16_bit(int(self.c_p['protocol_limits_dac'][0]))
        self.c_p['protocol_data'][5:7] = split_16_bit(int(self.movement_speed))
        if self.c_p['portenta_command_2'] == 1:
            self.c_p['protocol_data'][0] = 4 # Protocol activates Ay - B autoaligning
        else:
            self.c_p['protocol_data'][0] = 2 # Protocol activates By
        self.experiment_start_time = time()

        if not self.c_p['recording']:
            self.auto_controller.zoom_2_particles()
            sleep(0.5)
            self.auto_controller.main_window.toggle_recording()
        if not self.auto_controller.data_saver.is_saving():
            self.auto_controller.main_window.record_data()

    def find_protocol_limits(self):
        """
        Locates appropriate positions for the movement protocol of the experiment (run on the
        electronics controller for smooth operation). The limits are based on the max_force set in 
        the initiation of the protocol.
        """

        # Find limit on moving down first, then on moving up.
        dy = (self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1]) \
            * self.c_p['microns_per_pix']
        radii_sum = (self.c_p['pipette_particle_location'][3] 
                     + self.c_p['Trapped_particle_position'][3]) * self.c_p['microns_per_pix']
                    
        if not self.limits_found[1] :
            self.moving_down = True
            position_limit = 0.8
            self.auto_controller.move_up_and_down(
                max_dist=2,
                force_limit_lower=self.max_force,
                force_limit_upper=self.max_force,
                position_limit=position_limit
                )
            y_force = np.abs(np.mean(self.auto_controller.data_channels['F_total_Y'].get_data(200)))
            if (dy < radii_sum*position_limit 
                or np.abs(y_force) > self.max_force
                or  self.c_p['piezo_B'][1]>63800
                or self.c_p['piezo_A'][1]>63800):

                self.limits_found[1] = True
                if self.c_p['portenta_command_2'] == 1:
                    self.c_p['protocol_limits_dac'][1] = self.c_p['piezo_B'][1]
                else:
                    self.c_p['protocol_limits_dac'][1] = self.c_p['piezo_A'][1]
            

        if not self.limits_found[0] and self.limits_found[1]:
            self.moving_down = False
            self.auto_controller.move_up_and_down(
                max_dist=2,
                force_limit_lower=self.max_force,
                force_limit_upper=100,
                position_limit=1)

            if (dy - radii_sum >= self.max_separation
                or self.c_p['piezo_B'][1]<1200
                or self.c_p['piezo_A'][1]<1200):
                self.limits_found[0] = True
                if self.c_p['portenta_command_2'] == 1:
                    self.c_p['protocol_limits_dac'][0] = self.c_p['piezo_B'][1]
                else:
                    self.c_p['protocol_limits_dac'][0] = self.c_p['piezo_A'][1]
            y_force = np.mean(self.auto_controller.data_channels['F_total_Y'].get_data(1000))
            
            if np.abs(y_force) > 70:
                # Restart experiment, particles have gotten stuck together.
                self.terminate_protocol(True)
                self.restart_experiment()
                return False
        return True
    
    def terminate_protocol(self):
        """
        Safely stop the protocol running on the microcontroller and terminate the saving of data.
        """
        self.limits_found = [False,False]
        self.c_p['protocol_data'][0] = 0
        if self.auto_controller.data_saver.is_saving():
            # We call the main_window function to ensure that the saving is indicated in the main window too.
            self.auto_controller.main_window.record_data()

    def run(self) -> str:
        """
        Full autonomous procedure for electrostatic measurements.
        """

        if not (self.auto_controller.particle_trapped 
                and self.auto_controller.particle_in_pipette
                and self.auto_controller.check_particles_can_touch_with_piezo()):
            # Terminate the protocol (does nothing if it is not running)
            self.terminate_protocol()

            # Set the defualt stage
            self.stage = "collecting_particles"
            # Collect two particles and aling them
            self.auto_controller.collect_and_align_2_particles() 
            return self.stage

        if self.stage == "collecting_particles":
            self.stage = "aliging_particles"            

        if self.stage == "aliging_particles":
            if self.align_particles():
                self.stage = "finding_protocol_limits"
            else:
                return self.stage

        if self.stage == "finding_protocol_limits":
            if self.find_protocol_limits():
                self.initiate_movement_protocol()
                self.stage = "executing_protocol"
                self.experiment_start_time = time()
            return self.stage

        if self.stage == "executing_protocol":
            if time()-self.experiment_start_time > self.c_p['measurement_time']:
                self.reset_experiment()
        
        return self.stage
        

class ParticleCharacterization(AutonomousProtocol):
    """
    Performs a particle characterization experiment. Selects a large particle, 
    traps it and moves it to a target position. Then measures the hydrodynamic drag by moving
    position of the particle.
    """

    def __init__(self, c_p, auto_controller, size_threshold=3):
        self.c_p = c_p
        self.auto_controller = auto_controller
        self.running = False
        self.stage = "collecting_particles"
        self.size_threshold = size_threshold

    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True
        self.stage= 'collecting_particles' # Starting step and also default

    def stop(self) -> None:
        self.running = False
        self.stage = 'collecting_particles' # Starting step

    def reset_experiment(self) -> None:
        self.stage = "collecting_particles"
        self.auto_controller.restart_experiment()
        self.c_p['stokes_exp'] = "startup"
        self.c_p['stokes_stage'] = "stokes_startup"

    def run(self):

        if not self.auto_controller.particle_trapped: 
            self.c_p['move_piezo_2_target'] = True
            self.c_p['particle_type'] = 1
            self.c_p['search_and_trap'] = True
            self.c_p['stokes_stage'] = "stokes_startup"
            self.c_p['stokes_test_running'] = False

            if self.auto_controller.data_saver.is_saving():
                self.auto_controller.main_window.record_data()
                self.c_p['stokes_test_running'] = False
            return "Collecting particle"

        if self.auto_controller.data_channels['particle_trapped'].get_data(1)[0] == 0:
            return "Checking particle"

        # Idea here, only measure particles above a certain size threshold, e.g 3 microns in diameter.
        radii = self.auto_controller.data_channels['trapped_particle_radii'].get_data(10)
        if (np.mean(radii[radii>0]) < self.c_p['stokes_size_threshold'] 
            or self.c_p['multiple_particles_trapped']): 
            # Note that YOLO has a tendency to oversetimate the radii.
            if np.mean(radii[radii>0]) < self.c_p['stokes_size_threshold']:
                self.auto_controller.data_saver.snapshot(
                    filename_save=self.c_p['filename'] + "_small_particle"
                    )                
                print("Small particle trapped, restarting experiment")                
            else:                
                print("Many particles trapped, restarting experiment")            
            self.reset_experiment()
            return "resetting experiment"
        print("Large particle trapped")

        # A particle is trapped the next step is to go to the target start position.
        if self.c_p['stokes_stage'] == 'stokes_startup':
            distance = self.calc_distance(self.get_position(), self.c_p['stokes_left_pos'])

            if distance > 100:
                # Far from starting position, move to start position
                if not self.auto_controller.motor_controller.is_moving():
                    self.auto_controller.motor_controller.move_to_location(
                        self.c_p['stokes_center_pos'], self.c_p)
                    return
                else:
                    # we are moving to start position
                    return

            # Close to starting position. we can start the stokes test
            self.c_p['stokes_test_running'] = True
            # Autoalign and wait a little
            print("Autoaligning before stokes test")
            self.c_p['portenta_command_2'] = 1
            sleep(1)
            self.c_p['portenta_command_2'] = 0

            self.c_p['stokes_stage'] = "performing_test"


        # We are ready to start the stokes test
        if self.c_p['stokes_stage'] == "performing_test":

            # Check that data is being recorded
            if not self.data_saver.is_saving():
                self.data_saver.snapshot()
                self.main_window.record_data()
            
            if self.c_p['stokes_test_running']:
                # Test is underway and a particle is trapped, no need to do anything here
                return "Stokes test running"
            else:
                # If the test is finished terminate
                self.c_p['stokes_stage'] = "stokes_startup"

        
        # Since we have gotten down here the stokes test is finished.
        # Stop saving data and restart the experiment
        if self.data_saver.is_saving():
                self.main_window.record_data()
        self.restart_experiment()
        self.c_p['stokes_exp'] = "startup"
        return "Experiment restarted"


class RBCStretching(AutonomousProtocol):
    """
    Autonomous red blood cell stretching experiment. Stretches a red blood cell by changing the
    trapping power. Records a video of the process.
    """

    def __init__(self, c_p, auto_controller):
        self.c_p = c_p
        self.auto_controller = auto_controller
        self.running = False
        self.stage = "collecting_particles"
        self.RBC_timer = 0
        self.RBC_timeout = 5
        self.RBC_protocol_started = False
        self.RBC_protocol_started = False

    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True
        self.stage= 'collecting_particles' # Starting step and also default

    def stop(self) -> None:
        self.running = False
        self.stage = 'collecting_particles' # Starting step

    def reset_experiment(self) -> None:
        self.stage = "collecting_particles"
        self.auto_controller.restart_experiment()

    def run(self):
        """
        Autonomous red blood cell experiments
        By looking at the image, if there are no particles (RBC) then it will flow some particles
        and try again. Once there is a particle in view it will trap that and start the experiment
        changing the current according to the power_protocol_currents protocol.
        After the protocol is finished it will restart the experiment and try again.        
        """
        if not self.particle_trapped:
            self.c_p['move_piezo_2_target'] = True
            self.c_p['portenta_command_2'] = 0
            self.c_p['centering_on'] = True
            self.RBC_protocol_started = False
            self.c_p['laser_power_protocol_running'] = False
            if self.auto_controller.particles_in_view:
                self.RBC_timer = time()
            elif time() - self.RBC_timer > self.RBC_timeout:
                self.restart_experiment() # Flush more particles into the mix
                self.RBC_timer = time()

                # Move back to a safe position if need be.
                x = self.motor_controller.get_x_position()
                y = self.motor_controller.get_y_position()
                if ((x-self.c_p['pipette_location_chamber'][0])**2 + 
                    (y-self.c_p['pipette_location_chamber'][1])**2 > 500):
                    if not self.auto_controller.motor_controller.is_moving():
                        self.auto_controller.motor_controller.move_to_location(
                            self.c_p['pipette_location_chamber'])
                        print(f"Moving to pipette location {self.c_p['pipette_location_chamber']}")

        if self.auto_controller.particle_trapped and not self.RBC_protocol_started:
            self.c_p['centering_on'] = False
            self.c_p['search_and_trap'] = False
            self.c_p['laser_power_protocol_running'] = True
            self.RBC_protocol_started = True
        
        if self.RBC_protocol_started and not self.c_p['laser_power_protocol_running']:
            self.auto_controller.restart_experiment()
            self.RBC_protocol_started = False

class AutoControlWidget(QWidget):
    """
    A QWidget that provides an interface for controlling various automated functions
    related to particle manipulation and experiments.
    Parameters:
    - c_p: A dictionary containing configuration parameters.
    - data_channels: A dictionary of data channels for real-time data access.
    - motor_controller: An instance of a motor controller for piezo and stage movements.
    Methods:
    - init_ui: Initializes the user interface components.
    - refresh: Updates the UI based on the current state of the system.
    - center_particle: Toggles centering on a particle.
    - trap_particle: Toggles trapping of a particle.
    - drop_particles: Drops trapped particles.
    - search_and_trap: Toggles searching and trapping of particles.
    - toggle_z_focus_pipette_trapped: Toggles z-focus between trap and pipette.
    - move_piezo_2_saved_positions: Moves piezo to saved positions.
    - touch_particles: Initiates touching of particles.
    - stretch_molecule: Initiates stretching of a molecule.
    - toggle_autonomous_experiment: Toggles autonomous experiment mode.
    - move2pipette_tip: Moves the trapped particle to the pipette tip.
    - move2area_above: Moves the trapped particle to an area above the pipette tip.
    - suck_into_pipette: Initiates sucking the particle into the pipette.
    - toggle_pipette_focus: Toggles focus on the pipette.
    - external_restart: Forces an external restart of the system.
    - auto_calibrate: Starts or stops the auto-calibration process.
    - load_auto_calibration: Loads a saved auto-calibration file.
    - create_toggle_button: Helper function to create toggle buttons.
    - create_spin_box: Helper function to create spin boxes.
    - create_double_spin_box: Helper function to create double spin boxes.
    - create_label: Helper function to create labels.
    - create_section_label: Helper function to create section labels.
    - calculate_polynomial_coefficients: Calculates polynomial coefficients for calibration.    
    """
    def __init__(self, c_p, data_channels, motor_controller):
        super().__init__()
        self.c_p = c_p
        self.data_channels = data_channels
        self.motor_controller = motor_controller
        self.init_ui()

        # Initiate a timer to alert the user in case one of the protocols finishes.
        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def init_ui(self):
        """
        Initializes the user interface components.
        """
        layout = QVBoxLayout()
        self.setWindowTitle("Auto Controller")

        # Toggle Buttons Section
        self.create_toggle_buttons(layout)

        # Search and Trap Section
        self.create_search_and_trap_section(layout)

        # Capillary Controls Section
        self.create_capillary_controls(layout)

        # Pipette Controls Section
        self.create_pipette_controls(layout)

        # Focus and Movement Section
        self.create_focus_and_movement_controls(layout)

        # Autonomous Experiment Section
        self.create_autonomous_experiment_controls(layout)

        self.setLayout(layout)

    def create_toggle_buttons(self, layout):
        self.toggle_centering_button = self.create_toggle_button(
            'Centering on', self.center_particle, self.c_p['centering_on'])
        layout.addWidget(self.toggle_centering_button)

        self.toggle_trap_button = self.create_toggle_button(
            'Trap particle', self.trap_particle, self.c_p['trap_particle'])
        layout.addWidget(self.toggle_trap_button)

    def center_particle(self):
        """
        Function that seeks to keep a particle in the center 
        of the image. 
        """
        self.c_p['centering_on'] = not self.c_p['centering_on']

    def trap_particle(self):
        self.c_p['trap_particle'] = not self.c_p['trap_particle']
    
    def drop_particles(self):
        """
        Moves the piezos to drop the trapped particle
        """
        self.c_p['drop_particle'] = not self.drop_particles_button.isChecked()

    def search_and_trap(self):
        """
        Toggles searching and also trapping of particle
        """
        self.c_p['search_and_trap'] = not self.c_p['search_and_trap']

    def toggle_z_focus_pipette_trapped(self):
        self.c_p['focus_z_trap_pipette'] = not self.z_focus_button.isChecked()
    
    def move_piezo_2_saved_positions(self):
        self.c_p['move_piezo_2_target'] = not self.move_piezo_2_saved_positions_button.isChecked()

    def touch_particles(self):
        self.c_p['touch_particles'] = not self.touch_particles_button.isChecked()
        self.c_p['touch_counter'] = 0
    
    def stretch_molecule(self):
        self.c_p['stretch_molecule'] = not self.stretch_molecule_button.isChecked()

    def toggle_autonomous_experiment(self):
        self.c_p['autonomous_experiment'] = not self.autonomous_button.isChecked()

    def move2pipette_tip(self):
        self.c_p['move_particle2pipette'] = not self.move_particle2pipette_button.isChecked()

    def move2area_above(self):
        self.c_p['move_to_pipette_tip'] = not self.move2area_above_button.isChecked()

    def suck_into_pipette(self):
        self.c_p['suck_into_pipette'] = not self.suck_into_pipette_button.isChecked()

    def toggle_pipette_focus(self):
        self.c_p['focus_pipette'] = not self.pipette_focus_button.isChecked()
        if self.c_p['focus_pipette']:
            self.c_p['pipette_focus_startup'] = True
            self.c_p['pipette_sharpnesses'] = []
            self.c_p['pipette_sharpness_Z_pos'] = []
    
    def external_restart(self):
        # Forces a restart
        self.c_p['external_restart'] = True

    def auto_calibrate(self):
        self.c_p['calibration_running'] = not self.c_p['calibration_running']
        if self.c_p['calibration_running']:
            self.c_p['calibration_start'] = True

    def load_auto_calibration(self):
        calibration_file_path = QFileDialog.getOpenFileName(
            self, "Load Auto Calibration", "", "Calibration Files (*.npy)")[0]
        if calibration_file_path:
            try:
                calibration = np.load(calibration_file_path, allow_pickle=True)
                if np.shape(calibration) == np.shape(self.c_p['calibration_points']):
                    self.c_p['calibration_points'] = calibration
                    print("Auto calibration loaded successfully.")
                    self.c_p['calibration_performed'] = True
                    self.APX_coeffs = self.calculate_polynomial_coefficients(
                        self.c_p['calibration_points'][:,:,2],
                        self.c_p['calibration_points'][:,:,3],
                        self.c_p['calibration_points'][:,:,0])
                    self.APY_coeffs = self.calculate_polynomial_coefficients(
                        self.c_p['calibration_points'][:,:,2],
                        self.c_p['calibration_points'][:,:,3],
                        self.c_p['calibration_points'][:,:,1])
                    self.BPX_coeffs = self.calculate_polynomial_coefficients(
                        self.c_p['calibration_points'][:,:,4],
                        self.c_p['calibration_points'][:,:,5],
                        self.c_p['calibration_points'][:,:,0])
                    self.BPY_coeffs = self.calculate_polynomial_coefficients(
                        self.c_p['calibration_points'][:,:,4],
                        self.c_p['calibration_points'][:,:,5],
                        self.c_p['calibration_points'][:,:,1])
                else:
                    print("Calibration file shape does not match current calibration points.")
            except Exception as e:
                print(f"Error loading auto calibration: {e}")

    def create_search_and_trap_section(self, layout):
        search_and_trap_layout = QHBoxLayout()

        self.search_and_trap_button = self.create_toggle_button('Search and trap',
                                                                self.search_and_trap,
                                                                self.c_p['search_and_trap'])
        search_and_trap_layout.addWidget(self.search_and_trap_button)

        self.capillary_selection = QComboBox()
        self.capillary_selection.addItems(["Capillary 1", "Capillary 2"])
        self.capillary_selection.setCurrentIndex(0)
        self.capillary_selection.currentIndexChanged.connect(self.capillary_selection_changed)
        search_and_trap_layout.addWidget(self.capillary_selection)

        layout.addLayout(search_and_trap_layout)

    def create_capillary_controls(self, layout):
        capillary_1_layout = self.create_capillary_section('Capillary 1',
                                                           self.set_capillary_1_position,
                                                           self.goto_capillary_1)
        layout.addLayout(capillary_1_layout)

        capillary_2_layout = self.create_capillary_section('Capillary 2',
                                                           self.set_capillary_2_position,
                                                           self.goto_capillary_2)
        layout.addLayout(capillary_2_layout)

    def create_pipette_controls(self, layout):
        pipette_layout = QHBoxLayout()

        self.set_pipette_position_button = QPushButton('Set pipette position')
        self.set_pipette_position_button.pressed.connect(self.set_pipette_position)
        self.set_pipette_position_button.setToolTip("""Sets the approximate location of the 
                                                    pipette to the current motor position.""")
        pipette_layout.addWidget(self.set_pipette_position_button)

        self.set_go2pipette_button = QPushButton('Go to pipette')
        self.set_go2pipette_button.pressed.connect(self.goto_pipette)
        self.set_go2pipette_button.setToolTip("Moves the pipette to the pipette position.")
        pipette_layout.addWidget(self.set_go2pipette_button)

        self.center_pipette_button = QPushButton('Center pipette')
        self.center_pipette_button.pressed.connect(self.toggle_center_pipette)
        self.center_pipette_button.setCheckable(True)
        self.center_pipette_button.setChecked(self.c_p['center_pipette'])
        layout.addWidget(self.center_pipette_button)

        layout.addLayout(pipette_layout)

    def create_focus_and_movement_controls(self, layout):
        self.z_focus_button = self.create_toggle_button('Toggle z focus',
                                                        self.toggle_z_focus_pipette_trapped,
                                                        self.c_p['focus_z_trap_pipette'])
        layout.addWidget(self.z_focus_button)

        self.pipette_focus_button = self.create_toggle_button('Toggle pipette focus',
                                                              self.toggle_pipette_focus,
                                                              self.c_p['focus_pipette'])
        layout.addWidget(self.pipette_focus_button)

        self.move2area_above_button = self.create_toggle_button('Move to area above pipette',
                                                                self.move2area_above,
                                                                self.c_p['move_to_pipette_tip'])
        layout.addWidget(self.move2area_above_button)

        self.move_particle2pipette_button = self.create_toggle_button(
            'Move particle to pipette',
            self.move2pipette_tip,
            self.c_p['move_particle2pipette'])
        layout.addWidget(self.move_particle2pipette_button)

        self.suck_into_pipette_button = self.create_toggle_button('Suck into pipette',
                                                                  self.suck_into_pipette,
                                                                  self.c_p['suck_into_pipette'])
        layout.addWidget(self.suck_into_pipette_button)

        self.move_piezo_2_saved_positions_button = self.create_toggle_button(
            'Move piezo to zero positions',
            self.move_piezo_2_saved_positions,
            self.c_p['move_piezo_2_target'])
        layout.addWidget(self.move_piezo_2_saved_positions_button)

        self.touch_particles_button = self.create_toggle_button('Touch particles',
                                                                self.touch_particles,
                                                                self.c_p['touch_particles'])
        layout.addWidget(self.touch_particles_button)

        self.stretch_molecule_button = self.create_toggle_button('Stretch molecule',
                                                                 self.stretch_molecule,
                                                                 self.c_p['stretch_molecule'])
        layout.addWidget(self.stretch_molecule_button)

    def create_autonomous_experiment_controls(self, layout):
        self.autonomous_button = self.create_toggle_button('Autonomous experiment',
                                                           self.toggle_autonomous_experiment,
                                                           self.c_p['autonomous_experiment'])
        layout.addWidget(self.autonomous_button)

        self.autonomous_experiment_type_box = QComboBox()
        self.autonomous_experiment_type_box.addItems(self.c_p['autonomous_experiment_types'])
        self.autonomous_experiment_type_box.setCurrentIndex(0)
        self.autonomous_experiment_type_box.currentIndexChanged.connect(
                self.autonomous_experiment_type_changed)
        layout.addWidget(self.autonomous_experiment_type_box)

        self.drop_particles_button = self.create_toggle_button(
            'Drop particles', self.drop_particles, self.c_p['drop_particle'])
        layout.addWidget(self.drop_particles_button)

        self.auto_calibrate_button = self.create_toggle_button(
            'Auto calibrate', self.auto_calibrate, self.c_p['calibration_running'])
        layout.addWidget(self.auto_calibrate_button)

        self.load_calibration_button = QPushButton('Load position calibration')
        self.load_calibration_button.pressed.connect(self.load_auto_calibration)
        self.load_calibration_button.setToolTip(
            "Loads the auto calibration file.\n Needs to be the same shape as current calibration")
        layout.addWidget(self.load_calibration_button)
        self.restart_button  = QPushButton('Restart experiment')
        self.restart_button.pressed.connect(self.external_restart)
        layout.addWidget(self.restart_button)

    def create_toggle_button(self, text, slot_function, initial_state):
        button = QPushButton(text)
        button.pressed.connect(slot_function)
        button.setCheckable(True)
        button.setChecked(initial_state)
        return button

    def create_capillary_section(self, capillary_name, set_function, goto_function):
        layout = QHBoxLayout()

        set_button = QPushButton(f'Set {capillary_name} position')
        set_button.pressed.connect(set_function)
        set_button.setToolTip(
            f"Sets the approximate location of {capillary_name.lower()} to the current motor \
              position.")
        layout.addWidget(set_button)

        goto_button = QPushButton(f'Go to {capillary_name}')
        goto_button.pressed.connect(goto_function)
        goto_button.setToolTip(f"Moves the pipette to the {capillary_name.lower()} position.")
        layout.addWidget(goto_button)

        return layout

    def refresh(self):
        """
        Updates the various toggle buttons to reflect which routines are being executed, e.g. 
        "searching for a particle".
        """
        self.z_focus_button.setChecked(self.c_p['focus_z_trap_pipette'])
        self.pipette_focus_button.setChecked(self.c_p['focus_pipette'])
        self.move2area_above_button.setChecked(self.c_p['move_to_pipette_tip'])
        self.move_particle2pipette_button.setChecked(self.c_p['move_particle2pipette'])
        self.toggle_centering_button.setChecked(self.c_p['centering_on'])
        self.toggle_trap_button.setChecked(self.c_p['trap_particle'])
        self.search_and_trap_button.setChecked(self.c_p['search_and_trap'])
        self.center_pipette_button.setChecked(self.c_p['center_pipette'])

        self.move_piezo_2_saved_positions_button.setChecked(self.c_p['move_piezo_2_target'])
        self.touch_particles_button.setChecked(self.c_p['touch_particles'])
        self.stretch_molecule_button.setChecked(self.c_p['stretch_molecule'])
        self.drop_particles_button.setChecked(self.c_p['drop_particle'])
        self.autonomous_button.setChecked(self.c_p['autonomous_experiment'])
        self.suck_into_pipette_button.setChecked(self.c_p['suck_into_pipette'])
        self.auto_calibrate_button.setChecked(self.c_p['calibration_running'])

    def set_capillary_1_position(self):
        self.set_position('capillary_1_position')

    def set_capillary_2_position(self):
        self.set_position('capillary_2_position')

    def set_pipette_position(self):
        self.set_position('pipette_location_chamber')

    def set_position(self, key):
        x = int(self.motor_controller.get_x_position())
        y = int(self.motor_controller.get_y_position())
        z = int(self.motor_controller.get_z_position())
        self.c_p[key] = [x, y, z]

    def goto_capillary_1(self):
        self.motor_controller.move_to_location(self.c_p['capillary_1_position'])

    def goto_capillary_2(self):
        self.motor_controller.move_to_location(self.c_p['capillary_2_position'])

    def goto_pipette(self):
        self.motor_controller.move_to_location(self.c_p['pipette_location_chamber'])

    def capillary_selection_changed(self):
        self.c_p['particle_type'] = self.capillary_selection.currentIndex() + 1
        print(f"Particle type set to {self.c_p['particle_type']}")

    def autonomous_experiment_stage_changed(self):
        state = self.autonomous_experiment_stage_box.currentText()
        self.c_p['autocontroller_current_step'] = state
        print(f"Autonomous experiment stage set to {state}")

    def autonomous_experiment_type_changed(self):
        state = self.autonomous_experiment_type_box.currentText()
        self.c_p['autonomous_experiment_type'] = state
        print(f"Autonomous experiment type set to {state}")

    def toggle_center_pipette(self):
        """
        Moves the stage so that the pipette is in the center of the image
        """
        self.c_p['center_pipette'] = not self.center_pipette_button.isChecked()

        # If the button was just pressed then we should find the exact pipette location.
        if self.c_p['center_pipette']:
            self.c_p['pipette_located'] = False
        else:
            
            self.c_p['move_to_location'] = False

class SelectLaserPosition(MouseTool):
    """
    Used for determining the position of the laser in the image.

    Left click for selecting position of laser A and right click for selecting the position 
    of laser B.
    """
    def __init__(self, c_p):
        self.c_p = c_p
        self.pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
        self.pen2 = QtGui.QPen(QtGui.QColor(255, 0, 0))

    def draw(self, qp):
        """
        Draws the laser positions (A and B) in the camera feed based on where they should be when
        measuring their location with the PSD detectors.
        """
        qp.setPen(self.pen)
        r = 10
        x = int((self.c_p['laser_position_A_predicted'][0] - self.c_p['AOI'][0])
                 / self.c_p['image_scale'] - r/2)
        y = int((self.c_p['laser_position_A_predicted'][1] - self.c_p['AOI'][2])
                / self.c_p['image_scale'] - r/2)
        qp.drawEllipse(x,y, r, r)

        qp.setPen(self.pen2)
        x = int((self.c_p['laser_position_B_predicted'][0] - self.c_p['AOI'][0])
                / self.c_p['image_scale'] - r/2)
        y = int((self.c_p['laser_position_B_predicted'][1] - self.c_p['AOI'][2])
                / self.c_p['image_scale'] - r/2)
        qp.drawEllipse(x,y, r, r)

    def mousePress(self):
        
        if self.c_p['mouse_params'][0] == 1:
            self.c_p['laser_position_A'] = np.array(
                self.c_p['mouse_params'][1:3]) * self.c_p['image_scale']
            self.c_p['laser_position_A'][0] += self.c_p['AOI'][0]
            self.c_p['laser_position_A'][1] += self.c_p['AOI'][2]
            print("Laser A position set to: ", self.c_p['laser_position_A'])
        elif self.c_p['mouse_params'][0] == 2:
            self.c_p['laser_position_B'] = np.array(
                self.c_p['mouse_params'][1:3]) * self.c_p['image_scale']
            self.c_p['laser_position_B'][0] += self.c_p['AOI'][0]
            self.c_p['laser_position_B'][1] += self.c_p['AOI'][2]
            print("Laser B position set to: ", self.c_p['laser_position_A'])

    def mouseRelease(self):
        pass
    def mouseDoubleClick(self):
        pass
    def mouseMove(self):
        pass
    def getToolName(self):
        return "Laser position"
    def getToolTip(self):
        return """Click on the screen where the laser is located\n Used to tell the auto controll 
        functions where to expect particles to be trappable."""

class AutoControllerThread(Thread):
    """
    Thread for running the auto controller in the background. Also handles the deep learning
    image analysis. Can run the different AutonomousProtocols. To add more Autonomous protocols 
    add them in the self.protocols in the __init__ below and in the autonomous_experiment_types
    control_parameter (just a string with the name) to have it toggleable from the control widget.
    """
    def __init__(self, c_p, data_channels, object_tracker, motor_controller,main_window, data_saver,
                 pipette_pump):
        super().__init__()

        self.c_p = c_p
        self.particles_in_view = False
        self.data_channels = data_channels
        self.object_tracker = object_tracker
        self.motor_controller = motor_controller
        self.search_direction = 1 
        self.y_lim_pos = 1 # search limits
        self.x_lim_pos = 0.1
        self.main_window = main_window
        self.data_saver = data_saver
        self.pipette_pump = pipette_pump

        self.AOI0 = self.c_p['AOI'][0]
        self.AOI1 = self.c_p['AOI'][1]
        self.AOI2 = self.c_p['AOI'][2]
        self.AOI3 = self.c_p['AOI'][3]

        # Create the protocols
        self.protocols = {
            'DNA pulling':DNAPulling(self.c_p, self),
            'Elctrostatic repulsion': ElectrostaticRepulsion(self.c_p, self),
            'Particle characterization': ParticleCharacterization(self.c_p, self),
            'RBC Stretching': RBCStretching(self.c_p, self),
        }

        # DNA attachment parameters
        self.DNA_move_direction = 0
        # Number of bits we need to change the piezo dac to move 1 micron, approximate
        self.bits_per_pixel = 500 / self.c_p['microns_per_pix']
        # approximate particle-particle-distance at which we should see a force ramp in the DNA
        
        self.last_move_time = 0

        # pipette-z focus parameters
        self.sharpnesses_mapped = False
  
        self.touch_counter_limit = 15 # should be 20
        self.pipette_view_counter = 0 # Counts how many frames the pipette has been out of view.
        self.z_move_counter = 0
        
        self.go_trap_timer = 0
        self.in_trapping_area = False
        self.trapping_counter = 0

        # Autocalibration parameters
        self.calibration_timer = 0
        self.calibration_x = 0
        self.calibration_y = 0
        # How long to wait after moving the piezos before taking recording the data.
        self.calibration_wait_time = 1        
        # Direction of next x-step, moves back and forth, "forwards" or "backwards"
        self.calibration_dir = "forwards"
        
        self.grid_step = int(60_000 / self.c_p['grid_size'])
        self.APX_coeffs = None
        self.APY_coeffs = None
        self.BPX_coeffs = None
        self.BPY_coeffs = None
        self.PSD_A_P_sum = 10_000
        self.PSD_B_P_sum = 10_000
        # Used to terminate the autocalibration if no particles are found in the image for a while.
        self.calibration_particles_counter = 0

        # Stokes test parameters
        self.move_counter = 0

        self.moving_down = True # Used to determine if we are moving up/down when touching particles
        self.protocol_started = False
        self.limits_found = [False, False]
        self.moving_to_trap = False
        self.offset_y = 140 # Offset in pixels when moving to area above pipette
        self.zoomed_in = False

    def check_multiple_trapped(self):
        """
        Check if there are multiple particles trapped in the pipette.
        """
        # This is too sensitive when pushing together.
        if not self.c_p['particle_trapped'] or not self.c_p['tracking_on']:
            self.c_p['multiple_particles_trapped'] = False
            return

        if not self.particle_trapped:
            self.c_p['multiple_particles_trapped'] = False
            return
        # Had an issue with false multi trapped detection when molecules were broken
        fx = (self.data_channels['F_total_X'].data[self.data_channels['F_total_X'].index-15_000]
              + self.data_channels['F_total_X'].get_data(1)[0])
        fy = (self.data_channels['F_total_Y'].data[self.data_channels['F_total_Y'].index-15_000]
              + self.data_channels['F_total_Y'].get_data(1)[0])

        xy_force = (fx/2)**2 + (fy/2)**2
        if xy_force > 50: # If there is a large force then this may cause a deviation in z
            self.c_p['multiple_particles_trapped'] = False
            return

        distance_trapped_pipette_particle = np.linalg.norm(np.array(
            self.c_p['pipette_particle_location'][0:2])
            - np.array(self.c_p['Trapped_particle_position'][0:2]))

        if distance_trapped_pipette_particle*self.c_p['microns_per_pix'] < 4:
            self.c_p['multiple_particles_trapped'] = False
            return
        if self.motor_controller.is_moving():
            return False
        
        z_predictions = self.data_channels['trapped_particle_z_position'].get_data(10)
        mean = np.mean(z_predictions)
        self.c_p['multiple_particles_trapped'] = np.abs(mean) > 2

    def compute_gradient_sharpness(self):
        """
        Computes the sharpness of the gradient within a specified region of interest (ROI) in the
        image.

        Returns:
            sharpness (float): The mean gradient magnitude within the ROI.
                Returns None if there is an error during computation.
        """
        
        pipette_left = int(self.c_p['pipette_location'][0] - self.c_p['pipette_location'][2]/2)
        pipette_right = int(self.c_p['pipette_location'][0] + self.c_p['pipette_location'][2]/2)
        pipette_top = int(self.c_p['pipette_location'][1])
        pipette_bottom = int(self.c_p['pipette_location'][1] + self.c_p['pipette_location'][3])

        # Compute the gradients using the Sobel operator
        try:
            image_crop = self.c_p['image'][pipette_top:pipette_bottom, pipette_left:pipette_right]
            grad_x = cv2.Sobel(image_crop, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image_crop, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute the gradient magnitude
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Compute the mean of the gradient magnitudes
            sharpness = np.mean(grad_magnitude)
            
            return sharpness
        except Exception as e:
            return None

    def check_trapped(self, threshold=5_000):
        """
        Check if a particle is trapped in the pipette based on the predicted particle positions.

        Args:
            threshold (int): The threshold distance for determining if a particle is trapped.

        Returns:
            bool: True if a particle is trapped, False otherwise.
        """
        if len(self.c_p['predicted_particle_positions']) < 1:
            self.particles_in_view = False
            self.c_p['particle_trapped'] = False
            return False
        self.particles_in_view = True

        LX = self.c_p['laser_position'][0] - self.c_p['AOI'][0]
        LY = self.c_p['laser_position'][1] - self.c_p['AOI'][2]
        self.c_p['Trapped_particle_position'][0:2], idx = self.find_closest_particle([LX, LY],True)
        self.c_p['Trapped_particle_position'][3] = self.c_p['predicted_particle_radii'][idx]

        # Check if we can get also the z-position, different units tough.
        # Set to None if no z-position found.
        if self.c_p['z-tracking']:
            try:
                self.c_p['Trapped_particle_position'][2] = self.c_p['z-predictions'][idx]
            except IndexError as ie:
                self.c_p['Trapped_particle_position'][2] = None
        else:
            self.c_p['Trapped_particle_position'][2] = None

        trap_dist = ((self.c_p['Trapped_particle_position'][0]-LX)**2
                     + (self.c_p['Trapped_particle_position'][1]-LY)**2)
        self.c_p['particle_trapped'] = trap_dist < threshold
        return idx

    
    def check_in_pipette(self, threshold=3_000, offset=np.array([0, 0]), trapped_idx=None):
        """
        Checks if a particle is in the pipette. Saves the result of this in the approriate 
        data_channel and control parameter (c_p). Check is performed by measuring the distance
        between the particle position and pipette tip position.
        """
        if (not self.c_p['pipette_located'] or not self.c_p['tracking_on']
            or self.c_p['pipette_location'] is None):
            self.data_channels['particle_in_pipette'].put_data(0)
            return
        try:
             # Take into account that particle is above the pipette
            potential_pos, idx = self.find_closest_particle(
                np.array(self.c_p['pipette_tip_location']) - offset, True)
        except TypeError as te:
            print("Error in finding closest particle to pipette", te)
            return
        if potential_pos is None:
            self.c_p['particle_in_pipette'] = False
            self.data_channels['particle_in_pipette'].put_data(1)
            return
        
        # Use particle redii to determine position, check that it has not been updated erroneously.        
        radii = self.data_channels['pipette_particle_radii'].get_data(1)[0]
        if ((self.c_p['pipette_tip_location'][0] - potential_pos[0])**2/4 + 
            ((self.c_p['pipette_tip_location'][1]-radii) - potential_pos[1])**2 < threshold):
            self.c_p['particle_in_pipette'] = True
            self.data_channels['particle_in_pipette'].put_data(2)
            self.c_p['pipette_particle_location'][0:2] = potential_pos
            self.c_p['pipette_particle_location'][3] = self.c_p['predicted_particle_radii'][idx]
            if trapped_idx is not None and trapped_idx == idx:
                self.c_p['particle_trapped'] = False # Particle in pipette, not in trap.
            try:
                if self.c_p['z-tracking']:
                    self.c_p['pipette_particle_location'][2] = self.c_p['z-predictions'][idx]
                else:
                    self.c_p['pipette_particle_location'][2] = None
            except IndexError as ie:
                self.c_p['pipette_particle_location'][2] = None

        else:
            self.data_channels['particle_in_pipette'].put_data(1)

            self.c_p['particle_in_pipette'] = False

    def move_piezos_2_saved_positions(self, positions, threshold=5e-4):
        """
        Moves the piezos to a saved position (on the PSDs, usually (0,0)). Usefull for resetting
        after having done a trapping. Returns False if the piezos are not at the target position.
        Returns True if the piezos are at the target position or if they cannot be moved further.
        """

        # Check if we are autoalignin any of the piezos.
        if self.c_p['portenta_command_2'] != 0:
            return True
        # positions = np.copy(self.c_p['piezo_target_positions'])
        current_a_x = np.mean(self.data_channels['PSD_A_P_X'].get_data(1000)[0])
        current_a_y = np.mean(self.data_channels['PSD_A_P_Y'].get_data(1000)[0])
        current_a_sum = np.mean(self.data_channels['PSD_A_P_sum'].get_data(1000)[0])

        current_b_x = np.mean(self.data_channels['PSD_B_P_X'].get_data(1000)[0])
        current_b_y = np.mean(self.data_channels['PSD_B_P_Y'].get_data(1000)[0])
        current_b_sum = np.mean(self.data_channels['PSD_B_P_sum'].get_data(1000)[0])

        # Check that we are not auto-aliging piezo A, then move piezo a towards target position
        if current_a_sum < 0:
            current_a_sum = 20_000
        if current_b_sum < 0:
            current_b_sum = 20_000
    
        # Will give a number roughly in the range -0.5 to 0.5
        dax = (positions[0] - current_a_x)/current_a_sum 
        day = (positions[1] - current_a_y)/current_a_sum
        dbx = (positions[2] - current_b_x)/current_b_sum
        dby = (positions[3] - current_b_y)/current_b_sum
        

        if (np.abs(dax)<threshold and np.abs(day) < threshold
            and np.abs(dbx)<threshold and np.abs(dby)<threshold):
            return True

        prefac = 30_000 # Moving the whole distance can be unstable.
        ax_n = convert_to_uint16(self.data_channels['dac_ax'].get_data(1)[0] + dax*prefac)
        if ax_n == 65535 or ax_n == 0:
            return True
        self.c_p['piezo_A'][0] = ax_n

        ay_n = convert_to_uint16(self.data_channels['dac_ay'].get_data(1)[0] + day*prefac)
        if ay_n == 65535 or ay_n == 0:
            return True
        self.c_p['piezo_A'][1] = ay_n

        bx_n = convert_to_uint16(self.data_channels['dac_bx'].get_data(1)[0] + dbx*prefac)
        if bx_n == 65535 or bx_n == 0:
            return True
        self.c_p['piezo_B'][0] = bx_n

        by_n = convert_to_uint16(self.data_channels['dac_by'].get_data(1)[0] + dby*prefac)
        if by_n == 65535 or by_n == 0:
            return True
        self.c_p['piezo_B'][1] = by_n

        return False

    def drop_particle(self):
        """
        Function that drops the trapped particle(s) by moving the lasers.
        Used when multiple particles are in the trap but you really only want one.

        Moves away one of the lasers so that the particle is no longer trapped.
        Moving both lasers risks leaving the particle in the same position so that it is
        trapped again when the lasers are reset.
        """
        if not self.c_p['particle_trapped']:
            return True
        
        # Turn off auto-aligning
        if self.c_p['portenta_command_2'] == 1:
            self.c_p['portenta_command_2'] = 0
            if self.data_channels['dac_ax'].get_data(1)[0] > 32768:
                x = 2000
            else:
                x = 60000
            if self.data_channels['dac_ay'].get_data(1)[0] > 32768:
                y = 2000
            else:
                y = 60000
            self.c_p['piezo_A'] = np.uint16([x, y])
        else:
            self.c_p['portenta_command_2'] = 0
            if self.data_channels['dac_bx'].get_data(1)[0] > 32768:
                x = 2000
            else:
                x = 60000
            if self.data_channels['dac_by'].get_data(1)[0] > 32768:
                y = 2000
            else:
                y = 60000
            self.c_p['piezo_B'] = np.uint16([x, y])
        
        return False

    def drop_particle_and_reset_trap(self):
        if self.drop_particle():
            self.c_p['move_piezo_2_target'] = True
            return True
        return False

    def is_point_in_rectangle(self, point, top_left, bottom_right):
        """
        Determine if the point (px, py) is inside the rectangle defined by top-left and bottom-right
        coordinates.
        """
        px, py = point
        tlx, tly = top_left
        brx, bry = bottom_right
        return tlx <= px <= brx and tly <= py <= bry
    
    def check_close_to_Pipette(self, offset=40):
        """
        Checks the predicted particle position for particles that are far away enough from the
        pipette to not be stuck on it
        and use this as possible particles to trap.
        """
        if not self.c_p['pipette_located']:
            #  No pipette in view, we can safely return all particles.
            return self.c_p['predicted_particle_positions']

        # Extract the full extent of the pipette, particles really close to it are not interesting
        # to trap since they are likely to be stuck.
        top_left = self.c_p['pipette_location'][:2]
        top_left[0] -= self.c_p['pipette_location'][2]/2
        top_left[1] -= offset
        bottom_right = self.c_p['pipette_location'][:2]
        bottom_right[0] += self.c_p['pipette_location'][2]/2
        bottom_right[1] = self.c_p['AOI'][3] # Bottom location of AOI
        particle_positions = []

        for pos in self.c_p['predicted_particle_positions']:
            if not self.is_point_in_rectangle(pos, top_left, bottom_right):
                particle_positions.append(pos)

        return np.array(particle_positions)

    def center_on_particle(self, center):
        """
        Move the stage to center the specified center(usually optical trap postion), on the closest
        particle.
        """
        # Center is the laser position on which to center the particle.
       
        if not self.c_p['tracking_on']:
            print("Cannot center particle without tracking on")            
            self.motor_controller.stop_moving()
            return False
        
        center_particle_position, index = self.find_closest_particle(
            center, True, self.check_close_to_Pipette())

        # Check that we are not moving
        if center_particle_position is None:
            
            self.motor_controller.stop_moving()
            self.moving_to_trap = False
            return False
        
        dx = (center[0] - center_particle_position[0]) * self.c_p['ticks_per_pixel']
        dy = (center[1] - center_particle_position[1]) * self.c_p['ticks_per_pixel']
        # Do not move along z, (very) far from focus predictions are unreliable (and the only 
        # ones that would have been relevant).
        
        # If we have trapped a particle then return true        
        if self.c_p['particle_trapped']:
            self.motor_controller.stop_moving()
            self.moving_to_trap = False
            if self.particle_trapped:
                return True
            return False

        fac = 100 / ((dx**2 + dy**2)**0.5)
        if self.c_p['mouse_params'][0] == 0:
            self.motor_controller.move_at_speed(int(dy*fac), int(dx*fac)) # Changed here
            print("Moving at speed x:{dx} y:{dy} ")
            self.movin2trap = True
            return False
        
    def find_and_trap_particle(self):
        """
        Function that moves to the predefined capillary area and tries to retrieve a particle from
        there.
        """

        fludics_channel = ('capillary_1_fluidics_channel' if self.c_p['particle_type'] == 1 else
                           'capillary_2_fluidics_channel')
        # If already moving, return
        if self.motor_controller.is_moving():
            self.c_p['target_pressures'][self.c_p[fludics_channel][0]] = 0
            self.c_p['valves_open'][self.c_p[fludics_channel][2]] = False
            return False
        
        x = self.motor_controller.get_x_position()
        y = self.motor_controller.get_y_position()

        # If a particle is trapped, go 2 the pipette
        if self.c_p['particle_trapped']:
            # If we are already at the pipette with a trapped particle, return true
            if ((x-self.c_p['pipette_location_chamber'][0])**2 + 
                (y-self.c_p['pipette_location_chamber'][1])**2 < 1000):
                self.c_p['target_pressures'][self.c_p[fludics_channel][0]] = 0
                self.c_p['valves_open'][self.c_p[fludics_channel][2]] = False
                return True
            # We are not at the pipette but we have a particle so we should move 2 the pipette
            self.motor_controller.move_to_location(self.c_p['pipette_location_chamber'])
            return False
        
        # No particle trapped and not moving, if we are far from the capillary 
        # then we should move to the capillary
        capillary_position = (self.c_p['capillary_1_position'] if self.c_p['particle_type'] == 1
                              else self.c_p['capillary_2_position'])
        if ((x-capillary_position[0])**2+(y-capillary_position[1])**2 > 100_000
            and not self.moving_to_trap):
            # Don't generally want to change the z focus when moving automatically
            capillary_position[2] = self.motor_controller.get_z_position()
            self.motor_controller.move_to_location(capillary_position)
            return False
        
        # If we are close to the capillary, we should center the particle, we do this by toggling
        # the correct trapping loop
        self.c_p['centering_on'] = True

        # Should also push out some particles. We are in position and trying to trap
        self.c_p['target_pressures'][self.c_p[fludics_channel][0]] = self.c_p[fludics_channel][1]
        self.c_p['valves_open'][self.c_p[fludics_channel][2]] = True
        return False

        
    def find_true_laser_position(self):
        
        if not self.check_trapped() or not self.c_p['tracking_on']:
            print("No particles detected close enough to current laser position")
            return
        LX = self.c_p['laser_position'][0] - self.c_p['AOI'][0]
        LY = self.c_p['laser_position'][1] - self.c_p['AOI'][2]

        min_pos = self.find_closest_particle([LX, LY])

        self.c_p['laser_position'] = [min_pos[0]+self.c_p['AOI'][0], min_pos[1]+self.c_p['AOI'][2]]
        print("Laser position set to: ", self.c_p['laser_position'])

    def find_closest_particle(self, reference_position, return_idx, particle_positions=None):
        """
        Locates the particle closest the reference position.
        """

        if particle_positions is None:
            particle_positions = self.c_p['predicted_particle_positions']

        if len(particle_positions) == 0:
            if return_idx:
                return None, None
            return None
        try:
            LX = reference_position[0]
            LY = reference_position[1]
            min_x = 1000
            min_dist = 2e10
            min_y = 1000
            idx = 0
            min_idx = 0
            for x,y in particle_positions:

                if (x-LX)**2+(y-LY)**2<min_dist:
                    min_dist = (x-LX)**2+(y-LY)**2
                    min_x = x
                    min_y = y
                    min_idx = idx
                idx += 1
            if return_idx:
                return [min_x, min_y], min_idx
            return [min_x, min_y]
        except Exception as e:
            if return_idx:
                return None, None
            return None
        
    def pipette_proximity_check(self, move_to_particle=True):
        """
        Checks if the stage is close enough to the pipette to push the particle to it with just the
        piezos.
        """
        # Calculate the distance to the target position
        # If we are close enough to the target position then we return.

        # Remade this function to be more reliable and 
        if move_to_particle:
            # Move to particle in pipette            
            return self.check_particles_can_touch_with_piezo()
        x_min, x_max, y_min, y_max = self.approximate_piezo_moveable_area()



        # print("Checking pipette proximity",  x_min, x_max, y_min, y_max)
        return (x_min < self.c_p['pipette_tip_location'][0] < x_max and
                 y_min < self.c_p['pipette_tip_location'][1] 
                 - self.c_p['Trapped_particle_position'][3] < y_max)

    def move_to_pipette_tip(self, move_to_particle=True,piezo_y_target=32000, tolerance=4):
        """
        Moves the motors to be centered above the particle in the pipette, or alternatively above
        the pipette tip.
        """
        if not self.c_p['pipette_located']:
            return False
        
        # We put the particle quite far up to make it esier to touch the particles after having
        # moved to the pipette. But only if we are autoaligning.
        piezo_y_target = int(piezo_y_target)
        piezo_x_target = 32768
        if not self.move_particle_to_piezo_position([piezo_x_target, piezo_y_target]):
            return False
        
        # Calculate the distance to the taret position
        if move_to_particle:
            dx = self.c_p['pipette_particle_location'][0] - self.c_p['Trapped_particle_position'][0]  
            dy = ((self.c_p['pipette_particle_location'][1] - self.offset_y)
                  - self.c_p['Trapped_particle_position'][1])
        else:
            dx = self.c_p['pipette_tip_location'][0] - self.c_p['Trapped_particle_position'][0]  
            dy = ((self.c_p['pipette_tip_location'][1] - self.offset_y)
                  - self.c_p['Trapped_particle_position'][1])

        in_position = self.pipette_proximity_check(move_to_particle)

        # Had to move this to be able to do the check also from the main experiment thread.
        if in_position:            
            return True

        # Give some time between the moves(0.4 seconds)
        if time() - self.last_move_time < 0.4:
            return False

        # Move the stage
        num = 1.05 # 0.9 # Factor to not move to far at a time
        current_x = self.motor_controller.get_x_position()
        current_y = self.motor_controller.get_y_position()
        self.c_p['minitweezers_target_pos'][1] = int(current_y - dy*self.c_p['ticks_per_pixel']*num)
        if (dy*self.c_p['ticks_per_pixel'])**2 < 200:
            self.c_p['minitweezers_target_pos'][0] = int(
                current_x + dx*self.c_p['ticks_per_pixel']*num)

        self.last_move_time = time()
        if not in_position:
            self.c_p['move_to_location'] = True
            return False
        if self.motor_controller.is_moving():
            return False
        return in_position

    
    def find_max_sharpness(self, starting_guess, max_sharpness, direction,range=200, step=5):
        """
        Function that finds the maximum sharpness in a given direction from a starting position.
        Used when focusing things like the pipette.
        """
        sharpness = self.compute_gradient_sharpness()

        if sharpness is not None and sharpness>0.85*max_sharpness:
            return True

        z_pos =self.motor_controller.get_z_position()
        # Check that we don't move too far
        if direction>0:
            if z_pos > starting_guess + range:
                print("No sharpness found, moving to starting position")
                self.c_p['minitweezers_target_pos'][2] = starting_guess
                self.c_p['move_to_location'] = True
                return True
        else:
            if z_pos < starting_guess - range:
                print("No sharpness found, moving to starting position")
                self.c_p['minitweezers_target_pos'][2] = True
                self.c_p['move_to_location'] = True
                return True

        # Take a step in the correcet direction
        if not self.motor_controller.is_moving():
            self.c_p['minitweezers_target_pos'][2] = int(z_pos + step*direction)
            self.c_p['move_to_location'] = True
        return False

    def z_focus_pipette(self, range=200, step=10, direction=1):
        """
        Feedback algorithm that moves the stage in the z-direction to find the best focus for the
        pipette. Will move in one direction until it finds a sharpness peak and then move back to
        the best position. Uses the sharpness of the pipette as a measure of the focus quality.        
        """

        if self.c_p['pipette_focus_startup']:
            self.start_z = self.motor_controller.get_z_position()
            self.c_p['minitweezers_target_pos'][2] = int(self.start_z + range*direction)
            self.c_p['move_to_location'] = True
            self.c_p['pipette_focus_startup'] = False
            self.c_p['pipette_sharpnesses'] = []
            self.c_p['pipette_sharpness_Z_pos'] = []
            self.sharpnesses_mapped = False
            return False
        if self.motor_controller.is_moving():
            return False
        if self.particle_in_pipette:
            if (self.c_p['pipette_particle_location'][2] is not None 
                and np.abs(self.c_p['pipette_particle_location'][2]+self.c_p['z-offset'])<0.5):
                z_positions = self.data_channels['pipette_particle_z_position'].get_data(10)
                if (not 0 in z_positions 
                    and np.abs(np.mean(z_positions[-3:])+self.c_p['z-offset'])<0.7):
                    self.c_p['pipette_focus_startup'] = True
                    print("in good focus")
                    return True
                else:
                    print(f"Not in good focus, devious z position{z_positions}")


        self.c_p['pipette_sharpness_Z_pos'].append(self.motor_controller.get_z_position())
        if not self.c_p['pipette_located']:
            self.c_p['pipette_sharpnesses'].append(-1)
        else:
            sharpness = self.compute_gradient_sharpness()
            if sharpness is not None:
                self.c_p['pipette_sharpnesses'].append(sharpness)
            else:
                self.c_p['pipette_sharpnesses'].append(-1)
            print(sharpness)

        if direction > 0 and self.c_p['pipette_sharpness_Z_pos'][-1] > self.start_z - range:
            if not self.sharpnesses_mapped:
                self.c_p['minitweezers_target_pos'][2] = int(
                    self.motor_controller.get_z_position() - step)
                self.c_p['move_to_location'] = True
                return False
        if direction < 0 and self.c_p['pipette_sharpness_Z_pos'][-1] < self.start_z + range:
            if not self.sharpnesses_mapped:
                self.c_p['minitweezers_target_pos'][2] = int(
                    self.motor_controller.get_z_position() + step)
                self.c_p['move_to_location'] = True
                return False
        sharpness_id = np.argmax(self.c_p['pipette_sharpnesses'])            
        if not self.sharpnesses_mapped:
            self.c_p['pipette_sharpnesses'][self.c_p['pipette_sharpnesses']==None] = -1
            best_position = self.c_p['pipette_sharpness_Z_pos'][sharpness_id]
            self.sharpnesses_mapped = True
            self.c_p['minitweezers_target_pos'][2] = int(best_position)
            self.c_p['move_to_location'] = True
            return False
        best_position = self.c_p['pipette_sharpness_Z_pos'][sharpness_id]
        if self.find_max_sharpness(
            int(best_position+direction*step*5),
            np.max(self.c_p['pipette_sharpnesses']),
            direction):
            print("Finding max sharpness")
            self.c_p['pipette_focus_startup'] = True
            return True
        return False

    def z_focus(self, limit=0.4): 
        """
        The system compares the position of the particle in the trap with that of 
        the particle in the pipette, moving the z-position of the stage until they match.
        return:
            True if the system has reached the correct z-position
            False if the system needs to move further or no z-position was found.
        """

        # DO ininital checks to ensure we are set-up for this
        if not self.c_p['tracking_on'] and self.c_p['z-tracking']:
            return False
        if len(self.c_p['predicted_particle_positions']) == 0:
            return False
        
        # Find z-position of particle in pipette
        if (self.c_p['Trapped_particle_position'][2] is None
            or self.c_p['pipette_particle_location'][2] is None):
            return False
         # Puts it to a reasonable scale.
        dz = -(self.c_p['Trapped_particle_position'][2] - self.c_p['pipette_particle_location'][2])
        # Move the stage towards the correct position
        print(f"Time to z-move. {dz},{self.c_p['Trapped_particle_position'][2] },\
              {self.c_p['pipette_particle_location'][2]}")
        if np.abs(dz) < limit:
            print(f"Increasing the z-move counter to: {self.z_move_counter}")
            self.z_move_counter += 1
            if self.z_move_counter>8:
                self.z_move_counter = 0
                print("In good z-position, hooray!")
                return True
            return False
        if dz > 0:
            self.c_p['minitweezers_target_pos'][2] = int(self.motor_controller.get_z_position() + 4)
        if dz < 0:
            self.c_p['minitweezers_target_pos'][2] = int(self.motor_controller.get_z_position() - 4)
        self.z_move_counter = 0
        self.c_p['minitweezers_target_pos'][0] = int(self.motor_controller.get_x_position())
        self.c_p['minitweezers_target_pos'][1] = int(self.motor_controller.get_y_position())

        self.c_p['move_to_location'] = True
        return False        

    def approximate_piezo_moveable_area(self,max_move_microns=9.5):
        """
        Approximates the area in which the particle can be moved with the piezo.
        """
        max_move_pixels = max_move_microns/self.c_p['microns_per_pix']
        # Extract how far we have moved along the x
        x_piezo = self.data_channels['dac_bx'].get_data(1)[0]
        y_piezo = ((self.data_channels['dac_ay'].get_data(1)[0] 
                    + self.data_channels['dac_by'].get_data(1)[0])/2)
        x_particle = self.c_p['Trapped_particle_position'][0]
        y_particle = self.c_p['Trapped_particle_position'][1]
        x_min = x_particle - ((65536 - x_piezo) / 65536) * max_move_pixels
        x_max = x_particle + (x_piezo / 65536) * max_move_pixels
        y_min = y_particle - (y_piezo / 65536) * max_move_pixels
        y_max = y_particle + ((65536 - y_piezo) / 65536) * max_move_pixels
        return x_min, x_max, y_min, y_max

    def check_limit(self, number):
        if number < 0:
            return 0
        if number > 65535:
            return 65535
        return int(number)

    def move_particle_to_piezo_position(self, piezo_position, tolerance=10_000, step_size=2000):
        """
        Gradually moves the trap to the target position with autoalign turned on.
        Will autoalign piezo A if autoalign is not already set. Will do so in steps so as not to
        loose the particles. Returns true if the particle distance squared in DAC steps to the
        target position within the tolerance of the target position.
        """

        # Check which piezo to move and read the current position
        if self.c_p['portenta_command_2'] == 1:
            key = 'piezo_B'
            current_x = self.data_channels['dac_bx'].get_data(1)[0]
            current_y = self.data_channels['dac_by'].get_data(1)[0]
        elif self.c_p['portenta_command_2'] == 2:
            key = 'piezo_A'
            current_x = self.data_channels['dac_ax'].get_data(1)[0]
            current_y = self.data_channels['dac_ay'].get_data(1)[0]
        else:
            self.c_p['portenta_command_2'] = 1 # Autoaliging A
            key = 'piezo_B'
            current_x = self.data_channels['dac_bx'].get_data(1)[0]
            current_y = self.data_channels['dac_by'].get_data(1)[0]

        dx = piezo_position[0] - current_x
        dy = piezo_position[1] - current_y
        if dx**2 + dy**2 > tolerance:
            # Moving to target position.
            if dx < 0:
                self.c_p[key][0] = int(current_x - min(step_size, -dx))
            else:
                self.c_p[key][0] = int(current_x + min(step_size, dx))
            if dy < 0:
                self.c_p[key][1] = int(current_y - min(step_size, -dy))
            else:
                self.c_p[key][1] = int(current_y + min(step_size, dy))
            return False
        return True

    def move_particle_to_position_with_piezo(self, target_position, piezo="piezo_b",
                                             movement_threshold=3, movement_lim=2000):
        """
        Moves the particle with the piezo to the target position.
        Target position should be given in pixels in the image.
        """

        x_min, x_max, y_min, y_max = self.approximate_piezo_moveable_area()
        if not (x_min < target_position[0] < x_max and y_min < target_position[1] < y_max):
            print("Target position outside of moveable area")
            print(target_position, x_min, x_max, y_min, y_max)
            return False

        dx = target_position[0] - self.c_p['Trapped_particle_position'][0]
        dy = target_position[1] - self.c_p['Trapped_particle_position'][1]
        if dx**2+dy**2<(movement_threshold/self.c_p['microns_per_pix'])**2:
            return True

        if piezo == "piezo_b":
            current_piezo_pos_x = self.data_channels['dac_bx'].get_data(1)[0]
            current_piezo_pos_y = self.data_channels['dac_by'].get_data(1)[0]
            if dx > 0:
                new_pos = current_piezo_pos_x - (min(int(dx*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_B'][0] = self.check_limit(new_pos)
            if dx < 0:
                new_pos = current_piezo_pos_x + (min(int(-dx*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_B'][0] = self.check_limit(new_pos)
            if dy > 0:
                new_pos = current_piezo_pos_y + (min(int(dy*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_B'][1] = self.check_limit(new_pos)
            if dy < 0:
                new_pos = current_piezo_pos_y - (min(int(-dy*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_B'][1] = self.check_limit(new_pos)
        else:
            current_piezo_pos_x = self.data_channels['dac_ax'].get_data(1)[0]
            current_piezo_pos_y = self.data_channels['dac_ay'].get_data(1)[0]
            if dx > 0:
                new_pos = current_piezo_pos_x + (min(int(dx*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_A'][0] = self.check_limit(new_pos)
            if dx < 0:
                new_pos = current_piezo_pos_x - (min(int(-dx*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_A'][0] = self.check_limit(new_pos)
            if dy > 0:
                new_pos = current_piezo_pos_y + (min(int(dy*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_A'][1] = self.check_limit(new_pos)
            if dy < 0:
                new_pos = current_piezo_pos_y - (min(int(-dy*self.bits_per_pixel), movement_lim))
                self.c_p['piezo_A'][1] = self.check_limit(new_pos)

        return False

    def check_particles_can_touch_with_piezo(self):
        """
        Checks if the particles are close enough to each other to touch 
        when moved with only the piezos(laser).
        """
        if len(self.c_p['predicted_particle_positions']) < 2:
            return False
        x_min, x_max, y_min, y_max = self.approximate_piezo_moveable_area()
        pipette_particle_top = (self.c_p['pipette_particle_location'][1]
                                - self.c_p['pipette_particle_location'][3])
        pipette_particle_bottom_reachable = y_max + self.c_p['pipette_particle_location'][3]
        return (x_min < self.c_p['pipette_particle_location'][0] < x_max and
                pipette_particle_bottom_reachable > pipette_particle_top)            

    def align_trapped_and_pipette_X(self, move_dist=100, limit=0.1):
        """
        Aligns the trapped particle and the pipette in the X direction.

        Args:
            move_dist (int, optional): The distance to move the piezo stage. Defaults to 100.
            limit (float, optional): The limit for the alignment in microns. Defaults to 0.1.

        Returns:
            bool: True if alignment is finished, False otherwise.
        """

        dx = ((self.c_p['pipette_particle_location'][0] - self.c_p['Trapped_particle_position'][0])
              * self.c_p['microns_per_pix'])

        if dx > limit:
            if self.c_p['portenta_command_2'] == 1: # Autoaligning A
                self.c_p['piezo_B'][0] = max(int(self.c_p['piezo_B'][0] - move_dist), 1000)
            else:
                self.c_p['piezo_A'][0] = min(int(self.c_p['piezo_A'][0] + move_dist), 65000)
            return True

        if dx < -limit:
            if self.c_p['portenta_command_2'] == 1:
                self.c_p['piezo_B'][0] = min(int(self.c_p['piezo_B'][0] + move_dist), 65000)
            else:
                self.c_p['piezo_A'][0] = max(int(self.c_p['piezo_A'][0] - move_dist), 1000)
            return True

        return False

    def touch_trapped_2_pieptte_particle(self,
                                         max_dist=6,
                                         force_limit_lower=10,
                                         position_limit=0.85):
        """
        force_limit - the amount of force, in pN which the particles are allowed to be pushed
        together with.
        position_limit - the distance in particle radii that the particles are allowed to be
            pushed together with.
        max_dist - the maximum distance in microns that the particles are allowed to be pushed
            together with.

        returns a description of what the function is doing or why it is not doing anything.
        """
        
        if not self.particle_in_pipette or not self.particle_trapped:
            print(f"No particle in pipette or trap, pipette {self.particle_in_pipette}, \
                  trapped {self.particle_trapped}")
            self.c_p['touch_particles'] = False
            self.c_p['touch_counter'] = 0
            self.c_p['autocontroller_current_step'] = 'checking_pipette'
            return 'particle missing'

        if not self.c_p['z-tracking']:            
            self.c_p['z-tracking'] = True
            return 'z-tracking not on'

        if not self.check_particles_can_touch_with_piezo():
            print("Particles too far apart")
            return "Particles too far apart"

        # Check that we are autoaligning.
        if self.c_p['portenta_command_2'] == 0:
            self.c_p['portenta_command_2'] = 1
            return "No autoalign, turning it on"

        # Align in x unless DNA is tugging on the particle
        tot_force = self.data_channels['trapped_x_force'].get_data(1)[0]**2
        tot_force += self.data_channels['trapped_y_force'].get_data(1)[0]**2
        if self.align_trapped_and_pipette_X() and tot_force < 10:
            print("Aligning in x")
            return "Aligning in x"

        # Check z-alignment, changed to an average to get a more reliable reading.
        #dz = 10 * (self.trapped_bead_z.get_average() - self.pipette_bead_z.get_average())
        z_bead = np.mean(self.data_channels['trapped_particle_z_position'].get_data(10))
        z_pipette = np.mean(self.data_channels['pipette_particle_z_position'].get_data(10))
        dz = (z_bead - z_pipette)

        # this will always be positive since the pipette is below the trapped particle
        dy = self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1] 
        radii_sum = (self.c_p['pipette_particle_location'][3]
                     + self.c_p['Trapped_particle_position'][3])

        # Fix z-alginment if the particles are not touching and are missaligned in z
        if np.abs(dz) > 1 and dy > (radii_sum*1.2+2/self.c_p['microns_per_pix']):
            self.c_p['focus_z_trap_pipette'] = True
            return 'need aligning in z'
        
        self.move_up_and_down(
            max_dist=max_dist,
            force_limit_lower=force_limit_lower,
            position_limit=position_limit)
        y_force = np.abs(np.mean(self.data_channels['F_total_Y'].get_data(20)))

        if y_force > 100:
            self.c_p['touch_particles'] = False
            self.c_p['touch_counter'] = 0
            self.c_p['autocontroller_current_step'] = 'checking_pipette'
            self.restart_experiment()
            # Particle stuck
            return "Particle stuck"
            


    def zoom_2_particles(self, width=20, height=20, pipette_particle=True):
        """
        Zooms to the area around the particle being experimented on. Useful when recording videos
        as part of measurement.
        """
        
        if not self.c_p['particle_trapped']:
            print("No particle trapped")
            return
        # 10 microns wide
        x_min = max(int(self.c_p['AOI'][0] + self.c_p['Trapped_particle_position'][0]
                        - (width/2)/self.c_p['microns_per_pix']),0)
        x_max = min(int(x_min + width/self.c_p['microns_per_pix']),self.c_p['camera_width'])
        y_min = max(int(self.c_p['AOI'][2] + self.c_p['Trapped_particle_position'][1]
                        - height/self.c_p['microns_per_pix']),0)
        if pipette_particle:
            if self.particle_in_pipette and self.c_p['particle_in_pipette']:
                y_max = min(int(self.c_p['AOI'][2] + self.c_p['pipette_particle_location'][1]
                                + height/self.c_p['microns_per_pix']),self.c_p['camera_height'])
            else:
                return "No particle in pipette"
        else:
            y_max = min(int(y_min + 2*height/self.c_p['microns_per_pix']),self.c_p['camera_height'])

        self.zoomed_in = True
        # Save the old AOI to be able to zoom out again.
        self.AOI0 = self.c_p['AOI'][0]
        self.AOI1 = self.c_p['AOI'][1]
        self.AOI2 = self.c_p['AOI'][2]
        self.AOI3 = self.c_p['AOI'][3]

        self.c_p['AOI'] = [x_min, x_max, y_min, y_max]
        self.c_p['new_settings_camera'] = [True, 'AOI']

    def reset_zoom_level(self):
        """
        Resets the zoom level to the original level.
        """
        self.zoomed_in = False
        self.c_p['AOI'] = [self.AOI0, self.AOI1, self.AOI2, self.AOI3]
        self.c_p['new_settings_camera'] = [True, 'AOI']

    def move_up_and_down(self,
                         max_dist=5,
                         force_limit_lower=1000,
                         force_limit_upper=1000,
                         position_limit=0.8,
                         direction=None,
                         verbose=False):

        """
        Moves the trapped particle up and down to touch the pipette particle and look for a
        molecule strand. Force limits refers to the maximum force okay to reach when moving down and
        touching the particles(limit lower) and moving up and pulling the molecule(limit upper).
        Position limit is the distance in particle radii that the particles are allowed to be pushed
        together with.
        """

        # this will always be positive since the pipette is below the trapped particle
        dy = self.c_p['pipette_particle_location'][1] - self.c_p['Trapped_particle_position'][1] 
        radii_sum = (self.c_p['pipette_particle_location'][3]
                     + self.c_p['Trapped_particle_position'][3])
        current_force = np.mean(self.data_channels['F_total_Y'].get_data(10))

        if direction is not None:
            if direction == 'down':
                self.moving_down = True
            else:
                self.moving_down = False
 
        if self.moving_down:
            # Check if the particles are touching, either force or radii can tell us this
            if np.abs(current_force) > force_limit_lower or dy < radii_sum*position_limit:
                self.moving_down = False
                if verbose:
                    print("Changing direction of movement to up",
                          current_force > force_limit_lower,
                          dy < radii_sum*position_limit)

                return 'touching particles'
            if self.c_p['portenta_command_2'] == 1:
                self.c_p['piezo_B'][1] = min(64000, int(self.c_p['piezo_B'][1] + 200))
                if self.c_p['piezo_B'][1] >=64000:
                    self.moving_down = False
            else:
                self.c_p['piezo_A'][1] = min(64000, int(self.c_p['piezo_A'][1] + 200))
                if self.c_p['piezo_A'][1] >=64000:
                    self.moving_down = False
        else:
            # We are moving up, check if we have reached the max_distance in particle radiis.
            if (-current_force > force_limit_upper 
                or dy > (max_dist/self.c_p['microns_per_pix'] + radii_sum)):
                self.moving_down = True
                if verbose:
                    print("Changing direction of movement to down",
                          current_force,
                          force_limit_upper,
                          dy,
                          max_dist/self.c_p['microns_per_pix'],
                          radii_sum)

                self.c_p['touch_counter'] += 1
                return 'touching particles'
            if self.c_p['portenta_command_2'] == 1:
                self.c_p['piezo_B'][1] = max(1000, int(self.c_p['piezo_B'][1] - 200))
                if self.c_p['piezo_B'][1] <= 1000:
                    self.moving_down = True
            else:
                self.c_p['piezo_A'][1] = max(1000, int(self.c_p['piezo_A'][1] - 200))
                if self.c_p['piezo_A'][1] <= 1000:
                    self.moving_down = True

        return 'touching particles'

    def put_particle_in_pipette(self):
        """
        Moves the particle from the trap to the pipette.
        """
        if not self.c_p['particle_trapped']:
            return False

        x_min, x_max, y_min, y_max = self.approximate_piezo_moveable_area()
        if (not (x_min < self.c_p['pipette_tip_location'][0] < x_max and
                 y_min < self.c_p['pipette_tip_location'][1] 
                 - self.c_p['Trapped_particle_position'][3] < y_max)):
            print("Pipette location outside of moveable area")

        # Move the particle to the pipette
        target_position = [
            self.c_p['pipette_tip_location'][0],
            self.c_p['pipette_tip_location'][1] - self.c_p['Trapped_particle_position'][3]]

        if self.c_p['portenta_command_2'] == 1:
            return self.move_particle_to_position_with_piezo(
                target_position, piezo='piezo_b', movement_threshold=1.5)
        return self.move_particle_to_position_with_piezo(
            target_position, piezo='piezo_a', movement_threshold=1.5)
    
    def terminate_protocol(self, terminate_recording=False):
        self.c_p['protocol_data'][0] = 0
        self.limits_found = [False, False]
        self.protocol_started = False
        if self.data_saver.is_saving():
            self.main_window.record_data()
        if terminate_recording and self.c_p['recording']:
            # Will turn off recording after having finished the experiment, regardless of what the
            # user thinks of this.
            self.main_window.toggle_recording()
            self.reset_zoom_level()


    def update_lasers_position_from_PSDs(self):
        """
        Estimates the laser positions based on the psd readings. Assumes that the laser_position_A
        and laser_position_B are correct and set to while the PSD_position_reading was 0.        
        """
        self.PSD_A_P_sum = np.mean(self.data_channels['PSD_A_P_sum'].get_data(10))
        self.PSD_B_P_sum = np.mean(self.data_channels['PSD_B_P_sum'].get_data(10))
        if self.PSD_A_P_sum == 0 or self.PSD_B_P_sum == 0:
            return
        if self.APX_coeffs is not None:
            chunck_length = 10

            apx = np.mean(
                self.data_channels['PSD_A_P_X'].get_data(chunck_length)) / self.PSD_A_P_sum
            apy = np.mean(
                self.data_channels['PSD_A_P_Y'].get_data(chunck_length)) / self.PSD_A_P_sum
            bpx = np.mean(
                self.data_channels['PSD_B_P_X'].get_data(chunck_length)) / self.PSD_B_P_sum
            bpy = np.mean(
                self.data_channels['PSD_B_P_Y'].get_data(chunck_length)) / self.PSD_B_P_sum

            laser_a_x = self.poly2d(self.APX_coeffs,apx,apy) / self.c_p['microns_per_pix']
            laser_a_y = self.poly2d(self.APY_coeffs,apx,apy) / self.c_p['microns_per_pix']
            laser_b_x = self.poly2d(self.BPX_coeffs,bpx,bpy) / self.c_p['microns_per_pix']
            laser_b_y = self.poly2d(self.BPY_coeffs,bpx,bpy) / self.c_p['microns_per_pix']

        else:
            psd_a_x = np.mean(self.data_channels['PSD_A_P_X'].get_data(10)) / self.PSD_A_P_sum
            psd_a_y = np.mean(self.data_channels['PSD_A_P_Y'].get_data(10)) / self.PSD_A_P_sum
            psd_b_x = np.mean(self.data_channels['PSD_B_P_X'].get_data(10)) / self.PSD_B_P_sum 
            psd_b_y = np.mean(self.data_channels['PSD_B_P_Y'].get_data(10)) / self.PSD_B_P_sum 

            # For some reason this only display something once the tracking is turned on.
            laser_a_x = (self.c_p['laser_position_A'][0]
                         + (self.c_p['laser_a_transfer_matrix'][0] * psd_a_x
                            + self.c_p['laser_a_transfer_matrix'][1] * psd_a_y)
                            / self.c_p['microns_per_pix'])
            laser_a_y = (self.c_p['laser_position_A'][1]
                         + (self.c_p['laser_a_transfer_matrix'][2] * psd_a_x
                            + self.c_p['laser_a_transfer_matrix'][3] * psd_a_y)
                            / self.c_p['microns_per_pix'])
            laser_b_x = (self.c_p['laser_position_B'][0]
                         + (self.c_p['laser_b_transfer_matrix'][0] * psd_b_x
                            + self.c_p['laser_b_transfer_matrix'][1] * psd_b_y)
                            / self.c_p['microns_per_pix'])
            laser_b_y = (self.c_p['laser_position_B'][1]
                         + (self.c_p['laser_b_transfer_matrix'][2] * psd_b_x
                            + self.c_p['laser_b_transfer_matrix'][3] * psd_b_y)
                            / self.c_p['microns_per_pix'])

        self.c_p['laser_position_A_predicted'] = np.array([laser_a_x, laser_a_y])
        self.c_p['laser_position_B_predicted'] = np.array([laser_b_x, laser_b_y])
        self.c_p['laser_position'] = (self.c_p['laser_position_A_predicted']
                                      + self.c_p['laser_position_B_predicted']) / 2

    def suck_particle_into_pipette(self):
        """
        Puts the trapped particle inside the pipette.
        """
        if not self.c_p['particle_trapped'] or not self.c_p['pipette_located']:
            return "Particle not trapped or pipette not located"
        
        # estimate if we are supposed to go to the pipette with motors or with the piezo
        x_min, x_max, y_min, y_max = self.approximate_piezo_moveable_area()

        # Move the particle to the pipette using the piezo
        distance_x = self.c_p['pipette_tip_location'][0] - self.c_p['Trapped_particle_position'][0]
        distance_y = (self.c_p['pipette_tip_location'][1] - self.c_p['Trapped_particle_position'][1]
                      - self.c_p['Trapped_particle_position'][3]) # Accounting for radii now.
        if distance_x**2 + distance_y**2 < (2/self.c_p['microns_per_pix'])**2:
            print(f"Close enough to suck")
            self.c_p['move_to_pipette_tip'] = False

            self.c_p['move_particle2pipette'] = False
            self.pipette_pump.activate_suction()
            sleep(1)
            self.pipette_pump.deactivate_suction()
            self.motor_controller.move_to_location(self.c_p['pipette_location_chamber'])
            # Turn of auto-aligning
            self.c_p['portenta_command_2'] = 0
            self.c_p['move_piezo_2_target'] = True
            
            return "Finished"

        if not self.pipette_proximity_check(False) and not self.c_p['move_particle2pipette']:

            self.c_p['move_to_pipette_tip'] = True
            print("Pipette location outside of moveable area. moving with motors",
                  x_max, x_min, y_max, y_min, self.c_p['pipette_tip_location'])
            return "Moving with motors to pipette location"

        # Move the particle to the pipette using the piezo
        print("Piezos moving to put particle close to pipette")
        self.put_particle_in_pipette()
    
    def restart_experiment(self):
        """
        Drops the trapped particle and restarts the experiment with a new particle.
        Does this by moving the piezos away from their current position and then flushing the
        chamber.
        """
        self.c_p['drop_particle'] = True
        print("Restaring experiment by dropping particle and flushing chamber.")
        # Setting some parameters to false
        self.c_p['search_and_trap'] = False
        self.c_p['centering_on'] = False
        self.c_p['move_to_pipette_tip'] = False
        self.c_p['touch_particles'] = False
        self.c_p['autocontroller_current_step'] = 'checking_pipette'

        central_id = self.c_p['central_fluidics_channel'][0]
        tmp = self.c_p['target_pressures'][central_id]
        self.pipette_pump.activate_suction()
        self.c_p['pipette_pump_on'] = True
        sleep(1)
        self.c_p['target_pressures'][central_id] = self.c_p['central_fluidics_channel'][1]
        
        # Opening valves
        for idx,_ in enumerate(self.c_p['valves_open']):
            print("Opening valves for flushing")
            self.c_p['valves_open'][idx] = True

        sleep(2)
        
        self.c_p['target_pressures'][central_id] = tmp

        # Closing valves
        for idx,_ in enumerate(self.c_p['valves_open']):
            self.c_p['valves_open'][idx] = False
            print("Closing valves")
        sleep(1)
        self.pipette_pump.deactivate_suction()
        self.c_p['pipette_pump_on'] = False
        self.c_p['portenta_command_2'] = 0
        return "Restarting experiment"

    def get_2_particles(self):
        """
        A generic method for getting two particles, one in trap and one in pipette.
        When this returns True there are two particles, one in the trap and the other 
        in the pipette aligned and ready for experiments.
        """

        if self.c_p['multiple_particles_trapped']:
            # Drops the trapped particle and restarts the experiment with a new particle.
            self.c_p['autocontroller_current_step'] = 'checking_pipette'
            if self.c_p['particle_type'] == 2:
                # Does not really matter if we have two particles when going to put one in the
                # pipette.
                print("Many particles trapped, restarting experiment")
                self.restart_experiment()
                return False

        if self.c_p['autocontroller_current_step'] == 'checking_pipette':
            print("Checking pipette")

            self.c_p['centering_on'] = False
            self.c_p['touch_particles'] = False
            self.c_p['search_and_trap'] = False
            self.c_p['move_to_pipette_tip'] = False

            x = self.motor_controller.get_x_position()
            y = self.motor_controller.get_y_position()

            if ((x-self.c_p['pipette_location_chamber'][0])**2
                + (y-self.c_p['pipette_location_chamber'][1])**2 > 200):
                if not self.motor_controller.is_moving():
                    self.motor_controller.move_to_location(self.c_p['pipette_location_chamber'])
                print(f"Moving to pipette location {self.c_p['pipette_location_chamber']}")
                return False

            if self.motor_controller.is_moving():
                return False

            if (not self.c_p['focus_pipette'] and
                self.c_p['autocontroller_current_step'] != 'focusing_pipette'):
                self.c_p['autocontroller_current_step'] = 'focusing_pipette'
                self.c_p['focus_pipette'] = True
                self.c_p['move_to_pipette_tip'] = False
                self.c_p['suck_into_pipette'] = False
                print("Starting focusing of pipette")
                return False

        if (self.c_p['focus_pipette']
            and self.c_p['autocontroller_current_step'] == 'focusing_pipette'):
            print("Focusing pipette")
            self.c_p['touch_particles'] = False
            self.c_p['move_to_pipette_tip'] = False
            self.c_p['suck_into_pipette'] = False
            return False

        # If we are close to the pipette then we should check what is in the pipette
        if (self.c_p['autocontroller_current_step'] == 'focusing_pipette'
            and not self.motor_controller.is_moving()):
            print("Checking pipette contents and updating z position of pipette")
            if not self.c_p['pipette_located']:
                self.c_p['autonomous_experiment'] = False
                print("Pipette not located, stopping experiment")

            # 4 cases, particle trapped and particle in piptte can both be either true or false
            if not self.particle_in_pipette and not self.particle_trapped:
                print("Collecting particle type 1")
                self.c_p['search_and_trap'] = True
                self.c_p['particle_type'] = 1 # Get particle for the pipette
                self.c_p['autocontroller_current_step'] = 'searching_for_particle_1'
                self.c_p['move_piezo_2_target'] = True
                self.c_p['portenta_command_2'] = 0#1
                self.c_p['suck_into_pipette'] = False
                return False

            if not self.particle_in_pipette and self.particle_trapped:
                print("Pipette suction time, hooray!")
                self.c_p['suck_into_pipette'] = True
                return False

            if ((self.particle_in_pipette or self.c_p['particle_in_pipette'])
                and not self.particle_trapped):
                # Go get particle type number 2
                print("Collecting particle type 2")
                self.c_p['search_and_trap'] = True
                self.c_p['particle_type'] = 2 # Get particle with DNA
                self.c_p['autocontroller_current_step'] = 'searching_for_particle_2'
                self.c_p['move_piezo_2_target'] = True
                self.c_p['portenta_command_2'] = 0
                self.c_p['suck_into_pipette'] = False
                return False

            if self.particle_in_pipette and self.particle_trapped:
                # if the trapped particle is of type 1 then we shuld drop it and get a 
                # type 2 particle
                self.c_p['suck_into_pipette'] = False
                if self.c_p['particle_type'] == 1:
                    # Warns user that it might not be the correct particle type
                    # Not a serious matter, will just loose a little time if that is the case.
                    print("Warning may: Have particle type 1 in trap not target type 2")
                    
                print(" Ready for next step")
                return True
        
        # We are searching for a new particle to trap, let the other loop do it's job
        if (self.c_p['autocontroller_current_step'] == 'searching_for_particle_1'
            or self.c_p['autocontroller_current_step'] == 'searching_for_particle_2'):
            # Once the loop is finished it automatically turns off the search_and_trap and goes to 
            # the pipette, use this as a cue to move on by aligning the pipette and the particle 
            # in it.
            if not self.c_p['search_and_trap']:
                self.c_p['autocontroller_current_step'] = 'checking_pipette'
            return False
        return self.particle_in_pipette and self.particle_trapped
    
    def collect_and_align_2_particles(self):
        """
        This function will collect 2 particles, one in the pipette the other in the trap.
         It will then align them so that they can be "touched" together. Essentially preparing
         an electrostatic measurement or a single molecule experiment.
        """
        if not self.particle_trapped:
            self.c_p['move_piezo_2_target'] = True
            # Turn off autoalignment
            self.c_p['portenta_command_2'] = 0 # No autoalign
        else:
            self.c_p['portenta_command_2'] = 1
        if not self.get_2_particles():
            return False
        
        # Here things can go wrong
        self.c_p['autocontroller_current_step'] = 'move_to_pipette_tip'

        if self.c_p['autocontroller_current_step'] == 'move_to_pipette_tip':
            print("In move_to_pipette_tip step, Check: ", self.pipette_proximity_check(True))
            # Check trapped, if not go back to checking pipette
            if not self.particle_trapped:
                self.c_p['move_to_pipette_tip'] = False
                self.c_p['autocontroller_current_step'] = 'checking_pipette'
                return False

            # Check if we should move with the motors, if so autoalign and move towards the right
            #  area
            if not self.pipette_proximity_check(True):
                # Move 2 area above the pipette and toggle autoaligning
                self.c_p['portenta_command_2'] = 2
                self.c_p['move_to_pipette_tip'] = True
                self.c_p['autocontroller_current_step'] = 'move_to_pipette_tip'
                return False
            # If we should not move with the motors then we should move with the piezo and initiate
            # the "touching particles protocol"
            self.c_p['move_to_pipette_tip'] = False
            self.c_p['autocontroller_current_step'] = 'checking_pipette'

            return True
        

    def full_auto_experiment(self):
        """
        The full autonomous protocol for doing a molecule stretching experiment.

        1 Checks what is in the trap, if nothing is trapped then we should not autoalign.
        2 If the current step is check pipette then we focus the pipette
        3 Check pipette contents
        """

        protocol = self.protocols[self.c_p['autonomous_experiment_type']]

        # This prevents issues if you change protocol in the widget but don't turn it off first.
        for key in self.protocols:
            if key != self.c_p['autonomous_experiment_type']:
                self.protocols[key].stop()

        if not protocol.is_running():
            protocol.start()
        
        return protocol.run()

    def get_position(self):
        x = int(self.motor_controller.get_x_position())
        y = int(self.motor_controller.get_y_position())
        z = int(self.motor_controller.get_z_position())
        return [x, y, z]
    
    def calc_distance(self, pos1, pos2):
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)

    def run_stokes_test(self):
        """
        Function to run the stokes test autonomously.
        """

        if not self.c_p['stokes_test_running']:    
            return

        match self.c_p['stokes_test_step']:
            case "startup":
                # At startup go to the left position
                distance = self.calc_distance(self.get_position(), self.c_p['stokes_left_pos'])
                if distance < 100 and not self.motor_controller.is_moving():
                    self.c_p['stokes_test_step'] = "left2right"
                elif not self.motor_controller.is_moving():
                    self.motor_controller.move_to_location(self.c_p['stokes_left_pos'])
            case "left2right":
                distance = self.calc_distance(self.get_position(), self.c_p['stokes_right_pos'])
                if distance < 100 and not self.motor_controller.is_moving():
                    # We have stopped near the right position, time to move in the other direction
                    self.c_p['stokes_test_step'] = "right2left"
                    self.move_counter += 1
                if distance>100 and not self.motor_controller.is_moving():
                    self.motor_controller.move_to_location(self.c_p['stokes_right_pos'])

            case "right2left":
                distance = self.calc_distance(self.get_position(), self.c_p['stokes_left_pos'])
                if distance < 100 and not self.motor_controller.is_moving():
                    # We have stopped near the left position, time to move in the other direction
                    self.c_p['stokes_test_step'] = "left2right"
                    self.move_counter += 1

                if distance > 100 and not self.motor_controller.is_moving():
                    self.motor_controller.move_to_location(self.c_p['stokes_left_pos'])

                if self.move_counter >= self.c_p['stokes_max_move_count']*2:
                    self.c_p['stokes_test_step'] = "left2center"
                    self.move_counter = 0

            case "left2center":
                # We are moving from left to center to then move up and down.
                distance = self.calc_distance(self.get_position(), self.c_p['stokes_center_pos'])
                if distance < 100 and not self.motor_controller.is_moving():
                    self.c_p['stokes_test_step'] = "down2up"
                #
                if distance > 100 and not self.motor_controller.is_moving():
                    self.motor_controller.move_to_location(self.c_p['stokes_center_pos'])

            case "down2up":
                distance = self.calc_distance(self.get_position(), self.c_p['stokes_up_pos'])

                if distance < 100 and not self.motor_controller.is_moving():
                    self.c_p['stokes_test_step'] = "up2down"
                    self.move_counter += 1

                if distance > 100 and not self.motor_controller.is_moving():
                    self.motor_controller.move_to_location(self.c_p['stokes_up_pos'])

            case "up2down":
                distance = self.calc_distance(self.get_position(), self.c_p['stokes_down_pos'])

                if distance < 100 and not self.motor_controller.is_moving():
                    self.c_p['stokes_test_step'] = "down2up"
                    self.move_counter += 1

                if distance > 100 and not self.motor_controller.is_moving():
                    self.motor_controller.move_to_location(self.c_p['stokes_down_pos'])

                if self.move_counter >= self.c_p['stokes_max_move_count']*2:
                    self.c_p['stokes_test_step'] = "startup"
                    self.move_counter = 0
                    self.c_p['stokes_test_running'] = False

    
    def add_prediction_to_data(self):
        """
        Adds the predicted position of the trapped particle and the particle in the pipette to the
        data channels along with a time-stamp that indicates when the prediction was made.
        """

        if self.c_p['particle_trapped'] and self.c_p['Trapped_particle_position'][2] is not None:
            self.data_channels['trapped_particle_x_position'].put_data(
                self.c_p['Trapped_particle_position'][0]*self.c_p['microns_per_pix'])
            self.data_channels['trapped_particle_y_position'].put_data(
                self.c_p['Trapped_particle_position'][1]*self.c_p['microns_per_pix'])
            self.data_channels['trapped_particle_z_position'].put_data(
                self.c_p['Trapped_particle_position'][2])
            self.data_channels['trapped_particle_radii'].put_data(
                self.c_p['Trapped_particle_position'][3]*self.c_p['microns_per_pix'])
            self.data_channels['trapped_x_force'].put_data(
                np.mean(self.data_channels['F_total_X'].get_data(500)))
            self.data_channels['trapped_y_force'].put_data(
                np.mean(self.data_channels['F_total_Y'].get_data(500)))
        else:
            self.data_channels['trapped_particle_x_position'].put_data([0])
            self.data_channels['trapped_particle_y_position'].put_data([0])
            self.data_channels['trapped_particle_z_position'].put_data([0])
            self.data_channels['trapped_particle_radii'].put_data([0])
            self.data_channels['trapped_x_force'].put_data([0])
            self.data_channels['trapped_y_force'].put_data([0])

        if self.c_p['particle_in_pipette'] and self.c_p['pipette_particle_location'][2] is not None:
            self.data_channels['pipette_particle_x_position'].put_data(
                self.c_p['pipette_particle_location'][0]*self.c_p['microns_per_pix'])
            self.data_channels['pipette_particle_y_position'].put_data(
                self.c_p['pipette_particle_location'][1]*self.c_p['microns_per_pix'])
            self.data_channels['pipette_particle_z_position'].put_data(
                self.c_p['pipette_particle_location'][2])
            self.data_channels['pipette_particle_radii'].put_data(
                self.c_p['pipette_particle_location'][3]*self.c_p['microns_per_pix'])
        else:
            self.data_channels['pipette_particle_x_position'].put_data([0])
            self.data_channels['pipette_particle_y_position'].put_data([0])
            self.data_channels['pipette_particle_z_position'].put_data([0])
            self.data_channels['pipette_particle_radii'].put_data([0])
        self.data_channels['prediction_time'].put_data(self.data_channels['T_time'].get_data(1))

    def analyze_frame(self):
        if self.c_p['image'] is None:
            # The image was not captured yet
            return
        try:
            self.object_tracker.analyze_frame(self.c_p['image'])
        except Exception as e:
            print("Error in analyze frame: ", e)
            return
        (
            self.c_p['predicted_particle_positions'],
            self.c_p['predicted_particle_radii'],
        ) = self.object_tracker.predict_particle_positions()
        
        # Pipette detection        
        (
            self.c_p['pipette_location'],
            self.c_p['pipette_tip_location'],
            self.c_p['pipette_in_frame']
        ) = self.object_tracker.predict_pipette_position()

        if self.c_p['accurate_tip_detection_needed']:
            self.object_tracker.pipette_tilt_detection(self.c_p)
        else:
            self.c_p['pipette_tip_location'][0] -= (
                self.c_p['pipette_tilt'] * self.c_p['pipette_location'][3])

        if not self.c_p['pipette_in_frame']:
            self.pipette_view_counter += 1
            if self.pipette_view_counter > 4:
                self.c_p['pipette_located'] = False
        else:
            self.c_p['pipette_located'] = True

        if self.c_p['z-tracking']:
            self.c_p['z-predictions'] = self.object_tracker.predict_z_positions(
                self.c_p['image'], self.c_p['predicted_particle_positions'])
            if len(self.c_p['z-predictions']) >=1:
                self.c_p['z-predictions'] -= self.c_p['z-offset']
             

        trapped_idx = self.check_trapped()
        self.check_in_pipette(trapped_idx=trapped_idx)
        self.particle_trapped = sum(self.data_channels['particle_trapped'].get_data(10))>5
        self.particle_in_pipette = sum(self.data_channels['particle_in_pipette'].get_data(10)==2)>5
        # Add data to data channels
        self.add_prediction_to_data()
        self.data_channels['particle_trapped'].put_data(self.c_p['particle_trapped'])
        self.check_multiple_trapped() 

    def run(self):
        """
        Currently this is primarily used to test different routines for automation.
        """

        prev_time = time()
        while self.c_p['program_running']:

            # First the frame is tracked and the (laser) positions are updated
            self.c_p['loop_execution_time'] = time() - prev_time 
            prev_time = time()
            if self.c_p['move_piezo_2_target']:
                if self.move_piezos_2_saved_positions(self.c_p['piezo_target_positions']):
                    self.c_p['move_piezo_2_target'] = False

            self.update_lasers_position_from_PSDs()
            if self.c_p['image'] is None:
                continue
            if self.c_p['tracking_on']:
                self.analyze_frame()                                
            else:
                sleep(0.1)
                continue
            
            # If the frame was tracked then we may proceed to the automation routines if any of
            # them are toggled on.
            if self.c_p['calibration_running']:
                if self.auto_calibration():
                    self.c_p['calibration_running'] = False
                    self.c_p['calibration_start'] = True
                continue

            if self.c_p['centering_on']:
                center = ([self.c_p['laser_position'][0] - self.c_p['AOI'][0],
                           self.c_p['laser_position'][1] - self.c_p['AOI'][2]])
                self.center_on_particle(center)
                if self.c_p['particle_trapped']:
                    self.c_p['centering_on'] = False
                    if self.moving_to_trap:
                        self.moving_to_trap = False
                        self.motor_controller.stop_moving()

            if self.c_p['search_and_trap']: # Changed from elif to if
                if self.find_and_trap_particle():
                    self.c_p['search_and_trap'] = False

            elif self.c_p['find_laser_position']:
                self.find_true_laser_position()
                self.c_p['find_laser_position'] = False

            elif self.c_p['move_to_pipette_tip']:
                 if self.move_to_pipette_tip(move_to_particle=self.c_p['particle_in_pipette']):
                    self.c_p['move_to_pipette_tip'] = False
                     
            elif self.c_p['touch_particles']:
                message = self.touch_trapped_2_pieptte_particle()
                print(message)
                self.limits_found = [False, False]
                self.protocol_started = False
                self.c_p['protocol_data'][0] = 0

            elif self.c_p['focus_pipette']:
                if self.z_focus_pipette():
                    self.c_p['pipette_location_chamber'][2] = self.motor_controller.get_z_position()
                    self.c_p['focus_pipette'] = False

            if self.c_p['drop_particle']:
                if self.drop_particle_and_reset_trap():
                    self.c_p['drop_particle'] = False
            
            if self.c_p['suck_into_pipette']:
                if not self.particle_trapped:
                    print("Cannot suck without a particle in the trap")
                    self.c_p['suck_into_pipette'] = False
                else:
                    res = self.suck_particle_into_pipette()
                    if res == "Finished":
                        self.c_p['suck_into_pipette'] = False
                
            if self.c_p['external_restart']:
                self.restart_experiment()
                self.c_p['external_restart'] = False

            if self.c_p['focus_z_trap_pipette']:
                # RESET THE pipette focus
                if self.particle_in_pipette and self.particle_trapped:
                    if self.z_focus(): 
                        self.c_p['focus_z_trap_pipette'] = False
                else:
                    print("Focusing pipette instead of particle in it")
                    self.c_p['focus_z_trap_pipette'] = False
                    self.c_p['focus_pipette'] = True

            if self.c_p['autonomous_experiment']:
                 self.full_auto_experiment()
            elif self.protocols[self.c_p['autonomous_experiment_type']].is_running():
                self.protocols[self.c_p['autonomous_experiment_type']].stop()

            # Stokes test part
            if self.c_p['stokes_test_running']:
                print("Stokes test called from run loop")
                self.run_stokes_test()
                continue

    def auto_calibration(self, nbr_averging_points=2000):
        """
        Protocol for automatically moving the trapped particle in a grid pattern recording the 
        laser positions and forces at the different locations.
        uses the range 3000 to 63000 on the piezos to make the grid.

        The data is stored in a 3-array, calibration_points ordered as follows:
        particle position (x,y)
        laser positions (A_x, A_y, B_x, B_y) (normalized to the sum of the PSD signals)
        force (A_x, A_y, B_x, B_y, Z)
        DAC values (A_x, A_y, B_x, B_y)
        Laser power (A, B)

        """

        # If we are at the start of a calibration, reset the grid and set the start position.
        if self.c_p['calibration_start']:

            # Here we check if we are far from the starting position. If that is the case then
            # The system moves gradually to the starting position so as not to loose the particle
            target_x = int(3000 + self.calibration_x * self.grid_step)
            target_y = int(3000 + self.calibration_y * self.grid_step)
            if not self.move_particle_to_piezo_position([target_x, target_y]):
                return False

            # We are close to the target position and can start the calibration            
            self.calibration_x = 0
            self.calibration_y = 0
            self.calibration_timer = time()
            self.PSD_A_P_sum = np.mean(
                self.data_channels['PSD_A_P_sum'].get_data(nbr_averging_points))
            self.PSD_B_P_sum = np.mean(
                self.data_channels['PSD_B_P_sum'].get_data(nbr_averging_points))
            
            self.c_p['calibration_start'] = False
            self.calibration_dir = "forwards"
            self.c_p['portenta_command_2'] = 1 # Autoaliging A
            self.c_p['piezo_B'][0] = target_x
            self.c_p['piezo_B'][1] = target_y
            print(f"""Moving to position {self.c_p['piezo_B']}, step {self.calibration_x},
                  {self.calibration_y}""")
            return False

        # Check that there is a particle in view.
        if not self.particles_in_view:
            self.calibration_particles_counter += 1
            if self.calibration_particles_counter > 7:
                print("No particles in view, cannot calibrate")
                self.c_p['calibration_start'] = True
                self.c_p['calibration_running'] = False
                self.c_p['calibration_performed'] = False
                self.c_p['portenta_command_2'] = 0 # Turning of autoaligning
                self.calibration_particles_counter = 0
                return False
        else:
            self.calibration_particles_counter = 0
        # Check that we are not moving the piezo, if we are then we cannot calibrate

        # Set the position of the piezos
        self.c_p['portenta_command_2'] = 1 # Autoalign A
        target_x = int(3000 + self.calibration_x * self.grid_step)
        target_y = int(3000 + self.calibration_y * self.grid_step)
        x_pos_nok = np.abs(target_x-self.data_channels['dac_bx'].get_data(1)[0]) > 1000
        y_pos_nok = np.abs(target_y-self.data_channels['dac_by'].get_data(1)[0]) > 1000

        if time()-self.calibration_timer < self.calibration_wait_time or x_pos_nok or y_pos_nok:
            print(f"Time not elapsed {x_pos_nok} {y_pos_nok} \
                  {time()-self.calibration_timer} < {self.calibration_wait_time}")
            if x_pos_nok or y_pos_nok:
                print(f"Moving to position {target_x}, {target_y}")
                self.move_particle_to_piezo_position([target_x, target_y])
            return False
        
        # Update time to current time
        self.calibration_timer = time()
        # Record the data
        # Particle position in image
        x_position = ((self.c_p['Trapped_particle_position'][0] + self.c_p['AOI'][0])
                      * self.c_p['microns_per_pix'])
        y_position = ((self.c_p['Trapped_particle_position'][1] + self.c_p['AOI'][2])
                      * self.c_p['microns_per_pix'])

        cx = self.calibration_x
        cy = self.calibration_y
        self.c_p['calibration_points'][cx, cy, 0] = x_position
        self.c_p['calibration_points'][cx, cy, 1] = y_position

        # Laser positions
        self.c_p['calibration_points'][cx, cy, 2] = np.mean(
            self.data_channels['PSD_A_P_X'].get_data(nbr_averging_points)) / self.PSD_A_P_sum 
        self.c_p['calibration_points'][cx, cy, 3] = np.mean(
            self.data_channels['PSD_A_P_Y'].get_data(nbr_averging_points)) / self.PSD_A_P_sum 
        self.c_p['calibration_points'][cx, cy, 4] = np.mean(
            self.data_channels['PSD_B_P_X'].get_data(nbr_averging_points)) / self.PSD_B_P_sum
        self.c_p['calibration_points'][cx, cy, 5] = np.mean(
            self.data_channels['PSD_B_P_Y'].get_data(nbr_averging_points)) / self.PSD_B_P_sum

        # Force readings
        self.c_p['calibration_points'][cx, cy, 6] = np.mean(
            self.data_channels['PSD_A_F_X'].get_data(nbr_averging_points))
        self.c_p['calibration_points'][cx, cy, 7] = np.mean(
            self.data_channels['PSD_A_F_Y'].get_data(nbr_averging_points))
        self.c_p['calibration_points'][cx, cy, 8] = np.mean(
            self.data_channels['PSD_B_F_X'].get_data(nbr_averging_points))
        self.c_p['calibration_points'][cx, cy, 9] = np.mean(
            self.data_channels['PSD_B_F_Y'].get_data(nbr_averging_points))
        self.c_p['calibration_points'][cx, cy, 10] = np.mean(
            self.data_channels['F_total_Z'].get_data(nbr_averging_points))

        # DAC values
        self.c_p['calibration_points'][cx, cy, 11] = self.data_channels['dac_ax'].get_data(1)[0]
        self.c_p['calibration_points'][cx, cy, 12] = self.data_channels['dac_ay'].get_data(1)[0]
        self.c_p['calibration_points'][cx, cy, 13] = self.data_channels['dac_bx'].get_data(1)[0]
        self.c_p['calibration_points'][cx, cy, 14] = self.data_channels['dac_by'].get_data(1)[0]
        self.c_p['calibration_points'][cx, cy, 15] = \
            self.data_channels['Laser_A_power'].get_data(1)[0]
        self.c_p['calibration_points'][cx, cy, 16] = \
            self.data_channels['Laser_B_power'].get_data(1)[0]

        # Move to the next position
        if self.calibration_dir== "forwards" and cx < self.c_p['grid_size']-1:
            cx += 1

        elif self.calibration_dir == "forwards" and cx >= self.c_p['grid_size']-1:
            cy += 1
            self.calibration_dir = "backwards"
        elif self.calibration_dir == "backwards" and cx > 0:
            cx -= 1
        elif self.calibration_dir == "backwards" and cx <= 0:
            cy += 1
            self.calibration_dir = "forwards"

            # If we have reached the end, save the data and return True
        if cy >= self.c_p['grid_size']:
            # Save the calibration array
            calibration_file = self.c_p['recording_path']+'/'+self.c_p['filename']+'auto_calib'
            print(f"Calibration finished, saving to file{calibration_file}")
            np.save(calibration_file, self.c_p['calibration_points'])
            self.c_p['calibration_performed'] = True

            # Create new interpolators for laser position
            # This is way too slow, find a better way to do it

            self.APX_coeffs = self.calculate_polynomial_coefficients(
                self.c_p['calibration_points'][:,:,2],
                self.c_p['calibration_points'][:,:,3],
                self.c_p['calibration_points'][:,:,0])
            self.APY_coeffs = self.calculate_polynomial_coefficients(
                self.c_p['calibration_points'][:,:,2],
                self.c_p['calibration_points'][:,:,3],
                self.c_p['calibration_points'][:,:,1])
            self.BPX_coeffs = self.calculate_polynomial_coefficients(
                self.c_p['calibration_points'][:,:,4],
                self.c_p['calibration_points'][:,:,5],
                self.c_p['calibration_points'][:,:,0])
            self.BPY_coeffs = self.calculate_polynomial_coefficients(
                self.c_p['calibration_points'][:,:,4],
                self.c_p['calibration_points'][:,:,5],
                self.c_p['calibration_points'][:,:,1])
            self.calibration_x = cx
            self.calibration_y = cy
            return True
        self.calibration_x = cx
        self.calibration_y = cy

        if (np.abs(target_x-self.data_channels['dac_bx'].get_data(1)[0]) > 5000
            or np.abs(target_y-self.data_channels['dac_by'].get_data(1)[0]) > 5000):
            # We are far from the target position, move to the target position
            self.move_particle_to_piezo_position([target_x, target_y])
        else:
            self.c_p['piezo_B'][0] = target_x
            self.c_p['piezo_B'][1] = target_y
        print(f"Moving to position {self.c_p['piezo_B']}, step {cx}, {self.calibration_y}")
        return False

    def calculate_polynomial_coefficients(self,x,y,z):
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        # Create the design matrix for a 2D quadratic (degree 2) polynomial:
        # f(x, y) = a + b*x + c*y + d*x^2 + e*xy + f*y^2
        X = np.column_stack([np.ones(x.shape), x, y, x**2, x*y, y**2])

        # Fit the polynomial using least squares
        coeffs, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
        return coeffs

    def poly2d(self,coeffs, x, y):
        return (coeffs[0] + coeffs[1]*x + coeffs[2]*y + 
                coeffs[3]*x**2 + coeffs[4]*x*y + coeffs[5]*y**2)
