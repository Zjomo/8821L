import numpy as np
from queue import Queue

from PyQt6.QtWidgets import (
 QCheckBox, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import QTimer


class ControlParametersViewer(QWidget):
    """
    Window for displaying the current values of the data channels. Used primarily when testing
    new features and one needs to check values of many parameters but don't want to print directly
    to the command line.
    """
    def __init__(self, c_p):
        super().__init__()
        self.c_p = c_p
        self.setWindowTitle("Control parameters viewer")
        self.resize(800, 800)

        self.vBox = QVBoxLayout()
        self.CreateTable()
        self.vBox.addWidget(self.table)

        self.timer = QTimer()
        self.timer.setInterval(500) # sets the fps of the timer
        self.timer.timeout.connect(self.set_data)

        self.timer.start()
        self.setLayout(self.vBox)

    def CreateTable(self):

        self.table = QTableWidget(len(self.c_p), 2)  # +2 now for the fps row and a buffer
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])

        for idx, parameter in enumerate(self.c_p):
            self.table.setItem(idx, 0, QTableWidgetItem(f"{parameter}"))
            self.table.setItem(idx, 1, QTableWidgetItem(f"{self.c_p[parameter]}"))

    def set_data(self):
        for idx, parameter in enumerate(self.c_p):
            self.table.setItem(idx, 1, QTableWidgetItem(f"{self.c_p[parameter]}"))
            


def default_c_p():
    """
    Initiates the control parameters to default values.

    Returns
    -------
    c_p : TYPE
        DESCRIPTION. The control parameters. Dictionary containing the important parameters and
        values that need to be shared between different threads or easily monitored from the control
        parameters window.
    """
    c_p = {
            # General c_p of the UI
           'program_running': True,
           'mouse_params': [0, 0, 0, 0, 0, 0],
           'click_tools': [],
           'central_circle_on': False,

           # Camera and image parameters
           'image': np.ones([500, 500,1]),
           'image_idx': 0, # Index of snapshot image, used in naming.
           'color': "mono",  # Options are mono and color
           'new_settings_camera': [False, None],
           'camera_width': 1920, # Default camera width, will automatically update when connecting
           'camera_height': 1080,
           'recording': False,
           'exposure_time': 5000,
           'fps': 50,  # Frames per second of camera, measured
           'target_frame_rate': 50, # Target frame rate of the camera, can limit the framerate.
           'filename': 'measurement_',
           'video_name': 'Video',
           'video_format': 'avi',
           'available_video_formats': ['avi','mp4','npy'],
           'image_format': 'png',
           'available_image_formats': ['png','jpg','npy'],
           'data_file_format': 'npy',
           'available_data_formats': ['npy','csv','h5','mat'],
           'image_gain': 1, 
           'image_offset': 0,
           'AOI':[0,1000,0,1000], # Area of interest of camera
           'target_AOI':[0,1000,0,1000], # Target area of interest of camera 
           # (not all cameras accept all AOIs even if they fit the sensor)
           'recording_path': 'TestData/',           
           'bitrate': '30000000', # Bitrate of video to be saved
           'frame_queue': Queue(maxsize=2_000_000),  # Frame buffer essentially
           'image_scale': 1,

           'offset_channels':[
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_A_F_X', 'PSD_A_F_Y', 'PSD_A_F_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',
                            'PSD_B_F_X', 'PSD_B_F_Y', 'PSD_B_F_sum',
                            'Photodiode_A','Photodiode_B',
                            'message',
                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos'],

            # These are the channels which are sampled once per sample cycle. Default 
            'single_sample_channels':[
                            'Motor_x_pos', 'Motor_y_pos', 'Motor_z_pos',
                            'message', # This is used for debugging, i.e sending data from the
                            # controller to the computer for testing new functionality.
                            'dac_ax','dac_ay','dac_bx','dac_by',
                            'PSD_Force_A_saved',
            ],

            # These channels are sampled multiple times per sample cycle.
            'multi_sample_channels':[
                            'PSD_A_P_X', 'PSD_A_P_Y', 'PSD_A_P_sum',
                            'PSD_A_F_X', 'PSD_A_F_Y', 'PSD_A_F_sum',
                            'PSD_B_P_X', 'PSD_B_P_Y', 'PSD_B_P_sum',
                            'PSD_B_F_X', 'PSD_B_F_Y', 'PSD_B_F_sum',
                            'Photodiode_A','Photodiode_B',
                            'T_time','Time_micros_low','Time_micros_high', # Moved this up
                            ],

            # These channels have values calculated from the "mulit sample channels" 
            # (i.e force converted from PSD reading)
            'derived_PSD_channels': ['F_A_X','F_A_Y','F_B_X','F_B_Y','F_A_Z','F_B_Z',
                                     'F_total_X','F_total_Y','F_total_Z',
                                     'Position_A_X', 'Position_A_Y','Position_B_X','Position_B_Y',
                                     'Position_X','Position_Y'
                                     ],
            'prediction_channels': ['trapped_particle_x_position', 'trapped_particle_y_position',
                                    'trapped_particle_z_position', 'pipette_particle_x_position', 
                                    'pipette_particle_y_position', 'pipette_particle_z_position',
                                    'pipette_particle_radii', 'trapped_particle_radii', 
                                    'particle_trapped', 'particle_in_pipette', 'prediction_time',
                                    'trapped_x_force', 'trapped_y_force'],

            'save_idx': 0, # Index of the saved data
            'averaging_interval': 1_000, # How many samples to average over in the data channels
            # window

            # Piezo actuator outputs (for moving the lasers)
            # Target values for the piezos, in the range 0 to 2^15
            'piezo_A': np.uint16([32768, 32768]), 
            'piezo_B': np.uint16([32768, 32768]),
            'portenta_command_1': 0, # Command to send to the portenta, zero force etc.
            # Command to send to the portenta,  laser feedback control controls:
            # 0=> no autoalign, 1=> autoalign on A, 2=> autoalign on B
            'portenta_command_2': 0, # 

            'PSD_means':  np.uint16([0,0,0,0]), # Means of the PSD channels
            'PSD_force_means':  np.array([0,0,0,0]), # Means of the PSD channels
            'PSD_position_means': np.array([0,0,0,0]), # Means of the PSD channels

            # Deep learning tracking
            'tracking_on': True,
            'z-tracking': True,
            'draw_pipette': False, # If the pipette should be drawn in the image
            'draw_particles': False, # If the particles should be drawn in the image
            'draw_z_text': False, # If the z-position of the particles should be drawn in the image
            'draw_TnP_particles': False, # If the particles in the trap and pipette should be drawn
            # in the image
            'draw_lasers': False, # If the laser positions should be drawn in the image
            'draw_force': False,

            'z-predictions': np.array([]),
            'z-threshold': 8, # Threshold for the z-tracking
            'z-offset': 0, # Offset for the z-tracking used to set a focused particles z-position
            # to 0.
            'predicted_particle_radii': np.array([]),
 
            'pipette_in_frame': False,
            'pipette_located': False,
            'particle_trapped': False,
            'particle_in_pipette': False,
            'accurate_tip_detection_needed': True,
            'multiple_particles_trapped': False,
            'pipette_tilt': 0, # tilt of the pipette, used to find the tip position.
            'Trapped_particle_position': [0,0,0,0], # Position of the trapped particle in the image;
            # x,y,z, radii in pixels
            'pipette_particle_location': [1200,1200,0,0], # Location of the pipette particle in the
             # image,
            'pipette_location': [0,0,0,0], # Location of the pipette in the image,x,y position of 
            # tip as well as width and height of pipette(in this order).
            'pipette_tip_location': [0,0], # Location of the pipette tip in the image


            # Autocontroller parameters
            'loop_execution_time': 0, # Time it takes to execute the loop
            'autocontroller_current_step': 'checking_pipette', # The current step of the 
            # autocontroller
            'autonomous_experiment_type': 'DNA pulling', # The type of autonomous experiment to be
            # performed, alternatives are molecule_stretching and electrostatic(currently)
            'autonomous_experiment_types': ['DNA pulling','Elctrostatic repulsion',
                                            'Particle characterization', 'RBC Stretching'],

            'autonomous_experiment': False, # If this is toggled we will try to do a full autonomous
            # experiment.
            'autonomous_experiment_states': ['checking_pipette','focusing_pipette',
                                             'searching_for_particle_1','sucking_into_pipette',
                                             'searching_for_particle_2','move_to_pipette_tip',
                                             'touching_particles'],
            'external_restart': False,

            'centering_on': False,
            'trap_particle': False,
            'search_and_trap': False,
            'focus_z_trap_pipette': False, # Focus the particle in the trap with the one in the
            # pipette parameters for focusing pipette
            'drop_particle': False,
            'center_pipette': False,
            'suck_into_pipette': False, # used to suck particle into pipette.
            'move_to_pipette_tip': False,
            'move_avoiding_particles': False,
            'pipette_z_found': False,
            'move_piezo_2_target': False,
            'touch_particles': False,
            'touch_counter': 0, # Keeps track of how many times we have tried to attach a molecule
             #  and failed. Needs to be reset when we have successfully attached a molecule or when
             # starting on a new molecule drop_particle': False, # If set to true the piezos will 
             # move to drop the particle(s) in the trap. Is a control parameter to allow for easy
             # monitoring and debugging of new protocols.
            'stretch_molecule': False,
            'move_particle2pipette': False,
            'find_laser_position': False, # Updates the laser position to the current closest 
            # particle
            'focus_pipette': False, # Focus the pipette by optimizing sharpness
            'pipette_focus_startup': True,
            'pipette_sharpnesses': [],
            'pipette_sharpness_Z_pos': [],

            'piezo_target_positions': [0,0,0,0], # A position rading of the PSDs can be saved and
             #  moved to on the position sensors, by default this is position 0,0 on both lasers.
            'laser_position_A': [2660, 1502.3255814], # Default
            'laser_position_B': [2660, 1502.3255814], # Default,
            'laser_position_A_predicted': [2660, 1502.3255814],
            'laser_position_B_predicted': [2660, 1502.3255814],
            'laser_position': [2660, 1502.3255814], # Updated as the average of position A and B
            
            # Position of the capillaries that push out beads 1 and 2, 1 being the beads going to 
            # # the pipette and 2 the beads going to the trap
            'capillary_1_position': [0,0,0], 
            'capillary_2_position': [0,0,0],
            'pipette_location_chamber': [0,0,0], # Location of the pipette in the chamber,
             # motor coordinates
            'particle_type': 1, 
            'capillary_1_fluidics_channel': [0, 30, 3], 
            # first number is the channel numnber(0,1 or 2) and the second is the pressure when
            # pushing particles[mbar]. The last is the index of the valve used.
            'capillary_2_fluidics_channel': [2, 30, 3],
            'central_fluidics_channel': [1, 50,1], # The central channel where the pipette is.            

            # Autonomous protocol parameters parmeters
            'protocol_limits_dac': [20_000, 40_000], # The limits of the protocol in DAC units,
            'measurement_time': 600, # Time(seconds) during which experiments will be performed 
            # on a single particle         

            # Minitweezers controller parameters            
            'blue_led': 0, # Wheter the blue led is on or off, 0 for on and 1 for off

            # Minitweezers protocols parameters
            'protocol_running': False,
            'protocol_type': 'Constant speed',
            'protocol_data': np.uint8(np.zeros(13)),            

            # Automatic position calibration parameters
            'grid_size': 10, # Number of datapoints used when moving in the grid
            'calibration_points': np.zeros([10,10,17]),
            'calibration_start': True, # Used to tell if the calibration should be reset 
            # (started from scratch).
            'calibration_running': False,
            'calibration_performed': False, # Sets to true when a new calibration has been performed
            # and this should be updated in the read-portenta thread

            # Red blood cells (RBC) experiment parameters
            'laser_power_protocol_running': False,
            'power_protocol_currents': [ 
                                    [87,88,20], # 5 mW / laser
                                    [96,98,5], # 10 mW per laser
                                    [87,88,10],
                                    [110,116,5], # 20
                                    [87,88,10],
                                    [137,147,5], # 42
                                    [87,88,10],
                                    [162,175,5], # 60
                                    [87,88,10],
                                    [187,203,5], # 80
                                    [87,88,20],
                                ], # Ordedered as [laser_A_current, laser_B_current, duration]

            # Laser parameters
            'laser_A_current': 249, # Current in mA
            'laser_B_current': 235, # Current in mA
            'laser_A_current_current': 0, # Current in mA
            'laser_B_current_current': 0, # Current in mA
            'laser_A_on': False,
            'laser_B_on': False,
            # Microfluidics system  parameters
            'target_pressures': np.array([0.0, 0.0 , 0.0, 0.0]),
            'current_pressures': np.array([0.0, 0.0 , 0.0, 0.0]),
            # Valve parameters
            'valve_controller': None,
            'valves_controller_connected': False,
            'valves_open': [False,False,False,False,False,False,False,False],

            # Parameters for the electronics pump for the pipette
            'pipette_pump_on': False,
            'pipette_pump_target_power': 8,
            'pipette_pump_current_power': 0,

            # Minitweezers motors
            'motor_x_target_speed': 0,
            'motor_y_target_speed': 0,
            'motor_z_target_speed': 0,
            'minitweezers_goto_speed': 10_000,
            'minitweezers_target_pos': [0,0,0], # Should these start at 0?
            'minitweezers_target_speed': [0,0,0],
            'motor_travel_speed': [2_000, 2_000], # 5000 was somewhat high Speed of move to location.
            'move_to_location': False, # Deterimnes if  the motors move to a location rather than
            #  listen to the speed
            'saved_positions':[],

            # Stokes test parameters. Note that stokes test is used during calibration
            'stokes_left_pos': [0,0,0],
            'stokes_right_pos': [0,0,0],
            'stokes_up_pos': [0,0,0],
            'stokes_down_pos': [0,0,0],
            'stokes_center_pos': [0,0,0],
            'stokes_size_threshold': 2,
            'stokes_stage': "stokes_startup",
            'stokes_test_running': False,
            'stokes_test_step': "startup",
            'stokes_max_move_count': 3,

            'steppers_connected': [False, False, False], # 
        }
    return c_p


from dataclasses import dataclass
@dataclass
class DataChannel:
    """
    Class for storing and retrieving data for a specific channel.
    1. name: Name of the data channel.
    2. unit: Unit of the data channel.
    3. data: Numpy array to store the data.
    4. saving_toggled: Boolean to indicate if the data channel should be saved.
    """
    name: str
    unit: str
    data: np.array
    saving_toggled: bool = True
    sample_rate: float = 15625 # Default sample rate for any channel
    max_len: int = 10_000_000 # 1_000_000 default
    index: int = 0
    full: bool = False
    max_retrivable: int = 1 # number of datapoints which have been saved.

    def __post_init__(self):
        # Preallocate memory for the maximum length
        self.data = np.zeros(self.max_len)

    def put_data(self, d):
        try:
            if len(d) > self.max_len:
                return
        except TypeError:
            d = [d]

        if self.index + len(d) >= self.max_len:
            end_points = self.max_len - self.index
            self.data[self.index:] = d[:end_points]
            self.data[:len(d) - end_points] = d[end_points:]
            self.full = True
            self.index = (self.index + len(d)) % self.max_len
            self.max_retrivable = self.max_len
        else:
            self.data[self.index:self.index + len(d)] = d
            self.index += len(d)
            self.max_retrivable = max(self.index, self.max_retrivable)

    def get_data(self, nbr_points, offset=0):
        offset = max(0, offset)
        nbr_points = min(nbr_points, self.max_retrivable)
        if nbr_points <= self.index-offset:
            return self.data[self.index-nbr_points-offset:self.index-offset]
        else:
            return np.concatenate(
                [self.data[self.index-nbr_points-offset:], self.data[:self.index-offset]])

    def get_data_spaced(self, nbr_points, spacing=1):
        nbr_points = min(nbr_points, self.max_retrivable)
        final = self.index
        final -= final % spacing
        start = final - (final % spacing) - (nbr_points * spacing)
        if start >= 0:
            return self.data[start:final:spacing]
        else:
            last = (nbr_points * spacing + start) % self.max_len
            return np.concatenate([self.data[start::spacing], self.data[:last:spacing]])

def get_data_dicitonary_smarttrap():
    """
    Creates and returns a dictionary of data channels used in the control parameters.
    Each key in the returned dictionary is a string representing the channel name, and each value 
    is a DataChannel object initialized with the channel's name, unit, a default value list ([0]),
    and a boolean indicating if the channel is saved.
    Returns:
        dict: A dictionary mapping channel names to their corresponding DataChannel objects.
    """

    data = [
    # ['Time', 'Seconds', False], # Time measured by the computer.
    # ['prediction_time','microseconds', True],
    # ['particle_trapped','boolean', False],
    # ['trapped_particle_x_position','microns', True],
    # ['trapped_particle_y_position','microns', True],
    # ['trapped_particle_z_position','microns', True],
    # ['trapped_x_force', 'pN', True],
    # ['trapped_y_force', 'pN', True],
    # ['trapped_particle_radii','microns', True],
    # ['particle_in_pipette','bool', False], # Can take values,1,2,0 - 1 No particle in pipette, 
    # # 2- particle in pipette, 0 no pipette visible
    # ['pipette_particle_x_position','microns', True],
    # ['pipette_particle_y_position','microns', True],
    # ['pipette_particle_z_position','microns', True],
    # ['pipette_particle_radii','microns', True],
    # ['Temperature', 'Celsius', False],
    # ['Motor_x_pos', 'ticks', True],
    # ['Motor_y_pos','ticks', True],
    # ['Motor_z_pos', 'ticks', True],
    # ['Motor_x_speed','microns/s', True],
    # ['Motor_y_speed','microns/s', True],
    # ['Motor_z_speed','microns/s', True],
    # ['Motor time','microseconds', True],
    ['PSD_A_P_X','bits', True],
    ['PSD_A_P_Y','bits', True],
    ['PSD_A_P_sum','bits', True],
    ['PSD_A_F_X', 'bits', True],
    ['PSD_A_F_Y','bits', True],
    ['PSD_A_F_sum','bits', True],
    ['PSD_A_F_sum_compensated','bits', True],
    ['PSD_B_P_X', 'bits', True],
    ['PSD_B_P_Y','bits', True],
    ['PSD_B_P_sum','bits', True],
    ['PSD_B_F_X', 'bits', True],
    ['PSD_B_F_Y','bits', True],
    ['PSD_B_F_sum','bits', True],
    ['PSD_B_F_sum_compensated','bits', True],
    ['Photodiode_A','bits', True],
    ['Photodiode_B','bits', True],
    ['Laser_A_power','mW', True],
    ['Laser_B_power','mW', True],
    ['T_time','microseconds', True], # Time measured on the controller
    ['Time_micros_high','microseconds', False],
    ['Time_micros_low','microseconds', False],
    ['F_A_X','pN', False],
    ['F_A_Y','pN', False],
    ['F_B_X','pN', False],
    ['F_B_Y','pN', False],
    ['F_A_Z','pN', False],
    ['F_B_Z','pN', False],
    ['F_total_X','pN', True],
    ['F_total_Y','pN', True],
    ['F_total_Z','pN', True],
    ['Position_A_X','microns', False],
    ['Position_A_Y','microns', False],
    ['Position_B_X','microns', False],
    ['Position_B_Y','microns', False],
    ['Position_X','microns', True],
    ['Position_Y','microns', True],
    ['PSD_Force_A_saved','pN', False],
    ['Photodiode/PSD SUM A','a.u.', False],
    ['Photodiode/PSD SUM B','a.u.', False],
    ['message','string', False],
    ['dac_ax','bits', False],
    ['dac_ay','bits', False],
    ['dac_bx','bits', False],
    ['dac_by','bits', False],
    ]

    # These channels are not sampled as frequently which is relevant when plotting and saving data

    data_dict = {}
    for channel in data:
        data_dict[channel[0]] = DataChannel(channel[0], channel[1], [0], channel[2])

    sample_rate = 1116
    data_motor = [
    ['Motor_x_pos', 'ticks', True],
    ['Motor_y_pos','ticks', True],
    ['Motor_z_pos', 'ticks', True],
    ['Motor_x_speed','microns/s', True],
    ['Motor_y_speed','microns/s', True],
    ['Motor_z_speed','microns/s', True],
    ['Motor time','microseconds', True],
    ]
    for channel in data_motor:
        data_dict[channel[0]] = DataChannel(channel[0], channel[1], [0], channel[2],sample_rate)
    
    # TODO move this to a separate data dictionary
    sample_rate_tracking = 30
    data_tracked = [
        ['prediction_time','microseconds', True],
        ['particle_trapped','boolean', False],
        ['trapped_particle_x_position','microns', True],
        ['trapped_particle_y_position','microns', True],
        ['trapped_particle_z_position','microns', True],
        ['trapped_x_force', 'pN', True],
        ['trapped_y_force', 'pN', True],
        ['trapped_particle_radii','microns', True],
        ['particle_in_pipette','bool', False], # Can take values,1,2,0 - 1 No particle in pipette, 
        # 2- particle in pipette, 0 no pipette visible
        ['pipette_particle_x_position','microns', True],
        ['pipette_particle_y_position','microns', True],
        ['pipette_particle_z_position','microns', True],
        ['pipette_particle_radii','microns', True],
        ['Time', 'Seconds', False], # Time measured by the computer. Currently not used

    ]
    for channel in data_tracked:
        data_dict[channel[0]] = DataChannel(channel[0], channel[1], [0], channel[2],sample_rate_tracking)
    
    return data_dict


class CurrentValueWindow(QWidget):
    """
    Window for displaying the current values of the data channels.
    """
    def __init__(self, c_p, data_channels):
        super().__init__()
        self.c_p = c_p
        self.data_channels = data_channels
        self.setWindowTitle("Data channels viewer")
        self.resize(800, 800)

        self.vBox = QVBoxLayout()
        self.CreateTable()
        self.vBox.addWidget(self.table)        

        self.timer = QTimer()
        self.timer.setInterval(500) # sets the fps of the timer
        self.timer.timeout.connect(self.set_data)

        self.timer.start()
        self.setLayout(self.vBox)   

    def CreateTable(self):

        self.table = QTableWidget(len(self.data_channels)+2, 6)
        self.table.setHorizontalHeaderLabels(
            ["Channel", "Value", "Mean", "Standard dev", "Unit", "Save"])

        for idx, channel in enumerate(self.data_channels):
            self.table.setItem(idx, 0, QTableWidgetItem(f"{self.data_channels[channel].name}"))
            self.table.setItem(idx, 1, QTableWidgetItem(
                f"{self.data_channels[channel].get_data(1)}"))
            data = self.data_channels[channel].get_data(self.c_p['averaging_interval'])
            if data is not None and len(data) > 1:
                self.table.setItem(idx, 2, QTableWidgetItem(f"{np.mean(data)}"))
            else:
                self.table.setItem(idx, 2, QTableWidgetItem(""))

            if data is not None and len(data) > 1:
                self.table.setItem(idx, 3, QTableWidgetItem(f"{np.std(data)}"))
            else:
                self.table.setItem(idx, 3, QTableWidgetItem(""))
            self.table.setItem(idx,4, QTableWidgetItem(f"{self.data_channels[channel].unit}"))

            # Create a QCheckBox, connect it to the toggle_save method and add it to the table
            save_checkbox = QCheckBox()
            save_checkbox.stateChanged.connect(lambda state, ch=channel: self.toggle_save(state,
                                                                                          ch))
            save_checkbox.setChecked(self.data_channels[channel].saving_toggled)

            save_checkbox.setStyleSheet("""
                QCheckBox::indicator {
                    width: 30px;
                    height: 30px;
                }
                """)
            self.table.setCellWidget(idx, 5, save_checkbox)

        # Set the 'fps' value in the last row
        fps_idx = len(self.data_channels)
        self.table.setItem(fps_idx, 0, QTableWidgetItem("FPS"))
        self.table.setItem(fps_idx, 1, QTableWidgetItem(f"{self.c_p['fps']}"))
        for i in range(2, 6):  # Clear the rest of the columns for the FPS row
            self.table.setItem(fps_idx, i, QTableWidgetItem(""))

    def toggle_save(self, state, channel):
        """
        Toggle the saving_toggled property of the DataChannel when the checkbox is toggled.
        """
        self.data_channels[channel].saving_toggled = bool(state)
    
    def set_data(self):
        
        for idx, channel in enumerate(self.data_channels):
            data = self.data_channels[channel].get_data(self.c_p['averaging_interval'])
            self.table.setItem(idx,1, QTableWidgetItem(
                f"{self.data_channels[channel].get_data(1)}"))
            if data is not None and len(data) > 1:
                self.table.setItem(idx,2, QTableWidgetItem(f"{round(np.mean(data),6)}"))
            else:
                self.table.setItem(idx,2, QTableWidgetItem(f"{data}"))
            if data is not None and len(data) > 1:
                self.table.setItem(idx,3, QTableWidgetItem(f"{round(np.std(data),6)}"))
            else:
                self.table.setItem(idx,3, QTableWidgetItem(f"{data}"))

        # Refresh the 'fps' value in the last row
        fps_idx = len(self.data_channels)
        self.table.setItem(fps_idx, 1, QTableWidgetItem(f"{self.c_p['fps']}"))
