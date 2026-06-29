"""
This file contains the system specific parameters such as which ports are used and calibration 
factors.
Provides functions for loading these configurations which are called from the interface once this 
is initiated. Update the configs below to fit your own system.
------------------------------------------
Functions:

- get_defualt_config: Loads a default configuration of the SmartTrap system. Defines ports,
calibration parameters and paths.
- get_data_dicitonary_smarttrap: Defines the data channels used by the SmartTrap. Here you can add
more channels with data if need be. These can be saved and plotted.
- create_devices: Creates the objects for controlling various hardware devices of the SmartTrap.
"""

import numpy as np
from control_parameters import DataChannel


def get_defualt_config():
    """
    Here is the default configuration of the smartTrap system
    """
    config = {
        ############### Ports #########################
        'COM_port': 'COM8', # COM port used with the tweezers controller
        'objective_stepper_port': 'COM10',
        'laser_A_port':'COM12',
        'laser_B_port':'COM11',
        'pump_adress': 'COM5',
        'valve_adress': 'COM3',
        'pipette_pump_adress': 'COM6',

        # Paths
        'recording_path': 'TestData/',
        'yolo_path': "NeuralNetworks/YOLOV5Weights.pt",
        'default_z_model_path': "NeuralNetworks/Z_model_large_range.pth",

        'camera_type': 'Basler', # Implemented options are Basler and Thorlabs

        ############ Calibration parameters##################
        # Generally the calibration parameters are system specific, dependent on factors like
        # pixel size of the camera and the alignment.

        # Position calibration
        'laser_a_transfer_matrix': np.array([ 13.62547902 , 0.39582976, -0.98140442, 
                                                 13.65848671]), 
        'laser_b_transfer_matrix': np.array([ -13.75365959 , -2.95635977,-2.87762914,
                                                16.21314373]), 
        'PSD_to_pos': [14.08,13.80,13.89,13.04],  

        # Force calibration
        'PSD_to_force': np.array([0.0173, 0.0164, 0.0178, 0.0182]), # Calibration - system specific
        'Photodiode_sum_to_force': [1200,-700,210], # The calibration factor for the photodiode/PSD sum channel to force
        'reflection_A': 0.0693, # Used to calculate the actual laser power in the sample.
        'reflection_B': 0.0816,
        'sum2power_A': 0.00692*94/135,
        'sum2power_B': 0.00682*94/135,
        'reflection_fac': 1.0057, # Factor relatets to the compensation when calculating the true sum readings.

        # Camera and motor calibration 
        'microns_per_pix': 1/(21.022), #*(4.8/2.74), # Note this parameter is system dependent! 
        'ticks_per_micron': 6.24, # How many ticks per micron
        'microns_per_tick': 1/6.24, # How many microns per tick
        'ticks_per_pixel': 6.24/(18.28*1.15), # How many pixels per micron
        # other
        'valves_used': [1,3], # indices of the valves are used
    }
    return config

def get_data_dicitonary_smarttrap():
    """
    Creates and returns a dictionary of data channels used in the control parameters.
    Each key in the returned dictionary is a string representing the channel name, and each value 
    is a DataChannel object initialized with the channel's name, unit, a default value list ([0]),
    and a boolean indicating if the channel is saved.
    Returns:
        dict: A dictionary mapping channel names to their corresponding DataChannel objects.
    """

    # move these to the config.
    data = [
    ['Time', 'Seconds', False], # Time measured by the computer.
    ['prediction_time','microseconds', True],
    ['particle_trapped','(bool)', False],
    ['trapped_particle_x_position','microns', True],
    ['trapped_particle_y_position','microns', True],
    ['trapped_particle_z_position','microns', True],
    ['trapped_x_force', 'pN', True],
    ['trapped_y_force', 'pN', True],
    ['trapped_particle_radii','microns', True],
    ['particle_in_pipette','(boolish)', False], # Can take values,1,2,0 - 1 No particle in pipette, 
    # 2- particle in pipette, 0 no pipette visible
    ['pipette_particle_x_position','microns', True],
    ['pipette_particle_y_position','microns', True],
    ['pipette_particle_z_position','microns', True],
    ['pipette_particle_radii','microns', True],
    ['Temperature', 'Celsius', False],
    ['Motor_x_pos', 'ticks', True],
    ['Motor_y_pos','ticks', True],
    ['Motor_z_pos', 'ticks', True],
    ['Motor_x_speed','microns/s', True],
    ['Motor_y_speed','microns/s', True],
    ['Motor_z_speed','microns/s', True],
    ['Motor time','microseconds', True],
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

    data_dict = {}
    for channel in data:
        data_dict[channel[0]] = DataChannel(channel[0], channel[1], [0], channel[2])
    return data_dict


def create_devices(c_p, data_channels):
    """
    Creates and initializes the various hardware controllers used in the optical tweezers
    setup, including cameras, object tracking, motor controls, lasers, and microfluidics
    controllers. This method sets up the necessary connections and configurations for each
    controller based on the provided control parameters (`c_p`). It handles exceptions
    that may arise during the initialization of each controller, ensuring that the system
    can continue to operate even if some components fail to initialize.

    # NOTE that it is in here changes are made to use different controllers
    """

    # Set up the camera
    camera = None
    try:
        # Cameras from two manufacturors are currently implemented, Thorlabs and Basler.
        # They use different classes. To change manufacturor change the camera_type in the
        # control parameters
        if c_p['camera_type'] == "Thorlabs":
            print("Thorlabs camera selected")
            from thorlabs_scientific_cameras import ThorlabsScientificCamera as TSC
            camera = TSC()
        else:
            print("Basler camera selected")
            from basler_cameras import BaslerCamera
            camera = BaslerCamera()            
        c_p['camera_width'] = camera.get_sensor_size()[0]
        c_p['camera_height'] = camera.get_sensor_size()[1]
    except Exception as E:
        print(f"Camera error!\n{E}")


    # Set up the object tracker
    from smarttrap_tracker import ObjectTrackerYOLO, ParticleCNN  # noqa: F401
    object_tracker = ObjectTrackerYOLO(
        YOLO_model_path=c_p['yolo_path'],
        z_model_path=c_p['default_z_model_path'],
        particle_size_limits = [1.3/c_p['microns_per_pix'], 7/c_p['microns_per_pix']],
        )

    # Set up the motor controllers
    from smarttrap_motors import ObjectiveMotor, SmartTrapMotor

    motor_controller = SmartTrapMotor(c_p, data_channels)
    
    objective_motor = ObjectiveMotor()
    objective_motor.connect(c_p['objective_stepper_port'])

    from OSTech_laser_controller import OSTechLaserController
    laser_A = OSTechLaserController()
    laser_A.connect(c_p['laser_A_port'])

    laser_B = OSTechLaserController()
    laser_B.connect(c_p['laser_A_port'])

    # Set up the microfluidics controllers
    try:

        from elvesys_pump import (
            ElvesysMicrofluidicsController, MUXWireValveController, PipettePump)
        microfluidics_controller = ElvesysMicrofluidicsController()
        microfluidics_controller.connect(c_p['pump_adress'])
    except Exception as E:
        print(E)
        print("Could not connect to the microfluidics pump")
        from microfluidics_controllers import TestMicrofluidicsController
        microfluidics_controller = TestMicrofluidicsController()
    try:
        valve_controller = MUXWireValveController()
        valve_controller.connect(c_p['valve_adress'])
    except Exception as E:        
        from microfluidics_controllers import TestValveController
        print(E)
        valve_controller = TestValveController()
        print("Could not connect to the valve controller")
    try:
        pipette_pump = PipettePump()    
        pipette_pump.connect(c_p['pipette_pump_adress'])
    except Exception as E:
        print(E)
        from microfluidics_controllers import TestPipettePump
        pipette_pump = TestPipettePump()
        print("Could not connect to the pipette pressure controller")
    
    from smarttrap_driver import SmartTrapDriver
    instrument_driver = SmartTrapDriver(c_p, data_channels)

    return camera, object_tracker, motor_controller, objective_motor, laser_A, laser_B, microfluidics_controller, valve_controller, pipette_pump, instrument_driver