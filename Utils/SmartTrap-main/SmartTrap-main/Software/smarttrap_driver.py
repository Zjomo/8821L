"""
The various classes used for communications with the SmartTrap electronics controller are defined
in here.
----------------------------------------------------------------
Classes:

- PortentaCommsProcess: A separate proccess which only reads and writes to the serial USB bus of the
microcontroller (an arudino portenta) on the SmartTrap electronics controller.
- SmartTrapDriver: The driver class for the SmartTrap controller. Implements the InstrumentDriver
protocol.
"""

from multiprocessing import Process, Value, Array, Queue
from time import sleep, time

import numpy as np
import serial
from scipy.interpolate import griddata
from instrument_driver import InstrumentDriver

class PortentaCommsProcess(Process):
    """
    PortentaCommsProcess is a multiprocessing.Process subclass that manages serial communication
    between a host computer and a Portenta device (e.g., for minitweezers control). It handles
    sending data to the Portenta, reading incoming data, and placing received data into a queue
    for use by the main process. The process maintains a persistent serial connection, attempts
    reconnection on failure, and uses shared data structures for inter-process communication.
    """
    def __init__(self, portenta_data, outdata, com_port, running):
        """
        This function should get the data it is to send to the minitweezers. As well as a boolean 
        which determines if the process should continue running.

        Also takes the outdata which is a queue that the process can put data into. 
        This data is then read by the main process.

        portenta_data should be a queue while portenta_commands should be a shared array or pipe.
        """

        super().__init__()
        self.outdata = outdata
        self.portenta_data = portenta_data
        self.com_port = com_port
        self.running = running
        self.daemon  = True
        self.serial_channel = None

    def send_data_to_portenta(self):
        """
        Sends a data package via USB to the arduino portenta on the system controller.
        """

        if self.serial_channel is None:
            try:
                self.serial_channel = serial.Serial(
                    self.com_port, baudrate=5000000, timeout=.001, write_timeout=0.001)
                print("Reconnected")
            except Exception as ex:
                self.serial_channel = None
            return

        self.serial_channel.reset_output_buffer()
        self.outdata[0:2] = [123, 123]
        try:
            self.serial_channel.write(self.outdata)
        except serial.serialutil.SerialTimeoutException as e:
            pass
        except serial.serialutil.SerialException as e:
            print(f"Serial exception: {e}")
            self.serial_channel = None
        
    def read_portenta(self):
        """
        Reads the data from the serial port and returns it as a numpy array.
        """
        chunk_length = 256 # Number of 16 bit numbers sent each time.
        if self.serial_channel is None:
            return None
        try:
            bytes_to_read = self.serial_channel.in_waiting
            if bytes_to_read < chunk_length:
                return None
            raw_data = self.serial_channel.read(bytes_to_read)
        except serial.serialutil.SerialException as e:
            print(f"Serial exception: {e}")
            self.serial_channel = None
            return None

        if len(raw_data)<2 or raw_data[0] != 123 or raw_data[1] != 123:
                print('Wrong start bytes')
                return None

        # Instead of returning we put it in a buffer
        self.portenta_data.put_nowait(np.frombuffer(raw_data, dtype=np.uint16))

    def run(self):
        # Open the serial port in the child process
        try:
            self.serial_channel = serial.Serial(
                self.com_port, baudrate=5000000, timeout=.001, write_timeout=0.001)
        except Exception as ex:
            print("No comm port!")
            self.serial_channel = None
            print(ex)
        
        print('Serial channel opened')

        while self.running.value:
            self.send_data_to_portenta()
            self.read_portenta()
            sleep(1e-4)

        print("Portenta comms process stopped.")
        if self.serial_channel is not None:        
            self.serial_channel.close()
            print('Serial connection to minitweezers closed')


class SmartTrapDriver(InstrumentDriver):
    """
    This is the main thread that handles communications with the SmartTrap controller.
    It continously reads data from the controller using the PortentaCommsProcess which, by running
    in a separate process, allows it to communicate without being influenced by other processes.
    """
    def __init__(self, c_p, data_channels):        
        self.c_p = c_p
        self.portenta_data = Queue()
        self.data_channels = data_channels
        self.outdata = Array('i', np.uint8(np.zeros(48)))
        self.running_process = Value('i', 1)
        self.indata = np.uint8(np.zeros(64))
        self.start_time = time()
        self.reset_motor_pos = True
        self.connected = False
        nbr_multisamples = 1

        self.channel_array = self.c_p['single_sample_channels'].copy()
        for _ in range(nbr_multisamples):
            self.channel_array.extend(self.c_p['multi_sample_channels'])
        
        self.APX_interpolator = None
        self.APY_interpolator = None
        self.BPX_interpolator = None
        self.BPY_interpolator = None
        self.AFX_interpolator = None
        self.AFY_interpolator = None
        self.BFX_interpolator = None
        self.BFY_interpolator = None

    def prepare_portenta_commands(self):
        """
        Prepares data for sending to the portenta.
        """
        
        # Start bytes for the portenta
        self.outdata[0:2] = [123, 123]

        # Send the target position and speed
        for i in range(3):
            self.outdata[2 + i * 2] = np.uint16(
                self.c_p['minitweezers_target_pos'][i] + 32768)>> 8
            self.outdata[3 + i * 2] = np.uint16(
                self.c_p['minitweezers_target_pos'][i] + 32768)& 0xFF
            # Changed x and y position to account for different connections
            self.outdata[8 + i * 2] = np.uint16(
                self.c_p[f'motor_{["y", "x", "z"][i]}_target_speed'] + 32768) >> 8
            self.outdata[9 + i * 2] = np.uint16(
                self.c_p[f'motor_{["y", "x", "z"][i]}_target_speed'] + 32768) & 0xFF

        # Send the piezo voltages
        for i in range(2):
            self.outdata[14 + i * 2] = self.c_p['piezo_A'][i] >> 8
            self.outdata[15 + i * 2] = self.c_p['piezo_A'][i] & 0xFF
            self.outdata[18 + i * 2] = self.c_p['piezo_B'][i] >> 8
            self.outdata[19 + i * 2] = self.c_p['piezo_B'][i] & 0xFF

        self.outdata[22] = self.c_p['motor_travel_speed'][0] >> 8
        self.outdata[23] = self.c_p['motor_travel_speed'][0] & 0xFF
        self.outdata[24] = self.c_p['portenta_command_1']
        if self.c_p['portenta_command_1'] in [1,2,4,5]:
            # End Set zero values
            self.outdata[26] = self.c_p['PSD_means'][0] >> 8
            self.outdata[27] = self.c_p['PSD_means'][0] & 0xFF
            self.outdata[28] = self.c_p['PSD_means'][1] >> 8
            self.outdata[29] = self.c_p['PSD_means'][1] & 0xFF

            self.outdata[30] = self.c_p['PSD_means'][2] >> 8
            self.outdata[31] = self.c_p['PSD_means'][2] & 0xFF
            self.outdata[32] = self.c_p['PSD_means'][3] >> 8
            self.outdata[33] = self.c_p['PSD_means'][3] & 0xFF
            print("Sent zeroing data")

        if self.c_p['portenta_command_1'] == 3:
            # Sends the calibration data. I.e force conversion factors
            # Rescaling the values and converting to uint before sending
            psd_to_force_fac = 100_000
            AX = np.uint16(self.c_p['PSD_to_force'][0]*psd_to_force_fac)
            self.outdata[26] = AX >> 8
            self.outdata[27] = AX & 0xFF

            AY = np.uint16(self.c_p['PSD_to_force'][1]*psd_to_force_fac)
            self.outdata[28] = AY >> 8
            self.outdata[29] = AY & 0xFF

            BX = np.uint16(self.c_p['PSD_to_force'][2]*psd_to_force_fac)
            self.outdata[30] = BX >> 8
            self.outdata[31] = BX & 0xFF

            BY = np.uint16(self.c_p['PSD_to_force'][3]*psd_to_force_fac)
            self.outdata[32] = BY >> 8
            self.outdata[33] = BY & 0xFF
            
        self.outdata[25] = self.c_p['portenta_command_2']
        # Sometimes 
        self.c_p['portenta_command_1'] = 0
        self.outdata[34] = self.c_p['blue_led']
        self.outdata[35:] = self.c_p['protocol_data']

        # Check if we are moving to a location from the host GUI
        if (np.abs(self.c_p['motor_x_target_speed']) > 0 or
            np.abs(self.c_p['motor_y_target_speed']) > 0 or
            np.abs(self.c_p['motor_z_target_speed']) > 0):
            self.reset_target_positions()
            
            self.outdata[42] = 1 # 1 indicates that we are moving at constant speed set by the host
            # ensure that we don't do an accidental move to target location by updating current 
            # target position
        elif self.c_p['move_to_location']:
            self.outdata[42] = 0 # indicates a move to location on the portenta.
        else:
            self.outdata[42] = 1 # 1 indicates that we are moving at constant speed set by the host
        self.outdata[44] = (self.c_p['minitweezers_goto_speed'])>> 8
        self.outdata[45] = (self.c_p['minitweezers_goto_speed'])& 0xFF

    def reset_target_positions(self):
        self.c_p['minitweezers_target_pos'] = np.array(
            [self.data_channels['Motor_x_pos'].get_data(1)[0],
             self.data_channels['Motor_y_pos'].get_data(1)[0],
             self.data_channels['Motor_z_pos'].get_data(1)[0]])


    def calc_quote_fast(self, quote, channel1, channel2, chunk_length, scale=1):
        D1 = self.data_channels[channel1].get_data(chunk_length)
        D2 = self.data_channels[channel2].get_data(chunk_length).astype(float)
        D2[D2==0] = np.inf
        self.data_channels[quote].put_data(scale*D1/D2)

    def calculate_quotes_fast(self, chunk_length):        
        """
        Converts from raw readings (in bits) to positions and quote Photodiode reading / sum reading
        """
        # Calibration to get true positions, too slow to run in real time.
        if self.c_p['calibration_performed']:
            self.create_interpolators()
            self.c_p['calibration_performed'] = False

        self.calc_quote_fast('Position_A_X','PSD_A_P_X','PSD_A_P_sum',
                             chunk_length, self.c_p['PSD_to_pos'][0])
        self.calc_quote_fast('Position_A_Y','PSD_A_P_Y','PSD_A_P_sum',
                             chunk_length, self.c_p['PSD_to_pos'][1])
        self.calc_quote_fast('Position_B_X','PSD_B_P_X','PSD_B_P_sum',
                             chunk_length, self.c_p['PSD_to_pos'][2])
        self.calc_quote_fast('Position_B_Y','PSD_B_P_Y','PSD_B_P_sum',
                             chunk_length, self.c_p['PSD_to_pos'][3])
        self.calc_quote_fast('Photodiode/PSD SUM A','Photodiode_A','PSD_A_F_sum', chunk_length)
        self.calc_quote_fast('Photodiode/PSD SUM B','Photodiode_B','PSD_B_F_sum', chunk_length)

        self.data_channels['Position_X'].put_data(
            (self.data_channels['Position_A_X'].get_data(chunk_length)
             - self.data_channels['Position_B_X'].get_data(chunk_length))/2)
        self.data_channels['Position_Y'].put_data(
            (self.data_channels['Position_A_Y'].get_data(chunk_length)
             + self.data_channels['Position_B_Y'].get_data(chunk_length))/2)
        
    def calc_true_powers(self, chunk_length):
        """
        Calculates the power (in mW) of the lasers.
        Compensate for the reflections of the lasers to get the true PSD sum readings.
        ( What we would get if there were no reflections)
        """

        self.data_channels['PSD_A_F_sum_compensated'].put_data(
            (self.data_channels['PSD_A_F_sum'].get_data(chunk_length)
             - self.data_channels['PSD_B_F_sum'].get_data(chunk_length)*self.c_p['reflection_B'])
             * self.c_p['reflection_fac'])
        self.data_channels['PSD_B_F_sum_compensated'].put_data(
            (self.data_channels['PSD_B_F_sum'].get_data(chunk_length)
             - self.data_channels['PSD_A_F_sum'].get_data(chunk_length)*self.c_p['reflection_A'])
             * self.c_p['reflection_fac'])

        # Calculate the laser powers from the PSD readings
        self.data_channels['Laser_A_power'].put_data(
            self.data_channels['PSD_A_F_sum_compensated'].get_data(chunk_length)
            * self.c_p['sum2power_A'])
        self.data_channels['Laser_B_power'].put_data(
            self.data_channels['PSD_B_F_sum_compensated'].get_data(chunk_length)
            * self.c_p['sum2power_B'])

    def calc_forces(self, chunk_length):
        """
        Calculate force and put them into the data channels.
        """
        self.data_channels['F_A_X'].put_data(
            self.data_channels['PSD_A_F_X'].get_data(chunk_length) * self.c_p['PSD_to_force'][0])
        self.data_channels['F_A_Y'].put_data(
            self.data_channels['PSD_A_F_Y'].get_data(chunk_length) * self.c_p['PSD_to_force'][1])
        self.data_channels['F_B_X'].put_data(
            self.data_channels['PSD_B_F_X'].get_data(chunk_length) * self.c_p['PSD_to_force'][2])
        self.data_channels['F_B_Y'].put_data(
            self.data_channels['PSD_B_F_Y'].get_data(chunk_length) * self.c_p['PSD_to_force'][3])

        self.data_channels['F_A_Z'].put_data(
            self.data_channels['Photodiode/PSD SUM A'].get_data(chunk_length)
            * self.c_p['Photodiode_sum_to_force'][0])
        self.data_channels['F_B_Z'].put_data(
            self.data_channels['Photodiode/PSD SUM B'].get_data(chunk_length)
            * self.c_p['Photodiode_sum_to_force'][1])

        self.data_channels['F_total_X'].put_data(
            self.data_channels['F_A_X'].get_data(chunk_length)
            - self.data_channels['F_B_X'].get_data(chunk_length))
        self.data_channels['F_total_Y'].put_data(
            self.data_channels['F_A_Y'].get_data(chunk_length)
            + self.data_channels['F_B_Y'].get_data(chunk_length))
        self.data_channels['F_total_Z'].put_data(
            self.data_channels['F_A_Z'].get_data(chunk_length)
            + self.data_channels['F_B_Z'].get_data(chunk_length)
            + self.c_p['Photodiode_sum_to_force'][2])

    def calc_speeds(self, chunk_length, diff = 5):
        """
        Calculates the speeds of the motors, putting it into the data channels.
        """
        if self.data_channels['Motor_x_pos'].max_retrivable < chunk_length+diff:
            self.data_channels['Motor_x_speed'].put_data(
                np.zeros(self.data_channels['Motor_x_pos'].max_retrivable))
            self.data_channels['Motor_y_speed'].put_data(
                np.zeros(self.data_channels['Motor_y_pos'].max_retrivable))
            self.data_channels['Motor_z_speed'].put_data(
                np.zeros(self.data_channels['Motor_z_pos'].max_retrivable))
            return
        x_data = self.data_channels['Motor_x_pos'].get_data(chunk_length+diff)
        y_data = self.data_channels['Motor_y_pos'].get_data(chunk_length+diff)
        z_data = self.data_channels['Motor_z_pos'].get_data(chunk_length+diff)
        time_data = self.data_channels['Motor time'].get_data(chunk_length+diff)
        dx = -(x_data[diff:] - x_data[:-diff])
        dy = -(y_data[diff:] - y_data[:-diff])
        dz = -(z_data[diff:] - z_data[:-diff])
        dt = -(time_data[diff:] - time_data[:-diff]) * 1e-6 # Conveting to seconds here
        self.data_channels['Motor_x_speed'].put_data(dx * self.c_p['microns_per_tick'] / dt)
        self.data_channels['Motor_y_speed'].put_data(dy * self.c_p['microns_per_tick'] / dt)
        self.data_channels['Motor_z_speed'].put_data(dz * self.c_p['microns_per_tick'] / dt)

    def read_data(self):
        """
        Reads the data in the buffer of the separate process.
        """
        if self.portenta_data.empty():
            return None
        data = []
        while not self.portenta_data.empty():
            data.append(self.portenta_data.get_nowait())

        return np.concatenate(data)
    
    def update_piezodac_data(self):
        if self.c_p['portenta_command_2'] == 1:
            self.c_p['piezo_A'] = np.int32([self.data_channels['dac_ax'].get_data_spaced(1)[0],
                                            self.data_channels['dac_ay'].get_data_spaced(1)[0]])
        if self.c_p['portenta_command_2'] == 2:
            self.c_p['piezo_B'] = np.int32([self.data_channels['dac_bx'].get_data_spaced(1)[0],
                                            self.data_channels['dac_by'].get_data_spaced(1)[0]])

    def read_data_to_channels(self, chunk_length=256):
        """
        Reads all the data channels and sort the data into the correct data channels.
        """
        # Chunk length is the number of 16 bit numbers sent each time
        # i.e half the number of bytes sent.
        numbers = self.read_data()
        if numbers is None:
            sleep(0.001)
            return
        numbers = numbers.astype(int)

        L = len(numbers)
        nbr_chunks = int(L/chunk_length)
        nbr_channels = len(self.c_p['multi_sample_channels'])

        zero_offset = 1 + len(self.c_p['single_sample_channels'])
        unused_indices = (chunk_length-zero_offset) % nbr_channels

        # Single sample channels
        for idx, channel in enumerate(self.c_p['single_sample_channels']):
            data = numbers[idx+1:L:chunk_length]
            if channel in self.c_p['offset_channels']:
                data -= 32768
            self.data_channels[channel].put_data(data)
        data_length_single = len(data)

        # Compute starts and stops once
        base_starts = zero_offset + np.arange(nbr_chunks) * chunk_length
        base_stops = chunk_length * np.arange(1, nbr_chunks+1) - unused_indices
        indices = np.concatenate(
            [np.arange(start, stop, nbr_channels) for start, stop in zip(base_starts, base_stops)])

        # Multi sample channels
        for idx, channel in enumerate(self.c_p['multi_sample_channels']):
            if channel == 'T_time':
                continue
            data = numbers[indices+idx]
            if channel in self.c_p['offset_channels']:
                data -= 32768
            self.data_channels[channel].put_data(data)

        data_length = len(indices)
        low = self.data_channels['Time_micros_low'].get_data(data_length).astype(np.uint32)
        high = self.data_channels['Time_micros_high'].get_data(data_length).astype(np.uint32)
        T_time = (high << 16) | low
        self.data_channels['T_time'].put_data(T_time)
        self.data_channels['Motor time'].put_data(T_time[::14]) # 14 data points per chunk

        if self.reset_motor_pos or (time()- self.start_time)>1:
            self.reset_target_positions()
            self.reset_motor_pos = False
        self.start_time = time()

        self.calculate_quotes_fast(data_length)
        self.calc_true_powers(data_length)
        self.calc_forces(data_length)
        
        self.calc_speeds(data_length_single, diff=5)
        self.update_piezodac_data()

    def create_fast_interpolator(self, x_grid, y_grid, z_values, method='cubic',size=40):
        """
        Creates an interpolator function for irregularly sampled 2D grid data.
        
        Parameters:
        x_grid (numpy.ndarray): 1D array of x-coordinates.
        y_grid (numpy.ndarray): 1D array of y-coordinates.
        z_values (numpy.ndarray): 2D array of z-values corresponding to (x, y) coordinates.
        method (str): Interpolation method ('linear', 'nearest', 'cubic'). Default is 'cubic'.
        
        Returns:
        function: Interpolator function that takes (x, y) coordinates and returns interpolated
            z-values.
        """
        # Flatten the input grids
        points = np.c_[x_grid.ravel(), y_grid.ravel()]
        values = z_values.ravel()

        # Create the interpolation grid
        xi = np.linspace(x_grid.min(), x_grid.max(), size)
        yi = np.linspace(y_grid.min(), y_grid.max(), size)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Pre-compute interpolation on a fine grid
        zi_grid = griddata(points, values, (xi_grid, yi_grid), method=method)
        
        # Also pre-compute nearest neighbor interpolation for extrapolation
        zi_nearest = griddata(points, values, (xi_grid, yi_grid), method='nearest')

        # Replace NaNs in the zi_grid with nearest neighbor values
        nan_mask = np.isnan(zi_grid)
        zi_grid[nan_mask] = zi_nearest[nan_mask]

        def interpolator(x, y):
            """
            Interpolates the z-value for given (x, y) coordinates.
            
            Parameters:
            x (float or numpy.ndarray): x-coordinate(s) for interpolation.
            y (float or numpy.ndarray): y-coordinate(s) for interpolation.
            
            Returns:
            numpy.ndarray: Interpolated z-value(s).
            """
            x_index = np.searchsorted(xi, x)
            y_index = np.searchsorted(yi, y)
            x_index = np.clip(x_index, 0, len(xi) - 1)
            y_index = np.clip(y_index, 0, len(yi) - 1)
            return zi_grid[y_index, x_index]

        return interpolator

    
    def create_interpolators(self):

        # Create interpolators for the calibration points
        self.APX_interpolator = self.create_fast_interpolator(
            self.c_p['calibration_points'][:,:,2],
            self.c_p['calibration_points'][:,:,3],
            self.c_p['calibration_points'][:,:,0])
        
        self.APY_interpolator = self.create_fast_interpolator(
            self.c_p['calibration_points'][:,:,2],
            self.c_p['calibration_points'][:,:,3],
            self.c_p['calibration_points'][:,:,1])

        self.BPX_interpolator = self.create_fast_interpolator(
            self.c_p['calibration_points'][:,:,4],
            self.c_p['calibration_points'][:,:,5],
            self.c_p['calibration_points'][:,:,0])
        self.BPY_interpolator = self.create_fast_interpolator(
            self.c_p['calibration_points'][:,:,4],
            self.c_p['calibration_points'][:,:,5],
            self.c_p['calibration_points'][:,:,1])

        self.AFX_interpolator = self.create_fast_interpolator(
            self.c_p['calibration_points'][:,:,2],
            self.c_p['calibration_points'][:,:,3],
            self.c_p['calibration_points'][:,:,6])
        self.AFY_interpolator = self.create_fast_interpolator(
            self.c_p['calibration_points'][:,:,2],
            self.c_p['calibration_points'][:,:,3],
            self.c_p['calibration_points'][:,:,7])

        self.BFX_interpolator = self.create_fast_interpolator(
            self.c_p['calibration_points'][:,:,4],
            self.c_p['calibration_points'][:,:,5],
            self.c_p['calibration_points'][:,:,8])
        self.BFY_interpolator = self.create_fast_interpolator(
            self.c_p['calibration_points'][:,:,4],
            self.c_p['calibration_points'][:,:,5],
            self.c_p['calibration_points'][:,:,9])

        self.Z_interpolator = self.create_fast_interpolator(
            self.c_p['calibration_points'][:,:,2],
            self.c_p['calibration_points'][:,:,3],
            self.c_p['calibration_points'][:,:,10])

        self.APX_0 = self.APX_interpolator([0],[0])[0]
        self.APY_0 = self.APY_interpolator([0],[0])[0]
        self.BPX_0 = self.BPX_interpolator([0],[0])[0]
        self.BPY_0 = self.BPY_interpolator([0],[0])[0]

    def interpolate_position(self, chunk_length):
        # Interpolate the position of the particles from the calibration points
        self.data_channels['Position_A_X'].put_data(
            self.APX_interpolator(
                self.data_channels['PSD_A_P_X'].get_data(chunk_length),
                self.data_channels['PSD_A_P_Y'].get_data(chunk_length))-self.APX_0) 
        self.data_channels['Position_A_Y'].put_data(
            self.APY_interpolator(
                self.data_channels['PSD_A_P_X'].get_data(chunk_length),
                self.data_channels['PSD_A_P_Y'].get_data(chunk_length))-self.APY_0) 
        self.data_channels['Position_B_X'].put_data(
            self.BPX_interpolator(
                self.data_channels['PSD_B_P_X'].get_data(chunk_length),
                self.data_channels['PSD_B_P_Y'].get_data(chunk_length))-self.BPX_0)
        self.data_channels['Position_B_Y'].put_data(
            self.BPY_interpolator(
                self.data_channels['PSD_B_P_X'].get_data(chunk_length),
                self.data_channels['PSD_B_P_Y'].get_data(chunk_length))-self.BPY_0)

    def calculate_compensated_force(self, chunk_length):

        self.data_channels['F_A_X'].put_data(
            (self.data_channels['PSD_A_F_X'].get_data(chunk_length) - 
             self.AFX_interpolator(self.data_channels['PSD_A_P_X'].get_data(chunk_length),
                                   self.data_channels['PSD_A_P_Y'].get_data(chunk_length)))
                                   * self.c_p['PSD_to_force'][0])
        self.data_channels['F_A_Y'].put_data(
            (self.data_channels['PSD_A_F_Y'].get_data(chunk_length) - 
             self.AFY_interpolator(self.data_channels['PSD_A_P_X'].get_data(chunk_length),
                                   self.data_channels['PSD_A_P_Y'].get_data(chunk_length)))
                                   * self.c_p['PSD_to_force'][1])

        self.data_channels['F_B_X'].put_data(
            (self.data_channels['PSD_B_F_X'].get_data(chunk_length) - 
             self.BFX_interpolator(self.data_channels['PSD_B_P_X'].get_data(chunk_length),
                                   self.data_channels['PSD_B_P_Y'].get_data(chunk_length)))
                                   * self.c_p['PSD_to_force'][2])
        self.data_channels['F_B_Y'].put_data(
            (self.data_channels['PSD_B_F_Y'].get_data(chunk_length) - 
             self.BFY_interpolator(self.data_channels['PSD_B_P_X'].get_data(chunk_length),
                                   self.data_channels['PSD_B_P_Y'].get_data(chunk_length)))
                                   * self.c_p['PSD_to_force'][3])
        
    def is_connected(self):
        return self.connected

    def connect(self):
        if not self.connected:
            self.communication_process = PortentaCommsProcess(
                self.portenta_data, self.outdata, self.c_p['COM_port'], self.running_process)
            self.communication_process.start()
            self.connected = True
    
    def disconnect(self):
        if self.connected:
            print("Setting running process to 0")
            self.running_process.value = 0
            sleep(0.5)
            self.communication_process.join()
            self.connected = False

    def run(self):
        self.prepare_portenta_commands()
        self.read_data_to_channels()
        sleep(1e-3)

