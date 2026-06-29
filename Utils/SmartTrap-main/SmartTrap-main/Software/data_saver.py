"""
Here the functionality used to save data from the instrument is defined.

---------------------------------------------
Classes:

- SaverThreadInterface: An abstarct class for defining a data saver.
- DataSaverThread: The default data saver. A separate thread which saves the data from the SmartTrap.
"""

import numpy as np
from threading import Thread
from time import sleep
import pickle
import os
import pandas as pd
import abc
import cv2
import h5py

class SaverThreadInterface(abc.ABC, Thread):
    @abc.abstractmethod
    def start_saving(self):
        pass

    @abc.abstractmethod
    def is_saving(self):
        pass

    @abc.abstractmethod
    def stop_saving(self):
        pass

    @abc.abstractmethod
    def snapshot(self):
        pass

class DataSaverThread(SaverThreadInterface):
    """
    Thread used to save data in the background while the program is running.
    """

    def __init__(self, c_p, data_channels):
        Thread.__init__(self)
        self.c_p = c_p # Common control parameters
        self.data_channels = data_channels # Data which is to be saved
        self.running = True
        self.sleep_time = 0.1
        self.start_idx = 0
        self.start_idx_motors = 0
        self.start_idx_prediction = 0
        self.saving = False
        self.data_idx = 0
        self.max_sample_rate = 1000
        self.save_indices = {}

        for key in data_channels:
            sample_rate = data_channels[key].sample_rate
            self.max_sample_rate = max(self.max_sample_rate, sample_rate) 
            if not sample_rate in self.save_indices:
                self.save_indices[sample_rate] = [key, 0,0] # Name of first channel with this sample rate and corresponding data index
            self.max_save = int(data_channels[key].max_len*0.5)
        print(f"Max save is {self.max_save}")
        
    
    def set_saving_indices(self, index=1):
        """
        Saves the current indices of the various data channels.
        The index parameter indicates if it is to be saved in the first or last slot for saving. e.g
        setting index=1 indicates saving start index and index=2 indicates the last
        """
        for sample_rate in self.save_indices:
            data_channel_name = self.save_indices[sample_rate][0]
            # Set save index of the data channel
            self.save_indices[sample_rate][index] = self.data_channels[data_channel_name].index

    def set_filename(self):
        self.filename = self.c_p['recording_path'] + '/' + self.c_p['filename'] + str(self.data_idx) +"."+self.c_p['data_file_format']

    def start_saving(self):        
        self.set_saving_indices(1)
        
        self.set_saving_indices(2)

        self.saving = True
        # self.data_idx += 1
        self.set_filename()
        print("Saving started")

    def save_csv_data(self, data):
        # Convert data to DataFrame
        max_len = max(len(v) for v in data.values())

        # Pad all arrays with NaN
        padded_data = {}
        max_len = max(len(v) if hasattr(v, '__len__') else 1 for v in data.values())

        for k, v in data.items():
            v_list = list(v) if hasattr(v, '__len__') else [v]
            padded_data[k] = v_list + [np.nan] * (max_len - len(v_list))
        df_new = pd.DataFrame(padded_data)

        # Append mode for CSV
        if not os.path.exists(self.filename):
            # Create new file and write headers
            df_new.to_csv(self.filename, mode='w', index=False)  
        else:
            # Append without headers
            df_new.to_csv(self.filename, mode='a', header=False, index=False)

    def save_hdf5_data(self, data):
        if not os.path.exists(self.filename):
            with h5py.File(self.filename, 'w') as f:
                for key, arr in data.items():
                    f.create_dataset(
                        key,
                        data=arr,
                        maxshape=(None,) + arr.shape[1:],  # allow growth along axis 0
                        chunks=True
                    )
        else:
            with h5py.File(self.filename, 'a') as f:  # 'a' = append/update mode
                for key, arr in data.items():
                    dset = f[key]
                    old_len = dset.shape[0]
                    new_len = old_len + arr.shape[0]
                    dset.resize((new_len,) + dset.shape[1:])
                    dset[old_len:new_len] = arr

    def save_mat_data(self, data):
        """
        Docstring for save_mat_data
        This function saves the data in MATLAB .mat format.
        If last_save is True, it will also create a final .mat file with all data combined.
        Uses the H5PY library to save data in HDF5 format and then conver this to .mat format.
        """
        self.filename = self.c_p['recording_path'] + '/' + self.c_p['filename'] + str(self.data_idx) +".h5"
        self.save_hdf5_data(data)
        if not self.saving:
            with h5py.File(self.filename, 'r') as f:
                data_loaded = {key: f[key][()] for key in f.keys()}
            # Delete the intermediate HDF5 file
            os.remove(self.filename)
            # Save the data in .mat format
            from scipy.io import savemat
            self.filename = self.c_p['recording_path'] + '/' + self.c_p['filename'] + str(self.data_idx) +".mat"            
            savemat(self.filename, data_loaded)           


    def save_numpy_data(self,data):
        self.data_idx += 1
        self.set_filename()
        with open(self.filename, 'wb') as f:
            pickle.dump(data, f)

    def is_saving(self):
        return self.saving

    def snapshot(self, filename_save=None):
        """
        Captures a snapshot of what the camera is viewing and saves that
        in the fileformat specified by the image_format parameter.
        """
        if filename_save is None or isinstance(filename_save, bool):
            filename_save = self.c_p['filename']
        idx = str(self.c_p['image_idx'])
        filename = (self.c_p['recording_path'] + '/' + filename_save + 'image_' + idx + '.' +
                    self.c_p['image_format'])
        if self.c_p['image_format'] == 'npy':
            np.save(filename[:-4], self.c_p['image'])
        else:
            cv2.imwrite(filename, cv2.cvtColor(self.c_p['image'],
                                           cv2.COLOR_RGB2BGR))

        self.c_p['image_idx'] += 1

    def get_data_dict(self):
        # Set the stop indices
        self.set_saving_indices(2) 

        sleep(0.1) # Waiting for all channels to reach this point


        data = {}
        for sample_rate in self.save_indices:
            if self.save_indices[sample_rate][1] == self.save_indices[sample_rate][2]:
                self.save_indices[sample_rate][2] += 1

        for channel in self.data_channels:
            if self.data_channels[channel].saving_toggled:
                # Handle the different rates at which the channels are sampled to get the right data
                # to be saved.
                start_index = self.save_indices[self.data_channels[channel].sample_rate][1]
                stop_index = self.save_indices[self.data_channels[channel].sample_rate][2]
    
                if start_index < stop_index:
                    data[channel] = self.data_channels[channel].data[start_index:
                                                                        stop_index]
                else:
                    data[channel] = np.concatenate(
                        [self.data_channels[channel].data[start_index:],
                            self.data_channels[channel].data[:stop_index]])

        # Update the save indices so that the previous stop is the new start
        for sample_rate in self.save_indices:
            self.save_indices[sample_rate][1] = self.save_indices[sample_rate][2]
        return data


    def stop_saving(self):
        """
        Stops the data saving process, collects the recorded data from all relevant channels,
        and saves it to a file.
        This method finalizes the current data recording session by:
        - Setting the saving flag to False.
        - Determining the stop indices for each data channel.
        - Handling cases where channels may not have sampled correctly by adjusting indices.
        - Extracting the relevant data slices for each channel, accounting for channels sampled at
          different rates.
        - Saving the collected data to a file using pickle.
        - Incrementing the data file index for future recordings.
        The method ensures that data from all channels is synchronized as much as possible,
        and handles wrap-around cases where the stop index is less than the start index.
        """

        self.saving = False
        print("Saving stopped")
        data = self.get_data_dict()
        # May need to offload this saving to a different process to improve performance
        if self.c_p['data_file_format'] == 'csv':
            self.save_csv_data(data)
        elif self.c_p['data_file_format'] == 'h5':
            self.save_hdf5_data(data)
        elif self.c_p['data_file_format'] == 'mat':
            self.save_mat_data(data)        
        else:
            self.save_numpy_data(data)
        self.data_idx += 1 # Moved here from start saving


    def run(self):
        while self.c_p['program_running']:
            # Will implement continous saving here.
            if self.saving:
                # Check if we have moved far enough to save the data
                key = self.save_indices[self.max_sample_rate][0]
                start_index = self.save_indices[self.max_sample_rate][1]
                data_index =self.data_channels[key].index
                if data_index<start_index or (data_index-start_index)>self.max_save:                    
                    data = self.get_data_dict()
                    if self.c_p['data_file_format'] == 'csv':
                        self.save_csv_data(data)
                    elif self.c_p['data_file_format'] == 'h5':
                        self.save_hdf5_data(data)
                    elif self.c_p['data_file_format'] == 'mat':
                        self.save_mat_data(data)
                    else:
                        self.save_numpy_data(data)
            sleep(self.sleep_time)
        if not self.c_p['program_running'] and self.saving:
            self.stop_saving()
