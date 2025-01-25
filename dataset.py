import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler,RobustScaler, StandardScaler

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch


import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TimeDataset(Dataset):
    def __init__(self, raw_data, df_mts_spike, edge_index, scale=True, win=[200, 50], out_pred=50, val_data=False, scaler=None):
        self.raw_data = raw_data
        self.df_mts_spike = df_mts_spike
        self.edge_index = edge_index
        self.win_size = win[0]
        self.stride = win[1]
        self.out_pred = out_pred
        self.val_data = val_data
        self.scaler = scaler if scaler else (StandardScaler() if scale else None)

        # Scale data according to whether it's validation or training
        self.data = self.__scale_data__()

        # Align and ensure the spike data matches the raw_data shape
        self.spike_data = None
        if df_mts_spike is not None:
            self.spike_data = self.__prepare_spike_data__()

        # Precompute valid indices to avoid overlaps
        self.valid_indices = self.__compute_valid_indices__()

    def __len__(self):
        return len(self.valid_indices)

    def __scale_data__(self):
        data_copy = self.raw_data.copy()  # Avoid modifying the original data

        if self.scaler:
            if self.val_data:
                # If it's validation data, use the scaler to transform only
                data_copy[:] = self.scaler.transform(data_copy)
            else:
                # For training data, fit and transform
                data_copy[:] = self.scaler.fit_transform(data_copy)

        return data_copy

    def __prepare_spike_data__(self):
        """
        Align and prepare the spike data to match the shape and time steps of raw_data.
        """

        if len(self.df_mts_spike) != len(self.raw_data):
            raise ValueError("Spike data length must match raw_data length.")
        
        # Ensure spike data is a torch tensor with matching time dimension
        return torch.tensor(self.df_mts_spike.values).float()

    def __compute_valid_indices__(self):
        valid_indices = []
        for idx in range(0, len(self.data) - self.win_size - self.out_pred + 1, self.stride):
            s_begin = idx
            s_end = s_begin + self.win_size
            s_begin_y = s_end
            s_end_y = s_begin_y + self.out_pred
            
            # Ensure no overlap between seq_x and seq_y
            if s_end <= s_begin_y:
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]  # Map to precomputed valid index
        s_begin = idx
        s_end = s_begin + self.win_size

        # Get the input window for raw data
        seq_x = self.data.values[s_begin:s_end]
        seq_x = torch.tensor(seq_x).permute(1, 0)

        # Get the corresponding spike data window
        if self.spike_data is not None:
            spike_x = self.spike_data[s_begin:s_end].permute(1, 0)


        # Get the target window
        s_begin_y = s_end
        s_end_y = s_begin_y + self.out_pred
        seq_y = self.data.values[s_begin_y:s_end_y]
        seq_y = torch.tensor(seq_y).permute(1, 0)

        if self.spike_data is not None:
            spike_y = self.spike_data[s_begin_y:s_end_y].permute(1, 0)
            return seq_x, seq_y, spike_y
        
        return seq_x, seq_y 


        #seq_x = seq_x.unsqueeze(-1)  # Change shape from [28, 200] to [28, 200, 1]
        #seq_y = seq_y.unsqueeze(-1)
"""
class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, scale=True, win=[200, 50], out_pred=50, val_data=False, scaler=None):
        self.raw_data = raw_data
        self.edge_index = edge_index
        self.win_size = win[0]
        self.stride = win[1]
        self.out_pred = out_pred
        self.val_data = val_data
        self.scaler = scaler if scaler else (StandardScaler() if scale else None)

        # Scale data according to whether it's validation or training
        self.data = self.__scale_data__()

    def __len__(self):
        return  (len(self.data) - self.win_size) // self.stride

    def __scale_data__(self):
        data_copy = self.raw_data.copy()  # Avoid modifying the original data

        if self.scaler:
            if self.val_data:
                # If it's validation data, use the scaler to transform only
                data_copy[:] = self.scaler.transform(data_copy)
            else:
                # For training data, fit and transform
                data_copy[:] = self.scaler.fit_transform(data_copy)

        return data_copy

    def __getitem__(self, idx):
        s_begin = idx * self.stride
        s_end = s_begin + self.win_size

        if s_end > len(self.data):
            s_begin = len(self.data)- self.out_pred
            s_end = len(self.data)

        # Get the input window
        seq_x = self.data.values[s_begin:s_end]
        seq_x = torch.tensor(seq_x).permute(1, 0)

        # Get the target window
        s_begin_y = s_end
        s_end_y = s_begin_y + self.out_pred

        # Adjust if the target exceeds data length
        if s_end_y > len(self.data):
            s_begin_y = len(self.data) - self.out_pred
            s_end_y = len(self.data)

        seq_y = self.data.values[s_begin_y:s_end_y]
        seq_y = torch.tensor(seq_y).permute(1, 0)

        print(f's_begin : {s_begin}')
        print(f's_end : {s_end}')

        print(f's_begin_y : {s_begin_y}')
        print(f's_end_y : {s_end_y}')


        return seq_x, seq_y

"""
    



class TimeDatasetFull(Dataset):
    def __init__(self, raw_data, edge_index, scale=True, win=[200, 50]):
        self.raw_data = raw_data
        self.edge_index = edge_index

        self.win_size = win[0]
        self.stride = win[1]

        # Convert raw_data to tensor and initialize scaling
        self.data = raw_data
        self.scaler = StandardScaler() if scale else None
        self.__scale_data__()

        # Create and store the windowed version of the entire dataset
        self.inputs = self.__create_windows__()

    def __scale_data__(self):
        """Scale the data if the scaler is provided."""
        if self.scaler:
            self.data[:] = self.scaler.fit_transform(self.data)

    def __create_windows__(self):
        """Generate and return all overlapping windows for the input time series."""
        num_samples = (len(self.data) - self.win_size) // self.stride + 1
        inputs = []

        for i in range(num_samples):
            # Define the start and end of the window for inputs (X)
            s_begin = i * self.stride
            s_end = s_begin + self.win_size

            # Ensure windowing stays within bounds
            if s_end > len(self.data):
                s_begin = len(self.data) - self.win_size
                s_end = len(self.data)

            # Create input window sequence (X)
            seq_x = self.data.values[s_begin:s_end]
            seq_x = torch.tensor(seq_x).permute(1, 0)  # Reshape to [features, time]
            inputs.append(seq_x)

        # Stack all the windows into a tensor
        return torch.stack(inputs)

    def __len__(self):
        """Return the number of windows created."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """Return all the windowed sequences and edge index."""
        # Return the full windowed tensor and the edge index
        return self.inputs, self.edge_index.long()