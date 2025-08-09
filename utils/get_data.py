import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
import os

class NavierStokesDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "data/" 
        #please download from https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download
        h5_file = h5py.File(os.path.join(path, "ns_incom_inhom_2d_512-0.h5"), "r")
        #print(list(h5_file.keys())) #['force', 'particles', 't', 'velocity']
        data = np.array(h5_file['velocity'])  # (4, 1000, 512, 512, 2)
        our_data = data[0]  
        print('our data shape', our_data.shape)
        h5_file.close()

        self.flow = our_data[:800,:,:,:]

    def __len__(self):
        return 600

    def max_index(self):
        return 600

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)
        data = data.permute(0, 3, 1, 2)
        return data


class EvalNavierStokesDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "data/" 
        #please download from https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download
        h5_file = h5py.File(os.path.join(path, "ns_incom_inhom_2d_512-0.h5"), "r")
        print(list(h5_file.keys())) #['force', 'particles', 't', 'velocity']
        data = np.array(h5_file['velocity'])  # (4, 1000, 512, 512, 2)
        our_data = data[0]  
        h5_file.close()

        self.flow = our_data[800:1000,:,:,:]

    def __len__(self):
        return 1 

    def max_index(self):
        return 1

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)
        data = data.permute(0, 3, 1, 2)
        return data


class ReacDiffDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "data/" 
        #please download from https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download
        h5_file = h5py.File(os.path.join(path, "2D_diff-react_NA_NA.h5"), "r")
        num_samples = len(h5_file.keys())
        seed = 0 
        seed = str(seed).zfill(4)
        data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 2]
        print('original data shape', data.shape)
        h5_file.close()

        self.flow = data[:80,:,:,:]

    def __len__(self):
        return 60

    def max_index(self):
        return 60

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)
        data = data.permute(0, 3, 1, 2)
        return data


class EvalReacDiffDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "data/" 
        #please download from https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download
        h5_file = h5py.File(os.path.join(path, "2D_diff-react_NA_NA.h5"), "r")
        num_samples = len(h5_file.keys())
        seed = 0 
        seed = str(seed).zfill(4)
        data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 2]
        h5_file.close()
        
        self.flow = data[80:100,:,:,:]

    def __len__(self):
        return 1 

    def max_index(self):
        return 1

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)
        data = data.permute(0, 3, 1, 2)
        print('original data shape', data.shape)
        return data


class ShalloWaterDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "data/" 
        #please download from https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download
        h5_file = h5py.File(os.path.join(path, "2D_rdb_NA_NA.h5"), "r")  
        num_samples = len(h5_file.keys())
        seed = 0 
        seed = str(seed).zfill(4)
        data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 1]

        mean = np.mean(data)
        std = np.std(data)
        standardized_data = (data - mean) / std

        # Min-max scaling to [-1, 1]
        min_val = np.min(standardized_data)
        max_val = np.max(standardized_data)

        # Normalize to [-1, 1]
        scaled_data = 2 * (standardized_data - min_val) / (max_val - min_val) - 1

        print('original data shape', data.shape)
        print('mean: ', mean)
        print('stdev: ', std)
        h5_file.close()
        
        self.flow = scaled_data[:80,:,:,:]

    def __len__(self):
        return 60

    def max_index(self):
        return 60

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)
        data = data.permute(0, 3, 1, 2)
        return data


class EvalShallowWaterDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample
        path = "data/" 
        #please download from https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download
        h5_file = h5py.File(os.path.join(path, "2D_rdb_NA_NA.h5"), "r")
        num_samples = len(h5_file.keys())
        seed = 0 #np.random.randint(0, num_samples) 
        seed = str(seed).zfill(4)
        data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 1]
       
        mean = np.mean(data)
        std = np.std(data)
        standardized_data = (data - mean) / std

        # Min-max scaling to [-1, 1]
        min_val = np.min(standardized_data)
        max_val = np.max(standardized_data)

        # Normalize to [-1, 1]
        scaled_data = 2 * (standardized_data - min_val) / (max_val - min_val) - 1

        h5_file.close()
        
        self.flow = scaled_data[80:100,:,:,:]

    def __len__(self):
        return 1 

    def max_index(self):
        return 1

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)
        data = data.permute(0, 3, 1, 2)
        return data


class SimpleFlowDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample

        # Read dataset
        data = np.load('data/small_flow.npy') #data file is proivided in the repository
        print('original data shape', data.shape) #(151, 64, 64)
        data = data.reshape(data.shape[0],64,64,1)
        self.flow = data[:125,:,:,:]

    def __len__(self):
        return 100

    def max_index(self):
        return 100

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)
        data = data.permute(0, 3, 1, 2)
        return data


class EvalSimpleFlowDataset(Dataset):

    def __init__(self, snapshots_per_sample=5):

        self.snapshots_per_sample = snapshots_per_sample

        # Read dataset
        data = np.load('data/small_flow.npy') 
        data = data.reshape(data.shape[0],64,64,1)
        self.flow = data[125:150,:,:,:]

    def __len__(self):
        return 1 

    def max_index(self):
        return 1

    def __getitem__(self, index, time_idx=0):
        
        prefinals = []
        for i in range(index, index + self.snapshots_per_sample):
                prefinals.append(torch.Tensor(self.flow[i]).float())

        data = torch.stack(prefinals)
        data = data.permute(0, 3, 1, 2)
        return data
