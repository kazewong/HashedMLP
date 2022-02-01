import h5py
import torch
from torch.utils.data import Dataset

class GridDataset(Dataset):

    def __init__(self, data_file, keys,stride= 1):
        data_file = h5py.File(data_file,'r')
        data = []
        n_dim = len(data_file[keys[0]].shape)
        slices = tuple([slice(None,None,stride) for i in range(n_dim)])
        for i in keys:
            data.append(torch.tensor(data_file[i][slices][...,None]))
        data = torch.concat(data,dim=-1).float()

        self.original_shape = data.shape
        axis = [torch.arange(data.shape[i]) for i in range(n_dim)]
        coord = torch.stack(torch.meshgrid(*axis),-1).float()
        for i in range(n_dim):
            coord[...,i] = coord[...,i]/data.shape[i]
        coord = torch.flatten(coord,end_dim=-2)
        data = torch.flatten(data,end_dim=-2)
        
        self.coord = coord
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.coord[idx],self.data[idx]