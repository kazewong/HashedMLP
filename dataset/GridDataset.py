import h5py
import torch
from torch.utils.data import Dataset

from pytorch_lightning import LightningDataModule

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

class GridDataModule(LightningDataModule):
    def __init__(self, data_file, keys,stride= 1, batch_size=1024,num_workers=0):
        super().__init__()
        self.data_file = data_file
        self.keys = keys
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = GridDataset(self.data_file, self.keys, self.stride)
        length = torch.tensor([int(dataset.__len__()*0.8),dataset.__len__()-int(dataset.__len__()*0.8)])
        self.train_data, self.val_data = torch.utils.data.random_split(dataset, length)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
