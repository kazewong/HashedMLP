from numpy import int64, uint32
import torch
import torch.nn as nn
import numpy as np
import itertools



class HashedInterpolator(nn.Module):
    """
    Single level Hashed interpolator.

    Args:
        n_dim: Dimension of the input space.
        n_entries: Number of entries in the hash table.
        n_feature: Number of features.
        grids: The input size of the object. (n_1, n_2, ..., n_dim)

    """

    def __init__(self, n_dim: int, n_entries: int, n_feature: int, grids: torch.tensor,device='cuda'):
        super().__init__()
        self.n_dim = n_dim
        self.n_entries = n_entries
        self.n_feature = n_feature
        self.grids = grids
        self.device = device
        self.large_prime = torch.tensor([1, 19349663, 83492791, 48397621]).to(device)
        self.register_parameter(name='hash_table',param=nn.Parameter(((torch.rand(n_entries,n_feature)-0.5)*2e-4).to(device)))
        self.bit_table = torch.tensor(list(itertools.product([0,1],repeat=n_dim))).to(device)
        self.index_list = torch.repeat_interleave(torch.arange(n_dim)[None,:],2**n_dim,dim=0).to(device)

    def findGrid(self,position):
        """
        Find the corresponding lower corner of the input position.
        Assume the position are in the range of [0,1]^n_dim.
    
        Args:
            position: A tensor of shape (batch_size, n_dim)
        Returns:

        """
        corner_index = torch.floor(position*self.grids).int()
        return corner_index
        

    def getData(self,lower_corner):
        """
        Get data using lower corner indices and data array.

        Args:
            lower_corner: A tensor of shape (batch_size, n_dim)
            data: A tensor of shape (n_shape...)
        Returns:
            A tensor of shape (batch_size, n_dim)
        """
        corner_index = torch.repeat_interleave(lower_corner[:,None],2**self.n_dim,dim=1)+self.bit_table
        corner_hash = self.hashing(corner_index)
        corner_value = self.hash_table[corner_hash]
        return corner_value


    def ndLinearInterpolation(self,lower_corner,corner_value,position)->torch.tensor:
        """
        N dimensional linear interpolation.

        Args:
            corner_value: A tensor of shape (batch_size, 2**n_dim)
            position: A tensor of shape (batch_size, n_dim)
        Returns:
            A tensor of shape (batch_size)
        """
        lower_limit = lower_corner/self.grids
        upper_limit = (lower_corner+1)/self.grids
        size = (upper_limit - lower_limit)[0]
        coef_list = torch.stack([(position-lower_limit)/size,(upper_limit-position)/size],axis=-1)
        result = corner_value*torch.prod(coef_list[:,self.index_list,self.bit_table],dim=2)
        return torch.sum(result,axis=1)

    def hashing(self, index: torch.tensor) -> torch.tensor:
        """
        Hash function to turn input indicies into a hash table.

        Args:
            index: A tensor of shape (batch_size, n_dim)
        Returns:
            A tensor of shape (batch_size)

        """
        hash = torch.zeros(index.shape[:-1],dtype=torch.int32).to(self.device)
        for i in range(self.n_dim):
            hash = torch.bitwise_xor(hash,index[...,i]*self.large_prime[i]) % self.n_entries
        return hash

    def forward(self, position: torch.tensor) -> torch.tensor:
        lower_corner = self.findGrid(position)
        corner_value = self.getData(lower_corner)
        result = torch.stack([self.ndLinearInterpolation(lower_corner,corner_value[...,i],position) for i in range(self.n_feature)],axis=-1)
        return result

class MLP(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_layers: int, act = nn.ReLU()):
        super().__init__()
        layers = np.array([nn.Linear(n_hidden, n_hidden) for i in range(n_layers)])
        layers = np.insert(layers,np.arange(1,n_layers+1),act)
        self.output = nn.Sequential(nn.Linear(n_input, n_hidden), act, *layers, nn.Linear(n_hidden, n_output))

    def forward(self, x):
        return self.output(x)

class HashedMLP(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_layers: int,
                    n_entries: int, n_feature: int, base_grids: torch.tensor, 
                    n_level: int, n_factor: float, n_auxin:int=0, act = nn.ReLU()):
        super().__init__()
        self.level = n_level
        self.interpolator = nn.ModuleList([HashedInterpolator(n_input, n_entries, n_feature,base_grids*n_factor**i) for i in range(n_level)])
        self.MLP = MLP(n_feature*n_level+n_auxin, n_output, n_hidden, n_layers, act)

    def forward(self, x):
        x = torch.concat([self.interpolator[i](x) for i in range(self.level)],dim=1)
        return self.MLP(x)