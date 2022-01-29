from numpy import int64, uint32
import torch
import torch.nn as nn
import itertools

large_prime = [1, 19349663, 83492791, 48397621]

class HashedInterpolator(nn.Module):
    def __init__(self, n_dim: int, n_entries: int, n_feature: int, grids: torch.tensor):
        super().__init__()
        self.n_dim = n_dim
        self.n_entries = n_entries
        self.n_feature = n_feature
        self.grids = grids
        self.hash_table = nn.Parameter((torch.rand(n_entries,n_feature)-0.5)*2e-4)
        self.bit_table = torch.tensor(list(itertools.product([0,1],repeat=n_dim)))
        self.index_list = torch.repeat_interleave(torch.arange(n_dim)[None,:],2**n_dim,dim=0)

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


    def ndLinearInterpolation(self,corner_value,position)->torch.tensor:
        """
        N dimensional linear interpolation.

        Args:
            corner_value: A tensor of shape (batch_size, 2**n_dim)
            position: A tensor of shape (batch_size, n_dim)
        Returns:
            A tensor of shape (batch_size)
        """
        coef_list = torch.stack([position,1-position],axis=-1)
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
        hash = torch.zeros(index.shape[:-1],dtype=torch.int32)
        for i in range(self.n_dim):
            hash = torch.bitwise_xor(hash,index[...,i]*large_prime[i]) % self.n_entries
        return hash

    def forward(self, position: torch.tensor) -> torch.tensor:
        lower_corner = self.findGrid(position)
        corner_value = self.getData(lower_corner)
        result = torch.stack([self.ndLinearInterpolation(corner_value[...,i],position) for i in range(self.n_feature)],axis=-1)
        return result