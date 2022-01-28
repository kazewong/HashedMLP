from numpy import int64, uint32
import torch
import torch.nn as nn
import itertools

large_prime = [1, 19349663, 83492791, 48397621]

class HashInterpolator(nn.Module):
    def __init__(self, n_dim: int, n_entries: int, n_feature: int):
        super().__init__()
        self.n_dim = n_dim
        self.n_entries = n_entries
        self.n_feature = n_feature
        self.hash_table = nn.Parameter((torch.rand(n_entries,n_feature)-0.5)*2e-4)
        self.bit_table = torch.tensor(list(itertools.product([0,1],repeat=n_dim)))
        self.index_list = torch.repeat_interleave(torch.arange(n_dim)[None,:],2**n_dim,dim=0)

    def findGrid(self,position):
        pass

    def ndLinearInterpolation(self,corner_value,position):
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
        hash = torch.zeros(index.shape[0],dtype=torch.int32)
        for i in range(self.n_dim):
            hash = torch.bitwise_xor(hash,index[:,i]*large_prime[i]) % self.n_entries
        return hash

    def forward(self, index: torch.tensor) -> torch.tensor:
        pass