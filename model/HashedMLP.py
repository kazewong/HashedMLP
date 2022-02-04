"""Pytorch code for hashed neural fields, adapted from code written by Kaze Wong.
"""

from __future__ import annotations
from typing import Iterable

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
    grids: torch.Tensor
    bit_table: torch.Tensor
    hash_table: torch.Tensor
    index_list: torch.Tensor

    def __init__(self, n_entries: int, n_feature: int, grids: torch.tensor, sparse: bool=False):
        super().__init__()

        n_dim = len(grids)
        self.n_dim = n_dim
        self.n_feature = n_feature
        self.n_entries = n_entries

        self.hash_table = torch.nn.Embedding(n_entries, n_feature, sparse=sparse)

        self.register_buffer(name='grids', tensor=grids)
        self.register_buffer(name='bit_table', tensor=torch.tensor(list(itertools.product([0,1],repeat=n_dim))))
        self.register_buffer(name='index_list', tensor=torch.repeat_interleave(torch.arange(n_dim)[None,:],2**n_dim,dim=0))
        self.register_buffer(name='large_prime', tensor=torch.tensor([1, 19349663, 83492791, 48397621]))

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
        corner_index = torch.repeat_interleave(lower_corner[:,None], 2**self.n_dim, dim=1) + self.bit_table
        corner_hash = self.hashing(corner_index)
        corner_value = self.hash_table(corner_hash)
        return corner_value


    def ndLinearInterpolation(self, lower_corner: torch.Tensor, corner_value: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """
        N dimensional linear interpolation.

        Args:
            corner_value: A tensor of shape (batch_size, 2**n_dim)
            position: A tensor of shape (batch_size, n_dim)
        Returns:
            A tensor of shape (batch_size)
        """
        lower_limit = lower_corner / self.grids
        upper_limit = (lower_corner + 1) / self.grids
        size = (upper_limit[0] - lower_limit[0])

        coef_list = torch.stack([(position-lower_limit) / size, (upper_limit-position) / size], dim=-1)
        prod_coef = torch.prod(coef_list[:, self.index_list, self.bit_table], dim=-1)
        result = torch.matmul(corner_value.unsqueeze(-2), prod_coef.unsqueeze_(-1)).squeeze_(-1).squeeze_(-1)
        return result

    def hashing(self, index: torch.Tensor) -> torch.Tensor:
        """
        Hash function to turn input indicies into a hash table.

        Args:
            index: A tensor of shape (batch_size, n_dim)
        Returns:
            A tensor of shape (batch_size)

        """
        hash = index[..., 0] * self.large_prime[0]
        for i in range(1, self.n_dim):
            hash.bitwise_xor_(index[..., i] * self.large_prime[i]).remainder_(self.n_entries)
        return hash

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("HashedInterpolator.forward"):
            lower_corner = self.findGrid(position)

            with torch.autograd.profiler.record_function("HashedInterpolator.fetch_data"):
                corner_value = self.getData(lower_corner)

            with torch.autograd.profiler.record_function("HashedInterpolator.linear_interpolation"):
                result = torch.stack([self.ndLinearInterpolation(lower_corner, corner_value[...,i], position) for i in range(self.n_feature)],axis=-1)
            return result


class MLP(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_layers: int, act = nn.ReLU()):
        super().__init__()
        layers = itertools.chain.from_iterable([(nn.Linear(n_hidden, n_hidden), act) for _ in range(n_layers)])
        self.output = nn.Sequential(nn.Linear(n_input, n_hidden), act, *layers, nn.Linear(n_hidden, n_output))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


class MultigridInterpolator(torch.nn.Module):
    def __init__(self, levels: int, entries_per_level: int, base_grid: Iterable[int], features: int=2, factor: float=1.5, sparse: bool=False):
        super().__init__()

        base_grid = torch.tensor(base_grid)

        self.interpolators = torch.nn.ModuleList([
            HashedInterpolator(entries_per_level, features, base_grid * factor ** i, sparse=sparse) for i in range(levels)
        ])

        self.output_dim = features * levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat([interpolator(x) for interpolator in self.interpolators], dim=-1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


class HashedMLP(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, n_layers: int,
                    n_entries: int, n_feature: int, base_grids: torch.Tensor, 
                    n_level: int, n_factor: float, n_auxin:int=0, act = nn.ReLU()):
        super().__init__()

        base_grids = torch.as_tensor(base_grids)

        self.level = n_level
        self.interpolator = nn.ModuleList([HashedInterpolator(n_entries, n_feature, base_grids*n_factor**i) for i in range(n_level)])
        self.MLP = MLP(n_feature * n_level + n_auxin, output_dim, hidden_dim, n_layers, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.concat([self.interpolator[i](x) for i in range(self.level)],dim=1)
        return self.MLP(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)
