"""Pytorch code for hashed neural fields, adapted from code written by Kaze Wong.
"""

from __future__ import annotations
from typing import Collection, Union, Sequence

import torch
import torch.nn as nn
import numpy as np
import itertools


class HashedGrid(torch.nn.Module):
    """This class implements functionality to interpret a hash-table as a grid of values.

    This class models a module representing a grid of values, i.e. a module which can be called
    using an array of integer indices.

    Instead of storing the values in a dense grid, instead this class stores the values in a hash-table
    of the given size. Note that this essentially randomly compresses the storage with arbitrary collisions.
    """
    primes: torch.Tensor

    def __init__(self, num_entries: int, num_features: int, max_norm: float=None, norm_type: float=2, sparse: bool=False):
        """Create a new `HashedGrid` instance with the given number of entries and features.
        Optionally, enables sparse gradients for the underlying hash table.
        """
        super().__init__()

        self.hash_table = torch.nn.Embedding(num_entries, num_features, max_norm=max_norm, norm_type=norm_type, sparse=sparse)
        self.register_buffer('primes', tensor=torch.tensor([1, 19349663, 83492791, 48397621]))

    def hash_index(self, index: torch.Tensor) -> torch.Tensor:
        hash = index[..., 0] * self.primes[0]
        for i in range(1, index.shape[-1]):
            hash.bitwise_xor_(index[..., i] * self.primes[i]).remainder_(self.hash_table.num_embeddings)
        return hash

    def forward(self, idx: torch.Tensor):
        hash = self.hash_index(idx)
        return self.hash_table(hash)


class DenseGrid(torch.nn.Module):
    """This class implements a dense grid of the given size.
    """
    strides: torch.Tensor

    def __init__(self, grid_size: Sequence[int], num_features: int, max_norm: float=None, norm_type: float=2, sparse: bool=False):
        super().__init__()

        grid_size = torch.tensor([int(x) + 1 for x in grid_size])
        self.grid = torch.nn.Embedding(torch.prod(grid_size).item(), num_features, max_norm=max_norm, norm_type=norm_type, sparse=sparse)

        strides = torch.cat((torch.ones([1], dtype=torch.long), grid_size.cumprod(0)[:-1]))
        self.register_buffer('strides', strides)

    def forward(self, idx: torch.Tensor):
        # Integer matmul not well supported, use mul + sum instead.
        idx_linear = torch.mul(idx, self.strides.to(idx.dtype)).sum(-1)
        return self.grid(idx_linear)


class Interpolator(torch.nn.Module):
    """This class implements functionality to interpolate from a coarse grid into a fine grid.

    This class may be used to adapt a module which represents values sampled on a coarse integer
    grid to a continuous field through linear interpolation. The field is interpreted to be represented
    on the domain ``[0, 1]^d``.
    """
    box_size: torch.Tensor
    grid_size: torch.Tensor
    grid_steps: torch.Tensor
    bit_table: torch.Tensor
    index_list: torch.Tensor

    def __init__(self, grid_values: torch.nn.Module, grid_size: Sequence[int], box_size: Sequence[float]=None):
        """Create a new interpolator from the given grid specification.

        Parameters
        ----------
        grid_values : torch.nn.Module
            A module which can be called with an integer tensor of shape ``[batch, d]``, and returns a tensor of size ``[batch, f]``
            where ``f`` denotes the number of features.
        grid_size : torch.Tensor
            A tensor representing the size of the grid in each dimension.
            This should be a one-dimensional integer tensor [n_1, n_2, ..., n_d],
            where d in the number of dimensions, and each n_i denotes the size of the grid in the i-th dimension.
        box_size : Sequence[float], optional
        """
        super().__init__()

        n_dim = len(grid_size)
        self.grid_values = grid_values

        self.register_buffer('grid_size', torch.as_tensor(grid_size))
        self.register_buffer('box_size', torch.as_tensor(box_size) if box_size is not None else torch.ones(n_dim))
        self.register_buffer('grid_steps', self.box_size / self.grid_size)

        # Precomputed buffers for facilitating the interpolation.
        self.register_buffer('bit_table', tensor=torch.tensor(list(itertools.product([0,1], repeat=n_dim))))
        self.register_buffer('index_list', tensor=torch.repeat_interleave(torch.arange(n_dim).unsqueeze_(0), 2**n_dim, dim=0))

    def find_grid(self, position: torch.Tensor) -> torch.Tensor:
        corner_index = position.div(self.grid_steps).floor_().int()
        return corner_index

    def linear_interpolation(self, position: torch.Tensor, lower_corner: torch.Tensor, corner_values: torch.Tensor) -> torch.Tensor:
        lower_limit = lower_corner.mul(self.grid_steps)
        upper_limit = lower_corner.add(1).mul(self.grid_steps)

        coef_list = torch.stack([
            position.sub(lower_limit).to(self.grid_steps).div_(self.grid_steps),
            upper_limit.sub(position).to(self.grid_steps).div_(self.grid_steps)], dim=-1)

        coef_list.sub_(1).neg_()

        prod_coef = torch.prod(coef_list[..., self.index_list, self.bit_table], dim=-1)
        result = torch.matmul(prod_coef.unsqueeze_(-2), corner_values).squeeze_(-2)
        return result

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        # Find lower corner indices
        lower_corner = self.find_grid(position)
        # Expand to indices for all corner positions
        corner_index = torch.repeat_interleave(lower_corner.unsqueeze(-2), 2**len(self.grid_size), dim=1).add_(self.bit_table)
        # Get values at corner positions
        corner_values = self.grid_values(corner_index)
        # Interpolate corner values
        result = self.linear_interpolation(position, lower_corner, corner_values)
        return result


class MLP(torch.nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_layers: int, act = nn.ReLU()):
        super().__init__()

        if n_layers == 0:
            if n_input != n_output:
                raise ValueError("Input and output dimension must match for MLP with 0 layers.")
            self.output = torch.nn.Identity()
        elif n_layers == 1:
            self.output = torch.nn.Linear(n_input, n_output)
        else:
            layers = itertools.chain.from_iterable([(nn.Linear(n_hidden, n_hidden), act) for _ in range(n_layers - 2)])
            self.output = nn.Sequential(nn.Linear(n_input, n_hidden), act, *layers, nn.Linear(n_hidden, n_output))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


def make_grid_with_size(num_entries: int, num_features: int, grid_size: torch.Tensor, max_norm: float=None, norm_type: float=2, sparse: bool = False) -> Union[HashedGrid, DenseGrid]:
    """Makes a grid with the given size.

    If the number of entries is larger than that would be required for a dense grid, this function returns a `DenseGrid`.
    Otherwise, returns a `HashedGrid`.
    """
    if torch.prod(torch.as_tensor(grid_size) + 1).item() < num_entries:
        return DenseGrid(grid_size, num_features, max_norm=max_norm, norm_type=norm_type, sparse=sparse)
    else:
        return HashedGrid(num_entries, num_features, max_norm=max_norm, norm_type=norm_type, sparse=sparse)


class MultigridInterpolator(torch.nn.Module):
    """Multi-grid interpolator with optional hashing.

    This module implements a multi-grid interpolator.
    For a given position it queries a set of interpolators at different resolutions,
    and stacks the obtained values.

    """
    def __init__(self, levels: int, entries_per_level: int, base_grid: Collection[int], final_grid: Collection[int], box_size: Collection[float]=None,
                 features: int=2, max_norm: float=None, norm_type: float=2, sparse: bool=False):
        """Create a new multi-grid interpolator.

        Parameters
        ----------
        levels : int
            Number of levels of the multi-grid.
        entries_per_level : int
            Maximum number of entries per level. This class automatically switches between a dense grid
            and a hashed grid representation when the number of entries for a dense grid exceeds this value.
        base_grid : Collection[int]
            A list of integers representing the size of the base grid (i.e. the number of grid cells
            at the coarsest subdivision).
        final_grid : Collection[int]
            A list of integers representing the size of the final grid (i.e. the number of grid cells
            at the finest subdivision). Must be of the same length as `base_grid`.
        box_size : Collection[float], optional
            If not `None`, a list of float representing the extent of the box in each dimension.
            Must be of the same length as `base_grid`. Otherwise, the box size is assumed to be 1
            in all dimensions.
        features : int
            Dimension of the feature space.
        factor : float
            Growth factor for grid resolution at each level
        max_norm : float, optional
            If not `None`, embedding values at each grid will be rescaled to have a maximum norm of `max_norm`.
        norm_type : float, optional
            The p of p-norm to compute for the max-norm option.
        sparse : bool, optional
            If `True`, indicates that sparse gradients should be used for hashed embeddings.
            Otherwise, dense gradients are used. See `torch.nn.Embedding` for more information.
        """

        super().__init__()

        base_grid_t = torch.as_tensor(base_grid)
        final_grid_t = torch.as_tensor(final_grid)
        factor = torch.exp((final_grid_t.log() - base_grid_t.log()) / (levels - 1))

        def _make_interpolator(level):
            grid_size = base_grid_t.mul(factor**level).round_().int()
            grid = make_grid_with_size(entries_per_level, features, grid_size, max_norm=max_norm, norm_type=norm_type, sparse=sparse)
            return Interpolator(grid, grid_size, box_size)

        self.interpolators = torch.nn.ModuleList([_make_interpolator(level) for level in range(levels)])

        self.output_dim = features * levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat([interpolator(x) for interpolator in self.interpolators], dim=-1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)
