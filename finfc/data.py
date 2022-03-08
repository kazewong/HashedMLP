from __future__ import annotations

import concurrent.futures
import itertools
from typing import Collection, Iterable, Generic, List, Tuple, TypeVar

import numpy as np
import torch
import torch.utils.data


def _compute_strides(shape: Collection[int]) -> List[int]:
    strides = [1]
    for dim in reversed(shape[1:]):
        strides.append(strides[-1] * dim)
    return list(reversed(strides))


class FieldIndexDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """This dataset produces a sequence of (multi_idx, linear_idx) pairs representing indices
    in a multi-dimensional grid.
    """
    def __init__(self, shape: Collection[int], box_size: Collection[float]=None):
        """Create a new `FieldIndexDataset`.

        Parameters
        ----------
        shape : Collection[int]
            A collection of integers representing the grid size in each dimension
        box_size : Collection[int]
            A collection of floating point values representing the box size in each dimension.
        """
        super().__init__()

        self._shape = np.array(shape)

        if box_size is None:
            box_size = np.ones_like(self._shape)

        self._box_size = np.array(box_size)
        self._len = int(torch.prod(torch.tensor(self._shape)))
        self._grid_ratio = self._box_size / self._shape

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = np.asarray(idx)

        multi_idx = np.stack(np.unravel_index(idx, self._shape), axis=-1)
        multi_idx = multi_idx * self._grid_ratio

        return torch.from_numpy(multi_idx), torch.as_tensor(idx)


class FieldDataset(FieldIndexDataset):
    def __init__(self, field: torch.Tensor, box_size: Collection[float]=None):
        super().__init__(field.shape[:-1], box_size=box_size)

        self.field = field.contiguous()
        self._field_flat = self.field.flatten(end_dim=-2)

    def __getitem__(self, idx):
        multi_idx, idx = super().__getitem__(idx)
        return multi_idx, self._field_flat[idx]


class FastRandomSampler(torch.utils.data.Sampler[torch.Tensor], torch.utils.data.IterableDataset[torch.Tensor]):
    """Fast random sampler which uses some structured tactics to sample from very large datasets.

    This sampler does not attempt to sample completely uniformly from the given dataset,
    but rather produces a block random sample of the dataset (with some adjustments to avoid
    obvious block artifacts).

    """
    def __init__(self, blocks: Collection[int], batch_size: int, generator: torch.Generator=None):
        self._numel = int(torch.prod(torch.tensor(blocks)))
        self._len = self._numel // batch_size

        self.generator = generator
        self._blocks = blocks
        self._strides = _compute_strides(blocks)

        self.batch_size = batch_size

        if self._numel & (self._numel - 1) != 0:
            # length is not power of 2
            raise ValueError('Total length must be a power of 2!')

    def __len__(self):
        return self._len

    def __iter__(self):
        # Powers of 5 are units of Z / 2^n Z, so we can use a random such power to generate
        # a global bijection of Z / 2^n Z onto itself through multiplication.
        # We use this to avoid obvious blocking patterns in the sample, so that the linear index does
        # not align directly with the original grid.
        exponent = int(torch.randint(self._numel, tuple(), generator=self.generator))
        bijection = pow(5, exponent, self._numel)

        block_perms = [torch.randperm(b, generator=self.generator) for b in self._blocks]

        for i in range(self._len):
            idxs = np.unravel_index(np.arange(i * self.batch_size, (i + 1) * self.batch_size), self._blocks)
            multi_idx = [b[i] for b, i in zip(block_perms, idxs)]
            linear_idx = sum(s * i for s, i in zip(self._strides, multi_idx))
            linear_idx *= bijection
            linear_idx %= self._numel
            yield linear_idx
