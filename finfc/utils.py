from __future__ import annotations

from typing import Optional

import numpy as np
import pytorch_lightning
import pytorch_lightning.callbacks
import torch
import torch.nn


from . import _configs


def create_trainer(config: _configs.BaseTrainingConfig, callbacks: list=None, kwargs: dict=None) -> pytorch_lightning.Trainer:
    """Utility function to create a pytorch-lightning trainer from common configuration elements.

    Parameters
    ----------
    config : BaseTrainingConfig
        The configuration to use when creating the trainer.
    callbacks : list, optional
        Optional list of additional callbacks to use
    kwargs : dict, optional
        Optional dictionary of additional keyword arguments to pass to the trainer.
    """

    if callbacks is None:
        callbacks = []
    else:
        callbacks = list(callbacks)

    callbacks.append(pytorch_lightning.callbacks.LearningRateMonitor(logging_interval='step'))

    _kwargs = {
        'callbacks': callbacks,
        'max_epochs': config.max_epochs,
    }

    if config.gpus > 0:
        _kwargs['devices'] = config.gpus
        _kwargs['accelerator'] = 'gpu'
    else:
        _kwargs['accelerator'] = 'cpu'

    try:
        precision = int(config.precision)
    except ValueError:
        precision = config.precision
    _kwargs['precision'] = precision

    _kwargs.update(config.lightning)

    if kwargs is not None:
        _kwargs.update(kwargs)

    return pytorch_lightning.Trainer(**_kwargs)


def load_data(path: str, key: str=None):
    """Generic data loading function which supports multiple storage formats.

    This function is intended as a generic data loading mechanism to load a tensor
    from a file on disk. It supports different formats (deduced from the file extension),
    as well as special functionality to fetch sub-parts of some formats.

    Supported formats are:
    - .npy: Numpy array file. The array is loaded as-is.
    - .npz: Numpy zip file. The array is loaded from the given key if specified.
            Otherwise, the first array is loaded.
    - .h5 / .hdf5: HDF5 file. The array is loaded from the specified key.

    Parameters
    ----------
    path : str
        Path to the file to load.
    key : str
        Key to the sub-part of the data to load.

    Returns
    -------
    torch.Tensor
        The loaded tensor.
    """
    if path.endswith('.npy'):
        data = np.load(path)
    elif path.endswith('.npz'):
        archive = np.load(path)

        if key is None:
            data = next(iter(archive.values()))
        else:
            data = archive[key]
    elif path.endswith('.h5') or path.endswith('.hdf5'):
        if key is None:
            raise ValueError('HDF5 files must specify a key')

        import h5py
        f = h5py.File(path, 'r')
        data = np.asarray(f[key])
    else:
        raise ValueError('Unsupported file format.')

    return torch.from_numpy(data)


def ensure_channels(field: torch.Tensor, num_channels: Optional[int]) -> torch.Tensor:
    """Ensures that the given field has a channels dimension.

    This function performs the following:
    - If the number of channels is given and greater than 1,
        checks if the last dimension has the given size.
    - If the number of channels is 1, checks if the last dimension is of size 1.
        If not, expands the tensor in the last dimension.
    - If the number of channels is `None`, attempts to guess whether the tensor should be
        expanded in the last dimension (by default: expands if the last dimension has size greater than 32)
    """
    if num_channels is None:
        if field.shape[-1] > 32:
            return field.unsqueeze(-1)
    else:
        if num_channels > 1:
            if field.shape[-1] != num_channels:
                raise ValueError(f'Field has {field.shape[-1]} channels, but expected {num_channels}')
        else:
            if field.shape[-1] != 1:
                return field.unsqueeze(-1)

    return field
