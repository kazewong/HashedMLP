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
