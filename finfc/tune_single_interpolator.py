"""This module implements training of a single dense interpolator.

The goal is to evaluate different tuning in order to understand potential optimization
issues even in the absence of compression.

"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import omegaconf
import pytorch_lightning
import pytorch_lightning.callbacks
import torch
import torch.utils.data

from ._configs import BaseTrainingConfig
from . import data, models, utils

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class ModelConfig:
    hidden_size: int = 64
    feature_size: int = 8
    mlp_layers: int = 2
    sparse: bool = False

@dataclasses.dataclass
class DataConfig:
    dim: int = 2
    size: int = 512
    normalize: bool = True

@dataclasses.dataclass
class RandomDataConfig(DataConfig):
    _target_: str = 'cosmology.models.compression.hashed_neural_field.tune_single_interpolator._make_random_data'
    dim: int = 2
    size: int = 512
    distribution: str = "uniform"
    seed: int = 42

@dataclasses.dataclass
class FieldDataConfig(DataConfig):
    _target_: str = 'cosmology.models.compression.hashed_neural_field.tune_single_interpolator._make_field_data'
    path: str = '/home/wzhou/data/fields/camels_IllustrisTNG/kaze_mass_256.npy'
    transform: str = 'log'

@dataclasses.dataclass
class OptimConfig:
    lr_interpolator: float = 1e4
    lr_mlp: float = 1e-2
    optim_interpolator: str = 'sgd'
    optim_mlp: str = 'adam'
    normalize_interpolator_gradients: bool = True

@dataclasses.dataclass
class TrainingConfig(BaseTrainingConfig):
    defaults: List[Any] = dataclasses.field(default_factory=lambda: [
        "_self_",
        {'data': 'random'}
    ])
    model: ModelConfig = ModelConfig()
    data: DataConfig = omegaconf.MISSING
    optim: OptimConfig = OptimConfig()


def _ensure_list(l):
    if not isinstance(l, list):
        return [l]
    else:
        return l


class SingleInterpolatorModel(pytorch_lightning.LightningModule):
    hparams: TrainingConfig

    def __init__(self, config: TrainingConfig):
        super().__init__()

        self.save_hyperparameters(config)

        grid_size = [config.data.size] * config.data.dim

        if config.model.mlp_layers > 0:
            self.featurization = models.Interpolator(models.DenseGrid(grid_size, config.model.feature_size, sparse=self.hparams.model.sparse), grid_size)
            self.mlp = models.MLP(config.model.feature_size, 1, config.model.hidden_size, config.model.mlp_layers, torch.nn.GELU())
        else:
            self.featurization = models.Interpolator(models.DenseGrid([config.data.size] * config.data.dim, 1, sparse=self.hparams.model.sparse), grid_size)
            self.mlp = torch.nn.Identity()

        self.criterion = utils.ElasticNetLoss()
        self.automatic_optimization = False

        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x) -> torch.Tensor:
        feat = self.featurization(x)
        return self.mlp(feat)

    def training_step(self, batch, batch_idx):
        optimizers = _ensure_list(self.optimizers())
        schedulers = _ensure_list(self.lr_schedulers())

        x, y = batch
        predicted = self.forward(x)
        loss = self.criterion(predicted, y)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        self.manual_backward(loss)

        if self.hparams.optim.normalize_interpolator_gradients:
            for n, t in self.featurization.named_parameters():
                grad_norm = t.grad.norm()
                self.log('grad_norm/{}'.format(n), grad_norm)
                t.grad.div_(grad_norm)

        for opt in optimizers:
            opt.step()

        for sched in schedulers:
            sched.step()

        with torch.no_grad():
            l1_loss = self.l1_loss(predicted, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('l1', l1_loss, prog_bar=True)

        return loss

    def _make_optimizer_interpolator(self, batches_per_epoch: int):
        if self.hparams.optim.optim_interpolator == 'adam':
            if self.hparams.model.sparse:
                optim = torch.optim.SparseAdam(self.featurization.parameters(), lr=self.hparams.optim.lr_interpolator)
            else:
                 optim = torch.optim.Adam(self.featurization.parameters(), lr=self.hparams.optim.lr_interpolator)
        elif self.hparams.optim.optim_interpolator == 'sgd':
            optim = torch.optim.SGD(self.featurization.parameters(), lr=self.hparams.optim.lr_interpolator, momentum=0.9, nesterov=True)
        else:
            raise ValueError('Unknown optimizer type for interpolator: {}'.format(self.hparams.optim.optim_interpolator))

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.max_epochs * batches_per_epoch)
        return {'optimizer': optim, 'lr_scheduler': lr_scheduler}

    def _make_optimizer_mlp(self, batches_per_epoch: int):
        parameters = list(self.mlp.parameters())

        if len(parameters) == 0:
            return []

        if self.hparams.optim.optim_mlp == 'adam':
            optim = torch.optim.Adam(self.mlp.parameters(), lr=self.hparams.optim.lr_mlp)
        elif self.hparams.optim.optim_mlp == 'sgd':
            optim = torch.optim.SGD(self.mlp.parameters(), lr=self.hparams.optim.lr_mlp, momentum=0.9, nesterov=True)
        else:
            raise ValueError('Unknown optimizer type for MLP: {}'.format(self.hparams.optim.optim_mlp))

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.max_epochs * batches_per_epoch, eta_min=0.0)
        return [{'optimizer': optim, 'lr_scheduler': lr_scheduler}]


    def configure_optimizers(self):
        batches_per_epoch = self.hparams.data.size ** self.hparams.data.dim // self.hparams.batch_size

        optimizers = [self._make_optimizer_interpolator(batches_per_epoch)]
        optimizers.extend(self._make_optimizer_mlp(batches_per_epoch))

        return optimizers

class NeuralFieldDataModule(pytorch_lightning.LightningDataModule):
    hparams: RandomDataConfig
    _dataset: data.FieldDataset
    field: Optional[torch.Tensor]

    def __init__(self, config: RandomDataConfig, batch_size: int, num_data_workers: int):
        super().__init__()
        self.save_hyperparameters(config)

        self._dataset = None
        self.batch_size = batch_size
        self.field = None
        self.num_data_workers = num_data_workers

    def setup(self, stage: str=None):
        if self._dataset is not None:
            return

        self.field = hydra.utils.instantiate(self.hparams)

        if self.hparams.normalize:
            field_max = self.field.max()
            field_min = self.field.min()
            self.field.sub_(field_min).div_(field_max - field_min)

        self.num_channels = self.field.shape[-1]
        self._dataset = data.FieldDataset(self.field)

    def train_dataloader(self):
        # automatic batching is too slow, our dataset handle queries with
        # lists of indices and batches for us, so do that instead.

        # We need a dataloader for our sampler!
        # The sampler is a bit computationally expensive, so farming it to its own thread is beneficial
        sampler = torch.utils.data.DataLoader(
            data.FastRandomSampler(self.field.shape[:-1], batch_size=self.batch_size),
            batch_size=None, shuffle=False, num_workers=1, persistent_workers=True)

        dataloader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=self.num_data_workers,
            persistent_workers=self.num_data_workers > 0,
            pin_memory=True)

        return dataloader

def _make_random_data(size: int, dim: int, distribution: str, seed: int, *_, **__) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)

    if distribution != 'uniform':
        raise NotImplementedError()

    return torch.rand([size] * dim + [1], generator=gen)


def _make_field_data(size: int, dim: int, path: str, transform: str, *_, **__) -> torch.Tensor:
    logger.info('Loading field data from %s', path)
    field = torch.from_numpy(np.load(path))
    logger.info('Loaded field data of shape %s', field.shape)

    if transform == 'log':
        field = field.log1p_()

    return field


@hydra.main(config_name='config', config_path=None)
def main(config: TrainingConfig):
    dm = NeuralFieldDataModule(config.data, config.batch_size, num_data_workers=4)
    dm.setup()

    config.data.size = dm.field.shape[0]
    config.data.dim = len(dm.field.shape) - 1

    model = SingleInterpolatorModel(config)

    ckpt_callback = pytorch_lightning.callbacks.ModelCheckpoint(every_n_epochs=0)
    trainer = utils.create_trainer(config, [ckpt_callback], {'log_every_n_steps': 2})
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('config', node=TrainingConfig)
    cs.store('random', node=RandomDataConfig, group='data')
    cs.store('field', node=FieldDataConfig, group='data')
    main()
