from __future__ import annotations

from ._configs import *
from . import data, models

import hydra
import numpy as np
import torch
import torch.nn
import torch.utils.data


import pytorch_lightning
import pytorch_lightning.callbacks


def _ensure_list(x):
    if not isinstance(x, list):
        return [x]
    return x


class NeuralFieldModel(pytorch_lightning.LightningModule):
    hparams: TrainingConfig

    def __init__(self, config: TrainingConfig, field: torch.Tensor=None):
        super().__init__()
        self.save_hyperparameters(config)

        self.featurization = models.MultigridInterpolator(
            levels=config.model.hash.num_levels,
            entries_per_level=2 ** config.model.hash.num_entries_log2,
            base_grid=[config.model.hash.base_grid] * 3,
            features=config.model.hash.num_features,
            sparse=config.model.sparse)

        self.mlp = models.MLP(
            n_input=self.featurization.output_dim,
            n_output=config.data.num_channels,
            n_hidden=config.model.hidden_size,
            n_layers=config.model.mlp_layers,
            act=torch.nn.GELU())

        if field is not None:
            self.register_buffer('field', field)
        else:
            self.field = None

        self.criterion = torch.nn.L1Loss()

        # Disable automatic optimization to split sparse / dense gradients.
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("hash_features"):
            features = self.featurization(x)
        with torch.autograd.profiler.record_function("mlp"):
            values = self.mlp(features)
        return values

    def training_step(self, batch, batch_idx):
        x, y = batch
        predicted = self.forward(x)

        if self.field is not None:
            y = self.field[y]

        loss = self.criterion(predicted, y)
        self.log('train_loss', loss, prog_bar=True)

        opts = _ensure_list(self.optimizers())
        for opt in opts:
            opt.zero_grad()

        self.manual_backward(loss)

        for opt in opts:
            opt.step()

        for sched in _ensure_list(self.lr_schedulers()):
            sched.step(loss)

        return loss

    def on_train_epoch_end(self) -> None:
        pass


    def _configue_optim_sparse(self):
        optim_mlp = torch.optim.Adam(
            self.mlp.parameters(),
            lr=self.hparams.optim.learning_rate)

        optim_hash = torch.optim.SparseAdam(
            self.featurization.parameters(),
            lr=self.hparams.optim.learning_rate)

        patience = 5 * self.hparams.data.batches_per_epoch

        scheduler_mlp = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim_mlp, mode='min', factor=0.5, patience=patience, verbose=True)

        scheduler_hash = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim_hash, mode='min', factor=0.5, patience=patience, verbose=True)

        return [
            {
                'optimizer': optim_mlp,
                'lr_scheduler': scheduler_mlp,
                'monitor': 'train_loss'
            }, 
            {
                'optimizer': optim_hash,
                'lr_scheduler': scheduler_hash,
                'monitor': 'train_loss'
            }
        ]

    def _configure_optim_standard(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.optim.learning_rate)
        patience = 5 * self.hparams.data.batches_per_epoch
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=patience, verbose=True)
        return [{'optimizer': optim, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}]

    def configure_optimizers(self):
        if self.hparams.model.sparse:
            return self._configue_optim_sparse()
        else:
            return self._configure_optim_standard()


class NeuralFieldDataModule(pytorch_lightning.LightningDataModule):
    hparams: DataConfig
    _dataset: data.FieldDataset
    num_channels: Optional[int]

    def __init__(self, config: DataConfig, batch_size: int):
        super().__init__()
        self.save_hyperparameters(config)

        self._dataset = None
        self.num_channels = None
        self.batch_size = batch_size
        self.field = None


    def setup(self, stage: str=None):
        if self._dataset is not None:
            return

        self.field = torch.from_numpy(np.load(self.hparams.path))
        self.num_channels = self.field.shape[-1]

        if self.hparams.load_field_on_gpu:
            self._dataset = data.FieldIndexDataset(self.field.shape[:-1])
        else:
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
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True)

        return dataloader


@hydra.main(config_name='config', config_path=None)
def main(config: TrainingConfig):
    kwargs = {
        'devices': 'auto',
        'accelerator': 'gpu',
        'precision': 16,
        'callbacks': [
            pytorch_lightning.callbacks.LearningRateMonitor(logging_interval='step'),
            pytorch_lightning.callbacks.ModelCheckpoint(save_top_k=1, monitor='train_loss', mode='min')],
    }

    trainer = pytorch_lightning.Trainer(
        **kwargs,
        max_epochs=config.max_epochs)

    dm = NeuralFieldDataModule(config.data, config.batch_size)
    dm.setup()

    config.data.num_channels = dm.num_channels
    config.data.batches_per_epoch = len(dm._dataset) // dm.batch_size

    if config.data.load_field_on_gpu:
        field = dm.field.to(torch.float16).view(-1, dm.field.shape[-1])
    else:
        field = None

    model = NeuralFieldModel(config, field=field)
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('config', node=TrainingConfig)
    main()
