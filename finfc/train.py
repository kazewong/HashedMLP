from __future__ import annotations

import functools
import logging
from typing import Sequence

import hydra
import numpy as np
import torch
import torch.nn
import torch.utils.data
import torchmetrics
import torchmetrics.image.psnr

import pytorch_lightning
import pytorch_lightning.callbacks

from ._configs import *
from . import data, models, utils, prediction, preprocessing

logger = logging.getLogger(__name__)


def generate_grid_for_slice(grid_sizes: Sequence[int], box_sizes: Sequence[float]=None, like: torch.Tensor=None):
    dtype = like.dtype if like is not None else torch.float32
    device = like.device if like is not None else torch.device('cpu')

    grids = [
        torch.linspace(0, b, g + 2, dtype=dtype, device=device)[1:-1]
        for g, b in zip(grid_sizes, box_sizes)
    ]
    return torch.cartesian_prod(*grids)

def render_slice(model: torch.nn.Module, grid_sizes: Sequence[int], slice_depth: float, box_sizes: Sequence[float]) -> torch.Tensor:
    if box_sizes is None:
        box_sizes = [1] * len(grid_sizes)

    slice_grid_size = grid_sizes[:-1]
    slice_box_size = box_sizes[:-1]

    points = generate_grid_for_slice(slice_grid_size, slice_box_size, like=next(iter(model.parameters())))
    points = torch.cat([points, torch.full_like(points[..., 0:1], slice_depth)], dim=1)

    values = model(points)
    return values.view(*slice_grid_size, values.shape[-1])

def _normalize(x: torch.Tensor, x_min: float=None, x_max: float=None) -> torch.Tensor:
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    return x.sub(x_min).div_(x_max - x_min)


class NeuralFieldModel(pytorch_lightning.LightningModule):
    """Main model class for neural field training.
    """
    hparams: TrainingConfig
    criterion: prediction.AbstractLoss

    def __init__(self, config: TrainingConfig, field: torch.Tensor=None):
        """Create a new `NeuralFieldModel` from the given configuration.

        Parameters
        ----------
        config : TrainingConfig
            The configuration to use for the model.
        field : torch.Tensor, optional
            If not `None`, a tensor representing the entire field.
            This option is mainly used for optimization of data transfers,
            but does not appear to yield any performance gains.
        """
        super().__init__()
        self.save_hyperparameters(config)

        self.criterion: prediction.AbstractLoss = hydra.utils.instantiate(config.loss)

        self.featurization = models.MultigridInterpolator(
            levels=config.model.hash.num_levels,
            entries_per_level=2 ** config.model.hash.num_entries_log2,
            base_grid=config.model.hash.base_grid,
            final_grid=config.model.hash.final_grid,
            box_size=config.model.hash.box_size,
            features=config.model.hash.num_features,
            sparse=config.model.sparse)

        self.mlp = models.MLP(
            n_input=self.featurization.output_dim,
            n_output=config.data.num_channels * self.criterion.num_outputs,
            n_hidden=config.model.hidden_size,
            n_layers=config.model.mlp_layers,
            act=torch.nn.GELU())

        self.preprocessor = preprocessing.make_from_config_list(config.model.preprocessing)

        if field is not None:
            self.register_buffer('field', field)
        else:
            self.field = None

        self.l1_metric = torchmetrics.MeanAbsoluteError()
        self.psnr = torchmetrics.image.psnr.PeakSignalNoiseRatio(data_range=2)

        self.automatic_optimization = False

    def _evaluate_field(self, x: torch.Tensor):
        with torch.autograd.profiler.record_function("hash_features"):
            features = self.featurization(x)
        with torch.autograd.profiler.record_function("mlp"):
            values = self.mlp(features)

        values = values.view(*values.shape[:-1], self.hparams.data.num_channels, self.criterion.num_outputs)
        return values

    def forward(self, x: torch.Tensor, normalized: bool=True) -> torch.Tensor:
        values = self._evaluate_field(x)
        values = self.criterion.predict(values)

        if not normalized and self.preprocessor is not None:
            values = self.preprocessor(values)

        return values

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()

        x, y = batch
        output = self._evaluate_field(x)

        if self.field is not None:
            y = self.field[y]

        loss, predicted = self.criterion(output, self.criterion.preprocess(y))

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        self.manual_backward(loss)

        if self.hparams.optim.normalize_interpolator_gradient:
            for n, p in self.featurization.named_parameters():
                if p.requires_grad:
                    grad_norm = p.grad.norm()
                    p.grad.div_(grad_norm)
                    self.log('grad_norm/' + n, grad_norm.detach())

        for opt in optimizers:
            opt.step()
        for sched in self.lr_schedulers():
            sched.step()

        self.log('train_loss', loss.detach(), prog_bar=True)

        with torch.no_grad():
            l1 = self.l1_metric(predicted, y)
            self.log('l1', l1, prog_bar=True)
            self.log('metrics/l1', self.l1_metric, on_step=False, on_epoch=True)

            self.psnr(predicted, y)
            self.log('metrics/psnr', self.psnr)

        return loss

    def on_train_epoch_end(self) -> None:
        image = render_slice(self, self.hparams.model.hash.final_grid, 0.5, self.hparams.model.hash.box_size)

        # Revert normalization to display in 0-1 range as expected
        # by tensorboard logger
        if self.hparams.data.normalize:
            image = _normalize(image, -1, 1)
        else:
            image = _normalize(image)

        self.logger.experiment.add_image('slice', image, self.trainer.global_step, dataformats='HWC')

    def _make_optimizer_interpolator(self, batches_per_epoch: int):
        optim = torch.optim.SGD(self.featurization.parameters(), lr=self.hparams.optim.lr_interpolator, momentum=0.9, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.max_epochs * batches_per_epoch)
        return {'optimizer': optim, 'lr_scheduler': lr_scheduler}

    def _make_optimizer_mlp(self, batches_per_epoch: int):
        optim = torch.optim.Adam(self.mlp.parameters(), lr=self.hparams.optim.lr_mlp)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.max_epochs * batches_per_epoch, eta_min=0.0)
        return {'optimizer': optim, 'lr_scheduler': lr_scheduler}

    def configure_optimizers(self):
        if self.hparams.model.sparse:
            raise NotImplementedError("Sparse gradients not implemented.")
        return [
            self._make_optimizer_interpolator(self.hparams.data.batches_per_epoch),
            self._make_optimizer_mlp(self.hparams.data.batches_per_epoch)]


def _make_normalize_preprocessor_config(normalize: Optional[str]=None):
    if normalize is None:
        return None

    if normalize == 'linear':
        return RescalePreprocessorConfig()

    if normalize.startswith('quantile'):
        num_quantiles = int(normalize.split('_')[1])
        return QuantilePreprocessorConfig(num_quantiles=num_quantiles)

    raise ValueError(f"Unknown normalization type: {normalize}")


class NeuralFieldDataModule(pytorch_lightning.LightningDataModule):
    hparams: DataConfig
    _dataset: data.FieldDataset
    num_channels: Optional[int]

    def __init__(self, config: DataConfig, batch_size: int, box_size: Sequence[float]=None):
        super().__init__()
        self.save_hyperparameters(config)

        self._dataset = None
        self.num_channels = None
        self.batch_size = batch_size
        self.field = None
        self.preprocessor_config = []
        self.box_size = box_size

    def setup(self, stage: str=None):
        if self._dataset is not None:
            return

        self.field = utils.load_data(self.hparams.path, self.hparams.key)
        self.field = utils.ensure_channels(self.field, self.hparams.num_channels)

        if self.box_size is None:
            field_shape = np.array(self.field.shape[:-1])
            self.box_size = (field_shape / np.min(field_shape)).tolist()

        preprocessors = []

        if self.hparams.transform == 'log' or self.hparams.transform == 'log1p':
            self.preprocessor_config.append(LogPreprocessorConfig())
            log_preprocessor = preprocessing.LogPreprocessor()
            self.field = log_preprocessor.preprocess(self.field)
            preprocessors.append(log_preprocessor)

        normalize_config = _make_normalize_preprocessor_config(self.hparams.normalize)
        if normalize_config is not None:
            preprocessor = preprocessing.make_from_config(normalize_config)
            self.field = preprocessor.preprocess(self.field)
            self.preprocessor_config.append(normalize_config)
            preprocessors.append(preprocessor)

        self.preprocessor = preprocessing.ComposePreprocessor(preprocessors)

        self.num_channels = self.field.shape[-1]

        if self.hparams.load_field_on_gpu:
            self._dataset = data.FieldIndexDataset(self.field.shape[:-1], box_size=self.box_size)
        else:
            self._dataset = data.FieldDataset(self.field, box_size=self.box_size)

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


@hydra.main(config_name='config', config_path='conf')
def main(config: TrainingConfig):
    config.data.path = hydra.utils.to_absolute_path(config.data.path)

    ckpt_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        save_top_k=1, every_n_epochs=config.max_epochs,
        filename='model')

    trainer = utils.create_trainer(config, [ckpt_callback], {'log_every_n_steps': 2})
    dm = NeuralFieldDataModule(
        config.data, config.batch_size,
        box_size=config.model.hash.box_size)
    dm.setup()

    config.data.num_channels = dm.num_channels
    config.data.batches_per_epoch = len(dm._dataset) // dm.batch_size
    config.model.preprocessing = dm.preprocessor_config

    # Set final grid and box size from dataset if they are unset
    if config.model.hash.final_grid is None:
        config.model.hash.final_grid = dm.field.shape[:-1]

    if config.model.hash.box_size is None:
        logger.info('Using inferred box size for data: %s', dm.box_size)
        config.model.hash.box_size = dm.box_size

    if config.data.load_field_on_gpu:
        field = dm.field.to(torch.float16).view(-1, dm.field.shape[-1])
    else:
        field = None

    model = NeuralFieldModel(config, field=field)

    # Load preprocessor parameters induced from the datamodule
    model.preprocessor.load_state_dict(dm.preprocessor.state_dict())

    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('config', node=TrainingConfig)
    cs.store('l1', node=L1LossConfig, group='loss')
    cs.store('cross_entropy', node=CrossEntropyLossConfig, group='loss')
    main()
