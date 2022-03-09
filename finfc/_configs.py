from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import omegaconf

@dataclasses.dataclass
class PreprocessorConfig:
    name: str

@dataclasses.dataclass
class LogPreprocessorConfig(PreprocessorConfig):
    name: str = 'log'

@dataclasses.dataclass
class RescalePreprocessorConfig(PreprocessorConfig):
    name: str = 'rescale'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    output_min: float = -1
    output_max: float = 1

@dataclasses.dataclass
class QuantilePreprocessorConfig(PreprocessorConfig):
    name: str = 'quantile'
    num_quantiles: int = 8
    output_min: float = -1
    output_max: float = 1


@dataclasses.dataclass
class LossConfig:
    _target_: str
    lower: float = -1
    upper: float = 1

@dataclasses.dataclass
class L1LossConfig(LossConfig):
    _target_: str = 'finfc.prediction.L1Loss'

@dataclasses.dataclass
class CrossEntropyLossConfig(LossConfig):
    _target_: str = 'finfc.prediction.CrossEntropyRegressionLoss'
    num_grid: int = 256


@dataclasses.dataclass
class HashConfig:
    """Configuration for the multigrid hash problem.

    Attributes
    ----------
    base_grid : List[int]
        Size of the grid at the coarsest level.
    final_grid : List[int], optional
        Size of the grid at the finest level. If `None`, the final grid size will be
        set to the native resolution of the loaded data.
    box_size : List[float], optional
        Size of the box in each dimension. If `None`, the box size will be set so that the
        shortest side has length 1, and each grid cell is cubic.
    num_levels : int
        Number of grid levels.
    num_features : int
        Dimension of feature vector (hash entry)
    num_entries_log2 : int
        Number of entries in the hash table (log 2).
    """
    base_grid: List[int] = dataclasses.field(default_factory=lambda: [16] * 3)
    final_grid: Optional[List[int]] = None
    box_size: Optional[List[float]] = None
    num_levels: int = 8
    num_features: int = 2
    num_entries_log2: int = 24

@dataclasses.dataclass
class ModelConfig:
    hash: HashConfig = HashConfig()
    hidden_size: int = 64
    mlp_layers: int = 2
    sparse: bool = False
    preprocessing: List[PreprocessorConfig] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class OptimConfig:
    lr_mlp: float = 1e-2
    lr_interpolator: float = 1e1
    weight_decay: float = 1e-5
    normalize_interpolator_gradient: bool = True

@dataclasses.dataclass
class DataConfig:
    """Configuration for data source.

    Attributes
    ----------
    path : str
        Path to a numpy data file containing the field to compress
    key : str, optional
        If not `None`, a key which specifies which array to load from the data file.
        Useful for loading from `.npz` or `.h5` files.
    num_channels : int, optional
        The number of channels present in the data.
    batches_per_epoch: int, optional
        The number of batches corresponding to one epoch through the data.
    num_workers : int
        The number of data workers to use
    transform : str
        If not `None`, a string representing the transformation to apply to the data.
    normalize : str
        If not `None`, a string representing the normalization to apply to the data.
        Valid values are:
        - linear: rescales the data linearly
        - quantile_{n}: rescales the data using a piecewise-linear quantile transformation,
            the string is parsed to obtain the number of quantiles.
            E.g. specify 'quantile_8' for 8 quantiles.
    """
    path: str = omegaconf.MISSING
    key: Optional[str] = None
    num_channels: Optional[int] = None
    batches_per_epoch: Optional[int] = None
    num_workers: int = 16
    load_field_on_gpu: bool = False
    transform: Optional[str] = None
    normalize: Optional[str] = 'linear'

@dataclasses.dataclass
class BaseTrainingConfig:
    gpus: int = 1
    precision: str = '32'
    batch_size: int = 2 ** 20
    max_epochs: int = 100
    lightning: Dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class TrainingConfig(BaseTrainingConfig):
    defaults: List[Any] = dataclasses.field(default_factory=lambda: ['_self_', {'loss': 'l1'}])
    loss: LossConfig = omegaconf.MISSING
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    data: DataConfig = DataConfig()
