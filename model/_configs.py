from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

from cosmology.models import utils


@dataclasses.dataclass
class HashConfig:
    """Configuration for the multigrid hash problem.

    Attributes
    ----------
    base_grid : int
        Size of the grid at the coarsest level.
    num_levels : int
        Number of grid levels.
    num_features : int
        Dimension of feature vector (hash entry)
    num_entries_log2 : int
        Number of entries in the hash table (log 2).
    """
    base_grid: int = 16
    num_levels: int = 8
    num_features: int = 2
    num_entries_log2: int = 24

@dataclasses.dataclass
class ModelConfig:
    hash: HashConfig = HashConfig()
    hidden_size: int = 64
    mlp_layers: int = 2
    sparse: bool = True

@dataclasses.dataclass
class OptimConfig:
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5

@dataclasses.dataclass
class DataConfig:
    path: str = '/home/wzhou/data/fields/camels_IllustrisTNG/snapshot0_lmass.npy'
    num_channels: Optional[int] = None
    batches_per_epoch: Optional[int] = None
    num_workers: int = 16
    load_field_on_gpu: bool = False

@dataclasses.dataclass
class TrainingConfig:
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    data: DataConfig = DataConfig()
    batch_size: int = 2 ** 17
    max_epochs: int = 100
    lightning: Dict[str, Any] = dataclasses.field(default_factory=dict)
