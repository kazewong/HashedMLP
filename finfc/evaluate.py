"""Script and functionality to evaluate encoded field.
"""

from __future__ import annotations
import copy
import dataclasses
import functools
import logging
import os

from typing import Callable, List, Optional, Sequence

import hydra
import omegaconf
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import properties, train


def make_sample_grid(grid: Sequence[int], box: Sequence[float]=None, device=None) -> torch.Tensor:
    """Creates a vector of grid locations corresponding to a n-dimensional grid.
    """
    if box is None:
        box = [1.0] * len(grid)

    samples = [torch.arange(g, device=device, dtype=torch.float32).add_(0.5).mul_(b / g) for b, g in zip(box, grid)]
    return torch.cartesian_prod(*samples)


def sample_field(field: Callable[[torch.Tensor], torch.Tensor], grid: Sequence[int], box: Sequence[float]=None, device=None):
    """Samples a given field at the given grid resolution.
    """
    samples = make_sample_grid(grid, box=box, device=device)

    sample_batches = torch.split(samples, 2 ** 20)

    result = []

    for b in sample_batches:
        result.append(field(b))

    result = torch.cat(result, dim=0)
    return result.view(*grid)


def load_reference(path, boxsize: float=1.0):
    data = np.load(path)
    data = np.squeeze(data, -1)
    ps = properties.power_spectrum(data, boxsize=boxsize)
    return data, ps


def plot_power_spectrum(center_ref, ps_ref, center, ps, title: str=None):
    fig, ax = plt.subplots()
    ax.loglog(center_ref, ps_ref, label='ref')
    ax.loglog(center, ps, label='compressed')
    ax.set_xlabel('k (h / Mpc)')
    ax.set_ylabel('P(k) (Mpc / h)^3')

    if title is not None:
        ax.set_title(title)
    ax.legend()

    return fig


def plot_field_difference(slice_compressed, slice_ref):
    vmin = min(np.min(slice_compressed), np.min(slice_ref))
    vmax = max(np.max(slice_compressed), np.max(slice_ref))

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    ax1.imshow(slice_compressed, vmin=vmin, vmax=vmax, cmap='viridis')
    ax1.set_title('Compressed')
    ax2.imshow(slice_ref, vmin=vmin, vmax=vmax, cmap='viridis')
    ax2.set_title('Reference')
    ax3.imshow(slice_compressed - slice_ref, cmap='viridis')
    ax3.set_title('Difference')

    return fig


def compute_l1_difference(field: torch.Tensor, reference: torch.Tensor, preprocessor: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        field_p = preprocessor(field)
        reference_p = preprocessor(reference)
        l1 = torch.nn.functional.l1_loss(field_p, reference_p)

    return l1


@dataclasses.dataclass
class EvaluationConfig:
    checkpoint_path: str = omegaconf.MISSING
    grid_size: Optional[List[int]] = None
    boxsize: float = 1.0
    save_field: bool = False


@hydra.main(config_name='conf', config_path=None)
def main(config: EvaluationConfig):
    logger = logging.getLogger(__name__)

    model: train.NeuralFieldModel = train.NeuralFieldModel.load_from_checkpoint(
        hydra.utils.to_absolute_path(config.checkpoint_path))

    reference_path = hydra.utils.to_absolute_path(model.hparams.data.path)
    logger.info('Loading reference data from %s', reference_path)
    data_ref, ps_ref = load_reference(reference_path, config.boxsize)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dimension = len(data_ref.shape)
    logger.info('Dimension of reference data: %d', dimension)

    if config.grid_size is not None:
        logger.info('Overriding grid size to %s', config.grid_size)
        grid_size = config.grid_size
    else:
        grid_size = model.hparams.model.hash.final_grid
        logger.info('Using grid size from model: %s', grid_size)

    if model.hparams.data.transform == 'log':
        logger.info('Visualizing in the logarithmic domain')
        transform = np.log1p
    else:
        transform = lambda x: x

    box_size = model.hparams.model.hash.box_size

    model = model.eval()
    model = model.to(device=device)

    with torch.no_grad():
        field = sample_field(functools.partial(model, normalized=False), grid_size, box=box_size, device=device)
        field = field.cpu()

    l1 = compute_l1_difference(
        field,
        torch.from_numpy(data_ref),
        copy.deepcopy(model.preprocessor).cpu().preprocess)
    logging.info('Average L1 difference: %f', l1.item())

    ps = properties.power_spectrum(field, boxsize=config.boxsize)

    ps_ref_center, ps_ref_power = ps_ref
    ps_center, ps_power = ps

    fig = plot_power_spectrum(
        ps_ref_center, ps_ref_power, ps_center, ps_power,
        title=f'')
    fig.set_size_inches(8, 6)
    fig.savefig('power_spectrum.pdf', bbox_inches='tight')
    fig.savefig('power_spectrum.png', bbox_inches='tight', facecolor='w')

    slice_compressed = transform(field[..., field.shape[-1] // 2].numpy())
    slice_ref = transform(data_ref[..., data_ref.shape[-1] // 2])

    fig = plot_field_difference(slice_compressed, slice_ref)
    fig.set_size_inches(19, 6)
    fig.savefig('fields.png', bbox_inches='tight', facecolor='w')

    save_path = os.path.abspath('power_spectrum.npz')
    logger.info('Saving power spectrum to %s', save_path)

    np.savez(save_path,
        ps_ref_center=ps_ref_center,
        ps_ref_power=ps_ref_power,
        ps_center=ps_center,
        ps_power=ps_power)

    if config.save_field:
        nf_path = os.path.abspath('field.npy')
        logger.info('Saving neural field to %s', nf_path)
        np.save(nf_path, field.to(dtype=torch.float32).numpy())


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('conf', node=EvaluationConfig)
    main()

