"""Module used to compute properties of fields.
"""

from __future__ import annotations

from typing import Collection, List

import numpy as np
import torch


def _fftk(shape: Collection[int], boxsize: float, symmetric=True) -> List[np.ndarray]:
    k = []
    for d in range(len(shape)):
        kd = np.fft.fftfreq(shape[d])
        kd *= 2 * np.pi / boxsize * shape[d]

        kdshape = np.ones(len(shape), dtype=int)

        if symmetric and d == len(shape) - 1:
            kd = kd[:shape[d] // 2 + 1]

        kdshape[d] = len(kd)

        kd = kd.reshape(kdshape)
        k.append(kd)
    return k

def power_spectrum(field: np.ndarray, boxsize: float=1.0, k=None, symmetric=True):
    """Computes power spectrum from given array."""
    field = torch.as_tensor(field)

    # Torch fft is much faster than numpy, so use that instead
    c1: torch.Tensor = torch.fft.rfftn(field)
    c1.div_(c1[(0,) * len(c1.shape)].item())
    c1[(0,) * len(c1.shape)] = 0

    x = c1.abs().square_()

    if k is None:
        k = _fftk(field.shape, boxsize, symmetric)
        k = np.sqrt(sum(kk ** 2 for kk in k))

    k = torch.from_numpy(k).to(dtype=x.dtype)

    # CPU device required for histogram
    x = x.to(device='cpu')

    H, edges = torch.histogram(k, weight=x, bins=field.shape[0])
    N, edges = torch.histogram(k, bins=edges)
    center = edges[1:] + edges[:-1]
    power = H * boxsize ** 3 / N

    power[0] = np.nan

    return 0.5 * center, power


