"""Tests for models in hashed_neural_field.py."""

import numpy as np
import pytest
import torch

from finfc import models
from finfc import preprocessing


def test_interpolator_with_hashed_grid():
    with torch.random.fork_rng():
        torch.random.manual_seed(42)

        hashed_grid = models.HashedGrid(100, 2)
        interpolator = models.Interpolator(hashed_grid, [64, 64])
        points = torch.rand(5, 2)

        result = interpolator(points)
        assert int(result.sum().detach() * 1e6) == -1332248


def test_interpolator_with_dense_grid():
    with torch.random.fork_rng():
        torch.random.manual_seed(42)

        dense_grid = models.DenseGrid((64, 64), 2)
        interpolator = models.Interpolator(dense_grid, [64, 64])
        points = torch.rand(5, 2)

        result = interpolator(points)
        assert int(result.sum().detach() * 1e6) == -4061850


def test_interpolator_with_dense_grid_rectangular():
    with torch.random.fork_rng():
        torch.random.manual_seed(42)

        box_size = torch.tensor([1.0, 2.0])

        dense_grid = models.DenseGrid((64, 128), 2)
        interpolator = models.Interpolator(dense_grid, [64, 128], box_size)
        points = torch.rand(5, 2) * box_size

        result = interpolator(points)
        assert int(result.sum().detach() * 1e6) == -820403


def test_interpolator_with_dense_grid_anisotropic():
    with torch.random.fork_rng():
        torch.random.manual_seed(42)
        box_size = torch.tensor([1.0, 2.0])

        dense_grid = models.DenseGrid((64, 64), 2)
        interpolator = models.Interpolator(dense_grid, [64, 64], box_size)
        points = torch.rand(10, 2) * box_size

        result = interpolator(points)
        assert int(result.sum().detach() * 1e6) == -5392059


def test_multigrid_interpolator():
    with torch.random.fork_rng():
        torch.random.manual_seed(42)

        interpolator = models.MultigridInterpolator(3, 100, [4, 4], [32, 32])
        points = torch.rand(5, 2)

        result = interpolator(points)
        assert int(result.sum().detach() * 1e6) == -5333495


def test_torch_interpolation():
    with torch.random.fork_rng():
        torch.random.manual_seed(42)

        x = torch.rand((10,))

        breakpoints = torch.linspace(0, 1, steps=5)
        values = torch.rand(breakpoints.shape)

        result = preprocessing.interpolate_piecewise_linear(x, breakpoints, values)
        expected = torch.from_numpy(np.interp(x.numpy(), breakpoints.numpy(), values.numpy())).to(result)
        assert torch.allclose(result, expected)
