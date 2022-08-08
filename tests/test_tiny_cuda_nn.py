"""Test for implementations based on tinycudann
"""

from finfc import models
import torch

def test_interpolator_tinycudann():
    hashed_grid = models.TCNNMultigridInterpolator(4, 128, [4, 4], [32, 32])
    points = torch.rand(5, 2)

    result = hashed_grid(points)
    assert result is not None
