from __future__ import annotations

from typing import Iterable

import torch

from ._configs import PreprocessorConfig, LogPreprocessorConfig, RescalePreprocessorConfig


class ComposePreprocessor(torch.nn.Module):
    """Utility to compose preprocessing steps.

    """
    def __init__(self, transforms: Iterable[torch.nn.Module]):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transform in reversed(self.transforms):
            x = transform(x)
        return x

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform.preprocess(x)
        return x


class LogPreprocessor(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.expm1()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.log1p()


class RescalePreprocessor(torch.nn.Module):
    """Preprocessor corresponding to a single linear rescaling.
    """
    def __init__(self, min_value: float, max_value: float, output_min: float=-1, output_max: float=1):
        super().__init__()

        self.min_value = min_value
        self.max_value = max_value
        self.output_min = output_min
        self.output_max = output_max

        self._a = (output_max - output_min) / (max_value - min_value)
        self._b = output_min - self._a * min_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.add(-self._b / self._a, x, alpha=1 / self._a)

    def preprocess(self, y: torch.Tensor) -> torch.Tensor:
        return torch.add(self._b, y, alpha=self._a)

class LazyRescalePreprocessor(torch.nn.Module):
    max_value: torch.Tensor
    min_value: torch.Tensor

    def __init__(self, output_min: float=-1, output_max: float=1):
        super().__init__()

        self.output_min = output_min
        self.output_max = output_max

        self.register_buffer('min_value', torch.tensor(0.0))
        self.register_buffer('max_value', torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = (self.max_value - self.min_value) / (self.output_max - self.output_min)
        b = self.min_value - a * self.output_min
        return x.mul(a).add(b)

    def preprocess(self, y: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.min_value.copy_(y.min())
            self.max_value.copy_(y.max())

        a = (self.output_max - self.output_min) / (self.max_value - self.min_value)
        b = self.output_min - a * self.min_value

        return y.mul(a).add(b)


def interpolate_piecewise_linear(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Evaluate piecewise linear function.

    This function is similar to `np.interp` implemented in pytorch.
    It evaluates a piecewise linear function defined implicitly by its value `fp`
    at locations `xp`.
    """
    slope = fp.new_zeros(len(fp) + 1)
    slope[1:-1] = fp.diff().div_(xp.diff())

    buckets = torch.bucketize(x, xp)
    x_slope = slope[buckets]

    buckets.sub_(1)
    x_lb = xp[buckets]
    f_lb = fp[buckets]

    if x.requires_grad:
        return x.sub(x_lb).mul(x_slope).add(f_lb)
    else:
        return x.sub(x_lb).mul_(x_slope).add_(f_lb)


class LinearQuantilePreprocessor(torch.nn.Module):
    quantiles: torch.Tensor
    breakpoints: torch.Tensor

    def __init__(self, quantiles: torch.Tensor, min=-1, max=1):
        super().__init__()

        self.register_parameter('quantiles', torch.nn.Parameter(quantiles))
        self.register_buffer('breakpoints', torch.linspace(min, max, len(quantiles)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return interpolate_piecewise_linear(x, self.breakpoints, self.quantiles)

    def preprocess(self, y: torch.Tensor) -> torch.Tensor:
        return interpolate_piecewise_linear(y, self.quantiles, self.breakpoints)


class LazyLinearQuantilePreprocessor(torch.nn.Module):
    quantiles: torch.Tensor
    breakpoints: torch.Tensor

    def __init__(self, num_quantiles: int, min=-1, max=1):
        super().__init__()

        self.register_buffer('quantiles', torch.zeros(num_quantiles))
        self.register_buffer('breakpoints', torch.linspace(min, max, num_quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return interpolate_piecewise_linear(x, self.breakpoints, self.quantiles)

    def preprocess(self, y: torch.Tensor) -> torch.Tensor:
        if self.training:
            y_flat = y.view(-1)

            num_sample_points = min(2 ** 16, len(y_flat))
            sample_idx = torch.randint(0, len(y_flat), (num_sample_points,))
            sample_y = y_flat[sample_idx]

            # Set quantiles to the values computed from the sample.
            quantile_probs = torch.linspace(0, 1, len(self.quantiles), device=y.device, dtype=y.dtype)[1:-1]
            self.quantiles[1:-1] = torch.quantile(sample_y, quantile_probs)

            # Do not sample minimum and maximum values
            self.quantiles[0] = torch.min(y)
            self.quantiles[-1] = torch.max(y)

        return interpolate_piecewise_linear(y, self.quantiles, self.breakpoints)


def make_from_config(config: PreprocessorConfig):
    # Use name-based lookup for compatibility with omegaconf structured configs
    # which may change the type of the configuration.

    if config.name == 'log':
        return LogPreprocessor()
    elif config.name == 'rescale':
        return LazyRescalePreprocessor(
            output_min=config.output_min,
            output_max=config.output_max)
    elif config.name == 'quantile':
        return LazyLinearQuantilePreprocessor(
            num_quantiles=config.num_quantiles,
            min=config.output_min,
            max=config.output_max)
    else:
        raise ValueError(f"Unknown preprocessor config: {config}")

def make_from_config_list(configs: Iterable[PreprocessorConfig]):
    return ComposePreprocessor([make_from_config(config) for config in configs])
