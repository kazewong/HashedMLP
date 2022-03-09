"""This module contains functionality to apply different types of losses and collect various
metrics for regression problems.

"""

from __future__ import annotations

import abc

import torch
import torch.nn
import torch.nn.functional

class AbstractLoss(abc.ABC, torch.nn.Module):
    @property
    @abc.abstractmethod
    def num_outputs(self) -> int:
        """Property representing the number of model outputs required for prediction of this loss.

        Depending on the type of loss (e.g. l1 vs cross-entropy), the loss may require more than
        a single scalar output. This value corresponds to the last dimension of the output array
        passed to this loss's `__call__` method.
        """
        pass

    @abc.abstractmethod
    def forward(self, output: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the predicted value and loss for the given abstract loss.

        Parameters
        ----------
        output : torch.Tensor
            Output from the model used for prediction
        y_true : torch.Tensor
            Representation of the true value obtained from the `preprocess` function.

        Returns
        -------
        loss : torch.Tensor
            Scalar jnp array representing the average loss.
        predicted : torch.Tensor
            jnp array representing the predicted value at each element of the batch.
        """

    @abc.abstractmethod
    def predict(self, output: torch.Tensor) -> torch.Tensor:
        """Computes the predicted value for the given output."""

    def preprocess(self, y_true: torch.Tensor) -> torch.Tensor:
        """Preprocess the true label into a form more suitable for use by
        this loss.

        This function may be used to preprocess true labels to specific
        encodings which may be lead to more efficient computation.

        By default, this function simply returns the value unchanged.
        """
        return y_true

    def __call__(self, output: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(output, y_true, weight)


def _reduce_with_weights(losses, weights=None):
    if weights is None:
        return torch.mean(losses)

    return torch.sum(losses * weights) / torch.sum(weights)


class L1Loss(AbstractLoss):
    lower: float
    upper: float

    def __init__(self, lower: float, upper: float):
        super().__init__()
        self.lower = lower
        self.upper = upper

    @property
    def num_outputs(self) -> int:
        return 1

    def forward(self, output: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        output = output.squeeze(-1)
        y_pred = self.predict(output)

        if weights is None:
            loss = torch.nn.functional.l1_loss(y_pred, y_true)
        else:
            losses = torch.abs(y_pred - y_true)
            loss = _reduce_with_weights(losses, weights)

        return loss, y_pred

    def predict(self, output):
        y_pred = torch.sigmoid(output) * (1.05 * (self.upper - self.lower)) + self.lower
        return y_pred

    def preprocess(self, y_true):
        return y_true


class CrossEntropyLoss(AbstractLoss):
    num_classes: int

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @property
    def num_outputs(self) -> int:
        return self.num_classes

    def forward(self, output: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        y_pred = self.predict(output)

        if weights is not None:
            losses = torch.nn.functional.cross_entropy(output, y_true, reduction='none')
            loss = _reduce_with_weights(losses, weights)
        else:
            loss = torch.nn.functional.cross_entropy(output, y_true)

        return loss, y_pred

    def predict(self, output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(output, axis=-1)

    def preprocess(self, y_true: torch.Tensor):
        return y_true


class CrossEntropyRegressionLoss(AbstractLoss):
    """This loss implements a cross-entropy loss for regression problems
    based on a linear discretization of the target prediction space.
    """

    lower: float
    upper: float
    num_grid: int = 50

    def __init__(self, lower: float, upper: float, num_grid: int):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.num_grid = num_grid

    @property
    def num_outputs(self) -> int:
        return self.num_grid

    def forward(self, output: torch.Tensor, y_true_onehot: torch.Tensor, weights: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        output_logits = torch.nn.functional.log_softmax(output, dim=-1)
        losses = (output_logits * y_true_onehot).sum(axis=-1).mul(-1)
        loss = _reduce_with_weights(losses, weights)

        # Adjust loss to compute excess deviance.
        loss_saturated = _reduce_with_weights(torch.xlogy(y_true_onehot, y_true_onehot), weights)
        loss = loss - loss_saturated

        y_pred = self.predict_from_probs(output_logits.exp())

        return loss, y_pred

    def predict(self, output: torch.Tensor):
        grid_prob = torch.softmax(output, dim=-1)
        return self.predict_from_probs(grid_prob)

    def predict_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        grid = torch.linspace(self.lower, self.upper, self.num_grid, dtype=probs.dtype, device=probs.device)
        y_pred = torch.sum(probs * grid, axis=-1)
        return y_pred

    def preprocess(self, y_true: torch.Tensor) -> torch.Tensor:
        y_true = torch.as_tensor(y_true)
        grid = torch.linspace(self.lower, self.upper, self.num_grid, dtype=y_true.dtype, device=y_true.device)
        insert_idx = torch.searchsorted(grid, y_true, right=True).sub_(1).clamp_(0, self.num_grid - 2)

        one_hot_left = torch.nn.functional.one_hot(insert_idx, num_classes=self.num_grid)
        one_hot_right = torch.nn.functional.one_hot(insert_idx + 1, num_classes=self.num_grid)

        grid_proportion = torch.unsqueeze((y_true - grid[insert_idx]) / (grid[1] - grid[0]), axis=-1)
        return (1 - grid_proportion) * one_hot_left + grid_proportion * one_hot_right
