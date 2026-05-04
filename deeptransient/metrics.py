"""Evaluation metrics for TransientNet and CloudyNet."""

from __future__ import annotations

import numpy as np
import torch


def mean_squared_error_per_attribute(
    preds: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Return per-attribute MSE averaged over the batch.

    Args:
        preds:   Predicted attribute scores ``(N, 40)``.
        targets: Ground-truth attribute scores ``(N, 40)``.

    Returns:
        Per-attribute MSE tensor of shape ``(40,)``.
    """
    return ((preds - targets) ** 2).mean(dim=0)


def mean_attribute_error(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Return the scalar mean MSE across all attributes (as a percentage).

    This matches the primary evaluation metric reported in the paper
    (Table 2): average per-attribute MSE × 100.

    Args:
        preds:   ``(N, 40)`` predicted scores.
        targets: ``(N, 40)`` ground-truth scores.

    Returns:
        Mean MSE in percentage (lower is better).
    """
    mse = mean_squared_error_per_attribute(preds, targets)
    return float(mse.mean().item()) * 100.0


def normalised_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Return the normalised accuracy used by Lu et al. (2014).

    Normalised accuracy is defined as::

        norm_acc = max((accuracy - 0.5) / 0.5, 0)

    where *accuracy* is the fraction of correctly classified images.

    Args:
        preds:   Class logits or probabilities ``(N, 2)``.
        targets: Integer class labels ``(N,)``; ``0`` = sunny, ``1`` = cloudy.

    Returns:
        Normalised accuracy in ``[0, 1]``.
    """
    pred_class = preds.argmax(dim=1)
    acc = (pred_class == targets).float().mean().item()
    return max((acc - 0.5) / 0.5, 0.0)


def five_fold_normalised_accuracy(
    fold_accuracies: list[float],
) -> tuple[float, float]:
    """Compute mean ± std of normalised accuracy over five folds.

    Args:
        fold_accuracies: List of raw (un-normalised) per-fold accuracies.

    Returns:
        ``(mean_norm_acc, std_norm_acc)`` both as percentages.
    """
    norm = [max((a - 0.5) / 0.5, 0.0) * 100.0 for a in fold_accuracies]
    return float(np.mean(norm)), float(np.std(norm))
