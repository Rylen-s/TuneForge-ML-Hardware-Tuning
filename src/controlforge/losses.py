from __future__ import annotations

import torch
from torch.nn import functional as F


def heteroscedastic_loss(mean: torch.Tensor, log_variance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inverse_variance = torch.exp(-log_variance)
    return (0.5 * (inverse_variance * (mean - target).square() + log_variance)).mean()


def latency_ranking_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pairwise ranking preserves configuration-selection quality, not only MAE."""
    predicted_delta = prediction[:, None, 0] - prediction[None, :, 0]
    target_order = torch.sign(target[:, None, 0] - target[None, :, 0])
    upper = torch.triu(torch.ones_like(predicted_delta, dtype=torch.bool), diagonal=1)
    return F.softplus(predicted_delta[upper] * -target_order[upper]).mean()
