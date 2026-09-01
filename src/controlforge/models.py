from __future__ import annotations

import torch
from torch import nn


class PerformanceSurrogate(nn.Module):
    """Predicts metric means and aleatoric uncertainty from benchmark records."""
    def __init__(self, feature_dim: int = 12, hidden_dim: int = 96, layers: int = 2, architecture: str = "transformer"):
        super().__init__()
        if architecture == "mlp":
            blocks: list[nn.Module] = [nn.Linear(feature_dim, hidden_dim), nn.SiLU()]
            for _ in range(layers - 1):
                blocks += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            self.encoder = nn.Sequential(*blocks)
        elif architecture == "transformer":
            self.scalar_embedding = nn.Linear(1, hidden_dim)
            self.position = nn.Parameter(torch.zeros(1, feature_dim, hidden_dim))
            layer = nn.TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=hidden_dim * 3, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        else:
            raise ValueError("architecture must be mlp or transformer")
        self.architecture = architecture
        self.mean = nn.Linear(hidden_dim, 3)
        self.log_variance = nn.Linear(hidden_dim, 3)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.architecture == "mlp":
            hidden = self.encoder(features)
        else:
            tokens = self.scalar_embedding(features.unsqueeze(-1)) + self.position
            hidden = self.encoder(tokens).mean(dim=1)
        return self.mean(hidden), self.log_variance(hidden).clamp(-8, 5)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self(features)[0]
