from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    seed: int = 7
    device: str = "cpu"
    architecture: str = "transformer"
    optimizer: str = "adamw"
    feature_dim: int = 12
    hidden_dim: int = 96
    layers: int = 2
    batch_size: int = 128
    steps: int = 1500
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    curriculum_steps: int = 800
    profile_noise: float = 0.02
    ranking_weight: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)
