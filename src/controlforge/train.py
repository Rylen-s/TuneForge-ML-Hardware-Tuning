from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import BenchmarkDataset
from .losses import heteroscedastic_loss, latency_ranking_loss
from .models import PerformanceSurrogate
from .optim import build_optimizer


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train(config: TrainConfig) -> tuple[PerformanceSurrogate, list[dict]]:
    set_seed(config.seed)
    device = torch.device(config.device)
    dataset = BenchmarkDataset(seed=config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    model = PerformanceSurrogate(config.feature_dim, config.hidden_dim, config.layers, config.architecture).to(device)
    optimizer = build_optimizer(config.optimizer, model.parameters(), config.lr, config.weight_decay)
    history, iterator = [], iter(loader)
    for step in range(config.steps):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        features, targets = batch["features"].to(device), batch["targets"].to(device)
        # Curriculum gradually exposes noisy profiler measurements after clean synthetic records.
        noise = config.profile_noise * min(1.0, step / max(config.curriculum_steps, 1))
        features = features + torch.randn_like(features) * noise
        mean, log_variance = model(features)
        loss = heteroscedastic_loss(mean, log_variance, targets) + config.ranking_weight * latency_ranking_loss(mean, targets)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip); optimizer.step()
        if step % 100 == 0 or step == config.steps - 1:
            history.append({"step": step, "loss": float(loss.detach().cpu()), "profile_noise": noise})
    return model, history
