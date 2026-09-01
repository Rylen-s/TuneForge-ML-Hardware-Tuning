from __future__ import annotations

import numpy as np
import torch

from .models import PerformanceSurrogate
from .simulator import HardwareTuningSimulator, configuration_space


@torch.no_grad()
def evaluate(model: PerformanceSurrogate, tasks: int = 100, seed: int = 100, noise: float = 0.0) -> dict[str, float]:
    """Measures configuration-selection regret versus exhaustive valid search."""
    model.eval(); device = next(model.parameters()).device
    simulator, configs = HardwareTuningSimulator(seed), configuration_space()
    regrets, satisfied, latency_errors = [], [], []
    for _ in range(tasks):
        workload, hardware = simulator.sample_workload()
        feature_matrix = np.asarray([np.concatenate([workload, hardware, config.features()]) for config in configs])
        predicted = model.predict(torch.tensor(feature_matrix, dtype=torch.float32, device=device)).cpu().numpy()
        actual = np.asarray([simulator.benchmark(workload, hardware, config, noise=noise) for config in configs])
        feasible = (actual[:, 1] <= 12 * hardware[2]) & (actual[:, 2] >= 0.97)
        selected = int(np.argmin(np.where(feasible, predicted[:, 0], np.inf)))
        oracle = float(actual[feasible, 0].min())
        regrets.append(float(actual[selected, 0] / oracle - 1.0))
        satisfied.append(float(feasible[selected]))
        latency_errors.append(float(abs(predicted[selected, 0] - actual[selected, 0])))
    return {"mean_regret": float(np.mean(regrets)), "constraint_satisfaction": float(np.mean(satisfied)), "selected_latency_mae_ms": float(np.mean(latency_errors))}
