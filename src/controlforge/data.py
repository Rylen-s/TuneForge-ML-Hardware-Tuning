from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .simulator import HardwareTuningSimulator, configuration_space


class BenchmarkDataset(Dataset):
    """Reproducible benchmark-log data engine with a stable feature contract."""
    def __init__(self, records: int = 8_000, seed: int = 7, noise: float = 0.0):
        simulator, configs = HardwareTuningSimulator(seed), configuration_space()
        features, targets = [], []
        for _ in range(records):
            workload, device = simulator.sample_workload()
            config = configs[simulator.rng.integers(len(configs))]
            features.append(np.concatenate([workload, device, config.features()]))
            targets.append(simulator.benchmark(workload, device, config, noise=noise))
        self.features = torch.tensor(np.asarray(features), dtype=torch.float32)
        self.targets = torch.tensor(np.asarray(targets), dtype=torch.float32)

    def __len__(self) -> int: return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index], "targets": self.targets[index]}
