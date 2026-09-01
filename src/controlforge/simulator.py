from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Configuration:
    batch_size: int
    precision: str
    compiled: bool
    layout: str
    streams: int

    def features(self) -> np.ndarray:
        return np.array([
            np.log2(self.batch_size) / 8,
            {"fp32": 0.0, "bf16": 0.5, "int8": 1.0}[self.precision],
            float(self.compiled),
            float(self.layout == "channels_last"),
            self.streams / 4,
        ], dtype=np.float32)


class HardwareTuningSimulator:
    """Deterministic stand-in for an expensive profiler/benchmark service.

    Workloads contain [model_size, sequence_length, arithmetic_intensity,
    memory_intensity]; devices contain [compute, bandwidth, memory_capacity].
    Outputs are latency-ms, memory-GB, and quality retention in [0, 1].
    """
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def sample_workload(self) -> tuple[np.ndarray, np.ndarray]:
        workload = self.rng.uniform([0.15, 0.05, 0.1, 0.1], [1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        device = self.rng.uniform([0.4, 0.4, 0.45], [1.2, 1.2, 1.2]).astype(np.float32)
        return workload, device

    def benchmark(self, workload: np.ndarray, device: np.ndarray, config: Configuration, noise: float = 0.0) -> np.ndarray:
        size, sequence, arithmetic, memory = workload
        compute, bandwidth, capacity = device
        precision_speed = {"fp32": 1.0, "bf16": 0.68, "int8": 0.45}[config.precision]
        precision_memory = {"fp32": 1.0, "bf16": 0.56, "int8": 0.31}[config.precision]
        compile_gain = 0.80 if config.compiled else 1.0
        layout_gain = 0.88 if config.layout == "channels_last" and memory > arithmetic else 1.0
        parallel_gain = 1.0 / (1.0 + 0.18 * (config.streams - 1))
        batch_penalty = 0.55 + 0.18 * np.log2(config.batch_size)
        latency = 22 * size * (arithmetic / compute + memory / bandwidth) * precision_speed * compile_gain * layout_gain * parallel_gain * batch_penalty
        memory_gb = 3.2 * size * sequence * config.batch_size / 16 * precision_memory + 0.4 * config.streams
        quality = 1.0 - ({"fp32": 0.0, "bf16": 0.003, "int8": 0.025}[config.precision]) - 0.006 * (config.streams - 1)
        oom = memory_gb > 12 * capacity
        result = np.array([latency, memory_gb, max(0.0, quality)], dtype=np.float32)
        if oom: result[:2] = [5000.0, memory_gb]
        if noise: result[:2] *= self.rng.normal(1.0, noise, 2).astype(np.float32)
        return result


def configuration_space() -> list[Configuration]:
    return [Configuration(batch, precision, compiled, layout, streams)
            for batch in (1, 2, 4, 8, 16, 32)
            for precision in ("fp32", "bf16", "int8")
            for compiled in (False, True)
            for layout in ("contiguous", "channels_last")
            for streams in (1, 2, 4)]
