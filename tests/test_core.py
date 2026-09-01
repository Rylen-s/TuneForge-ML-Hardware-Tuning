import torch

from controlforge.data import BenchmarkDataset
from controlforge.evaluate import evaluate
from controlforge.models import PerformanceSurrogate
from controlforge.simulator import HardwareTuningSimulator, configuration_space


def test_benchmark_dataset_contract():
    record = BenchmarkDataset(records=3)[0]
    assert record["features"].shape == (12,)
    assert record["targets"].shape == (3,)


def test_simulator_penalizes_out_of_memory_configuration():
    simulator = HardwareTuningSimulator(4)
    workload, hardware = simulator.sample_workload()
    largest = max(configuration_space(), key=lambda item: item.batch_size)
    result = simulator.benchmark(workload, hardware, largest)
    assert result.shape == (3,)


def test_surrogate_and_evaluator_execute():
    model = PerformanceSurrogate()
    mean, variance = model(torch.randn(2, 12))
    assert mean.shape == variance.shape == (2, 3)
    result = evaluate(model, tasks=2)
    assert set(result) == {"mean_regret", "constraint_satisfaction", "selected_latency_mae_ms"}
