# TuneForge

TuneForge is platform for hardware-aware ML performance tuning. It trains a surrogate model to predict a workload configuration's latency, memory use, and quality retention, then chooses the fastest configuration that meets memory and quality constraints.

The included simulator makes experiments reproducible on a laptop. It is designed to later accept measurements from real profilers or benchmark scripts.

## Project flow

```text
workload + hardware + runtime configuration
                    │
                    ▼
          benchmark simulator / profiler
          latency, memory, quality labels
                    │
                    ▼
           PyTorch performance surrogate
        predicts metrics and uncertainty
                    │
                    ▼
    select lowest-latency valid configuration
                    │
                    ▼
     evaluate regret vs exhaustive search
```

## Setup

Requires Python 3.10+ and PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Run

Train the default Transformer surrogate:

```bash
controlforge train --steps 1500 --architecture transformer --optimizer adamw
```

Try the MLP or optimizer alternatives:

```bash
controlforge train --steps 1500 --architecture mlp --optimizer lion
controlforge train --steps 1500 --architecture transformer --optimizer shampoo_lite
```

Training writes the latest model and metrics to `runs/latest.pt`.

Evaluate the checkpoint with simulated noisy profiling measurements:

```bash
controlforge evaluate --checkpoint runs/latest.pt --profile-noise 0.03
```

Run the tests:

```bash
pytest
```

## File map

```text
src/controlforge/
├── cli.py          command-line train and evaluate entry points
├── config.py       experiment configuration
├── simulator.py    deterministic workload/hardware benchmark environment
├── data.py         benchmark-record dataset
├── models.py       MLP and feature-Transformer performance surrogate
├── losses.py       uncertainty-aware regression and ranking losses
├── optim.py        AdamW, Lion, and Shampoo-lite optimizer selection
├── train.py        training loop and noise curriculum
└── evaluate.py     regret and constraint-satisfaction evaluation

tests/test_core.py  core dataset, model, simulator, and evaluator checks
```

## Metrics

- `mean_regret`: latency gap between the selected configuration and the best valid configuration.
- `constraint_satisfaction`: fraction of selections meeting memory and quality requirements.
- `selected_latency_mae_ms`: latency prediction error for the selected configuration.
