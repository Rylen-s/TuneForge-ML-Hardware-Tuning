# TuneForge — Evidence-Grounded Hardware Performance Tuning

`TuneForge` is the ML core for an evidence-grounded agent that diagnoses and tunes ML workloads for a target hardware budget. Given a workload description, a device profile, and measured profiler signals, it selects legal runtime configurations—batch size, precision, compilation, memory layout, and parallelism—to meet a latency/throughput/memory/quality objective.

This is a real, bounded problem: choosing serving and training settings is expensive, highly hardware-dependent, and normally requires repeated manual profiling. Every proposed configuration is verified by an executable benchmark harness; the agent is not rewarded for plausible explanations.

## Why this is a better project

The original GRPO/MCP idea is valuable, but an arbitrary tool-use sandbox risks reading as generic LLM post-training. TuneForge keeps the agentic and verifiable aspects while giving them a concrete systems-and-ML purpose. It also creates first-class evidence for the training skills a recruiter asked for.

| Skill | Concrete project component |
| --- | --- |
| Architecture design | Workload/device/profile encoder with GRU/Transformer and uncertainty-aware performance heads |
| Losses and training dynamics | Heteroscedastic latency regression, pairwise ranking, constraint-violation penalty, calibration metrics |
| Curriculum | Synthetic clean profiles → injected measurement noise → real device traces → out-of-distribution workloads |
| Optimizers | Controlled AdamW, Lion, and Shampoo-style preconditioner ablation at matched token/example budget |
| Pre-training → RL | Train a performance surrogate on benchmark logs; SFT an agent on successful traces; GRPO on verifier rewards |
| Data engine | Versioned workload/config/device records, profiling queue, deduplication, data-quality checks, hard-example mining |
| Evaluation | Held-out workload families/devices, constraint satisfaction, regret versus exhaustive search, calibration, tool-use accuracy |

## Project shape

```text
workload + device spec + profile trace
                 │
          performance surrogate
       predicts latency / memory / quality
                 │
      agent chooses profiling or tuning tools
                 │
      executable benchmark verifier ──► reward + new trace
```

The initial local environment is a deterministic hardware-cost simulator so supervised training and tests are reproducible. The included code covers the performance-surrogate/data/evaluation core; the tool-using LLM is the third milestone. Swap the simulator for adapters that call `torch.profiler`, Triton benchmark scripts, vLLM, or a real serving endpoint once the data contract is stable.

## Milestones

1. **Supervised baseline:** collect configuration trials and train a calibrated latency/memory surrogate; report MAE, rank correlation, and constraint calibration.
2. **Search policy:** compare random search, Bayesian optimization, surrogate-guided beam search, and a prompted tool agent on regret at fixed benchmark budget.
3. **Agent finetuning:** SFT on successful tuning traces, then GRPO with exact verifier rewards (valid configuration, constraints met, improvement versus baseline). Keep LLM judging out of the primary reward.
4. **Robustness:** test unseen workload shapes, noisy/missing counters, and a distinct GPU/CPU architecture. Report failures, not just average wins.
5. **Scale:** distributed data collection plus DDP/FSDP after profiling demonstrates a need; quantify examples/sec, time-to-target quality, and cost.

## Resume-quality claim, after measurement

> Built an evidence-grounded hardware tuning agent that used profiler-derived telemetry to select ML runtime configurations under latency, memory, and quality constraints. Trained an uncertainty-aware performance surrogate with ranking and constraint-aware losses, then fine-tuned a tool-using policy with verifier-based GRPO; evaluated regret, constraint satisfaction, and cross-workload robustness against exhaustive and random-search baselines.

Use real task counts, hardware, and held-out measurements in the final bullet—never placeholder percentages.
