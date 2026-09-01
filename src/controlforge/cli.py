from __future__ import annotations

import argparse
from pathlib import Path
import torch

from .config import TrainConfig
from .evaluate import evaluate
from .models import PerformanceSurrogate
from .train import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="controlforge")
    commands = parser.add_subparsers(dest="command", required=True)
    train_command = commands.add_parser("train")
    train_command.add_argument("--steps", type=int, default=1500)
    train_command.add_argument("--architecture", choices=["mlp", "transformer"], default="transformer")
    train_command.add_argument("--optimizer", choices=["adamw", "lion", "shampoo_lite"], default="adamw")
    train_command.add_argument("--seed", type=int, default=7)
    evaluate_command = commands.add_parser("evaluate")
    evaluate_command.add_argument("--checkpoint", required=True)
    evaluate_command.add_argument("--profile-noise", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        config = TrainConfig(steps=args.steps, architecture=args.architecture, optimizer=args.optimizer, seed=args.seed)
        model, history = train(config)
        metrics = evaluate(model, seed=config.seed)
        Path("runs").mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "config": config.to_dict(), "history": history, "metrics": metrics}, "runs/latest.pt")
        print({"metrics": metrics, "last_train": history[-1], "checkpoint": "runs/latest.pt"})
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = TrainConfig(**checkpoint["config"])
        model = PerformanceSurrogate(config.feature_dim, config.hidden_dim, config.layers, config.architecture)
        model.load_state_dict(checkpoint["model"])
        print(evaluate(model, noise=args.profile_noise))


if __name__ == "__main__": main()
