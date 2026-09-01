from __future__ import annotations

import torch


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 3e-4, betas=(0.9, 0.99), weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None: continue
                state = self.state[parameter]
                if not state: state["momentum"] = torch.zeros_like(parameter)
                momentum, grad = state["momentum"], parameter.grad
                update = momentum.lerp(grad, 1 - group["betas"][0])
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.sign(), alpha=-group["lr"])
                momentum.lerp_(grad, 1 - group["betas"][1])
        return loss


def build_optimizer(name: str, parameters, lr: float, weight_decay: float):
    if name == "adamw": return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "lion": return Lion(parameters, lr=lr, weight_decay=weight_decay)
    if name == "shampoo_lite": return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=(0.95, 0.999))
    raise ValueError("optimizer must be adamw, lion, or shampoo_lite")
