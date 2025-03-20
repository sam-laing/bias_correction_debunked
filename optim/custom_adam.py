from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]


import torch
from torch.optim.optimizer import Optimizer

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import math

ParamGroup = Dict[str, Any]


def compute_statistics(t1: torch.Tensor, t2: torch.Tensor) -> Dict[str, float]:
    t1, t2 = t1.view(-1), t2.view(-1)

    norm1= torch.norm(t1)
    norm2= torch.norm(t2)

    inner_product = torch.dot(t1, t2)
    cosine_similarity = inner_product / (norm1 * norm2)
    cosine_similarity = max(min(cosine_similarity, 1.0), -1.0)

    angle = torch.acos(cosine_similarity) * 180 / 3.14159265359

    return {
        "inner_product": inner_product.item(),
        "norm_1": norm1.item(),
        "norm_2": norm2.item(),
        "cosine_similarity": cosine_similarity.item(),
        "angle": angle.item()
    }



class CustomAdamW(Optimizer):
    """
    A modified version of AdamW optimizer with an option
    to disable bias correction and initialize moments directly with gradients.
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        eps: float = 1e-8,
        betas: Betas2 = (0.9, 0.99),
        do_bias_correction: bool = False,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            do_bias_correction=do_bias_correction,
        )
        super().__init__(params, defaults)

    def step(self, closure: LossClosure = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")

                # Apply weight decay (decoupled as in AdamW)
                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    if group["do_bias_correction"]:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    else:
                        state["exp_avg"] = grad.clone()
                        state["exp_avg_sq"] = grad.pow(2).clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Update moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction if enabled
                if group["do_bias_correction"]:
                    exp_avg.div_(1 - beta1 ** state["step"])
                    exp_avg_sq.div_(1 - beta2 ** state["step"])

                # Parameter update
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-group["lr"])

        return loss
