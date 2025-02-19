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
from torch import Tensor
import math

ParamGroup = Dict[str, Any]



class CustomAdam(Optimizer):
    def __init__(
            self, 
            params: Params,
            lr: float = 1e-4,
            eps: float = 1e-8,
            betas: Betas2 = (0.9, 0.99),
            do_bias_correction: bool = False,
            weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, do_bias_correction=do_bias_correction)
        super().__init__(params, defaults)     


    def step(self, closure: OptLossClosure = None):
        loss = None  
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue  
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("None of that sparse stuff here")
                
                if group["weight_decay"] != 0:
                    p.data.mul_(1-group["lr"] * group["weight_decay"])

                state = self.state[p]

                if len(state) == 0:

                    state["step"] = 0
                    # State initialization
                    state["exp_avg"] = grad
                    state["exp_avg_sq"] = grad**2
                    if group["do_bias_correction"]:
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["nested_exp_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)   


                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                beta1, beta2 = group["betas"]
                state["step"] += 1


                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                if group["do_bias_correction"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    exp_avg = exp_avg / bias_correction1
                    exp_avg_sq = exp_avg_sq / bias_correction2

                step = exp_avg / (exp_avg_sq.sqrt() + group["eps"])
                p.data.add_(step, alpha=-group["lr"])

        return loss