import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union


# Define types for better type hints
Params = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
LossClosure = Optional[Callable[[], float]]


def compute_statistics(t1: torch.Tensor, t2: torch.Tensor) -> Dict[str, float]:
    """   
    Compute the angle between the two tensors, their inner products and their norms
    """
    assert t1.shape == t2.shape, "t1 and t2 must have the same shape"
    
    t1_flat, t2_flat = t1.view(-1), t2.view(-1)
    norm_t1, norm_t2 = torch.norm(t1_flat).item(), torch.norm(t2_flat).item()
    # basically if effectively zero, no point computing
    if norm_t1 < 1e-8 or norm_t2 < 1e-8:
        return {
            "inner_product": 0.0,
            "norm_t1": norm_t1,
            "norm_t2": norm_t2,
            "cosine_similarity": 0.0,
            "angle": 0.0
        }

    inner_product = torch.dot(t1_flat, t2_flat).item()
    cosine_similarity = inner_product / (norm_t1 * norm_t2)
    cosine_similarity = max(min(cosine_similarity, 1.0), -1.0)
    angle = torch.acos(torch.tensor(cosine_similarity)).item() * 180 / 3.14159265359
    return {
        "inner_product": inner_product,
        "norm_t1": norm_t1,
        "norm_t2": norm_t2,
        "cosine_similarity": cosine_similarity,
        "angle": angle
    }

class CustomSGD(Optimizer):
    """
    A modified version of SGD optimizer with momentum and an option
    to disable bias correction and initialize momentum directly with gradients.
    Tracks statistics between momentum and gradient.
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0.0,
        do_bias_correction: bool = True,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            do_bias_correction=do_bias_correction,
        )
        super().__init__(params, defaults)
        
        # Initialize storage for statistics
        self.layerwise_statistics = {}
        self.global_statistics = {
            'angle': [], 
            'cosine_similarity': [],
            'inner_product': [],
            'grad_norm': [],
            'momentum_norm': []
        }
        self.param_names = {}
        
        # Auto-name parameters
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                self.param_names[p] = f"layer_{i}_{j}"

    def step(self, closure: LossClosure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Reset statistics for this step
        self.layerwise_statistics = {}
        
        # For global statistics
        all_grads = []
        all_momentums = []
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")

                # Apply weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                    if not group["do_bias_correction"]:
                        state["momentum_buffer"].copy_(grad)

                momentum_buffer = state["momentum_buffer"]
                
                # Only compute statistics after the first step
                if state["step"] > 0:
                    # Compute layerwise statistics
                    self.layerwise_statistics[self.param_names[p]] = compute_statistics(momentum_buffer, grad)
                    
                    # Store tensors for global statistics
                    all_grads.append(grad.flatten())
                    all_momentums.append(momentum_buffer.flatten())

                state["step"] += 1

                # Update momentum buffer
                momentum_buffer.mul_(group["momentum"]).add_(grad, alpha=1 - group["dampening"])

                # Apply bias correction if enabled
                if group["do_bias_correction"]:
                    momentum_corrected = momentum_buffer.clone().div_(1 - group["momentum"] ** state["step"])
                else:
                    momentum_corrected = momentum_buffer

                # Parameter update
                p.data.add_(momentum_corrected, alpha=-group["lr"])

        # Compute global statistics if we have data
        if all_grads and all_momentums:
            all_grad_tensor = torch.cat(all_grads)
            all_momentum_tensor = torch.cat(all_momentums)
            global_stats = compute_statistics(all_momentum_tensor, all_grad_tensor)
            
            # Store global statistics
            for key, value in global_stats.items():
                if key == 'norm_t1':
                    self.global_statistics['grad_norm'].append(value)
                elif key == 'norm_t2':
                    self.global_statistics['momentum_norm'].append(value)
                else:
                    self.global_statistics[key].append(value)
                
        return loss

    def get_layerwise_statistics(self):
        return self.layerwise_statistics

    def get_global_statistics(self):
        return self.global_statistics
