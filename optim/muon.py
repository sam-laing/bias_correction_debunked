import torch
import math
from torch.optim import Optimizer

# the muon optimizer with option to just orthogonalize rather than approximate
@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    r"""
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """

    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

@torch.compile
def get_svd(G, eps=1e-7):
    """
    Compute the SVD of G using torch.linalg.svd. This is a wrapper around the function to ensure
    that we can use it with torch.compile.
    """
    X = G.float()
    X /= (X.norm() + eps) # ensure top singular value <= 1, eps to prevent NaNs

    if len(X.shape) == 1:
        raise ValueError("G is a vector, shouldn't compute SVD")
    elif len(X.shape) == 2:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        S = S.unsqueeze(0)
        return U, S, Vh
    else:
        raise ValueError("G is not a matrix, cannot compute SVD ... should be done in optimizer")


def orthogonalise(G):
    if G.size(0)>G.size(1):
        G = G.T

    U, S, Vh = get_svd(G)

    if G.size(0) > G.size(1):
        return (U @ Vh).T

    return U @ Vh 


class Muon(Optimizer):
    def __init__(
            self, params, lr=1e-3, momentum=0.95, nesterov=False, ns_steps=3, eps=1e-7,
            orthogonalize=False, weight_decay=0.0, adjust_lr=True, precon_nuclear=False,
            adam_betas=(0.95, 0.95), adam_eps=1e-8
            ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, orthogonalize=orthogonalize, eps=eps,
            weight_decay=weight_decay, adjust_lr=adjust_lr, 
            precon_nuclear=precon_nuclear,
            adam_betas=adam_betas, adam_eps=adam_eps
        )
        self._optimizer_types = {}
        
        super().__init__(params, defaults)
        
        self._identify_optimizer_types()

    def _identify_optimizer_types(self):
        """Identify which parameters should use Adam vs Muon"""
        all_params = []
        for group in self.param_groups:
            all_params.extend(group['params'])
        
        # Mark the last parameter as Adam 
        last_param_id = id(all_params[-1]) if all_params else None
        
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    # Use Muon for 2D parameters that aren't the last one
                    if p.ndim == 2 and id(p) != last_param_id:
                        self._optimizer_types[id(p)] = 'muon'
                    else:
                        self._optimizer_types[id(p)] = 'adam'

    def _muon_step(self, p, g, group, state):
        """Perform Muon optimization step"""
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        orthogonalize = group["orthogonalize"]
        weight_decay = group["weight_decay"]
        adjust_lr = group["adjust_lr"]
        precon_nuclear = group["precon_nuclear"]
        eps = group["eps"]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)
            
        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(g)
        g = g.add(buf, alpha=momentum) if nesterov else buf

        if orthogonalize:
            update = orthogonalise(g.reshape(len(g), -1)).view(g.shape)
        else:
            update = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=ns_steps).view(g.shape)

        effective_lr = lr
        if adjust_lr:
            effective_lr = 0.2 * math.sqrt(max(g.size(-2), g.size(-1))) * lr

        if precon_nuclear:
            U, S, Vh = get_svd(g.reshape(len(g), -1), eps=eps)
            nuclear_norm = S.abs().sum()
            if nuclear_norm > 0:
                effective_lr *= nuclear_norm

        if weight_decay > 0:
            p.data.add_(p.data, alpha=-weight_decay * lr)

        p.data.add_(update, alpha=-effective_lr)

    def _adam_step(self, p, g, group, state):
        """Perform Adam optimization step"""
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["adam_betas"]
        eps = group["adam_eps"]
        
        if 'step' not in state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        
        state['step'] += 1
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        
        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        
        p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        if weight_decay > 0:
            p.data.add_(p.data, alpha=-weight_decay * lr)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                    
                state = self.state[p]
                optimizer_type = self._optimizer_types.get(id(p), 'muon')
                
                if optimizer_type == "muon":
                    self._muon_step(p, g, group, state)
                else: 
                    self._adam_step(p, g, group, state)
                    
        return None

    def add_param_group(self, param_group):
        """Add parameter group while maintaining optimizer type tracking"""
        super().add_param_group(param_group)
        
        # Update optimizer types for new parameters
        all_params = []
        for group in self.param_groups:
            all_params.extend(group['params'])
        
        last_param_id = id(all_params[-1]) if all_params else None
        
        for p in param_group['params']:
            if p.requires_grad:
                if p.ndim == 2 and id(p) != last_param_id:
                    self._optimizer_types[id(p)] = 'muon'
                else:
                    self._optimizer_types[id(p)] = 'adam'