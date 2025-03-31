from optim import CustomAdamW
from torch.optim import Optimizer
import torch

def initialize_optimizer(param_groups, cfg):
    """
    Initialize an optimizer from the config file
    """

    if cfg.optimizer is None:

        return CustomAdamW(
            param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), 
            eps=cfg.eps, weight_decay=cfg.weight_decay, 
            do_bias_correction=cfg.do_bias_correction
            )
    
    elif cfg.optimizer == "muon":
        # definitely need some fine grained logic where 1d layers just have adamw or SGD
        # and other layers muon with flattening stuff for more than 2d

        from .muon import Muon  
        momentum = cfg.momentum if cfg.momentum is not None else cfg.beta1 

        return None # do later 
    





def initialize_scheduler(optimizer:Optimizer, optim_steps:int, cfg):
    """
    Initialize a scheduler from the config file
    """
    if cfg.scheduler is None:
        return None
    

    
    # i would like to infer the total number of steps based on batch size, epochs and dataset type

    
    if cfg.scheduler == "warmup_cosine":
        # check if lr_start, lr_max, lr_end are defined in cfg
        assert cfg.warmup_steps is not None, "warmup_steps must be defined if warmup_steps is defined"
        assert cfg.lr_start is not None, "lr_start must be defined if warmup_steps is defined"
        #assert cfg.lr_max is not None, "lr_max must be defined if warmup_steps is defined"
        assert cfg.lr_end is not None, "lr_end must be defined if warmup_steps is defined"
        
        from .lr_schedule import WarmupCosine
        warmup_steps = cfg.warmup_steps * optim_steps # proportion of total steps x total steps
        
        return WarmupCosine(
            optimizer=optimizer,
            lr_start=cfg.lr_start,
            lr_max=cfg.lr,
            lr_end=cfg.lr_end,
            warmup_steps=warmup_steps,
            T=optim_steps
        )
    
    elif cfg.scheduler == "warmup_step":
        assert cfg.warmup_steps is not None, "warmup_steps must be defined if warmup_steps is defined"
        assert cfg.lr_start is not None, "lr_start must be defined if warmup_steps is defined"
        assert cfg.step_size is not None, "step_size must be defined if warmup_steps is defined"
        assert cfg.gamma is not None, "gamma must be defined if warmup_steps is defined"
        
        from .lr_schedule import WarmupStep
        warmup_steps = cfg.warmup_steps * optim_steps
        return WarmupStep(
            optimizer=optimizer,
            lr_start=cfg.lr_start,
            lr_max=cfg.lr,
            lr_end=cfg.lr_end,
            warmup_steps=warmup_steps,
            step_size=cfg.step_size,
            gamma=cfg.gamma
        )
        
    elif cfg.scheduler == "wsd":
        # check if lr_start, lr_max, lr_end are defined in cfg
        assert cfg.warmup_steps is not None, "warmup_steps must be defined if warmup_steps is defined"
        assert cfg.lr_start is not None, "lr_start must be defined if warmup_steps is defined"
        assert cfg.lr_end is not None, "lr_end must be defined if warmup_steps is defined"
        assert cfg.cooldown_start_step is not None, "cooldown_start_step must be defined if wsd is defined"
        assert cfg.cooldown_steps is not None, "cooldown_steps must be defined if wsd is defined"
        
        from .lr_schedule import WSD
        warmup_steps = cfg.warmup_steps * optim_steps
        cooldown_start_step = cfg.cooldown_start_step * optim_steps
        cooldown_steps = cfg.cooldown_steps * optim_steps

        return WSD(
            optimizer=optimizer,
            lr_start=cfg.lr_start,
            lr_end=cfg.lr_end,
            warmup_steps=warmup_steps,
            cooldown_start_step=cooldown_start_step,
            cooldown_steps=cooldown_steps
        )

    elif cfg.scheduler == "step":
        assert cfg.step_size is not None, "step_size must be defined if step is defined"
        assert cfg.gamma is not None, "gamma must be defined if step is defined"
        
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    
    elif cfg.scheduler == "multistep":
        gamma = 0.5 if cfg.gamma is None else cfg.gamma

        # divide epochs into milestones 3 times evenly
        milestones = [int(cfg.epochs * i / 3) for i in range(1, 4)]
      
        from torch.optim.lr_scheduler import MultiStepLR
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    

    elif cfg.scheduler == "warmup_multistep":
        gamma = 0.5 if cfg.gamma is None else cfg.gamma

        # divide epochs into milestones 3 times evenly
        milestones = [int(cfg.epochs * i / 3) for i in range(1, 4)]

        assert cfg.warmup_steps is not None, "warmup_steps must be defined if warmup_steps is defined"

        assert cfg.lr_start is not None, "lr_start must be defined if warmup_steps is defined"

        from .lr_schedule import WarmupMultiStep  

        warmup_steps = cfg.warmup_steps * optim_steps
        return WarmupMultiStep(
            optimizer=optimizer,
            lr_start=cfg.lr_start,
            warmup_steps=warmup_steps,
            milestones=milestones,
            gamma=gamma
        )
    



    
    



    return None