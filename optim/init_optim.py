from optim import CustomAdamW
import torch

def initialize_optimizer(param_groups, cfg):
    """
    Initialize an optimizer from the config file
    """
    if cfg.do_bias_correction:
        return torch.optim.AdamW(
            param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), 
            eps=cfg.eps, weight_decay=cfg.weight_decay
        )

    return CustomAdamW(
        param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), 
        eps=cfg.eps, weight_decay=cfg.weight_decay, 
        do_bias_correction=cfg.do_bias_correction)


DATASET_SIZES =  {
    "cifar10": 50000,
    "imagenet": 1281167,
    "cifar100": 50000
}

def initialize_scheduler(optimizer, cfg):
    """
    Initialize a scheduler from the config file
    """
    if cfg.scheduler is None:
        return None
    

    
    # i would like to infer the total number of steps based on batch size, epochs and dataset type
    if cfg.dataset == "cifar10":
        total_steps = (40000 // cfg.batch_size + 1) * cfg.epochs
        

    
    if cfg.warmup_steps is not None:


        warmup_steps = cfg.warmup_steps


    return None