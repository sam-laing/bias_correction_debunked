from optim import CustomAdamW
import torch

def initialize_optimizer(param_groups, cfg):
    """
    Initialize an optimizer from the config file
    """
    return CustomAdamW(
        param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), 
        eps=cfg.eps, weight_decay=cfg.weight_decay, 
        bias_correction=cfg.do_bias_correction)

def initialize_scheduler(optimizer, cfg):
    """
    Initialize a scheduler from the config file
    """
    # infer number of training steps from cfg.epochs and cfg.batch_size 
    num_training_steps = cfg.epochs * (cfg.batch_size + 1)

    #cfg.warmup is a proportion of the total number of training steps


    if cfg.scheduler == "linear":
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(cfg.warmup * num_training_steps), num_training_steps=num_training_steps
        )
    elif cfg.scheduler == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(cfg.warmup * num_training_steps), num_training_steps=num_training_steps
        )
    elif cfg.scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
    else:
        raise NotImplementedError(f"Scheduler {cfg.scheduler} not implemented or misspelled")

