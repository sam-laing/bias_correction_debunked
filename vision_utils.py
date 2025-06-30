import torch 
import os  
import numpy as np
import matplotlib.pyplot as plt
import wandb
from itertools import product
from collections import namedtuple
import yaml

import numpy as np

def mixup_datapoints(x, y, device, alpha=0.2):
    '''
    given a minibatch of data x and labels y, 
    generate an idx of samples to mixup with

    p is the amount of weight to put on the first sample
    it will be generated from a beta distribution with parameter alpha

    '''
    idx = torch.randperm(x.shape[0]).to(device)
    lam = np.random.beta(alpha, alpha)

    x = lam * x + (1 - lam) * x[idx]
    y = lam * y + (1 - lam) * y[idx]
    return x, y

def cutmix_datapoints(x, y, device, alpha=0.2):
    """
    given minibatch x,y
    do a randomized cutmix operation on x
    and take convex combination on y with area of the cut region
    """
    rand_idx = torch.randperm(x.shape[0]).to(device)
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    # fill in the cut region with the data from the random index
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_idx, :, bbx1:bbx2, bby1:bby2]
    lam1 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y = lam1 * y + (1 - lam1) * y[rand_idx]

    return x, y


def rand_bbox(size, lam):
    W, H = size[-2], size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def init_wandb(cfg):
  """Initalizes a wandb run"""
  #os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
  os.environ["WANDB__SERVICE_WAIT"] = "600"
  os.environ["WANDB_SILENT"] = "true"
  wandb_run_name = f"{cfg.model}, bias_corr={cfg.do_bias_correction}, zero_init={cfg.zero_init}, sched={cfg.scheduler}, lr={cfg.lr}, wd={cfg.weight_decay}, bs={cfg.batch_size}, b1={cfg.beta1}, b2={cfg.beta2}, seed={cfg.seed}"
  tags = [  
      f"model: {cfg.model}", 
      f"dataset: {cfg.dataset}",
      f"bias_corr: {cfg.do_bias_correction}",
      f"sched: {cfg.scheduler}",
      f"val_size: {cfg.val_size}",
      f"lr: {cfg.lr}",
      f"wd: {cfg.weight_decay}",
      f"bs: {cfg.batch_size}",
      f"b1: {cfg.beta1}",
      f"b2: {cfg.beta2}",
      f"seed: {cfg.seed}",
      f"optimizer: {cfg.optimizer}",
      f"one_seeded: {cfg.one_seeded}",
      f"zero_init: {cfg.zero_init}",
      f"cutmix: {cfg.cutmix}",
      f"cutmix_prob: {cfg.cutmix_probability}",
    ]
  if hasattr(cfg, "dropout"):
    tags.append(f"dropout: {cfg.dropout}")

  if cfg.dataset == "tiny_imagenet" and cfg.model == "ViT":
    tags.extend([
      f"img_size: {cfg.image_size}",
      f"patch_size: {cfg.patch_size}",
      f"num_heads: {cfg.num_heads}",
      f"num_layers: {cfg.num_layers}",
      f"hidden_dim: {cfg.hidden_dim}",
      f"mlp_dim: {cfg.mlp_dim}",
    ])

  wandb.init(
    project=cfg.wandb_project,
    group=cfg.dataset, 
    name=wandb_run_name, 
    dir=cfg.wandb_dir,
    config=cfg._asdict(),
    tags = tags
  )

def load_config(path, job_idx=None):
  """
  Parse a yaml file and return the correspondent config as a namedtuple.
  If the config files has multiple entries, returns the one corresponding to job_idx.
  """
  
  with open(path, 'r') as file:
    config_dict = yaml.safe_load(file)
  Config = namedtuple('Config', config_dict.keys())

  if job_idx is None:
    cfg = config_dict
    sweep_size = 1

  else:
    keys = list(config_dict.keys())
    values = [val if isinstance(val, list) else [val] for val in config_dict.values()]
    combinations = list(product(*values))

    sweep_size = len(combinations)
    if job_idx >= sweep_size:
      raise ValueError("job_idx exceeds the total number of hyperparam combinations.")

    combination = combinations[job_idx]
    cfg = {keys[i]: combination[i] for i in range(len(keys))}
  
  return Config(**cfg), sweep_size

"""
def log(cfg, epoch, train_loss, val_loss, val_accuracy, lr):
  # Logs metrics to wandb
  
  wandb.log({
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_accuracy': val_accuracy,
    'lr': lr
  }, step=epoch)


  #print(f"Epoch: {epoch} Step:  Train Loss: {train_loss} Val Loss: {val_loss} Val Acc: {val_accuracy}")
"""


def log(cfg, epoch, train_loss, val_loss, val_accuracy, lr):
  """Logs metrics to wandb"""

  metrics = {
  "train_loss": train_loss, 
  "lr": lr
  }
  if (val_loss is not None) and (val_accuracy is not None):
    metrics["val_loss"] = val_loss
    metrics["val_accuracy"] = val_accuracy

  wandb.log(metrics, step=epoch)



def log_test_summary(cfg, test_loss, test_accuracy):
  """Logs test metrics to wandb"""
  wandb.log({
    'test_loss': test_loss,
    'test_accuracy': test_accuracy
  }, step=cfg.epochs)
  
  wandb.run.summary["final_test_loss"] = test_loss
  wandb.run.summary["final_test_accuracy"] = test_accuracy
  wandb.run.summary["optimizer"] = cfg.optimizer if hasattr(cfg, 'optimizer') else "adam"
  wandb.run.summary["bias_correction"] = cfg.do_bias_correction


def print_columns(columns_list, is_head=False, is_final_entry=False):
    """Print formatted columns with separators."""
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-"*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-"*len(print_string))


# things to log
LOGGING_COLUMNS_LIST = ["epoch", "train_loss", "val_loss", "val_acc", "lr"]


def print_training_details(variables, is_head=False, is_final_entry=False):
    """Format and print training details in a clean tabular format."""
    formatted = []
    for col in LOGGING_COLUMNS_LIST:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_head=is_head, is_final_entry=is_final_entry)



import shutil
def maybe_make_dir(cfg, job_idx=None):
  """Creates an experiment directory if checkpointing is enabled"""
  if not cfg.save_intermediate_checkpoints and not cfg.save_last_checkpoint:
    return
  if cfg.resume and cfg.resume_exp_name is None:  # if resuming from the same exp
    return
  
  exp_name = f"{cfg.model}, bias_corr={cfg.do_bias_correction}, sched={cfg.scheduler}, lr={cfg.lr}, wd={cfg.weight_decay}, b1={cfg.beta1}, b2={cfg.beta2}"
  exp_dir = os.path.join(cfg.out_dir, exp_name)
  if job_idx is not None:  # subfolder for each job in the sweep
    exp_dir = os.path.join(exp_dir, f"job_idx_{job_idx}")

  if os.path.exists(exp_dir):
    if not cfg.over_write:
      raise ValueError(f"Found existing exp_dir at {exp_dir}.")
    print(f"Removing experiment dir: {exp_dir}")
    shutil.rmtree(exp_dir)

  print(f"Creating experiment directory: {exp_dir}")
  os.makedirs(exp_dir, exist_ok=True)
  with open(os.path.join(exp_dir, 'config.yaml'), 'w') as file:
    yaml.dump(cfg._asdict(), file, default_flow_style=False)


def print_master(msg):
  """Prints only in master process if using multiple GPUs."""
  rank = os.environ.get('RANK', -1)
  ddp = int(rank) != -1
  master_process = (not ddp) or (int(rank) == 0)
  
  if master_process:
    print(msg)