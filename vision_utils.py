import torch 
import os  
import numpy as np
import matplotlib.pyplot as plt
import wandb
from itertools import product
from collections import namedtuple
import yaml

def init_wandb(cfg):
  """Initalizes a wandb run"""
  #os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
  os.environ["WANDB__SERVICE_WAIT"] = "600"
  os.environ["WANDB_SILENT"] = "true"
  wandb_run_name = f"bias_corr={cfg.do_bias_correction}, sched={cfg.scheduler}, lr={cfg.lr}, wd={cfg.weight_decay}, b1={cfg.beta1}, b2={cfg.beta2}"
  wandb.init(
    project=cfg.wandb_project, 
    name=wandb_run_name, 
    dir=cfg.wandb_dir,
    config=cfg._asdict()
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


def log(cfg, epoch, train_loss, val_loss, val_accuracy, lr):
  """Logs metrics to wandb"""
  wandb.log({
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_accuracy': val_accuracy,
    'lr': lr
  }, step=epoch)

  print(f"Epoch: {epoch} Step:  Train Loss: {train_loss} Val Loss: {val_loss} Val Acc: {val_accuracy}")