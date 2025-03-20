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
  wandb_run_name = f"{cfg.model}, bias_corr={cfg.do_bias_correction}, sched={cfg.scheduler}, lr={cfg.lr}, wd={cfg.weight_decay}, b1={cfg.beta1}, b2={cfg.beta2}"
  wandb.init(
    project=cfg.wandb_project,
    group=cfg.dataset, 
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

  #print(f"Epoch: {epoch} Step:  Train Loss: {train_loss} Val Loss: {val_loss} Val Acc: {val_accuracy}")

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