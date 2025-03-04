import torch
import torch.nn as nn
import torch.nn.functional as F

from optim import CustomAdam
from torch.optim import NAdam


from absl import app, flags 
from collections import defaultdict  

import vision_utils

import yaml  

# purely for debugging on interactive node
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class ConfigData:
    # General settings
    num_workers: int = 4
    
    # Model Configuration
    model: Literal["resnet18", "resnet50", "wide_resnet50_2", "resnet101", "ViT", "basic_cnn"] = "resnet18"
    
    # Training Parameters
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    do_bias_correction: bool = False
    
    # Scheduler Configuration
    scheduler: Literal["linear", "cosine", "constant"] = "cosine"
    warmup: float = 0.1  # Proportion of training steps for warmup
    
    # Logging Configuration
    use_wandb: bool = False
    wandb_project: str = "bias_correction"
    wandb_dir: str = "./wandb_logs"
    wandb_api_key: str = ""  # Leave empty to use environment variable or wandb login

cfg = ConfigData()


flags.DEFINE_string("config", "config/config.yaml", "Path to config file")
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS

def main(_):
    #CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
    #cfg = vision_utils.load_config(CFG_PATH)
    #cfg = vision_utils.load_config(FLAGS.config)
    #cfg = vision_utils.load_config(CFG_PATH)
    pass 

if __name__ == "__main__":
    app.run(main)




   













if __name__ == "__main__":
    app.run(main)