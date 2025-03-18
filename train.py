import torch
import torch.nn as nn
import torch.nn.functional as F

from optim import initialize_optimizer, initialize_scheduler

from engine.torch_engine import TorchEngine


from absl import app, flags 
from collections import defaultdict  

import vision_utils
from torch_utils import pytorch_setup, destroy_ddp
from models.vision_models import construct_model

import yaml  
from data import get_dataloaders
# purely for debugging on interactive node
from dataclasses import dataclass
from typing import Optional, Literal

flags.DEFINE_string("config", "config/config.yaml", "Path to config file")
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS


def main(_):
    CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
    cfg = vision_utils.load_config(CFG_PATH, JOB_IDX)
    local_rank, world_size, device, master_process = pytorch_setup(cfg)

    train_loader, val_loader, test_loader = get_dataloaders(cfg, local_rank, world_size)

    model = construct_model(cfg)

    criterion = nn.CrossEntropyLoss()

    optimizer = initialize_optimizer(model.parameters(), cfg)
    scheduler = initialize_scheduler(optimizer, cfg)
    



    








if __name__ == "__main__":



    app.run(main)




   













if __name__ == "__main__":
    app.run(main)