import torch
import torch.nn as nn
import torch.nn.functional as F

from optim import initialize_optimizer, initialize_scheduler

from engine.torch_engine import TorchEngine


from absl import app, flags 
from collections import defaultdict  

import vision_utils
from torch_utils import pytorch_setup, destroy_ddp
from models.vision_models.construct import construct_model

import yaml  
from data.vision_datasets.dataloaders import get_loaders
# purely for debugging on interactive node
from dataclasses import dataclass
from typing import Optional, Literal

flags.DEFINE_string("config", "config/config.yaml", "Path to config file")
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS


def main(_):
    CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
    #cfg = vision_utils.load_config(CFG_PATH, JOB_IDX)
    # In train.py
    cfg, _ = vision_utils.load_config(CFG_PATH, JOB_IDX)
    local_rank, world_size, device, master_process = pytorch_setup(cfg)

    train_loader, val_loader, test_loader = get_loaders(cfg)

    print("Train loader loaded")

    # construct a model 
    model = construct_model(cfg)
    model = model.to(device)
    print("Model constructed")

    criterion = nn.CrossEntropyLoss()

    optimizer = initialize_optimizer(model.parameters(), cfg)
    print("Optimizer initialized")

    print("Starting training loop test")
    # do a test batch
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(out.shape)
        print(y.shape)
        print(loss)
        break
    

    #compute total number of steps
    
    scheduler = initialize_scheduler(optimizer, cfg)

    print("making the engine")
    engine = TorchEngine(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,  # Pass the scheduler if you have one
        device=device
    )
    print("engine made")
    print("training an epoch")
    epoch_loss = engine.train_one_epoch(train_loader)
    print(epoch_loss)
    print("training done")
    print("validating")
    val_loss, acc = engine.validate(val_loader)
    print(val_loss)
    print(acc)

    print("validation done")



    print("Tests passed")

if __name__ == "__main__":

    app.run(main)
