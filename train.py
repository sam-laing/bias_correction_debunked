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
    # In train.py
    cfg, _ = vision_utils.load_config(CFG_PATH, JOB_IDX)

    if (cfg.deterministic) and (cfg.one_seeded):
        cfg.sampler_seed = cfg.seed

        print(f"sampler seed: {cfg.sampler_seed}, seed: {cfg.seed} - should be same")


    local_rank, world_size, device, master_process = pytorch_setup(cfg)

    #if master_process:
    #   vision_utils.maybe_make_dir(cfg, JOB_IDX)

    if cfg.use_wandb and master_process:
        vision_utils.init_wandb(cfg)

    print("Bias Correction set to", cfg.do_bias_correction)

    train_loader, val_loader, test_loader = get_loaders(cfg)

    print("Train loader loaded")

    # construct a model 
    model = construct_model(cfg)
    model = model.to(device)
    print("Model constructed")

    criterion = nn.CrossEntropyLoss()

    optimizer = initialize_optimizer(model.parameters(), cfg)
    print("Optimizer initialized")

    """
    #count the number of each class label in the validation set
    class_counts = defaultdict(int)
    for i, (x, y) in enumerate(val_loader):
        for label in y:
             class_counts[label.item()] += 1
    print("Class counts in validation set")
    print(class_counts)
    """

    

    #compute total number of steps
    optim_steps = (len(train_loader) + 1) * cfg.epochs
    scheduler = initialize_scheduler(optimizer, optim_steps, cfg)

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

    print("====== Starting the training loop ======")

    for epoch in range(cfg.epochs):
        train_loss = engine.train_one_epoch(train_loader)
        val_loss, val_accuracy = None, None
        if len(val_loader) != 0:
            val_loss, val_accuracy = engine.validate(val_loader)
        lr = optimizer.param_groups[0]['lr']
        if cfg.use_wandb and master_process:
            vision_utils.log(cfg, epoch, train_loss, val_loss, val_accuracy, lr)
        if master_process:
            print(f"Epoch: {epoch} Step:  Train Loss: {train_loss} Val Loss: {val_loss} Val Acc: {val_accuracy}")

    print("====== Training finished ======")

    #test
    test_loss, test_accuracy = engine.validate(test_loader)
    if master_process:
        print(f"Test Loss: {test_loss} Test Accuracy: {test_accuracy}")

    test_loss, test_accuracy = engine.validate(test_loader)
    if master_process:
        print(f"Test Loss: {test_loss} Test Accuracy: {test_accuracy}")
    
    if cfg.use_wandb:
        vision_utils.log_test_summary(cfg, test_loss, test_accuracy)



if __name__ == "__main__":
    app.run(main)
