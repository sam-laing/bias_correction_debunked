import torch
import torch.nn as nn
import torch.nn.functional as F

from optim import initialize_optimizer, initialize_scheduler

from engine.torch_engine import TorchEngine


from collections import defaultdict  

import vision_utils
from torch_utils import pytorch_setup, destroy_ddp
from models.vision_models.construct import construct_model

import yaml  
from data.vision_datasets.dataloaders import get_loaders
# purely for debugging on interactive node
from dataclasses import dataclass
from typing import Optional, Literal, Union, List


from vision_utils import LOGGING_COLUMNS_LIST, print_training_details

from absl import app, flags 
flags.DEFINE_string("config", "config/config.yaml", "Path to config file")
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS


import time



def main(_):
    CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
    cfg, _ = vision_utils.load_config(CFG_PATH, JOB_IDX)
 
    if (cfg.deterministic) and (cfg.one_seeded):
        #need a workaround for this
        #cfg.sampler_seed = cfg.seed

        print(f"sampler seed: {cfg.sampler_seed}, seed: {cfg.seed} - should be same")

    local_rank, world_size, device, master_process = pytorch_setup(cfg)

    if cfg.use_wandb and master_process:
        vision_utils.init_wandb(cfg)

    print("Bias Correction set to", cfg.do_bias_correction)
    # construct a model 
    model = construct_model(cfg)
    model = model.to(device)

    # diplay some information about the model and the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")



    print("Model constructed")
    train_loader, val_loader, test_loader = get_loaders(cfg)
    print(len(train_loader))
    print(f"Loaded {cfg.dataset} dataset")



    criterion = nn.CrossEntropyLoss()
    optimizer = initialize_optimizer(model.parameters(), cfg)
    print("Optimizer initialized")

    #compute total number of steps
    optim_steps = (len(train_loader) + 1) * cfg.epochs
    scheduler = initialize_scheduler(optimizer, optim_steps, cfg)

    print("making the engine")
    engine = TorchEngine(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device
    )
    print("engine made")


    variables = {col: col for col in LOGGING_COLUMNS_LIST}
    print_training_details(variables, is_head=True)

    print("====== Starting the training loop ======")
    actual_train_and_val_time = 0
    best_val_accuracy = 0.0
    for epoch in range(cfg.epochs):
        
        train_loss = engine.train_one_epoch(train_loader)
        val_loss, val_accuracy = None, None
        if val_loader is not None:
            val_loss, val_accuracy = engine.validate(val_loader)
        lr = optimizer.param_groups[0]['lr']
        
        # Log to wandb if enabled
        if cfg.use_wandb and master_process:
            vision_utils.log(cfg, epoch, train_loss, val_loss, val_accuracy, lr)
        
        # Track best validation accuracy
        if val_accuracy and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        
        # Print training details in a nice format
        if master_process:
            variables = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_accuracy,
                "lr": lr
            }
            print_training_details(variables)

    print("\n====== Training finished ======")

    # Test with the final model
    test_loss, test_accuracy = engine.validate(test_loader)
    
    if master_process:
        # Final summary row
        variables = {
            "epoch": "final",
            "train_loss": None,
            "val_loss": None,
            "val_acc": best_val_accuracy,
            "lr": None
        }
        print_training_details(variables, is_final_entry=True)
        
        # Print the test results
        print(f"\nFinal results:")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
    
    if cfg.use_wandb:
        vision_utils.log_test_summary(cfg, test_loss, test_accuracy)



if __name__ == "__main__":
    app.run(main)
