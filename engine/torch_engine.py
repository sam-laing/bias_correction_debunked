import torch

from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

from models.vision_models.construct import construct_model
from optim import initialize_optimizer, initialize_scheduler

import time

class TorchEngine(torch.nn.Module):
    def __init__(self, cfg, model, optimizer, criterion, scheduler=None, device=torch.device('cuda')):
        super(TorchEngine, self).__init__()
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

    def forward(self, x):
        return self.model(x)

    def train_one_epoch(self, train_loader, timed=False):
        if timed:
            start_time = time.time()
        self.model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            running_loss += loss.item()
        if timed:
            elapsed_time = time.time() - start_time
            return running_loss / len(train_loader), elapsed_time
        
        return running_loss / len(train_loader)

    def validate(self, val_loader, timed=False):
        """   
        can also be used for test set

        Returns: 
        --------
        val_loss: float
            Average validation loss
        accuracy: float
            Validation accuracy
            
        """
        if timed:
            start_time = time.time()
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        if timed:
            elapsed_time = time.time() - start_time
            return running_loss / len(val_loader), correct / total, elapsed_time
        
        return running_loss / len(val_loader), correct / total














    

