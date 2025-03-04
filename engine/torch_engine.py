import torch

from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

class TorchEngine(torch.nn.Module):
    def __init__(self, cfg, model, optimizer, criterion, scheduler=None, device=torch.device('cuda')):
        super(TorchEngine, self).__init__()
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

    def forward(self, x):
        return self.model(x)

    def train_one_epoch(self, train_loader):
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
        
        return running_loss / len(train_loader)

    def validate(self, val_loader):
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
        
        return running_loss / len(val_loader), correct / total

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            print(f"Epoch: {epoch} Train Loss: {train_loss} Val Loss: {val_loss} Val Acc: {val_acc}")












    

