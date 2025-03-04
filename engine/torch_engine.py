import torch

from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext



def train_one_epoch(cfg, ):
    pass