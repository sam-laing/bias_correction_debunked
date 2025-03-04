import torch
import torch.nn as nn
import torch.nn.functional as F

from optim import CustomAdam
from torch.optim import NAdam


from absl import app, flags 
from collections import defaultdict  

import vision_utils

flags.DEFINE_string("config", "config/config.yaml", "Path to config file")
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS

def main(_):
    CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx










