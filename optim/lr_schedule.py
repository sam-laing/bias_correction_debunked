"""Custom implementation of LR schedules."""

import math

class WarmupCosine(object):
  """Linear warmup followed by cosine decay"""
  def __init__(self, optimizer, lr_start, lr_max, lr_end, warmup_steps, T):
    self.optimizer = optimizer
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.lr_end = lr_end
    self.warmup_steps = warmup_steps
    self.T = T
    self.iter = 0
    for group in self.optimizer.param_groups:
      group["lr"] = lr_start

  def schedule(self, t):
    """returns lr(t), where t is the current step"""
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max-self.lr_start)/self.warmup_steps * t
    elif t <= self.T:
      progress = (t-self.warmup_steps) / (self.T-self.warmup_steps)
      return self.lr_end + 0.5 * (self.lr_max-self.lr_end) * (1 + math.cos(math.pi * progress))
    return self.lr_end

  def step(self):
    self.iter += 1
    lr = self.schedule(self.iter)
    for group in self.optimizer.param_groups:
      group["lr"] = lr

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)


class WSD(object):
  """Trapezoidal schedule / WSD: (linear) Warmup, Stable, (linear) Decay"""
  def __init__(self, optimizer, lr_start, lr_max, lr_end, warmup_steps, cooldown_start_step, cooldown_steps):
    self.optimizer = optimizer
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.lr_end = lr_end
    self.warmup_steps = warmup_steps
    self.cooldown_start_step = cooldown_start_step
    self.cooldown_steps = cooldown_steps
    self.iter = 0
    
    for group in self.optimizer.param_groups:
      group["lr"] = lr_start

  def schedule(self, t):
    """returns lr(t), where t is the current step"""
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max-self.lr_start)/self.warmup_steps * t
    elif t <= self.cooldown_start_step:
      return self.lr_max
    return self.lr_max + (self.lr_end-self.lr_max)/self.cooldown_steps * (t-self.cooldown_start_step)

  def step(self):
    """computes new lr and sets it in self.optimizer"""
    self.iter += 1
    lr = self.schedule(self.iter)
    for group in self.optimizer.param_groups:
      group["lr"] = lr

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)


class WarmupConstant(object):
  """Linear Warmup + Constant LR"""
  def __init__(self, optimizer, lr_start, lr_max, warmup_steps):
    self.optimizer = optimizer
    self.lr_start = lr_start
    self.lr_max = lr_max
    self.warmup_steps = warmup_steps
    self.iter = 0
    for group in self.optimizer.param_groups:
      group["lr"] = lr_start

  def schedule(self, t):
    """returns lr(t), where t is the current step"""
    if t <= self.warmup_steps:
      return self.lr_start + (self.lr_max-self.lr_start)/self.warmup_steps * t
    return self.lr_max

  def step(self):
    """computes new lr and sets it in self.optimizer"""
    self.iter += 1
    lr = self.schedule(self.iter)
    for group in self.optimizer.param_groups:
      group["lr"] = lr

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)


class WarmupStep(object):
    """   
    Linear warmup for a certain number of steps, followed by step decay at fixed intervals.
    The learning rate is reduced by a factor of gamma every step_size steps after warmup.
    """

    def __init__(
        self, optimizer, lr_start, lr_max, lr_end, warmup_steps, step_size, gamma
    ):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_max = lr_max  # Peak LR after warmup
        self.lr_end = lr_end  # Minimum LR
        self.warmup_steps = warmup_steps
        self.step_size = step_size  # Steps between decay
        self.gamma = gamma  # Decay factor
        self.iter = 0
        
        # Initialize optimizer with starting learning rate
        for group in self.optimizer.param_groups:
            group["lr"] = lr_start

    def schedule(self, t):
        """returns lr(t), where t is the current step"""
        if t <= self.warmup_steps:
            # Linear warmup phase
            return self.lr_start + (self.lr_max - self.lr_start) / self.warmup_steps * t
        else:
            # Step decay phase - reduce by gamma factor every step_size steps
            post_warmup_steps = t - self.warmup_steps
            # Calculate which step "plateau" we're on
            step_index = post_warmup_steps // self.step_size
            # Apply gamma reduction for each step
            lr = self.lr_max * (self.gamma ** step_index)
            # Ensure we don't go below minimum LR
            return max(lr, self.lr_end)

    def step(self):
        """computes new lr and sets it in self.optimizer"""
        self.iter += 1
        lr = self.schedule(self.iter)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def state_dict(self):
        """Returns scheduler state for checkpointing"""
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        """Loads scheduler state from checkpoint"""
        self.__dict__.update(state_dict)


class WarmupMultiStep(object):
    """
    Linear warmup for a certain number of steps, followed by multi-step decay.
    The learning rate is reduced by a factor of gamma at specified milestones.
    """

    def __init__(
        self, optimizer, lr_start, lr_max, lr_end, warmup_steps, milestones, gamma
    ):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_max = lr_max  # Peak LR after warmup
        self.lr_end = lr_end  # Minimum LR
        self.warmup_steps = warmup_steps
        self.milestones = milestones  # List of steps to reduce LR
        self.gamma = gamma  # Decay factor
        self.iter = 0
        
        # Initialize optimizer with starting learning rate
        for group in self.optimizer.param_groups:
            group["lr"] = lr_start
        
    def schedule(self, t):
        """returns lr(t), where t is the current step"""
        if t <= self.warmup_steps:
            # Linear warmup phase
            return self.lr_start + (self.lr_max - self.lr_start) / self.warmup_steps * t
        else:
            # Multi-step decay phase
            lr = self.lr_max
            for milestone in self.milestones:
                if t >= milestone:
                    lr *= self.gamma
            # Ensure we don't go below minimum LR
            return max(lr, self.lr_end)
    def step(self):
        """computes new lr and sets it in self.optimizer"""
        self.iter += 1
        lr = self.schedule(self.iter)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
    
    def state_dict(self):
        """Returns scheduler state for checkpointing"""
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}
    
    def load_state_dict(self, state_dict):
        """Loads scheduler state from checkpoint"""
        self.__dict__.update(state_dict)
    

      