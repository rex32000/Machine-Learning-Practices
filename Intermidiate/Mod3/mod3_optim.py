#optimizer of breakout
import math
import torch
import torch.optim as optim

class SharedAdam(optim.Adam):

    def __init__(self, params, lr=le-3, betas = (0.9, .999), eps = le-8, weight_decay=0):
       super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
       for group in self.param_groups:
           for p in group['params']:
               