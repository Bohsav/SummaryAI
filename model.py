import torch
from torch import nn


# TODO: According to the plan this is going to be the base model
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("")
