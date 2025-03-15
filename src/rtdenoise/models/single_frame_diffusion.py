import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

"""
This class will have two implementations:
- A non-latent diffusion method which is color stable and hopefully is easier to train.
- A latent diffusion method which is not color stable and will use a CFM-like architecture for gather operations
"""


