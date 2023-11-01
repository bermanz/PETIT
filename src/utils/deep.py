"""Deep-Learning Infra Module"""

import random
from enum import Enum, auto
import numpy as np

import torch


def set_deterministic(training_func):
    """A decorator for ensuring a deterministic training (use to wrap the training routing)"""

    def training_wrap(*args, **kwargs):
        torch.backends.cudnn.deterministic = True
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        return training_func(*args, **kwargs)

    return training_wrap


class NetPhase(Enum):
    """Enumeration for the network phase"""

    train = auto()
    val = auto()
    test = auto()
