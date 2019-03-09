import torch.nn as nn
from ..parameterized import Parameterized
from ..hparam_registry import HParams


def register_module(name):
    return Parameterized.register_object(name)

def hparam_registry():
    return HParams()

class Module(nn.Module, Parameterized):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        Parameterized.__init__(self, *args, **kwargs)
        self.init_network()                 

    def init_network(self):
        pass
