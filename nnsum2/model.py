import torch.nn as nn
from .parameterized import Parameterized


class Module(nn.Module, Parameterized):
    def __init__(self):
        super(Module, self).__init__()
        self.init_network()                 


#from .hparam import hparam_registry


class Model(nn.Module):

    @staticmethod
    def hp(default=None):
        def mark_hyperparameter(func):
            func.is_hyperparameter = True
            if default is not None:
                func.default = default
            return func
        return mark_hyperparameter



#    @staticmethod
#    def hparam_registry(cls):
 #       setattr(cls, "_hparam_registry", hparam_registry())

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

        for name, method in self.__class__.__dict__.items():
            if hasattr(method, "is_hyperparameter"):
                print(name, method)
                if name in kwargs:
                    value = kwargs[name]
                elif hasattr(method, "default"):
                    value = method.default
                else:
                    raise Exception("No value provided for {}".format(name))
                setattr(self, "_{}".format(name), value)
                setattr(self, name, property(method)) 

