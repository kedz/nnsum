from collections import OrderedDict


class hparam_registry(object):
    
    def __init__(self):
        self._hparams = OrderedDict()
        self._defaults = {}

    def __call__(self, default=None):
        def wrapper(func):
            self._hparams[func.__name__] = func
            if default is not None:
                self._defaults[func.__name__] = default
            return property(func)
        return wrapper

    def get_hparams(self):
        return self._hparams.keys()

    def has_default(self, hparam):
        return hparam in self._defaults

    def get_default(self, hparam):
        return self._defaults[hparam]
