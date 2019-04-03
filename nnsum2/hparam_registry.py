from collections import OrderedDict


class HParams(object):

    def __init__(self):
        self.hparams = OrderedDict()
        self.defaults = {}
        self.submodules = set()

    def register(self, default=None, required=True, type=None):
        def wrapper(func):
            fname = func.__name__
            _locals = {}
            exec("def {}(self): return self._{}".format(fname, fname), 
                 globals(), _locals)
            new_func = _locals[fname]
            self.hparams[fname] = new_func
            if not required or default is not None:
                self.defaults[fname] = default

            if type == "submodule":
                self.submodules.add(fname)
            return property(new_func)           
        return wrapper

    def __call__(self, *args, **kwargs):
        return self.register(*args, **kwargs)

    def __iter__(self):
        for hparam in self.hparams:
            yield hparam

    def has_default(self, hparam):
        return hparam in self.defaults

    def is_submodule(self, hparam):
        return hparam in self.submodules
