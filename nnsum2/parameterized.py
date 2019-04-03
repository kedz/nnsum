from .hparam_registry import HParams


class Parameterized(object):

    OBJECT_REGISTRY = {}

    @classmethod
    def register_object(cls, name):
        def wrapper(cls):
            cls.OBJECT_REGISTRY[name] = cls
            return cls
        return wrapper

    def __init__(self, *args, **kwargs):
        
        hp_reg = self.get_hparam_registry()

        for hparam in hp_reg:
            if hparam in kwargs:
                value = kwargs[hparam]
            elif hp_reg.has_default(hparam):
                value = hp_reg.defaults[hparam]
            else:
                raise Exception("No value provided for {} for class {}".format(
                    hparam, self.__class__))

            if hp_reg.is_submodule(hparam) and isinstance(value, dict):
                value = build_module(value)

            setattr(self, "_{}".format(hparam), value)

        self.init_object()

    def init_object(self):
        pass

    @staticmethod
    def create_registry():
        return HParams()

    def get_hparam_registry(self):
        regs = [obj for name, obj in self.__class__.__dict__.items()
                if isinstance(obj, HParams)]
        if len(regs) == 0:
            raise Exception("No HParams registry associated with this class.")
        elif len(regs) > 1:
            raise Exception(
                "Multiple HParams registries associated with this class.")
        return regs[0]

def build_object(params):
    if isinstance(params, (list, tuple)):
        return [build_object(p) for p in params]
    else:
        object_name = params.get("__modulename__", None)
        if object_name is None:
            raise Exception(
                "Hyperparameter dict does not contain __modulename__")
        elif object_name not in Parameterized.OBJECT_REGISTRY:
            raise Exception(
                "{} not registered!".format(object_name))
        return Parameterized.OBJECT_REGISTRY[object_name](**params)
