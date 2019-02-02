import ipywidgets as widgets


class SGDWidget(object):
    def __init__(self):

        self._w_lr = widgets.FloatText(value=.1, description="LR:")
        self._w_weight_decay = widgets.FloatText(value=0., description="L2")
        self._w_momentum = widgets.FloatText(value=0., description="Momentum")
        self._w_dampening = widgets.FloatText(value=0., 
                                              description="Dampening")
        self._w_nesterov = widgets.Dropdown(options=["True", "False"],
                                            value="False",
                                            description="Nesterov Momentum")

        self._w_main = widgets.VBox([
            widgets.HBox([self._w_lr, self._w_weight_decay]),
            widgets.HBox([self._w_momentum, self._w_dampening]),
            widgets.HBox([self._w_nesterov]),
        ])

    @property
    def lr(self):
        return self._w_lr.value

    @lr.setter
    def lr(self, new_val):
        self._w_lr.value = new_val

    @property
    def weight_decay(self):
        return self._w_weight_decay.value

    @weight_decay.setter
    def weight_decay(self, new_val):
        self._w_weight_decay.value = new_val

    @property
    def momentum(self):
        return self._w_momentum.value

    @momentum.setter
    def momentum(self, new_val):
        self._w_momentum.value = new_val

    @property
    def dampening(self):
        return self._w_dampening.value

    @dampening.setter
    def dampening(self, new_val):
        self._w_dampening.value = new_val

    @property
    def nesterov(self):
        return self._w_nesterov.value == "True"

    @nesterov.setter
    def nesterov(self, new_val):
        self._w_nesterov.value = "True" if new_val else "False"

    def __call__(self):
        return self._w_main

    def get_optimizer(self):
        return {"type": "SGD", "lr": self.lr, 
                "weight_decay": self.weight_decay, "momentum": self.momentum,
                "dampening": self.dampening, "nesterov": self.nesterov}

    def set_params(self, params):
        self.lr = params["lr"]
        self.weight_decay = params["weight_decay"]
        self.momentum = params["momentum"]
        self.dampening = params["dampening"]
        self.nesterov = params["nesterov"]
