import ipywidgets as widgets


class AdamWidget(object):
    def __init__(self):

        self._w_lr = widgets.FloatText(value=1e-3, description="LR:")
        self._w_weight_decay = widgets.FloatText(value=0., description="L2")
        self._w_beta1 = widgets.FloatText(value=0.9, description="Beta1")
        self._w_beta2 = widgets.FloatText(value=0.999, description="Beta2")
        self._w_eps = widgets.FloatText(value=1e-8, description="Eps.")
        self._w_amsgrad = widgets.Dropdown(options=["True", "False"],
                                           value="False",
                                           description="AMS Grad")

        self._w_main = widgets.VBox([
            widgets.HBox([self._w_lr, self._w_weight_decay]),
            widgets.HBox([self._w_beta1, self._w_beta2]),
            widgets.HBox([self._w_eps, self._w_amsgrad]),
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
    def beta1(self):
        return self._w_beta1.value

    @beta1.setter
    def beta1(self, new_val):
        self._w_beta1.value = new_val

    @property
    def beta2(self):
        return self._w_beta2.value

    @beta2.setter
    def beta2(self, new_val):
        self._w_beta2.value = new_val

    @property
    def eps(self):
        return self._w_eps.value

    @eps.setter
    def eps(self, new_val):
        self._w_eps.value = new_val

    @property
    def amsgrad(self):
        return self._w_amsgrad.value == "True"

    @amsgrad.setter
    def amsgrad(self, new_val):
        self._w_amsgrad.value = "True" if new_val else "False"

    def __call__(self):
        return self._w_main

    def get_optimizer(self):
        return {"type": "Adam", "lr": self.lr, 
                "weight_decay": self.weight_decay, "eps": self.eps,
                "betas": (self.beta1, self.beta2), 
                "amsgrad": self.amsgrad}

    def set_params(self, params):
        self.lr = params["lr"]
        self.weight_decay = params["weight_decay"]
        self.eps = params["eps"]
        self.beta1 = params["betas"][0]
        self.beta2 = params["betas"][1]
        self.amsgrad = params["amsgrad"]
