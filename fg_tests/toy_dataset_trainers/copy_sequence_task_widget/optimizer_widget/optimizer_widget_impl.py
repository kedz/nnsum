import ipywidgets as widgets

from .sgd_widget import SGDWidget
from .adam_widget import AdamWidget

from collections import OrderedDict


class OptimizerWidget(object):
    def __init__(self):

        self._optimizer_params = OrderedDict()
        self._w_optimizers = widgets.Dropdown(options=[], value=None,
                                              description="Optimizer:")
        self._w_algos = widgets.Dropdown(options=["SGD", "Adam"], 
                                         value="SGD", description="Algo:")
        self._w_algos.observe(self._algo_select, names=["value"])
        self._w_optimizer_name = widgets.Text(value="optimizer-1",
                                              description="Name:")
        self._w_optimizers.observe(self._select_callback, names=["value"])
        self._w_algo_widgets = {"SGD": SGDWidget(), "Adam": AdamWidget()}
        self._w_current_algo = self._w_algo_widgets["SGD"]
        self._w_message = widgets.Label()
        self._w_create_button = widgets.Button(description="Create Optimizer")
        self._w_create_button.on_click(self.create_button_click)

        self._w_main = widgets.VBox([
            widgets.Label("Optimizers"),
            self._w_optimizers,
            widgets.HBox([self._w_algos, self._w_optimizer_name]),
            self._w_algo_widgets["SGD"](),
            self._w_create_button,
            self._w_message,
        ])

        self._callbacks = []
        self._new_opt_callbacks = []
        self._w_optimizers.observe(self._new_opts_event, names=["options"])

    def __call__(self):
        return self._w_main

    def register_new_optimizers(self, callback):
        self._new_opt_callbacks.append(callback)

    def _new_opts_event(self, change):
        for callback in self._new_opt_callbacks:
            callback(self._optimizer_params)
    
    def register_callback(self, callback):
        self._callbacks.append(callback)

    @property
    def optimizer_name(self):
        return self._w_optimizer_name.value.strip()

    @optimizer_name.setter
    def optimizer_name(self, new_name):
        self._w_optimizer_name.value = new_name

    def create_button_click(self, button):
        params = self._w_current_algo.get_optimizer()
        name = self.optimizer_name
        params["name"] = name
        if name in self._optimizer_params or name == "":
            self._w_message.value = (
                "Name must be unique non-empty string."
            )
            return
        else:
            self._w_message.value = ""

        self._optimizer_params[name] = params
        self._w_optimizers.options = self._optimizer_params.keys()
        self._w_optimizers.value = name
        
        self.optimizer_name = "optimizer-{}".format(
            len(self._optimizer_params) + 1)


        for callback in self._callbacks:
            callback(params)
        self._optimizer_params

    def _select_callback(self, change):
        name = change["new"]
        params = self._optimizer_params[name]
        self._w_current_algo = self._w_algo_widgets[params["type"]]
        
        self._w_main.children = list(self._w_main.children[:3]) + \
                [self._w_current_algo()] + list(self._w_main.children[4:])
        self._w_current_algo.set_params(params)
        self._w_algos.value = params["type"]

    def _algo_select(self, change):
        name = change["new"]
        self._w_current_algo = self._w_algo_widgets[name]
        self._w_main.children = list(self._w_main.children[:3]) + \
                [self._w_current_algo()] + list(self._w_main.children[4:])

    def get_params(self):
        return self._optimizer_params
