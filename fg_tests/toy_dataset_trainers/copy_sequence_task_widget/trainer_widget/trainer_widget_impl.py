import ipywidgets as widgets

from collections import OrderedDict


class TrainerWidget(object):
    def __init__(self):

        self._trainer_params = OrderedDict()

        self._w_trainers = widgets.Dropdown(options=[], value=None,
                                            description="Trainer:")
        self._w_trainers.observe(self._select_callback, names=["value"])
        self._w_trainer_name = widgets.Text(value="trainer-1", 
                                            description="Name:")
        self._w_batch_size = widgets.IntText(value=16, 
                                             description="Batch Size:")
        self._w_epochs = widgets.IntText(value=20, description="Epochs:")
        self._w_create_button = widgets.Button(description="Create Trainer")
        self._w_create_button.on_click(self.create_button_click)
        self._w_message = widgets.Label()

        self._w_main = widgets.VBox([
            widgets.Label("Trainers"),
            self._w_trainers,
            self._w_trainer_name,
            widgets.HBox([self._w_batch_size, self._w_epochs]),
            self._w_create_button,
            self._w_message,
        ])

        self._callbacks = []

    def __call__(self):
        return self._w_main

    @property
    def trainer_name(self):
        return self._w_trainer_name.value.strip()

    @trainer_name.setter
    def trainer_name(self, new_val):
        self._w_trainer_name.value = new_val

    @property
    def batch_size(self):
        return self._w_batch_size.value

    @batch_size.setter
    def batch_size(self, new_val):
        self._w_batch_size.value = new_val

    @property
    def epochs(self):
        return self._w_epochs.value

    @epochs.setter
    def epochs(self, new_val):
        self._w_epochs.value = new_val

    def create_button_click(self, button):
        name = self.trainer_name
        if name in self._trainer_params or name == "":
            self._w_message.value = (
                "Name must be unique non-empty string."
            )
            return
        else:
            self._w_message.value = ""
        
        params = {
            "name": self.trainer_name,
            "batch_size": self.batch_size,
            "epochs": self.epochs
        }
        self._trainer_params[name] = params
        self._w_trainers.options = self._trainer_params.keys()
        self._w_trainers.value = name
        self.trainer_name = "trainer-{}".format(len(self._trainer_params) + 1)

        for callback in self._callbacks:
            callback(self._trainer_params)

    def _select_callback(self, change):
        params = self._trainer_params[change["new"]]
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]

    def register_new_trainers(self, callback):
        self._callbacks.append(callback)

    def get_params(self):
        return self._trainer_params
