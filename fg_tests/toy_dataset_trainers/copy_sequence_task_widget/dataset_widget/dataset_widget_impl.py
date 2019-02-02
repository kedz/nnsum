import ipywidgets as widgets

from .create_dataset_widget import CreateDatasetWidget
from .explore_dataset_widget import ExploreDatasetWidget

from collections import OrderedDict


class DatasetWidget(object):
    def __init__(self):

        self._dataset_params = OrderedDict()
        self._w_datasets = widgets.Dropdown(options=[], value=None,
                                            description="Datasets:")
        self._w_datasets.observe(self._select_callback, names='value')

        self._w_create_dataset = CreateDatasetWidget()
        self._w_create_dataset.register_callback(self._create_callback)

        self._w_tabs = widgets.Tab([self._w_create_dataset(), 
                                    widgets.Label()])
        self._w_tabs.set_title(0, "Create Dataset")
        self._w_tabs.set_title(1, "Explore")
        
        self._w_message = widgets.Label()
       
        self._w_main = widgets.VBox([
            widgets.Label("Datasets"),
            self._w_datasets,
            self._w_tabs,
            self._w_message,    
        ])

        self._callbacks = []

    def _create_callback(self, dataset_params):
        name = dataset_params["name"]
        if name in self._dataset_params or name == "":
            self._w_message.value = (
                "Name must be unique non-empty string."
            )
            return
        else:
            self._w_message.value = ""
        
        self._dataset_params[name] = dataset_params
        self._w_datasets.options = self._dataset_params.keys()
        self._w_create_dataset.dataset_name = "dataset-{}".format(
            len(self._dataset_params) + 1)

        for callback in self._callbacks:
            callback(self._dataset_params)

    def _select_callback(self, change):
        name = change["new"]
        dataset_params = self._dataset_params[name]
        self._w_create_dataset.vocab_size = dataset_params["vocab_size"]
        self._w_create_dataset.min_length = dataset_params["min_length"]
        self._w_create_dataset.max_length = dataset_params["max_length"]
        self._w_create_dataset.train_size = dataset_params["train_size"]
        self._w_create_dataset.test_size = dataset_params["test_size"]
        self._w_create_dataset.train_seed = dataset_params["train_seed"]
        self._w_create_dataset.test_seed = dataset_params["test_seed"]
        
        if name != change["old"]:
            self._w_tabs.children = [self._w_create_dataset(), 
                                     ExploreDatasetWidget(dataset_params)()]

    def __call__(self):
        return self._w_main

    def register_new_datasets(self, callback):
        self._callbacks.append(callback)

    def get_params(self):
        return self._dataset_params
