import ipywidgets as widgets

from collections import OrderedDict

from .example_prediction_widget import ExamplePredictionWidget


class ResultsExplorerWidget(object):
    def __init__(self):

        self._trained_models = OrderedDict()
        self._w_trained_models = widgets.Dropdown(
            options=self._trained_models.keys(),
            description="Models")

        self._w_example_explorer = OrderedDict()

        self._w_main = widgets.VBox([
            self._w_trained_models,
            widgets.Label()
        ])
        

        self._w_trained_models.observe(self._select_callback, names=["value"])
              


    def update_models(self, trained_models):
        self._trained_models.update(trained_models)
        self._w_trained_models.options = trained_models.keys()
   
    def _select_callback(self, change):
        name = self._w_trained_models.value
        if name not in self._w_example_explorer:
            self._w_example_explorer[name] = ExamplePredictionWidget(
                self._trained_models[name])
        self._w_main.children = [self._w_trained_models, 
                                 self._w_example_explorer[name]()]

    def __call__(self):
        return self._w_main
