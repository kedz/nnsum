import ipywidgets as widgets

from collections import OrderedDict

from .training_run_widget import TrainingRunWidget


class TrainModelWidget(object):
    def __init__(self, datasets_widget, models_widget, optimizers_widget,
                 trainers_widget):

        self._trained_models = OrderedDict()
        self._dataset_params = datasets_widget.get_params()
        self._model_params = models_widget.get_params()
        self._optimizer_params = optimizers_widget.get_params()
        self._trainer_params = trainers_widget.get_params()


        self._w_trained_models = widgets.Dropdown(options=[],
                                                  description="Trained Model:")
        self._w_trained_models.observe(self._select_trained, names=["value"])
        self._w_train_name = widgets.Text(value="trained-model-1",
                                          description="Name:")
          
        self._w_datasets = widgets.Dropdown(
            options=self._dataset_params.keys(),
            description="Dataset:")
        self._w_models = widgets.Dropdown(
            options=self._model_params.keys(),
            description="Model:")
        self._w_optimizers = widgets.Dropdown(
            options=self._optimizer_params.keys(),
            description="Optimizer:")
        self._w_trainers = widgets.Dropdown(
            options=self._trainer_params.keys(),
            description="Trainer:")
        self._w_start_button = widgets.Button(description="Train Model")
        self._w_start_button.on_click(self._start_callback)
        self._w_train_runs = {}

        datasets_widget.register_new_datasets(self._new_dataset)
        models_widget.register_new_models(self._new_model)
        optimizers_widget.register_new_optimizers(self._new_optimizer)
        trainers_widget.register_new_trainers(self._new_trainer)

        self._w_main = widgets.VBox([
            widgets.HBox([
                widgets.VBox([self._w_datasets, self._w_models,
                              self._w_optimizers, self._w_trainers]),
                widgets.VBox([
                    self._w_trained_models,
                    self._w_train_name,
                    self._w_start_button,
                ]) 
            ]),
            widgets.Label(),
        ])

    @property
    def run_name(self):
        return self._w_train_name.value.strip()

    @run_name.setter
    def run_name(self, new_name):
        self._w_train_name.value = new_name

    @property
    def dataset(self):
        return self._w_datasets.value

    @property
    def model(self):
        return self._w_models.value

    @property
    def optimizer(self):
        return self._w_optimizers.value

    @property
    def trainer(self):
        return self._w_trainers.value

    def _new_dataset(self, new_datasets):
        val = self._w_datasets.value
        self._w_datasets.options = new_datasets.keys()
        self._w_datasets.value = val
        self._dataset_params = new_datasets

    def _new_model(self, new_models):
        val = self._w_models.value
        self._w_models.options = new_models.keys()
        self._w_models.value = val
        self._model_params = new_models

    def _new_optimizer(self, new_optimizers):
        val = self._w_optimizers.value
        self._w_optimizers.options = new_optimizers.keys()
        self._w_optimizers.value = val
        self._optimizer_params = new_optimizers

    def _new_trainer(self, new_trainers):
        val = self._w_trainers.value
        self._w_trainers.options = new_trainers.keys()
        self._w_trainers.value = val
        self._trainer_params = new_trainers

    def _start_callback(self, button):
        
        if self.dataset is None:
            return
        if self.model is None:
            return 
        if self.optimizer is None:
            return
        if self.trainer is None:
            return 

        if self.run_name in self._trained_models or self.run_name == "":
            return

        dataset_params = self._dataset_params[self.dataset]
        model_params = self._model_params[self.model]
        optimizer_params = self._optimizer_params[self.optimizer]
        trainer_params = self._trainer_params[self.trainer]

        run_name = self.run_name
        w = TrainingRunWidget(dataset_params, model_params, 
                              optimizer_params, trainer_params,
                              name=run_name)
        self._w_train_runs[self.run_name] = w
        self._w_trained_models.options = \
            list(self._w_trained_models.options) + [run_name]

                             
        self._w_main.children = list(self._w_main.children[:-1]) + [w()] 
        self.run_name = "trained-model-{}".format(
            len(self._w_trained_models.options) + 1)
        self._w_trained_models.value = run_name

        def finished_callback(model_stats):
            print("Saving", run_name)
            self._trained_models[run_name] = model_stats

        w.set_finished_callback(finished_callback)
        w.start_train()

    def __call__(self):
        return self._w_main

    def _select_trained(self, change):
        name = change["new"]
        w = self._w_train_runs[name]
        self._w_main.children = list(self._w_main.children[:-1]) + [w()] 
