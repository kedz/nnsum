import ipywidgets as widgets
from .dataset_widget import DatasetWidget
from .model_widget import ModelWidget
from .optimizer_widget import OptimizerWidget
from .trainer_widget import TrainerWidget
from .train_model_widget import TrainModelWidget





#from .dataset_creator_widget import DatasetCreatorWidget
#from .model_creator_widget import ModelCreatorWidget
#from .model_trainer_widget import ModelTrainerWidget
#from .model_explorer_widget import ModelExplorerWidget


class CopySequenceTaskWidget(object):
    def __init__(self):

        self._w_dataset = DatasetWidget()
        self._w_model = ModelWidget()
        self._w_optimizer = OptimizerWidget()
        self._w_trainer = TrainerWidget()
        self._w_train_model = TrainModelWidget(
            self._w_dataset,
            self._w_model,
            self._w_optimizer,
            self._w_trainer)

#        self._dataset_creator = DatasetCreatorWidget()
#        self._model_creator = ModelCreatorWidget()
#        self._model_explorer = ModelExplorerWidget()
#        self._model_trainer = ModelTrainerWidget(self._dataset_creator,
                               # i#                 self._model_creator,
                                #                 self._model_explorer)

        self._main_widget = widgets.VBox([
            self._w_dataset(),
            widgets.Label(" "),
            self._w_model(),
            widgets.Label(" "),
            self._w_optimizer(),
            widgets.Label(" "),
            self._w_trainer(),
            widgets.Label(" "),
            self._w_train_model(),
        ])

    def __call__(self):
        return self._main_widget
