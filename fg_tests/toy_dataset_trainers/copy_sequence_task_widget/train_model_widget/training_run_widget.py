import ipywidgets as widgets

import matplotlib
import matplotlib.pyplot as plt
#plt.style.use('ggplot')

import os
import threading
from tempfile import NamedTemporaryFile

from .train_run import start_train_run


class TrainingRunWidget(object):
    def __init__(self, dataset_params, model_params, optimizer_params,
                 trainer_params, name=None):

        self._dataset_params = dataset_params
        self._model_params = model_params
        self._optimizer_params = optimizer_params
        self._trainer_params = trainer_params

        self._w_train_progress = widgets.IntProgress(
            value=0, min=0, max=trainer_params["epochs"], step=1,
            description='Training:', bar_style='')
        self._w_train_status = widgets.Label("Epoch 0/{}".format(
            trainer_params["epochs"]))
        self._w_valid_progress = widgets.IntProgress(
            value=0, min=0, max=trainer_params["epochs"], step=1,
            description='Validation:', bar_style='')
        self._w_valid_status = widgets.Label("Epoch 0/{}".format(
            trainer_params["epochs"]))
        self._w_plot = widgets.Image(format="png")

        self._w_main = widgets.VBox([
            widgets.Label(str(name)),
            widgets.HBox([self._w_train_progress, self._w_train_status]),
            widgets.HBox([self._w_valid_progress, self._w_valid_status]),
            widgets.Box([self._w_plot]),
        ])

        self._finished_callback = None
        self.init_plots()
        
    def __call__(self):
        return self._w_main

    def set_finished_callback(self, callback):
        self._finished_callback = callback

    def train_progress_tick(self):

        self._w_train_progress.value += 1
        self._w_train_status.value = "Epoch {}/{}".format(
            self._w_train_progress.value,
            self._w_train_progress.max)

    def valid_progress_tick(self):

        self._w_valid_progress.value += 1
        self._w_valid_status.value = "Epoch {}/{}".format(
            self._w_valid_progress.value,
            self._w_valid_progress.max)

    def init_plots(self):

        matplotlib.use('Agg')
        self._fig = plt.figure(figsize=(10, 3))
        self._xent_ax = self._fig.add_subplot(1,2,1)
        self._xent_ax.set_xlim(1, self._trainer_params["epochs"])
        self._xent_ax.set_xticks(range(1, self._trainer_params["epochs"] + 1))
        self._xent_ax.set_xlabel("Epochs")
        self._xent_ax.set_ylabel("X-Entropy")
        self._xent_ax.set_ylim(0, 100)
        self._train_xent_line, = self._xent_ax.plot(
            -1, -1, label="Train")
        self._valid_xent_line, = self._xent_ax.plot(
            -1, -1, label="Valid")
        self._max_xent = float("-inf")
        self._xent_ax.legend(loc=1)

        self._acc_ax = self._fig.add_subplot(1,2,2)
        self._acc_ax.set_xlim(1, self._trainer_params["epochs"])
        self._acc_ax.set_xticks(range(1, self._trainer_params["epochs"] + 1))
        self._acc_ax.set_xlabel("Epochs")
        self._acc_ax.set_ylabel("Accuracy")
        self._acc_ax.set_ylim(0, 1)
        self._valid_acc_line, = self._acc_ax.plot(
            -1, -1, label="Valid")
        self._acc_ax.legend(loc=4)


        self._image_file = NamedTemporaryFile(suffix=".png", delete=False)

        self._fig.savefig(self._image_file.name)
        with open(self._image_file.name, "rb") as fp:
            image = fp.read()
            self._w_plot.value = image

    def update_train_plot(self, train_stats):
        train_xent = train_stats["x-entropy"]
        epochs = [x for x in range(1, len(train_xent) + 1)]
        self._train_xent_line.set_xdata(epochs)
        self._train_xent_line.set_ydata(train_xent)
        self._max_xent = max(train_xent[-1], self._max_xent)

        self._xent_ax.set_ylim(0, self._max_xent + .5)
        self._fig.canvas.draw()
        self._fig.savefig(self._image_file.name)

        with open(self._image_file.name, "rb") as fp:
            image = fp.read()
            self._w_plot.value = image

    def update_valid_plot(self, valid_stats):
        valid_xent = valid_stats["x-entropy"]
        valid_acc = valid_stats["accuracy"]
        epochs = [x for x in range(1, len(valid_xent) + 1)]
        self._valid_xent_line.set_xdata(epochs)
        self._valid_xent_line.set_ydata(valid_xent)
        self._max_xent = max(valid_xent[-1], self._max_xent)
        self._xent_ax.set_ylim(0, self._max_xent + .5)

        self._valid_acc_line.set_xdata(epochs)
        self._valid_acc_line.set_ydata(valid_acc)

        self._fig.canvas.draw()
        self._fig.savefig(self._image_file.name)

        with open(self._image_file.name, "rb") as fp:
            image = fp.read()
            self._w_plot.value = image

    def start_train(self):
        worker = threading.Thread(target=self.train_worker)
        worker.start()

    def train_worker(self):

        train_stats = {"x-entropy": []}
        valid_stats = {"x-entropy": [], "accuracy": []}

        def train_callback(stats):
            train_stats["x-entropy"].append(stats["x-entropy"])
            self.train_progress_tick()
            self.update_train_plot(train_stats)

        def valid_callback(stats):
            valid_stats["x-entropy"].append(stats["x-entropy"])
            valid_stats["accuracy"].append(stats["accuracy"])
            self.update_valid_plot(valid_stats)
            self.valid_progress_tick()

        model_stats = start_train_run(
            self._dataset_params, self._model_params,
            self._optimizer_params, self._trainer_params,
            train_complete_callback=train_callback,
            valid_complete_callback=valid_callback)

        plt.close(self._fig)
        if os.path.isfile(self._image_file.name):
            os.remove(self._image_file.name)

        if self._finished_callback:
            self._finished_callback(model_stats)
        return


        import time
        for epoch in range(1, self._trainer_params["epochs"] + 1):

            time.sleep(1)
            self.train_progress_tick()
            import torch
            tr_xent = torch.exp(torch.FloatTensor(1).normal_()).item()
            train_stats["x-entropy"].append(tr_xent)
            #print(train_stats)
            self.update_train_plot(train_stats)

            va_xent = torch.exp(torch.FloatTensor(1).normal_()).item()
            va_acc = torch.FloatTensor(1).uniform_(0,1).item()
            valid_stats["x-entropy"].append(va_xent)
            valid_stats["accuracy"].append(va_acc)
            self.update_valid_plot(valid_stats)
            self.valid_progress_tick()

    def __del__(self):
        
        if os.path.isfile(self._image_file.name):
            os.remove(self._image_file.name)
