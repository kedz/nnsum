from pathlib import Path
from ..parameterized import Parameterized
from ..hparam_registry import HParams

import torch
import numpy as np
import random
import json


@Parameterized.register_object("trainer.training_loop")
class TrainingLoop(Parameterized):
    
    hparams = HParams()

    @hparams(default=5)
    def max_epochs(self):
        pass

    @hparams(type="submodule")
    def training_minibatches(self):
        pass

    @hparams(type="submodule")
    def validation_minibatches(self):
        pass

    @hparams(type="submodule")
    def model(self):
        pass

    @hparams()
    def loss_functions(self):
        pass

    @hparams()
    def metrics(self):
        pass

    @hparams(default=None, required=False)
    def example_loggers(self):
        pass

    @hparams()
    def optimizer(self):
        pass

    @hparams()
    def lr_scheduler(self):
        pass

    @property
    def epoch(self):
        return self._epoch

    @hparams()
    def experiment_directory(self):
        pass

    @property
    def model_checkpoint_directory(self):
        return Path(self.experiment_directory) / "model_checkpoints"

    @property
    def output_checkpoint_directory(self):
        return Path(self.experiment_directory) / "output_checkpoints"

    @property
    def results_path(self):
        exp_dir = Path(self.experiment_directory)
        return exp_dir / "{}.results.jsonl".format(exp_dir.name)

    @hparams(default=0)
    def seed(self):
        pass

    @hparams(default=-1)
    def device(self):
        pass

    def init_object(self):

        if self.example_loggers is None:
            self._example_loggers = {}

        self._epoch = 0

        EPOCH_STRLEN = str(len(str(self.max_epochs)))
        TR_BATCHES_STRLEN = str(len(str(len(self.training_minibatches))))
        VA_BATCHES_STRLEN = str(len(str(len(self.validation_minibatches))))
        self._train_template = (
            "TRAIN EPOCH-{:0" + EPOCH_STRLEN +"d}-{:0" + TR_BATCHES_STRLEN 
            + "d}/{:0" + TR_BATCHES_STRLEN +"d}  LOSS={:5.3f}"
        )       
        self._valid_template = (
            "VALID EPOCH-{:0" + EPOCH_STRLEN +"d}-{:0" + VA_BATCHES_STRLEN 
            + "d}/{:0" + VA_BATCHES_STRLEN +"d}  LOSS={:5.3f}"
        )       

        self._set_seed() 
        self.model.initialize_parameters()
        if self.device > -1:
            self._model = self.model.cuda(self.device)
            self.training_minibatches.device = self.device
            self.validation_minibatches.device = self.device

        self.model_checkpoint_directory.mkdir(exist_ok=True, parents=True)
        self.output_checkpoint_directory.mkdir(exist_ok=True, parents=True)
    
        self.optimizer.params = self.model.parameters()
        self.lr_scheduler.optimizer = self.optimizer.optim

    def _set_seed(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        #torch.backends.cudnn.deterministic = True                        

    def _checkpoint_model(self, epoch_result):
        name = self.model_checkpoint_directory.parent.name
        ckpt_path = self.model_checkpoint_directory / "{}.ckpt.{}.pth".format(
            name, self.epoch)
        torch.save(self.model, ckpt_path)

    def _open_output_checkpoints(self, mode):

        # default is to save outputs during validation and not during training.
        default = True if mode == "validation" else False
        open_files = {}
        for name, logger in self.example_loggers.items():
            if logger.get("save_" + mode, default):
                output_dir = self.output_checkpoint_directory / mode
                output_dir.mkdir(exist_ok=True, parents=True)
                path = output_dir / "{}.ckpt.{}.txt".format(name, self.epoch)
                open_files[name] = path.open("w")
        return open_files

    def _close_output_checkpoints(self, output_file_pointers):
        for fp in output_file_pointers.values():
            fp.flush()
            fp.close()

    def _apply_example_logger(self, batch, forward_state, mode, output_fps):
        # default is to log examples during validation and not during training.
        default = True if mode == "validation" else False
        output_texts = {}
        for name, logger in self.example_loggers.items():
            if logger.get("save_" + mode, default):
                logger["module"](batch, forward_state, output_fps[name])

    def run(self):
        if self.device > -1:
            self._model = self.model.cuda(self.device)
            self.training_minibatches.device = self.device
            self.validation_minibatches.device = self.device

        self.optimizer.params = self.model.parameters()
        self.lr_scheduler.optimizer = self.optimizer.optim

        with self.results_path.open("w") as results_fp:
            for _ in range(self.max_epochs):
                
                self._epoch += 1
                train_losses_metrics = self.train_epoch()
                valid_losses_metrics = self.valid_epoch()
                print()
                
                epoch_result = {
                    "epoch": self.epoch,
                    "training": train_losses_metrics,
                    "validation": valid_losses_metrics,
                }

                #summary_metric = eval(self.metric_selector)(epoch_result)
                self.lr_scheduler.step(epoch_result)
                #print("SUMMARY_METRIC =", summary_metric)

                print(json.dumps(epoch_result), file=results_fp, flush=True)
                self._checkpoint_model(epoch_result)


    def apply_loss_functions(self, forward_state, batch):
        
        loss = 0
        for lf in self.loss_functions.values():
            loss += lf["weight"] * lf["module"](forward_state, batch)
        return loss

    def apply_metrics(self, forward_state, batch, mode):
        for metric in self.metrics.values():
            if metric.get(mode, True if mode == "validation" else False):
                metric["module"](forward_state, batch)

    def train_epoch(self):
        self.model.train()
        output_pointers = self._open_output_checkpoints("training")
        
        batches = self.training_minibatches
        
        for loss_func in self.loss_functions.values():
            loss_func["module"].reset()
        for metric in self.metrics.values():
            metric["module"].reset()

        total_loss = 0
        for step, batch in enumerate(batches, 1):
            self.optimizer.zero_grad()
        
            forward_state = self.model(batch)
            self.apply_metrics(forward_state, batch, "training")

            loss = self.apply_loss_functions(forward_state, batch)
            total_loss += loss.item()
            loss.backward()

            self.optimizer.step()
         
            self._apply_example_logger(
                batch, forward_state, "training", output_pointers)
   
            status = self._train_template.format(
                self.epoch, step, len(batches), total_loss / step)
            print(status, end="\r" if step < len(batches) else "\n",
                  flush=True)

        loss_func_results = {
            name: loss_func["module"].mean()
            for name, loss_func in self.loss_functions.items()
        }
        loss_func_results["AVG_LOSS"] = total_loss / len(batches)      
        metric_results = {name: metric["module"].compute()
                          for name, metric in self.metrics.items()
                          if metric.get("training", False)}

        for metric in self.metrics.values():
            if metric.get("display_training", False) and \
                    metric.get("training", False):
                metric["module"].pretty_print()

        self._close_output_checkpoints(output_pointers)
 
        return {"loss_functions": loss_func_results, "metrics": metric_results}

    def valid_epoch(self):
        self.model.eval()
        output_pointers = self._open_output_checkpoints("validation")
        
        batches = self.validation_minibatches

        for loss_func in self.loss_functions.values():
            loss_func["module"].reset()
        for metric in self.metrics.values():
            metric["module"].reset()

        total_loss = 0
        for step, batch in enumerate(batches, 1):
        
            forward_state = self.model(batch)
            self.apply_metrics(forward_state, batch, "validation")

            loss = self.apply_loss_functions(forward_state, batch)
            total_loss += loss.item()

            self._apply_example_logger(
                batch, forward_state, "validation", output_pointers)

            status = self._valid_template.format(
                self.epoch, step, len(batches), total_loss / step)
            print(status, end="\r" if step < len(batches) else "\n",
                  flush=True)

        loss_func_results = {
            name: loss_func["module"].mean()
            for name, loss_func in self.loss_functions.items()
        }
        loss_func_results["AVG_LOSS"] = total_loss / len(batches)      
        metric_results = {name: metric["module"].compute()
                          for name, metric in self.metrics.items()
                          if metric.get("validation", True)}

        for metric in self.metrics.values():
            if metric.get("display_validation", False) and \
                    metric.get("validation", True):
                metric["module"].pretty_print()
            print()

        self._close_output_checkpoints(output_pointers)
        
        return {"loss_functions": loss_func_results, "metrics": metric_results}
