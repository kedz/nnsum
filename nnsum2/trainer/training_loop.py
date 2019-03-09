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

    @hparams()
    def metric_selector(self):
        pass

    @hparams()
    def metric_mode(self):
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
    def model_directory(self):
        pass

    @hparams(default=0)
    def seed(self):
        pass

    
    @property
    def results_path(self):
        return self.model_directory / "{}.results.jsonl".format(
            self.model_directory.name)

    @hparams(default=-1)
    def device(self):
        pass

    def init_object(self):
        self._epoch = 0

        self._model_directory = Path(self._model_directory)
        self._model_directory.mkdir(exist_ok=True, parents=True)

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
 
    def set_seed(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        #torch.backends.cudnn.deterministic = True                        

    def run(self):
       
        self.set_seed() 
        self.model.initialize_parameters()
        if self.device > -1:
            self._model = self.model.cuda(self.device)
            self.training_minibatches.device = self.device
            self.validation_minibatches.device = self.device

        ckpt_dir = self.model_directory / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        self.optimizer.params = self.model.parameters()
        self.lr_scheduler.optimizer = self.optimizer.optim

        with self.results_path.open("w") as results_fp:
            for _ in range(self.max_epochs):
                
                self._epoch += 1
                train_losses_metrics = self.train_epoch()
                valid_losses_metrics = self.valid_epoch()
                
                epoch_result = {
                    "epoch": self.epoch,
                    "training": train_losses_metrics,
                    "validation": valid_losses_metrics,
                }

                summary_metric = eval(self.metric_selector)(epoch_result)
                self.lr_scheduler.step(summary_metric)
                print("SUMMARY_METRIC =", summary_metric)

                print(json.dumps(epoch_result), file=results_fp, flush=True)
                ckpt_path = ckpt_dir / "{}.ckpt.{}.pth".format(
                    self.model_directory.name, self.epoch)
                torch.save(self.model, ckpt_path)
                
                #train_state_path = self.model_directory / "train_state.pth"
                #torch.save({"epoch": self.epoch, "lr_scheduler


    def apply_loss_functions(self, forward_state, batch):
        
        loss = 0
        for lf in self.loss_functions:
            loss += lf["weight"] * lf["module"](forward_state, batch)
        return loss

    def apply_metrics(self, forward_state, batch, mode):
        for metric in self.metrics:
            if metric.get(mode, True if mode == "validation" else False):
                metric["module"](forward_state, batch)

    def train_epoch(self):
        
        batches = self.training_minibatches
        
        for loss_func in self.loss_functions:
            loss_func["module"].reset()
        for metric in self.metrics:
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
            
            status = self._train_template.format(
                self.epoch, step, len(batches), total_loss / step)
            print(status, end="\r" if step < len(batches) else "\n",
                  flush=True)

        loss_func_results = {loss_func["name"]: loss_func["module"].mean()
                             for loss_func in self.loss_functions}
        loss_func_results["AVG_LOSS"] = total_loss / len(batches)      
        metric_results = {metric["name"]: metric["module"].compute()
                          for metric in self.metrics
                          if metric.get("training", False)}

        for metric in self.metrics:
            if metric.get("display_training", False) and \
                    metric.get("training", False):
                metric["module"].pretty_print()
 
        return {"loss_functions": loss_func_results, "metrics": metric_results}

    def valid_epoch(self):
        
        batches = self.validation_minibatches

        for loss_func in self.loss_functions:
            loss_func["module"].reset()
        for metric in self.metrics:
            metric["module"].reset()

        total_loss = 0
        for step, batch in enumerate(batches, 1):
        
            forward_state = self.model(batch)
            self.apply_metrics(forward_state, batch, "validation")

            loss = self.apply_loss_functions(forward_state, batch)
            total_loss += loss.item()

            status = self._valid_template.format(
                self.epoch, step, len(batches), total_loss / step)
            print(status, end="\r" if step < len(batches) else "\n",
                  flush=True)

        loss_func_results = {loss_func["name"]: loss_func["module"].mean()
                             for loss_func in self.loss_functions}
        loss_func_results["AVG_LOSS"] = total_loss / len(batches)      
        metric_results = {metric["name"]: metric["module"].compute()
                          for metric in self.metrics
                          if metric.get("validation", True)}

        for metric in self.metrics:
            if metric.get("display_validation", False) and \
                    metric.get("validation", True):
                metric["module"].pretty_print()
        
        return {"loss_functions": loss_func_results, "metrics": metric_results}
