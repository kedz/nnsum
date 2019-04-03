from pathlib import Path
from ..parameterized import Parameterized
from ..hparam_registry import HParams

import torch
import numpy as np
import random
import json


@Parameterized.register_object("trainer.multiclass_training_loop")
class MultiClassTrainingLoop(Parameterized):
    
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
    def meta_loss_functions(self):
        pass

    @hparams()
    def metrics(self):
        pass

#    @hparams()
#    def metric_selector(self):
#        pass

#    @hparams()
#    def metric_mode(self):
#        pass

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

    @hparams(default=True)
    def warm_start(self):
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
        if not self.warm_start:
            self.model.initialize_parameters()

        self.set_seed() 
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

                #summary_metric = eval(self.metric_selector)(epoch_result)
                self.lr_scheduler.step(epoch_result)
                #print("SUMMARY_METRIC =", summary_metric)

                print(json.dumps(epoch_result), file=results_fp, flush=True)
                ckpt_path = ckpt_dir / "{}.ckpt.{}.pth".format(
                    self.model_directory.name, self.epoch)
                torch.save(self.model, ckpt_path)
                
                #train_state_path = self.model_directory / "train_state.pth"
                #torch.save({"epoch": self.epoch, "lr_scheduler


    def apply_loss_functions(self, forward_state, batch):
        loss = 0
        for field, loss_funcs in self.loss_functions.items():
            field_loss = 0
            for name, lf in loss_funcs["modules"].items():
                field_loss += lf["weight"] * lf["module"](
                    forward_state[field], batch)
            loss += loss_funcs["weight"] * field_loss
        return loss

    def apply_meta_loss_functions(self, forward_states, batch):
        loss = 0
        for name, lf in self.meta_loss_functions.items():
            if self.epoch >= lf.get("start_epoch", -1):
                loss += lf["weight"] * lf["module"](forward_states, batch)
        return loss

    def apply_metrics(self, forward_states, batch, mode):
        default = True if mode == "validation" else False
        for field, metrics in self.metrics.items():
            for name, metric in metrics.items():
                if metric.get(mode, default):
                    metric["module"](forward_states[field], batch)

    def train_epoch(self):
        self.model.train()
        
        batches = self.training_minibatches
        
        for field, loss_funcs in self.loss_functions.items():
            for name, lf in loss_funcs["modules"].items():
                lf["module"].reset()
        for name, loss_func in self.meta_loss_functions.items():
            loss_func["module"].reset()
        for field, metrics in self.metrics.items():
            for name, metric in metrics.items():
                metric["module"].reset()

        total_loss = 0
        for step, batch in enumerate(batches, 1):
            self.optimizer.zero_grad()
        
            forward_states = self.model(batch)
            self.apply_metrics(forward_states, batch, "training")

            loss = self.apply_loss_functions(forward_states, batch)
            loss += self.apply_meta_loss_functions(forward_states, batch)
            total_loss += loss.item()
            loss.backward()

            self.optimizer.step()
            
            status = self._train_template.format(
                self.epoch, step, len(batches), total_loss / step)
            print(status, end="\r" if step < len(batches) else "\n",
                  flush=True)


                
        loss_func_results = {}
        for field, loss_funcs in self.loss_functions.items():
            lf_results = {}
            for name, lf in loss_funcs["modules"].items():
                lf_results[name] = lf["module"].mean()
            loss_func_results[field] = lf_results    
        loss_func_results["AVG_LOSS"] = total_loss / len(batches)      

        metric_results = {}
        for field, metrics in self.metrics.items():
            field_results = {}
            for name, metric in metrics.items():
                if metric.get("training", False):
                    field_results[name] = metric["module"].compute()
            metric_results[field] = field_results
        
#        for metric in self.metrics:
#            if metric.get("display_training", False) and \
#                    metric.get("training", False):
#                metric["module"].pretty_print()
 
        return {"loss_functions": loss_func_results, "metrics": metric_results}

    def valid_epoch(self):
        self.model.eval()
        for field, loss_funcs in self.loss_functions.items():
            for name, lf in loss_funcs["modules"].items():
                lf["module"].reset()
        for name, loss_func in self.meta_loss_functions.items():
            loss_func["module"].reset()

        for field, metrics in self.metrics.items():
            for name, metric in metrics.items():
                metric["module"].reset()
       
        batches = self.validation_minibatches

        total_loss = 0
        for step, batch in enumerate(batches, 1):
        
            forward_states = self.model(batch)
            self.apply_metrics(forward_states, batch, "validation")

            loss = self.apply_loss_functions(forward_states, batch)
            loss += self.apply_meta_loss_functions(forward_states, batch)
            total_loss += loss.item()

            status = self._valid_template.format(
                self.epoch, step, len(batches), total_loss / step)
            print(status, end="\r" if step < len(batches) else "\n",
                  flush=True)

        loss_func_results = {}
        for field, loss_funcs in self.loss_functions.items():
            lf_results = {}
            for name, lf in loss_funcs["modules"].items():
                lf_results[name] = lf["module"].mean()
            loss_func_results[field] = lf_results    

        loss_func_results["AVG_LOSS"] = total_loss / len(batches)      

        metric_results = {}
        for field, metrics in self.metrics.items():
            field_results = {}
            for name, metric in metrics.items():
                if metric.get("validation", True):
                    field_results[name] = metric["module"].compute()
            metric_results[field] = field_results
 
#        for metric in self.metrics:
#            if metric.get("display_validation", False) and \
#                    metric.get("validation", True):
#                metric["module"].pretty_print()
        
        return {"loss_functions": loss_func_results, "metrics": metric_results}
