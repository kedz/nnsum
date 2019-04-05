from pathlib import Path
from ..parameterized import Parameterized
from ..hparam_registry import HParams

import torch
import numpy as np
import random
import json


@Parameterized.register_object("trainer.decoder_model_training_loop")
class DecoderModelTrainingLoop(Parameterized):
    
    hparams = HParams()

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
    def optimizer(self):
        pass

    @hparams()
    def lr_scheduler(self):
        pass

    @hparams(default=5)
    def max_epochs(self):
        pass

    @hparams()
    def search_algorithms(self):
        pass

    @hparams()
    def rerankers(self):
        pass

    @hparams()
    def postprocessors(self):
        pass

    @hparams()
    def metrics(self):
        pass
 
    @hparams(default=-1)
    def device(self):
        pass
 
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
 
    @property
    def epoch(self):
        return self._epoch

    def init_object(self):
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

        for name in self.rerankers.keys():
            if name in self.search_algorithms:
                raise Exception((
                    "duplicate name: {},".formant(name) +
                    "search algorithms and rerankers must have " +
                    "different names."))

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
        for name, postproc in self.postprocessors.items():
            if postproc.get("save_" + mode, default):
                output_dir = self.output_checkpoint_directory / name
                output_dir.mkdir(exist_ok=True, parents=True)
                path = output_dir / "{}.ckpt.{}.txt".format(name, self.epoch)
                open_files[name] = path.open("w")
        return open_files

    def _close_output_checkpoints(self, output_file_pointers):
        for fp in output_file_pointers.values():
            fp.flush()
            fp.close()

    def _reset_losses_and_metrics(self):
        for loss_func in self.loss_functions.values():
            loss_func["module"].reset()
        for metric in self.metrics.values():
            metric["module"].reset()

    def _apply_search_algos(self, batch, encoded_inputs, mode):
        
        # default is to search during validation and not during training.
        default = True if mode == "validation" else False
        search_states = {}
        for name, search_algo in self.search_algorithms.items():
            if search_algo.get(mode, default):
                search_states[name] = search_algo["module"](
                    self.model, batch, encoded_inputs=encoded_inputs)
        return search_states

    def _apply_rerankers(self, batch, search_states, mode):

        # default is to rerank during validation and not during training.
        default = True if mode == "validation" else False
        for name, reranker in self.rerankers.items():
            if reranker.get(mode, default):
                search_states[name] = reranker["module"](
                    batch, search_states[reranker["input"]])
        return search_states

    def _apply_postprocessors(self, batch, search_states,
                              mode, output_fps):
        # default is to postprocess during validation and not during training.
        default = True if mode == "validation" else False
        output_texts = {}
        for name, postprocessor in self.postprocessors.items():
            if postprocessor.get(mode, default):
                search_state = search_states[postprocessor["input"]]
                texts = postprocessor["module"](batch, search_state)
            
                if name in output_fps:
                    self._write_postproc_output(
                        batch["target_reference_strings"], texts, 
                        output_fps[name])

                output_texts[name] = texts 
        return output_texts

    def _write_postproc_output(self, batch_refs, batch_hyps, fp):
        lines = []
        for refs, hyp in zip(batch_refs, batch_hyps):
            lines.extend(["REF: {}".format(ref) for ref in refs])
            lines.append("HYP: {}\n".format(hyp))
        print("\n".join(lines), file=fp)

    def _apply_metrics(self, batch, search_states, postprocessed_texts,
                       mode):
        # default is to postprocess during validation and not during training.
        default = True if mode == "validation" else False
        for metric in self.metrics.values():
            if metric.get(mode, default):
                if metric["input_source"] == "search":
                    metric_input = search_states[metric['input']]
                elif metric["input_source"] == "postprocessor":
                    metric_input = postprocessed_texts[metric['input']]
                metric["module"](batch, metric_input)

    def _apply_loss_functions(self, forward_state, batch):
        loss = 0
        for lf in self.loss_functions.values():
            loss += lf["weight"] * lf["module"](forward_state, batch)
        return loss

    def _epoch_results(self, avg_loss, mode):
        # default is to apply metrics during validation and not during 
        # training.
        default = True if mode == "validation" else False
        loss_func_results = {name: lf["module"].mean()
                             for name, lf in self.loss_functions.items()}
        loss_func_results["__avgloss__"] = avg_loss
        metric_results = {name: metric["module"].compute()
                          for name, metric in self.metrics.items()
                          if metric.get(mode, default)}
        return {"loss_functions": loss_func_results, "metrics": metric_results}

    def _display_metrics(self, mode):
        # default is to apply metrics during validation and not during 
        # training.
        default = True if mode == "validation" else False
        for name, metric in self.metrics.items():
            if metric.get("display_" + mode, default) and \
                    metric.get(mode, default):
                print("metric: {}".format(name))
                metric["module"].pretty_print()

    def _train_epoch(self, batches):
        
        self.model.train()
        output_pointers = self._open_output_checkpoints("training")
        self._reset_losses_and_metrics() 
        total_loss = 0

        for step, batch in enumerate(batches, 1):
            self.optimizer.zero_grad()
            encoded_inputs = self.model.encode(batch)

            # Optionally perform generation for monitoring outputs or for 
            # taking evaluation metrics on generated outputs. 
            search_states = self._apply_search_algos(
                batch, encoded_inputs, "training")

            # Optionally perform reranking of search candidates. This updates
            # search states. 
            self._apply_rerankers(
                batch, search_states, "training")

            # Optionally perform postprocessing on final search states.
            # Postprocessed output is optionally written to files in 
            # output_pointers and also cached for use by evaluation metrics.
            postprocessed_texts = self._apply_postprocessors(
                batch, search_states, "training", output_pointers)

            # Optionally apply evaluation metrics.
            self._apply_metrics(
                batch, search_states, postprocessed_texts, "training")

            # Complete the forward pass of the decoder and compute the 
            # loss function. 
            forward_state = self.model(batch, encoded_inputs=encoded_inputs)
            loss = self._apply_loss_functions(forward_state, batch)
            total_loss += loss.item()
            
            # Perform the backward pass and update the model weights. 
            loss.backward()
            self.optimizer.step()
            
            status = self._train_template.format(
                self.epoch, step, len(batches), total_loss / step)
            print(status, end="\r" if step < len(batches) else "\n",
                  flush=True)

        avg_loss = total_loss / len(batches)
        results = self._epoch_results(avg_loss, "training")
        self._display_metrics("training")
        self._close_output_checkpoints(output_pointers)
        return results

    def _valid_epoch(self, batches):
        
        self.model.eval()
        output_pointers = self._open_output_checkpoints("validation")
        self._reset_losses_and_metrics()
        total_loss = 0

        for step, batch in enumerate(batches, 1):
            encoded_inputs = self.model.encode(batch)

            # Optionally perform generation for monitoring outputs or for 
            # taking evaluation metrics on generated outputs. 
            search_states = self._apply_search_algos(
                batch, encoded_inputs, "validation")

            # Optionally perform reranking of search candidates. This updates
            # search states. 
            self._apply_rerankers(
                batch, search_states, "validation")

            # Optionally perform postprocessing on final search states.
            # Postprocessed output is optionally written to files in 
            # output_pointers and also cached for use by evaluation metrics.
            postprocessed_texts = self._apply_postprocessors(
                batch, search_states, "validation", output_pointers)

            # Optionally apply evaluation metrics.
            self._apply_metrics(
                batch, search_states, postprocessed_texts, "validation")

            # Complete the forward pass of the decoder and compute the 
            # loss function. 
            forward_state = self.model(batch, encoded_inputs=encoded_inputs)
            loss = self._apply_loss_functions(forward_state, batch)
            total_loss += loss.item()
            
            status = self._valid_template.format(
                self.epoch, step, len(batches), total_loss / step)
            print(status, end="\r" if step < len(batches) else "\n",
                  flush=True)

        avg_loss = total_loss / len(batches)
        results = self._epoch_results(avg_loss, "validation")
        self._display_metrics("validation")
        self._close_output_checkpoints(output_pointers)
        return results
       
    def run(self):
       
        with self.results_path.open("w") as results_fp:
            for _ in range(self.max_epochs):

                self._epoch += 1
                train_result = self._train_epoch(self.training_minibatches)
                valid_result = self._valid_epoch(self.validation_minibatches)
                print()
                
                epoch_result = {
                    "epoch": self.epoch,
                    "training": train_result, 
                    "validation": valid_result,
                }

                self.lr_scheduler.step(epoch_result)

                #print(json.dumps(epoch_result), file=results_fp, flush=True)
                #self._checkpoint_model(epoch_result)
