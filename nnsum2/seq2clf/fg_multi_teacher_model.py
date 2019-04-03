import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry
import numpy as np


@register_module("seq2clf.fg_multi_teacher_model")
class FGMultiTeacherModel(Module):

    hparams = hparam_registry()

    @hparams()
    def models(self):
        pass

    def init_network(self):
        self._inputs_vocab = [model.input_embedding_context.vocab
                              for model in self.models.values()][0]

    def train(self):
        for model in self.models.values():
            model.train()

    def eval(self):
        for model in self.models.values():
            model.eval()
        return self

    def cuda(self, device):
        for name in self.models.keys():
            self.models[name] = self.models[name].cuda(device)
        return self

    def parameters(self):
        for model in self.models.values():
            for param in model.parameters():
                yield param

    def forward(self, inputs):
        return {name: model(inputs) for name, model in self.models.items()}

    def predict(self, inputs, forward_states=None):
        if forward_states is None:
            forward_states = self(inputs)
        return {field: state["target_log_probability"].argmax(1)
                for field, state in forward_states.items()}

    def initialize_parameters(self):
        for model in self.models.values():
            model.initialize_parameters()


    def _expand_labels(self, labels, beam_size):
        return {field: label.view(-1,1).repeat(1, beam_size).view(-1)
                for field, label in labels.items()}

    def rerank_beam(self, beam_outputs, labels):

        steps, batches, beams = beam_outputs.size()
        beam_labels = self._expand_labels(labels, beams)
        beam_outputs_flat = beam_outputs.permute(1, 2, 0).contiguous().view(
            batches * beams, steps)

        teacher_inputs = {
            "source_input_features": {
                "tokens": beam_outputs_flat,
            },
            "source_mask": beam_outputs_flat.eq(self._inputs_vocab.pad_index)
        }
        
        predictions = self.predict(teacher_inputs)
        errors = []
        for field, prediction in predictions.items():    
            errors.append(
                (prediction != beam_labels[field]).long())
        total_errors = sum(errors).view(batches, beams)
        # Using numpy's stable sort.
        argsorts = np.argsort(total_errors.data.numpy(), 1)

        sorted_errors = []
        sorted_beams = []
        for beam, error, argsort in zip(
                beam_outputs.split(1, 1), total_errors.split(1, 0), argsorts):
            sorted_beams.append(beam[:,:,argsort])
            sorted_errors.append(error[:,argsort])
        sorted_errors = torch.cat(sorted_errors, 0)
        sorted_beams = torch.cat(sorted_beams, 1)

        return sorted_errors, sorted_beams

   
    def local_errors(self, inputs, true_labels):
        inputs_mask = inputs.eq(0)

        #print(inputs)
        forward_states = self.forward({
            "source_input_features": {"tokens": inputs},
            "source_mask": inputs_mask,
        })

        losses = inputs.new().float().new(inputs.size()).fill_(0)

        stop_mask = inputs.eq(3)
        na_errors = stop_mask.new(inputs.size(0))
        #print(inputs)

        predictions = self.predict(None, forward_states)
        
       
        
        for field, pred in predictions.items():
            errors = true_labels[field] != pred
            if torch.any(errors):
                if field != "name":
                    na_errors = na_errors | (true_labels[field].ne(0) & pred.eq(0))
                    non_na_errors = errors & pred.ne(0)
                else:
                    non_na_errors = errors 
                #print(non_na_errors)
                if torch.any(non_na_errors):


                
                    #print(field)
                    #print(forward_states[field]["gates"])
                    #print(non_na_errors)

                    #for i, row in enumerate(inputs):
                        #if not non_na_errors[i]:
                        #    continue
                        #print(self.models[field].label_vocab[true_labels[field][i].item()])
                        #print(self.models[field].label_vocab[pred[i].item()])
                        #print(" ".join([self._inputs_vocab[idx.item()] for idx in row]))

                    losses.data.add_(
                        forward_states[field]["gates"].masked_fill(
                            ~non_na_errors.view(-1, 1), 0))
                    #input()
                
            else:
                losses.data.add_(
                    -forward_states[field]["gates"].masked_fill(
                        errors.view(-1, 1), 0))
                
                #print(forward_states[field]["gates"])
                #print("true", true_labels[field])
                #print("pred", pred)
                #print(true_labels[field].ne(0) & pred.eq(0))
                #print(na_errors)
                #losses.data.masked_fill_(
                #    stop_mask.masked_fill(errors, 0), -1)
                #print(losses)

                #input()
        #print("losses")
        #print(na_errors)
        #print(na_errors.view(-1, 1) & stop_mask)
        losses.data.masked_fill_(
            na_errors.view(-1, 1) & stop_mask, 1.0).clamp_(0, 1.0)
        #print(losses)
        #input()
        return losses
