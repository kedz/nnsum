import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry
from nnsum2.layers import (ConvSeq1D, ConvMaxPoolingSeq1D, StandardizerSeq1D,
                           MaxPoolingSeq1D)

@register_module("seq2clf.fg_teacher_model_v2")
class FGTeacherModelV2(Module):

    hparams = hparam_registry()

    @hparams(type="submodule")
    def input_embedding_context(self):
        pass

    @hparams()
    def label_vocab(self):
        pass

    @hparams()
    def input_dims(self):
        pass

    @hparams(default=5)
    def gate_filter_width(self):
        pass

    @hparams(default=5)
    def classifier_filter_width(self):
        pass

    @hparams(default=0.0)
    def dropout(self):
        pass

    def init_network(self):

        self._gating_layer1 = ConvSeq1D(
            kernel_width=self.gate_filter_width,
            input_features=self.input_dims,
            output_features=self.input_dims,
            padding=self.gate_filter_width // 2,
            mask_mode="unsafe")
        self._gating_layer2 = nn.Linear(self.input_dims, 1)

        self._label_layer1 = ConvMaxPoolingSeq1D(
            kernel_width=self.classifier_filter_width,
            input_features=self.input_dims,
            output_features=self.input_dims,
            padding=self.classifier_filter_width // 2,
            mask_mode="unsafe",
            activation="ReLU",
            dropout=self.dropout,
        )
        self._label_layer2 = nn.Linear(self.input_dims, len(self.label_vocab))


        if "N/A" in self.label_vocab:
            self._na_index = self.label_vocab["N/A"]
            self._na_layer_1 = StandardizerSeq1D()
            self._na_layer_2 = MaxPoolingSeq1D(squeeze_singleton=True)
            self._na_layer_3 = nn.Linear(1, 1)
        else:
            self._na_index = None

#        self._lbl_layer1 = nn.Conv2d(
#            1, self.input_dims, 
#            (self.classifier_filter_width, self.input_dims),
#            padding=(self.classifier_filter_width // 2, 0))
#        self._lbl_layer2 = nn.Linear(self.input_dims, len(self.label_vocab))

    def gate_network(self, embeddings, inputs_mask):
        hidden, gates_mask = self._gating_layer1(
            embeddings, inputs_mask=inputs_mask)
        hidden = F.dropout(
            hidden, p=self.dropout, training=self.training, inplace=True)
        hidden = torch.relu(hidden)
        gate_logits = self._gating_layer2(hidden)
        gates = torch.sigmoid(gate_logits)
        gates = gates.masked_fill(gates_mask.unsqueeze(-1), 0.)

        gated_emb = gates * embeddings
        return gated_emb, gates

    def label_network(self, embeddings, inputs_mask):
        hidden_state, _ = self._label_layer1(
            embeddings, inputs_mask=inputs_mask)
        label_logits = self._label_layer2(hidden_state)

        if self._na_index is not None:
            na_mask = label_logits.new().byte().new(label_logits.size())
            na_mask.zero_()
            na_mask[:, self._na_index].fill_(1)
            label_logits = label_logits.masked_fill(na_mask, float("-inf"))
        else:
            na_mask = None

        label_probs = torch.softmax(label_logits, dim=1)
        return label_probs, na_mask

    def na_network(self, gates, inputs_mask):
#        print(gates.size())
#        print(gates.squeeze(-1).unfold(1,4,1).size())
        na_logits = self._na_layer_3(
            -1 * gates.squeeze(-1).unfold(1,4,1).sum(2).max(1)[0].unsqueeze(-1))
#        input()
#        
#        
#        gates_standardized, _ = self._na_layer_1(gates, inputs_mask)
#        max_gates, _ = self._na_layer_2(gates_standardized, inputs_mask)
#        na_logits = self._na_layer_3(max_gates)
        return torch.sigmoid(na_logits)

        print(max_gates)
        input()
        
        
        mask = gates.squeeze(2).eq(0)



        a = (gates).squeeze(2)
#        print(mask.size())
#        print(a.size())
        anorm = (a - a.mean(1, keepdim=True)) / a.std(1, keepdim=True)
#        print(anorm.size())
        anorm = anorm.masked_fill(mask, 0)
        na_logits = self._na_linear(anorm.topk(10, dim=1)[0].mean(1, keepdim=True))
        #print(anorm.topk(10, dim=1)[0].size())
        #print(anorm.topk(10, dim=1)[0].sum(1))
#        input()
#        print(anorm)
#        print(anorm.size())
#        print(anorm.topk(5, dim=1)[0].size())
#        input()
#        na_logits = self._na_layer(1 - gates.unsqueeze(1)).squeeze(1).mean(1)
        na_probs = torch.sigmoid(na_logits)
        return na_probs

    def forward(self, inputs):
        
        emb = self.input_embedding_context(inputs["source_input_features"])
        if self.input_embedding_context.transpose:
            import warnings
            warnings.warn("input_embedding_context is needlessly transposing.")
            emb = emb.permute(1, 0, 2).contiguous()

        gated_embeddings, gate_probs = self.gate_network(
            emb, inputs["source_mask"])
        
        label_value_probs, na_mask = self.label_network(
            gated_embeddings, inputs["source_mask"])

        if self._na_index is not None:
            na_probs = self.na_network(gate_probs, inputs["source_mask"])
            label_probs = (1 - na_probs) * label_value_probs

            target_probs = label_probs.masked_scatter(na_mask, na_probs)
        else:
            target_probs = label_value_probs

        target_log_probs = torch.log(target_probs)    

        return {"target_log_probability": target_log_probs,
                "gates": gate_probs.squeeze(2)}

    def predict(self, inputs):
        return self.forward(inputs)["target_log_probability"].argmax(1)

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            print(name)
            if name == "_gating_layer2.bias":
                nn.init.constant_(param, -3.)    
                print(param)
            elif name == "_na_layer.bias":
                nn.init.constant_(param, -2.)    
                print(param)
            elif "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)    
            else:
                nn.init.normal_(param)    
