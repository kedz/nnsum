import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("seq2clf.fg_teacher_model")
class FGTeacherModel(Module):

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

        if "N/A" in self.label_vocab:
            self._na_index = self.label_vocab["N/A"]
            self._na_layer = nn.Conv2d(1, 1, (self.gate_filter_width, 1))
        else:
            self._na_index = None

        self._lbl_layer1 = nn.Conv2d(
            1, self.input_dims, 
            (self.classifier_filter_width, self.input_dims),
            padding=(self.classifier_filter_width // 2, 0))
        self._lbl_layer2 = nn.Linear(self.input_dims, len(self.label_vocab))

        self._gate_layer1 = nn.Conv2d(1, self.input_dims, 
            (self.gate_filter_width, self.input_dims),
            padding=(self.gate_filter_width // 2, 0))
        self._gate_layer2 = nn.Linear(self.input_dims, 1)

    def gate_network(self, embeddings):
        embeddings = embeddings.unsqueeze(1)

        preact = self._gate_layer1(embeddings).squeeze(3).permute(0, 2, 1)
        act = torch.relu(preact)
        act_dro = F.dropout(act, p=self.dropout, training=self.training,
                            inplace=True)
        gate_logit = self._gate_layer2(act_dro)
        gate_probs = torch.sigmoid(gate_logit)
        return gate_probs

    def label_network(self, embeddings):
        embeddings = embeddings.unsqueeze(1)
        preact = self._lbl_layer1(embeddings).squeeze(3)
        max_pooling = preact.max(2)[0]
        act = torch.relu(max_pooling)
        act_dro = F.dropout(act, p=self.dropout, training=self.training,
                            inplace=True)
        label_logits = self._lbl_layer2(act_dro)

        if self._na_index is not None:
            na_mask = label_logits.new().byte().new(label_logits.size())
            na_mask.zero_()
            na_mask[:, self._na_index].fill_(1)
            label_logits = label_logits.masked_fill(na_mask, float("-inf"))
        else:
            na_mask = None

        label_probs = torch.softmax(label_logits, dim=1)
        return label_probs, na_mask

    def na_network(self, gates):
        na_logits = self._na_layer(1 - gates.unsqueeze(1)).squeeze(1).mean(1)
        na_probs = torch.sigmoid(na_logits)
        return na_probs

    def forward(self, inputs):
        
        for name, vocab in self.input_embedding_context.named_vocabs.items():
            gate_mask = inputs["source_input_features"][name].eq(
                vocab.pad_index).unsqueeze(2)
            break

        emb = self.input_embedding_context(inputs["source_input_features"])
        
        if self.input_embedding_context.transpose:
            import warnings
            warnings.warn("input_embedding_context is needlessly transposing.")
            emb = emb.permute(1, 0, 2).contiguous()

        gate_probs = self.gate_network(emb).masked_fill(gate_mask, 0)

        gated_emb = gate_probs * emb

        if self._na_index is not None:
            na_probs = self.na_network(gate_probs)
            label_value_probs, na_mask = self.label_network(gated_emb)
            label_probs = (1 - na_probs) * label_value_probs

            target_probs = label_probs.masked_scatter(na_mask, na_probs)
        else:
            target_probs, na_mask = self.label_network(gated_emb)

        target_log_probs = torch.log(target_probs)    

        return {"target_log_probability": target_log_probs,
                "gates": gate_probs.squeeze(2)}

    def predict(self, inputs):
        return self.forward(inputs)["target_log_probability"].argmax(1)

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            print(name)
            if name == "_gate_layer2.bias":
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
