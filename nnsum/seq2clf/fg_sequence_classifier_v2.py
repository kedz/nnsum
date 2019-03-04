import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSequenceClassifierV2(nn.Module):
    def __init__(self, src_ec, tgt_ec):
        super(FGSequenceClassifierV2, self).__init__()
        self.source_ec = src_ec
        self.target_ec = tgt_ec
        self._na_layer = nn.Linear(3, 1)
        self._lbl_layer1 = nn.Conv2d(1, 512, (5, src_ec.output_size),
                                    padding=(2,0))
        self._lbl_layer2 = nn.Linear(512, len(tgt_ec.vocab) - 1)

        self._gate_layer1 = nn.Conv2d(1, 512, (5, src_ec.output_size),
                                    padding=(2,0))
        self._gate_layer2 = nn.Linear(512, 1)


    def _na_prob_network(self, gates):
        top_gates = gates.squeeze(2).topk(dim=1, k=3)[0]
        na_logits = self._na_layer(1 - top_gates)
        na_probs = torch.sigmoid(na_logits)
        return na_probs
    
    def _label_prob_network(self, embs):
        embs = embs.unsqueeze(1)
        preact = self._lbl_layer1(embs).squeeze(3).max(2)[0]
        act = F.dropout(torch.relu(preact), p=.25, training=self.training,
                        inplace=True)
        lbl_logit = self._lbl_layer2(act)
        lbl_prob = torch.softmax(lbl_logit, dim=1)
        return lbl_prob

    def _gate_network(self, embs):
        embs = embs.unsqueeze(1)
        preact = self._gate_layer1(embs).squeeze(3).permute(0, 2, 1)
        act = F.dropout(torch.relu(preact), p=.25, training=self.training,
                        inplace=True)
        gate_logit = self._gate_layer2(act)
        #print(gate_logit)
        gates = torch.sigmoid(gate_logit)
        #print(gates)
        #input()
        return gates

    def forward(self, inputs):
        emb = self.source_ec(inputs["source_input_features"])
        gate_probs = self._gate_network(emb)
        na_prob = self._na_prob_network(gate_probs)
        
        gated_embs = gate_probs * emb
        label_probs = self._label_prob_network(gated_embs)

        probs = torch.cat([na_prob, (1 - na_prob) * label_probs], dim=1)
        log_probs = torch.log(probs)
        return {"target_log_probability": log_probs, 
                "encoder_state": {"gates": gate_probs.squeeze(2)}}

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
