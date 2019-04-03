import torch
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("loss_functions.fg_meta_gate_div")
class FGMetaGateDiv(Module):

    hparams = hparam_registry()

    def init_network(self):
        self.reset()
 
    def reset(self):
        self._total_loss = 0
        self._total_inputs = 0

    def mean(self):
        if self._total_inputs > 0:
            return self._total_loss / self._total_inputs
        else:
            raise RuntimeError("Must have processed at least one batch.")

    def _entropy(self, P):
        return - (torch.log(P) * P).sum(1)

    @hparams()
    def label_vocabs(self):
        pass

    def forward(self, forward_states, batch):
        eps = 1e-8
        #print()
        all_dists = []
        all_entropy = []
        all_ignore = []
        for field, fs in forward_states.items():
        #    print(field)
            gates = (fs["gates"] + eps) 
            gate_dist = (gates / gates.sum(1, keepdim=True))
            H = self._entropy(gate_dist)
            
            if "N/A" in self.label_vocabs[field]:
                na_idx = self.label_vocabs[field]["N/A"]
                ignore = batch["target_labels"][field].eq(na_idx)
            else:
                ignore = batch["target_labels"][field].clone().fill_(0).byte()
            gate_dist = gate_dist.masked_fill(ignore.view(-1, 1), 0.)

            #weight = 1 / (~ignore).float().sum()
            H = H.masked_fill(ignore, 0.)
            #print(ignore)
            #print(weight)
            #print(H)
            all_entropy.append(H.view(-1, 1))
            all_dists.append(gate_dist.unsqueeze(1))
            all_ignore.append(ignore.view(-1, 1))
        #print()
        all_ignore = torch.cat(all_ignore, dim=1)
        #print(all_ignore)
        weights = (~all_ignore).float().sum(1, keepdim=True)
        #print(weights)
        #print(torch.cat(all_entropy, dim=1))
        #print(torch.cat(all_dists, dim=1).size())
        #print(torch.cat(all_dists, dim=1)[:,:,:4])
        avg_dist = (torch.cat(all_dists, dim=1).sum(1) / weights)
        avg_dist_H = self._entropy(avg_dist)
        #print(avg_dist_H)
        avg_H = (sum(all_entropy) / weights).view(-1)
        #print(avg_H)
        total_loss = -(avg_dist_H - avg_H).sum()
        self._total_loss += total_loss.item()
        self._total_inputs += avg_H.size(0)
        
        return total_loss / avg_H.size(0) 
