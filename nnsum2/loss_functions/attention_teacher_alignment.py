import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module import Module, register_module, hparam_registry


@register_module("loss_functions.attention_teacher_alignment")
class AttentionTeacherAlignment(Module):

    hparams = hparam_registry()

    @hparams(default="context_attention")
    def attention_field(self):
        pass

    @hparams()
    def teacher(self):
        pass

    @hparams()
    def mrs_vocab(self):
        pass

    def init_network(self):
        self.reset()
        self.teacher.eval()

        self._map = {}
        for i, mr in self.mrs_vocab.enumerate():
            if "_" in mr:
                field = mr.split("_", 1)[0]
                if field == "NAME":
                    self._map[i] = "name"
                elif field == "NEAR":
                    self._map[i] = "near"
                elif field == "AREA":
                    self._map[i] = "area"
                elif field == "FAMILYFRIENDLY":
                    self._map[i] = "family_friendly"
                elif field == "FOOD":
                    self._map[i] = "food"
                elif field == "PRICERANGE":
                    self._map[i] = "price_range"
                elif field == "CUSTOMERRATING":
                    self._map[i] = "customer_rating"
                elif field == "EATTYPE":
                    self._map[i] = "eat_type"
                else:
                    raise Exception("Bad mr:", mr)
 
    def reset(self):
        self._total_loss = 0
        self._total_inputs = 0

    def mean(self):
        if self._total_inputs > 0:
            return self._total_loss / self._total_inputs
        else:
            raise RuntimeError("Must have processed at least one batch.")

    def forward(self, forward_state, batch):
        
        if "max_references" in batch:
            self._total_inputs += 1
            return 0 
        attention = forward_state[self.attention_field]
        tgt_size = attention.size(0)
        teacher_inputs = {
            'source_input_features':  batch["target_output_features"]
        }
        teacher_forward_states = self.teacher(teacher_inputs)


        mrs = batch["source_input_features"]["mrs"]
        batch_size, src_seq_size = mrs.size()
        mrs = mrs.data.tolist()

        unnormalized_reference = []
        for batch in range(batch_size):
            refs = []
            for step in range(src_seq_size):
                field = self._map.get(mrs[batch][step], None)
                if field is None:
                    refs.append(attention.new(tgt_size, 1, 1).fill_(0))
                else:
                    gates = teacher_forward_states[field]["gates"][batch]
                    refs.append(gates.view(-1, 1, 1))
            refs = torch.cat(refs, 2)
            unnormalized_reference.append(refs)
        
        unnormalized_reference = torch.cat(unnormalized_reference, 1)
        normalizer = unnormalized_reference.sum(2, keepdim=True)
        normalizer = normalizer.masked_fill(normalizer.eq(0), 1)
        reference_attention = unnormalized_reference / normalizer
        avg_loss = ((reference_attention - attention) ** 2).mean()
        self._total_inputs += attention.numel()
        self._total_loss += (avg_loss.item() * attention.numel())
        
        return avg_loss 

