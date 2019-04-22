import torch
from ..parameterized import Parameterized
from ..hparam_registry import HParams
import numpy as np


@Parameterized.register_object("example_loggers.gw_fg_example_logger")
class GWFGExampleLogger(Parameterized):
    
    hparams = HParams()

    @hparams()
    def source_vocab(self):
        pass

    @hparams()
    def source_field(self):
        pass

    @hparams()
    def target_vocab(self):
        pass

    @hparams()
    def target_field(self):
        pass

    def init_object(self):
        pass

    def _format_target_attention(self, tokens, attention):

        tmp = ["{:" + str(len(t)) + "}" for t in tokens]
        grid = [[" " * len(t) for t in tokens] for i in range(11)]

        for i, a in enumerate(attention):
            level = int(np.round(a / .1 ))
            for l in range(level):
                grid[11 - l - 1][i] = tmp[i].format("*" * len(tokens[i]))
        
        text = "\n".join([" ".join(g) for g in grid] + [" ".join(tokens)])
        return text

    def __call__(self, batch, forward_state, fp):
 
        bs = batch["prediction_targets"].size(0)
        probs = torch.softmax(forward_state["target_logits"], dim=1)
        pred_probs, pred_indices = torch.sort(probs, 1, descending=True)

        for b in range(bs):
            true_token = self.source_vocab[batch["prediction_targets"][b]]
            target_tokens = [
                self.target_vocab[idx]
                for idx in batch["target_features"][self.target_field][b]
                if idx != self.target_vocab.pad_index
            ]
            attn = forward_state["attention"][0,b,:len(target_tokens)].tolist()
            target_text = self._format_target_attention(target_tokens, attn)

            source_tokens = [
                self.source_vocab[idx]
                for idx in batch["source_features"][self.source_field][b]
                if idx != self.source_vocab.pad_index
            ]

            print("\n===\n", file=fp)

            print("SUMMARY", file=fp)
            print(target_text, file=fp)
            print(file=fp)
            print("INPUT", file=fp)
            print(" ".join(source_tokens), file=fp)
            print(file=fp)
            print("TARGET:  ", true_token, file=fp)

            pred_tokens = [self.source_vocab[idx]
                           for idx in pred_indices[b,:10]]
            pred_tokens_probs = ["{:0.3f}".format(p)
                                 for p in pred_probs[b,:10]]
            print(
                "  PRED:", 
                "  ".join(["{}|{}".format(t, p) 
                           for t, p in zip(pred_tokens, pred_tokens_probs)]),
                file=fp)
            print(file=fp)
