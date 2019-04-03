import torch
import numpy as np
from ..parameterized import Parameterized
from ..hparam_registry import HParams
from nnsum.seq2seq.search_state import SearchState


@Parameterized.register_object("seq2seq.fg_teacher_reranker")
class FGTeacherReranker(Parameterized):

    hparams = HParams()

    @hparams()
    def teacher_model(self):
        pass

    @hparams(default=-1)
    def device(self):
        pass

    def init_object(self):
        self.teacher_model.eval()
        #if self.device > -1:
        #    for name in self.teacher_models.keys():
        #        model = self.teacher_models[name]
        #        self.teacher_models[name] = model.cuda(self.device)
        #for model in self.teacher_models.values():
        #    model.eval()

    def __call__(self, batch, search_state):
        
        output = search_state.get_result("output")
        seq_size, batch_size, beam_size = output.size()
        output = output.permute(1, 2, 0).contiguous()
        output_flat = output.view(batch_size * beam_size, seq_size)
        
        teacher_input = {"source_input_features": {"tokens": output_flat}}

        all_true_labels = []
        all_pred_labels = []
        for field, teacher in self.teacher_model.models.items():
            true_labels = batch["source_labels"][field]
            all_true_labels.append(true_labels.view(-1, 1))

            pred_labels = teacher.predict(teacher_input)
            all_pred_labels.append(pred_labels.view(-1, 1))

        all_true_labels = torch.cat(all_true_labels, 1).view(batch_size, 1, -1)
        all_pred_labels = torch.cat(all_pred_labels, 1).view(
            batch_size, beam_size, -1)

        errors = all_true_labels.ne(all_pred_labels)
        avg_errors = errors.float().mean(2)

        # Using numpy's merge sort because torch's default sort is not stable.
        # We want a stable sort because the outputs are already ordered by
        # the scoring used in beam search, and when deciding between two
        # candidates with the same classification error, we should prefer
        # the one with the highest average token log likelihood.
        I = np.argsort(avg_errors.cpu().numpy(), 1, kind='mergesort')
        sorted_output = torch.stack(
            [o.squeeze(0)[i] for o, i in zip(output.split(1,0), I)])

        return SearchState(output=sorted_output.permute(2, 0, 1))
