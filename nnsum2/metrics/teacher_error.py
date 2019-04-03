from ..parameterized import Parameterized
from ..hparam_registry import HParams

from nnsum.seq2seq.search_state import SearchState


@Parameterized.register_object("metrics.teacher_error")
class TeacherError(Parameterized):

    hparams = HParams()

    @hparams()
    def teacher(self):
        pass

    def reset(self):
        self._errors = {name: 0 for name in self.teacher.models.keys()}
        self._total = 0

    def init_object(self):
        self.reset()
        self.teacher.eval()

    def __call__(self, batch, search_state):

        if isinstance(search_state, SearchState):
            output = search_state["output"]
        else:
            output = search_state.get_result("output")

        if output.dim() == 3:
            output = output[:,:,0]

        output = output.t()
        mask = output.eq(0)
        max_length = (~mask).long().sum(1).max()
        if output.size(1) > max_length:
            output = output[:,:max_length]
            mask = mask[:,:max_length]
        self._total += output.size(0)

        #print(output.t())
        teacher_input = {"source_input_features": {"tokens": output},
                         "source_mask": mask}
        preds = self.teacher.predict(teacher_input)
        for field, pred_labels in preds.items():
            true_labels = batch["source_labels"][field]
            errors = (true_labels != pred_labels).long().sum().item()
            self._errors[field] += errors
#            print(field)
#            for row in output.t().tolist():
#                print(" ".join(
#                    [model.input_embedding_context.vocab[idx] for idx in row
#                     if idx != model.input_embedding_context.vocab.pad_index]))
            #print([model.label_vocab[idx] for idx in pred_state.tolist()])

            #print(pred_state)
            #print(batch["source_labels"])

        #input()

        #exit()



    def compute(self):
        if self._total == 0:
            raise RuntimeError("Must have processed at least one batch.")
        return {f: e / self._total for f, e in self._errors.items()}

    def pretty_print(self):
        errors = self.compute()
        error_strings = ["Teacher Error Rates ::"]
        for field, error_rate in self.compute().items():
            error_strings.append("{}={:4.3f}".format(field, error_rate))
        print("  ".join(error_strings))
