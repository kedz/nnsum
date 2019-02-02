import ipywidgets as widgets

import matplotlib
import matplotlib.pyplot as plt

from nnsum.datasets import CopyDataset
from nnsum.data.seq2seq_batcher import (
        batch_source, batch_target, batch_copy_alignments,
        batch_pointer_data)


class ExamplePredictionWidget(object):
    def __init__(self, run_data):
        self._run_data = run_data

        print(run_data.keys())
        self._datasets = self.make_datasets(run_data["dataset_params"])    
        self._model = run_data["model"]

        self._w_part = widgets.Dropdown(options=self._datasets.keys(),
                                        value="test",
                                        description="Part:") 
        self._w_example = widgets.IntSlider(
            min=1, value=1, max=len(self._datasets["test"]),
            description="Example:")
        self._w_source = widgets.Label()
        self._w_expected_target = widgets.Label()
        self._w_expected_target_dec_vcb = widgets.Label()
        self._w_predicted_target = widgets.Label()
        self._w_alignment = widgets.Output()

        def on_value_change(change):
            if change["new"] is not None:
                self._show_example()
        self._w_example.observe(on_value_change, names=["value"])

        self._m_main = widgets.VBox([
            widgets.HBox([self._w_part, self._w_example]), 
            widgets.Label("Source"),
            self._w_source,
            widgets.Label("Expected Target"),
            self._w_expected_target,
            widgets.Label("Expected Target (Decoder Vocab)"),
            self._w_expected_target_dec_vcb,
            widgets.Label("Predicted Target"),
            self._w_predicted_target,
            self._w_alignment,
        ])

        with self._w_alignment:
            matplotlib.use('nbAgg')
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(1,1,1)
            plt.show(self._fig)

    def make_datasets(self, dataset_params):
        train_dataset = CopyDataset(dataset_params["vocab_size"],
                                    dataset_params["min_length"],
                                    dataset_params["max_length"],
                                    dataset_params["test_size"],
                                    random_seed=dataset_params["train_seed"])
        test_dataset = CopyDataset(dataset_params["vocab_size"],
                                   dataset_params["min_length"],
                                   dataset_params["max_length"],
                                   dataset_params["test_size"],
                                   random_seed=dataset_params["test_seed"])
        return {"train": train_dataset, "test": test_dataset}

    @property
    def example(self):
        index = self._w_example.value - 1
        part = self._w_part.value
        return self._datasets[part][index]

    def make_batch_example(self, example, src_vocabs, tgt_vocabs):
        batch = batch_source([example["source"]], src_vocabs)
        batch.update(batch_target([example["target"]], tgt_vocabs))
        batch.update(batch_pointer_data([example["source"]], tgt_vocabs,
                                        targets=[example["target"]]))
        return batch




    def convert_prediction(self, output, tgt_vcb, ext_vcb):
        result = []
        offset = len(tgt_vcb)
        for idx in output.tolist():
            if idx >= offset:
                result.append(ext_vcb[idx - offset])
            else:
                result.append(tgt_vcb[idx])
        return result

    def _show_example(self):
        src_vocabs = self._model.encoder.embedding_context.named_vocabs
        tgt_ec = self._model.decoder.embedding_context
        tgt_vocabs = tgt_ec.named_vocabs

        self._w_source.value = " ".join(self.example["source"]["tokens"])
        self._w_expected_target.value = " ".join(
            self.example["target"]["tokens"])

        tgt_vcb = tgt_vocabs["tokens"]
        self._w_expected_target_dec_vcb.value = " ".join(
            [tgt_vcb[tgt_vcb[tok]]
             for tok in self.example["target"]["tokens"]])
        batch = self.make_batch_example(self.example, src_vocabs, tgt_vocabs)
        greedy_search = self._model.greedy_decode(batch, max_steps=1000)
        pred_tokens = self.convert_prediction(
            greedy_search.get_result("output").t()[0], tgt_vcb,
            batch.get("extended_vocab", None))

        



#        pred_tokens = tgt_ec.convert_index_tensor(
#            greedy_search.get_result("output").t()[0])
        self._w_predicted_target.value = " ".join(pred_tokens[:-1])

        ctx_attn = greedy_search.get_result("context_attention")[:,0].t()
        with self._w_alignment:
            self._ax.matshow(ctx_attn.data.numpy())
            self._ax.set_xticklabels([''] + pred_tokens)
            self._ax.set_yticklabels(
                ['', "<START>"] + self.example["target"]["tokens"])
            



#        self._widgets["expected_target"].value = " ".join(
#                ex["target"]["tokens"])


#        ex = 

    def __call__(self):
        return self._m_main
