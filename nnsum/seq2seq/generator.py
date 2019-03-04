from nnsum.data.seq2seq_batcher import batch_source, batch_pointer_data
from sacremoses import MosesDetokenizer


class ConditionalGenerator(object):
    def __init__(self, model, max_steps=1000, replace_unknown=True,
                 detokenize=False):
        self._model = model
        self._max_steps = max_steps
        self._replace_unknown = replace_unknown
        self._detok = MosesDetokenizer() if detokenize else None
        self._detokenize = detokenize

    @property
    def replace_unknown(self):
        return self._replace_unknown
 
    def _clean_outputs(self, outputs, tgt_vocab, ext_vocab=None,
                       attention=None, source_tokens=None):

        tokens = []

        for step, idx in enumerate(outputs.tolist()):
            if idx == tgt_vocab.stop_index:
                break
            if idx != tgt_vocab.unknown_index:
                if idx < len(tgt_vocab):
                    tokens.append(tgt_vocab[idx])
                else:
                    tokens.append(ext_vocab[idx - len(tgt_vocab)])
            else:
                if self.replace_unknown and attention is not None:
                    tokens.append(source_tokens[attention[step].max(0)[1]])
                else:
                    tokens.append(tgt_vocab.unknown_token)
        return tokens

    def generate(self, conditioning, return_state=False):
        batch = batch_source(
            [conditioning], self._model.encoder.embedding_context.named_vocabs)
        batch.update(
            batch_pointer_data(
                [conditioning], 
                self._model.decoder.embedding_context.named_vocabs))

        self._model.eval()
        search = self._model.greedy_decode(batch, max_steps=self._max_steps)

        tokens = self._clean_outputs(
            search.get_result("output").t()[0],
            self._model.decoder.embedding_context.vocab,
            ext_vocab=batch.get("extended_vocab", None),
            attention=search.get_result("context_attention")[:,0,1:],
            source_tokens=conditioning["tokens"])

        if self._detokenize:
            out_str = self._apply_detokenize(tokens)
        else:
            out_str = " ".join(tokens)

        if return_state:
            return out_str, search
        else:
            return out_str

    def generate_from_batch(self, batch):
        self._model.eval()
        search = self._model.greedy_decode(batch, max_steps=self._max_steps)

        tokens_name = self._model.decoder.embedding_context.name

        all_tokens = []
        outputs = search.get_result("output").cpu().t()
        attn = search.get_result("context_attention")
        attn = attn.cpu() if attn is not None else attn

        for i, output in enumerate(outputs):
            all_tokens.append(
                self._clean_outputs(
                    output,
                    self._model.decoder.embedding_context.vocab,
            ext_vocab=batch.get("extended_vocab", None),
            attention=attn[:,i,1:] if attn is not None else None,
            source_tokens=batch["source_tokens"][i]))

        if self._detokenize:
            return [self._apply_detokenize(tokens) for tokens in all_tokens]
        return all_tokens

    def _apply_detokenize(self, tokens):
        string = self._detok.detokenize(tokens).replace("<SENT>", "")\
            .replace(" - ", "-")
        return string
