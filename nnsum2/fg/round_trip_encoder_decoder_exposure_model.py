from ..module import Module, register_module, hparam_registry
from nnsum2.seq2seq.searches import GreedySearch
from nnsum2.attention import BiLinearKernel
import torch


@register_module("fg.round_trip_encoder_decoder_exposure_model")
class RoundTripEncoderDecoderExposureModel(Module):

    hparams = hparam_registry()

    @hparams()
    def round_trip_model(self):
        pass

    @hparams(default=1000)
    def max_steps(self):
        pass

    @property
    def encoder(self):
        return self.round_trip_model.encoder

    @property
    def decoder(self):
        return self.round_trip_model.decoder 

    def init_network(self):
        self._hidden_state_kernel = BiLinearKernel()
        from warnings import warn
        warn("Setting dropout to 0!")
        self.set_dropout(0)

    def _compute_alignment_error(self, ref_states, ref_mask, 
                                 search_states, search_mask):

        # ref_states (ref length x batch states x hidden size)
        # search_states (search length x batch states x hidden size)
        
        # Removed ref_states from computation graph. We are only going to 
        # backpropagate through the search states. 
        ref_states = ref_states.detach()

        # attention (batch x query length x hidden size) 
        attention, _ = self._hidden_state_kernel(
            search_states.permute(1, 0, 2), ref_states, key_mask=search_mask)
        attention = attention.detach()

        # aligned_search_states (batch size x ref length x hidden size)
        aligned_search_states = torch.bmm(
            attention, search_states.permute(1, 0, 2))

        # reference state-wise squared error
        el_se = ((ref_states.permute(1,0,2) - aligned_search_states) ** 2) \
            .sum(dim=2)
        
        el_se = el_se.masked_fill(ref_mask, 0)

        # numer is total squared error
        # denom is number of non masked reference steps * hidden size
        return el_se.sum() / ((~ref_mask).float().sum() * ref_states.size(2))

    def encode(self, inputs):
        encoder_output, encoder_state = self.round_trip_model.encoder(
            inputs["source_input_features"], 
            inputs["source_lengths"])

        context = {"encoder_output": encoder_output,
                   "source_mask": inputs.get("source_mask", None),
                   "source_extended_vocab_map": inputs.get(
                       "source_extended_vocab_map", None),
                   "extended_vocab": inputs.get("extended_vocab", None),
                   "controls": inputs.get("controls", None)}
        return context, encoder_state

    def forward(self, inputs, encoded_inputs=None):

        model = self.round_trip_model

        if encoded_inputs is None:
            encoded_inputs = self.encode(inputs)

        context, encoder_state = encoded_inputs

        vocab = model.decoder.output_embedding_context.vocab

        search_state = GreedySearch(
            model.decoder, 
            {"encoder_state": encoder_state}, context, 
            max_steps=self.max_steps, return_incomplete=True)

        decoder_state = model.decoder(
            encoder_state, inputs["target_input_features"], context, None)

        search_mask = search_state.get_result("output").eq(vocab.pad_index)
        alignment_error = self._compute_alignment_error(
            decoder_state["rnn_output"],
            inputs["target_mask"],
            search_state.get_result("rnn_output"),
            search_mask)

        decoder_state["search_alignment_error"] = (alignment_error, ())

        embeddings = model.encoder.embedding_context.embeddings.weight
        recon_logits = model.reconstructor(
			context["encoder_output"],
            inputs["cloze_indices"],
            decoder_state["rnn_output"], 
            embeddings,
            targets_mask=inputs["target_mask"].t())
        decoder_state["cloze_logits"] = (
            recon_logits, ("batch", "sequence", "vocab"))
        return decoder_state

    def initialize_parameters(self):
        self.round_trip_model.initialize_parameters()

    def set_dropout(self, dropout):
        self.round_trip_model.set_dropout(dropout)
