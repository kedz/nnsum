import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNSeq2SeqModel(nn.Module):
    def __init__(self, source_embedding_context, target_embedding_context,
                 hidden_size=300, 
                 num_layers=2):
        super(RNNSeq2SeqModel, self).__init__()
        self.src_embedding_context = source_embedding_context
        self.src_encoder = nn.GRU(
            source_embedding_context.output_size, 
            hidden_size, num_layers=num_layers)
        self.tgt_embedding_context = target_embedding_context
        self.tgt_encoder = nn.GRU(
            target_embedding_context.embedding_size,
            hidden_size, num_layers=num_layers)
        self.predictor = nn.Linear(
            hidden_size, len(target_embedding_context.vocab))

    def forward(self, inputs):
        emb = self.src_embedding_context(inputs["source_features"])
        emb_packed = pack_padded_sequence(emb, inputs["source_lengths"],
                                          batch_first=False)

        enc_out_packed, state = self.src_encoder(emb_packed)
        enc_out, _ = pad_packed_sequence(enc_out_packed)

#        print(enc_out)
        
#        print(inputs.keys())        
        decoder_inputs = inputs["target_input_features"]["tokens"].t()

        logits = []
        for step in range(decoder_inputs.size(0)):
            inp_step = decoder_inputs[step:step + 1]
            inp_emb = self.tgt_embedding_context(inp_step)
#            print(inp_emb.size())
            dec_out, state = self.tgt_encoder(inp_emb, state)
#            print(dec_out.size())
            logits_step = self.predictor(dec_out.squeeze(0))
            logits.append(logits_step)

        return logits
