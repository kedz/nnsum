import argparse
import pathlib
import nnsum
from nnsum.util import batch_pad_and_stack_vector
import torch
from sacremoses import MosesTokenizer, MosesDetokenizer
from collections import OrderedDict


detokenizer = MosesDetokenizer()

def postprocess(token_sequences):
    pptexts = []
    for token_seq in token_sequences:
        string = " ".join(token_seq)
        text = [detokenizer.detokenize(s.replace(" </DS>", "").split(" ")) 
                for s in string.split("<SENT>")]
        pptexts.append(" ".join(text))
    return pptexts

def decode(model, inputs, beam_size, reranker=None):
    if beam_size is None or beam_size == 1:
        return postprocess(model.decode(inputs))
    else:
        beam_decoded = model.beam_decode(inputs, beam_size=beam_size)
        if reranker is not None:
            reranker_func = make_reranker(reranker, inputs)
            beam_decoded = reranker_func(beam_decoded)
        
        strings = [postprocess(beam) for beam in beam_decoded]
        strings = [b[0] for b in strings]
        return strings

def make_reranker(clf, inputs):
    def rerank(batch_beams):
        batch_size = len(batch_beams)
        beam_size = len(batch_beams[0])
        rr_vcb = clf.source_embedding_context.vocab
        tgt_vcbs = clf.target_embedding_context.named_vocabs

        ground_truth = OrderedDict()
        for cls, tgt_vcb in tgt_vcbs.items():
            true_labels = [tgt_vcb[data["data"].get(cls, "(n/a)")]
                           for data in inputs["orig_data"]]
            true_labels = torch.LongTensor(true_labels).view(-1, 1).repeat(
                1, beam_size).view(-1)
            ground_truth[cls] = true_labels

        reranker_inputs = []
        for beam in batch_beams:
            for seq in beam:
                reranker_inputs.append(
                    torch.LongTensor(
                        [rr_vcb.start_index] + [rr_vcb[tok] for tok in seq]))
        reranker_inputs = batch_pad_and_stack_vector(reranker_inputs, 
                                                     rr_vcb.pad_index)
        scores_flat = clf.score({"source_features": reranker_inputs},
                                ground_truth)      
        scores = scores_flat.view(batch_size, beam_size)

        rerankings = torch.sort(scores, 1, descending=True)[1]
        
        for b, ranks in enumerate(rerankings):
            batch_beams[b] = [batch_beams[b][r.item()] 
                              for r in ranks]
        return batch_beams

    return rerank

def main():
    parser = argparse.ArgumentParser("Generate text from data.")
    parser.add_argument("--inputs", type=pathlib.Path, required=True,
                        help="Path to input source.")
    parser.add_argument("--output", type=pathlib.Path, required=True,
                        help="Path to output source.")
    parser.add_argument("--model", type=pathlib.Path, required=True,
                        help="Path to model.")
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--reranker", type=pathlib.Path, required=False,
                        default=None)

    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
   
    model = torch.load(args.model).cpu()
    model.eval()

    if args.reranker is not None:
        reranker = torch.load(args.reranker).cpu()
        reranker.eval()
    else:
        reranker = None
 
    source = nnsum.data.RAMDataset(args.inputs)
    loader = nnsum.data.Seq2SeqDataLoader(
        source,
        model.encoder.embedding_context.named_vocabs,
        model.decoder.embedding_context.named_vocabs,
        batch_size=8,
        pin_memory=False,
        shuffle=False,
        sorted=False,
        include_original_data=True,
        num_workers=1)
 
    with args.output.open("w") as fp:
        for batch in loader:
            strings = decode(model, batch, args.beam_size, reranker)
            for data, text in zip(batch["orig_data"], strings):
                print(text)
                fp.write(text)
                fp.write("\n")

if __name__ == "__main__":
    main()
