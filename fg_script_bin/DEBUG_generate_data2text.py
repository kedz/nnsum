import argparse
import sys
import pathlib
import nnsum
import torch
from sacremoses import MosesDetokenizer
import textwrap


detokenizer = MosesDetokenizer()

def postprocess(token_sequences):
    pptexts = []
    for token_seq in token_sequences:
        string = " ".join(token_seq)
        text = [detokenizer.detokenize(s.replace(" </DS>","").split(" ")) 
                for s in string.split("<SENT>")]
        pptexts.append(" ".join(text))
    return pptexts

def references_iter(path):
    refs = []
    with path.open("r") as fp:
        for line in fp:
            if line.strip() == '':
                yield refs
                refs = []
            else:
                refs.append(line.strip())

def pretty_print_data(data, fp):
    keys = sorted(data.keys())
    items = []
    for key in keys:
        text = "{}={}".format(key, data[key]).replace(" ", "_")
        items.append(text)
    print("   ".join(items), file=fp)

def pack_lines(left_col, right_col, fp, col_width=70):

    wrapper = textwrap.TextWrapper(width=col_width, subsequent_indent="    ")
    line_tmp = "{:"+ str(col_width) + "s}     {:" + str(col_width) + "s}"
    right_col = [l for t in right_col for l in wrapper.wrap(t)]
    left_col = [l for t in left_col for l in wrapper.wrap(t)]

    if len(left_col) < len(right_col):
        diff = len(right_col) - len(left_col)
        left_col = left_col + [""] * diff

    if len(right_col) < len(left_col):
        diff = len(left_col) - len(right_col)
        right_col = right_col + [""] * diff

    for ll, rl in zip(left_col, right_col):
        print(line_tmp.format(ll, rl), file=fp)

def main():
    parser = argparse.ArgumentParser("Generate text from data.")
    parser.add_argument("--source", type=pathlib.Path, required=True,
                        help="Path to input source json.")
    parser.add_argument("--target", type=pathlib.Path, required=True,
                        help="Path to input source json.")
#    parser.add_argument("--output", type=pathlib.Path, required=True,
#                        help="Path to output source.")
    parser.add_argument("--model", type=pathlib.Path, required=True,
                        help="Path to model.")
    parser.add_argument("--beam-size", type=int, default=16,
                        help="Beam size.")

    args = parser.parse_args()
#    args.output.parent.mkdir(exist_ok=True, parents=True)

    ri = references_iter(args.target)

    model = torch.load(args.model).cpu()
    model.eval()
 
    source = nnsum.data.RAMDataset(args.source)
    loader = nnsum.data.Seq2SeqDataLoader(
        source,
        model.encoder.embedding_context.named_vocabs,
        model.decoder.embedding_context.named_vocabs,
        batch_size=2,
        pin_memory=False,
        shuffle=False,
        sorted=False,
        include_original_data=True,
        num_workers=8)
 
    #with args.output.open("w") as fp:

    try:
        fp = sys.stdout

        for source in loader:
            orig_data = source["orig_data"]
            greedy_best_sequences = postprocess(model.decode(source))
            nbest_seqs = model.beam_decode(
                source, beam_size=args.beam_size)
            nbest_seqs = [postprocess(beam) for beam in nbest_seqs]

            for i, data in enumerate(orig_data):

                print("Data", file=fp)
                print("----", file=fp)
                pretty_print_data(data["data"], fp)
                print("", file=fp)

                left_col = ["Greedy Decoding",
                            "---------------",
                            greedy_best_sequences[i],
                            " "]
 
                left_col.append(" ")
                left_col.append("Beam Decoding")
                left_col.append("-------------")
                for j in range(args.beam_size):
                    nbest_seq = nbest_seqs[i][j]
                    left_col.append("[{}] {}".format(j, nbest_seq))
                    left_col.append(" ")

                  #  , (seq, score, lp) in enumerate(zip(beam_seqs[b], beam_scores[b], beam_lps[b]), 1):
#                #    seq_toks = model.decoder.embedding_context.convert_index_tensor(seq)
#                #    seq_str = postprocess([seq_toks])[0]
#                #    left_col.append("[{}] ({:3.3f}|{:3.3f}) {}".format(
#                #        i, score, lp, seq_str))
#               
                right_col = ["References",
                             "----------"]
                for j, ref in enumerate(next(ri), 1):
                    right_col.append("[{}] {}".format(j, ref))
                    right_col.append("")
                
                pack_lines(left_col, right_col, fp)
#
                print("\n\n\n", file=fp, flush=True)
    finally:
        pass

if __name__ == "__main__":
    main()
