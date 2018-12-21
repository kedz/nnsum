import argparse
import torch
import sys
import pathlib
import nnsum
import torch
from sacremoses import MosesDetokenizer
import textwrap
from colorama import Fore, Style
import numpy as np
from collections import OrderedDict


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

    wrapper = textwrap.TextWrapper(width=col_width, subsequent_indent="    ",
                                   drop_whitespace=False)
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
    parser.add_argument("--clf", type=pathlib.Path, required=True,
                        help="Path to classifier.")
    parser.add_argument("--beam-size", type=int, default=16,
                        help="Beam size.")

    args = parser.parse_args()
#    args.output.parent.mkdir(exist_ok=True, parents=True)

    ri = references_iter(args.target)

    model = torch.load(args.model).cpu()
    model.eval()
    clf = torch.load(args.clf).cpu()
    clf.eval()

 
    label_vocabs = clf.target_embedding_context.named_vocabs

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

   
    tgt_emb_ctx = model.decoder.embedding_context

    try:
        fp = sys.stdout

        for source in loader:
            orig_data = source["orig_data"]
            greedy_best_idx_seqs = model.decode(source, return_tokens=False)
            greedy_best_tkn_seqs = tgt_emb_ctx.convert_index_tensor(
                greedy_best_idx_seqs)
            greedy_best_seqs = postprocess(greedy_best_tkn_seqs)
            greedy_best_labels = clf.predict_labels(
                {"source_features": greedy_best_idx_seqs})

#            greedy_best_sequences = postprocess(model.decode(source))
#            bh_sz, bm_sz, seq_len = nbest_seqs.size()
#            nbest_seqs_flat = nbest_seqs.view(bh_sz * bm_sz, seq_len)
##
#            pred_labels_log_probs = clf.log_probs(
#                {"source_features": nbest_seqs_flat})
#            for label, lp in pred_labels_log_probs.items():
#                pred_labels_log_probs[label] = lp.view(bh_sz, bm_sz, -1)
##
##


            gt_labels = OrderedDict()
            for cls, vcb in label_vocabs.items():
                batch_labels = []
                for i, data in enumerate(orig_data):
                    batch_labels.extend( 
                        [vcb[data["data"].get(cls, "(n/a)")]] * (args.beam_size ** 2))
                gt_labels[cls] = torch.LongTensor(batch_labels)


            def rescore_beam(lp, hist, next_tokens):
                bh, bm, bm = next_tokens.size()
                if hist is None:
                    hist = next_tokens.data.new(bh * bm * bm, 1).fill_(2)
                    clf_input = torch.cat(
                        [hist, next_tokens.view(bh * bm * bm, 1)], 1)
                    lps = clf.score(
                        {"source_features": clf_input}, gt_labels)
                    cur_step = clf_input.size(1)
                    return .25 * lps.view(bh, bm, bm) + .75 * lp / cur_step
                else:
                    
                    hist = hist.unsqueeze(1).repeat(1, bm, 1).view(
                        bh * bm * bm, -1)
                    next_tokens = next_tokens.view(bh * bm * bm, 1)

                    clf_input = torch.cat(
                        [hist, next_tokens], 1)
                    
                    lps = clf.score(
                        {"source_features": clf_input}, gt_labels)
                    cur_step = clf_input.size(1)
                    return .15 * lps.view(bh, bm, bm) + .85 * lp / cur_step


            orig_nbest_seqs, orig_nbest_scores = model.beam_decode(
                source, beam_size=args.beam_size, return_scores=True,
                return_tokens=True)
            orig_nbest_seqs = [postprocess(beam) for beam in orig_nbest_seqs]



            nbest_seqs, nbest_scores = model.beam_decode(
                source, beam_size=args.beam_size, return_scores=True,
                return_tokens=True, rescoring_func=rescore_beam)
            nbest_seqs = [postprocess(beam) for beam in nbest_seqs]




            for i, data in enumerate(orig_data):
                
                
                

                print("Data", file=fp)
                print("----", file=fp)
                pretty_print_data(data["data"], fp)
                print("", file=fp)

                for cls, pred_label in greedy_best_labels[i].items():
                    print("{:20s}: {:20s} | {:20s}".format(cls, pred_label, data["data"].get(cls, "(n/a)")))

                print()

                print("Greedy Decoding", file=fp)
                print("---------------", file=fp)
                print(greedy_best_seqs[i])

                print()

                
#                continue
#                left_col = ["Greedy Decoding",
#                            "---------------",
#                            greedy_best_sequences[i],
#                            " "]

                left_col = []
                right_col = []
#                right_col = ["Predicted Labels", 
#                             "----------------"]
 
                beam_lps = []
                left_col.append("Beam Decoding")
                left_col.append("-------------")
                right_col.append("Beam Decoding Rescored")
                right_col.append("----------------------")
                for j in range(args.beam_size):
                    nbest_seq = orig_nbest_seqs[i][j]
                    nbest_score = orig_nbest_scores[i,j].item()
                    left_col.append(
                        "[{}] ({:3.3f}) {}".format(j, nbest_score, nbest_seq))

                    nbest_seq = nbest_seqs[i][j]
                    nbest_score = nbest_scores[i,j].item()
                    right_col.append(
                        "[{}] ({:3.3f}) {}".format(j, nbest_score, nbest_seq))
#                    cand_lps = []
#                    for cls, vcb in label_vocabs.items():
#                        true = vcb[data["data"].get(cls, "(n/a)")]
#                        cand_lps.append(
#                            pred_labels_log_probs[cls][i][j][true].item())
#                    beam_lps.append(sum(cand_lps) / len(cand_lps))
#                    label_str = ""
#                    for label, lp in pred_labels_log_probs.items():
#                        pred_label = label_vocabs[label][
#                            torch.max(lp[i][j], 0)[1].item()]
#                        if data["data"].get(label, "(N/A)") == pred_label:
#                            pred_label = "**" + pred_label + "**"
#                        label_str += "{}={}  ".format(label, pred_label)  
                        
#                    right_col.append(label_str)
                    left_col.append(" ")
                    right_col.append(" ")
#                
                pack_lines(left_col, right_col, fp)
                input()
#
                print("\n\n\n", file=fp, flush=True)
                input()
    finally:
        pass

if __name__ == "__main__":
    main()
