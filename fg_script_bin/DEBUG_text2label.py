import torch
import nnsum
import argparse
import pathlib
from colorama import Fore, Style
import numpy as np


def main():
    parser = argparse.ArgumentParser("Generate text from data.")
    parser.add_argument("--source", type=pathlib.Path, required=True,
                        help="Path to input source json.")
    parser.add_argument("--target", type=pathlib.Path, required=True,
                        help="Path to input source json.")

    parser.add_argument("--model", type=pathlib.Path, required=True,
                        help="Path to model.")

    args = parser.parse_args()

    source = nnsum.data.RAMDataset(args.source)
    target = nnsum.data.RAMDataset(args.target)

    model = torch.load(args.model).cpu()
    model.eval()

    loader = nnsum.data.Seq2ClfDataLoader(
        nnsum.data.AlignedDataset(source, target),
        model.source_embedding_context.named_vocabs,
        model.target_embedding_context.named_vocabs,
        batch_size=32,
        pin_memory=False,
        shuffle=False,
        num_workers=8,
        include_original_data=True)

    ent_thresh = .3

    for batch in loader:
        data = batch["orig_data"]
        pred_labels, attention = model.predict_labels(
            batch, return_attention=True)
        logits, _ = model(batch)

        print(logits)
        input()

        for a in attention:
            print(a)
        for b, data_b in enumerate(data):
            print()
#            print(data_b[0])
            print(data_b[1]["labels"])

            print()
            print(data_b[0]["text"])
            print()
            print("{:20s}   {:25s} : {:25s}".format(
                "label_type", "predicted", "true")) 
            print("-" * 80)
            entropies = {}

            for cls, pred_label in pred_labels[b].items():
                probs = torch.softmax(logits[cls][b], dim=0)
                logprobs = torch.log_softmax(logits[cls][b], dim=0)
                entropy = -(probs * logprobs).sum().item()
                entropies[cls] = entropy
                max_ent = torch.log(torch.tensor(float(probs.size(0)))).item()
                print("{:20s}   {:25s} : {:25s} : {:6.3f} : {:6.3f}".format(
                          cls, pred_label, data_b[1]["labels"].get(cls, "(N/A)"),
                          entropy, max_ent))
                 
            #print( batch["targets"].keys()

            #input()
            #print(attention[0][b])
           
            print()
            print("{:20s}".format("token"), end=" ")
            for label in batch["targets"].keys():
                if entropies[label] < ent_thresh:
                    print((Fore.GREEN + "{:10s}" + Style.RESET_ALL).format(label[:10]),
                      end=" : ")
                else:
                    print("{:10s}".format(label[:10]),
                      end=" : ")
            print()
            print("-"*80)
            for i, token in enumerate(["START"] + data_b[0]["tokens"]["tokens"] + ["STOP"]):
                
                print("{:20s}".format(token), end=" ")
                for j, attn in enumerate(attention):
                    a = attn[b][i].item() / .1
                    if a < 6:
                        a = 0
                    else:
                        a = int(np.round(a))
                    
                    print("{:10s}".format("*" * a),
                          end=" : ")
                print()

            print()
            input()
        
        #model.predict_labels(batch)

if __name__ == "__main__":
    main()
