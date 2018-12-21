import torch
import nnsum
import argparse
import pathlib
from colorama import Fore, Style
import numpy as np
from nltk import word_tokenize


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
        pred_labels = model.predict_labels(
            batch)
        logits = model(batch)

        for b, data_b in enumerate(data):
            print()
#            print(data_b[0])
            print(data_b[1]["labels"])

            print()
            tokens = []
            for word in word_tokenize(data_b[0]["text"]):
                if word.lower() in model.source_embedding_context.vocab:
                    tokens.append(word)
                else:
                    tokens.append(Fore.YELLOW + word + Style.RESET_ALL)
            print(" ".join(tokens))
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
                true_label = data_b[1]["labels"].get(cls, "(n/a)")
                if true_label == pred_label:
                    true_label = Fore.GREEN + true_label + Style.RESET_ALL
                    pred_label = Fore.GREEN + pred_label + Style.RESET_ALL
                print("{:20s}   {:25s} : {:25s} : {:6.3f} : {:6.3f}".format(
                          cls, pred_label, true_label,
                          entropy, max_ent))
                 
            input()

if __name__ == "__main__":
    main()
