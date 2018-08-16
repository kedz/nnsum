import nnsum.io

import torch
import numpy as np
import pandas as pd
import rouge_papier

import argparse
import os
import sys
import json
from collections import defaultdict
import logging

logging.getLogger().setLevel(logging.INFO)


def check_dir(path):
    if path != "" and not os.path.exists(path):
        os.makedirs(path)

def collect_reference_paths(reference_dir):
    ids2refs = defaultdict(list)
    for filename in os.listdir(reference_dir):
        id = filename.rsplit(".", 2)[0]
        ids2refs[id].append(os.path.join(reference_dir, filename))
    return ids2refs

def compute_rouge(model, dataset, reference_dir, output_dir, 
                  remove_stopwords=True,
                  summary_length=100):

    model.eval()

    hist = {}
    ids2refs = collect_reference_paths(reference_dir)
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))

    ordered_ids = []

    with rouge_papier.util.TempFileManager() as manager:

        path_data = []
        for i, batch in enumerate(dataset.iter_batch(), 1):
            sys.stdout.write("{}/{}\r".format(i, max_iters))
            sys.stdout.flush()
             
            texts, positions = model.predict(
                batch.inputs, batch.metadata, return_indices=True,
                max_length=summary_length+25)
            for pos_b in positions:
                for p in pos_b:
                    p = int(p)
                    hist[p] = hist.get(p, 0) + 1
            for b, text in enumerate(texts):
                id = batch.metadata.id[b]
                summary = "\n".join(text)                
                summary_path = os.path.join(
                    output_dir, "{}.summary".format(id))
                with open(summary_path, "w") as sfp:
                    sfp.write(summary)
                ref_paths = ids2refs[id]
                path_data.append([summary_path, ref_paths])
                ordered_ids.append(id)

        print("")
        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(
            config_path, max_ngram=2, lcs=True, 
            remove_stopwords=remove_stopwords,
            length=summary_length)
        df.index = ordered_ids + ["average"]
        df = pd.concat([df[:-1].sort_index(), df[-1:]], axis=0)
        return df, hist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--references", nargs="+", required=True)
    parser.add_argument("--summary-outputs", nargs="+", required=True)
    parser.add_argument("--part-names", type=str,  nargs="+", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--summary-length", default=100, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    #parser.add_argument(
     #   "--remove-pos", default=None, 
     #   choices=["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ",
     #            "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    #             "SCONJ", "SYM", "VERB", "X"],              
     #   nargs="+")

    args = parser.parse_args()

    assert len(args.inputs) == len(args.summary_outputs)
    assert len(args.inputs) == len(args.references)
    assert len(args.inputs) == len(args.part_names)
#    if args.remove_pos is not None:
#        remove_pos = set(args.remove_pos)
#    else:
#        remove_pos = None

    logging.info(" Loading model from {}.".format(args.model_path))
    model = torch.load(
        args.model_path, map_location=lambda storage, loc: storage)
    if args.gpu > -1:
        model.cuda(args.gpu)
    logging.info(" Model loaded.")

    vocab = model.embeddings.vocab

    results = {}
    for inputs_path, ref_dir, outputs_dir, name in zip(
            args.inputs, args.references, args.summary_outputs, 
            args.part_names):
        logging.info(" Loading inputs.")
        dataset = nnsum.io.make_sds_prediction_dataset(
            inputs_path,
            vocab,
            batch_size=args.batch_size,
            gpu=args.gpu)

        check_dir(outputs_dir)
        rouge_df, hist = compute_rouge(
            model, dataset, ref_dir,
            outputs_dir,
            remove_stopwords=True,
            summary_length=args.summary_length)

        results[name] = {"rouge": rouge_df.to_dict(), "hist": hist}

        print(rouge_df[-1:])
   
    check_dir(os.path.dirname(args.results))
    print("Writing results to {} ...".format(args.results))
    with open(args.results, "w") as fp:
        fp.write(json.dumps(results))

if __name__ == "__main__":
    main()
