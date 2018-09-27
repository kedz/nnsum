import argparse
import sys
import os
import torch
import json
from collections import defaultdict
from dataset import Dataset
import numpy as np
import rouge_papier
import pandas as pd

def check_dir(path):
    if path != "" and not os.path.exists(path):
        os.makedirs(path)

def filter_tokens(sentence, remove_pos):
    if remove_pos is None:
        return sentence["tokens"]
    else:
        tokens = [t if p not in remove_pos else "_UNK_" 
                  for t, p in zip(sentence["tokens"], sentence["pos"])]
        return tokens 

def make_dataset(input_path, tok2idx, batch_size, gpu, sent_limit=500,
                 remove_pos=None):

    inputs = []
    sent_lengths = []
    num_sents = []
    num_tokens = []
    ids = []
    texts = []
    print(remove_pos)

    with open(input_path, "r") as inp_fp:
        for inp_line in inp_fp:
            example = json.loads(inp_line)
            if len(example["inputs"]) == 0:
                continue
            texts.append([inp["text"] for inp in example["inputs"]])
            ids.append(example["id"])
            tokens = []
            sent_length = [] 
            #real_sent_length = []
            for sent in example["inputs"][:sent_limit]:
                filtered_tokens = filter_tokens(sent, remove_pos)
                sent_length.append(len(filtered_tokens))
                for token in filtered_tokens:
                    tokens.append(tok2idx.get(token, 1)) 
            inputs.append(tokens)
            sent_lengths.append(sent_length)
            num_tokens.append(len(tokens))
            num_sents.append(len(sent_length))

    num_sents = torch.LongTensor(num_sents)
    max_sents = num_sents.max()
    num_tokens = torch.LongTensor(num_tokens)
    max_tokens = num_tokens.max()

    for tokens in inputs:
        if len(tokens) < max_tokens:
            tokens.extend([0] * (max_tokens - len(tokens)))
    for sent_length in sent_lengths:
        if len(sent_length) < max_sents:
            sent_length.extend([0] * (max_sents - len(sent_length)))

    sent_lengths = torch.LongTensor(sent_lengths)
    inputs = torch.LongTensor(inputs)
    layout = [
        ["inputs", [
            ["tokens", "tokens"],
            ["token_counts", "num_tokens"],
            ["sentence_lengths", "sent_lengths"],
            ["word_count", "sent_lengths"],
            ["sentence_counts", "num_sents"],]
        ],
        ["metadata", [
            ["id", "id"],
            ["text","text"]]
        ]
    ]

    dataset = Dataset(
        (inputs, num_tokens, "tokens"),
        (num_tokens, "num_tokens"),
        (num_sents, "num_sents"),
        (sent_lengths, num_sents, "sent_lengths"),
        (ids, "id"),
        (texts, "text"),
        batch_size=batch_size,
        gpu=gpu,
        lengths=num_sents,
        layout=layout,
        shuffle=True)
    return dataset

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
        df, conf = rouge_papier.compute_rouge(
            config_path, max_ngram=2, lcs=True, 
            remove_stopwords=remove_stopwords,
            length=summary_length, return_conf=True)
        df.index = ordered_ids + ["average"]
        df = pd.concat([df[:-1].sort_index(), df[-1:]], axis=0)
        return df, conf, hist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--valid-inputs", required=True)
    parser.add_argument("--test-inputs", required=True)
    parser.add_argument("--valid-summaries", required=True)
    parser.add_argument("--test-summaries", required=True)
    parser.add_argument("--pred-valid-summaries", required=True)
    parser.add_argument("--pred-test-summaries", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--summary-length", default=100, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--remove-pos", default=None, 
        choices=["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ",
                 "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
                 "SCONJ", "SYM", "VERB", "X"],              
        nargs="+")

    args = parser.parse_args()

    if args.remove_pos is not None:
        remove_pos = set(args.remove_pos)
    else:
        remove_pos = None

    model = torch.load(args.model, map_location=lambda storage, loc: storage)
    if args.gpu > -1:
        model.cuda(args.gpu)

    with open(args.vocab, "r") as fp:
        idx2tok, tok2idx = json.load(fp)

    valid_data = make_dataset(
        args.valid_inputs, tok2idx, 
        batch_size=args.batch_size,
        gpu=args.gpu,
        sent_limit=1000, remove_pos=remove_pos)

    test_data = make_dataset(
        args.test_inputs, tok2idx, 
        batch_size=args.batch_size,
        gpu=args.gpu,
        sent_limit=1000, remove_pos=remove_pos)

    check_dir(args.pred_valid_summaries)
    check_dir(args.pred_test_summaries)
    valid_df, valid_conf, valid_hist = compute_rouge(
        model, valid_data, args.valid_summaries, 
        args.pred_valid_summaries,
        remove_stopwords=True,
        summary_length=args.summary_length)

    test_df, test_conf, test_hist = compute_rouge(
        model, test_data, args.test_summaries, 
        args.pred_test_summaries,
        remove_stopwords=True,
        summary_length=args.summary_length)

    print(test_df[-1:])
    print(test_conf)

    check_dir(os.path.dirname(args.results))
    print("Writing results to {} ...".format(args.results))
    with open(args.results, "w") as fp:
        d = {"valid": {"rouge": valid_df.to_dict(), 
                       "conf": valid_conf.to_dict(), 
                       "hist": valid_hist},
             "test": {"rouge": test_df.to_dict(), 
                      "conf": test_conf.to_dict(), 
                      "hist": test_hist}}
        fp.write(json.dumps(d))

if __name__ == "__main__":
    main()
