import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset
import random
from util import initialize_vocab_and_embeddings 
from model_util import cheng_and_lapata_extractor_model
from collections import defaultdict
import rouge_papier


def check_dir(path):
    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

def make_dataset(input_path, label_path, tok2idx, batch_size, gpu,
                 sent_limit=500):

    inputs = []
    sent_lengths = []
    num_sents = []
    num_tokens = []
    targets = []
    ids = []
    texts = []

    with open(input_path, "r") as inp_fp, open(label_path, "r") as lbl_fp:
        for inp_line, lbl_line in zip(inp_fp, lbl_fp):
            example = json.loads(inp_line)
            labels = json.loads(lbl_line)
            if len(example["inputs"]) == 0:
                continue
            assert example["id"] == labels["id"]
            texts.append([inp["text"] for inp in example["inputs"]])
            ids.append(example["id"])
            targets.append(labels["labels"][:sent_limit])
            tokens = []
            sent_length = [] 
            for sent in example["inputs"][:sent_limit]:
                sent_length.append(len(sent["tokens"]))
                for token in sent["tokens"]:
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
    for sent_length, tgt in zip(sent_lengths, targets):
        if len(sent_length) < max_sents:
            tgt.extend([-1] * (max_sents - len(tgt)))
            sent_length.extend([0] * (max_sents - len(sent_length)))

    targets = torch.LongTensor(targets)
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
        ["targets", "targets"],
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
        (targets, num_sents, "targets"),
        (ids, "id"),
        (texts, "text"),
        batch_size=batch_size,
        gpu=gpu,
        lengths=num_sents,
        layout=layout,
        shuffle=True)
    return dataset

def train(optimizer, model, dataset, weight=None, grad_clip=5):
    model.train()
    total_xent = 0
    total_els = 0
    
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        optimizer.zero_grad()
        
        logits = model(batch.inputs, decoder_supervision=batch.targets.float())
        mask = batch.targets.gt(-1).float()
        total_sentences_batch = batch.inputs.sentence_counts.data.sum()
        
        if weight is not None:
            mask.data.masked_fill_(batch.targets.data.eq(1), weight)

        bce = F.binary_cross_entropy_with_logits(
            logits, batch.targets.float(),
            weight=mask, 
            size_average=False)

        avg_bce = bce / total_sentences_batch
        avg_bce.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-grad_clip, grad_clip)
        optimizer.step()

        total_xent += bce.data[0]
        total_els += total_sentences_batch

        sys.stdout.write(
            "train: {}/{} XENT={:0.6f}\r".format(
                n_iter, max_iters, total_xent / total_els))
        sys.stdout.flush()

    return total_xent / total_els

def validate(model, dataset, summary_dir, weight=None, remove_stopwords=True, 
             summary_length=100):
    model.eval()
    total_xent = 0
    total_els = 0
    
    max_iters = int(np.ceil(dataset.size / dataset.batch_size))
    
    for n_iter, batch in enumerate(dataset.iter_batch(), 1):
        
        logits = model(batch.inputs)
        mask = batch.targets.gt(-1).float()
        total_sentences_batch = batch.inputs.sentence_counts.data.sum()
        
        if weight is not None:
            mask.data.masked_fill_(batch.targets.data.eq(1), weight)

        bce = F.binary_cross_entropy_with_logits(
            logits, batch.targets.float(),
            weight=mask, 
            size_average=False)

        avg_bce = bce / total_sentences_batch

        total_xent += bce.data[0]
        total_els += total_sentences_batch

        sys.stdout.write(
            "valid: {}/{} XENT={:0.6f}\r".format(
                n_iter, max_iters, total_xent / total_els))
        sys.stdout.flush()

    rouge_df, hist = compute_rouge(
        model, dataset, summary_dir, remove_stopwords=remove_stopwords,
        summary_length=summary_length)
    r1, r2 = rouge_df.values[0].tolist()    
    
    return total_xent / total_els, r1, r2, hist

def collect_reference_paths(reference_dir):
    ids2refs = defaultdict(list)
    for filename in os.listdir(reference_dir):
        id = filename.rsplit(".", 2)[0]
        ids2refs[id].append(os.path.join(reference_dir, filename))
    return ids2refs

def compute_rouge(model, dataset, reference_dir, remove_stopwords=True,
                  summary_length=100):

    model.eval()

    hist = {}
    ids2refs = collect_reference_paths(reference_dir)

    with rouge_papier.util.TempFileManager() as manager:

        path_data = []
        for batch in dataset.iter_batch():
            texts, positions = model.predict(
                batch.inputs, batch.metadata, return_indices=True,
                max_length=summary_length)
            for pos_b in positions:
                for p in pos_b:
                    hist[p] = hist.get(p, 0) + 1
            for b, text in enumerate(texts):
                id = batch.metadata.id[b]
                summary = "\n".join(text)                
                summary_path = manager.create_temp_file(summary)
                ref_paths = ids2refs[id]
                path_data.append([summary_path, ref_paths])

        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(
            config_path, max_ngram=2, lcs=False, 
            remove_stopwords=remove_stopwords,
            length=summary_length)
        return df[-1:], hist

def main():

    import argparse

    parser = argparse.ArgumentParser()

    # Input File Parameters
    parser.add_argument("--train-inputs", type=str, required=True)
    parser.add_argument("--train-labels", type=str, required=True)
    parser.add_argument("--valid-inputs", type=str, required=True)
    parser.add_argument("--valid-labels", type=str, required=True)
    parser.add_argument("--valid-summary-dir", required=True, type=str)

    # Training Parameters parameters
    parser.add_argument("--seed", default=48929234, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weighted", default=False, action="store_true")
    parser.add_argument("--sent-limit", default=500, type=int)
    parser.add_argument("--teacher-forcing", default=5, type=int)

    # Output File Locations
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)

    # Word Embedding Parameters
    parser.add_argument("--embedding-size", default=50, type=int)
    parser.add_argument(
        "--pretrained-embeddings", type=str, required=False, default=None)
    parser.add_argument(
        "--fix-embeddings", action="store_true", default=False)
    parser.add_argument("--embedding-dropout", default=0.0, type=float)
    parser.add_argument("--at-least", default=3, type=int)

    # Sentence Encoder Parameters
    parser.add_argument(
        "--sent-encoder", default="avg", choices=["cnn", "avg", "rnn"])
    parser.add_argument(
        "--sent-dropout", default=.25, type=float)
    parser.add_argument(
        "--sent-filter-windows", default=[1, 2, 3, 4, 5, 6], type=int, 
        nargs="+")
    parser.add_argument(
        "--sent-feature-maps", default=[25, 25, 50, 50, 50, 50], 
        type=int, nargs="+")

    # Document Encoder Parameters
    parser.add_argument(
        "--doc-rnn-hidden-size", default=100, type=int)
    parser.add_argument(
        "--doc-rnn-bidirectional", action="store_true", default=False)
    parser.add_argument(
        "--doc-rnn-dropout", default=.25, type=float)

    # MLP Parameters
    parser.add_argument(
        "--mlp-layers", default=[100], type=int, nargs="+")
    parser.add_argument(
        "--mlp-dropouts", default=[.25], type=float, nargs="+")

    # ROUGE Parameters
    parser.add_argument(
        "--remove-stopwords", action="store_true", default=False)
    parser.add_argument(
        "--summary-length", default=100, type=int)
    
##    parser.add_argument("--shuffle-doc", default=False, action="store_true")
#    parser.add_argument(
#        "--sentence-extractor", default="c&l", 
#        choices=["c&l", "simple", "rnn"])
#    parser.add_argument("--attention", default=None, choices=["dot"])

    args = parser.parse_args()

    if len(args.sent_feature_maps) != len(args.sent_filter_windows):
        print("--sent-feature-maps and --sent- filter-windows must have same "
              "number of arguments!")
        sys.exit(1)
    
    if len(args.mlp_layers) != len(args.mlp_dropouts):
        print("--mlp-layers and --mlp-dropouts must have same number",
              "of arguments!")
        sys.exit(1)
 
    check_dir(args.vocab)
    check_dir(args.model_path)
    check_dir(args.results_path)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tok2idx, idx2tok, embeddings = initialize_vocab_and_embeddings(
        args.train_inputs, args.embedding_size, 
        dropout=args.embedding_dropout, 
        trainable=not args.fix_embeddings,
        embeddings_path=args.pretrained_embeddings)

    print("Writing vocab to {} ...".format(args.vocab))
    with open(args.vocab, "w") as fp:
        fp.write(json.dumps([idx2tok, tok2idx]))

    train_data = make_dataset(
        args.train_inputs, args.train_labels, tok2idx, 
        batch_size=args.batch_size,
        gpu=args.gpu,
        sent_limit=args.sent_limit)

    valid_data = make_dataset(
        args.valid_inputs, args.valid_labels, tok2idx, 
        batch_size=args.batch_size,
        gpu=args.gpu,
        sent_limit=args.sent_limit)

    model = cheng_and_lapata_extractor_model(
        embeddings,
        sent_dropout=args.sent_dropout,
        sent_encoder_type=args.sent_encoder,
        sent_filter_windows=args.sent_filter_windows,
        sent_feature_maps=args.sent_feature_maps,
        doc_rnn_hidden_size=args.doc_rnn_hidden_size,
        doc_rnn_bidirectional=args.doc_rnn_bidirectional,
        doc_rnn_dropout=args.doc_rnn_dropout,
        mlp_layers=args.mlp_layers,
        mlp_dropouts=args.mlp_dropouts)

    if args.gpu > -1:
        model.cuda(args.gpu)

    if args.weighted:
        print("\nComputing class weights...")
        label_counts = [0, 0]
        labels, counts = np.unique(
            train_data.targets.numpy(), return_counts=True)
        for label, count in zip(labels, counts):
            label_counts[label] = count
        print("Counts y=0: {}, y=1 {}".format(*label_counts))
        weight = label_counts[0] / label_counts[1]
        print("Reweighting y=1 by {}\n".format(weight))
    else:
        weight = None

    print("\nInitializing weights")
    for name, param in model.named_parameters():
        if "emb" not in name and "weight" in name:
            print(name, "Xavier Normal Initialization", 
                "({})".format(",".join([str(x) for x in param.data.size()])))
            nn.init.xavier_normal(param)    
        elif "decoder_start" in name:
            print(name, "Normal Initialization", 
                "({})".format(",".join([str(x) for x in param.data.size()])))
            nn.init.normal(param)    
        elif "emb" not in name and "bias" in name:
            print(name, "Constant Initialization = 0",
                "({})".format(",".join([str(x) for x in param.data.size()])))
            nn.init.constant(param, 0)    
        else:
            print(name, "???", 
                "({})".format(",".join([str(x) for x in param.data.size()])))

    optim = torch.optim.Adam(model.parameters(), lr=.0001) 

    train_xents = []
    valid_results = []

    best_rouge_2 = 0
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        print("=== {:4d} ===".format(epoch))
        
#        if args.shuffle_doc:
#            training_data = shuffle_data(training_data)

        if args.teacher_forcing < epoch and \
                model.document_decoder.teacher_forcing:
            print("Disabling teach forcing.")
            model.document_decoder.teacher_forcing = False

        train_xent = train(optim, model, train_data, weight=weight)
        train_xents.append(train_xent)
        
        valid_result = validate(
            model, valid_data, args.valid_summary_dir, 
            weight=weight,
            remove_stopwords=args.remove_stopwords, 
            summary_length=args.summary_length)
        valid_results.append(valid_result)
        print(("Epoch {} :: Train xent: {:0.3f} | Valid xent: {:0.3f} | " \
               "R1: {:0.3f} | R2: {:0.3f}").format(
                  epoch, train_xents[-1], *valid_results[-1]))

        if valid_results[-1][-2] > best_rouge_2:
            best_rouge_2 = valid_results[-1][-2]
            best_epoch = epoch
            if args.model_path is not None:
                print("Saving model ...")
                torch.save(model, args.model_path)

    print("Best epoch: {}  ROUGE-1 {:0.3f}  ROUGE-2 {:0.3f}".format(
        best_epoch, *valid_results[best_epoch - 1][1:-1]))
    
    if args.results_path is not None:
        results = {"training": {"cross-entropy": train_xents},
                   "validation": {
                       "cross-entropy": [x[0] for x in valid_results], 
                       "position_histogram": [x[3] for x in valid_results], 
                       "rouge-1": [x[1] for x in valid_results],
                       "rouge-2": [x[2] for x in valid_results]}}
        print("Writing results to {} ...".format(args.results_path))
        with open(args.results_path, "w") as fp:
            fp.write(json.dumps(results)) 
   
if __name__ == "__main__":
    main()
