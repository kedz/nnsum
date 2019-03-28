import torch
import random

import nnsum
import logging
from rouge_score import RougeScorer
from ref_loader import *

logging.getLogger().setLevel(logging.INFO)

def main():

    args = nnsum.cli.training_argparser().parse_args()

    random.seed(args["trainer"]["seed"])
    torch.manual_seed(args["trainer"]["seed"])
    torch.cuda.manual_seed_all(args["trainer"]["seed"])

    print(args["trainer"])
    print()
    print(args["emb"])
    print()
    print(args["enc"])
    print()
    print(args["ext"])

    print("Initializing vocabulary and embeddings.")
    embedding_context = nnsum.io.initialize_embedding_context(
        args["trainer"]["train_inputs"], **args["emb"])

    print("Loading training data.")
    if args["trainer"]["shuffle_sents"]:
        print("Shuffling sentences!")
    train_data = nnsum.data.SampleCacheDataset(
        embedding_context.vocab,
        args["trainer"]["train_inputs"],
        targets_dir=args["trainer"]["train_labels"],
        sentence_limit=args["trainer"]["sentence_limit"],
        num_samples=args["trainer"]["raml_samples"],
        temperature=args["trainer"]["raml_temp"],
        shuffle_sents=args["trainer"]["shuffle_sents"])
    train_loader = nnsum.data.SampleCacheDataLoader(
        train_data, batch_size=args["trainer"]["batch_size"],
        num_workers=args["trainer"]["loader_workers"])

    print("Loading validation data.")
    val_data = nnsum.data.SummarizationDataset(
        embedding_context.vocab,
        args["trainer"]["valid_inputs"],
        targets_dir=args["trainer"]["valid_labels"],
        references_dir=args["trainer"]["valid_refs"],
        sentence_limit=args["trainer"]["sentence_limit"])
    val_loader = nnsum.data.SummarizationDataLoader(
        val_data, batch_size=args["trainer"]["batch_size"],
        num_workers=args["trainer"]["loader_workers"])

    if args["trainer"]["weighted"]:
        weight = nnsum.trainer.compute_class_weights(
            args["trainer"]["train_labels"],
            args["trainer"]["loader_workers"],
            sentence_limit=args["trainer"]["sentence_limit"])
    else:
        weight = None

    model = torch.load(args["trainer"]["mrt_model"], map_location=lambda storage, loc: storage)
    if args["trainer"]["gpu"] > -1:
        print("Placing model on device: {}".format(args["trainer"]["gpu"]))
        model.cuda(args["trainer"]["gpu"])
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

    # mrt stuff
    stopwords = set([word.strip().lower() for word in open(
                      args["trainer"]["stopwords"]).readlines()]) if args["trainer"]["stopwords"] else set()
    scorer = RougeScorer(stopwords=stopwords, word_limit=args["trainer"]["summary_length"])
    
    ids2refs = get_ids2refs(args["trainer"]["train_refs"])
    refs_dict = get_refs_dict(ids2refs, stopwords, word_limit=args["trainer"]["summary_length"])
    
    nnsum.trainer.labels_mrt_trainer(
        model, optimizer, train_loader, val_loader, 
        scorer, refs_dict, alpha=args["trainer"]["mrt_alpha"], 
        num_samples=args["trainer"]["mrt_samples"],
        pos_weight=weight, max_epochs=args["trainer"]["epochs"],
        summary_length=args["trainer"]["summary_length"],
        remove_stopwords=args["trainer"]["remove_stopwords"],
        gpu=args["trainer"]["gpu"],
        teacher_forcing=args["trainer"]["teacher_forcing"],
        model_path=args["trainer"]["model"],
        results_path=args["trainer"]["results"],
        valid_metric=args["trainer"]["valid_metric"])

if __name__ == "__main__":
    main()
