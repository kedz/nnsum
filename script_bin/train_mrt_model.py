import nnsum.trainer
import nnsum.io
import nnsum.module
import torch
import os
import argparse
import datetime
import logging
import json
import random
from nnsum.model.mrt_model import MRTModel
from nnsum.trainer import collect_reference_paths
from threading import Thread
import multiprocessing
from nnsum.utils.rouge_score import RougeScorer

logging.getLogger().setLevel(logging.INFO)

def check_dir(path):
    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

def _get_refs(ids2refs, stopwords, word_limit, refs_dict, ids):
  for id in ids:
    sens = [line.strip().split(" ") for line in open(ids2refs[id][0]).readlines()]
    tokens = [i for s in sens for i in s][:word_limit]
    d = dict()
    refs_dict[id] = d
    for t in tokens:
      if t not in d:
        d[t] = 1
      else:
        d[t] += 1
    if len(refs_dict) % (int(len(ids2refs)/3)) == 0 or len(refs_dict) == len(ids2refs):
      print("loading refs, done with %d from %d" % (len(refs_dict), len(ids2refs)))

def get_refs_dict(ids2refs, stopwords, word_limit):
  refs_dict = dict()
  threads = []
  keys = list(ids2refs.keys())
  step = int(len(ids2refs) / multiprocessing.cpu_count()) + 1
  for i in range(multiprocessing.cpu_count()):
    ids = [keys[idx] for idx in range(i*step, min(len(ids2refs),(i+1)*step))]
    t = Thread(target=_get_refs, args=(ids2refs, stopwords, word_limit, refs_dict, ids))
    threads.append(t)
    t.start()
  for t in threads: t.join()
  return refs_dict

def main():
    parser = argparse.ArgumentParser("Train an mrt model based extractive summarizer.")
    parser.add_argument("--train-inputs", type=str, required=True)
    parser.add_argument("--train-labels", type=str, required=True)
    parser.add_argument("--valid-inputs", type=str, required=True)
    parser.add_argument("--valid-labels", type=str, required=True)
    parser.add_argument("--train-refs", type=str, required=True)
    parser.add_argument("--valid-refs", type=str, required=True)
    parser.add_argument("--pretrained-model", type=str, required=True)
    parser.add_argument("--alt-rouge", default=False, action="store_true")

    # Output File Locations
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)

    # Training Parameters parameters
    parser.add_argument("--seed", default=48929234, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weighted", default=False, action="store_true")
    parser.add_argument("--sent-limit", default=None, type=int)
    parser.add_argument("--stopwords", default="stopwords.txt", type=str)

    # ROUGE Parameters
    parser.add_argument(
        "--remove-stopwords", action="store_true", default=False)
    parser.add_argument(
        "--summary-length", default=100, type=int) 
    
    nnsum.module.EmbeddingContext.update_command_line_options(parser)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
 
    if args.model_path:
        check_dir(args.model_path)
    if args.results_path:
        check_dir(args.results_path)

    logging.info(" Loading pretrained model for mrt training.")
    ids2refs_tr = collect_reference_paths(args.train_refs)
    ids2refs_vl = collect_reference_paths(args.valid_refs)

    ids2refs = {**ids2refs_tr, **ids2refs_vl}

    try:
      stopwords = set([word.strip() for word in open(args.stopwords).readlines()])
    except:
      stopwords = set()
      logging.warning(" Failed to load stopwords, training without.")

    ref_dicts = get_refs_dict(ids2refs, stopwords, args.summary_length)

    try:
      pretrained_model = torch.load(args.pretrained_model, map_location=lambda storage, loc: storage)
      logging.info(" Loaded pretrained model from %s successfully" % args.pretrained_model)
    except:
      raise Exception("mrt without pretrained model is not supported.")

    scorer = RougeScorer(stopwords, word_limit = args.summary_length)

    alt_rouge = (ref_dicts,scorer) if args.alt_rouge else None

    logging.info(" Alt rouge validation is set to: %s" % args.alt_rouge)
   
    model = MRTModel(ref_dicts, pretrained_model, scorer=scorer, stopwords=stopwords)

    logging.info(" Initializing vocabulary and embeddings.")
    embedding_context = nnsum.io.initialize_embedding_context(
        args.train_inputs, args.embedding_size, at_least=args.at_least,
        top_k=args.top_k, word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        update_rule=args.update_rule,
        embeddings_path=args.pretrained_embeddings,
        filter_pretrained=args.filter_pretrained)

    logging.info(" Loading training data.")
    train_data = nnsum.io.make_sds_dataset(
        args.train_inputs, args.train_labels, embedding_context.vocab,
        batch_size=args.batch_size,
        sent_limit=args.sent_limit,
        gpu=args.gpu)

    logging.info(" Loading validation data.")
    valid_data = nnsum.io.make_sds_dataset(
        args.valid_inputs, args.valid_labels, embedding_context.vocab,
        batch_size=args.batch_size,
        sent_limit=args.sent_limit,
        gpu=args.gpu)
    
    if args.weighted:
        weight = nnsum.trainer.compute_class_weights(train_data)
    else:
        weight = None

    if args.gpu > -1:
        logging.info(" Placing model on device: {}".format(args.gpu))
        model.cuda(args.gpu)

    train_times = []
    valid_times = []
    train_xents = []
    valid_results = []

    best_rouge = 0
    best_epoch = None

    optim = torch.optim.Adam(model.parameters(), lr=.0001) 

    start_time = datetime.datetime.utcnow()
    logging.info(" Training start time: {}".format(start_time))

    for epoch in range(1, args.epochs + 1):
        logging.info(" *** Epoch {:4d} ***".format(epoch))
     
        train_start_time = datetime.datetime.utcnow()
        train_xent = nnsum.trainer.train_epoch(
            optim, model, train_data, pos_weight=weight, mrt=True)
        train_xents.append(train_xent)
        train_epoch_time = datetime.datetime.utcnow() - train_start_time
        train_times.append(train_epoch_time)
        logging.info(" Training avg. risk: {:6.5f}".format(train_xent))
        logging.info(" Training time: {}".format(train_epoch_time))

        valid_start_time = datetime.datetime.utcnow()
        valid_result = nnsum.trainer.validation_epoch(
            model, valid_data, args.valid_refs, pos_weight=weight,
            remove_stopwords=args.remove_stopwords, 
            summary_length=args.summary_length, mrt=True, alt_rouge=alt_rouge)
        valid_results.append(valid_result)
        valid_epoch_time = datetime.datetime.utcnow() - valid_start_time
        valid_times.append(valid_epoch_time)
        logging.info((" Validation avg. risk: {:6.5f} ROUGE-1: {:5.2f}" \
                      " ROUGE-2: {:5.2f}").format(*valid_result))
        logging.info(" Validation time: {}".format(valid_epoch_time))

        if valid_result[-1] > best_rouge:
            logging.info(" Best model @ epoch {}".format(epoch))
            best_rouge = valid_result[-1]
            best_epoch = epoch

            if args.model_path is not None:
                logging.info(" Saving model to {} ...".format(args.model_path))
                torch.save(model, args.model_path)
                logging.info(" Model saved.")

        if epoch < args.epochs:
            avg_tt = sum(train_times, datetime.timedelta(0)) / len(train_times)
            avg_vt = sum(valid_times, datetime.timedelta(0)) / len(valid_times)
            avg_epoch_time = avg_tt + avg_vt
            time_remaining = avg_epoch_time * (args.epochs - epoch) 
            ect = time_remaining + datetime.datetime.utcnow()
            time_elapsed = datetime.datetime.utcnow() - start_time
            logging.info(" Time elaspsed: {}".format(time_elapsed))
            logging.info(" Estimated completion time: {}\n".format(ect))

    logging.info(" Finished training @ {}\n".format(datetime.datetime.utcnow()))
    logging.info(" Best epoch: {}  ROUGE-1 {:0.3f}  ROUGE-2 {:0.3f}".format(
        best_epoch, *valid_results[best_epoch - 1][1:]))
    
    if args.results_path is not None:
        results = {"training": {"cross-entropy": train_xents},
                   "validation": {
                       "cross-entropy": [x[0] for x in valid_results], 
                       "rouge-1": [x[1] for x in valid_results],
                       "rouge-2": [x[2] for x in valid_results]}}
        logging.info(" Writing results to {} ...".format(args.results_path))
        with open(args.results_path, "w") as fp:
            fp.write(json.dumps(results)) 

if __name__ == "__main__":
    main()
