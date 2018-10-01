import argparse
import pathlib
import logging

import torch
import nnsum
import random


logging.getLogger().setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser("Train a seq2seq extractive summarizer.")
    parser.add_argument("--train-inputs", type=pathlib.Path, required=True)
    parser.add_argument("--train-labels", type=pathlib.Path, required=True)
    parser.add_argument("--valid-inputs", type=pathlib.Path, required=True)
    parser.add_argument("--valid-labels", type=pathlib.Path, required=True)
    parser.add_argument("--valid-refs", type=pathlib.Path, required=True)

    # Output File Locations
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)

    # Training Parameters parameters
    parser.add_argument("--seed", default=48929234, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weighted", default=False, action="store_true")
    parser.add_argument("--sent-limit", default=None, type=int)

    # ROUGE Parameters
    parser.add_argument(
        "--remove-stopwords", action="store_true", default=False)
    parser.add_argument(
        "--summary-length", default=100, type=int) 

    nnsum.module.EmbeddingContext.update_command_line_options(parser)
    nnsum.model.Seq2SeqModel.update_command_line_options(parser)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
 
    print("Initializing vocabulary and embeddings.")
    embedding_context = nnsum.io.initialize_embedding_context(
        args.train_inputs,
        args.embedding_size,
        at_least=args.at_least,
        top_k=args.top_k,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        update_rule=args.update_rule,
        embeddings_path=args.pretrained_embeddings,
        filter_pretrained=args.filter_pretrained)

    print("Loading training data.")
    train_data = nnsum.data.SingleDocumentDataset(
        embedding_context.vocab,
        args.train_inputs, 
        labels_dir=args.train_labels, 
        sentence_limit=args.sent_limit)
    train_loader = train_data.dataloader(
        batch_size=args.batch_size,
        gpu=args.gpu)

    print("Loading validation data.")
    val_data = nnsum.data.SingleDocumentDataset(
        embedding_context.vocab, 
        args.valid_inputs, 
        labels_dir=args.valid_labels, 
        references_dir=args.valid_refs,
        sentence_limit=args.sent_limit)
    val_loader = valid_data.dataloader(
        batch_size=args.batch_size,
        gpu=args.gpu)

    if args.weighted:
        weight = nnsum.trainer.compute_class_weights(train_data)
    else:
        weight = None

    print("Building model.")
    model = nnsum.model.Seq2SeqModel.model_builder(
        embedding_context,
        sent_dropout=args.sent_dropout,
        sent_encoder_type=args.sent_encoder,
        sent_filter_windows=args.sent_filter_windows,
        sent_feature_maps=args.sent_feature_maps,
        sent_rnn_hidden_size=args.sent_rnn_hidden_size,
        sent_rnn_bidirectional=args.sent_rnn_bidirectional,
        doc_rnn_hidden_size=args.doc_rnn_hidden_size,
        doc_rnn_bidirectional=args.doc_rnn_bidirectional,
        doc_rnn_dropout=args.doc_rnn_dropout,
        doc_rnn_layers=args.doc_rnn_layers,
        mlp_layers=args.mlp_layers,
        mlp_dropouts=args.mlp_dropouts,
        attention=args.attention)

    if args.gpu > -1:
        print("Placing model on device: {}".format(args.gpu))
        model.cuda(args.gpu)

    model.initialize_parameters(logger=logging.getLogger())

    optimizer = torch.optim.Adam(model.parameters(), lr=.0001) 

    nnsum.trainer.labels_mle_trainer(
        model, optimizer, train_dataloader, valid_dataloader, 
        pos_weight=weight, max_epochs=args.epochs,
        summary_length=args.summary_length,
        remove_stopwords=args.remove_stopwords)
    
    
    exit()



    train_times = []
    valid_times = []
    train_xents = []
    valid_results = []

    best_rouge = 0
    best_epoch = None



    trainer = nnsum.trainer.create_trainer(
        model, optim, pos_weight=weight)


    trainer.run(train_dataloader, max_epochs=args.epochs)


    start_time = datetime.datetime.utcnow()
    logging.info(" Training start time: {}".format(start_time))

    for epoch in range(1, args.epochs + 1):
        logging.info(" *** Epoch {:4d} ***".format(epoch))
     
        train_start_time = datetime.datetime.utcnow()
        train_xent = nnsum.trainer.train_epoch(
            optim, model, train_dataloader, pos_weight=weight)
        train_xents.append(train_xent)
        train_epoch_time = datetime.datetime.utcnow() - train_start_time
        train_times.append(train_epoch_time)
        logging.info(" Training avg. x-entropy: {:6.5f}".format(train_xent))
        logging.info(" Training time: {}".format(train_epoch_time))

        valid_start_time = datetime.datetime.utcnow()
        valid_result = nnsum.trainer.validation_epoch(
            model, valid_dataloader, pos_weight=weight,
            remove_stopwords=args.remove_stopwords, 
            summary_length=args.summary_length)
        valid_results.append(valid_result)
        valid_epoch_time = datetime.datetime.utcnow() - valid_start_time
        valid_times.append(valid_epoch_time)
        logging.info((" Validation avg. x-entropy: {:6.5f} ROUGE-1: {:5.2f}" \
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
