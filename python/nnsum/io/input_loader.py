from .dataset import Dataset
import logging
import torch
try:
    import ujson as json
except ImportError:
    import json


def sds_token_iter(inputs_path):
    with open(inputs_path, "r") as fp:
        for line in fp:
            example = json.loads(line)
            for sentence in example["inputs"]:
                for token in sentence["tokens"]:
                    yield token

def load_sds_inputs(inputs_path, vocab, sent_limit=None):

    tokens = []
    sentence_lengths = []
    num_sentences = []
    num_tokens = []
    pretty_sentence_lengths = []
    texts = []
    ids = []
   
    logging.info(" Reading inputs from {}".format(inputs_path))
 
    with open(inputs_path, "r") as fp:
        for line in fp:
            example = json.loads(line)
            ids.append(example["id"])
            if sent_limit is not None:
                sentences = example["inputs"][:sent_limit]
            else:
                sentences = example["inputs"]

            doc_text = []
            doc_tokens = []
            doc_sentence_lengths = [] 
            doc_pretty_sentence_lengths = []

            for sentence in sentences:
                doc_text.append(sentence["text"])
                doc_sentence_lengths.append(len(sentence["tokens"]))
                doc_tokens.extend([vocab.index(t) for t in sentence["tokens"]])
                doc_pretty_sentence_lengths.append(sentence["word_count"])

            tokens.append(doc_tokens)
            sentence_lengths.append(doc_sentence_lengths)
            num_tokens.append(len(doc_tokens))
            num_sentences.append(len(doc_sentence_lengths))
            texts.append(doc_text)
            pretty_sentence_lengths.append(doc_sentence_lengths)

    num_sentences = torch.LongTensor(num_sentences)
    max_sents = int(num_sentences.max())
    num_tokens = torch.LongTensor(num_tokens)
    max_tokens = int(num_tokens.max())

    for t in tokens:
        if len(t) < max_tokens:
            t.extend([0] * (max_tokens - len(t)))
    for sl in sentence_lengths:
        if len(sl) < max_sents:
            #tgt.extend([-1] * (max_sents - len(tgt)))
            sl.extend([0] * (max_sents - len(sl)))

    ids = tuple(ids)
    tokens = torch.LongTensor(tokens)
    sentence_lengths = torch.LongTensor(sentence_lengths)

    logging.info(" Read {} documents, {} sentences, and {} tokens".format(
        tokens.size(0), num_sentences.sum(), num_tokens.sum()))
    num_unk = tokens.eq(vocab.unknown_index).sum()
    logging.info(" {} tokens unknown ({:6.2f}%)".format(
        num_unk, 100 * num_unk / num_tokens.sum()))
    
    data = {"ids": ids, 
            "texts": texts,
            "tokens": tokens, 
            "num_tokens": num_tokens,
            "num_sentences": num_sentences, 
            "sentence_lengths": sentence_lengths,
            "pretty_sentence_lengths": pretty_sentence_lengths}
    return data
    
def load_sds_labels(path, sent_limit=None):

    labels = []
    num_sentences = []
    ids = []
   
    logging.info(" Reading labels from {}".format(path))
 
    with open(path, "r") as fp:
        for line in fp:
            example = json.loads(line)

            doc_labels = example["labels"]
            if sent_limit is not None:
                doc_labels = doc_labels[:sent_limit]
 
            ids.append(example["id"])
            labels.append(doc_labels)
            num_sentences.append(len(doc_labels))

    num_sentences = torch.LongTensor(num_sentences)
    max_sents = int(num_sentences.max())

    for doc_labels in labels:
        if len(doc_labels) < max_sents:
            doc_labels.extend([-1] * (max_sents - len(doc_labels)))

    ids = tuple(ids)
    labels = torch.LongTensor(labels)

    tot_sentences = num_sentences.sum()
    tot_extracts = labels.eq(1).sum()

    logging.info(" Read {} documents, {} sentences.".format(
        labels.size(0), tot_sentences))
    logging.info(" {} extract labels ({:6.2f}%)".format(
        tot_extracts, 100 * tot_extracts / tot_sentences))

    return {"ids": ids, "labels": labels, "num_sentences": num_sentences}


def make_sds_dataset(inputs_path, labels_path, vocab, batch_size=32, gpu=-1,
                     sent_limit=None, shuffle=True):
    inputs_data = load_sds_inputs(inputs_path, vocab, sent_limit=sent_limit)
    labels_data = load_sds_labels(labels_path, sent_limit=sent_limit)
    if inputs_data["ids"] != labels_data["ids"]:
        raise Exception("Different ids in inputs and label data.")
    
    if (inputs_data["num_sentences"] != labels_data["num_sentences"]).any():
        raise Exception("Different num_sentences in inputs and labels data.")
    
    layout = [
        ["inputs",
            [
             ["tokens", "tokens"],
             ["num_tokens", "num_tokens"],
             ["num_sentences", "num_sentences"],
             ["sentence_lengths", "sentence_lengths"],
            ]
        ],
        ["targets", "targets"],
        ["metadata", [["id", "id"], 
                      ["texts","texts"], 
                      ["sentence_lengths", "pretty_sentence_lengths"]]]
    ]

    dataset = Dataset(
        (inputs_data["tokens"], inputs_data["num_tokens"], "tokens"),
        (inputs_data["num_tokens"], "num_tokens"),
        (inputs_data["num_sentences"], "num_sentences"),
        (inputs_data["sentence_lengths"], inputs_data["num_sentences"], 
         "sentence_lengths"),
        (labels_data["labels"], inputs_data["num_sentences"], "targets"),
        (inputs_data["ids"], "id"),
        (inputs_data["texts"], "texts"),
        (inputs_data["pretty_sentence_lengths"], "pretty_sentence_lengths"),
        batch_size=batch_size,
        gpu=gpu,
        lengths=inputs_data["num_sentences"],
        layout=layout,
        shuffle=True)
    return dataset 

def make_sds_prediction_dataset(inputs_path, vocab, batch_size=32, gpu=-1,
                                sent_limit=None, shuffle=True):
    inputs_data = load_sds_inputs(inputs_path, vocab, sent_limit=sent_limit)
    
    layout = [
        ["inputs",
            [
             ["tokens", "tokens"],
             ["num_tokens", "num_tokens"],
             ["num_sentences", "num_sentences"],
             ["sentence_lengths", "sentence_lengths"],
            ]
        ],
        ["metadata", [["id", "id"], 
                      ["texts","texts"], 
                      ["sentence_lengths", "pretty_sentence_lengths"]]]
    ]

    dataset = Dataset(
        (inputs_data["tokens"], inputs_data["num_tokens"], "tokens"),
        (inputs_data["num_tokens"], "num_tokens"),
        (inputs_data["num_sentences"], "num_sentences"),
        (inputs_data["sentence_lengths"], inputs_data["num_sentences"], 
         "sentence_lengths"),
        (inputs_data["ids"], "id"),
        (inputs_data["texts"], "texts"),
        (inputs_data["pretty_sentence_lengths"], "pretty_sentence_lengths"),
        batch_size=batch_size,
        gpu=gpu,
        lengths=inputs_data["num_sentences"],
        layout=layout,
        shuffle=shuffle)
    return dataset 
