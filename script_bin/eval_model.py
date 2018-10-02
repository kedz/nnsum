import argparse
import pathlib
import ujson as json

import torch
import nnsum
import pandas as pd
import rouge_papier


def main():
    parser = argparse.ArgumentParser(
        "Evaluate nnsum models using original Perl ROUGE script.")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--sentence-limit", default=None, type=int)
    parser.add_argument("--summary-length", type=int, default=100)
    parser.add_argument(
        "--remove-stopwords", action="store_true", default=False)
    parser.add_argument(
        "--inputs", type=pathlib.Path, required=True)
    parser.add_argument(
        "--refs", type=pathlib.Path, required=True)
    parser.add_argument(
        "--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--results", type=pathlib.Path, required=False, default=None)
 
    args = parser.parse_args() 

    print("Loading model...", end="", flush=True)
    model = torch.load(args.model, map_location=lambda storage, loc: storage)
    if args.gpu > -1:
        model.cuda(args.gpu)
    vocab = model.embeddings.vocab
    print(" OK!")

    data = nnsum.data.SingleDocumentDataset(
        vocab,
        args.inputs,
        references_dir=args.refs,
        sentence_limit=args.sentence_limit)
    loader = data.dataloader(
        batch_size=args.batch_size)

    ids = []
    path_data = []
    model.eval()
    with rouge_papier.util.TempFileManager() as manager:
        with torch.no_grad():
            for step, batch in enumerate(loader, 1):
                batch = batch.to(args.gpu)
                print("generating summaries {} / {} ...".format(
                        step, len(loader)),
                    end="\r" if step < len(loader) else "\n", flush=True)
                texts = model.predict(batch, max_length=args.summary_length)
                
                for text, ref_paths in zip(texts, batch.reference_paths):
                    summary = "\n".join(text)                
                    summary_path = manager.create_temp_file(summary)
                    path_data.append(
                        [summary_path, [str(x) for x in ref_paths]])
                ids.extend(batch.id)

        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(
            config_path, max_ngram=2, lcs=True, 
            remove_stopwords=args.remove_stopwords,
            length=args.summary_length)
        df.index = ids + ["average"]
        df = pd.concat([df[:-1].sort_index(), df[-1:]], axis=0)
        print(df[-1:])
       
        if args.results:
            records = df[:-1].to_dict("records")

            results = {"idividual": {id: record 
                                     for id, record in zip(ids, records)},
                       "average": df[-1:].to_dict("records")[0]}
            args.results.parent.mkdir(parents=True, exist_ok=True)
            with args.results.open("w") as fp:
                fp.write(json.dumps(results))
        
if __name__ == "__main__":
    main()
