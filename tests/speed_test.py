import argparse
import pathlib
import time

import nnsum

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=pathlib.Path, required=True)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--refs", type=pathlib.Path, required=True)
    parser.add_argument(
        "--processes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 35])
    args = parser.parse_args()

    vocab = nnsum.io.initialize_embedding_context(
        args.inputs, 1, at_least=1,
        word_dropout=0.0,
        embedding_dropout=0.0,
        update_rule="update-all").vocab

    dataset = nnsum.data.SingleDocumentDataset(
        vocab, args.inputs,
        labels_dir=args.labels, 
        references_dir=args.refs,
        sentence_limit=50)

    print("Inputs path: {}".format(args.inputs))
    print("Labels path: {}".format(args.labels))
    print("Human References: {}".format(args.refs))

    for num_processes in args.processes:

        loader = dataset.dataloader(
            batch_size=32, 
            num_workers=num_processes)
        print("Num processes:", num_processes)
        
        start_time = time.time()
        for batch in loader:
            pass
        stop_time = time.time()
        print((stop_time - start_time) / len(loader))


if __name__ == "__main__":
    main()
