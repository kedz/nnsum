import argparse
import pathlib
import nnsum
import torch
from sacremoses import MosesTokenizer, MosesDetokenizer


detokenizer = MosesDetokenizer()

def postprocess(token_sequences):
    pptexts = []
    for token_seq in token_sequences:
        string = " ".join(token_seq)
        text = [detokenizer.detokenize(s.split()) 
                for s in string.split("<SENT>")]
        pptexts.append(" ".join(text))
    return pptexts

def main():
    parser = argparse.ArgumentParser("Generate text from data.")
    parser.add_argument("--inputs", type=pathlib.Path, required=True,
                        help="Path to input source.")
    parser.add_argument("--output", type=pathlib.Path, required=True,
                        help="Path to output source.")
    parser.add_argument("--model", type=pathlib.Path, required=True,
                        help="Path to model.")

    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
   
    model = torch.load(args.model).cpu()
 
    source = nnsum.data.RAMDataset(args.inputs)
    loader = nnsum.data.Seq2SeqDataLoader(
        source,
        model.encoder.embedding_context.named_vocabs,
        model.decoder.embedding_context.named_vocabs,
        batch_size=32,
        pin_memory=False,
        shuffle=False,
        sorted=False,
        include_original_data=True,
        num_workers=1)
 
    with args.output.open("w") as fp:
        for batch in loader:
            token_sequences = model.decode(batch)
            strings = postprocess(token_sequences)
            for data, text in zip(batch["orig_data"], strings):
                print(text)
                fp.write(text)
                fp.write("\n")

if __name__ == "__main__":
    main()
