import argparse
import sys
import pathlib
from collections import defaultdict

from nnsum.module import EmbeddingContext
from nnsum.module import sentence_encoder as sent_enc
from nnsum.module import sentence_extractor as sent_ext
from nnsum.model import SummarizationModel


class ModuleArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ModuleArgumentParser, self).__init__(
            *args, usage=argparse.SUPPRESS, **kwargs)

class ModuleArgumentSelector(object):

    def __init__(self, name, desc=""):
        self._name = name
        self._modules = {}
        self._help_msg = {}
        self._desc = desc

    def add_module_opts(self, name, argparser, help=""):
        self._modules[name] = argparser
        self._help_msg[name] = help

    def print_help(self):
        print("  {} ARG [... ARG_OPTIONS]".format(self._name)) 
        print("  {}".format(self._desc))
        print("  Choices:")
        for mod_name in self._modules.keys():
            print("      {} {}".format(
                mod_name, ":: " + self._help_msg[mod_name]))
            print("             " + "\n             ".join(self._modules[mod_name].format_help().split("\n")))
            
    def parse_args(self, args=None):
        if args is None:
            args = sys.argv[1:]
        if len(args) == 0:
            self.print_help()
            exit()
        if args[0] not in self._modules or args[0] in ["-h", "-H", "--help"]:
            self.print_help()
            exit()
        else:
            r = vars(self._modules[args[0]].parse_args(args[1:]))
            r["OPT"] = args[0] 
            return r

class MultiModuleParser(object):
    def __init__(self, prog, description=""):
        self._prog = prog
        self._description = description
        self._modules = {}
        self._mod_list = []

    def add_module(self, name, argparser, help=""):
        assert name not in self._modules
        self._modules[name] = argparser
        self._mod_list.append(name)

    def build_usage_string(self):
        buf = "Usage: {}".format(self._prog)
        for name in self._mod_list:
            buf += " {} {}_ARGS".format(name, name[2:].upper())
        return buf

    def print_help(self):
        print(self.build_usage_string())
        for name in self._mod_list:
            self._modules[name].print_help()

    def parse_args(self, args=None):
        if args is None:
            args = sys.argv[1:]

        if len(args) == 0 or args[0] in ["-h", "-H", "--help"]:
            self.print_help()
            sys.exit()

        found_idxs = [i for i, v in enumerate(args) if v in self._modules]
        found_idxs.append(len(args))

        results = {}

        for i, start in enumerate(found_idxs[:-1]):
            stop = found_idxs[i + 1]
            sub_args = args[start + 1:stop]
            params = self._modules[args[start]].parse_args(sub_args)
            if not isinstance(params, dict):
                params = vars(params)
            results[args[start][2:]] = params

        missing_args = []
        for name in self._mod_list:
            if name[2:] not in results:
                missing_args.append(name)
        if len(missing_args) > 0:
            self.print_help()
            print("Missing the following arguments: {}".format(
                ", ".join(missing_args)))
            sys.exit()

        return results   

def training_argparser():

    train_parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
    train_parser.add_argument(
        "--train-inputs", type=pathlib.Path, required=True,
        help="Path to directory of training input json files.")
    train_parser.add_argument(
        "--train-labels", type=pathlib.Path, required=True,
        help="Path to directory of training label json files.")
    train_parser.add_argument(
        "--valid-inputs", type=pathlib.Path, required=True,
        help="Path to directory of validation input json files.")
    train_parser.add_argument(
        "--valid-labels", type=pathlib.Path, required=True,
        help="Path to directory of validation label json files.")
    train_parser.add_argument(
        "--valid-refs", type=pathlib.Path, required=True,
        help="Path to directory of validation human reference summaries.")
   
    train_parser.add_argument("--seed", default=48929234, type=int)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", default=32, type=int)
    train_parser.add_argument("--gpu", default=-1, type=int)
    train_parser.add_argument("--teacher-forcing", default=25, type=int)
    train_parser.add_argument("--sentence-limit", default=50, type=int)
    train_parser.add_argument(
        "--weighted", action="store_true", default=False,
        help="Upweight positive labels to make them proportional to the " \
             "negative labels.")   
    train_parser.add_argument("--loader-workers", type=int, default=8)
    train_parser.add_argument("--raml-samples", type=int, default=25)
    train_parser.add_argument("--raml-temp", type=float, default=.05)
    train_parser.add_argument("--summary-length", type=int, default=100)
    train_parser.add_argument(
        "--remove-stopwords", action="store_true", default=False)
    train_parser.add_argument(
        "--shuffle-sents", action="store_true", default=False,
        help="Shuffle training sentence order.")
    train_parser.add_argument("--model", type=pathlib.Path, default=None,
                              required=False)
    train_parser.add_argument("--results", type=pathlib.Path, default=None,
                              required=False)

    emb_parser = EmbeddingContext.argparser()

    enc_parser = ModuleArgumentSelector(
        "--enc", desc="Select sentence encoder module and settings.")
    avg_parser = sent_enc.AveragingSentenceEncoder.argparser()
    enc_parser.add_module_opts(
        "avg", avg_parser, help="Averaging sentence encoder.")
    cnn_parser = sent_enc.CNNSentenceEncoder.argparser()
    enc_parser.add_module_opts(
        "cnn", cnn_parser, help="Convolutional sentence encoder.")
    rnn_parser = sent_enc.RNNSentenceEncoder.argparser()
    enc_parser.add_module_opts(
        "rnn", rnn_parser, help="RNN sentence encoder.")

    ext_parser = ModuleArgumentSelector(
        "--ext", desc="Select sentence extractor module and settings.")
    rnn_ext_parser = sent_ext.RNNSentenceExtractor.argparser()
    ext_parser.add_module_opts(
        "rnn", rnn_ext_parser, help="RNN sentence extractor.")
    s2s_ext_parser = sent_ext.Seq2SeqSentenceExtractor.argparser()
    ext_parser.add_module_opts(
        "s2s", s2s_ext_parser, help="Seq2Seq sentence extractor.")
    cl_ext_parser = sent_ext.ChengAndLapataSentenceExtractor.argparser()
    ext_parser.add_module_opts(
        "cl", cl_ext_parser, help="Cheng & Lapata sentence extractor.")
    sr_ext_parser = sent_ext.SummaRunnerSentenceExtractor.argparser()
    ext_parser.add_module_opts(
        "sr", sr_ext_parser, help="SummaRunner sentence extractor.")


    parser = MultiModuleParser("train_model.py")
    parser.add_module("--trainer", train_parser)
    parser.add_module("--emb", emb_parser)
    parser.add_module("--enc", enc_parser)
    parser.add_module("--ext", ext_parser)

    return parser

def create_model_from_args(embedding_context, args):
   
    sent_encoder_type = args["enc"]["OPT"]
    del args["enc"]["OPT"]

    if sent_encoder_type == "avg":
        encoder = sent_enc.AveragingSentenceEncoder(
            embedding_context.embedding_size, **args["enc"])
    elif sent_encoder_type == "cnn":
        encoder = sent_enc.CNNSentenceEncoder(
            embedding_context.embedding_size, **args["enc"])
    elif sent_encoder_type == "rnn":
        encoder = sent_enc.RNNSentenceEncoder(
            embedding_context.embedding_size, **args["enc"])
    else:
        raise Exception("Bad encoder type: {}".format(sent_encoder_type))

    sent_extractor_type = args["ext"]["OPT"]
    del args["ext"]["OPT"]
    if sent_extractor_type == "rnn":
        extractor = sent_ext.RNNSentenceExtractor(
            encoder.size, **args["ext"])
    elif sent_extractor_type == "s2s":
        extractor = sent_ext.Seq2SeqSentenceExtractor(
            encoder.size, **args["ext"])
    elif sent_extractor_type == "cl":
        extractor = sent_ext.ChengAndLapataSentenceExtractor(
            encoder.size, **args["ext"])
    elif sent_extractor_type == "sr":
        extractor = sent_ext.SummaRunnerSentenceExtractor(
            encoder.size, **args["ext"])
    else:
        raise Exception("Bad extractor type: {}".format(sent_encoder_type))

    model = SummarizationModel(embedding_context, encoder, extractor)

    return model
