import argparse
import pathlib


def training_argparser():
    parser = argparse.ArgumentParser("Train a faithful generation model.")
    parser.add_argument("--train-source", type=pathlib.Path, required=True,
                        help="Path to training source file or directory")
    parser.add_argument("--train-target", type=pathlib.Path, required=True,
                        help="Path to training target file or directory")

    return parser
