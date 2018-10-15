import unittest

import tempfile
import shutil
import pathlib
import json

import torch
from nnsum.data import SummarizationDataset
from nnsum.io import Vocab


class TestSummarizationDataset(unittest.TestCase):

    def create_dummy_data(self):
        word_list= ["this", "is", "a", "test", ".", "another", "final", 
                    "sentence", "now"]
        vocab = Vocab.from_word_list(word_list)
        input_dir = tempfile.mkdtemp()
        input1_path = pathlib.Path(input_dir) / "ex1.json"
        input2_path = pathlib.Path(input_dir) / "ex2.json"
        input3_path = pathlib.Path(input_dir) / "ex3.json"
        input4_path = pathlib.Path(input_dir) / "ex4.json"
        input5_path = pathlib.Path(input_dir) / "ex5.json"
        input1_path.write_text(json.dumps(
            {"id": "ex1", 
             "inputs": [{"text": "This is a test.", 
                         "tokens": ["This", "is", "a", "test", "."]},
                        {"text": "Another test.", 
                         "tokens": ["Another", "test", "."]},  
                        {"text": "A final sentence now.",
                         "tokens": ["A", "final", "sentence", "now", "."]}]}))

        input2_path.write_text(json.dumps(
            {"id": "ex2", 
             "inputs": [{"text": "Another test.", 
                         "tokens": ["Another", "test", "."]},  
                        {"text": "A final sentence now.",
                         "tokens": ["A", "final", "sentence", "now", "."]}]}))

        input3_path.write_text(json.dumps(
            {"id": "ex3", 
             "inputs": [{"text": "This is a test.", 
                         "tokens": ["This", "is", "a", "test", "."]},
                        {"text": "Another test.", 
                         "tokens": ["Another", "test", "."]},  
                        {"text": "A final sentence now.",
                         "tokens": ["A", "final", "sentence", "now", "."]}]}))

        input4_path.write_text(json.dumps(
            {"id": "ex4", 
             "inputs": [{"text": "This is a test.", 
                         "tokens": ["This", "is", "a", "test", "."]},
                        {"text": "A final sentence now.",
                         "tokens": ["A", "final", "sentence", "now", "."]}]}))

        input5_path.write_text(json.dumps(
            {"id": "ex5", 
             "inputs": [{"text": "This is a test.", 
                         "tokens": ["This", "is", "a", "test", "."]},
                        {"text": "Another test.", 
                         "tokens": ["Another", "test", "."]},  
                        {"text": "A final sentence now.",
                         "tokens": ["A", "final", "sentence", "now", "."]}]}))


        label_dir = tempfile.mkdtemp()
        label1_path = pathlib.Path(label_dir) / "ex1.json"
        label2_path = pathlib.Path(label_dir) / "ex2.json"
        label3_path = pathlib.Path(label_dir) / "ex3.json"
        label4_path = pathlib.Path(label_dir) / "ex4.json"
        label5_path = pathlib.Path(label_dir) / "ex5.json"
        label1_path.write_text(json.dumps({"id": "ex1", "labels": [1, 0, 1]}))
        label2_path.write_text(json.dumps({"id": "ex2", "labels": [1, 0]}))
        label3_path.write_text(json.dumps({"id": "ex3", "labels": [1, 1, 0]}))
        label4_path.write_text(json.dumps({"id": "ex4", "labels": [0, 1]}))
        label5_path.write_text(json.dumps({"id": "ex5", "labels": [0, 1, 1]}))

        ref_dir = tempfile.mkdtemp()
        ref1a_path = pathlib.Path(ref_dir) / "ex1.a.txt"
        ref1b_path = pathlib.Path(ref_dir) / "ex1.b.txt"
        ref2_path = pathlib.Path(ref_dir) / "ex2.txt"
        ref3a_path = pathlib.Path(ref_dir) / "ex3.a.txt"
        ref3b_path = pathlib.Path(ref_dir) / "ex3.b.txt"
        ref4_path = pathlib.Path(ref_dir) / "ex4.txt"
        ref5_path = pathlib.Path(ref_dir) / "ex5.txt"
        ref1a_path.write_text("dummy")
        ref1b_path.write_text("dummy")
        ref2_path.write_text("dummy")
        ref3a_path.write_text("dummy")
        ref3b_path.write_text("dummy")
        ref4_path.write_text("dummy")
        ref5_path.write_text("dummy")
        

        return input_dir, label_dir, ref_dir, vocab



    def test_dataset_length(self):
        try:
            input_dir, label_dir, ref_dir, vocab = self.create_dummy_data()

            dataset = SummarizationDataset(vocab, input_dir) 

            self.assertTrue(len(dataset) == 5)
        
        
        finally:
            shutil.rmtree(input_dir)
            shutil.rmtree(label_dir)


    def test_dataset_getter(self):
        try:
            input_dir, label_dir, ref_dir, vocab = self.create_dummy_data()

            dataset = SummarizationDataset(vocab, input_dir) 

            for i in range(len(dataset)):
                self.assertTrue(dataset[i]["id"] == "ex{}".format(i + 1))
                self.assertTrue("targets" not in dataset[i])

            ref_doc = torch.LongTensor(
                [[vocab["another"], vocab["test"], vocab["."], vocab.pad_index,
                  vocab.pad_index],
                 [vocab["a"], vocab["final"], vocab["sentence"], vocab["now"],
                  vocab["."]]]) 

            self.assertTrue(torch.all(ref_doc == dataset[1]["document"]))
            self.assertTrue(
                "Another test." == dataset[1]["pretty_sentences"][0])
            self.assertTrue(
                "A final sentence now." == dataset[1]["pretty_sentences"][1])
            self.assertTrue(2 == dataset[1]["pretty_sentence_lengths"][0])
            self.assertTrue(4 == dataset[1]["pretty_sentence_lengths"][1])

            ref_sent_lens = torch.LongTensor([3, 5])
            self.assertTrue(
                torch.all(ref_sent_lens == dataset[1]["sentence_lengths"]))
            self.assertTrue(2 == dataset[1]["num_sentences"])
        
        finally:
            shutil.rmtree(input_dir)
            shutil.rmtree(label_dir)

    def test_targets(self):
        try:
            inputs_dir, targets_dir, ref_dir, vocab = self.create_dummy_data()

            dataset = SummarizationDataset(
                vocab, inputs_dir, targets_dir=targets_dir) 

            ref_labels = [
                torch.LongTensor([1, 0, 1]),
                torch.LongTensor([1, 0]),
                torch.LongTensor([1, 1, 0]),
                torch.LongTensor([0, 1]),
                torch.LongTensor([0, 1, 1])]
            for i in range(len(dataset)):
                self.assertTrue(
                    torch.all(dataset[i]["targets"] == ref_labels[i]))

        finally:
            shutil.rmtree(inputs_dir)
            shutil.rmtree(targets_dir)

    def test_references(self):
        try:
            inputs_dir, targets_dir, ref_dir, vocab = self.create_dummy_data()

            dataset = SummarizationDataset(
                vocab, inputs_dir, targets_dir=targets_dir,
                references_dir=ref_dir) 

            expected_paths = [
                [pathlib.Path(ref_dir) / "ex1.a.txt",
                 pathlib.Path(ref_dir) / "ex1.b.txt"],
                [pathlib.Path(ref_dir) / "ex2.txt"],
                [pathlib.Path(ref_dir) / "ex3.a.txt",
                 pathlib.Path(ref_dir) / "ex3.b.txt"],
                [pathlib.Path(ref_dir) / "ex4.txt"],
                [pathlib.Path(ref_dir) / "ex5.txt"]]

            for i in range(len(dataset)):
                self.assertTrue(
                    dataset[i]["reference_paths"] == expected_paths[i])

        finally:
            shutil.rmtree(inputs_dir)
            shutil.rmtree(targets_dir)


if __name__ == '__main__':
    unittest.main() 
