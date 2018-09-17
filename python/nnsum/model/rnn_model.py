from nnsum.model.summarization_model import SummarizationModel
from nnsum.module.sentence_encoder import (AveragingSentenceEncoder,
    CNNSentenceEncoder)
from nnsum.module.sentence_extractor import RNNSentenceExtractor

import logging


class RNNModel(SummarizationModel):

    @staticmethod
    def update_command_line_options(parser):

        # Sentence Encoder Parameters
        parser.add_argument(
            "--sent-encoder", default="avg", choices=["cnn", "avg", "rnn"])
        parser.add_argument(
            "--sent-dropout", default=.25, type=float)
        parser.add_argument(
            "--sent-filter-windows", default=[1, 2, 3, 4, 5, 6], type=int, 
            nargs="+")
        parser.add_argument(
            "--sent-feature-maps", default=[25, 25, 50, 50, 50, 50], 
            type=int, nargs="+")
        parser.add_argument(
            "--sent-rnn-hidden-size", default=100, type=int)
        parser.add_argument(
            "--sent-rnn-bidirectional", action="store_true", default=False)

        # Document Encoder Parameters
        parser.add_argument(
            "--doc-rnn-hidden-size", default=100, type=int)
        parser.add_argument(
            "--doc-rnn-bidirectional", action="store_true", default=False)
        parser.add_argument(
            "--doc-rnn-dropout", default=.25, type=float)
        parser.add_argument(
            "--doc-rnn-layers", default=1, type=int)
    
        # MLP Parameters
        parser.add_argument(
            "--mlp-layers", default=[100], type=int, nargs="+")
        parser.add_argument(
            "--mlp-dropouts", default=[.25], type=float, nargs="+")

    @staticmethod
    def model_builder(embedding_context, 
                      sent_dropout=.25,
                      sent_encoder_type="avg",
                      sent_feature_maps=[25, 25, 25],
                      sent_filter_windows=[1, 2, 3],
                      sent_rnn_hidden_size=200,
                      sent_rnn_cell="gru",
                      sent_rnn_bidirectional=True,
                      sent_rnn_layers=1,
                      doc_rnn_cell="gru", 
                      doc_rnn_hidden_size=150, 
                      doc_rnn_bidirectional=False,
                      doc_rnn_dropout=.25,
                      doc_rnn_layers=1,
                      mlp_layers=[100], 
                      mlp_dropouts=[.25]):

        if len(sent_feature_maps) != len(sent_filter_windows):
            raise Exception(
                "sent_feature_maps and sent_filter_windows must have same "
                "number of arguments!")
    
        if len(mlp_layers) != len(mlp_dropouts):
            raise Exception(
                "mlp_layers and mlp_dropouts must have same number",
                "of arguments!")

        if sent_encoder_type == "avg":
             sent_enc = AveragingSentenceEncoder(
                 embedding_context.embedding_size, dropout=sent_dropout)
             logging.info(" Sentence Encoder: " + repr(sent_enc))
        elif sent_encoder_type == "cnn":
             sent_enc = CNNSentenceEncoder(
                 embedding_context.embedding_size,
                 feature_maps=sent_feature_maps, 
                 filter_windows=sent_filter_windows,
                 dropout=sent_dropout)
             logging.info(" Sentence Encoder: " + repr(sent_enc))
        elif sent_encoder_type == "rnn":
             sentence_encoder = SentenceRNNEncoder(
                 embedding_layer.size,
                 sent_rnn_hidden_size,
                 dropout=sent_dropout,
                 bidirectional=sent_rnn_bidirectional,
                 num_layers=sent_rnn_layers,
                 cell=sent_rnn_cell)
    
        else:
            raise Exception("sentence_encoder must be 'rnn', 'cnn', or 'avg'")
 
        sent_ext = RNNSentenceExtractor(
            sent_enc.size, 
            doc_rnn_hidden_size,
            num_layers=doc_rnn_layers, 
            cell=doc_rnn_cell, 
            rnn_dropout=doc_rnn_dropout,
            bidirectional=doc_rnn_bidirectional,
            mlp_layers=mlp_layers, 
            mlp_dropouts=mlp_dropouts)

        return RNNModel(embedding_context, sent_enc, sent_ext)
