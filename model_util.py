from sentence_averaging_encoder import SentenceAveragingEncoder
from sentence_cnn_encoder import SentenceCNNEncoder
from sentence_rnn_encoder import SentenceRNNEncoder
from document_rnn_encoder import DocumentRNNEncoder
from simple_sentence_extractor import SimpleSentenceExtractor
from rnn_sentence_extractor import RNNSentenceExtractor
from chen_and_lapata_sentence_extractor import ChenAndLapataSentenceExtractor
from summarization_model import SummarizationModel


def chen_and_lapata_sent_model(embedding_layer, feature_maps=[50, 50, 50], 
                               filter_windows=[1, 2, 3], sentence_dropout=.25,
                               sentence_encoder="cnn",
                               sentence_rnn_hidden_size=200,
                               sentence_extractor="c&l",
                               attention=None, 
                               rnn_cell="GRU", rnn_hidden_size=150, 
                               rnn_encoder_bidirectional=False,
                               rnn_dropout=.25, rnn_layers=1,
                               mlp_layers=[100], mlp_dropouts=[.25]):

    if sentence_encoder == "cnn":
        sent_encoder = SentenceCNNEncoder(
             embedding_layer.size,
             feature_maps=feature_maps, 
             filter_windows=filter_windows,
             dropout=sentence_dropout)
    elif sentence_encoder == "avg":
         sent_encoder = SentenceAveragingEncoder(
             embedding_layer.size,
             dropout=sentence_dropout)
    elif sentence_encoder == "rnn":
         sent_encoder = SentenceRNNEncoder(
             embedding_layer.size,
             sentence_rnn_hidden_size,
             dropout=sentence_dropout,
             cell=rnn_cell)
    else:
        raise Exception("sentence_encoder must be 'cnn' or 'avg'")
        
    doc_encoder = DocumentRNNEncoder(
        sent_encoder.size, rnn_hidden_size,
        num_layers=rnn_layers, cell=rnn_cell, dropout=rnn_dropout,
        bidirectional=rnn_encoder_bidirectional)
   
    if sentence_extractor == "c&l": 
        doc_decoder = ChenAndLapataSentenceExtractor(
            sent_encoder.size, rnn_hidden_size,
            num_layers=rnn_layers, cell=rnn_cell, rnn_dropout=rnn_dropout,
            mlp_layers=mlp_layers, mlp_dropouts=mlp_dropouts)
    elif sentence_extractor == "simple":
         doc_decoder = SimpleSentenceExtractor(
            doc_encoder.size, 
            mlp_layers=mlp_layers, mlp_dropouts=mlp_dropouts)
    elif sentence_extractor == "rnn":
        doc_decoder = RNNSentenceExtractor(
            sent_encoder.size, rnn_hidden_size,
            num_layers=rnn_layers, cell=rnn_cell, rnn_dropout=rnn_dropout,
            mlp_layers=mlp_layers, mlp_dropouts=mlp_dropouts,
            attention=attention)
    else:   
        raise Exception("sentence_extractor must be 'c&l', 'simple', 'rnn'.")

    return SummarizationModel(
        embedding_layer, sent_encoder, doc_encoder, doc_decoder)
