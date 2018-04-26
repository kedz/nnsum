from sentence_averaging_encoder import SentenceAveragingEncoder
from sentence_cnn_encoder import SentenceCNNEncoder
from sentence_rnn_encoder import SentenceRNNEncoder
from document_rnn_encoder import DocumentRNNEncoder
from simple_sentence_extractor import SimpleSentenceExtractor
from rnn_sentence_extractor import RNNSentenceExtractor
from cheng_and_lapata_sentence_extractor import ChengAndLapataSentenceExtractor
from summarunner_sentence_extractor import SummaRunnerSentenceExtractor
from summarization_model import SummarizationModel


def simple_extractor_model(embedding_layer, 
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


    if sent_encoder_type == "avg":
         sentence_encoder = SentenceAveragingEncoder(
             embedding_layer.size,
             dropout=sent_dropout)
    elif sent_encoder_type == "cnn":
        sentence_encoder = SentenceCNNEncoder(
             embedding_layer.size,
             feature_maps=sent_feature_maps, 
             filter_windows=sent_filter_windows,
             dropout=sent_dropout)
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
 
    document_encoder = DocumentRNNEncoder(
        sentence_encoder.size, 
        doc_rnn_hidden_size,
        num_layers=doc_rnn_layers, 
        cell=doc_rnn_cell, 
        dropout=doc_rnn_dropout,
        bidirectional=doc_rnn_bidirectional)

    document_decoder = SimpleSentenceExtractor(
        document_encoder.size, 
        mlp_layers=mlp_layers, 
        mlp_dropouts=mlp_dropouts)

    return SummarizationModel(
        embedding_layer, sentence_encoder, document_encoder, document_decoder)


def cheng_and_lapata_extractor_model(embedding_layer, 
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

    if sent_encoder_type == "avg":
         sentence_encoder = SentenceAveragingEncoder(
             embedding_layer.size,
             dropout=sent_dropout)
    elif sent_encoder_type == "cnn":
        sentence_encoder = SentenceCNNEncoder(
             embedding_layer.size,
             feature_maps=sent_feature_maps, 
             filter_windows=sent_filter_windows,
             dropout=sent_dropout)
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
 
    document_encoder = DocumentRNNEncoder(
        sentence_encoder.size, 
        doc_rnn_hidden_size,
        num_layers=doc_rnn_layers, 
        cell=doc_rnn_cell, 
        dropout=doc_rnn_dropout,
        bidirectional=doc_rnn_bidirectional)

    document_decoder = ChengAndLapataSentenceExtractor(
            sentence_encoder.size, doc_rnn_hidden_size,
            num_layers=doc_rnn_layers, cell=doc_rnn_cell, 
            rnn_dropout=doc_rnn_dropout,
            mlp_layers=mlp_layers, mlp_dropouts=mlp_dropouts)

    return SummarizationModel(
        embedding_layer, sentence_encoder, document_encoder, document_decoder)

def summarunner_extractor_model(embedding_layer, 
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
                                dec_sent_size=300,
                                dec_doc_size=400,
                                dec_num_positions=50,
                                dec_num_segments=4,
                                dec_position_size=50,
                                dec_segment_size=50):

    if sent_encoder_type == "avg":
         sentence_encoder = SentenceAveragingEncoder(
             embedding_layer.size,
             dropout=sent_dropout)
    elif sent_encoder_type == "cnn":
        sentence_encoder = SentenceCNNEncoder(
             embedding_layer.size,
             feature_maps=sent_feature_maps, 
             filter_windows=sent_filter_windows,
             dropout=sent_dropout)
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
 
    document_encoder = DocumentRNNEncoder(
        sentence_encoder.size, 
        doc_rnn_hidden_size,
        num_layers=doc_rnn_layers, 
        cell=doc_rnn_cell, 
        dropout=doc_rnn_dropout,
        bidirectional=doc_rnn_bidirectional)

    document_decoder = SummaRunnerSentenceExtractor(
            document_encoder.size, sent_size=dec_segment_size,
            doc_size=dec_doc_size, num_positions=dec_num_positions,
            num_segments=dec_num_segments, position_size=dec_position_size,
            segment_size=dec_segment_size)

    return SummarizationModel(
        embedding_layer, sentence_encoder, document_encoder, document_decoder)
