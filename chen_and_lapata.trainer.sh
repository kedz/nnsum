DATA=/storage/data/kedzie-summarization-data/cnn-dailymail
GLOVE=/storage/data/kedzie-summarization-data/glove

PRETRAINED_EMBEDDINGS=${GLOVE}/glove.6B.200d.txt
CDTRAIN_INPUTS=${DATA}/inputs/cnn.dailymail.train.tokens.json
CDVALID_INPUTS=${DATA}/inputs/cnn.dailymail.valid.tokens.json
CDTRAIN_LABELS=${DATA}/labels/cnn.dailymail.train.json
CDVALID_LABELS=${DATA}/labels/cnn.dailymail.valid.json
VALID_SUMMARY_DIR=${DATA}/human-abstracts/valid



VOCAB="${DATA}/vocab/tmp.vocab.json"
RESULTS_PATH="${DATA}/results/tmp.results.json"
MODEL_PATH="${DATA}/models/tmp.model.bin"

python train_hierarchical_model.py \
    --train-inputs $CDVALID_INPUTS \
    --train-labels $CDVALID_LABELS \
    --valid-inputs $CDVALID_INPUTS \
    --valid-labels $CDVALID_LABELS \
    --valid-summary-dir $VALID_SUMMARY_DIR \
    --vocab ${VOCAB} \
    --results-path ${RESULTS_PATH} \
    --model-path ${MODEL_PATH} \
    --batch-size 3 \
    --embedding-size 4 \
    --sentence-dropout .25 \
    --rnn-hidden-size 300 \
    --rnn-dropout .25 \
    --mlp-layers 300 \
    --mlp-dropouts .25 \
    --sentence-encoder rnn \
    --sentence-rnn-hidden-size 2 \
    --sentence-extractor simple \
    --rnn-encoder-bidirectional \
    --attention dot \
    --weighted --gpu 0 --epochs 10


    #--pretrained-embeddings ${PRETRAINED_EMBEDDINGS} \
    #--fix-embeddings \
exit


VOCAB="${DATA}/vocab/gru.seq2seq.dot.avg.fixed.v1.vocab.json"
RESULTS_PATH="${DATA}/results/gru.seq2seq.dot.avg.fixed.v1.results.json"
MODEL_PATH="${DATA}/models/gru.seq2seq.dot.avg.fixed.v1.model.bin"

python train_hierarchical_model.py \
    --train-inputs $CDTRAIN_INPUTS \
    --train-labels $CDTRAIN_LABELS \
    --valid-inputs $CDVALID_INPUTS \
    --valid-labels $CDVALID_LABELS \
    --valid-summary-dir $VALID_SUMMARY_DIR \
    --vocab ${VOCAB} \
    --results-path ${RESULTS_PATH} \
    --model-path ${MODEL_PATH} \
    --batch-size 8 \
    --embedding-size 200 \
    --pretrained-embeddings ${PRETRAINED_EMBEDDINGS} \
    --fix-embeddings \
    --sentence-dropout .25 \
    --rnn-hidden-size 300 \
    --rnn-dropout .25 \
    --mlp-layers 300 \
    --mlp-dropouts .25 \
    --sentence-encoder avg \
    --sentence-extractor rnn \
    --attention dot \
    --weighted --gpu 0 --epochs 10



VOCAB="${DATA}/vocab/gru.seq2seq.avg.fixed.v1.vocab.json"
RESULTS_PATH="${DATA}/results/gru.seq2seq.avg.fixed.v1.results.json"
MODEL_PATH="${DATA}/models/gru.seq2seq.avg.fixed.v1.model.bin"

python train_hierarchical_model.py \
    --train-inputs $CDTRAIN_INPUTS \
    --train-labels $CDTRAIN_LABELS \
    --valid-inputs $CDVALID_INPUTS \
    --valid-labels $CDVALID_LABELS \
    --valid-summary-dir $VALID_SUMMARY_DIR \
    --vocab ${VOCAB} \
    --results-path ${RESULTS_PATH} \
    --model-path ${MODEL_PATH} \
    --batch-size 8 \
    --embedding-size 200 \
    --pretrained-embeddings ${PRETRAINED_EMBEDDINGS} \
    --fix-embeddings \
    --sentence-dropout .25 \
    --rnn-hidden-size 300 \
    --rnn-dropout .25 \
    --mlp-layers 300 \
    --mlp-dropouts .25 \
    --sentence-encoder avg \
    --sentence-extractor rnn \
    --weighted --gpu 0 --epochs 10

exit

VOCAB="${DATA}/vocab/bigru.avg.fixed.v1.vocab.json"
RESULTS_PATH="${DATA}/results/bigru.avg.fixed.v1.results.json"
MODEL_PATH="${DATA}/models/bigru.avg.fixed.v1.model.bin"

python train_hierarchical_model.py \
    --train-inputs $CDTRAIN_INPUTS \
    --train-labels $CDTRAIN_LABELS \
    --valid-inputs $CDVALID_INPUTS \
    --valid-labels $CDVALID_LABELS \
    --valid-summary-dir $VALID_SUMMARY_DIR \
    --vocab ${VOCAB} \
    --results-path ${RESULTS_PATH} \
    --model-path ${MODEL_PATH} \
    --batch-size 8 \
    --embedding-size 200 \
    --pretrained-embeddings ${PRETRAINED_EMBEDDINGS} \
    --fix-embeddings \
    --sentence-dropout .25 \
    --rnn-hidden-size 300 \
    --rnn-dropout .25 \
    --mlp-layers 300 \
    --mlp-dropouts .25 \
    --sentence-encoder avg \
    --sentence-extractor simple \
    --rnn-encoder-bidirectional \
    --weighted --gpu 0 --epochs 10



VOCAB="${DATA}/vocab/bigru.avg.learned.v1.vocab.json"
RESULTS_PATH="${DATA}/results/bigru.avg.learned.v1.results.json"
MODEL_PATH="${DATA}/models/bigru.avg.learned.v1.model.bin"

python train_hierarchical_model.py \
    --train-inputs $CDTRAIN_INPUTS \
    --train-labels $CDTRAIN_LABELS \
    --valid-inputs $CDVALID_INPUTS \
    --valid-labels $CDVALID_LABELS \
    --valid-summary-dir $VALID_SUMMARY_DIR \
    --vocab ${VOCAB} \
    --results-path ${RESULTS_PATH} \
    --model-path ${MODEL_PATH} \
    --batch-size 8 \
    --embedding-size 200 \
    --pretrained-embeddings ${PRETRAINED_EMBEDDINGS} \
    --sentence-dropout .25 \
    --rnn-hidden-size 300 \
    --rnn-dropout .25 \
    --mlp-layers 300 \
    --mlp-dropouts .25 \
    --sentence-encoder avg \
    --sentence-extractor simple \
    --rnn-encoder-bidirectional \
    --weighted --gpu 0 --epochs 10

exit


VOCAB="${DATA}/vocab/c&l.avg.fixed.v1.vocab.json"
RESULTS_PATH="${DATA}/results/c&l.avg.fixed.v1.results.json"
MODEL_PATH="${DATA}/models/c&l.avg.fixed.v1.model.bin"

python train_hierarchical_model.py \
    --train-inputs $CDTRAIN_INPUTS \
    --train-labels $CDTRAIN_LABELS \
    --valid-inputs $CDVALID_INPUTS \
    --valid-labels $CDVALID_LABELS \
    --valid-summary-dir $VALID_SUMMARY_DIR \
    --vocab ${VOCAB} \
    --results-path ${RESULTS_PATH} \
    --model-path ${MODEL_PATH} \
    --batch-size 8 \
    --embedding-size 200 \
    --filter-windows 1 2 3 4 5 6 7 \
    --feature-maps 25 25 25 50 50 50 50 \
    --sentence-dropout .25 \
    --rnn-hidden-size 300 \
    --rnn-dropout .25 \
    --mlp-layers 300 \
    --mlp-dropouts .25 \
    --pretrained-embeddings ${PRETRAINED_EMBEDDINGS} \
    --fix-embeddings \
    --sentence-encoder avg \
    --weighted --gpu 0 --epochs 10

VOCAB="${DATA}/vocab/c&l.avg.learned.v1.vocab.json"
RESULTS_PATH="${DATA}/results/c&l.avg.learned.v1.results.json"
MODEL_PATH="${DATA}/models/c&l.avg.learned.v1.model.bin"

python train_hierarchical_model.py \
    --train-inputs $CDTRAIN_INPUTS \
    --train-labels $CDTRAIN_LABELS \
    --valid-inputs $CDVALID_INPUTS \
    --valid-labels $CDVALID_LABELS \
    --valid-summary-dir $VALID_SUMMARY_DIR \
    --vocab ${VOCAB} \
    --results-path ${RESULTS_PATH} \
    --model-path ${MODEL_PATH} \
    --batch-size 8 \
    --embedding-size 200 \
    --filter-windows 1 2 3 4 5 6 7 \
    --feature-maps 25 25 25 50 50 50 50 \
    --sentence-dropout .25 \
    --rnn-hidden-size 300 \
    --rnn-dropout .25 \
    --mlp-layers 300 \
    --mlp-dropouts .25 \
    --pretrained-embeddings ${PRETRAINED_EMBEDDINGS} \
    --sentence-encoder avg \
    --weighted --gpu 0 --epochs 10
