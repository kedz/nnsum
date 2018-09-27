DATA=$1/duc-sds
GPU=$2
GLOVE=$1/glove

INPUTS_TRAIN=$DATA/inputs/duc-sds.inputs.train.json
INPUTS_VALID=$DATA/inputs/duc-sds.inputs.valid.json
INPUTS_TEST=$DATA/inputs/duc-sds.inputs.test.json

LABELS_TRAIN=$DATA/labels/duc-sds.labels.train.json
LABELS_VALID=$DATA/labels/duc-sds.labels.valid.json
LABELS_TEST=$DATA/labels/duc-sds.labels.test.json

SUMS_VALID=$DATA/human-abstracts/valid

VOCABDIR=${DATA}/vocab
RESULTSDIR=${DATA}/results
MODELSDIR=${DATA}/models



SEEDS="3423452 8747842 2347283 7234821 5247881"
SEED="3423452"
EMB_SIZES="50 100 200 300"
EMB_SIZE="200"

    echo "$SEED $EMB_SIZE"
    VOCAB="${VOCABDIR}/${SEED}.${EMB_SIZE}d.fixed.avg.s2s.vocab.json"
    RESULTS="${RESULTSDIR}/${SEED}.${EMB_SIZE}d.fixed.avg.s2s.results.json"
    MODEL="${MODELSDIR}/${SEED}.${EMB_SIZE}d.fixed.avg.s2s.bin"
    python train_seq2seq_model.py \
      --train-inputs $INPUTS_TRAIN \
      --train-labels $LABELS_TRAIN \
      --valid-inputs $INPUTS_VALID \
      --valid-labels $LABELS_VALID \
      --valid-summary-dir $SUMS_VALID \
      --vocab ${VOCAB} \
      --results-path ${RESULTS} \
      --model-path ${MODEL} \
      --weighted \
      --batch-size 32 \
      --gpu $GPU \
      --epochs 50 \
      --pretrained-embeddings ${GLOVE}/glove.6B.${EMB_SIZE}d.txt \
      --fix-embeddings \
      --embedding-dropout .25 \
      --embedding-size ${EMB_SIZE} \
      --sent-encoder avg \
      --sent-dropout .25 \
      --doc-rnn-hidden-size 300 \
      --doc-rnn-bidirectional \
      --doc-rnn-dropout .25 \
      --mlp-layers 100 \
      --mlp-dropouts .25 \
      --seed $SEED 
 
