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
EMB_SIZES="50 100 200 300"
EMB_SIZES="200"

for SEED in $SEEDS
do
  for EMB_SIZE in $EMB_SIZES
  do

    echo "$SEED $EMB_SIZE"
    VOCAB="${VOCABDIR}/${SEED}.${EMB_SIZE}d.fixed.cnn.c&l.vocab.json"
    RESULTS="${RESULTSDIR}/${SEED}.${EMB_SIZE}d.fixed.cnn.c&l.results.json"
    MODEL="${MODELSDIR}/${SEED}.${EMB_SIZE}d.fixed.cnn.c&l.bin"
    python "train_cheng&lapata_model.py" \
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
      --teacher-forcing 25 \
      --pretrained-embeddings ${GLOVE}/glove.6B.${EMB_SIZE}d.txt \
      --fix-embeddings \
      --embedding-dropout .25 \
      --embedding-size ${EMB_SIZE} \
      --sent-encoder cnn \
      --sent-dropout .25 \
      --sent-filter-windows 1 2 3 4 5 6 \
      --sent-feature-maps 25 25 50 50 50 50 \
      --doc-rnn-hidden-size 300 \
      --doc-rnn-dropout .25 \
      --mlp-layers 100 \
      --mlp-dropouts .25 \
      --seed $SEED 
 
    echo "$SEED $EMB_SIZE"
    VOCAB="${VOCABDIR}/${SEED}.${EMB_SIZE}d.learned.cnn.c&l.vocab.json"
    RESULTS="${RESULTSDIR}/${SEED}.${EMB_SIZE}d.learned.cnn.c&l.results.json"
    MODEL="${MODELSDIR}/${SEED}.${EMB_SIZE}d.learned.cnn.c&l.bin"
    python "train_cheng&lapata_model.py" \
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
      --teacher-forcing 25 \
      --pretrained-embeddings ${GLOVE}/glove.6B.${EMB_SIZE}d.txt \
      --embedding-dropout .25 \
      --embedding-size ${EMB_SIZE} \
      --at-least 10 \
      --sent-encoder cnn \
      --sent-dropout .25 \
      --sent-filter-windows 1 2 3 4 5 6 \
      --sent-feature-maps 25 25 50 50 50 50 \
      --doc-rnn-hidden-size 300 \
      --doc-rnn-dropout .25 \
      --mlp-layers 100 \
      --mlp-dropouts .25 \
      --seed $SEED 
 
    echo "$SEED $EMB_SIZE"
    VOCAB="${VOCABDIR}/${SEED}.${EMB_SIZE}d.fixed.avg.c&l.vocab.json"
    RESULTS="${RESULTSDIR}/${SEED}.${EMB_SIZE}d.fixed.avg.c&l.results.json"
    MODEL="${MODELSDIR}/${SEED}.${EMB_SIZE}d.fixed.avg.c&l.bin"
    python "train_cheng&lapata_model.py" \
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
      --teacher-forcing 25 \
      --pretrained-embeddings ${GLOVE}/glove.6B.${EMB_SIZE}d.txt \
      --fix-embeddings \
      --embedding-dropout .25 \
      --embedding-size ${EMB_SIZE} \
      --sent-encoder avg \
      --sent-dropout .25 \
      --doc-rnn-hidden-size 300 \
      --doc-rnn-dropout .25 \
      --mlp-layers 100 \
      --mlp-dropouts .25 \
      --seed $SEED 

    echo "$SEED $EMB_SIZE"
    VOCAB="${VOCABDIR}/${SEED}.${EMB_SIZE}d.learned.avg.c&l.vocab.json"
    RESULTS="${RESULTSDIR}/${SEED}.${EMB_SIZE}d.learned.avg.c&l.results.json"
    MODEL="${MODELSDIR}/${SEED}.${EMB_SIZE}d.learned.avg.c&l.bin"
    python "train_cheng&lapata_model.py" \
      --train-inputs $INPUTS_TRAIN \
      --train-labels $LABELS_TRAIN \
      --valid-inputs $INPUTS_VALID \
      --valid-labels $LABELS_VALID \
      --valid-summary-dir $SUMS_VALID \
      --vocab ${VOCAB} \
      --results-path ${RESULTS_PATH} \
      --model-path ${MODEL_PATH} \
      --weighted \
      --batch-size 32 \
      --gpu $GPU \
      --epochs 50 \
      --teacher-forcing 25 \
      --pretrained-embeddings ${GLOVE}/glove.6B.${EMB_SIZE}d.txt \
      --embedding-dropout .25 \
      --embedding-size ${EMB_SIZE} \
      --at-least 10 \
      --sent-encoder avg \
      --sent-dropout .25 \
      --doc-rnn-hidden-size 300 \
      --doc-rnn-dropout .25 \
      --mlp-layers 100 \
      --mlp-dropouts .25 \
      --seed $SEED 
 
  done
done

