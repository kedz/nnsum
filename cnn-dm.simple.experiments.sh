DATA=$1/cnn-dailymail
GPU=$2
GLOVE=$1/glove

INPUTS_TRAIN=$DATA/inputs/cnn.dm.spacy.input.train.json
INPUTS_VALID=$DATA/inputs/cnn.dm.spacy.input.valid.json
INPUTS_TEST=$DATA/inputs/cnn.dm.spacy.input.test.json

LABELS_TRAIN=$DATA/labels/cnn.dm.lim50.labels.train.json
LABELS_VALID=$DATA/labels/cnn.dm.lim50.labels.valid.json
LABELS_TEST=$DATA/labels/cnn.dm.lim50.labels.test.json

SUMS_VALID=$DATA/human-abstracts/valid

VOCABDIR=${DATA}/vocab
RESULTSDIR=${DATA}/results
MODELSDIR=${DATA}/models



SEEDS="3423452 8747842 2347283 7234821 5247881"
EMB_SIZES="50" # 100 200 300"
EMB_SIZES="200"

for SEED in $SEEDS
do
  for EMB_SIZE in $EMB_SIZES
  do

    echo "$SEED $EMB_SIZE"
    VOCAB="${VOCABDIR}/${SEED}.${EMB_SIZE}d.fixed.avg.simple.vocab.json"
    RESULTS="${RESULTSDIR}/${SEED}.${EMB_SIZE}d.fixed.avg.simple.results.json"
    MODEL="${MODELSDIR}/${SEED}.${EMB_SIZE}d.fixed.avg.simple.bin"
    python train_simple_model.py \
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
      --epochs 20 \
      --sent-limit 50 \
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
 
    echo "$SEED $EMB_SIZE"
    VOCAB="${VOCABDIR}/${SEED}.${EMB_SIZE}d.learned.avg.simple.vocab.json"
    RESULTS="${RESULTSDIR}/${SEED}.${EMB_SIZE}d.learned.avg.simple.results.json"
    MODEL="${MODELSDIR}/${SEED}.${EMB_SIZE}d.learned.avg.simple.bin"
    python train_simple_model.py \
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
      --epochs 20 \
      --sent-limit 50 \
      --pretrained-embeddings ${GLOVE}/glove.6B.${EMB_SIZE}d.txt \
      --embedding-dropout .25 \
      --embedding-size ${EMB_SIZE} \
      --at-least 10 \
      --sent-encoder avg \
      --sent-dropout .25 \
      --doc-rnn-hidden-size 300 \
      --doc-rnn-bidirectional \
      --doc-rnn-dropout .25 \
      --mlp-layers 100 \
      --mlp-dropouts .25 \
      --seed $SEED 
 


  done
done

for SEED in $SEEDS
do
  for EMB_SIZE in $EMB_SIZES
  do
    echo "$SEED $EMB_SIZE"
    VOCAB="${VOCABDIR}/${SEED}.${EMB_SIZE}d.fixed.cnn.simple.vocab.json"
    RESULTS="${RESULTSDIR}/${SEED}.${EMB_SIZE}d.fixed.cnn.simple.results.json"
    MODEL="${MODELSDIR}/${SEED}.${EMB_SIZE}d.fixed.cnn.simple.bin"
    python train_simple_model.py \
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
      --epochs 20 \
      --sent-limit 50 \
      --pretrained-embeddings ${GLOVE}/glove.6B.${EMB_SIZE}d.txt \
      --fix-embeddings \
      --embedding-dropout .25 \
      --embedding-size ${EMB_SIZE} \
      --sent-encoder cnn \
      --sent-dropout .25 \
      --sent-filter-windows 1 2 3 4 5 6 \
      --sent-feature-maps 25 25 50 50 50 50 \
      --doc-rnn-hidden-size 300 \
      --doc-rnn-bidirectional \
      --doc-rnn-dropout .25 \
      --mlp-layers 100 \
      --mlp-dropouts .25 \
      --seed $SEED 
 
    echo "$SEED $EMB_SIZE"
    VOCAB="${VOCABDIR}/${SEED}.${EMB_SIZE}d.learned.cnn.simple.vocab.json"
    RESULTS="${RESULTSDIR}/${SEED}.${EMB_SIZE}d.learned.cnn.simple.results.json"
    MODEL="${MODELSDIR}/${SEED}.${EMB_SIZE}d.learned.cnn.simple.bin"
    python train_simple_model.py \
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
      --epochs 20 \
      --sent-limit 50 \
      --pretrained-embeddings ${GLOVE}/glove.6B.${EMB_SIZE}d.txt \
      --embedding-dropout .25 \
      --embedding-size ${EMB_SIZE} \
      --at-least 10 \
      --sent-encoder cnn \
      --sent-dropout .25 \
      --sent-filter-windows 1 2 3 4 5 6 \
      --sent-feature-maps 25 25 50 50 50 50 \
      --doc-rnn-hidden-size 300 \
      --doc-rnn-bidirectional \
      --doc-rnn-dropout .25 \
      --mlp-layers 100 \
      --mlp-dropouts .25 \
      --seed $SEED 
 
  done
done
