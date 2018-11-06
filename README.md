# nnsum
An extractive neural network text summarization library for the EMNLP 2018 paper *Content Selection in Deep Learning Models of Summarization* (https://arxiv.org/abs/1810.12343).

- Data and preprocessing scripts are in a separate library (https://github.com/kedz/summarization-datasets). 
  If a dataset is publicly available the script will download it. 
The DUC and NYT datasets must be obtained separately before calling the preprocessing script.
  - To obtain the DUC 2001/2002 datasets: https://duc.nist.gov/data.html
  - To obtain the NYT dataset: https://catalog.ldc.upenn.edu/ldc2008t19
- Model implementation code is located in `nnsum`.
- Training and evaluation scripts are located in `script_bin`.

# Installation
1. Install pytorch using pip or conda.
2. run:
```bash 
git clone https://github.com/kedz/nnsum.git
cd nnsum
python setup.py install
```
3. Get the data: 
```bash
git clone https://github.com/kedz/summarization-datasets.git
cd summarization-datasets
python setup.py install
````
See README.md in summarization-datasets for details on how to get each dataset from the paper.

# Training A Model

All models from the paper can be trained from the same convenient training script: `script_bin/train_model.py`.
The general pattern for usage is:
```bash
python script_bin/train_model.py \
  --trainer TRAINER_ARGS --emb EMBEDDING_ARGS \
  --enc ENCODER_ARGS --ext EXTRACTOR_ARGS
```
Every model has a set of word embeddings, a sentence encoder, and a sentence extractor. 
Each argument section allows you to pick an architecture/options for that component.
For the most part, defaults match the paper's primary evaluation settings. 
For example, to train the CNN encoder with Seq2Seq extractor on gpu 0, run the following:

```
python script_bin/train_model.py \
    --trainer --train-inputs PATH/TO/INPUTS/TRAIN/DIR \
              --train-labels PATH/TO/LABELS/TRAIN/DIR \
              --valid-inputs PATH/TO/INPUTS/VALID/DIR \
              --valid-labels PATH/TO/LABELS/VALID/DIR \
              --valid-refs PATH/TO/HUMAN/REFERENCE/VALID/DIR \
              --weighted \
              --gpu 0 \
              --model PATH/TO/SAVE/MODEL \
              --results PATH/TO/SAVE/VALIDATION/SCORES \
              --seed 12345678 \
    --emb --pretrained-embeddings PATH/TO/200/DIM/GLOVE \
    --enc cnn \
    --ext s2s --bidirectional 
```

## Trainer Arguments
These arguments set the data to train on, batch sizes, training epochs, etc.
 - ```--train-inputs PATH``` Path to training inputs directory. (required)
 - ```--train-labels PATH``` Path to training labels directory. (required)
 - ```--valid-inputs PATH``` Path to validation inputs directory. (required)
 - ```--valid-labels PATH``` Path to validation labels directory. (required)
 - `--valid-refs PATH` Path to validation reference summaries directory. (required)
 - `--seed SEED` Random seed for getting repeatable results. 
 Due to randomness in cudnn drivers the CNN sentence encoder will still have randomness. 
 - `--epochs EPOCHS` Number of training epochs to run. (Default: 50)
 - `--batch-size BATCH_SIZE` Batch size for sgd. (Default: 32)
 - `--gpu GPU` Select gpu device. When set to -1, use cpu. (Default: -1)
 - `--teacher-forcing EPOCHS` Number of epochs to use teacher forcing. 
 After the EPOCHS training epoch, teacher forcing is disabled. Only effects the Cheng & Lapata extractor. (Default: 25)
 - `--sentence-limit LIMIT` Only load the first LIMIT sentences of each document. 
 Useful for saving memory/compute time when some of the data contains outlier documents with long length. (Default: 50)
 - `--weighted` When this flag is used, upweight positive labels to make them proportional to the negative labels.
 This is helpful because this is an inbalanced classification problem and we only want to select 3 or 4 sentences out of 30-50 sentences.
 - `--summary-length LENGTH` Set the maximum summary word length for ROUGE evaluation.
 - `--remove-stopwords` When flag is set, ROUGE is computed with stopwords removed.
 - `--model PATH` Location to save model. (Optional)
 - `--results PATH` Location to save results json. Stores per epoch training cross entropy, and per epoch validation cross-entropy, rouge-1, and rouge-2. (optional) 

## Embedding Arguments
These arguments set the word embeddings size, the path to pretrained embeddings, whether to fix the embeddings during learning, etc.

 - `--embedding-size SIZE` Dimenstion of word embeddings. Must match the size of pretrained embeddings if using pretrained. (Default: 200)
 - `--pretrained-embeddings PATH` Path to pretrained embeddings. Embeddings file should be a text file each line in the format: word e1 e2 e3...  where e1, e2, ... are the values of the embedding at that dimension. (optional)
 - `--top-k TOP_K` Only keep the TOP_K most frequent words in the embedding vocabulary where counts are based on the training dataset. Default keeps all words. Only active when not using pretrained embeddings. (Default: None)
 - `--at-least AT_LEAST` Keep only words that occur at least AT_LEAST times in the training dataset. Default keeps all words that occur at least once. Only active when not using pretrained embeddings. (Default: 1)
 - `--word-dropout PROB` Apply dropout to tokens in the input document with drop prob PROB. (Default: 0) 
 - `--embedding-dropout PROB` Apply dropout to the word embeddings with drop prob PROB. (default: .25)
 - `--update-rule RULE` Options are `update-all` and `fix-all`. `update-all` treats the embeddings as parameters to be learned during training. `fix-all` prevents word embeddings from being updated. (Default: `fix-all`)
 - `--filter-pretrained` When flag is set, apply `--top-k` and `--at-least` filters to pretrained embedding vocabulary.

## Encoder Arguments
The encoder arguments select for one of three sentence encoder architectures: `avg`, `rnn`, or `cnn`, and their various parameters e.g. `--enc avg` or `--enc cnn CNN_ARGUMENTS`. 
The sentence encoder takes a sentence, i.e. an arbitrarily long sequence of word embeddings and encodes them as a fixed length embedding. 
Below we describe the options for each architecture.
### Averaging Encoder
A sentence embedding is simply the average of the word embeddings.
 - `--dropout PROB`  Drop probability applied after averaging the word embeddings. (Default: .25)
### RNN Encoder
A sentence embedding is the final output state of an RNN applied to the word embedding sequence.
 - `--hidden-size SIZE` Size of the RNN output layer (and hidden if using LSTM cell). (Default: 200)
 - `--bidirectional` When flag is set, use a bi-directional RNN. Output is the concatenation of the final forward and backward outputs. 
 - `--dropout PROB` Drop propability applied to output layers of the RNN. (Default: .25)
 - `--num-layers NUM` Number layers in the RNN. (Default: 1)
 - `--cell CELL` RNN cell type to use. Options are `rnn`, `gru`, or `lstm`. (Default: `gru`)
### CNN Encoder
A sentence embedding is the output of a convolutional + relu + dropout layer. 
The output size is the sum of the `--feature-maps` argument.
 - `--dropout PROB`     Drop probability applied to convolutional layer output. (Default: .25)
 - `--filter-windows WIN [WIN ...]` Filter window widths, i.e. size in ngrams of each convolutional feature window. 
 Number of args must be the same length as `--feature-maps`. (Default: `1 2 3 4 5 6`)
 - `--feature-maps MAPS [MAPS ...]` Number of convolutional feature maps. Number of args must be the same length as `--filter-windows`. (Default: `25 25 50 50 50 50`)
             

## Extractor Arguments
The extractor arguments select for one of four sentence extractor architectures: `rnn`, `s2s`, `cl`, or `sr`, and their various parameters e.g. `--ext cl` or `--enc s2s S2S_ARGUMENTS`. 
The sentence extractor takes an arbitrarily long sequence of sentence embeddings and predicts whether each sentence should be included in the summary.
Below we describe the options for each architecture.
### RNN Extractor
Sentence embeddings are run through an RNN and then fed into a multi-layer perceptron MLP to predict sentence extraction.
 - `--hidden-size SIZE` Size of the RNN output layer/hidden layer. (Default: 300)
 - `--bidirectional` When flag is set, use a bi-directional RNN.
 - `--rnn-dropout PROB` Drop propability applied to output layers of the RNN. (Default: .25)
 - `--num-layers NUM` Number of layers in the RNN. (Default: 1)
 - `--cell CELL` RNN cell type to use. Options are `rnn`, `gru`, or `lstm`. (Default: `gru`)
 - `--mlp-layers SIZE [SIZE ...]` A list of sizes of the hidden layers in the MLP. Must be the same length as `--mlp-dropouts`. (Default: `100`)
 - `--mlp-dropouts PROB [PROBS ...]` A list the MLP hidden layer dropout probabilities. Must be the same length as `--mlp-layers`. (Default: .25)
### Seq2Seq Extractor
Sentence embeddings are run through a seq2seq based extractor with attention and MLP layer for predicting sentence extraction.
 - `--hidden-size SIZE` Size of the RNN output layer/hidden layer. (Default: 300)
 - `--bidirectional` When flag is set, use a bi-directional RNN.
 - `--rnn-dropout PROB` Drop propability applied to output layers of the RNN. (Default: .25)
 - `--num-layers NUM` Number of layers in the RNN. (Default: 1)
 - `--cell CELL` RNN cell type to use. Options are `rnn`, `gru`, or `lstm`. (Default: `gru`)
 - `--mlp-layers SIZE [SIZE ...]` A list of sizes of the hidden layers in the MLP. Must be the same length as `--mlp-dropouts`. (Default: `100`)
 - `--mlp-dropouts PROB [PROBS ...]` A list the MLP hidden layer dropout probabilities. Must be the same length as `--mlp-layers`. (Default: .25)
 
 ### Cheng & Lapata Extractor
 This is an implementation of the sentence extractive summarizer from: https://arxiv.org/abs/1603.07252
  - `--hidden-size SIZE` Size of the RNN output layer/hidden layer. (Default: 300)
 - `--bidirectional` When flag is set, use a bi-directional RNN.
 - `--rnn-dropout PROB` Drop propability applied to output layers of the RNN. (Default: .25)
 - `--num-layers NUM` Number of layers in the RNN. (Default: 1)
 - `--cell CELL` RNN cell type to use. Options are `rnn`, `gru`, or `lstm`. (Default: `gru`)
 - `--mlp-layers SIZE [SIZE ...]` A list of sizes of the hidden layers in the MLP. Must be the same length as `--mlp-dropouts`. (Default: `100`)
 - `--mlp-dropouts PROB [PROBS ...]` A list the MLP hidden layer dropout probabilities. Must be the same length as `--mlp-layers`. (Default: .25)
 
### SummaRunner Extractor
This is an implementation of the sentence extractive summarizer from: https://arxiv.org/abs/1611.04230
- `--hidden-size SIZE` Size of the RNN output layer/hidden layer. (Default: 300)
- `--bidirectional` When flag is set, use a bi-directional RNN.
- `--rnn-dropout PROB` Drop propability applied to output layers of the RNN. (Default: .25)
- `--num-layers NUM` Number of layers in the RNN. (Default: 1)
- `--cell CELL` RNN cell type to use. Options are `rnn`, `gru`, or `lstm`. (Default: `gru`)
- `--sentence-size SIZE` Dimension of sentence representation (after RNN layer) (Default: 100)
- `--document-size SIZE` Dimension of the document representation (Default: 100)
- `--segments SEG` Number of coarse position chunks, e.g. 4 segments means sentences in first quarter of the document would get the same segment embedding, the second quarter and so on. (Default: 4)
- `--max-position-weights NUM` The number of unique sentence position embeddings to use. The first NUM sentences will get unique sentence position embeddings. Documents longer than NUM will have the last sentence position embedding repeated. (Default: 50)
- `--segment-size SIZE` Dimension of segment embeddings. (Default: 16)
- `--position-size SIZE` Dimension of position embeddings. (Default: 16)

# Evaluating a Model
To get a model's ROUGE scores on the train, validation, or test set use `script_bin/eval_model.py`.

E.g.:
```
python eval_model.py \
  --inputs PATH/TO/INPUTS/DIR \
  --refs PATH/TO/REFERENCE/DIR \
  --model PATH/TO/MODEL \
  --results PATH/TO/WRITE/RESULTS \
  --summary-length 100 
```
Eval script parameters are described below:
 - `--batch-size BATCH_SIZE` Batch size to use when generating summaries. (Default: 32)
 - `--gpu GPU` Which gpu device to use. When GPU == -1, use cpu. (Default: -1)
 - `--sentence-limit LIMIT` Only read in the first LIMIT sentences in the document to be summarized. By default there is no limit. (Default: None)
 - `--summary-length LENGTH` The summary word length to use in ROUGE evauation. (Default: 100)
 - `--remove-stopwords` When flag is true ROUGE evaluation is done with stopwords removed.
 - `--inputs PATH` Path to input data directory. (required)
 - `--refs PATH` Path to human reference summary directory. (required) 
 - `--model PATH` Path to saved model to evaluate. (required)
 - `--results PATH` Path to write results json. Stores per document ROUGE scores and dataset average scores. (optional)
