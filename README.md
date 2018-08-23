# nnsum
An extractive neural network text summarization library for the EMNLP 2018 paper ''Content Selection in Deep Learning Models of Summarization."

Data and preprocessing code. If a dataset is publicly available the script will download it. 
The DUC and NYT datasets must be obtained separately before calling the preprocessing script.
To obtain the DUC 2001/2002 datasets: https://duc.nist.gov/data.html
To obtain the NYT dataset: https://catalog.ldc.upenn.edu/ldc2008t19
Model implementation code is located python/nnsum.
Training and evaluation scripts are located in script_bin.
Experiment settings/bash scripts for each table in the paper are located in experiment_scripts.

