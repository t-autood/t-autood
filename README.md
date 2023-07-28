# t-autood
This repo includes a pretrained meta learner and code that loads the meta learner to score different data sets.

1. Clone the repo to local machine: git clone https://github.com/t-autood/t-autood.git     - after this command, the "t-autood" sub-folder is created.
2. cd t-autood
3. tar xzvf pipelines.tar.gz    - after this command, the "pipelines" sub-folder is created. This folder includes 20 joblib files, each file includes a sklearn compatible pipeline. These 20 pipelines convert raw data into a new representation consumed by the meta learner.
4. tar xzvf benchmark_datasets.tar.gz    - after this command, the "benchmark_datasets" sub-folder is created. This folder includes 47 data sets - note that two files are removed since they are larger than 25MB - the limit of github.com. Each data file contains pre-computed outlier scores by 20 pipelines in "pipelines" sub-folder.
5. File model_s23283215.pkl is the pretrained LGBM meta learner, file top_pipeline_names.csv includes names of selected pipelines stored in "pipelines" sub-folder. These names are used by benchmark.py to select columns in data set files, preparing for meta learner's prediction.
6. To run the meta learner on pre-scored benchmark data sets in "benchmark_datasets" sub-folder: python benchmark.py  
7. To score a new data set cardio_odds.csv with the pretraind meta learner: python score_new_dataset.py  . This script uses 20 pipelines to fit and score the raw data set, converting it to the new representation consumed by the meta learner for detecting outliers in cardio_odds.csv data set.
