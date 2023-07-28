# t-autood

1. Clone the repo to local machine: git clone https://github.com/t-autood/t-autood.git  - should see the "t-autood" sub-folder created
2. cd t-autood
3. tar xzvf benchmark_datasets.tar.gz  - should see the "benchmark_datasets" sub-folder created
4. tar xzvf pipelines.tar.gz  - should see the "pipelines" sub-folder created
5. model_s23283215.pkl is the pretrained meta learner, top_pipeline_names.csv includes selected pipelines stored in "pipelines" sub-folder
6. To run the meta learner on pre-scored benchmark data sets stored in "benchmark_datasets" sub-folder: python benchmark.py
7. To score a new data set cardio_odds.csv with the pretraind meta learner: python score_new_dataset.py
