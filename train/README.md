## Overview
This folder includes pipelines and datasets for training our meta learner, T-AutoOD. We ran more than 200K experiments, each experiment includes a pair of pipeline and dataset and gives one column in one of the 520+ datasets.

## Pipelines
"pipelines" folder includes 400+ sklearn compatible pipelines in joblib format.

## Datasets
"data" folder includes 520+ datasets in .gzip format although the extension is .csv. All 520+ .csv gzip files have the same headers (or column names). You can load them with Pandas and concatenate them. To load with Pandas, use "compression='zip'", e.g., df = pd.read_csv('yprop_4_1_44039_rg_col0_50_c0.01.csv', compression='gzip'). Note that, the meta learner in the paper was trained with original numeric/float64 data values. However, the total amount of space for 520 files is too large for us to upload these files in float64 to github. So, we have to cast the numeric/float64 to float16. As a result, the uploaded gzip data files have values in float16.

"test_datasets_by_dsname.csv" includes list of datasets used for testing while "train_datasets_by_dsname.csv" includes a list of datasets for training. 

## Test with External benchmarks
The  whole set of 520+ datasets can be used for training the meta learner. In this case, you can test your meta learner against the benchmark data set in the benchmark_datasets.tar.gz in the main folder.
