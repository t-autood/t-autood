import glob
import joblib
import traceback
import numpy as np
import pandas as pd
from typing import List
from statistics import mean, median
from sklearn.metrics import roc_auc_score
from os.path import join, realpath, dirname, basename

def compute_roc_stats(
                        datafiles: List = None,
                        pipeline_names: List = None,
                        model = None
                    ):

    assert model is not None
    assert datafiles is not None
    assert pipeline_names is not None
     
    roc_score_lst = list()
    for ds_file_name in datafiles:
        try:
            te_df = pd.read_csv(ds_file_name, compression='gzip')
        except Exception as e:
            te_df = pd.read_csv(ds_file_name)

        try:
            y_test = te_df["y_true"]
            te_df.drop(columns=["y_true"], axis=1, inplace=True)
            X_test = te_df[pipeline_names].values
            y_score = model.predict_proba(X_test)[:, 1]                    
            roc_score = roc_auc_score(y_test, y_score)
            roc_score_lst.append(roc_score)
            print(f"Data set: {basename(ds_file_name)}: roc: {roc_score}")
        except Exception as e:
            roc_score_lst.append(np.nan)
            print(f"{traceback.format_exc()}, data set: {ds_file_name}")

    return roc_score_lst

if __name__ == "__main__":
    curr_dir = dirname(realpath(__file__))
    meta_model = joblib.load(join(curr_dir, "model_s23283215.pkl"))
    
    ppname_filepath = join(curr_dir, "top_pipeline_names.csv")
    df_ppname = pd.read_csv(ppname_filepath)
    pipeline_names = list(df_ppname["pipeline_name"].values) 

    datadir = join(join(curr_dir, "benchmark_datasets"), "*.csv")
    benchmark_datafiles = glob.glob(datadir)
    roc_score_list = compute_roc_stats(
                            datafiles=benchmark_datafiles, 
                            pipeline_names=pipeline_names, 
                            model=meta_model,
                        )
    print(f"mean: {mean(roc_score_list)}, median: {median(roc_score_list)}, max: {max(roc_score_list)}, min_:{min(roc_score_list)}")
    print("Done")