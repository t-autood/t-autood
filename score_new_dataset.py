import joblib
import pandas as pd
from os.path import join, realpath, dirname
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

if __name__ == "__main__":
    curr_dir = dirname(realpath(__file__))
    meta_model = joblib.load(join(curr_dir, "model_s23283215.pkl"))
    dataset_filepath = join(curr_dir, "cardio_odds.csv")
     
    df = pd.read_csv(dataset_filepath)
    if "label" in df.columns:
        df.drop("label", axis=1, inplace=True)

    X = df.values
    ppname_filepath = join(curr_dir, "top_pipeline_names.csv")
    df_ppname = pd.read_csv(ppname_filepath)
    pipeline_names = list(df_ppname["pipeline_name"].values) 

    odf = pd.DataFrame()
    for pname in pipeline_names:
        pp_name = join(curr_dir, f"pipelines/{pname}.joblib")
        pipeline = joblib.load(pp_name)
        pipeline.fit(X)
        y_score = pipeline.score_samples(X)
        odf[pname] = y_score.tolist()
        print(pp_name)
    
    ypred = meta_model.predict(X=odf.values)    
    print(ypred)
    
    print("Done")