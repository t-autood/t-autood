import joblib
import pandas as pd
from os.path import join, realpath, dirname
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

def _set_pipeline_random_seed(pipeline, pipeline_random_seed=3046024202):
    if isinstance(pipeline, FeatureUnion):
        steps = pipeline.transformer_list
    elif isinstance(pipeline, Pipeline):
        steps = pipeline.steps
    else:
        steps = [('estimator', pipeline)]  # an ordinary estimator

    for _, step_component in steps:
        if isinstance(step_component, Pipeline) or isinstance(step_component, FeatureUnion):
            _set_pipeline_random_seed(
                                            pipeline=step_component,
                                            pipeline_random_seed=pipeline_random_seed
            )
            continue

        if 'random_state' in step_component.get_params():
            step_component.set_params(random_state=pipeline_random_seed)

    return pipeline


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
        pp = joblib.load(pp_name)
        
        pipeline = _set_pipeline_random_seed(
            pipeline=pp,
            pipeline_random_seed=3046024202,
        )
        pipeline.fit(X)
        y_score = pipeline.score_samples(X)
        odf[pname] = y_score.tolist()
        print(pp_name)
    
    ypred = meta_model.predict(X=odf.values)    
    print(ypred)
    
    print("Done")