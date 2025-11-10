from sklearn.datasets import load_iris
import pandas as pd

def load():
    ds = load_iris(as_frame=True)
    X = ds.data
    y = ds.target
    target_names = list(ds.target_names)
    feature_names = list(ds.feature_names)
    return X, y, feature_names, target_names
