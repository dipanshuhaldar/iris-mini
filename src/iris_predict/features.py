from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def numeric_pipeline(num_cols):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

def make_preprocessor(feature_names):
    return ColumnTransformer([
        ("num", numeric_pipeline(feature_names), feature_names)
    ])
