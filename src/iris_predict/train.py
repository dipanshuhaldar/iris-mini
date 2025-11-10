import joblib, json, os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from .data import load
from .features import make_preprocessor
from .model import build_estimator

def train(cfg):
    X, y, feature_names, target_names = load()
    pre = make_preprocessor(feature_names)
    est = build_estimator(cfg["model"])
    pipe = Pipeline([("pre", pre), ("clf", est)])

    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["random_seed"])
    y_pred = cross_val_predict(pipe, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=target_names, output_dict=True)

    # fit on full data for final model
    pipe.fit(X, y)

    os.makedirs(cfg["artifacts_dir"], exist_ok=True)
    joblib.dump(pipe, f'{cfg["artifacts_dir"]}/model.joblib')
    with open(f'{cfg["artifacts_dir"]}/metrics.json', "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    return acc
