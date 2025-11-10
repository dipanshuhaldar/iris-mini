import joblib
import numpy as np

def load_model(path="artifacts/model.joblib"):
    return joblib.load(path)

def predict_one(sepal_len, sepal_wid, petal_len, petal_wid, model_path="artifacts/model.joblib"):
    pipe = load_model(model_path)
    X = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    proba = pipe.predict_proba(X)[0].tolist()
    pred = int(pipe.predict(X)[0])
    return {"pred_class_idx": pred, "proba": proba}
