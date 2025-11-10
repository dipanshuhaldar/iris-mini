from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def build_estimator(kind: str):
    if kind == "logreg":
        return LogisticRegression(max_iter=1000)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200)
    if kind == "svm":
        return SVC(probability=True)
    raise ValueError(f"Unknown model kind: {kind}")
