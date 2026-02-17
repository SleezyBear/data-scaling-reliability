from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_models():
    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1)
    }
    return models
