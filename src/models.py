from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_models(seed_model):
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed_model
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=seed_model,
            n_jobs=-1
        )
    }
    return models