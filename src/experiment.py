import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_loader import load_data
from models import get_models

REPEATS = 10
TEST_SIZE = 0.2

def evaluate_model(model, X, y):
    accuracies = []
    f1_scores = []
    auc_scores = []

    for i in range(REPEATS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=i
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_prob))

    return (
        np.mean(accuracies),
        np.mean(f1_scores),
        np.mean(auc_scores),
        np.std(auc_scores)
    )

def main():
    X, y = load_data()
    models = get_models()

    for name, model in models.items():
        acc, f1, auc, auc_std = evaluate_model(model, X, y)

        print(f"{name}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC-AUC : {auc:.4f}")
        print(f"  AUC Std : {auc_std:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    main()
