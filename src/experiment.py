import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_loader import load_data
from models import get_models

REPEATS = 10
TEST_SIZE = 0.2
SCALING_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]

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

def get_scaled_subset(X, y, fraction, seed):
    if fraction == 1.0:
        return X, y

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=fraction,
        random_state=seed
    )

    for train_idx, _ in splitter.split(X, y):
        return X[train_idx], y[train_idx]

def run_phase_1(X, y, models):
    print("\nPHASE 1: Clean Full Dataset Baseline")
    print("=" * 60)

    for name, model in models.items():
        acc, f1, auc, auc_std = evaluate_model(model, X, y)

        print(f"{name}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC-AUC : {auc:.4f}")
        print(f"  AUC Std : {auc_std:.4f}")
        print("-" * 40)

def run_phase_2(X, y, models):
    print("\nPHASE 2: Clean Dataset Scaling")
    print("=" * 60)

    results = []

    for frac in SCALING_LEVELS:
        print(f"\nDataset Fraction: {frac}")
        print("-" * 50)

        for name, base_model in models.items():
            acc_list = []
            f1_list = []
            auc_list = []

            for seed in range(REPEATS):
                X_subset, y_subset = get_scaled_subset(X, y, frac, seed)
                model = base_model.__class__(**base_model.get_params())
                acc, f1, auc, _ = evaluate_model(model, X_subset, y_subset)

                acc_list.append(acc)
                f1_list.append(f1)
                auc_list.append(auc)

            mean_acc = np.mean(acc_list)
            mean_f1 = np.mean(f1_list)
            mean_auc = np.mean(auc_list)
            auc_std = np.std(auc_list)

            print(f"{name}")
            print(f"  Accuracy: {mean_acc:.4f}")
            print(f"  F1 Score: {mean_f1:.4f}")
            print(f"  ROC-AUC : {mean_auc:.4f}")
            print(f"  AUC Std : {auc_std:.4f}")
            print("-" * 40)

            results.append({
                "model": name,
                "dataset_fraction": frac,
                "mean_accuracy": mean_acc,
                "mean_f1": mean_f1,
                "mean_auc": mean_auc,
                "auc_std": auc_std
            })

    df = pd.DataFrame(results)
    df.to_csv("phase2_clean_scaling_results.csv", index=False)
    print("\nSaved results to phase2_clean_scaling_results.csv")

def main():
    X, y = load_data()
    models = get_models()

    run_phase_1(X, y, models)
    run_phase_2(X, y, models)

if __name__ == "__main__":
    main()
