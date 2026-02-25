import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_data
from models import get_models
from noise import apply_label_noise, apply_feature_noise, apply_weak_samples
from visualization import run_visualization_pipeline


REPEATS = 10
TEST_SIZE = 0.2
SCALING_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]

RUN_EXPERIMENTS = True
RUN_VISUALIZATION = True

LABEL_NOISE_RATE = 0.1
FEATURE_NOISE_STD = 0.05
WEAK_SAMPLE_FRACTION = 0.2

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]
FEATURE_NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1]
WEAK_SAMPLE_LEVELS = [0.0, 0.1, 0.2, 0.3]


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_model(model, X, y):
    accuracies, f1_scores, auc_scores = [], [], []

    for i in range(REPEATS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=i
        )

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        y_pred = model_clone.predict(X_test)
        y_prob = model_clone.predict_proba(X_test)[:, 1]

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


def precompute_scaled_subsets(X, y):
    return {
        frac: get_scaled_subset(X, y, frac, seed=42)
        for frac in SCALING_LEVELS
    }


def run_phase_1(X, y, models):
    print("\nPHASE 1: Clean Full Dataset Baseline")
    print("=" * 60)

    for name, model in models.items():
        acc, f1, auc, auc_std = evaluate_model(model, X, y)

        print(name)
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC-AUC : {auc:.4f}")
        print(f"  AUC Std : {auc_std:.4f}")
        print("-" * 40)

    print("PHASE 1 COMPLETE")


def run_phase_2(scaled_data, models):
    print("\nPHASE 2: Clean Dataset Scaling")
    print("=" * 60)

    results = []

    for frac in SCALING_LEVELS:
        X_subset, y_subset = scaled_data[frac]

        for name, model in models.items():
            acc, f1, auc, auc_std = evaluate_model(model, X_subset, y_subset)

            results.append({
                "phase": "clean_scaling",
                "model": name,
                "dataset_fraction": frac,
                "mean_accuracy": acc,
                "mean_f1": f1,
                "mean_auc": auc,
                "auc_std": auc_std
            })

    pd.DataFrame(results).to_csv(
        f"{RESULTS_DIR}/phase2_clean_scaling_results.csv", index=False
    )

    print("PHASE 2 COMPLETE")


def run_phase_3(scaled_data, models):
    print("\nPHASE 3: Fixed Combined Noise")
    print("=" * 60)

    results = []

    for frac in SCALING_LEVELS:
        X_subset, y_subset = scaled_data[frac]
        X_subset = X_subset.copy()
        y_subset = y_subset.copy()

        X_subset = apply_feature_noise(X_subset, FEATURE_NOISE_STD, random_state=42)
        y_subset = apply_label_noise(y_subset, LABEL_NOISE_RATE, random_state=42)
        X_subset, y_subset = apply_weak_samples(
            X_subset, y_subset, WEAK_SAMPLE_FRACTION, 0.01, random_state=42
        )

        for name, model in models.items():
            acc, f1, auc, auc_std = evaluate_model(model, X_subset, y_subset)

            results.append({
                "phase": "fixed_combined_noise",
                "model": name,
                "dataset_fraction": frac,
                "mean_accuracy": acc,
                "mean_f1": f1,
                "mean_auc": auc,
                "auc_std": auc_std
            })

    pd.DataFrame(results).to_csv(
        f"{RESULTS_DIR}/phase3_fixed_noise_results.csv", index=False
    )

    print("PHASE 3 COMPLETE")


def run_phase_4_generic(
    scaled_data,
    models,
    noise_levels,
    noise_column,
    noise_function,
    output_filename
):
    print(f"\nPHASE 4: {noise_column} Interaction")
    print("=" * 60)

    results = []

    for noise_value in noise_levels:
        for frac in SCALING_LEVELS:
            X_subset, y_subset = scaled_data[frac]
            X_subset = X_subset.copy()
            y_subset = y_subset.copy()

            if noise_value > 0:
                X_subset, y_subset = noise_function(
                    X_subset, y_subset, noise_value
                )

            for name, model in models.items():
                acc, f1, auc, auc_std = evaluate_model(model, X_subset, y_subset)

                results.append({
                    "phase": f"{noise_column}_interaction",
                    "model": name,
                    "dataset_fraction": frac,
                    noise_column: noise_value,
                    "mean_accuracy": acc,
                    "mean_f1": f1,
                    "mean_auc": auc,
                    "auc_std": auc_std
                })

    pd.DataFrame(results).to_csv(
        f"{RESULTS_DIR}/{output_filename}", index=False
    )

    print("PHASE 4 COMPLETE")


def label_noise_wrapper(X, y, noise_value):
    return X, apply_label_noise(y, noise_value, random_state=42)


def feature_noise_wrapper(X, y, noise_value):
    return apply_feature_noise(X, noise_value, random_state=42), y


def weak_sample_wrapper(X, y, noise_value):
    return apply_weak_samples(X, y, noise_value, 0.01, random_state=42)


def run_phase_5_analysis(input_csv, output_csv, noise_column):
    print(f"\nPHASE 5: Reliability Analysis ({noise_column})")
    print("=" * 60)

    df = pd.read_csv(f"{RESULTS_DIR}/{input_csv}")

    df["cv"] = df["auc_std"] / df["mean_auc"]
    df["stability_score"] = 1 - df["cv"]

    clean_df = (
        df[df[noise_column] == 0.0]
        [["model", "dataset_fraction", "mean_auc"]]
        .rename(columns={"mean_auc": "clean_auc"})
    )

    df = df.merge(clean_df, on=["model", "dataset_fraction"], how="left")

    df["relative_auc_drop"] = (
        (df["clean_auc"] - df["mean_auc"]) / df["clean_auc"]
    )

    scaling_reference = (
        df[df["dataset_fraction"].isin([0.2, 1.0])]
        .pivot_table(
            index=["model", noise_column],
            columns="dataset_fraction",
            values="mean_auc"
        )
        .reset_index()
    )

    scaling_reference["scaling_gain"] = (
        scaling_reference[1.0] - scaling_reference[0.2]
    )

    df = df.merge(
        scaling_reference[["model", noise_column, "scaling_gain"]],
        on=["model", noise_column],
        how="left"
    )

    df["breakdown_flag"] = (
        (df["relative_auc_drop"] > 0.1) |
        (df["cv"] > 0.1) |
        (df["scaling_gain"] < 0)
    )

    df.to_csv(f"{RESULTS_DIR}/{output_csv}", index=False)

    print("PHASE 5 COMPLETE")


def main():
    if RUN_EXPERIMENTS:
        X, y = load_data()
        models = get_models()

        for name, model in models.items():
            if isinstance(model, RandomForestClassifier):
                model.set_params(n_jobs=-1)

        scaled_data = precompute_scaled_subsets(X, y)

        run_phase_1(X, y, models)
        run_phase_2(scaled_data, models)
        run_phase_3(scaled_data, models)

        run_phase_4_generic(
            scaled_data,
            models,
            NOISE_LEVELS,
            "label_noise",
            label_noise_wrapper,
            "phase4_label_noise_results.csv"
        )

        run_phase_4_generic(
            scaled_data,
            models,
            FEATURE_NOISE_LEVELS,
            "feature_noise_std",
            feature_noise_wrapper,
            "phase4_feature_noise_results.csv"
        )

        run_phase_4_generic(
            scaled_data,
            models,
            WEAK_SAMPLE_LEVELS,
            "weak_fraction",
            weak_sample_wrapper,
            "phase4_weak_sample_results.csv"
        )

        run_phase_5_analysis(
            "phase4_label_noise_results.csv",
            "phase5_label_noise_analysis.csv",
            "label_noise"
        )

        run_phase_5_analysis(
            "phase4_feature_noise_results.csv",
            "phase5_feature_noise_analysis.csv",
            "feature_noise_std"
        )

        run_phase_5_analysis(
            "phase4_weak_sample_results.csv",
            "phase5_weak_sample_analysis.csv",
            "weak_fraction"
        )

    if RUN_VISUALIZATION:
        run_visualization_pipeline()


if __name__ == "__main__":
    main()