import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_loader import load_data
from models import get_models
from noise import apply_label_noise, apply_feature_noise, apply_weak_samples

REPEATS = 10
TEST_SIZE = 0.2
SCALING_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]

LABEL_NOISE_RATE = 0.1
FEATURE_NOISE_STD = 0.05
WEAK_SAMPLE_FRACTION = 0.2

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]
FEATURE_NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1]
WEAK_SAMPLE_LEVELS = [0.0, 0.1, 0.2, 0.3]


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

        print(name)
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC-AUC : {auc:.4f}")
        print(f"  AUC Std : {auc_std:.4f}")
        print("-" * 40)

    print("PHASE 1 COMPLETE")


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
                print(f"{name} | Seed {seed+1}/{REPEATS}")
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

            print(name)
            print(f"  Accuracy: {mean_acc:.4f}")
            print(f"  F1 Score: {mean_f1:.4f}")
            print(f"  ROC-AUC : {mean_auc:.4f}")
            print(f"  AUC Std : {auc_std:.4f}")
            print("-" * 40)

            results.append({
                "phase": "clean_scaling",
                "model": name,
                "dataset_fraction": frac,
                "mean_accuracy": mean_acc,
                "mean_f1": mean_f1,
                "mean_auc": mean_auc,
                "auc_std": auc_std
            })

    pd.DataFrame(results).to_csv("phase2_clean_scaling_results.csv", index=False)

    print("\nSaved results to phase2_clean_scaling_results.csv")
    print("PHASE 2 COMPLETE")


def run_phase_3(X, y, models):
    print("\nPHASE 3: Fixed Combined Noise")
    print("=" * 60)

    results = []

    for frac in SCALING_LEVELS:
        print(f"\nDataset Fraction: {frac}")

        for name, base_model in models.items():
            acc_list = []
            f1_list = []
            auc_list = []

            for seed in range(REPEATS):
                print(f"{name} | Seed {seed+1}/{REPEATS}")

                X_subset, y_subset = get_scaled_subset(X, y, frac, seed)

                X_subset = apply_feature_noise(X_subset, FEATURE_NOISE_STD, seed)
                y_subset = apply_label_noise(y_subset, LABEL_NOISE_RATE, seed)
                X_subset, y_subset = apply_weak_samples(
                    X_subset, y_subset, WEAK_SAMPLE_FRACTION, 0.01, seed
                )

                model = base_model.__class__(**base_model.get_params())
                acc, f1, auc, _ = evaluate_model(model, X_subset, y_subset)

                acc_list.append(acc)
                f1_list.append(f1)
                auc_list.append(auc)

            print(f"{name} DONE")

            results.append({
                "phase": "fixed_combined_noise",
                "model": name,
                "dataset_fraction": frac,
                "mean_accuracy": np.mean(acc_list),
                "mean_f1": np.mean(f1_list),
                "mean_auc": np.mean(auc_list),
                "auc_std": np.std(auc_list)
            })

    pd.DataFrame(results).to_csv("phase3_fixed_noise_results.csv", index=False)

    print("PHASE 3 COMPLETE")


def run_phase_4_label_noise(X, y, models):
    print("\nPHASE 4A: Label Noise Interaction")
    print("=" * 60)

    results = []

    for noise_rate in NOISE_LEVELS:
        print(f"\nLabel Noise: {noise_rate}")

        for frac in SCALING_LEVELS:
            print(f" Dataset Fraction: {frac}")

            for name, base_model in models.items():
                acc_list, f1_list, auc_list = [], [], []

                for seed in range(REPEATS):
                    print(f"{name} | Seed {seed+1}/{REPEATS}")

                    X_subset, y_subset = get_scaled_subset(X, y, frac, seed)

                    if noise_rate > 0:
                        y_subset = apply_label_noise(y_subset, noise_rate, seed)

                    model = base_model.__class__(**base_model.get_params())
                    acc, f1, auc, _ = evaluate_model(model, X_subset, y_subset)

                    acc_list.append(acc)
                    f1_list.append(f1)
                    auc_list.append(auc)

                results.append({
                    "phase": "label_noise_interaction",
                    "model": name,
                    "dataset_fraction": frac,
                    "label_noise": noise_rate,
                    "mean_accuracy": np.mean(acc_list),
                    "mean_f1": np.mean(f1_list),
                    "mean_auc": np.mean(auc_list),
                    "auc_std": np.std(auc_list)
                })

    pd.DataFrame(results).to_csv("phase4_label_noise_results.csv", index=False)

    print("PHASE 4A COMPLETE")


def run_phase_4_feature_noise(X, y, models):
    print("\nPHASE 4B: Feature Noise Interaction")
    print("=" * 60)

    results = []

    for noise_std in FEATURE_NOISE_LEVELS:
        print(f"\nFeature Noise STD: {noise_std}")

        for frac in SCALING_LEVELS:
            print(f" Dataset Fraction: {frac}")

            for name, base_model in models.items():
                acc_list, f1_list, auc_list = [], [], []

                for seed in range(REPEATS):
                    print(f"{name} | Seed {seed+1}/{REPEATS}")

                    X_subset, y_subset = get_scaled_subset(X, y, frac, seed)

                    if noise_std > 0:
                        X_subset = apply_feature_noise(X_subset, noise_std, seed)

                    model = base_model.__class__(**base_model.get_params())
                    acc, f1, auc, _ = evaluate_model(model, X_subset, y_subset)

                    acc_list.append(acc)
                    f1_list.append(f1)
                    auc_list.append(auc)

                results.append({
                    "phase": "feature_noise_interaction",
                    "model": name,
                    "dataset_fraction": frac,
                    "feature_noise_std": noise_std,
                    "mean_accuracy": np.mean(acc_list),
                    "mean_f1": np.mean(f1_list),
                    "mean_auc": np.mean(auc_list),
                    "auc_std": np.std(auc_list)
                })

    pd.DataFrame(results).to_csv("phase4_feature_noise_results.csv", index=False)

    print("PHASE 4B COMPLETE")


def run_phase_4_weak_samples(X, y, models):
    print("\nPHASE 4C: Weak Sample Interaction")
    print("=" * 60)

    results = []

    for weak_fraction in WEAK_SAMPLE_LEVELS:
        print(f"\nWeak Fraction: {weak_fraction}")

        for frac in SCALING_LEVELS:
            print(f" Dataset Fraction: {frac}")

            for name, base_model in models.items():
                acc_list, f1_list, auc_list = [], [], []

                for seed in range(REPEATS):
                    print(f"{name} | Seed {seed+1}/{REPEATS}")

                    X_subset, y_subset = get_scaled_subset(X, y, frac, seed)

                    if weak_fraction > 0:
                        X_subset, y_subset = apply_weak_samples(
                            X_subset, y_subset, weak_fraction, 0.01, seed
                        )

                    model = base_model.__class__(**base_model.get_params())
                    acc, f1, auc, _ = evaluate_model(model, X_subset, y_subset)

                    acc_list.append(acc)
                    f1_list.append(f1)
                    auc_list.append(auc)

                results.append({
                    "phase": "weak_sample_interaction",
                    "model": name,
                    "dataset_fraction": frac,
                    "weak_fraction": weak_fraction,
                    "mean_accuracy": np.mean(acc_list),
                    "mean_f1": np.mean(f1_list),
                    "mean_auc": np.mean(auc_list),
                    "auc_std": np.std(auc_list)
                })

    pd.DataFrame(results).to_csv("phase4_weak_sample_results.csv", index=False)

    print("PHASE 4C COMPLETE")


def run_phase_5_label_noise_analysis():
    print("\nPHASE 5: Reliability Analysis")
    print("=" * 60)

    df = pd.read_csv("phase4_label_noise_results.csv")

    df["cv"] = df["auc_std"] / df["mean_auc"]
    df["stability_score"] = 1 - df["cv"]

    clean_df = df[df["label_noise"] == 0.0][
        ["model", "dataset_fraction", "mean_auc"]
    ].rename(columns={"mean_auc": "clean_auc"})

    df = df.merge(clean_df, on=["model", "dataset_fraction"], how="left")

    df["relative_auc_drop"] = (
        (df["clean_auc"] - df["mean_auc"]) / df["clean_auc"]
    )

    scaling_gain_list = []

    for model in df["model"].unique():
        for noise in df["label_noise"].unique():
            subset = df[
                (df["model"] == model) &
                (df["label_noise"] == noise)
            ].sort_values("dataset_fraction")

            gain = (
                subset[subset["dataset_fraction"] == 1.0]["mean_auc"].values[0]
                -
                subset[subset["dataset_fraction"] == 0.2]["mean_auc"].values[0]
            )

            scaling_gain_list.extend([gain] * len(subset))

    df["scaling_gain"] = scaling_gain_list

    df["breakdown_flag"] = (
        (df["relative_auc_drop"] > 0.1) |
        (df["cv"] > 0.1) |
        (df["scaling_gain"] < 0)
    )

    df.to_csv("phase5_label_noise_analysis.csv", index=False)

    print("PHASE 5 COMPLETE")


def main():
    X, y = load_data()
    models = get_models()

    run_phase_1(X, y, models)
    run_phase_2(X, y, models)
    run_phase_3(X, y, models)
    run_phase_4_label_noise(X, y, models)
    run_phase_4_feature_noise(X, y, models)
    run_phase_4_weak_samples(X, y, models)
    run_phase_5_label_noise_analysis()


if __name__ == "__main__":
    main()
