import os
import json
import tempfile
import hashlib
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone

from data_loader import load_data
from models import get_models
from noise import apply_label_noise, apply_feature_noise, apply_weak_samples
from visualization import run_visualization_pipeline
from statistical_tests import run_statistical_tests


MASTER_SEED = 42

REPEATS = 10
TEST_SIZE = 0.2
SCALING_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]

RUN_EXPERIMENTS = True
RUN_VISUALIZATION = True
RUN_STATISTICAL_TESTS = True

LABEL_NOISE_RATE = 0.1
FEATURE_NOISE_STD = 0.05
WEAK_SAMPLE_FRACTION = 0.2

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]
FEATURE_NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1]
WEAK_SAMPLE_LEVELS = [0.0, 0.1, 0.2, 0.3]

BREAKDOWN_THRESHOLD = 0.05

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.json")


def atomic_write_json(data, filepath):
    dirpath = os.path.dirname(filepath)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dirpath) as tmp:
        json.dump(data, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, filepath)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return {"completed": {}}
    try:
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    except:
        return {"completed": {}}


def save_checkpoint(checkpoint):
    atomic_write_json(checkpoint, CHECKPOINT_PATH)


def get_run_id(phase, *args):
    key = f"{phase}|" + "|".join(map(str, args))
    return hashlib.md5(key.encode()).hexdigest()


def derive_seed(master_seed, name):
    key = f"{master_seed}_{name}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


def evaluate_model(model, X, y, seed_split_base):
    accuracies, f1_scores, auc_scores = [], [], []

    for i in range(REPEATS):
        split_seed = seed_split_base + i

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=split_seed
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


def precompute_scaled_subsets(X, y, seed_scaling):
    return {
        frac: get_scaled_subset(X, y, frac, seed=seed_scaling)
        for frac in SCALING_LEVELS
    }


def print_phase(title):
    print("\n" + "="*50)
    print(title)
    print("="*50)


def run_phase5_enrichment():
    specs = [
        ("phase4_label_noise_results.csv", "label_noise",
         "phase5_label_noise_analysis.csv"),
        ("phase4_feature_noise_results.csv", "feature_noise_std",
         "phase5_feature_noise_analysis.csv"),
        ("phase4_weak_sample_results.csv", "weak_fraction",
         "phase5_weak_sample_analysis.csv"),
    ]

    for input_file, noise_column, output_file in specs:
        in_path = os.path.join(RESULTS_DIR, input_file)
        out_path = os.path.join(RESULTS_DIR, output_file)

        if not os.path.exists(in_path):
            print(f"  [Phase 5] {input_file} not found, skipping.")
            continue

        df = pd.read_csv(in_path)

        df["cv"] = df["auc_std"] / df["mean_auc"]

        baseline = (
            df[df[noise_column] == 0.0][["model", "dataset_fraction", "mean_auc"]]
            .rename(columns={"mean_auc": "baseline_auc"})
        )
        df = df.merge(baseline, on=["model", "dataset_fraction"], how="left")
        df["relative_auc_drop"] = (
            (df["baseline_auc"] - df["mean_auc"]) / df["baseline_auc"]
        )
        df["breakdown_flag"] = (
            df["relative_auc_drop"] > BREAKDOWN_THRESHOLD
        ).astype(int)
        df = df.drop(columns=["baseline_auc"])

        df.to_csv(out_path, index=False)
        print(f"  Saved: {output_file}")


def main():

    checkpoint = load_checkpoint()
    completed = checkpoint["completed"]

    if RUN_EXPERIMENTS:

        np.random.seed(MASTER_SEED)
        random.seed(MASTER_SEED)

        seed_scaling = derive_seed(MASTER_SEED, "scaling")
        seed_split_base = derive_seed(MASTER_SEED, "split")
        seed_label_noise = derive_seed(MASTER_SEED, "label_noise")
        seed_feature_noise = derive_seed(MASTER_SEED, "feature_noise")
        seed_weak = derive_seed(MASTER_SEED, "weak")
        seed_model = derive_seed(MASTER_SEED, "model")

        X, y = load_data()
        models = get_models(seed_model)

        scaled_data = precompute_scaled_subsets(X, y, seed_scaling)

        # ── Phase 1 ────────────────────────────────────────────────────────
        print_phase("PHASE 1: Baseline clean dataset evaluation")

        results_phase1 = []

        for name, model in models.items():
            run_id = get_run_id("p1", name)
            if run_id in completed:
                results_phase1.append(completed[run_id])
                continue

            acc, f1, auc, auc_std = evaluate_model(
                model, X, y, seed_split_base
            )

            row = {
                "phase": "baseline_clean",
                "model": name,
                "dataset_fraction": 1.0,
                "mean_accuracy": acc,
                "mean_f1": f1,
                "mean_auc": auc,
                "auc_std": auc_std
            }

            results_phase1.append(row)
            completed[run_id] = row
            save_checkpoint(checkpoint)

        pd.DataFrame(results_phase1).to_csv(
            f"{RESULTS_DIR}/phase1_baseline_results.csv", index=False
        )

        # ── Phase 2 ────────────────────────────────────────────────────────
        print_phase("PHASE 2: Dataset scaling without noise")

        results_phase2 = []

        for frac in SCALING_LEVELS:
            X_subset, y_subset = scaled_data[frac]

            for name, model in models.items():
                run_id = get_run_id("p2", name, frac)
                if run_id in completed:
                    results_phase2.append(completed[run_id])
                    continue

                acc, f1, auc, auc_std = evaluate_model(
                    model, X_subset, y_subset, seed_split_base
                )

                row = {
                    "phase": "clean_scaling",
                    "model": name,
                    "dataset_fraction": frac,
                    "mean_accuracy": acc,
                    "mean_f1": f1,
                    "mean_auc": auc,
                    "auc_std": auc_std
                }

                results_phase2.append(row)
                completed[run_id] = row
                save_checkpoint(checkpoint)

        pd.DataFrame(results_phase2).to_csv(
            f"{RESULTS_DIR}/phase2_clean_scaling_results.csv", index=False
        )

        # ── Phase 3 ────────────────────────────────────────────────────────
        print_phase("PHASE 3: Fixed combined noise injection")

        results_phase3 = []

        for frac in SCALING_LEVELS:
            X_subset, y_subset = scaled_data[frac]
            X_subset = X_subset.copy()
            y_subset = y_subset.copy()

            X_subset = apply_feature_noise(X_subset, FEATURE_NOISE_STD, random_state=seed_feature_noise)
            y_subset = apply_label_noise(y_subset, LABEL_NOISE_RATE, random_state=seed_label_noise)
            X_subset, y_subset = apply_weak_samples(X_subset, y_subset, WEAK_SAMPLE_FRACTION, 0.01, random_state=seed_weak)

            for name, model in models.items():
                run_id = get_run_id("p3", name, frac)
                if run_id in completed:
                    results_phase3.append(completed[run_id])
                    continue

                acc, f1, auc, auc_std = evaluate_model(
                    model, X_subset, y_subset, seed_split_base
                )

                row = {
                    "phase": "fixed_combined_noise",
                    "model": name,
                    "dataset_fraction": frac,
                    "mean_accuracy": acc,
                    "mean_f1": f1,
                    "mean_auc": auc,
                    "auc_std": auc_std
                }

                results_phase3.append(row)
                completed[run_id] = row
                save_checkpoint(checkpoint)

        pd.DataFrame(results_phase3).to_csv(
            f"{RESULTS_DIR}/phase3_fixed_noise_results.csv", index=False
        )

        # ── Phase 4 ────────────────────────────────────────────────────────
        print_phase("PHASE 4: Scaling x varying noise interactions")

        for noise_levels, noise_column, noise_func, filename, phase_tag in [
            (NOISE_LEVELS, "label_noise",
             lambda X, y, nv: (X, apply_label_noise(y, nv, random_state=seed_label_noise)),
             "phase4_label_noise_results.csv", "p4_label"),

            (FEATURE_NOISE_LEVELS, "feature_noise_std",
             lambda X, y, nv: (apply_feature_noise(X, nv, random_state=seed_feature_noise), y),
             "phase4_feature_noise_results.csv", "p4_feature"),

            (WEAK_SAMPLE_LEVELS, "weak_fraction",
             lambda X, y, nv: apply_weak_samples(X, y, nv, 0.01, random_state=seed_weak),
             "phase4_weak_sample_results.csv", "p4_weak")
        ]:

            results = []

            for noise_value in noise_levels:
                for frac in SCALING_LEVELS:
                    X_subset, y_subset = scaled_data[frac]
                    X_subset = X_subset.copy()
                    y_subset = y_subset.copy()

                    if noise_value > 0:
                        X_subset, y_subset = noise_func(X_subset, y_subset, noise_value)

                    for name, model in models.items():
                        run_id = get_run_id(phase_tag, name, frac, noise_value)
                        if run_id in completed:
                            results.append(completed[run_id])
                            continue

                        acc, f1, auc, auc_std = evaluate_model(
                            model, X_subset, y_subset, seed_split_base
                        )

                        row = {
                            "phase": f"{noise_column}_interaction",
                            "model": name,
                            "dataset_fraction": frac,
                            noise_column: noise_value,
                            "mean_accuracy": acc,
                            "mean_f1": f1,
                            "mean_auc": auc,
                            "auc_std": auc_std
                        }

                        results.append(row)
                        completed[run_id] = row
                        save_checkpoint(checkpoint)

            pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/{filename}", index=False)

        # ── Phase 5 ────────────────────────────────────────────────────────
        print_phase("PHASE 5: Noise analysis enrichment")
        run_phase5_enrichment()

    # ── Phase 6 ────────────────────────────────────────────────────────────
    if RUN_STATISTICAL_TESTS:
        print_phase("PHASE 6: Statistical testing")
        run_statistical_tests(REPEATS)

    # ── Phase 7 ────────────────────────────────────────────────────────────
    if RUN_VISUALIZATION:
        print_phase("PHASE 7: Visualization")
        run_visualization_pipeline()


if __name__ == "__main__":
    main()