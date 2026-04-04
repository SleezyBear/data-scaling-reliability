import os
import json
import tempfile
import hashlib
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from data_loader import load_all_datasets
from models import get_models
from noise import apply_label_noise, apply_feature_noise, apply_weak_samples
from visualization import run_visualization_pipeline
from statistical_tests import run_statistical_tests
from threshold_detection import run_threshold_detection


# ── Config ─────────────────────────────────────────────────────────────────

MASTER_SEED = 42

REPEATS = 10
TEST_SIZE = 0.2
SCALING_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]

RUN_EXPERIMENTS         = True
RUN_STATISTICAL_TESTS   = True
RUN_THRESHOLD_DETECTION = True
RUN_VISUALIZATION       = True

LABEL_NOISE_RATE      = 0.1
FEATURE_NOISE_STD     = 0.05
WEAK_SAMPLE_FRACTION  = 0.2

NOISE_LEVELS          = [0.0, 0.1, 0.2, 0.3]
FEATURE_NOISE_LEVELS  = [0.0, 0.02, 0.05, 0.1]
WEAK_SAMPLE_LEVELS    = [0.0, 0.1, 0.2, 0.3]

BREAKDOWN_THRESHOLD   = 0.05

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.json")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Checkpointing ──────────────────────────────────────────────────────────

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
    except (json.JSONDecodeError, OSError):
        return {"completed": {}}


def save_checkpoint(checkpoint):
    atomic_write_json(checkpoint, CHECKPOINT_PATH)


def get_run_id(phase, *args):
    key = f"{phase}|" + "|".join(map(str, args))
    return hashlib.md5(key.encode()).hexdigest()


# ── Seeds ──────────────────────────────────────────────────────────────────

def derive_seed(master_seed, name):
    key = f"{master_seed}_{name}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)


# ── Evaluation ─────────────────────────────────────────────────────────────

def _single_repeat(model, X, y, split_seed, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=split_seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    m = clone(model)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    y_prob = m.predict_proba(X_test)[:, 1]
    return (
        float(accuracy_score(y_test, y_pred)),
        float(f1_score(y_test, y_pred, zero_division=0)),
        float(roc_auc_score(y_test, y_prob))
    )


def evaluate_model(model, X, y, seed_split_base):
    results = Parallel(n_jobs=-1)(
        delayed(_single_repeat)(model, X, y, seed_split_base + i, TEST_SIZE)
        for i in range(REPEATS)
    )
    accuracies = [r[0] for r in results]
    f1_scores  = [r[1] for r in results]
    auc_scores = [r[2] for r in results]
    return (
        float(np.mean(accuracies)),
        float(np.mean(f1_scores)),
        float(np.mean(auc_scores)),
        float(np.std(auc_scores)),
        auc_scores  # raw per-repeat scores for Wilcoxon/Friedman
    )


# ── Scaling ─────────────────────────────────────────────────────────────────

def get_scaled_subset(X, y, fraction, seed):
    if fraction == 1.0:
        return X, y
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=fraction, random_state=seed
    )
    for train_idx, _ in splitter.split(X, y):
        return X[train_idx], y[train_idx]


def precompute_scaled_subsets(X, y, seed_scaling):
    return {
        frac: get_scaled_subset(X, y, frac, seed=seed_scaling)
        for frac in SCALING_LEVELS
    }


# ── Row builder ─────────────────────────────────────────────────────────────

def build_row(phase_label, dataset_name, model_name, frac,
              acc, f1, auc, auc_std, auc_reps, extra=None):
    row = {
        "phase": phase_label,
        "dataset": dataset_name,
        "model": model_name,
        "dataset_fraction": frac,
        "mean_accuracy": acc,
        "mean_f1": f1,
        "mean_auc": auc,
        "auc_std": auc_std,
    }
    for i, v in enumerate(auc_reps):
        row[f"auc_rep_{i}"] = v
    if extra:
        row.update(extra)
    return row


# ── Phase 5 enrichment ──────────────────────────────────────────────────────

def run_phase5_enrichment(dataset_name):
    specs = [
        (f"{dataset_name}_phase4_label_noise_results.csv",
         "label_noise",
         f"{dataset_name}_phase5_label_noise_analysis.csv"),
        (f"{dataset_name}_phase4_feature_noise_results.csv",
         "feature_noise_std",
         f"{dataset_name}_phase5_feature_noise_analysis.csv"),
        (f"{dataset_name}_phase4_weak_sample_results.csv",
         "weak_fraction",
         f"{dataset_name}_phase5_weak_sample_analysis.csv"),
    ]
    for input_file, noise_column, output_file in specs:
        in_path  = os.path.join(RESULTS_DIR, input_file)
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


# ── Print helper ────────────────────────────────────────────────────────────

def print_phase(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    checkpoint = load_checkpoint()
    completed  = checkpoint["completed"]
    dataset_names = []

    if RUN_EXPERIMENTS:
        np.random.seed(MASTER_SEED)
        random.seed(MASTER_SEED)

        seed_scaling       = derive_seed(MASTER_SEED, "scaling")
        seed_split_base    = derive_seed(MASTER_SEED, "split")
        seed_label_noise   = derive_seed(MASTER_SEED, "label_noise")
        seed_feature_noise = derive_seed(MASTER_SEED, "feature_noise")
        seed_weak          = derive_seed(MASTER_SEED, "weak")
        seed_model         = derive_seed(MASTER_SEED, "model")

        all_datasets  = load_all_datasets()
        dataset_names = list(all_datasets.keys())
        models        = get_models(seed_model)

        for dataset_name, (X, y) in all_datasets.items():
            print(f"\n{'='*60}")
            print(f"DATASET: {dataset_name}")
            print(f"{'='*60}")

            scaled_data = precompute_scaled_subsets(X, y, seed_scaling)

            # ── Phase 1: Clean full baseline ──────────────────────────────
            print_phase(f"PHASE 1: Clean full baseline [{dataset_name}]")
            p1_rows = []
            for name, model in models.items():
                run_id = get_run_id("p1", dataset_name, name)
                if run_id in completed:
                    p1_rows.append(completed[run_id])
                    continue
                acc, f1, auc, auc_std, auc_reps = evaluate_model(
                    model, X, y, seed_split_base
                )
                row = build_row("baseline_clean", dataset_name, name,
                                1.0, acc, f1, auc, auc_std, auc_reps)
                p1_rows.append(row)
                completed[run_id] = row
                save_checkpoint(checkpoint)
            pd.DataFrame(p1_rows).to_csv(
                os.path.join(RESULTS_DIR, f"{dataset_name}_phase1_baseline.csv"),
                index=False
            )

            # ── Phase 2: Clean scaling ────────────────────────────────────
            print_phase(f"PHASE 2: Clean scaling [{dataset_name}]")
            p2_rows = []
            for frac in SCALING_LEVELS:
                X_sub, y_sub = scaled_data[frac]
                for name, model in models.items():
                    run_id = get_run_id("p2", dataset_name, name, frac)
                    if run_id in completed:
                        p2_rows.append(completed[run_id])
                        continue
                    acc, f1, auc, auc_std, auc_reps = evaluate_model(
                        model, X_sub, y_sub, seed_split_base
                    )
                    row = build_row("clean_scaling", dataset_name, name,
                                    frac, acc, f1, auc, auc_std, auc_reps)
                    p2_rows.append(row)
                    completed[run_id] = row
                    save_checkpoint(checkpoint)
            pd.DataFrame(p2_rows).to_csv(
                os.path.join(RESULTS_DIR, f"{dataset_name}_phase2_clean_scaling_results.csv"),
                index=False
            )

            # ── Phase 3: Combined fixed noise ─────────────────────────────
            print_phase(f"PHASE 3: Fixed combined noise [{dataset_name}]")
            p3_rows = []
            for frac in SCALING_LEVELS:
                X_sub, y_sub = scaled_data[frac]
                X_sub = apply_feature_noise(X_sub.copy(), FEATURE_NOISE_STD, random_state=seed_feature_noise)
                y_sub = apply_label_noise(y_sub.copy(), LABEL_NOISE_RATE, random_state=seed_label_noise)
                X_sub, y_sub = apply_weak_samples(X_sub, y_sub, WEAK_SAMPLE_FRACTION, 0.01, random_state=seed_weak)
                for name, model in models.items():
                    run_id = get_run_id("p3", dataset_name, name, frac)
                    if run_id in completed:
                        p3_rows.append(completed[run_id])
                        continue
                    acc, f1, auc, auc_std, auc_reps = evaluate_model(
                        model, X_sub, y_sub, seed_split_base
                    )
                    row = build_row("fixed_combined_noise", dataset_name, name,
                                    frac, acc, f1, auc, auc_std, auc_reps)
                    p3_rows.append(row)
                    completed[run_id] = row
                    save_checkpoint(checkpoint)
            pd.DataFrame(p3_rows).to_csv(
                os.path.join(RESULTS_DIR, f"{dataset_name}_phase3_fixed_noise_results.csv"),
                index=False
            )

            # ── Phase 4: Isolated noise interactions ──────────────────────
            noise_specs = [
                (NOISE_LEVELS, "label_noise",
                 lambda X, y, nv: (X, apply_label_noise(y, nv, random_state=seed_label_noise)),
                 f"{dataset_name}_phase4_label_noise_results.csv", "p4_label"),

                (FEATURE_NOISE_LEVELS, "feature_noise_std",
                 lambda X, y, nv: (apply_feature_noise(X, nv, random_state=seed_feature_noise), y),
                 f"{dataset_name}_phase4_feature_noise_results.csv", "p4_feature"),

                (WEAK_SAMPLE_LEVELS, "weak_fraction",
                 lambda X, y, nv: apply_weak_samples(X, y, nv, 0.01, random_state=seed_weak),
                 f"{dataset_name}_phase4_weak_sample_results.csv", "p4_weak"),
            ]

            for noise_levels, noise_column, noise_func, filename, phase_tag in noise_specs:
                print_phase(f"PHASE 4: {noise_column} [{dataset_name}]")
                p4_rows = []
                for noise_value in noise_levels:
                    for frac in SCALING_LEVELS:
                        X_sub, y_sub = scaled_data[frac]
                        X_sub = X_sub.copy()
                        y_sub = y_sub.copy()
                        if noise_value > 0:
                            X_sub, y_sub = noise_func(X_sub, y_sub, noise_value)
                        for name, model in models.items():
                            run_id = get_run_id(phase_tag, dataset_name, name, frac, noise_value)
                            if run_id in completed:
                                p4_rows.append(completed[run_id])
                                continue
                            acc, f1, auc, auc_std, auc_reps = evaluate_model(
                                model, X_sub, y_sub, seed_split_base
                            )
                            row = build_row(
                                f"{noise_column}_interaction", dataset_name, name,
                                frac, acc, f1, auc, auc_std, auc_reps,
                                extra={noise_column: noise_value}
                            )
                            p4_rows.append(row)
                            completed[run_id] = row
                            save_checkpoint(checkpoint)
                pd.DataFrame(p4_rows).to_csv(
                    os.path.join(RESULTS_DIR, filename), index=False
                )

            # ── Phase 5: Enrichment ───────────────────────────────────────
            print_phase(f"PHASE 5: Noise analysis enrichment [{dataset_name}]")
            run_phase5_enrichment(dataset_name)

    # ── Phase 6: Statistical tests ─────────────────────────────────────────
    if RUN_STATISTICAL_TESTS:
        print_phase("PHASE 6: Statistical testing")
        if not dataset_names:
            dataset_names = ["creditcard", "give_me_some_credit", "adult_income"]
        run_statistical_tests(dataset_names, REPEATS)

    # ── Phase 7: Visualization ─────────────────────────────────────────────
    if RUN_VISUALIZATION:
        print_phase("PHASE 7: Visualization")
        run_visualization_pipeline()

    # ── Phase 8: Threshold detection ───────────────────────────────────────
    if RUN_THRESHOLD_DETECTION:
        print_phase("PHASE 8: Threshold detection")
        if not dataset_names:
            dataset_names = ["creditcard", "give_me_some_credit", "adult_income"]
        run_threshold_detection(dataset_names)


if __name__ == "__main__":
    main()