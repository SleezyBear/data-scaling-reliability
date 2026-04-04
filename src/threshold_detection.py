import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.signal import savgol_filter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ── 1. Saturation point ────────────────────────────────────────────────────

def detect_saturation_point(fractions, auc_values, smooth=True):
    fractions = np.array(fractions)
    auc_values = np.array(auc_values)
    if smooth and len(auc_values) >= 5:
        auc_smooth = savgol_filter(auc_values, window_length=5, polyorder=2)
    else:
        auc_smooth = auc_values
    gains = np.diff(auc_smooth) / np.diff(fractions)
    THRESHOLD = 0.005 / 0.2  # 0.5% AUC gain per 20% fraction step
    for i, gain in enumerate(gains):
        if gain < THRESHOLD:
            return float(fractions[i + 1])
    return None


def run_saturation_detection(dataset_name):
    path = os.path.join(RESULTS_DIR, f"{dataset_name}_phase2_clean_scaling_results.csv")
    if not os.path.exists(path):
        print(f"  [saturation] {path} not found, skipping.")
        return []
    df = pd.read_csv(path)
    results = []
    for model in df["model"].unique():
        mdf = df[df["model"] == model].sort_values("dataset_fraction")
        sat = detect_saturation_point(
            mdf["dataset_fraction"].values,
            mdf["mean_auc"].values
        )
        results.append({
            "dataset": dataset_name,
            "model": model,
            "saturation_fraction": sat if sat is not None else "none"
        })
        print(f"  [{dataset_name}] {model}: saturation at fraction={sat}")
    return results


# ── 2. Noise tolerance boundary ────────────────────────────────────────────

def detect_noise_tolerance_boundary(df, noise_column, model,
                                     dataset_fraction, alpha=0.05):
    sub = df[
        (df["model"] == model) &
        (df["dataset_fraction"] == dataset_fraction)
    ].sort_values(noise_column)

    rep_cols = [c for c in df.columns if c.startswith("auc_rep_")]
    if not rep_cols:
        return None

    clean_row = sub[sub[noise_column] == 0.0]
    if clean_row.empty:
        return None
    clean_scores = clean_row[rep_cols].values.flatten()

    for _, row in sub.iterrows():
        if row[noise_column] == 0.0:
            continue
        noisy_scores = row[rep_cols].values.flatten()
        try:
            _, p = wilcoxon(clean_scores, noisy_scores, alternative="greater")
        except ValueError:
            continue
        if p < alpha:
            return float(row[noise_column])
    return None


def run_noise_tolerance_detection(dataset_name):
    results = []
    noise_specs = [
        (f"{dataset_name}_phase4_label_noise_results.csv", "label_noise"),
        (f"{dataset_name}_phase4_feature_noise_results.csv", "feature_noise_std"),
        (f"{dataset_name}_phase4_weak_sample_results.csv", "weak_fraction"),
    ]
    for filename, noise_column in noise_specs:
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            print(f"  [tolerance] {path} not found, skipping.")
            continue
        df = pd.read_csv(path)
        for model in df["model"].unique():
            for frac in sorted(df["dataset_fraction"].unique()):
                boundary = detect_noise_tolerance_boundary(
                    df, noise_column, model, frac
                )
                results.append({
                    "dataset": dataset_name,
                    "noise_type": noise_column,
                    "model": model,
                    "dataset_fraction": frac,
                    "tolerance_boundary": boundary if boundary is not None else "none"
                })
                print(
                    f"  [{dataset_name}] {model} @ {frac:.0%} | "
                    f"{noise_column}: boundary={boundary}"
                )
    return results


# ── 3. Inflection point ────────────────────────────────────────────────────

def detect_inflection_point(noise_levels, auc_values, min_curvature=0.002):
    noise_levels = np.array(noise_levels)
    auc_values = np.array(auc_values)
    if len(auc_values) < 3:
        return None
    second_diff = np.diff(auc_values, n=2)
    if second_diff.size == 0 or np.min(second_diff) > -min_curvature:
        return None
    idx = int(np.argmin(second_diff))
    return float(noise_levels[idx + 1])


def run_inflection_detection(dataset_name):
    results = []
    noise_specs = [
        (f"{dataset_name}_phase4_label_noise_results.csv", "label_noise"),
        (f"{dataset_name}_phase4_feature_noise_results.csv", "feature_noise_std"),
        (f"{dataset_name}_phase4_weak_sample_results.csv", "weak_fraction"),
    ]
    for filename, noise_column in noise_specs:
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            print(f"  [inflection] {path} not found, skipping.")
            continue
        df = pd.read_csv(path)
        for model in df["model"].unique():
            for frac in sorted(df["dataset_fraction"].unique()):
                sub = df[
                    (df["model"] == model) &
                    (df["dataset_fraction"] == frac)
                ].sort_values(noise_column)
                inflection = detect_inflection_point(
                    sub[noise_column].values,
                    sub["mean_auc"].values
                )
                results.append({
                    "dataset": dataset_name,
                    "noise_type": noise_column,
                    "model": model,
                    "dataset_fraction": frac,
                    "inflection_point": inflection if inflection is not None else "none"
                })
                print(
                    f"  [{dataset_name}] {model} @ {frac:.0%} | "
                    f"{noise_column}: inflection={inflection}"
                )
    return results


# ── Master runner ──────────────────────────────────────────────────────────

def run_threshold_detection(dataset_names):
    all_saturation = []
    all_tolerance  = []
    all_inflection = []

    for dataset_name in dataset_names:
        print(f"\n  Saturation detection: {dataset_name}")
        all_saturation.extend(run_saturation_detection(dataset_name))

        print(f"\n  Noise tolerance detection: {dataset_name}")
        all_tolerance.extend(run_noise_tolerance_detection(dataset_name))

        print(f"\n  Inflection point detection: {dataset_name}")
        all_inflection.extend(run_inflection_detection(dataset_name))

    pd.DataFrame(all_saturation).to_csv(
        os.path.join(RESULTS_DIR, "phase8_saturation_points.csv"), index=False)
    pd.DataFrame(all_tolerance).to_csv(
        os.path.join(RESULTS_DIR, "phase8_noise_tolerance_boundaries.csv"), index=False)
    pd.DataFrame(all_inflection).to_csv(
        os.path.join(RESULTS_DIR, "phase8_inflection_points.csv"), index=False)

    print("\n  Saved: phase8_saturation_points.csv")
    print("  Saved: phase8_noise_tolerance_boundaries.csv")
    print("  Saved: phase8_inflection_points.csv")