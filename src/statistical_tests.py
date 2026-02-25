import os
import numpy as np
import pandas as pd
from scipy.stats import t

INPUT_FILE = os.path.join("results", "phase4_label_noise_results.csv")
OUTPUT_FILE = os.path.join("results", "phase7_statistical_tests.csv")

def summary_t_test(mean1, std1, mean2, std2, n):
    se = np.sqrt((std1**2 / n) + (std2**2 / n))
    if se == 0:
        return 0.0, 1.0
    t_stat = (mean1 - mean2) / se
    df = 2 * n - 2
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    return t_stat, p_value

def run_statistical_tests(repeats):
    df = pd.read_csv(INPUT_FILE)
    results = []

    models = df["model"].unique()
    dataset_sizes = sorted(df["dataset_fraction"].unique())
    noise_levels = sorted(df["label_noise"].unique())

    for model in models:
        for size in dataset_sizes:
            clean_row = df[
                (df["model"] == model) &
                (df["dataset_fraction"] == size) &
                (df["label_noise"] == 0.0)
            ]
            if clean_row.empty:
                continue
            mean_clean = clean_row["mean_auc"].values[0]
            std_clean = clean_row["auc_std"].values[0]
            for noise in noise_levels:
                if noise == 0.0:
                    continue
                noisy_row = df[
                    (df["model"] == model) &
                    (df["dataset_fraction"] == size) &
                    (df["label_noise"] == noise)
                ]
                if noisy_row.empty:
                    continue
                mean_noisy = noisy_row["mean_auc"].values[0]
                std_noisy = noisy_row["auc_std"].values[0]
                t_stat, p = summary_t_test(
                    mean_clean, std_clean,
                    mean_noisy, std_noisy,
                    repeats
                )
                results.append({
                    "test_type": "clean_vs_noise",
                    "model": model,
                    "dataset_fraction": size,
                    "label_noise": noise,
                    "t_stat": t_stat,
                    "p_value": p,
                    "significant": p < 0.05
                })

    min_size = dataset_sizes[0]
    max_size = dataset_sizes[-1]

    for model in models:
        for noise in noise_levels:
            row_small = df[
                (df["model"] == model) &
                (df["dataset_fraction"] == min_size) &
                (df["label_noise"] == noise)
            ]
            row_large = df[
                (df["model"] == model) &
                (df["dataset_fraction"] == max_size) &
                (df["label_noise"] == noise)
            ]
            if row_small.empty or row_large.empty:
                continue
            mean_small = row_small["mean_auc"].values[0]
            std_small = row_small["auc_std"].values[0]
            mean_large = row_large["mean_auc"].values[0]
            std_large = row_large["auc_std"].values[0]
            t_stat, p = summary_t_test(
                mean_large, std_large,
                mean_small, std_small,
                repeats
            )
            results.append({
                "test_type": "scaling_effect",
                "model": model,
                "dataset_fraction": f"{min_size}_vs_{max_size}",
                "label_noise": noise,
                "t_stat": t_stat,
                "p_value": p,
                "significant": p < 0.05
            })

    if len(models) >= 2:
        model_a = models[0]
        model_b = models[1]
        for size in dataset_sizes:
            for noise in noise_levels:
                row_a = df[
                    (df["model"] == model_a) &
                    (df["dataset_fraction"] == size) &
                    (df["label_noise"] == noise)
                ]
                row_b = df[
                    (df["model"] == model_b) &
                    (df["dataset_fraction"] == size) &
                    (df["label_noise"] == noise)
                ]
                if row_a.empty or row_b.empty:
                    continue
                mean_a = row_a["mean_auc"].values[0]
                std_a = row_a["auc_std"].values[0]
                mean_b = row_b["mean_auc"].values[0]
                std_b = row_b["auc_std"].values[0]
                t_stat, p = summary_t_test(
                    mean_a, std_a,
                    mean_b, std_b,
                    repeats
                )
                results.append({
                    "test_type": "model_comparison",
                    "model": f"{model_a}_vs_{model_b}",
                    "dataset_fraction": size,
                    "label_noise": noise,
                    "t_stat": t_stat,
                    "p_value": p,
                    "significant": p < 0.05
                })

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print("Phase 7 statistical testing complete.")