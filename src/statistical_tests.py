import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def get_rep_scores(row, rep_cols):
    return row[rep_cols].values.astype(float)


def wilcoxon_test(scores_a, scores_b, alternative="two-sided"):
    try:
        stat, p = wilcoxon(scores_a, scores_b, alternative=alternative)
        return float(stat), float(p)
    except ValueError:
        return float("nan"), 1.0


def summarize_paired_effect(scores_a, scores_b):
    deltas = np.array(scores_a, dtype=float) - np.array(scores_b, dtype=float)
    return {
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
    }


def holm_adjust(p_values):
    p = np.array(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    for i, idx in enumerate(order):
        adjusted[idx] = min(1.0, (m - i) * p[idx])

    # Enforce monotonicity in sorted order.
    for i in range(1, m):
        prev_idx = order[i - 1]
        curr_idx = order[i]
        adjusted[curr_idx] = max(adjusted[curr_idx], adjusted[prev_idx])

    return adjusted


def fdr_bh_adjust(p_values):
    p = np.array(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p

    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.empty(m, dtype=float)

    for i in range(m - 1, -1, -1):
        rank = i + 1
        value = (m / rank) * sorted_p[i]
        if i == m - 1:
            adjusted_sorted[i] = value
        else:
            adjusted_sorted[i] = min(value, adjusted_sorted[i + 1])

    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def apply_multiple_testing_corrections(results):
    if not results:
        return pd.DataFrame(results)

    df = pd.DataFrame(results)
    if "p_raw" not in df.columns:
        df["p_raw"] = df["p_value"]

    df["p_holm"] = np.nan
    df["p_fdr_bh"] = np.nan

    group_cols = ["test_type", "dataset", "noise_type"]
    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        pvals = df.loc[idx, "p_raw"].astype(float).values
        mask = np.isfinite(pvals)
        if not np.any(mask):
            continue

        valid_positions = np.where(mask)[0]
        valid_pvals = pvals[mask]
        holm_vals = holm_adjust(valid_pvals)
        fdr_vals = fdr_bh_adjust(valid_pvals)

        for pos, p_holm, p_fdr in zip(valid_positions, holm_vals, fdr_vals):
            row_idx = idx[pos]
            df.at[row_idx, "p_holm"] = float(p_holm)
            df.at[row_idx, "p_fdr_bh"] = float(p_fdr)

    # Primary decision column uses Holm (family-wise control).
    df["p_value"] = df["p_holm"].fillna(df["p_raw"])
    df["significant_uncorrected"] = df["p_raw"] < 0.05
    df["significant_holm"] = df["p_holm"] < 0.05
    df["significant_fdr_bh"] = df["p_fdr_bh"] < 0.05
    df["significant"] = df["significant_holm"].fillna(df["significant_uncorrected"])

    return df


def run_wilcoxon_clean_vs_noise(df, noise_column, dataset_name, rep_cols):
    results = []
    for model in df["model"].unique():
        for frac in sorted(df["dataset_fraction"].unique()):
            clean_row = df[
                (df["model"] == model) &
                (df["dataset_fraction"] == frac) &
                (df[noise_column] == 0.0)
            ]
            if clean_row.empty:
                continue
            clean_scores = get_rep_scores(clean_row.iloc[0], rep_cols)

            for noise_val in sorted(df[noise_column].unique()):
                if noise_val == 0.0:
                    continue
                noisy_row = df[
                    (df["model"] == model) &
                    (df["dataset_fraction"] == frac) &
                    (df[noise_column] == noise_val)
                ]
                if noisy_row.empty:
                    continue
                noisy_scores = get_rep_scores(noisy_row.iloc[0], rep_cols)
                stat, p = wilcoxon_test(clean_scores, noisy_scores, alternative="two-sided")
                effect = summarize_paired_effect(clean_scores, noisy_scores)
                results.append({
                    "test_type": "clean_vs_noise",
                    "dataset": dataset_name,
                    "noise_type": noise_column,
                    "model": model,
                    "dataset_fraction": frac,
                    "noise_level": noise_val,
                    "statistic": stat,
                    "p_raw": p,
                    "alternative": "two-sided",
                    "mean_delta": effect["mean_delta"],
                    "median_delta": effect["median_delta"],
                })
    return results


def run_wilcoxon_scaling_effect(df, noise_column, dataset_name, rep_cols):
    results = []
    fracs = sorted(df["dataset_fraction"].unique())
    min_frac, max_frac = fracs[0], fracs[-1]

    for model in df["model"].unique():
        for noise_val in sorted(df[noise_column].unique()):
            small_row = df[
                (df["model"] == model) &
                (df["dataset_fraction"] == min_frac) &
                (df[noise_column] == noise_val)
            ]
            large_row = df[
                (df["model"] == model) &
                (df["dataset_fraction"] == max_frac) &
                (df[noise_column] == noise_val)
            ]
            if small_row.empty or large_row.empty:
                continue
            large_scores = get_rep_scores(large_row.iloc[0], rep_cols)
            small_scores = get_rep_scores(small_row.iloc[0], rep_cols)
            stat, p = wilcoxon_test(
                large_scores,
                small_scores,
                alternative="two-sided"
            )
            effect = summarize_paired_effect(large_scores, small_scores)
            results.append({
                "test_type": "scaling_effect",
                "dataset": dataset_name,
                "noise_type": noise_column,
                "model": model,
                "dataset_fraction": f"{min_frac}_vs_{max_frac}",
                "noise_level": noise_val,
                "statistic": stat,
                "p_raw": p,
                "alternative": "two-sided",
                "mean_delta": effect["mean_delta"],
                "median_delta": effect["median_delta"],
            })
    return results


def run_pairwise_wilcoxon_bonferroni(model_names, all_model_scores):
    results = []
    pairs = list(itertools.combinations(range(len(model_names)), 2))
    n_comparisons = len(pairs)
    for i, j in pairs:
        stat, p = wilcoxon_test(
            all_model_scores[i], all_model_scores[j], alternative="two-sided"
        )
        p_corrected = min(p * n_comparisons, 1.0)
        effect = summarize_paired_effect(all_model_scores[i], all_model_scores[j])
        results.append({
            "model_a": model_names[i],
            "model_b": model_names[j],
            "statistic": stat,
            "p_raw": p,
            "p_bonferroni": p_corrected,
            "alternative": "two-sided",
            "mean_delta": effect["mean_delta"],
            "median_delta": effect["median_delta"],
            "significant_bonferroni": p_corrected < 0.05
        })
    return results


def run_friedman_pairwise_posthoc(df, noise_column, dataset_name, rep_cols):
    results = []
    model_names = list(df["model"].unique())

    for frac in sorted(df["dataset_fraction"].unique()):
        for noise_val in sorted(df[noise_column].unique()):
            model_scores = []
            valid_models = []

            for model in model_names:
                row = df[
                    (df["model"] == model) &
                    (df["dataset_fraction"] == frac) &
                    (df[noise_column] == noise_val)
                ]
                if not row.empty:
                    model_scores.append(get_rep_scores(row.iloc[0], rep_cols))
                    valid_models.append(model)

            if len(valid_models) < 3:
                continue

            try:
                f_stat, f_p = friedmanchisquare(*model_scores)
                f_stat, f_p = float(f_stat), float(f_p)
            except ValueError:
                f_stat, f_p = float("nan"), 1.0

            results.append({
                "test_type": "friedman",
                "dataset": dataset_name,
                "noise_type": noise_column,
                "dataset_fraction": frac,
                "noise_level": noise_val,
                "models": "|".join(valid_models),
                "statistic": f_stat,
                "p_raw": f_p,
                "alternative": "k-sample",
            })

            if f_p < 0.05:
                for ph in run_pairwise_wilcoxon_bonferroni(valid_models, model_scores):
                    results.append({
                        "test_type": "pairwise_wilcoxon_bonferroni",
                        "dataset": dataset_name,
                        "noise_type": noise_column,
                        "dataset_fraction": frac,
                        "noise_level": noise_val,
                        "models": f"{ph['model_a']}_vs_{ph['model_b']}",
                        "statistic": ph["statistic"],
                        "p_raw": ph["p_raw"],
                        "p_bonferroni": ph["p_bonferroni"],
                        "alternative": ph["alternative"],
                        "mean_delta": ph["mean_delta"],
                        "median_delta": ph["median_delta"],
                        "significant_bonferroni": ph["significant_bonferroni"],
                    })

    return results


def run_friedman_across_datasets(dataset_names):
    rows = []
    for dataset_name in dataset_names:
        path = os.path.join(RESULTS_DIR, f"{dataset_name}_phase1_baseline.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "model" not in df.columns or "mean_auc" not in df.columns:
            continue
        for _, row in df.iterrows():
            rows.append({
                "dataset": dataset_name,
                "model": row["model"],
                "mean_auc": float(row["mean_auc"]),
            })

    if not rows:
        return []

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="dataset", columns="model", values="mean_auc")
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return []

    rank_df = pivot.rank(axis=1, ascending=False, method="average")
    model_names = list(rank_df.columns)
    model_rank_vectors = [rank_df[m].values for m in model_names]

    results = []
    try:
        f_stat, f_p = friedmanchisquare(*model_rank_vectors)
        f_stat, f_p = float(f_stat), float(f_p)
    except ValueError:
        f_stat, f_p = float("nan"), 1.0

    results.append({
        "test_type": "friedman_across_datasets",
        "dataset": "ALL_DATASETS",
        "noise_type": "clean_full",
        "dataset_fraction": 1.0,
        "noise_level": 0.0,
        "models": "|".join(model_names),
        "statistic": f_stat,
        "p_raw": f_p,
        "alternative": "k-sample",
        "n_datasets": int(rank_df.shape[0]),
    })

    if f_p < 0.05:
        pairwise = run_pairwise_wilcoxon_bonferroni(model_names, model_rank_vectors)
        for ph in pairwise:
            results.append({
                "test_type": "pairwise_rank_wilcoxon_bonferroni",
                "dataset": "ALL_DATASETS",
                "noise_type": "clean_full",
                "dataset_fraction": 1.0,
                "noise_level": 0.0,
                "models": f"{ph['model_a']}_vs_{ph['model_b']}",
                "statistic": ph["statistic"],
                "p_raw": ph["p_raw"],
                "p_bonferroni": ph["p_bonferroni"],
                "alternative": ph["alternative"],
                "mean_delta": ph["mean_delta"],
                "median_delta": ph["median_delta"],
                "significant_bonferroni": ph["significant_bonferroni"],
                "n_datasets": int(rank_df.shape[0]),
            })

    rank_out = rank_df.reset_index().melt(
        id_vars="dataset", var_name="model", value_name="rank"
    )
    rank_out.to_csv(
        os.path.join(RESULTS_DIR, "phase6_dataset_model_ranks.csv"), index=False
    )

    return results


def run_statistical_tests(dataset_names, repeats):
    rep_cols_base = [f"auc_rep_{i}" for i in range(repeats)]
    all_results = []

    noise_specs = [
        ("phase4_label_noise_results.csv", "label_noise"),
        ("phase4_feature_noise_results.csv", "feature_noise_std"),
        ("phase4_weak_sample_results.csv", "weak_fraction"),
    ]

    for dataset_name in dataset_names:
        print(f"\n  Statistical tests: {dataset_name}")
        for filename, noise_column in noise_specs:
            path = os.path.join(RESULTS_DIR, f"{dataset_name}_{filename}")
            if not os.path.exists(path):
                print(f"    {path} not found, skipping.")
                continue

            df = pd.read_csv(path)
            rep_cols = [c for c in rep_cols_base if c in df.columns]
            if not rep_cols:
                print(f"    No per-repeat columns in {path}, skipping.")
                continue

            print(f"    Wilcoxon clean-vs-noise ({noise_column})...")
            all_results.extend(
                run_wilcoxon_clean_vs_noise(df, noise_column, dataset_name, rep_cols)
            )
            print(f"    Wilcoxon scaling effect ({noise_column})...")
            all_results.extend(
                run_wilcoxon_scaling_effect(df, noise_column, dataset_name, rep_cols)
            )
            print(f"    Friedman + pairwise Wilcoxon (Bonferroni) ({noise_column})...")
            all_results.extend(
                run_friedman_pairwise_posthoc(df, noise_column, dataset_name, rep_cols)
            )

    print("\n  Cross-dataset model ranking tests (clean full baseline)...")
    all_results.extend(run_friedman_across_datasets(dataset_names))

    corrected_df = apply_multiple_testing_corrections(all_results)

    out_path = os.path.join(RESULTS_DIR, "phase6_statistical_tests.csv")
    corrected_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")