import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pandas.plotting import parallel_coordinates


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

DATASET_NAMES = ["creditcard", "give_me_some_credit", "adult_income"]

DATASET_LABELS = {
    "creditcard": "Credit Card Fraud (0.17%)",
    "give_me_some_credit": "Give Me Some Credit (6.7%)",
    "adult_income": "Adult Income (24%)",
}

NOISE_SPECS = [
    ("phase5_label_noise_analysis.csv", "label_noise", "Label Noise"),
    ("phase5_feature_noise_analysis.csv", "feature_noise_std", "Feature Noise"),
    ("phase5_weak_sample_analysis.csv", "weak_fraction", "Weak Samples"),
]


def get_colors(n):
    return [cm.tab10(i / max(n, 1)) for i in range(n)]


def _read_csv_or_none(path):
    if not os.path.exists(path):
        print(f"  [skip] {path}")
        return None
    return pd.read_csv(path)


def _dataset_label(dataset_name):
    return DATASET_LABELS.get(dataset_name, dataset_name)


def _to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _coerce_threshold_value(series):
    return pd.to_numeric(series.replace("none", np.nan), errors="coerce")


def _safe_noise_tag(noise_label):
    return noise_label.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Basic plots
# ---------------------------------------------------------------------------

def plot_baseline_model_bars_per_dataset():
    for dataset_name in DATASET_NAMES:
        path = os.path.join(RESULTS_DIR, f"{dataset_name}_phase1_baseline.csv")
        df = _read_csv_or_none(path)
        if df is None or df.empty:
            continue

        needed = {"model", "mean_auc", "auc_std"}
        if not needed.issubset(df.columns):
            print(f"  [skip] Missing columns in {path}")
            continue

        df = df.sort_values("mean_auc", ascending=True)
        colors = get_colors(len(df))

        plt.figure(figsize=(10, 5))
        plt.barh(df["model"], df["mean_auc"], color=colors, edgecolor="grey")
        plt.errorbar(
            df["mean_auc"],
            df["model"],
            xerr=df["auc_std"],
            fmt="none",
            color="black",
            capsize=4,
        )
        plt.xlabel("Mean ROC-AUC")
        plt.title(f"Baseline Model Comparison | {_dataset_label(dataset_name)}")
        plt.xlim(max(0, float(df["mean_auc"].min()) - 0.05), 1.0)
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"{dataset_name}_baseline_model_comparison_bar.png"),
            dpi=180,
        )
        plt.close()


def plot_clean_scaling_lines_per_dataset():
    for dataset_name in DATASET_NAMES:
        path = os.path.join(RESULTS_DIR, f"{dataset_name}_phase2_clean_scaling_results.csv")
        df = _read_csv_or_none(path)
        if df is None or df.empty:
            continue

        needed = {"model", "dataset_fraction", "mean_auc"}
        if not needed.issubset(df.columns):
            print(f"  [skip] Missing columns in {path}")
            continue

        models = list(df["model"].unique())
        colors = get_colors(len(models))

        plt.figure(figsize=(9, 5))
        for i, model in enumerate(models):
            sub = df[df["model"] == model].sort_values("dataset_fraction")
            plt.plot(
                sub["dataset_fraction"],
                sub["mean_auc"],
                marker="o",
                linewidth=1.8,
                label=model,
                color=colors[i],
            )

        plt.xlabel("Dataset Fraction")
        plt.ylabel("Mean ROC-AUC")
        plt.title(f"Clean Scaling Curves | {_dataset_label(dataset_name)}")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"{dataset_name}_clean_scaling_lines_all_models.png"),
            dpi=180,
        )
        plt.close()


def plot_clean_scaling_cross_dataset_per_model():
    all_dfs = []
    for dataset_name in DATASET_NAMES:
        path = os.path.join(RESULTS_DIR, f"{dataset_name}_phase2_clean_scaling_results.csv")
        df = _read_csv_or_none(path)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["dataset"] = dataset_name
        all_dfs.append(df)

    if not all_dfs:
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    models = list(combined["model"].unique())
    colors = get_colors(len(DATASET_NAMES))

    for model in models:
        plt.figure(figsize=(9, 5))
        mdf = combined[combined["model"] == model]
        for i, dataset_name in enumerate(DATASET_NAMES):
            sub = mdf[mdf["dataset"] == dataset_name].sort_values("dataset_fraction")
            if sub.empty:
                continue
            plt.plot(
                sub["dataset_fraction"],
                sub["mean_auc"],
                marker="o",
                linewidth=1.8,
                label=_dataset_label(dataset_name),
                color=colors[i],
            )
        plt.xlabel("Dataset Fraction")
        plt.ylabel("Mean ROC-AUC")
        plt.title(f"Clean Scaling Across Datasets | {model}")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"cross_dataset_clean_scaling_{model}.png"), dpi=180)
        plt.close()


def plot_noise_response_curves():
    for dataset_name in DATASET_NAMES:
        for phase5_file, noise_col, noise_label in NOISE_SPECS:
            path = os.path.join(RESULTS_DIR, f"{dataset_name}_{phase5_file}")
            df = _read_csv_or_none(path)
            if df is None or df.empty:
                continue

            needed = {"model", "dataset_fraction", noise_col, "mean_auc", "auc_std"}
            if not needed.issubset(df.columns):
                print(f"  [skip] Missing columns in {path}")
                continue

            fracs = sorted(df["dataset_fraction"].unique())
            models = list(df["model"].unique())
            colors = get_colors(len(models))

            ncols = 2
            nrows = int(np.ceil(len(fracs) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows), sharey=True)
            axes = np.array(axes).reshape(-1)

            for idx, frac in enumerate(fracs):
                ax = axes[idx]
                fdf = df[df["dataset_fraction"] == frac].sort_values(noise_col)
                for i, model in enumerate(models):
                    sub = fdf[fdf["model"] == model]
                    if sub.empty:
                        continue
                    x = sub[noise_col].values
                    y = sub["mean_auc"].values
                    s = sub["auc_std"].values
                    ax.plot(x, y, marker="o", linewidth=1.6, color=colors[i], label=model)
                    ax.fill_between(x, y - s, y + s, color=colors[i], alpha=0.12)

                ax.set_title(f"Fraction={frac:.1f}")
                ax.set_xlabel(noise_label)
                ax.set_ylabel("Mean ROC-AUC")
                ax.grid(alpha=0.25)

            for idx in range(len(fracs), len(axes)):
                axes[idx].axis("off")

            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
            fig.suptitle(f"Noise Response Curves | {_dataset_label(dataset_name)} | {noise_label}", y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(
                os.path.join(PLOTS_DIR, f"{dataset_name}_{_safe_noise_tag(noise_label)}_noise_response_grid.png"),
                dpi=180,
            )
            plt.close()


def plot_cv_profiles():
    for dataset_name in DATASET_NAMES:
        for phase5_file, noise_col, noise_label in NOISE_SPECS:
            path = os.path.join(RESULTS_DIR, f"{dataset_name}_{phase5_file}")
            df = _read_csv_or_none(path)
            if df is None or df.empty or "cv" not in df.columns:
                continue

            models = list(df["model"].unique())
            colors = get_colors(len(models))
            fracs = sorted(df["dataset_fraction"].unique())
            target_frac = fracs[-1]

            plt.figure(figsize=(9, 5))
            fdf = df[df["dataset_fraction"] == target_frac].sort_values(noise_col)
            for i, model in enumerate(models):
                sub = fdf[fdf["model"] == model]
                if sub.empty:
                    continue
                plt.plot(
                    sub[noise_col],
                    sub["cv"],
                    marker="o",
                    linewidth=1.6,
                    color=colors[i],
                    label=model,
                )

            plt.xlabel(noise_label)
            plt.ylabel("Coefficient of Variation")
            plt.title(
                f"Stability Profile (CV) at Fraction={target_frac:.1f} | "
                f"{_dataset_label(dataset_name)} | {noise_label}"
            )
            plt.grid(alpha=0.3)
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(
                os.path.join(PLOTS_DIR, f"{dataset_name}_{_safe_noise_tag(noise_label)}_cv_profile.png"),
                dpi=180,
            )
            plt.close()


def plot_relative_drop_heatmaps():
    for dataset_name in DATASET_NAMES:
        for phase5_file, noise_col, noise_label in NOISE_SPECS:
            path = os.path.join(RESULTS_DIR, f"{dataset_name}_{phase5_file}")
            df = _read_csv_or_none(path)
            if df is None or df.empty or "relative_auc_drop" not in df.columns:
                continue

            for model in df["model"].unique():
                mdf = df[df["model"] == model]
                pivot = mdf.pivot_table(
                    index=noise_col,
                    columns="dataset_fraction",
                    values="relative_auc_drop",
                )
                if pivot.empty:
                    continue

                plt.figure(figsize=(8.8, 5.6))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", cbar_kws={"label": "Relative AUC Drop"})
                plt.title(f"Relative Performance Drop | {model} | {_dataset_label(dataset_name)} | {noise_label}")
                plt.xlabel("Dataset Fraction")
                plt.ylabel(noise_label)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(PLOTS_DIR, f"{dataset_name}_{_safe_noise_tag(noise_label)}_relative_drop_heatmap_{model}.png"),
                    dpi=180,
                )
                plt.close()


def plot_breakdown_probability_curves():
    for dataset_name in DATASET_NAMES:
        for phase5_file, noise_col, noise_label in NOISE_SPECS:
            path = os.path.join(RESULTS_DIR, f"{dataset_name}_{phase5_file}")
            df = _read_csv_or_none(path)
            if df is None or df.empty or "breakdown_flag" not in df.columns:
                continue

            models = list(df["model"].unique())
            colors = get_colors(len(models))

            plt.figure(figsize=(9, 5))
            for i, model in enumerate(models):
                mdf = df[df["model"] == model]
                # Averaged across all noise levels for a compact risk summary.
                risk = mdf.groupby("dataset_fraction")["breakdown_flag"].mean().reset_index()
                plt.plot(
                    risk["dataset_fraction"],
                    risk["breakdown_flag"],
                    marker="o",
                    linewidth=1.6,
                    color=colors[i],
                    label=model,
                )

            plt.xlabel("Dataset Fraction")
            plt.ylabel("Breakdown Probability")
            plt.title(f"Breakdown Risk Summary | {_dataset_label(dataset_name)} | {noise_label}")
            plt.ylim(-0.02, 1.02)
            plt.grid(alpha=0.3)
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(
                os.path.join(PLOTS_DIR, f"{dataset_name}_{_safe_noise_tag(noise_label)}_breakdown_probability_curves.png"),
                dpi=180,
            )
            plt.close()


def plot_cross_dataset_ranking_heatmap():
    rows = []
    for dataset_name in DATASET_NAMES:
        path = os.path.join(RESULTS_DIR, f"{dataset_name}_phase1_baseline.csv")
        df = _read_csv_or_none(path)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            rows.append(
                {
                    "dataset": _dataset_label(dataset_name),
                    "model": row["model"],
                    "mean_auc": row["mean_auc"],
                }
            )

    if not rows:
        return

    combined = pd.DataFrame(rows)
    pivot = combined.pivot_table(index="model", columns="dataset", values="mean_auc")
    if pivot.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={"label": "Mean ROC-AUC"})
    plt.title("Cross-Dataset Baseline Model Ranking")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cross_dataset_baseline_model_ranking_heatmap.png"), dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# Intermediate plots
# ---------------------------------------------------------------------------

def plot_robustness_small_multiples():
    for phase5_file, noise_col, noise_label in NOISE_SPECS:
        dfs = {}
        model_union = set()
        frac_union = set()

        for dataset_name in DATASET_NAMES:
            path = os.path.join(RESULTS_DIR, f"{dataset_name}_{phase5_file}")
            df = _read_csv_or_none(path)
            if df is None or df.empty:
                continue
            dfs[dataset_name] = df
            model_union.update(df["model"].unique())
            frac_union.update(df["dataset_fraction"].unique())

        if not dfs:
            continue

        models = sorted(model_union)
        fracs = sorted(frac_union)
        colors = get_colors(len(models))

        for frac in fracs:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
            for i, dataset_name in enumerate(DATASET_NAMES):
                ax = axes[i]
                if dataset_name not in dfs:
                    ax.axis("off")
                    continue
                ddf = dfs[dataset_name]
                subf = ddf[ddf["dataset_fraction"] == frac].sort_values(noise_col)
                if subf.empty:
                    ax.axis("off")
                    continue

                for j, model in enumerate(models):
                    mdf = subf[subf["model"] == model]
                    if mdf.empty:
                        continue
                    ax.plot(mdf[noise_col], mdf["mean_auc"], marker="o", linewidth=1.5, color=colors[j], label=model)

                ax.set_title(_dataset_label(dataset_name))
                ax.set_xlabel(noise_label)
                ax.set_ylabel("Mean ROC-AUC")
                ax.grid(alpha=0.25)

            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
            fig.suptitle(f"Robustness Small Multiples | {noise_label} | Fraction={frac:.1f}", y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            plt.savefig(
                os.path.join(PLOTS_DIR, f"robustness_small_multiples_{_safe_noise_tag(noise_label)}_frac_{str(frac).replace('.', 'p')}.png"),
                dpi=180,
            )
            plt.close()


def plot_pairwise_delta_against_best():
    for dataset_name in DATASET_NAMES:
        baseline_path = os.path.join(RESULTS_DIR, f"{dataset_name}_phase1_baseline.csv")
        bdf = _read_csv_or_none(baseline_path)
        if bdf is None or bdf.empty:
            continue

        best_model = bdf.sort_values("mean_auc", ascending=False).iloc[0]["model"]

        for phase5_file, noise_col, noise_label in NOISE_SPECS:
            path = os.path.join(RESULTS_DIR, f"{dataset_name}_{phase5_file}")
            df = _read_csv_or_none(path)
            if df is None or df.empty:
                continue

            fracs = sorted(df["dataset_fraction"].unique())
            target_frac = fracs[-1]
            sub = df[df["dataset_fraction"] == target_frac].copy()
            if sub.empty:
                continue

            best_sub = sub[sub["model"] == best_model][[noise_col, "mean_auc"]].rename(columns={"mean_auc": "best_auc"})
            merged = sub.merge(best_sub, on=noise_col, how="left")
            merged["delta_vs_best"] = merged["best_auc"] - merged["mean_auc"]

            other_models = [m for m in merged["model"].unique() if m != best_model]
            if not other_models:
                continue
            colors = get_colors(len(other_models))

            plt.figure(figsize=(9, 5))
            for i, model in enumerate(other_models):
                mdf = merged[merged["model"] == model].sort_values(noise_col)
                plt.plot(mdf[noise_col], mdf["delta_vs_best"], marker="o", linewidth=1.6, color=colors[i], label=f"{best_model} - {model}")

            plt.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
            plt.xlabel(noise_label)
            plt.ylabel("Delta AUC vs Best Baseline Model")
            plt.title(
                f"Pairwise Deltas vs {best_model} | {_dataset_label(dataset_name)} | "
                f"{noise_label} | Fraction={target_frac:.1f}"
            )
            plt.grid(alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(
                os.path.join(PLOTS_DIR, f"{dataset_name}_{_safe_noise_tag(noise_label)}_pairwise_delta_vs_best.png"),
                dpi=180,
            )
            plt.close()


def plot_threshold_summary_dotplots():
    sat_path = os.path.join(RESULTS_DIR, "phase8_saturation_points.csv")
    tol_path = os.path.join(RESULTS_DIR, "phase8_noise_tolerance_boundaries.csv")
    inf_path = os.path.join(RESULTS_DIR, "phase8_inflection_points.csv")

    sat = _read_csv_or_none(sat_path)
    tol = _read_csv_or_none(tol_path)
    inf = _read_csv_or_none(inf_path)

    if sat is not None and not sat.empty and "saturation_fraction" in sat.columns:
        sdf = sat.copy()
        sdf["value"] = _coerce_threshold_value(sdf["saturation_fraction"])
        sdf = sdf.dropna(subset=["value"])
        if not sdf.empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=sdf, x="value", y="model", hue="dataset", style="dataset", s=80)
            plt.xlabel("Saturation Fraction")
            plt.ylabel("Model")
            plt.title("Threshold Summary: Saturation Point")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "threshold_summary_saturation_dotplot.png"), dpi=180)
            plt.close()

    if tol is not None and not tol.empty and "tolerance_boundary" in tol.columns:
        tdf = tol.copy()
        tdf["value"] = _coerce_threshold_value(tdf["tolerance_boundary"])
        tdf = tdf.dropna(subset=["value"])
        if not tdf.empty:
            for noise_type in sorted(tdf["noise_type"].unique()):
                ndf = tdf[tdf["noise_type"] == noise_type]
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=ndf, x="value", y="model", hue="dataset", style="dataset", s=75)
                plt.xlabel("Noise Tolerance Boundary")
                plt.ylabel("Model")
                plt.title(f"Threshold Summary: Tolerance Boundary | {noise_type}")
                plt.grid(alpha=0.25)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(PLOTS_DIR, f"threshold_summary_tolerance_dotplot_{noise_type}.png"),
                    dpi=180,
                )
                plt.close()

    if inf is not None and not inf.empty and "inflection_point" in inf.columns:
        idf = inf.copy()
        idf["value"] = _coerce_threshold_value(idf["inflection_point"])
        idf = idf.dropna(subset=["value"])
        if not idf.empty:
            for noise_type in sorted(idf["noise_type"].unique()):
                ndf = idf[idf["noise_type"] == noise_type]
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=ndf, x="value", y="model", hue="dataset", style="dataset", s=75)
                plt.xlabel("Inflection Point")
                plt.ylabel("Model")
                plt.title(f"Threshold Summary: Inflection Point | {noise_type}")
                plt.grid(alpha=0.25)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(PLOTS_DIR, f"threshold_summary_inflection_dotplot_{noise_type}.png"),
                    dpi=180,
                )
                plt.close()


# ---------------------------------------------------------------------------
# Advanced plots
# ---------------------------------------------------------------------------

def _phase6_stats_df():
    path = os.path.join(RESULTS_DIR, "phase6_statistical_tests.csv")
    df = _read_csv_or_none(path)
    if df is None or df.empty:
        return None

    if "significant_holm" not in df.columns:
        if "significant" in df.columns:
            df["significant_holm"] = df["significant"]
        else:
            df["significant_holm"] = False

    if "p_holm" not in df.columns:
        if "p_value" in df.columns:
            df["p_holm"] = df["p_value"]
        else:
            df["p_holm"] = np.nan

    return df


def plot_significance_annotated_effect_heatmaps():
    df = _phase6_stats_df()
    if df is None:
        return

    cdf = df[df["test_type"] == "clean_vs_noise"].copy()
    if cdf.empty or "mean_delta" not in cdf.columns:
        return

    for dataset_name in sorted(cdf["dataset"].dropna().unique()):
        ddf = cdf[cdf["dataset"] == dataset_name]
        for noise_type in sorted(ddf["noise_type"].dropna().unique()):
            ndf = ddf[ddf["noise_type"] == noise_type]
            for model in sorted(ndf["model"].dropna().unique()):
                mdf = ndf[ndf["model"] == model].copy()
                if mdf.empty:
                    continue

                pivot_val = mdf.pivot_table(index="noise_level", columns="dataset_fraction", values="mean_delta")
                pivot_sig = mdf.pivot_table(index="noise_level", columns="dataset_fraction", values="significant_holm", aggfunc="max")
                if pivot_val.empty:
                    continue

                ann = pivot_val.copy().astype(object)
                for i in range(pivot_val.shape[0]):
                    for j in range(pivot_val.shape[1]):
                        val = pivot_val.iloc[i, j]
                        sig = False
                        if i < pivot_sig.shape[0] and j < pivot_sig.shape[1]:
                            sig = bool(pivot_sig.iloc[i, j]) if pd.notna(pivot_sig.iloc[i, j]) else False
                        ann.iloc[i, j] = f"{val:.3f}{'*' if sig else ''}" if pd.notna(val) else ""

                plt.figure(figsize=(8.6, 5.4))
                sns.heatmap(
                    pivot_val,
                    annot=ann,
                    fmt="",
                    cmap="coolwarm",
                    center=0.0,
                    cbar_kws={"label": "Mean Delta (Clean - Noisy)"},
                )
                plt.title(
                    f"Effect Heatmap (+ Holm significance *) | {model} | "
                    f"{_dataset_label(dataset_name)} | {noise_type}"
                )
                plt.xlabel("Dataset Fraction")
                plt.ylabel("Noise Level")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(PLOTS_DIR, f"effect_heatmap_significant_{dataset_name}_{noise_type}_{model}.png"),
                    dpi=180,
                )
                plt.close()


def plot_forest_effect_plots():
    df = _phase6_stats_df()
    if df is None:
        return

    for test_type in ["clean_vs_noise", "scaling_effect"]:
        tdf = df[df["test_type"] == test_type].copy()
        if tdf.empty or "mean_delta" not in tdf.columns:
            continue

        for dataset_name in sorted(tdf["dataset"].dropna().unique()):
            ddf = tdf[tdf["dataset"] == dataset_name]
            for noise_type in sorted(ddf["noise_type"].dropna().unique()):
                ndf = ddf[ddf["noise_type"] == noise_type].copy()
                if ndf.empty:
                    continue

                ndf["effect_spread"] = (ndf["mean_delta"] - ndf.get("median_delta", ndf["mean_delta"])).abs()
                ndf["label"] = (
                    ndf.get("model", "NA").astype(str)
                    + " | frac=" + ndf["dataset_fraction"].astype(str)
                    + " | noise=" + ndf["noise_level"].astype(str)
                )
                ndf = ndf.sort_values("mean_delta")

                max_rows = 24
                ndf = ndf.tail(max_rows)

                plt.figure(figsize=(10, max(5, 0.32 * len(ndf) + 2)))
                y_pos = np.arange(len(ndf))
                plt.errorbar(
                    ndf["mean_delta"],
                    y_pos,
                    xerr=ndf["effect_spread"],
                    fmt="o",
                    ecolor="grey",
                    capsize=3,
                    color="tab:blue",
                )
                sig = ndf["significant_holm"].fillna(False).values
                for i, is_sig in enumerate(sig):
                    if is_sig:
                        plt.scatter(ndf["mean_delta"].iloc[i], y_pos[i], color="red", s=35, zorder=3)
                plt.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
                plt.yticks(y_pos, ndf["label"], fontsize=7)
                plt.xlabel("Mean Delta Effect")
                plt.ylabel("Condition")
                plt.title(
                    f"Forest Effect Plot ({test_type}) | {_dataset_label(dataset_name)} | {noise_type}"
                )
                plt.grid(axis="x", alpha=0.25)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(PLOTS_DIR, f"forest_effect_{test_type}_{dataset_name}_{noise_type}.png"),
                    dpi=180,
                )
                plt.close()


def plot_critical_difference_like_ranking():
    rank_path = os.path.join(RESULTS_DIR, "phase6_dataset_model_ranks.csv")
    stats_df = _phase6_stats_df()
    ranks = _read_csv_or_none(rank_path)
    if stats_df is None or ranks is None or ranks.empty:
        return

    avg_rank = ranks.groupby("model")["rank"].mean().sort_values()
    if avg_rank.empty:
        return

    pr = stats_df[
        (stats_df["test_type"] == "pairwise_rank_wilcoxon_bonferroni")
        & (stats_df["dataset"] == "ALL_DATASETS")
    ].copy()

    plt.figure(figsize=(10, 4.8))
    x = avg_rank.values
    y = np.ones_like(x)
    plt.scatter(x, y, s=70, color="tab:blue")

    for i, model in enumerate(avg_rank.index):
        plt.text(x[i], y[i] + 0.06, model, ha="center", fontsize=9)

    # Draw non-significant pair connectors.
    level = 1.12
    if not pr.empty:
        p_col = "p_bonferroni" if "p_bonferroni" in pr.columns else "p_holm"
        for _, row in pr.iterrows():
            models = str(row.get("models", "")).split("_vs_")
            if len(models) != 2:
                continue
            m1, m2 = models
            if m1 not in avg_rank.index or m2 not in avg_rank.index:
                continue
            pval = row.get(p_col, np.nan)
            if pd.isna(pval) or pval < 0.05:
                continue
            x1, x2 = avg_rank[m1], avg_rank[m2]
            plt.plot([x1, x2], [level, level], color="grey", linewidth=1.2)
            level += 0.03

    xmin = max(0.5, float(np.floor(x.min() * 2) / 2 - 0.3))
    xmax = float(np.ceil(x.max() * 2) / 2 + 0.3)
    plt.xlim(xmin, xmax)
    plt.ylim(0.85, max(level + 0.05, 1.3))
    plt.yticks([])
    plt.xlabel("Average Rank (Lower is Better)")
    plt.title("Critical-Difference-Like Model Ranking (Cross-Dataset)")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "critical_difference_like_ranking.png"), dpi=180)
    plt.close()


def plot_dominance_maps():
    for dataset_name in DATASET_NAMES:
        for phase5_file, noise_col, noise_label in NOISE_SPECS:
            path = os.path.join(RESULTS_DIR, f"{dataset_name}_{phase5_file}")
            df = _read_csv_or_none(path)
            if df is None or df.empty:
                continue

            required = {"dataset_fraction", noise_col, "model", "mean_auc"}
            if not required.issubset(df.columns):
                continue

            rows = []
            for (frac, nlv), g in df.groupby(["dataset_fraction", noise_col]):
                gg = g.sort_values("mean_auc", ascending=False)
                if gg.empty:
                    continue
                best = gg.iloc[0]
                second_auc = gg.iloc[1]["mean_auc"] if len(gg) > 1 else np.nan
                margin = float(best["mean_auc"] - second_auc) if pd.notna(second_auc) else np.nan
                rows.append(
                    {
                        "dataset_fraction": frac,
                        "noise_level": nlv,
                        "best_model": best["model"],
                        "margin": margin,
                    }
                )

            if not rows:
                continue

            dmap = pd.DataFrame(rows)
            margin_pivot = dmap.pivot_table(index="noise_level", columns="dataset_fraction", values="margin")
            model_pivot = dmap.pivot_table(index="noise_level", columns="dataset_fraction", values="best_model", aggfunc="first")
            if margin_pivot.empty:
                continue

            annotations = model_pivot.copy().astype(str)
            for i in range(annotations.shape[0]):
                for j in range(annotations.shape[1]):
                    m = model_pivot.iloc[i, j]
                    v = margin_pivot.iloc[i, j]
                    if pd.isna(v):
                        annotations.iloc[i, j] = ""
                    else:
                        short = str(m)[:10]
                        annotations.iloc[i, j] = f"{short}\n{v:.3f}"

            plt.figure(figsize=(9, 5.5))
            sns.heatmap(
                margin_pivot,
                annot=annotations,
                fmt="",
                cmap="Blues",
                cbar_kws={"label": "Winner Margin (AUC gap to 2nd)"},
            )
            plt.title(f"Dominance Map | {_dataset_label(dataset_name)} | {noise_label}")
            plt.xlabel("Dataset Fraction")
            plt.ylabel(noise_label)
            plt.tight_layout()
            plt.savefig(
                os.path.join(PLOTS_DIR, f"{dataset_name}_{_safe_noise_tag(noise_label)}_dominance_map.png"),
                dpi=180,
            )
            plt.close()


def plot_parallel_coordinates_multiobjective():
    # Build multi-objective table from phase5 + phase8 summaries.
    all_phase5 = []
    for dataset_name in DATASET_NAMES:
        for phase5_file, noise_col, noise_label in NOISE_SPECS:
            path = os.path.join(RESULTS_DIR, f"{dataset_name}_{phase5_file}")
            df = _read_csv_or_none(path)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["dataset"] = dataset_name
            df["noise_type"] = noise_col
            all_phase5.append(df)

    if not all_phase5:
        return

    p5 = pd.concat(all_phase5, ignore_index=True)
    required = {"dataset", "model", "mean_auc", "cv", "breakdown_flag"}
    if not required.issubset(p5.columns):
        return

    agg = (
        p5.groupby(["dataset", "model"], as_index=False)
        .agg(
            mean_auc=("mean_auc", "mean"),
            mean_cv=("cv", "mean"),
            breakdown_prob=("breakdown_flag", "mean"),
        )
    )

    tol_path = os.path.join(RESULTS_DIR, "phase8_noise_tolerance_boundaries.csv")
    tdf = _read_csv_or_none(tol_path)
    if tdf is not None and not tdf.empty and "tolerance_boundary" in tdf.columns:
        tdf = tdf.copy()
        tdf["tolerance_boundary"] = _coerce_threshold_value(tdf["tolerance_boundary"])
        tdf = (
            tdf.groupby(["dataset", "model"], as_index=False)["tolerance_boundary"]
            .mean()
            .rename(columns={"tolerance_boundary": "mean_tolerance"})
        )
        agg = agg.merge(tdf, on=["dataset", "model"], how="left")
    else:
        agg["mean_tolerance"] = np.nan

    # Convert to beneficial metrics and min-max normalize per dataset.
    agg["stability_score"] = 1.0 / (1.0 + agg["mean_cv"].clip(lower=0))
    agg["resilience_score"] = 1.0 - agg["breakdown_prob"].clip(lower=0, upper=1)

    plot_cols = ["mean_auc", "stability_score", "resilience_score", "mean_tolerance"]

    for dataset_name in DATASET_NAMES:
        ddf = agg[agg["dataset"] == dataset_name].copy()
        if ddf.empty:
            continue

        for col in plot_cols:
            cmin, cmax = ddf[col].min(), ddf[col].max()
            if pd.isna(cmin) or pd.isna(cmax) or cmax == cmin:
                ddf[col + "_norm"] = 0.5
            else:
                ddf[col + "_norm"] = (ddf[col] - cmin) / (cmax - cmin)

        pc = ddf[
            [
                "model",
                "mean_auc_norm",
                "stability_score_norm",
                "resilience_score_norm",
                "mean_tolerance_norm",
            ]
        ].copy()
        pc = pc.rename(
            columns={
                "mean_auc_norm": "AUC",
                "stability_score_norm": "Stability",
                "resilience_score_norm": "Resilience",
                "mean_tolerance_norm": "Tolerance",
            }
        )

        plt.figure(figsize=(10, 5.5))
        parallel_coordinates(pc, class_column="model", colormap="tab10", linewidth=1.8)
        plt.title(f"Parallel Coordinates Multi-Objective View | {_dataset_label(dataset_name)}")
        plt.ylabel("Normalized Score (higher is better)")
        plt.ylim(0, 1)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"{dataset_name}_parallel_coordinates_multiobjective.png"),
            dpi=180,
        )
        plt.close()


def plot_statistical_evidence_dashboard():
    stats = _phase6_stats_df()
    if stats is None or stats.empty:
        return

    # Panel A: count of Holm-significant findings by model (where model column exists).
    model_sig_counts = (
        stats[stats["model"].notna()]
        .groupby("model", as_index=False)["significant_holm"]
        .sum()
        .rename(columns={"significant_holm": "significant_count"})
    )

    # Panel B: average absolute effect size by model where available.
    if "mean_delta" in stats.columns:
        model_abs_effect = (
            stats[stats["model"].notna() & stats["mean_delta"].notna()]
            .assign(abs_effect=lambda d: d["mean_delta"].abs())
            .groupby("model", as_index=False)["abs_effect"]
            .mean()
            .rename(columns={"abs_effect": "mean_abs_effect"})
        )
    else:
        model_abs_effect = pd.DataFrame(columns=["model", "mean_abs_effect"])

    # Panel C: best corrected p-value by model.
    pcol = "p_holm" if "p_holm" in stats.columns else "p_value"
    model_best_p = (
        stats[stats["model"].notna() & stats[pcol].notna()]
        .groupby("model", as_index=False)[pcol]
        .min()
        .rename(columns={pcol: "best_corrected_p"})
    )

    models = sorted(set(model_sig_counts["model"]).union(set(model_abs_effect["model"])).union(set(model_best_p["model"])))
    if not models:
        return

    a = model_sig_counts.set_index("model").reindex(models).fillna(0)["significant_count"]
    b = model_abs_effect.set_index("model").reindex(models).fillna(0)["mean_abs_effect"]
    c = model_best_p.set_index("model").reindex(models)
    c["best_corrected_p"] = c["best_corrected_p"].fillna(1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].bar(models, a.values, color="tab:blue")
    axes[0].set_title("Holm-Significant Findings")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(models, b.values, color="tab:orange")
    axes[1].set_title("Mean Absolute Effect Size")
    axes[1].set_ylabel("|Mean Delta|")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(models, c["best_corrected_p"].values, color="tab:green")
    axes[2].set_title("Best Corrected p-value")
    axes[2].set_ylabel("p")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=45)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Statistical Evidence Dashboard", y=1.02)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "statistical_evidence_dashboard.png"), dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def run_visualization_pipeline():
    print("\n  [Basic] Generating baseline and scaling plots...")
    plot_baseline_model_bars_per_dataset()
    plot_clean_scaling_lines_per_dataset()
    plot_clean_scaling_cross_dataset_per_model()

    print("  [Basic] Generating robustness/noise plots...")
    plot_noise_response_curves()
    plot_cv_profiles()
    plot_relative_drop_heatmaps()
    plot_breakdown_probability_curves()
    plot_cross_dataset_ranking_heatmap()

    print("  [Intermediate] Generating comparative synthesis plots...")
    plot_robustness_small_multiples()
    plot_pairwise_delta_against_best()
    plot_threshold_summary_dotplots()

    print("  [Advanced] Generating inferential and multi-objective plots...")
    plot_significance_annotated_effect_heatmaps()
    plot_forest_effect_plots()
    plot_critical_difference_like_ranking()
    plot_dominance_maps()
    plot_parallel_coordinates_multiobjective()
    plot_statistical_evidence_dashboard()

    print(f"\n  All plots saved to: {PLOTS_DIR}")
