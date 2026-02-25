import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_clean_scaling():
    df = pd.read_csv(f"{RESULTS_DIR}/phase2_clean_scaling_results.csv")
    models = df["model"].unique()

    for model in models:
        model_df = df[df["model"] == model]

        plt.figure()
        plt.plot(model_df["dataset_fraction"], model_df["mean_auc"], marker="o")
        plt.title(f"Clean Scaling - {model}")
        plt.xlabel("Dataset Fraction")
        plt.ylabel("Mean AUC")
        plt.grid(True)
        plt.savefig(f"{PLOTS_DIR}/clean_scaling_auc_{model}.png")
        plt.close()


def plot_noise_scaling(file_name, noise_column, noise_label):
    df = pd.read_csv(f"{RESULTS_DIR}/{file_name}")
    models = df["model"].unique()
    noise_levels = sorted(df[noise_column].unique())

    for model in models:
        plt.figure()
        model_df = df[df["model"] == model]

        for noise in noise_levels:
            subset = model_df[model_df[noise_column] == noise]
            subset = subset.sort_values("dataset_fraction")

            plt.plot(
                subset["dataset_fraction"],
                subset["mean_auc"],
                marker="o",
                label=f"{noise_label}={noise}"
            )

        plt.title(f"{noise_label} Scaling Effect - {model}")
        plt.xlabel("Dataset Fraction")
        plt.ylabel("Mean AUC")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{PLOTS_DIR}/{noise_label}_auc_{model}.png")
        plt.close()


def plot_cv_scaling(file_name, noise_column, noise_label):
    df = pd.read_csv(f"{RESULTS_DIR}/{file_name}")
    models = df["model"].unique()
    noise_levels = sorted(df[noise_column].unique())

    for model in models:
        plt.figure()
        model_df = df[df["model"] == model]

        for noise in noise_levels:
            subset = model_df[model_df[noise_column] == noise]
            subset = subset.sort_values("dataset_fraction")

            plt.plot(
                subset["dataset_fraction"],
                subset["cv"],
                marker="o",
                label=f"{noise_label}={noise}"
            )

        plt.title(f"{noise_label} Scaling vs CV - {model}")
        plt.xlabel("Dataset Fraction")
        plt.ylabel("Coefficient of Variation (CV)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{PLOTS_DIR}/{noise_label}_cv_{model}.png")
        plt.close()


def plot_heatmap(file_name, noise_column, noise_label):
    df = pd.read_csv(f"{RESULTS_DIR}/{file_name}")
    models = df["model"].unique()

    for model in models:
        model_df = df[df["model"] == model]

        pivot = model_df.pivot_table(
            index=noise_column,
            columns="dataset_fraction",
            values="relative_auc_drop"
        )

        plt.figure()
        sns.heatmap(pivot, annot=False)
        plt.title(f"{noise_label} Relative AUC Drop - {model}")
        plt.xlabel("Dataset Fraction")
        plt.ylabel(noise_label)
        plt.savefig(f"{PLOTS_DIR}/{noise_label}_heatmap_{model}.png")
        plt.close()


def plot_breakdown_rate(file_name, noise_column, noise_label):
    df = pd.read_csv(f"{RESULTS_DIR}/{file_name}")
    models = df["model"].unique()

    for model in models:
        model_df = df[df["model"] == model]

        breakdown_rate = (
            model_df.groupby("dataset_fraction")["breakdown_flag"]
            .mean()
            .reset_index()
        )

        plt.figure()
        plt.plot(
            breakdown_rate["dataset_fraction"],
            breakdown_rate["breakdown_flag"],
            marker="o"
        )

        plt.title(f"{noise_label} Breakdown Rate - {model}")
        plt.xlabel("Dataset Fraction")
        plt.ylabel("Breakdown Probability")
        plt.grid(True)
        plt.savefig(f"{PLOTS_DIR}/{noise_label}_breakdown_{model}.png")
        plt.close()


def run_visualization_pipeline():
    plot_clean_scaling()

    plot_noise_scaling(
        "phase5_feature_noise_analysis.csv",
        "feature_noise_std",
        "feature_noise"
    )
    plot_cv_scaling(
        "phase5_feature_noise_analysis.csv",
        "feature_noise_std",
        "feature_noise"
    )
    plot_heatmap(
        "phase5_feature_noise_analysis.csv",
        "feature_noise_std",
        "feature_noise"
    )
    plot_breakdown_rate(
        "phase5_feature_noise_analysis.csv",
        "feature_noise_std",
        "feature_noise"
    )

    plot_noise_scaling(
        "phase5_label_noise_analysis.csv",
        "label_noise",
        "label_noise"
    )
    plot_cv_scaling(
        "phase5_label_noise_analysis.csv",
        "label_noise",
        "label_noise"
    )
    plot_heatmap(
        "phase5_label_noise_analysis.csv",
        "label_noise",
        "label_noise"
    )
    plot_breakdown_rate(
        "phase5_label_noise_analysis.csv",
        "label_noise",
        "label_noise"
    )

    plot_noise_scaling(
        "phase5_weak_sample_analysis.csv",
        "weak_fraction",
        "weak_sample"
    )
    plot_cv_scaling(
        "phase5_weak_sample_analysis.csv",
        "weak_fraction",
        "weak_sample"
    )
    plot_heatmap(
        "phase5_weak_sample_analysis.csv",
        "weak_fraction",
        "weak_sample"
    )
    plot_breakdown_rate(
        "phase5_weak_sample_analysis.csv",
        "weak_fraction",
        "weak_sample"
    )