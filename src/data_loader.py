import os
import glob
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def find_csv_file(data_dir, allow_multiple=False):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No CSV file found in '{data_dir}'. Please place a dataset there."
        )
    if len(csv_files) > 1:
        if allow_multiple:
            chosen = csv_files[0]
            print(
                "  Multiple CSV files found; using first alphabetically: "
                f"{os.path.basename(chosen)}"
            )
            return chosen
        names = [os.path.basename(f) for f in csv_files]
        raise ValueError(
            f"Multiple CSV files found in '{data_dir}': {names}. Please keep only one."
        )
    return csv_files[0]


def load_csv_dataset(path):
    print(f"  Loading: {os.path.basename(path)}")
    df = pd.read_csv(path)
    target_col = df.columns[-1]
    print(f"  Target column: '{target_col}' ({df[target_col].nunique()} unique values)")
    X = df.drop(target_col, axis=1).values.astype(np.float64)
    y = df[target_col].values
    return X, y


def load_adult_income():
    print("  Fetching Adult Income dataset from OpenML...")
    dataset = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = dataset.frame.dropna()
    y = (df["class"].str.strip() == ">50K").astype(int).values
    X = pd.get_dummies(df.drop(columns=["class"])).values.astype(np.float64)
    pos_rate = y.mean()
    print(f"  Adult Income: {len(y)} samples, {pos_rate:.1%} positive")
    return X, y


def load_give_me_some_credit():
    gmc_path = os.path.join(DATA_DIR, "give_me_some_credit.csv")
    if os.path.exists(gmc_path):
        print(f"  Loading Give Me Some Credit from {gmc_path}")
        df = pd.read_csv(gmc_path).dropna()
        target_col = "SeriousDlqin2yrs"
        X = df.drop(columns=[target_col, "Unnamed: 0"], errors="ignore").values.astype(np.float64)
        y = df[target_col].values
        pos_rate = y.mean()
        print(f"  Give Me Some Credit: {len(y)} samples, {pos_rate:.1%} positive")
        return X, y
    else:
        raise FileNotFoundError(
            "Give Me Some Credit dataset not found. "
            "Download cs-training.csv from https://www.kaggle.com/c/GiveMeSomeCredit/data, "
            f"rename it to 'give_me_some_credit.csv' and place it in {DATA_DIR}/"
        )


def load_all_datasets():
    datasets = {}

    # Dataset 1: Credit Card Fraud (extreme imbalance ~0.17%)
    print("\n[Dataset 1/3] Credit Card Fraud (extreme imbalance)")
    fraud_path = os.path.join(DATA_DIR, "creditcard.csv")
    if not os.path.exists(fraud_path):
        raise FileNotFoundError(
            f"creditcard.csv not found in {DATA_DIR}/. "
            "Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    X, y = load_csv_dataset(fraud_path)
    datasets["creditcard"] = (X, y)
    print(f"  Positive rate: {y.mean():.3%}")

    # Dataset 2: Give Me Some Credit (mid imbalance ~6.7%)
    print("\n[Dataset 2/3] Give Me Some Credit (mid imbalance)")
    X, y = load_give_me_some_credit()
    datasets["give_me_some_credit"] = (X, y)

    # Dataset 3: Adult Income (balanced ~24%)
    print("\n[Dataset 3/3] Adult Income (balanced)")
    X, y = load_adult_income()
    datasets["adult_income"] = (X, y)

    return datasets


def load_data(path=None):
    """Single-dataset loader kept for backward compatibility."""
    if path is None:
        path = find_csv_file(DATA_DIR, allow_multiple=True)
    print(f"Loading dataset: {os.path.basename(path)}")
    X, y = load_csv_dataset(path)
    return X, y