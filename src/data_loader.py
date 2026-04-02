import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler


def find_data_file(data_dir=None):
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV file found in '{data_dir}'. Please place a dataset there.")
    if len(csv_files) > 1:
        names = [os.path.basename(f) for f in csv_files]
        raise ValueError(f"Multiple CSV files found in '{data_dir}': {names}. Please keep only one.")

    return csv_files[0]


def load_data(path=None):
    if path is None:
        path = find_data_file()

    print(f"Loading dataset: {os.path.basename(path)}")
    df = pd.read_csv(path)

    target_col = df.columns[-1]
    print(f"Using '{target_col}' as target column ({df[target_col].nunique()} unique values)")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values