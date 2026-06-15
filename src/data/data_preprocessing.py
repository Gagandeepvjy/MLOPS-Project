from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

try:
    from src.params import get_section
except ImportError:
    from params import get_section


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace({"?": pd.NA, "xxxxxxx": pd.NA, "xxxxxxxx": pd.NA}, inplace=True)
    df.drop(columns=["security_no", "referral_id"], errors="ignore", inplace=True)

    if "joining_date" in df.columns:
        df["joining_date"] = pd.to_datetime(df["joining_date"], errors="coerce")
        df["joining_year"] = df["joining_date"].dt.year
        df["joining_month"] = df["joining_date"].dt.month
        df["joining_day"] = df["joining_date"].dt.day
        df.drop(columns=["joining_date"], inplace=True)

    if "last_visit_time" in df.columns:
        time_series = pd.to_datetime(df["last_visit_time"], format="%H:%M:%S", errors="coerce")
        df["last_visit_seconds"] = (
            time_series.dt.hour.fillna(0).astype(int) * 3600
            + time_series.dt.minute.fillna(0).astype(int) * 60
            + time_series.dt.second.fillna(0).astype(int)
        )
        df.drop(columns=["last_visit_time"], inplace=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    for col in categorical_cols:
        df[col] = df[col].astype("string").fillna("Unknown")

    return df


def preprocess_data(
    raw_dir: Union[Path, str] = None,
    interim_dir: Union[Path, str] = None,
):
    project_dir = Path(__file__).resolve().parents[2]
    params = get_section("preprocessing")
    raw_dir = Path(raw_dir) if raw_dir else project_dir / params.get("raw_dir", "data/raw")
    interim_dir = Path(interim_dir) if interim_dir else project_dir / params.get("interim_dir", "data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)
    missing_values = params.get("missing_values", ["?", "xxxxxxx", "xxxxxxxx"])

    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected raw train/test files at {train_path} and {test_path}. Run data ingestion first."
        )

    train_df = pd.read_csv(train_path, na_values=missing_values)
    test_df = pd.read_csv(test_path, na_values=missing_values)

    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected raw train/test files at {train_path} and {test_path}. Run data ingestion first."
        )

    train_df = pd.read_csv(train_path, na_values=["?", "xxxxxxx", "xxxxxxxx"])
    test_df = pd.read_csv(test_path, na_values=["?", "xxxxxxx", "xxxxxxxx"])

    train_preprocessed = preprocess_dataframe(train_df)
    test_preprocessed = preprocess_dataframe(test_df)

    train_output = interim_dir / "train_preprocessed.csv"
    test_output = interim_dir / "test_preprocessed.csv"
    train_preprocessed.to_csv(train_output, index=False)
    test_preprocessed.to_csv(test_output, index=False)

    return train_output, test_output


if __name__ == "__main__":
    train_file, test_file = preprocess_data()
    print(f"Wrote preprocessed train data to {train_file}")
    print(f"Wrote preprocessed test data to {test_file}")
