from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.params import get_section
except ImportError:
    from params import get_section


def load_source_data(source_path: Union[Path, str]) -> pd.DataFrame:
    source_path = Path(source_path)
    df = pd.read_csv(source_path, na_values=["?", "xxxxxxx", "xxxxxxxx"])
    return df


def ingest_data(
    source_file: Union[Path, str] = None,
    raw_dir: Union[Path, str] = None,
    test_size: float = None,
    random_state: int = None,
):
    project_dir = Path(__file__).resolve().parents[2]
    params = get_section("data_ingestion")
    source_file = Path(source_file) if source_file else project_dir / params.get("source_file", "src/dataset.csv")
    raw_dir = Path(raw_dir) if raw_dir else project_dir / params.get("raw_dir", "data/raw")
    test_size = test_size if test_size is not None else params.get("test_size", 0.2)
    random_state = random_state if random_state is not None else params.get("random_state", 42)
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = load_source_data(source_file)
    target_col = "churn_risk_score"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in source data.")

    stratify_values = df[target_col] if df[target_col].nunique() > 1 else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path


if __name__ == "__main__":
    train_file, test_file = ingest_data()
    print(f"Wrote train data to {train_file}")
    print(f"Wrote test data to {test_file}")
