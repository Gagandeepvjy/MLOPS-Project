import json
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from params import get_section


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_feature_transformer(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", make_one_hot_encoder(), categorical_cols))

    transformer = ColumnTransformer(transformers=transformers, remainder="drop")
    return transformer


def load_interim_data(
    interim_dir: Union[Path, str] = None,
):
    project_dir = Path(__file__).resolve().parents[2]
    interim_dir = Path(interim_dir) if interim_dir else project_dir / "data" / "interim"
    train_file = interim_dir / "train_preprocessed.csv"
    test_file = interim_dir / "test_preprocessed.csv"
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Expected interim files at {train_file} and {test_file}. Run preprocessing first."
        )
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df


def transform_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = None,
    processed_dir: Union[Path, str] = None,
    models_dir: Union[Path, str] = None,
):
    project_dir = Path(__file__).resolve().parents[2]
    params = get_section("feature_engineering")
    target_col = target_col if target_col is not None else params.get("target_col", "churn_risk_score")
    processed_dir = Path(processed_dir) if processed_dir else project_dir / params.get("processed_dir", "data/processed")
    models_dir = Path(models_dir) if models_dir else project_dir / params.get("models_dir", "models")
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df.drop(columns=[target_col], errors="ignore")
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col], errors="ignore")
    y_test = test_df[target_col]

    transformer = build_feature_transformer(X_train)
    transformer.fit(X_train)

    train_transformed = transformer.transform(X_train)
    test_transformed = transformer.transform(X_test)

    feature_names = transformer.get_feature_names_out()
    train_features = pd.DataFrame(train_transformed, columns=feature_names)
    test_features = pd.DataFrame(test_transformed, columns=feature_names)
    train_features[target_col] = y_train.values
    test_features[target_col] = y_test.values

    train_output = processed_dir / "train_features.csv"
    test_output = processed_dir / "test_features.csv"
    train_features.to_csv(train_output, index=False)
    test_features.to_csv(test_output, index=False)

    vectorizer_path = models_dir / "vectorizer.pkl"
    joblib.dump(transformer, vectorizer_path)

    return train_output, test_output, vectorizer_path


if __name__ == "__main__":
    train_file, test_file = load_interim_data()
    train_output, test_output, vectorizer_path = transform_features(train_file, test_file)
    print(f"Wrote engineered train features to {train_output}")
    print(f"Wrote engineered test features to {test_output}")
    print(f"Saved feature transformer to {vectorizer_path}")
