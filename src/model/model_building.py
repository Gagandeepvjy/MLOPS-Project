from pathlib import Path
from typing import Any, Dict, Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from params import get_section


def train_model(
    processed_dir: Union[Path, str] = None,
    models_dir: Union[Path, str] = None,
    target_col: str = None,
    params: Dict[str, Any] = None,
):
    project_dir = Path(__file__).resolve().parents[2]
    config = params or get_section("model")
    target_col = target_col if target_col is not None else config.get("target_col", "churn_risk_score")
    processed_dir = Path(processed_dir) if processed_dir else project_dir / "data" / "processed"
    models_dir = Path(models_dir) if models_dir else project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    n_estimators = config.get("n_estimators", 100)
    random_state = config.get("random_state", 42)
    max_depth = config.get("max_depth")
    min_samples_split = config.get("min_samples_split", 2)
    min_samples_leaf = config.get("min_samples_leaf", 1)

    train_file = processed_dir / "train_features.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"Expected processed training file at {train_file}. Run feature engineering first.")

    train_df = pd.read_csv(train_file)
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in processed training data.")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)

    model_path = models_dir / "model.pkl"
    joblib.dump(model, model_path)
    return model_path


if __name__ == "__main__":
    model_path = train_model()
    print(f"Saved trained model to {model_path}")
