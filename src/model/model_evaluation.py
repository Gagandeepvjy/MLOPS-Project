import json
from datetime import datetime
from pathlib import Path
from typing import Union

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def load_test_data(
    processed_dir: Union[Path, str] = None,
):
    project_dir = Path(__file__).resolve().parents[2]
    processed_dir = Path(processed_dir) if processed_dir else project_dir / "data" / "processed"
    test_file = processed_dir / "test_features.csv"
    if not test_file.exists():
        raise FileNotFoundError(f"Expected processed test data at {test_file}. Run feature engineering first.")
    return pd.read_csv(test_file)


def load_model(
    models_dir: Union[Path, str] = None,
):
    project_dir = Path(__file__).resolve().parents[2]
    models_dir = Path(models_dir) if models_dir else project_dir / "models"
    model_file = models_dir / "model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Expected model file at {model_file}. Run model building first.")
    return joblib.load(model_file)


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    target_col: str = "churn_risk_score",
    metrics_dir: Union[Path, str] = None,
    reports_dir: Union[Path, str] = None,
):
    project_dir = Path(__file__).resolve().parents[2]
    metrics_dir = Path(metrics_dir) if metrics_dir else project_dir / "metrics"
    reports_dir = Path(reports_dir) if reports_dir else project_dir / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in processed test data.")

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred

    metrics = {
        "accuracy_score": float(accuracy_score(y_test, y_pred)),
        "precision_score": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_score": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc_score": float(roc_auc_score(y_test, y_proba)),
    }

    metrics_path = metrics_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    experiment_info = {
        "model_path": str(Path(__file__).resolve().parents[2] / "models" / "model.pkl"),
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "rows_evaluated": int(test_df.shape[0]),
        "feature_count": int(X_test.shape[1]),
    }

    experiment_path = reports_dir / "experiment.json"
    with experiment_path.open("w", encoding="utf-8") as fh:
        json.dump(experiment_info, fh, indent=2)

    model_info_path = reports_dir / "model_info.json"
    model_info = {
        "model_file": str(Path(__file__).resolve().parents[2] / "models" / "model.pkl"),
        "output_metrics": str(metrics_path),
        "generated_at": experiment_info["timestamp"],
    }
    with model_info_path.open("w", encoding="utf-8") as fh:
        json.dump(model_info, fh, indent=2)

    return metrics_path, experiment_path, model_info_path


if __name__ == "__main__":
    model = load_model()
    test_df = load_test_data()
    metrics_file, experiment_file, model_info_file = evaluate_model(model, test_df)
    print(f"Wrote metrics to {metrics_file}")
    print(f"Wrote experiment data to {experiment_file}")
    print(f"Wrote model info to {model_info_file}")
