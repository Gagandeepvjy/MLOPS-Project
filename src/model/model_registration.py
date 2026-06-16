import json
import os
from pathlib import Path
from typing import Any, Dict, Union

import dagshub
import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

try:
    from src.params import load_params
except ImportError:
    from params import load_params

project_dir = Path(__file__).resolve().parents[2]
env_path = project_dir / ".env"
load_dotenv(env_path)


def load_experiment_info(
    reports_dir: Union[Path, str] = None,
    experiment_file: str = "experiment.json",
):
    project_dir = Path(__file__).resolve().parents[2]
    reports_dir = Path(reports_dir) if reports_dir else project_dir / "reports"
    path = reports_dir / experiment_file
    if not path.exists():
        raise FileNotFoundError(f"Expected experiment info at {path}. Run model evaluation first.")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def register_model(
    reports_dir: Union[Path, str] = None,
    mlflow_tracking_uri: str = None,
    dagshub_repo_owner: str = None,
    dagshub_repo_name: str = None,
):
    proj_dir = Path(__file__).resolve().parents[2]
    reports_dir = Path(reports_dir) if reports_dir else proj_dir / "reports"
    mlflow_tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    dagshub_repo_owner = dagshub_repo_owner or os.getenv("DAGSHUB_REPO_OWNER")
    dagshub_repo_name = dagshub_repo_name or os.getenv("DAGSHUB_REPO_NAME")

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    if dagshub_repo_owner and dagshub_repo_name:
        dagshub.init(
        repo_owner=dagshub_repo_owner,
        repo_name=dagshub_repo_name,
        mlflow=True,
        dvc=False  # add this to avoid DVC remote conflict
        )

    experiment_info = load_experiment_info(reports_dir)
    all_params = load_params()
    mlflow_config = all_params.get("mlflow", {}) if isinstance(all_params, dict) else {}
    model_name = mlflow_config.get("model_name", "churn-risk-model")
    register_flag = mlflow_config.get("register_model", True)

    model_file = proj_dir / "models" / "model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Expected model file at {model_file}. Run model building first.")
    model = joblib.load(model_file)

    with mlflow.start_run() as run:
        flat_params = {}
        for section, section_params in all_params.items():
            if isinstance(section_params, dict):
                for key, value in section_params.items():
                    if value is not None:
                        flat_params[f"{section}.{key}"] = value
            else:
                flat_params[section] = section_params

        mlflow.log_params(flat_params)
        for metric_name, metric_value in experiment_info.get("metrics", {}).items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, artifact_path="model")
        if register_flag and model_name:
            mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)

        mlflow.log_artifact(str(reports_dir / "experiment.json"))
        mlflow.log_artifact(str(reports_dir / "model_info.json"))

    return experiment_info


if __name__ == "__main__":
    info = register_model()
    print("Registered experiment info to MLflow/DagsHub")
    print(info)
