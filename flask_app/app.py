import os
import time
import warnings
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request
from prometheus_client import CollectorRegistry, Counter, CONTENT_TYPE_LATEST, Histogram, generate_latest

ROOT_DIR = Path(__file__).resolve().parents[1]

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
load_dotenv(ROOT_DIR / ".env")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-this-secret-key")

registry = CollectorRegistry()
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

RAW_FEATURES = [
    "age",
    "gender",
    "region_category",
    "membership_category",
    "joining_date",
    "joined_through_referral",
    "preferred_offer_types",
    "medium_of_operation",
    "internet_option",
    "last_visit_time",
    "days_since_last_login",
    "avg_time_spent",
    "avg_transaction_value",
    "avg_frequency_login_days",
    "points_in_wallet",
    "used_special_discount",
    "offer_application_preference",
    "past_complaint",
    "complaint_status",
    "feedback",
]


def get_mlflow_client():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    dagshub_repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    dagshub_repo_name = os.getenv("DAGSHUB_REPO_NAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    if dagshub_repo_owner and dagshub_repo_name:
        import dagshub

        dagshub.init(repo_owner=dagshub_repo_owner, repo_name=dagshub_repo_name, mlflow=True)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    return mlflow.MlflowClient()


def get_latest_model_version(model_name):
    client = get_mlflow_client()
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_versions:
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
    return latest_versions[0].version if latest_versions else None


def load_model_and_vectorizer():
    model_name = os.getenv("MLFLOW_MODEL_NAME", "churn-risk-model")
    model = None

    # If a full inference pipeline was saved during training, prefer that.
    pipeline_path = ROOT_DIR / "models" / "pipeline.pkl"
    if pipeline_path.exists():
        app.logger.info("Loading full inference pipeline from %s", pipeline_path)
        pipeline = joblib.load(pipeline_path)
        return pipeline, None

    try:
        model_version = get_latest_model_version(model_name)
        if model_version:
            model_uri = f"models:/{model_name}/{model_version}"
            app.logger.info("Loading model from registry: %s", model_uri)
            model = mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        app.logger.warning("Unable to load model from MLflow registry: %s", exc)

    if model is None:
        local_model_path = ROOT_DIR / "models" / "model.pkl"
        app.logger.info("Loading local model from %s", local_model_path)
        model = joblib.load(local_model_path)

    vectorizer_path = ROOT_DIR / "models" / "vectorizer.pkl"
    app.logger.info("Loading vectorizer from %s", vectorizer_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


def build_input_dataframe(form_data):
    raw_values = {field: form_data.get(field, "") for field in RAW_FEATURES}
    df = pd.DataFrame([raw_values])
    # Some saved transformers expect an index column named 'Unnamed: 0' (CSV index).
    # Add a default value so transforms don't fail when that column is expected.
    if "Unnamed: 0" not in df.columns:
        df["Unnamed: 0"] = 0
    return df


def _load_select_options():
    select_options = {}
    try:
        sample_path = ROOT_DIR / "src" / "dataset.csv"
        if sample_path.exists():
            sample_df = pd.read_csv(sample_path)
            for col in RAW_FEATURES:
                if col in sample_df.columns:
                    uniques = sample_df[col].dropna().unique().tolist()
                    if 1 < len(uniques) <= 200:
                        select_options[col] = sorted(map(str, uniques))
    except Exception:
        return {}
    return select_options


def build_prediction_features(raw_df):
    df = raw_df.copy()
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

    if "churn_risk_score" in df.columns:
        df = df.drop(columns=["churn_risk_score"])

    return df


model, vectorizer = load_model_and_vectorizer()


@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template(
        "index.html",
        result=None,
        probability=None,
        form_data={},
        select_options=_load_select_options(),
    )
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    raw_df = build_input_dataframe(request.form)
    feature_df = build_prediction_features(raw_df)
    # If `vectorizer` is None, we assume `model` is a full pipeline that
    # accepts raw inputs (or will handle preprocessing internally).
    probability = None
    try:
        if vectorizer is None:
            # Try passing raw inputs first (pipeline may expect raw DataFrame).
            preds = model.predict(raw_df)
            # try predict_proba on raw inputs as well
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(raw_df)
                probability = float(proba[0][1]) if proba is not None and len(proba) > 0 else None
        else:
            features = vectorizer.transform(feature_df)
            preds = model.predict(features)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)
                probability = float(proba[0][1]) if proba is not None and len(proba) > 0 else None

        # unify prediction extraction
        prediction = preds[0] if hasattr(preds, "__len__") else preds
    except Exception:
        # fallback: if pipeline expects preprocessed features
        if vectorizer is None:
            features = feature_df
            if hasattr(model, "transform"):
                try:
                    features = model.transform(feature_df)
                except Exception:
                    pass
        else:
            features = vectorizer.transform(feature_df)

        preds = model.predict(features)
        prediction = preds[0] if hasattr(preds, "__len__") else preds
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)
            probability = float(proba[0][1]) if proba is not None and len(proba) > 0 else None

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    result_text = "Churn Risk: {}".format(prediction)
    return render_template(
        "index.html",
        result=result_text,
        probability=probability,
        form_data=request.form,
        select_options=_load_select_options(),
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
