import os
import sys
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
sys.path.insert(0, str(ROOT_DIR))

try:
    from src.data.data_preprocessing import preprocess_dataframe
except ImportError:
    from data.data_preprocessing import preprocess_dataframe

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
    return pd.DataFrame([raw_values])


def build_prediction_features(raw_df):
    preprocessed = preprocess_dataframe(raw_df)
    if "churn_risk_score" in preprocessed.columns:
        preprocessed = preprocessed.drop(columns=["churn_risk_score"])
    return preprocessed


model, vectorizer = load_model_and_vectorizer()


@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None, probability=None, form_data={})
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    raw_df = build_input_dataframe(request.form)
    feature_df = build_prediction_features(raw_df)
    features = vectorizer.transform(feature_df)

    prediction = model.predict(features)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][1])

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    result_text = "Churn Risk: {}".format(prediction)
    return render_template(
        "index.html",
        result=result_text,
        probability=probability,
        form_data=request.form,
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
