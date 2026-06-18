MLOPS Project
==============

A Dockerized Flask + MLflow churn prediction application built as an MLOps demo.

## What this project contains

- `flask_app/` — Flask application and Jinja2 UI templates for serving predictions
- `models/` — serialized model artifacts used at inference time
- `src/` — core project logic for data ingestion, preprocessing, feature engineering, model training, evaluation, and MLflow model registration
- `data/` — raw, interim, and processed data assets
- `mlruns/` — local MLflow experiment metadata and model artifacts
- `Dockerfile` — builds a container image for the Flask app
- `requirements.txt` — Python dependencies for the full project
- `flask_app/requirements.txt` — runtime requirements for the Flask service

## Quick start

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run locally

```bash
python flask_app/app.py
```

Open `http://localhost:5002` in your browser.

### Run with Docker

```bash
docker build -t mlops:latest .
docker run -p 8888:5002 mlops:latest
```

Open `http://localhost:8888` in your browser.

## How the Flask app works

- The Flask app is defined in `flask_app/app.py`.
- The model is loaded from MLflow registry if configured; otherwise it falls back to `models/model.pkl`.
- The feature preprocessing vectorizer is loaded from `models/vectorizer.pkl`.
- Dropdown option values on the UI are built from `src/dataset.csv`.

## Docker behavior

The Docker image copies:

- the root `requirements.txt` and `setup.py` for dependency installation
- `flask_app/` source files
- `models/` serialized artifacts
- `src/dataset.csv` so UI dropdowns are populated at runtime

The container starts the Flask app with `gunicorn` on port `5002`.

## Useful commands

```bash
git status
python flask_app/app.py
docker build --no-cache -t mlops:latest .
docker run -p 8888:5002 mlops:latest
```

## Notes

- `mlflow.txt` is ignored by git once it is removed from tracking with `git rm --cached mlflow.txt`.
- If the UI still appears stale after a change, rebuild the Docker image with `--no-cache`.
- The app logs metrics and can expose Prometheus metrics at `/metrics`.
