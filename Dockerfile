FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies first so Docker can cache this layer.
COPY requirements.txt ./
COPY setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the runtime files required for the Flask app.
COPY flask_app/ ./flask_app/
COPY models/ ./models/
COPY src/dataset.csv ./src/dataset.csv

EXPOSE 5002
WORKDIR /app/flask_app

CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app", "--timeout", "120"]
# CMD ["python", "app.py"]
