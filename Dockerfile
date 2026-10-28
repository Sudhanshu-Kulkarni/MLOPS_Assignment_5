# Lightweight production image — no TensorFlow.
# Model: scikit-learn Pipeline (StandardScaler + MLPClassifier) ~20MB
# Total image target: ~400MB (python:3.11-slim + sklearn + flask)

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Minimal system deps: curl for HEALTHCHECK only
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install production Python deps (no TensorFlow)
COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/     ./src/
COPY app/     ./app/
COPY params.yaml ./

# Copy only the inference artifacts (model + history) — NOT the 100MB+ npz data files
COPY models/classifier/model.joblib        ./models/classifier/model.joblib
COPY models/classifier/format.json         ./models/classifier/format.json
COPY models/training_history.json          ./models/training_history.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app.server:app", \
     "--workers", "1", "--timeout", "120", "--access-logfile", "-"]
