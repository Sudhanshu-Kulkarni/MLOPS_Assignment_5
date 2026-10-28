# CIFAR-10 Classifier — DVC + Docker + GitHub Actions CI/CD + AWS EC2

End-to-end MLOps deployment: DVC pipeline → Docker container → GitHub Actions CI/CD → AWS EC2.

## Architecture

```
GitHub push
    ↓
GitHub Actions CI/CD (.github/workflows/ci-cd.yaml)
    ├── Job 1: test   — runs 22 automated tests
    ├── Job 2: docker — builds Docker image, verifies container starts
    └── Job 3: deploy — SSH into EC2, pull latest, rebuild + restart container

EC2 Instance (Amazon Linux 2023)
    └── Docker container → Flask app on port 80
            ├── GET  /         — HTML upload form
            ├── GET  /health   — JSON health check
            └── POST /predict  — image → CIFAR-10 class prediction
```

## DVC Pipeline

```
data_collection → data_processing → feature_selection → model_training → model_evaluation
```

Run with:
```bash
dvc repro
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-web.txt
git init && dvc init
dvc repro
```

## Local Docker

```bash
docker build -t cifar10-classifier:latest .
docker run -p 8000:8000 cifar10-classifier:latest
# Open http://localhost:8000
```

## AWS Deployment

See `deploy/DEPLOYMENT.md` for full step-by-step instructions.

## Run tests

```bash
python test.py
```

22 tests across 5 test classes: UtilsTests, DataCollectionTests, ModelTests, FlaskAppTests, PipelineIntegrationTests.
