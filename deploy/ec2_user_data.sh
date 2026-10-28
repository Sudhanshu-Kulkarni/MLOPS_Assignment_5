#!/bin/bash
# EC2 User Data script — runs automatically on first boot of the EC2 instance.
# Amazon Linux 2023 (arm64 or x86_64)
set -euxo pipefail

# ── 1. System packages ────────────────────────────────────────────────────
dnf update -y
dnf install -y docker git python3-pip curl
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# ── 2. DVC with S3 support ────────────────────────────────────────────────
pip3 install --upgrade pip
pip3 install "dvc[s3]"

# ── 3. Clone the repo ─────────────────────────────────────────────────────
REPO="https://github.com/YOUR_USERNAME/MLOPS_Assignment_5.git"
APP_DIR=/opt/cifar10-classifier

rm -rf "$APP_DIR"
git clone "$REPO" "$APP_DIR"
cd "$APP_DIR"

# ── 4. Pull DVC artifacts from S3 (if remote configured) ─────────────────
if grep -q 's3://' .dvc/config 2>/dev/null; then
    dvc pull -f || echo "DVC pull failed — running pipeline locally"
fi

# ── 5. If no model yet, run the pipeline with synthetic data ──────────────
if [ ! -f models/training_history.json ]; then
    pip3 install -r requirements.txt -r requirements-web.txt
    python3 -c "
import yaml, pathlib
p = pathlib.Path('params.yaml')
cfg = yaml.safe_load(p.read_text())
cfg['data_collection']['dataset_name'] = 'synthetic'
cfg['data_collection']['max_samples'] = 500
cfg['model_training']['epochs'] = 3
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
"
    python3 -m src.data_collection
    python3 -m src.data_processing
    python3 -m src.feature_selection
    python3 -m src.model_training
    python3 -m src.model_evaluation
fi

# ── 6. Build and run Docker container ────────────────────────────────────
docker build -t cifar10-classifier:latest .
docker rm -f cifar10-app || true
docker run -d \
    --name cifar10-app \
    -p 80:8000 \
    --restart unless-stopped \
    cifar10-classifier:latest

echo "Deployment complete. App running on port 80."
curl -f http://localhost/health && echo "Health check passed."
