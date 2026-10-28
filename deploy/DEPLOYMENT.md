# AWS Deployment Guide

## Resources

| Resource | Value |
|---|---|
| Region | us-east-1 |
| S3 bucket | `<your-name>-cifar10-dvc-<account-id>` |
| DVC remote | `s3://<bucket>/dvc-storage` |
| GitHub | https://github.com/YOUR_USERNAME/MLOPS_Assignment_5 |
| EC2 Public URL | `http://<EC2_PUBLIC_IP>/` |

---

## Step 1 — Push DVC artifacts to S3

```bash
cd my_assignment_5
source ../my_assignment_1/venv/bin/activate

# Create S3 bucket
aws s3 mb s3://sudhanshu-cifar10-dvc-<account-id> --region us-east-1

# Add DVC remote
dvc remote add -d origin s3://sudhanshu-cifar10-dvc-<account-id>/dvc-storage

# Push model + data artifacts
dvc push
```

---

## Step 2 — Launch EC2 instance (AWS Console)

1. Go to EC2 → Launch Instance
2. **Name:** `cifar10-classifier`
3. **AMI:** Amazon Linux 2023 (free tier eligible)
4. **Instance type:** t2.micro (free tier) or t3.small
5. **Key pair:** create or use existing — download `.pem` file
6. **Security group:** allow HTTP (port 80) and SSH (port 22) from anywhere
7. **User data:** paste contents of `deploy/ec2_user_data.sh`
   - Edit `REPO=` to point to your GitHub repo
8. Launch instance — wait 10-15 minutes

---

## Step 3 — Verify deployment

```bash
# Replace with your EC2 public IP
curl http://<EC2_PUBLIC_IP>/health
```

Open in browser: `http://<EC2_PUBLIC_IP>/`

---

## Step 4 — Set up GitHub Secrets for CI/CD auto-deploy

In your GitHub repo → Settings → Secrets → Actions:

| Secret | Value |
|---|---|
| `EC2_HOST` | EC2 public IP |
| `EC2_SSH_KEY` | Contents of your `.pem` file |

After this, every push to `main` automatically deploys to EC2.

---

## Local Docker test

```bash
dvc repro                          # make sure pipeline has run
docker build -t cifar10-classifier:latest .
docker run -p 8000:8000 cifar10-classifier:latest
```

Open http://localhost:8000

---

## DVC pipeline stages

```
data_collection → data_processing → feature_selection → model_training → model_evaluation
```

All stages defined in `dvc.yaml`. Run with:
```bash
dvc repro
```
