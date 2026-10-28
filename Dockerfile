FROM python:3.11-slim

WORKDIR /app

# Prevent .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps: git (for DVC), libgomp (TF), curl (health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt

# Copy project source
COPY . .

# Port the Flask app listens on
EXPOSE 8000

# Healthcheck so Docker/EC2 knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn — 2 workers, 120s timeout (model load can be slow)
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app.server:app", \
     "--workers", "2", "--timeout", "120", "--access-logfile", "-"]
