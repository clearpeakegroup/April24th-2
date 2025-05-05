# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 1. System build deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential swig cmake git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better cache
COPY requirements.txt ./

# 2. Python deps (including DRL/ML deps)
# Install torch with CUDA support from the official PyTorch wheel index
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade 'pip<24.1' && \
    pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Copy install script and cache layer
COPY scripts/install_finrl.sh /tmp/install_finrl.sh
RUN chmod +x /tmp/install_finrl.sh && \
    /tmp/install_finrl.sh && \
    rm /tmp/install_finrl.sh

# Copy rest of the code last for best cache
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
