# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── requirements relative to context root ──
COPY requirements.txt .
# Note: FinRL is installed only in the backend container
RUN pip uninstall -y gradio gradio-client || true
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --force-reinstall gradio==4.16.0 \
    && pip check \
    && pip freeze > requirements.lock

# ── copy entire frontend source ──
COPY . .

EXPOSE 7860
CMD ["python", "gradio_app.py", "--headless"]
