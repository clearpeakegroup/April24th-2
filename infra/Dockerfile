# syntax=docker/dockerfile:1.4
FROM python:3.10-slim AS gradio

ARG GRADIO_VERSION
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install FinRL using centralized script
COPY scripts/install_finrl.sh /tmp/install_finrl.sh
RUN chmod +x /tmp/install_finrl.sh && \
    /tmp/install_finrl.sh && \
    rm /tmp/install_finrl.sh

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    && pip check \
    && pip freeze > requirements.lock \
    && pip install --no-cache-dir gradio==${GRADIO_VERSION}

COPY . .

EXPOSE 7860
CMD ["python3", "-u", "frontend/gradio_app.py"]
