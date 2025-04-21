FROM python:3.10-slim AS gradio
ARG GRADIO_VERSION=4.44.1
ENV GRADIO_VERSION=${GRADIO_VERSION}
# ── build & runtime image
FROM python:3.10-slim

# 1. system libs   (rarely change  → cached)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. requirements (use poetry if pyproject.toml exists, else pip)
COPY pyproject.toml* requirements.txt* ./
## ── AUTOMATICALLY GENERATED (do not edit below) ────────────
RUN if [ -f pyproject.toml ]; then \
      pip install poetry && poetry install --no-root; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi && \
      pip install poetry && poetry install --no-root; \
    else \
      pip install --no-cache-dir -r requirements.txt; \

# 3. source code
COPY . .

# Entrypoint is set by docker-compose command 
CMD ["python3", "-u", "frontend/gradio_app.py"]
