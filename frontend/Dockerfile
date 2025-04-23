FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── requirements relative to context root ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── copy entire frontend source ──
COPY . .

EXPOSE 7860
CMD ["python", "gradio_app.py", "--headless"]
