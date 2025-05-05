# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    && pip check \
    && pip freeze > requirements.lock

COPY . .

CMD ["python3"] 