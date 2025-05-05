FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential swig git libffi-dev libzmq3-dev && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN python3 -m pip install -r /tmp/requirements.txt

WORKDIR /app
COPY . /app

EXPOSE 8001
# copy shared package so it exists even before volumes mount

ENV PYTHONPATH=/shared:/app
CMD ["./entry.sh"]
