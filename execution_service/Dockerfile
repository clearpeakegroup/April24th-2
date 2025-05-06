FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential swig git libffi-dev libzmq3-dev && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/shared:/app
## CACHE-FRIENDLY LAYOUT ----------------
COPY requirements.txt /tmp/
RUN python3 -m pip install --no-cache-dir --upgrade pip==23.3.1
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# copy only code we really need
COPY backend      /app/backend
COPY shared       /app/shared
COPY execution_service /app/execution_service
## ------------------------------------------------
# -- CACHE FRIENDLY LAYOUT ------------------------------------------------
#  a) install python deps first
COPY execution_service/requirements.txt /tmp/
RUN python3 -m pip install --no-cache-dir --upgrade pip==23.3.1
# Install gym from wheel first to avoid metadata build error
RUN python3 -m pip install --no-cache-dir gym==0.21.0
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

#  b) then only copy the code we need, not dot-repo
COPY backend      /app/backend
COPY shared       /app/shared
COPY execution_service /app/execution_service
# -------------------------------------------------------------------------
VOLUME [ "/shared" ]

WORKDIR /app

# --- shared backend code & deps -------------------------------
COPY backend /app/backend
COPY backend/requirements.txt /tmp/backend-req.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir -r /tmp/backend-req.txt || true

COPY requirements.txt /tmp/
RUN python3 -m pip install -r /tmp/requirements.txt

# copy dirs individually to keep cache
COPY backend        /app/backend
COPY shared         /app/shared
COPY finrl_service   /app/finrl_service
COPY execution_service /app/execution_service

EXPOSE 8001
# copy shared package so it exists even before volumes mount

CMD ["./entry.sh"]
