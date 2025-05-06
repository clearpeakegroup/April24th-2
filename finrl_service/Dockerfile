# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1

# ── system deps: Python 3.10 + build tools ─────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip python3.10-dev \
        build-essential swig git libffi-dev libzmq3-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/shared:/app
## CACHE-FRIENDLY LAYOUT ----------------
COPY requirements.txt /tmp/
RUN python3 -m pip install --no-cache-dir --upgrade pip==23.3.1
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# copy only code we really need
COPY backend      /app/backend
COPY shared       /app/shared
COPY finrl_service /app/finrl_service
## ------------------------------------------------
# -- CACHE FRIENDLY LAYOUT ------------------------------------------------
#  a) install python deps first
COPY finrl_service/requirements.txt /tmp/
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

# ── pip bootstrap ──────────────────────────────────────────────────────
COPY requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir --upgrade pip==23.3.1

# FinRL (no deps) ‑‑ keep exactly one install
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir --no-deps \
    git+https://github.com/AI4Finance-Foundation/FinRL@v0.3.5#egg=FinRL

# Torch CUDA 11.8
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Remaining pinned deps (no duplicates)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir \
    stable-baselines3==1.7.0 \
    gym==0.21.0 ncps==0.0.6 ccxt

# copy project code last (keeps rebuilds fast)
# copy dirs individually to keep cache
COPY backend        /app/backend
COPY shared         /app/shared
COPY finrl_service   /app/finrl_service
COPY execution_service /app/execution_service

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir pyzmq>=25.1.0
# copy shared package so it exists even before volumes mount

CMD ["python3", "/app/rl_agent_tick_sub.py"]
