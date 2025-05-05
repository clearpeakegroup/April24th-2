FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- Python 3.10, pip, header files, and build tools -------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip python3.10-dev \
        build-essential swig git libffi-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade "pip==23.3.1" && \
    rm -rf /var/lib/apt/lists/*

# --- locked dependency set ---------------------------------------------
COPY requirements.txt /tmp/
RUN python3 -m pip install -r /tmp/requirements.txt \
        -f https://download.pytorch.org/whl/torch_stable.html

# (optional) COPY your strategy code here
# COPY . /app
# WORKDIR /app
# CMD ["python3","run_avalanche.py"]
