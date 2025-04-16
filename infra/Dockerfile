FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -tipsy && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy environment.yml
COPY ../backend/environment.yml /tmp/environment.yml

# Create the environment
RUN conda env create -f /tmp/environment.yml && conda clean -a

# Set the default environment
ENV CONDA_DEFAULT_ENV=finrl-backend
ENV PATH /opt/conda/envs/finrl-backend/bin:$PATH

# Set workdir
WORKDIR /workspace

CMD ["/bin/bash"] 