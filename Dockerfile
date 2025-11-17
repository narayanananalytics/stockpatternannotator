# Stock Pattern Annotator - Dockerfile
# Supports both CPU and GPU training
#
# Build for CPU:
#   docker build -t stockpatternannotator:cpu .
#
# Build for GPU:
#   docker build --build-arg CUDA_VERSION=11.8.0 -t stockpatternannotator:gpu .

ARG CUDA_VERSION=""
ARG PYTHON_VERSION=3.10

# Base image selection (CPU or GPU)
FROM ${CUDA_VERSION:+nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04}${CUDA_VERSION:+}${CUDA_VERSION:-python:${PYTHON_VERSION}-slim-bullseye} AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python if using CUDA base image
RUN if [ -n "$CUDA_VERSION" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python${PYTHON_VERSION} \
            python${PYTHON_VERSION}-dev \
            python3-pip \
            && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
            && rm -rf /var/lib/apt/lists/*; \
    fi

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    if [ -n "$CUDA_VERSION" ]; then \
        # GPU version - install CUDA-enabled PyTorch
        pip install torch --index-url https://download.pytorch.org/whl/cu118 && \
        pip install -r requirements.txt; \
    else \
        # CPU version
        pip install torch --index-url https://download.pytorch.org/whl/cpu && \
        pip install -r requirements.txt; \
    fi

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create directories for data persistence
RUN mkdir -p /data/models \
             /data/results \
             /data/databases \
             /data/tensorboard \
             /data/examples

# Copy and set entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set volume mount points
VOLUME ["/data"]

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command - start Python interactive shell
CMD ["python"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import stockpatternannotator; print('OK')" || exit 1

# Labels
LABEL maintainer="Stock Pattern Annotator"
LABEL version="0.4.1"
LABEL description="OHLC Pattern Detection and RL Trading with GPU support"
