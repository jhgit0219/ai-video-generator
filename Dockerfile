# AI Video Generator - Docker Image
# Supports both CPU and GPU (NVIDIA CUDA) execution

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies and Python 3.11 from deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Upgrade pip to latest version to get newest package indices
RUN pip3 install --upgrade pip

# Install Python dependencies
# Install PyTorch packages separately from CUDA index for speed
RUN pip3 install --no-cache-dir \
    torch==2.9.0 \
    torchaudio==2.9.0 \
    torchvision==0.24.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install all other packages from PyPI
RUN pip3 install --no-cache-dir -r requirements.txt

# Download spaCy language model
RUN python3 -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/input data/output data/temp_images logs weights

# Note: YOLO weights will be auto-downloaded by ultralytics on first run
# You can pre-load them by mounting your local weights directory:
# -v ./weights:/app/weights

# Expose port for web interface
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Environment variables for LLM models (can be overridden in docker-compose.yml)
ENV EFFECTS_LLM_MODEL=llama3
ENV DEEPSEEK_MODEL=deepseek-coder:6.7b

# Start Ollama in background and pull required models
CMD ollama serve & \
    sleep 5 && \
    echo "Pulling LLM models (this may take a few minutes on first run)..." && \
    ollama pull ${EFFECTS_LLM_MODEL} && \
    ollama pull ${DEEPSEEK_MODEL} && \
    echo "Models ready. Starting web interface..." && \
    python3 -m gradio_interface
