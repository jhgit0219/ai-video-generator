# Docker Development Guide

Complete guide for using Docker with this project and dockerizing future builds.

---

## Table of Contents

1. [Quick Start for Users](#quick-start-for-users)
2. [Understanding the Docker Setup](#understanding-the-docker-setup)
3. [Development Workflow](#development-workflow)
4. [Updating the Docker Image](#updating-the-docker-image)
5. [Dockerizing Future Projects](#dockerizing-future-projects)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Quick Start for Users

### First Time Setup

**Prerequisites:**
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- For GPU: NVIDIA Docker runtime ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

**Run the Application:**

```bash
# 1. Clone or navigate to project
cd ai-video-generator

# 2. Build the Docker image (one-time, ~5 minutes)
docker-compose build

# 3. Start with GPU support
docker-compose --profile gpu up

# OR start with CPU-only
docker-compose --profile cpu up

# 4. Access web interface
# Open browser to http://localhost:7860
```

**First startup takes 5-10 minutes** to download LLM models (~8GB). Subsequent starts are instant.

### Daily Usage

```bash
# Start the application
docker-compose --profile gpu up

# Stop the application (Ctrl+C, then)
docker-compose down

# View logs
docker-compose logs -f

# Restart after code changes
docker-compose restart
```

---

## Understanding the Docker Setup

### File Structure

```
ai-video-generator/
├── Dockerfile              # Main image definition
├── docker-compose.yml      # Multi-service orchestration
├── .dockerignore          # Files excluded from image
├── DOCKER_GUIDE.md        # User-facing documentation
└── DOCKER_DEVELOPMENT_GUIDE.md  # This file
```

### Dockerfile Breakdown

```dockerfile
# Base image with NVIDIA CUDA support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System dependencies (Python, FFmpeg, etc.)
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget curl ffmpeg

# Install Ollama for local LLM
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Download AI models (YOLO weights)
RUN wget https://github.com/ultralytics/.../yolo11x-seg.pt

# Expose web interface port
EXPOSE 7860

# Startup command
CMD ollama serve & \
    ollama pull llama3 && \
    ollama pull deepseek-coder:6.7b && \
    python3 -m gradio_interface
```

**Key Concepts:**

1. **Layer Caching** - Each `RUN` command creates a cached layer
2. **Build Context** - `.dockerignore` excludes files from image
3. **Base Image** - `nvidia/cuda` provides GPU support
4. **Runtime Dependencies** - System packages, Python packages, AI models
5. **Entrypoint** - `CMD` runs on container start

### docker-compose.yml Breakdown

```yaml
version: '3.8'

services:
  ai-video-generator-gpu:
    build: .
    runtime: nvidia  # Enable GPU access
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - EFFECTS_LLM_MODEL=llama3
    volumes:
      - ./data/input:/app/data/input      # Input scripts
      - ./data/output:/app/data/output    # Generated videos
      - ollama-data:/root/.ollama         # Persistent models
    ports:
      - "7860:7860"  # Web interface
    profiles:
      - gpu

volumes:
  ollama-data:  # Persists downloaded LLM models
```

**Key Concepts:**

1. **Profiles** - Separate GPU/CPU configurations
2. **Volumes** - Persist data and share files with host
3. **Environment Variables** - Configure runtime behavior
4. **Runtime** - `nvidia` enables GPU passthrough
5. **Ports** - Map container port to host port

### .dockerignore Breakdown

```
# Development artifacts
__pycache__/
*.pyc
venv/
.git/

# Large outputs (regenerated)
data/output/*
data/temp_images/*

# Test/debug files
tests/
debug_*.py
ignore/
```

**Purpose:** Reduce image size and build time by excluding unnecessary files.

---

## Development Workflow

### Scenario 1: Testing Code Changes Locally

**Option A: Mount Source Code (Fast Iteration)**

```yaml
# docker-compose.yml
volumes:
  - .:/app  # Mount entire project directory
  - ./data/input:/app/data/input
  - ./data/output:/app/data/output
  - ollama-data:/root/.ollama
```

```bash
# Start with mounted source
docker-compose --profile gpu up

# Edit code on host machine
# Changes reflected immediately in container
# Restart container to apply
docker-compose restart
```

**Option B: Rebuild Image (Production-like)**

```bash
# Make code changes
# Rebuild image with changes
docker-compose build --no-cache

# Start fresh container
docker-compose --profile gpu up
```

### Scenario 2: Adding New Python Dependencies

**Step 1: Update requirements.txt**

```bash
# On host machine
pip install new-package
pip freeze > requirements.txt
```

**Step 2: Rebuild Image**

```bash
# Rebuild with new dependencies
docker-compose build

# Start container
docker-compose --profile gpu up
```

### Scenario 3: Debugging Inside Container

```bash
# Enter running container
docker exec -it ai-video-gen-gpu bash

# Now you're inside the container
pwd  # /app
ls   # See application files
python3 main.py --script data/input/test.json

# Check logs
tail -f logs/app.log

# Test Ollama
ollama list  # See installed models
ollama run llama3 "Hello"

# Exit container
exit
```

### Scenario 4: Developing New Features

**Workflow:**

1. **Develop locally** (without Docker) using `venv`
2. **Test locally** with `python main.py`
3. **Update Dockerfile/docker-compose.yml** if needed
4. **Build and test in Docker**
5. **Commit changes** (including Docker files)

```bash
# Local development
source venv/bin/activate  # or venv\Scripts\activate on Windows
python main.py --script data/input/test.json

# Test in Docker
docker-compose build
docker-compose --profile gpu up

# Verify everything works
# Commit changes
git add Dockerfile docker-compose.yml requirements.txt
git commit -m "feat: add new feature"
```

---

## Updating the Docker Image

### When to Update

Update the Docker setup when:
- Adding/removing Python dependencies
- Changing system dependencies (apt packages)
- Updating base image (e.g., new CUDA version)
- Changing environment variables
- Modifying startup command

### How to Update

**1. Modify Dockerfile**

```dockerfile
# Example: Add new system package
RUN apt-get update && apt-get install -y \
    python3.10 \
    ffmpeg \
    redis-server  # NEW PACKAGE
```

**2. Update .dockerignore if Needed**

```
# Exclude new large files
new_feature_output/*
```

**3. Update docker-compose.yml**

```yaml
environment:
  - NEW_FEATURE_ENABLED=true  # New config
```

**4. Rebuild and Test**

```bash
# Clean rebuild
docker-compose build --no-cache

# Test
docker-compose --profile gpu up
```

**5. Update Documentation**

- Update `DOCKER_GUIDE.md` for users
- Update this file for developers

### Versioning Docker Images

```bash
# Tag image with version
docker build -t ai-video-gen:v1.0.0 .

# Tag as latest
docker build -t ai-video-gen:latest .

# Push to registry (optional)
docker push yourusername/ai-video-gen:v1.0.0
```

---

## Dockerizing Future Projects

### Step-by-Step Guide

#### 1. Choose Base Image

```dockerfile
# Python application
FROM python:3.10-slim

# GPU-accelerated Python app
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Node.js application
FROM node:18-alpine

# Multi-language (Python + Node)
FROM ubuntu:22.04
RUN apt-get install python3 nodejs npm
```

**Considerations:**
- **Size** - Alpine images are smallest (~5MB), Ubuntu is largest (~80MB)
- **Compatibility** - Some packages need full Ubuntu (not Alpine)
- **GPU** - Use NVIDIA CUDA base for GPU support

#### 2. Install System Dependencies

```dockerfile
# Update package lists
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    git \
    curl \
    wget \
    # Media processing
    ffmpeg \
    imagemagick \
    # Libraries
    libsm6 \
    libxext6 \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/*
```

**Best Practice:** Chain commands with `&&` to reduce layers.

#### 3. Install Application Dependencies

```dockerfile
# Set working directory
WORKDIR /app

# Copy dependency file first (for layer caching)
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code (after dependencies)
COPY . .
```

**Layer Caching:** Dependencies change less often than code, so install them first.

#### 4. Download/Prepare Models or Assets

```dockerfile
# Create directories
RUN mkdir -p models data/input data/output

# Download model weights
RUN wget https://example.com/model.pt -O models/model.pt

# Or pull from package registry
RUN python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base')"
```

#### 5. Configure Networking

```dockerfile
# Expose port(s)
EXPOSE 7860      # Web interface
EXPOSE 8000      # API server
```

#### 6. Define Startup Command

```dockerfile
# Simple command
CMD ["python3", "main.py"]

# Multiple services (use shell form)
CMD ollama serve & python3 app.py

# Override with docker-compose or docker run
ENTRYPOINT ["python3"]
CMD ["main.py"]  # Can be overridden
```

#### 7. Create docker-compose.yml

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
    environment:
      - DEBUG=false
      - API_KEY=${API_KEY}  # From .env file
    restart: unless-stopped

# Optional: Add database, cache, etc.
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  app-data:
```

#### 8. Create .dockerignore

```
# Version control
.git/
.gitignore

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
venv/
env/

# IDE
.vscode/
.idea/
*.swp

# Documentation
*.md
docs/

# Tests
tests/
test_*.py

# Large outputs
data/output/*
logs/*

# OS
.DS_Store
Thumbs.db
```

#### 9. Test and Iterate

```bash
# Build
docker build -t myapp:dev .

# Run
docker run -p 7860:7860 myapp:dev

# Or with docker-compose
docker-compose up

# Debug
docker exec -it <container-id> bash
```

### Common Patterns

#### Multi-Stage Build (Reduce Image Size)

```dockerfile
# Build stage
FROM python:3.10 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python3", "main.py"]
```

**Benefit:** Final image only has runtime dependencies, not build tools.

#### Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1
```

**Benefit:** Docker knows when app is ready and can restart if unhealthy.

#### Non-Root User (Security)

```dockerfile
# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

CMD ["python3", "main.py"]
```

**Benefit:** Reduced security risk if container is compromised.

---

## Troubleshooting

### Build Failures

**Problem:** `ERROR [internal] load metadata for docker.io/library/...`

**Solution:** Check internet connection, try different base image registry.

```bash
# Use mirror
FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04
```

---

**Problem:** `Package not found` during apt-get install

**Solution:** Update package lists first.

```dockerfile
RUN apt-get update && apt-get install -y package-name
```

---

**Problem:** `Permission denied` when copying files

**Solution:** Check file permissions on host, use `COPY --chown`.

```dockerfile
COPY --chown=appuser:appuser . /app
```

---

### Runtime Issues

**Problem:** GPU not detected inside container

**Solution:** Verify NVIDIA Docker runtime installed.

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# If fails, install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

---

**Problem:** `Address already in use` on port 7860

**Solution:** Change host port in docker-compose.yml.

```yaml
ports:
  - "8080:7860"  # Use port 8080 on host
```

---

**Problem:** Changes not reflected in container

**Solution:** Rebuild image or use volume mount.

```bash
# Option 1: Rebuild
docker-compose build --no-cache
docker-compose up

# Option 2: Mount source
# (Add to docker-compose.yml volumes)
- .:/app
```

---

**Problem:** Container exits immediately

**Solution:** Check logs for errors.

```bash
# View logs
docker-compose logs

# Run interactively
docker run -it myapp:dev bash
```

---

### Storage Issues

**Problem:** Disk full from Docker images/containers

**Solution:** Clean up unused resources.

```bash
# Remove unused images
docker image prune -a

# Remove all stopped containers
docker container prune

# Remove unused volumes
docker volume prune

# Nuclear option (careful!)
docker system prune -a --volumes
```

---

**Problem:** Models not persisting between restarts

**Solution:** Use named volumes.

```yaml
volumes:
  - ollama-data:/root/.ollama

volumes:
  ollama-data:
```

---

## Best Practices

### Dockerfile Best Practices

1. **Order commands by change frequency**
   ```dockerfile
   # Rarely changes
   FROM python:3.10
   RUN apt-get install ...

   # Changes occasionally
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   # Changes frequently
   COPY . .
   ```

2. **Minimize layers**
   ```dockerfile
   # Bad (3 layers)
   RUN apt-get update
   RUN apt-get install -y python3
   RUN rm -rf /var/lib/apt/lists/*

   # Good (1 layer)
   RUN apt-get update && \
       apt-get install -y python3 && \
       rm -rf /var/lib/apt/lists/*
   ```

3. **Use specific versions**
   ```dockerfile
   # Bad
   FROM python:latest

   # Good
   FROM python:3.10.12-slim
   ```

4. **Clean up in same layer**
   ```dockerfile
   RUN apt-get update && \
       apt-get install -y package && \
       # Clean up immediately
       rm -rf /var/lib/apt/lists/*
   ```

5. **Use .dockerignore**
   - Exclude `.git`, `node_modules`, `venv`, test files
   - Faster builds, smaller images

### docker-compose.yml Best Practices

1. **Use environment variables**
   ```yaml
   environment:
     - API_KEY=${API_KEY}  # From .env file
   ```

2. **Define restart policies**
   ```yaml
   restart: unless-stopped  # Auto-restart on failure
   ```

3. **Use named volumes**
   ```yaml
   volumes:
     - app-data:/data  # Named volume

   volumes:
     app-data:
   ```

4. **Health checks**
   ```yaml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:7860"]
     interval: 30s
     timeout: 10s
     retries: 3
   ```

5. **Resource limits** (production)
   ```yaml
   deploy:
     resources:
       limits:
         memory: 8G
         cpus: '4'
   ```

### Security Best Practices

1. **Run as non-root user**
   ```dockerfile
   USER appuser
   ```

2. **Don't include secrets in image**
   ```dockerfile
   # Use environment variables or secrets management
   ENV API_KEY=${API_KEY}
   ```

3. **Use specific base image versions**
   ```dockerfile
   FROM python:3.10.12-slim  # Not :latest
   ```

4. **Scan images for vulnerabilities**
   ```bash
   docker scan myapp:latest
   ```

5. **Minimize attack surface**
   - Use minimal base images (alpine, slim)
   - Remove unnecessary packages
   - Don't run SSH server in container

### Performance Best Practices

1. **Multi-stage builds** (reduce size)
2. **Layer caching** (order commands correctly)
3. **Use volumes** (don't copy large datasets)
4. **Resource limits** (prevent resource exhaustion)
5. **Health checks** (auto-restart on failure)

---

## Quick Reference

### Common Commands

```bash
# Build
docker build -t myapp:tag .
docker-compose build

# Run
docker run -p 7860:7860 myapp:tag
docker-compose up
docker-compose up -d  # Detached

# Stop
docker-compose down
docker-compose stop

# Logs
docker logs <container-id>
docker-compose logs -f

# Execute command in container
docker exec -it <container-id> bash
docker-compose exec app bash

# Cleanup
docker system prune -a
docker volume prune

# Info
docker ps              # Running containers
docker images          # Available images
docker volume ls       # Volumes
docker-compose ps      # Compose services
```

### File Checklist

When dockerizing a project, create:

- [ ] `Dockerfile` - Image definition
- [ ] `docker-compose.yml` - Multi-service orchestration
- [ ] `.dockerignore` - Build context exclusions
- [ ] `DOCKER_GUIDE.md` - User documentation
- [ ] `README.md` - Include Docker quick start
- [ ] `.env.example` - Environment variable template

### Testing Checklist

Before committing Docker setup:

- [ ] Build succeeds without errors
- [ ] Container starts successfully
- [ ] Application accessible on expected port
- [ ] Volumes persist data correctly
- [ ] Environment variables work
- [ ] GPU acceleration works (if applicable)
- [ ] Logs are accessible
- [ ] Container restarts on failure
- [ ] Documentation is complete

---

## Additional Resources

**Official Documentation:**
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

**This Project:**
- `DOCKER_GUIDE.md` - End-user deployment guide
- `docker-compose.yml` - Configuration examples
- `Dockerfile` - Reference implementation

---

**Last Updated:** 2025-11-07

This guide covers everything from basic usage to advanced dockerization patterns. Refer to it when developing new features or dockerizing future projects!
