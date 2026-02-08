FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git git-lfs curl ffmpeg build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch + CUDA 13 wheels
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130

# Copy and install repo requirements (if present)
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Optional small utilities
RUN git lfs install

# Default workdir and entrypoint
COPY . /workspace
WORKDIR /workspace
ENV PYTHONUNBUFFERED=1

# non-root user for safety (optional)
RUN useradd -ms /bin/bash dev && chown -R dev:dev /workspace
USER dev

CMD ["bash"]