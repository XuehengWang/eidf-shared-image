# CUDA 13.0 development image. Requires a CUDA-13-compatible NVIDIA driver
# on the GPU node. Built for linux/amd64 (EIDF A100 workers).
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    CUDA_HOME=/usr/local/cuda \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    ENABLE_KVCACHED=false \
    KVCACHED_AUTOPATCH=0
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git vim nano less tmux htop \
    build-essential cmake ninja-build pkg-config gdb strace \
    libnuma-dev libibverbs-dev libaio-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 + conda; separate from the distro Python.
RUN curl -fsSL -o /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && conda install -y python=3.12 \
    && conda clean -afy

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja

# PyTorch matching SGLang 0.5.20, with CUDA 13.0 runtime wheels.
# PyTorch publishes torchvision==0.28.0 for torch==2.13.0.
# Torchaudio is omitted: it is not necessary for these inference experiments.
RUN python -m pip install \
    torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu130

# Install the pinned SGLang release. The resolver must retain torch==2.13.0.
RUN python -m pip install "sglang==0.5.20"

# Ensure CUDA 13 kernels are sourced from SGLang's CUDA 13 index.
# --no-deps prevents kernel-index resolution from changing PyTorch.
RUN python -m pip install --upgrade --no-deps sglang-kernel \
    --index-url https://docs.sglang.ai/whl/cu130/

# kvcached builds with the installed PyTorch headers. Compiler/linker stubs
# are available in the devel image, but GPU runtime execution needs a driver.
RUN LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH} \
    python -m pip install "kvcached==0.1.6" --no-build-isolation

RUN python -m pip install ipython pytest psutil pandas \
    && python -m pip check

# Import checks do not require a GPU (but cannot prove GPU compatibility).
RUN python - <<'PYTHON'
import importlib.metadata as md
import sys
import torch
print('Python:', sys.version)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
for pkg in ('sglang', 'kvcached', 'sglang-kernel'):
    print(pkg, md.version(pkg))
assert torch.__version__.split('+')[0] == '2.13.0'
assert torch.version.cuda and torch.version.cuda.startswith('13.0')
PYTHON

WORKDIR /workspace
CMD ["/bin/bash", "-lc", "sleep infinity"]
