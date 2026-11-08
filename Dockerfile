# Experimental CUDA-12.8 build for EIDF nodes with NVIDIA R570 drivers.
# SGLang 0.5.20 upstream defaults to CUDA 13; GPU kernel compatibility
# is NOT guaranteed on this CUDA-12 stack. Validate before research use.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

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

# Keep torch on the CUDA-12 wheel channel rather than accidentally upgrading
# it to an incompatible CUDA-13 build through another package.
RUN python -m pip install \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# SGLang 0.5.20 wheels exist for cp312, but the CUDA-12 kernel wheel
# source is an experimental workaround, not a vendor-supported v0.5.20 lane.
RUN python -m pip install sglang-kernel \
    --index-url https://docs.sglang.ai/whl/cu129/

# Pin torch while resolving SGLang so it isn't silently upgraded.
RUN printf 'torch==2.10.0\ntorchvision==0.25.0\ntorchaudio==2.10.0\n' > /tmp/torch-constraints.txt \
    && python -m pip install -c /tmp/torch-constraints.txt 'sglang==0.5.20' \
    && LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH} \
       python -m pip install -c /tmp/torch-constraints.txt \
       'kvcached==0.1.6' --no-build-isolation \
    && python -m pip install ipython pytest psutil pandas \
    && python -m pip check \
    && python -c 'import sys,torch,sglang,kvcached; print(sys.version); print(torch.__version__, torch.version.cuda); print("SGLang and kvcached imported")'

WORKDIR /workspace
CMD ["/bin/bash", "-lc", "sleep infinity"]
